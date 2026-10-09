"""The JEM trainer: the MPS objective with SGLD for the generative term (D35, D71).

For a training batch x with labels y and SGLD negatives x⁻ (persistent chains),

    L = (1-β)·CE(x_adv) + (β/N)·L_gen(x)      (+ energy penalty)
    L_gen(x) = -f_y(x) + mean logsumexp_c f(x⁻)

L_gen is the contrastive estimate of -log p(x, y): the negatives' mean score
stands in for log Z, so its gradient is the JEM gradient. NAT is ``evasion:
null``: CE(x) in place of CE(x_adv); β=0 needs no negatives. x_adv is the shared PGD.
N = ``n_vars(x)``, the features and the class, as for the MPS (D86).

Validation and selection are the MPS's (:func:`bm4tc.core.objective.evaluate`,
argmin ``objective``, D8), with log Z estimated by a standardized, seeded SGLD
chain independent of the training sampler, so that weak training chains cannot
buy a low validation loss. The replay buffer is part of the best state and of the
checkpoint.
"""
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from bm4tc.core.attacks import build_attack
from bm4tc.core.embeddings import range_size_of, rel_to_abs
from bm4tc.core.jem.model import JEMMLP, JEMModelConfig
from bm4tc.core.jem.sampler import ReplayBuffer, SGLDConfig, SGLDSampler
from bm4tc.core.objective import evaluate, mix, n_vars, optimizer
from bm4tc.core.train import TrainConfig, curriculum_eps, evasion_config

logger = logging.getLogger(__name__)


@dataclass
class ValidSamplerConfig:
    """The standardized SGLD chain behind the validation log Z estimate."""
    num_steps: int = 100
    step_size: float = 0.05
    noise_std: float = 0.005
    reinit_probability: float = 0.1
    buffer_size: int = 2048
    batch_size: int = 256
    num_batches: int = 4
    seed: int = 12345


@dataclass
class JEMConfig:
    """Everything JEM-specific in a run config (the ``jem:`` node); the shared
    knobs (beta, epochs, optimizer, attack, ...) are the ``trainer:`` node."""
    model: JEMModelConfig = field(default_factory=JEMModelConfig)
    sampler: SGLDConfig = field(default_factory=SGLDConfig)
    valid_sampler: ValidSamplerConfig = field(default_factory=ValidSamplerConfig)
    energy_l2: float = 1e-4            # squared-score penalty on positives and negatives
    grad_clip: Optional[float] = 10.0
    input_noise_std: float = 0.01      # Gaussian noise on the clean training inputs


class JEMTrainer:
    """Fits a JEMMLP and keeps the best epoch on ``objective/valid``. The epoch
    record passed to ``on_epoch_end`` has the MPS Trainer's shape."""

    def __init__(self, model: JEMMLP, cfg: TrainConfig, jem: JEMConfig,
                 train_loader: DataLoader, valid_loader: DataLoader, device: torch.device,
                 seed: int = 0, buffer: Optional[dict] = None):
        if cfg.eval_every < 1:
            raise ValueError(f"eval_every must be >= 1, got {cfg.eval_every}")
        self.model, self.cfg, self.jem = model.to(device), cfg, jem
        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.device = device

        lo_hi = model.input_range
        self.sampler = SGLDSampler(
            jem.sampler, ReplayBuffer(jem.sampler.buffer_size, model.data_dim, lo_hi, seed=seed))
        if buffer is not None:
            self.sampler.buffer.load_state_dict(buffer)
        v = jem.valid_sampler
        self.valid_sampler = SGLDSampler(
            SGLDConfig(num_steps=v.num_steps, step_size=v.step_size, noise_std=v.noise_std,
                       reinit_probability=v.reinit_probability, buffer_size=v.buffer_size),
            ReplayBuffer(v.buffer_size, model.data_dim, lo_hi, seed=v.seed))

        self.attack = None
        if cfg.evasion is not None:
            evasion = evasion_config(cfg.evasion)
            if evasion.method != "PGD":
                raise ValueError(f"Training supports the PGD attack only, got {evasion.method!r}")
            self.attack = build_attack(evasion)
            self.range_size = range_size_of(model)
            self.eps_rel = float(evasion.eps_rel[0] if evasion.eps_rel else 0.1)
            self.eps_abs = rel_to_abs(self.eps_rel, self.range_size)
            self._start_abs = rel_to_abs(cfg.curriculum_eps_start_rel, self.range_size)

        self.best = {"objective": float("inf")}
        self.best_epoch = 0
        self._best_state = self._snapshot()

    # ── One step ────────────────────────────────────────────────────────────

    def _objective(self, data, labels, eps_abs):
        """(objective, penalty) on one batch (module docstring)."""
        cfg, beta = self.cfg, self.cfg.beta
        positives = data
        if self.jem.input_noise_std > 0:
            positives = (data + self.jem.input_noise_std * torch.randn_like(data)).clamp(
                *self.model.input_range)
        negatives = None
        if beta > 0:
            negatives = self.sampler.sample_training(self.model, len(data), self.device)

        logits = self.model(positives)
        ce = F.cross_entropy(logits, labels)
        dis = ce
        if self.attack is not None:
            self.model.eval()
            adv = self.attack.generate(model=self.model, naturals=data, labels=labels,
                                       eps_abs=eps_abs, device=self.device)
            self.model.train()
            dis = F.cross_entropy(self.model(adv), labels)

        gen = penalty = None
        if negatives is not None:
            neg_score = self.model.marginal_score(negatives.detach())
            gen = -logits.gather(1, labels[:, None]).mean() + neg_score.mean()
            if self.jem.energy_l2 > 0:
                pos_score = torch.logsumexp(logits, dim=-1)
                penalty = self.jem.energy_l2 * (pos_score.square().mean()
                                                + neg_score.square().mean())
        return mix(dis, gen, beta, n_vars(data)), penalty

    # ── Validation ──────────────────────────────────────────────────────────

    def _log_Z(self, epoch: int) -> float:
        """The validation estimate of log Z: the mean score of a standardized SGLD
        chain, seeded per epoch apart from the training randomness. nan at β=0."""
        if self.cfg.beta <= 0:
            return float("nan")
        v = self.jem.valid_sampler
        devices = []
        if torch.device(self.device).type == "cuda":
            index = torch.device(self.device).index
            devices = [torch.cuda.current_device() if index is None else index]
        scores = []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(v.seed + epoch)
            for _ in range(v.num_batches):
                initial, indices = self.valid_sampler.buffer.initial(
                    v.batch_size, v.reinit_probability, self.device)
                samples = self.valid_sampler.refine(self.model, initial)
                self.valid_sampler.buffer.update(indices, samples)
                with torch.no_grad():
                    scores.append(self.model.marginal_score(samples).mean().item())
        return sum(scores) / len(scores)

    def _validate(self, epoch: int) -> dict:
        return evaluate(
            self.model, self.valid_loader, self.device,
            log_Z=self._log_Z(epoch), beta=self.cfg.beta, attack=self.attack,
            eps_abs=self.eps_abs if self.attack is not None else 0.0,
        )

    # ── Loop ────────────────────────────────────────────────────────────────

    def _snapshot(self) -> dict:
        return {"model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                "buffer": {"data": self.sampler.buffer.data.clone()}}

    def _train_epoch(self, eps_abs: float) -> Optional[dict]:
        """Mean objective and penalty over the epoch; None if a loss is non-finite."""
        self.model.train()
        objectives, penalties, diagnostics = [], [], []
        for data, labels in self.train_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            objective, penalty = self._objective(data, labels, eps_abs)
            loss = objective if penalty is None else objective + penalty
            if not torch.isfinite(loss):
                logger.warning(f"Non-finite JEM loss ({loss.item()}); stopping.")
                return None
            self.optimizer.zero_grad()
            loss.backward()
            if self.jem.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.jem.grad_clip)
            self.optimizer.step()
            objectives.append(objective.item())
            penalties.append(0.0 if penalty is None else penalty.item())
            if self.cfg.beta > 0 and self.jem.sampler.track_diagnostics:
                diagnostics.append(self.sampler.last_diagnostics)
        n = len(objectives)
        self._diagnostics = {f"sgld/{k}": sum(d[k] for d in diagnostics) / len(diagnostics)
                             for k in (diagnostics[0] if diagnostics else {})}
        return {"objective": sum(objectives) / n, "penalty": sum(penalties) / n}

    def train(self, on_epoch_end: Optional[Callable[[int, Dict], None]] = None,
              output_dir: Optional[Path] = None):
        cfg = self.cfg
        self.optimizer = optimizer(self.model.parameters(), cfg.optimizer)
        patience = 0
        regime = "JEM-AT" if self.attack is not None else "JEM"
        logger.info(f"{regime} training begins (beta={cfg.beta:.3g}, "
                    f"{self.model.count_parameters()} parameters).")
        pbar = tqdm(range(1, cfg.max_epoch + 1), desc=regime, unit="ep", dynamic_ncols=True)
        for epoch in pbar:
            eps_abs = (curriculum_eps(cfg, epoch, self._start_abs, self.eps_abs)
                       if self.attack is not None else 0.0)
            train = self._train_epoch(eps_abs)
            if train is None:
                break
            if self.attack is not None:
                train["eps_rel"] = eps_abs / self.range_size
            record = {"train": train}
            validated = epoch % cfg.eval_every == 0
            if validated:
                valid = self._validate(epoch)
                logged = dict(valid)
                if "rob" in valid:
                    logged["rob"] = {self.eps_rel: valid["rob"]}
                record["valid"] = logged
                pbar.set_postfix(loss=f"{train['objective']:.4f}", acc=f"{valid['acc']:.4f}")
            record["diagnostics"] = self._diagnostics
            if on_epoch_end is not None:
                on_epoch_end(epoch, record)
            if validated:
                if math.isfinite(valid["objective"]) and valid["objective"] < self.best["objective"]:
                    self.best, self.best_epoch = dict(valid), epoch
                    self._best_state = self._snapshot()
                    patience = 0
                else:
                    patience += 1
            if patience > cfg.patience:
                logger.info(f"Early stopping after epoch {epoch}.")
                break

        self.model.load_state_dict(self._best_state["model"])
        self.sampler.buffer.load_state_dict(self._best_state["buffer"])
        if not math.isfinite(self.best["objective"]):
            logger.warning("No finite validation objective; skipping model save.")
        elif cfg.save and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(output_dir / "model"), replay_buffer=self._best_state["buffer"])
        logger.info("Training finished.")
