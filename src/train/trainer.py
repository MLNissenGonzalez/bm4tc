"""One trainer for ConditionalBornMachine: natural (NAT) and adversarial (AT) training.

The objective, for a training batch x with labels y, is

    L = (1-α)·[(1-cw)·L_dis(x_adv) + cw·L_dis(x)] + α·L_gen(x)      (+ norm penalty)

where x_adv is a PGD attack on x. NAT is the no-attack case (``evasion: null``):
then cw is irrelevant and the bracket is L_dis(x), so L = mixed_nll(x, α). The
generative term always sees clean data (D18): fitting p(x) to adversarial points
would work against detection and purification.

Every ``eval_every`` epochs the trainer validates the same objective on the
validation set (:func:`src.utils.train.evaluate`) and keeps the epoch with the
lowest ``objective`` (D8). ``patience`` counts validation events, not epochs.
"""

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from src.model import ConditionalBornMachine
from src.utils.embeddings import range_size_of, rel_to_abs
from src.utils.evasion import EvasionConfig, ProjectedGradientDescent, build_attack
from src.utils.train import (
    NormControlConfig,
    NormRegularizer,
    NormTracker,
    OptimizerConfig,
    evaluate,
    optimizer,
    resolve_log_target,
)

import logging

logger = logging.getLogger(__name__)

# Constant (not the run seed) so every seed in a sweep attacks the same validation
# samples, keeping cross-seed rob comparisons clean.
_ADV_SUBSET_SEED = 0


@dataclass
class TrainConfig:
    alpha: float = 0.0  # weight of the generative term
    max_epoch: int = 100
    batch_size: int = 64
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    patience: int = 250  # in validation events: eval_every=5, patience=10 is 50 epochs
    eval_every: int = 1
    # The training attack, an EvasionConfig; None is NAT. Untyped here because
    # OmegaConf 2.3 cannot merge a preset into an Optional[dataclass] field that
    # is None under Hydra; the Trainer checks it against EvasionConfig instead.
    evasion: Optional[Dict[str, Any]] = None
    clean_weight: float = 0.0  # cw above; AT only
    # Linear ramp of the training radius from curriculum_eps_start_rel to the full
    # radius, reached at epoch curriculum_end·max_epoch. Validation always attacks
    # at the full radius. AT only.
    curriculum: bool = False
    curriculum_eps_start_rel: float = 0.0
    curriculum_end: float = 1.0
    norm_control: NormControlConfig = field(default_factory=NormControlConfig)
    save: bool = False


def evasion_config(raw) -> EvasionConfig:
    """``TrainConfig.evasion`` as an EvasionConfig; an unknown key raises."""
    merged = OmegaConf.merge(OmegaConf.structured(EvasionConfig), raw)
    return OmegaConf.to_object(merged)


class Trainer:
    """Fits a ConditionalBornMachine and keeps the best epoch on ``objective/valid``."""

    def __init__(
        self,
        cbm: ConditionalBornMachine,
        cfg: TrainConfig,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        device: torch.device,
    ):
        if cfg.eval_every < 1:
            raise ValueError(f"eval_every must be >= 1, got {cfg.eval_every}")
        self.cbm = cbm
        self.cfg = cfg
        self.train_loader = train_loader
        self.valid_loader = valid_loader  # not shuffled: adv_indices are positions in it
        self.device = device

        self.best = {"objective": float("inf")}
        self.best_tensors = [t.cpu().clone().detach() for t in cbm.tensors]
        self._nc = cfg.norm_control
        self.norm_regularizer: NormRegularizer | None = None
        self._nc_log_target: float | None = None

        self.attack: ProjectedGradientDescent | None = None
        self.adv_indices: set[int] = set()
        self.clean_weight = 1.0  # without an attack the bracket is the clean term
        if cfg.evasion is not None:
            self._init_attack()

    # ------------------------------------------------------------------
    # Attack
    # ------------------------------------------------------------------

    def _init_attack(self):
        cfg = self.cfg
        evasion = evasion_config(cfg.evasion)
        if evasion.method != "PGD":
            raise ValueError(f"Training supports the PGD attack only, got {evasion.method!r}")
        self.attack = build_attack(evasion)
        self.clean_weight = cfg.clean_weight

        # The rel -> abs boundary for training: configs author eps_rel, the attack
        # is driven by absolute model-domain budgets.
        self.range_size = range_size_of(self.cbm)
        self.eps_rel = float(evasion.eps_rel[0] if evasion.eps_rel else 0.1)
        self.eps_abs = rel_to_abs(self.eps_rel, self.range_size)
        self._curriculum_start_abs = rel_to_abs(cfg.curriculum_eps_start_rel, self.range_size)
        logger.info(
            f"Attack budget: eps_rel={self.eps_rel:g} -> eps_abs={self.eps_abs:g} "
            f"(input range width {self.range_size:g})"
        )

        if cfg.alpha >= 1.0:
            logger.warning(
                "alpha=1 with an attack: the discriminative term vanishes, so training "
                "is clean generative NLL and the adversarial examples are discarded."
            )
        if cfg.clean_weight >= 1.0:
            logger.warning(
                "clean_weight=1: no adversarial examples enter the objective and no "
                "valid samples are attacked, so 'rob' is never reported."
            )

        # The validation attack subset: (1 - cw)·n positions in the valid loader's
        # order (stable: only the train split is shuffled), drawn once from a
        # constant seed so the rob curve is not perturbed by resampling.
        n = len(self.valid_loader.dataset)
        k = min(n, max(0, int(round((1.0 - cfg.clean_weight) * n))))
        gen = torch.Generator().manual_seed(_ADV_SUBSET_SEED)
        self.adv_indices = set(torch.randperm(n, generator=gen)[:k].tolist())
        logger.info(
            f"Attacking {k}/{n} valid samples every {cfg.eval_every} epoch(s); "
            f"patience={cfg.patience} valid events (~{cfg.patience * cfg.eval_every} epochs)."
        )

    def _eps_abs(self, epoch: int) -> float:
        """Training radius at ``epoch`` (the curriculum ramp, or the full radius)."""
        if not self.cfg.curriculum:
            return self.eps_abs
        end_epoch = self.cfg.curriculum_end * self.cfg.max_epoch
        progress = min(1.0, epoch / end_epoch)
        return self._curriculum_start_abs + progress * (self.eps_abs - self._curriculum_start_abs)

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _objective(self, data, labels, eps_abs, tracker: NormTracker) -> torch.Tensor:
        """The training objective on one batch (module docstring).

        ``mixed_nll(x, y, a) = (1-a)·L_dis + a·L_gen`` decomposes exactly, so both
        clean terms fold into one call at a rescaled alpha: with
        ``s = (1-a)·cw + a`` and ``a' = a/s``,

            s · mixed_nll(x, y, a') = (1-a)·cw·L_dis(x) + a·L_gen(x)

        which keeps an AT step at two forwards (one log Z) rather than three. At
        least one of the two weights is positive. Without an attack the objective
        is ``mixed_nll(x, y, a)``, computed directly: ``s = (1-a) + a`` need not be
        exactly 1.0 in floating point.
        """
        alpha, cw = self.cfg.alpha, self.clean_weight
        if self.attack is None:
            nll = self.cbm.mixed_nll(data, labels, alpha, debug=self._nc.debug)
            tracker.record_amp(self.cbm)
            return nll

        adv_w = (1.0 - alpha) * (1.0 - cw)
        s = (1.0 - alpha) * cw + alpha

        terms = []
        if adv_w > 0.0:
            self.cbm.eval()
            adv_data = self.attack.generate(
                born=self.cbm, naturals=data, labels=labels, eps_abs=eps_abs,
                device=self.device,
            )
            self.cbm.train()
            terms.append(adv_w * self.cbm.mixed_nll(adv_data, labels, alpha=0.0))
            # Amplitudes explode on the adversarial batch; record before the clean
            # forward overwrites the cache.
            tracker.record_amp(self.cbm)
        if s > 0.0:
            terms.append(s * self.cbm.mixed_nll(data, labels, alpha=alpha / s,
                                                debug=self._nc.debug))
            if adv_w <= 0.0:
                tracker.record_amp(self.cbm)

        return terms[0] if len(terms) == 1 else terms[0] + terms[1]

    # ------------------------------------------------------------------
    # Collapse diagnostics
    # ------------------------------------------------------------------

    def _diagnostics(self, data: torch.Tensor) -> Dict[str, float]:
        """log_Z and log|amp|² stats. Prefers the failing mixed_nll forward's
        stats (no extra contraction); falls back to a fresh no-grad recompute when
        it did not form them (e.g. alpha=0 leaves log_Z out)."""
        result: Dict[str, float] = self.cbm.forward_stats()
        _tiny = float(torch.finfo(torch.float32).tiny)

        if "log_Z" not in result:
            with torch.no_grad():
                try:
                    self.cbm.reset()
                    result["log_Z"] = self.cbm.log_partition_function().item()
                except Exception:
                    result["log_Z"] = float("nan")

        if "log_amp_sq_mean" not in result:
            with torch.no_grad():
                try:
                    amp = self.cbm.amplitudes(data)
                    log_abs_sq = 2.0 * torch.log(amp.abs().clamp(min=_tiny))
                    finite_mask = torch.isfinite(log_abs_sq)
                    finite = log_abs_sq[finite_mask]
                    result["log_amp_sq_mean"] = finite.mean().item() if finite.numel() else float("nan")
                    result["log_amp_sq_min"] = finite.min().item() if finite.numel() else float("nan")
                    result["log_amp_sq_max"] = finite.max().item() if finite.numel() else float("nan")
                    result["amp_nonfinite_count"] = int((~finite_mask).sum().item())
                    result["amp_nan_count"] = int(torch.isnan(log_abs_sq).sum().item())
                except Exception:
                    result["log_amp_sq_mean"] = float("nan")
                    result["log_amp_sq_min"] = float("nan")
                    result["log_amp_sq_max"] = float("nan")
                    result["amp_nonfinite_count"] = -1
                    result["amp_nan_count"] = -1
        return result

    @staticmethod
    def _format_diagnostics(d: Dict[str, float]) -> str:
        parts = []
        log_Z = d.get("log_Z", float("nan"))
        if not math.isfinite(log_Z):
            tag = "overflow" if log_Z > 0 else ("underflow" if log_Z < 0 else "nan")
            parts.append(f"norm {tag} (log_Z={log_Z:.4g})")
        else:
            headroom = d.get("log_Z_headroom", float("nan"))
            if math.isfinite(headroom):
                parts.append(f"log_Z={log_Z:.4g} (overflow headroom={headroom:.4g})")
            else:
                parts.append(f"log_Z={log_Z:.4g}")
        mean_ = d.get("log_amp_sq_mean", float("nan"))
        min_ = d.get("log_amp_sq_min", float("nan"))
        max_ = d.get("log_amp_sq_max", float("nan"))
        nf_count = d.get("amp_nonfinite_count", 0)
        if not math.isfinite(mean_):
            parts.append("amplitudes non-finite")
        else:
            s = f"log|amp|² mean={mean_:.4g} min={min_:.4g} max={max_:.4g}"
            if nf_count > 0:
                # The count comes from non-finite 2·log(|amp|.clamp(min=tiny)). The
                # clamp floors amplitude underflow, so a non-finite entry is either
                # an amplitude that overflowed float32 (+inf) or a degenerate
                # contraction (NaN); do not report the second as the first.
                nan_count = int(d.get("amp_nan_count", 0) or 0)
                if nan_count > 0 and nan_count >= nf_count:
                    cause = "NaN, degenerate contraction (zero amplitude?)"
                elif nan_count > 0:
                    cause = f"{nf_count - nan_count} overflow + {nan_count} NaN"
                else:
                    cause = "overflow"
                s += f" ({nf_count} non-finite → {cause})"
            parts.append(s)
        return "; ".join(parts)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def _train_epoch(self, eps_abs: float):
        """One pass over the training split. Sets ``_collapsed`` and stops early on
        a non-OOM error or a loss that stays non-finite after ``cbm.reset()``."""
        objectives, penalties = [], []
        self._collapsed = False
        tracker = NormTracker()
        self.cbm.train()

        for data, labels in self.train_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            self.step += 1

            try:
                nll = self._objective(data, labels, eps_abs, tracker)
                if not torch.isfinite(nll):
                    # Reset clears stale tensorkrowch contraction nodes; the retry
                    # is for recovery only.
                    self.cbm.reset()
                    nll = self._objective(data, labels, eps_abs, tracker)
                    if not torch.isfinite(nll):
                        diag = self._diagnostics(data)
                        logger.warning(
                            f"NaN/inf loss at step {self.step} (also after cbm.reset()): "
                            f"{self._format_diagnostics(diag)}"
                        )
                        self._collapsed = True
                        break
                    logger.warning(
                        f"NaN/inf loss at step {self.step} recovered after cbm.reset(); continuing."
                    )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA OOM at step {self.step}, re-raising.")
                    raise
                logger.warning(f"Training stopped at step {self.step}: {e}")
                self._collapsed = True
                break

            if self.norm_regularizer is not None:
                penalty = self.norm_regularizer(self.cbm)
                loss = nll + penalty
            else:
                penalty = None
                loss = nll

            # log Z is formed by mixed_nll (alpha>0) or the regularizer; read it
            # before optimizer.step() changes the parameters.
            tracker.record_logZ(self.cbm)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self._nc.hard_every > 0 and (self.step % self._nc.hard_every == 0):
                self.cbm.renormalize_(log_target=self._nc_log_target)

            objectives.append(nll.detach().cpu().item())
            penalties.append(penalty.detach().cpu().item() if penalty is not None else 0.0)

        n = len(objectives)
        self._train_objective = sum(objectives) / n if n else float("nan")
        self._train_penalty = sum(penalties) / n if n else float("nan")
        self._norm_stats = tracker.finalize(self.cbm)

    def _validate(self) -> dict:
        return evaluate(
            self.cbm, self.valid_loader, self.device,
            alpha=self.cfg.alpha,
            attack=self.attack,
            eps_abs=self.eps_abs if self.attack is not None else 0.0,
            clean_weight=self.clean_weight,
            adv_indices=self.adv_indices,
        )

    def _update(self, valid: dict):
        """Select on the validation objective (D8): keep the epoch if it is lower."""
        value = valid["objective"]
        if math.isfinite(value) and value < self.best["objective"]:
            self.best = dict(valid)
            self.best_tensors = [t.clone().detach() for t in self.cbm.tensors]
            self.best_epoch = self.epoch
            self.patience_counter = 0
        else:
            self.patience_counter += 1

    def _finish(self, output_dir: Optional[Path]):
        """Restore the best tensors and save them if asked."""
        self.cbm.initialize(tensors=self.best_tensors)
        self.cbm.reset()
        self.cbm.to("cpu")

        if not math.isfinite(self.best["objective"]):
            logger.warning(
                f"Best objective is {self.best['objective']} (no finite validation); "
                "skipping model save."
            )
        elif self.cfg.save and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.cbm.save(str(output_dir / "model"))
        logger.info("Training finished.")

    def train(
        self,
        on_epoch_end: Optional[Callable[[int, Dict], None]] = None,
        output_dir: Optional[Path] = None,
    ):
        """Run the training loop.

        ``on_epoch_end(epoch, record)`` gets ``{"train": {...}, "norm": {...}}``
        every epoch, plus ``"valid": {...}`` on validation epochs, with plain metric
        names (``rob`` as ``{eps_rel: value}``).
        """
        cfg = self.cfg
        self.step = 0
        self.epoch = 0
        self.best_epoch = 0
        self.patience_counter = 0
        self.epoch_times = []
        self._collapsed = False

        self.cbm.prepare(device=self.device)
        self._nc_log_target = resolve_log_target(self.cbm, self._nc)
        if self._nc.soft_strength > 0.0:
            self.norm_regularizer = NormRegularizer(
                strength=self._nc.soft_strength, log_target=self._nc_log_target
            )
        self.optimizer = optimizer(self.cbm.parameters(), cfg.optimizer)

        regime = "AT" if self.attack is not None else "NAT"
        logger.info(f"{regime} training begins (alpha={cfg.alpha:.3g}).")

        pbar = tqdm(range(cfg.max_epoch), desc=regime, unit="ep", dynamic_ncols=True)
        for epoch in pbar:
            epoch_start = time.perf_counter()
            self.epoch = epoch + 1

            eps_abs = self._eps_abs(self.epoch) if self.attack is not None else 0.0
            self._train_epoch(eps_abs)
            if self._collapsed:
                logger.info("Ending training early (see warning above).")
                break

            record = {
                "train": {"objective": self._train_objective, "penalty": self._train_penalty},
                "norm": self._norm_stats,
            }
            postfix = {
                "loss": f"{self._train_objective:.4f}",
                "logZ": f"{self._norm_stats.get('norm/log_Z_mean', float('nan')):.3g}",
            }
            if self.attack is not None:
                # The budget actually trained at, which follows the curriculum.
                record["train"]["eps_rel"] = eps_abs / self.range_size

            validated = self.epoch % cfg.eval_every == 0
            if validated:
                valid = self._validate()
                postfix["acc"] = f"{valid['acc']:.4f}"
                logged = dict(valid)
                if "rob" in valid:
                    # At the full radius, so the key is constant within a run; a
                    # subset estimator over n_rob samples.
                    postfix["rob"] = f"{valid['rob']:.4f}"
                    logged["rob"] = {self.eps_rel: valid["rob"]}
                record["valid"] = logged
            pbar.set_postfix(**postfix)

            if on_epoch_end is not None:
                on_epoch_end(self.epoch, record)

            if validated:
                self._update(valid)
            self.epoch_times.append(time.perf_counter() - epoch_start)

            if self.patience_counter > cfg.patience:
                logger.info(f"Early stopping after epoch {self.epoch}.")
                break

        self._finish(output_dir)
