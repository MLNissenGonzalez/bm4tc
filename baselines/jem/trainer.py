"""Conservative natural and adversarial trainers for the MLP/JEM baseline."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm

from src.utils.train import OptimizerConfig, optimizer

from .attacks import PGDConfig, pgd_classification
from .model import JEMMLP
from .sampler import SGLDSampler


@dataclass
class NaturalTrainerConfig:
    alpha: float = 0.0
    max_epoch: int = 100
    batch_size: int = 512
    patience: int = 30
    stop_crit: str = "acc"
    input_noise_std: float = 0.0
    energy_l2: float = 1e-4
    grad_clip: float | None = 10.0
    save: bool = True
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            name="adam", kwargs={"lr": 3e-4, "weight_decay": 0.0}
        )
    )


@dataclass
class ValidationSamplerConfig:
    """Standardized, stronger sampler used only for model selection."""

    num_steps: int = 100
    step_size: float = 0.05
    noise_std: float = 0.005
    reinit_probability: float = 0.1
    buffer_size: int = 2048
    batch_size: int = 256
    num_batches: int = 4
    seed: int = 12345


@dataclass
class AdversarialTrainerConfig:
    """Purely discriminative MLP adversarial training (alpha is intentionally absent)."""

    max_epoch: int = 100
    batch_size: int = 512
    patience: int = 20
    clean_weight: float = 0.2
    epsilon: float = 0.2
    attack_steps: int = 10
    attack_step_size: float | None = None
    eval_rob_freq: int = 5
    grad_clip: float | None = 10.0
    save: bool = True
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            name="adam", kwargs={"lr": 3e-4, "weight_decay": 0.0}
        )
    )


def evaluate_classifier(model, loader, device) -> dict[str, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            logits = model(data)
            loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    return {
        "dis_loss": loss_sum / total if total else float("nan"),
        "acc": correct / total if total else float("nan"),
    }


def evaluate_jem(
    model: JEMMLP,
    loader,
    sampler: SGLDSampler,
    alpha: float,
    cfg: ValidationSamplerConfig,
    device,
    epoch: int,
) -> dict[str, float]:
    """Validation CE and CD surrogate using a standardized negative sampler.

    The sampler configuration is independent of the candidate training sampler,
    preventing weak training chains from obtaining an artificially low CD loss.
    """
    model.eval()
    total, correct, ce_sum, pos_score_sum = 0, 0, 0.0, 0.0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            logits = model(data)
            ce_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            pos_score_sum += torch.logsumexp(logits, dim=-1).sum().item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    dis_loss = ce_sum / total
    positive_score = pos_score_sum / total

    cuda_devices = []
    if torch.device(device).type == "cuda":
        index = torch.device(device).index
        cuda_devices = [torch.cuda.current_device() if index is None else index]
    negatives, start_scores, displacements, boundary_fractions = [], [], [], []
    # Isolate validation randomness from training and make trials comparable.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(cfg.seed + epoch)
        if cuda_devices:
            torch.cuda.manual_seed_all(cfg.seed + epoch)
        for _ in range(cfg.num_batches):
            initial, indices = sampler.buffer.initial(
                cfg.batch_size, cfg.reinit_probability, device
            )
            with torch.no_grad():
                start_scores.append(model.marginal_score(initial).mean().item())
            samples = sampler.refine(
                model,
                initial,
                num_steps=cfg.num_steps,
                step_size=cfg.step_size,
                noise_std=cfg.noise_std,
            )
            sampler.buffer.update(indices, samples)
            with torch.no_grad():
                negatives.append(model.marginal_score(samples).mean().item())
                displacements.append(
                    (samples - initial).flatten(1).norm(dim=1).mean().item()
                )
                lo, hi = model.input_range
                at_boundary = ((samples <= lo + 1e-5) | (samples >= hi - 1e-5))
                boundary_fractions.append(at_boundary.float().mean().item())

    negative_score = sum(negatives) / len(negatives)
    px_cd_loss = negative_score - positive_score
    joint_cd_loss = dis_loss + px_cd_loss
    mixed_loss = dis_loss + alpha * px_cd_loss
    return {
        "dis_loss": dis_loss,
        "acc": correct / total,
        "px_cd_loss": px_cd_loss,
        "joint_cd_loss": joint_cd_loss,
        "gen_loss": joint_cd_loss,
        "mixed_loss": mixed_loss,
        "positive_marginal_score": positive_score,
        "negative_marginal_score": negative_score,
        "sgld_score_start": sum(start_scores) / len(start_scores),
        "sgld_score_gain": negative_score - sum(start_scores) / len(start_scores),
        "sgld_displacement_l2": sum(displacements) / len(displacements),
        "sgld_boundary_fraction": sum(boundary_fractions) / len(boundary_fractions),
    }


def make_json_logger(output_dir: Path, wandb_run=None):
    records: list[dict] = []
    path = output_dir / "log.json"

    def callback(epoch: int, metrics: dict) -> None:
        records.append({"epoch": epoch, **metrics})
        path.write_text(json.dumps(records, indent=2))
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, **metrics})

    return callback


class NaturalTrainer:
    def __init__(
        self,
        model,
        cfg,
        datahandler,
        sampler,
        validation_sampler,
        validation_sampler_cfg,
        device,
    ):
        self.model: JEMMLP = model
        self.cfg: NaturalTrainerConfig = cfg
        self.datahandler = datahandler
        self.sampler: SGLDSampler = sampler
        self.validation_sampler: SGLDSampler = validation_sampler
        self.validation_sampler_cfg: ValidationSamplerConfig = validation_sampler_cfg
        self.device = device
        self.best = {
            "acc": 0.0,
            "dis_loss": float("inf"),
            "mixed_loss": float("inf"),
            "px_cd_loss": float("inf"),
        }
        self.best_state = None
        self.best_buffer_state = None
        self.best_epoch = 0

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        totals = {
            "loss": 0.0,
            "dis": 0.0,
            "px_cd": 0.0,
            "joint_cd": 0.0,
            "sgld_score_gain": 0.0,
            "sgld_displacement_l2": 0.0,
            "sgld_boundary_fraction": 0.0,
        }
        count = 0
        for data, labels in self.datahandler.classification["train"]:
            data, labels = data.to(self.device), labels.to(self.device)
            positives = data
            if self.cfg.input_noise_std > 0:
                positives = (
                    data + self.cfg.input_noise_std * torch.randn_like(data)
                ).clamp(*self.model.input_range)
            negatives = None
            if self.cfg.alpha > 0:
                negatives = self.sampler.sample_training(
                    self.model, len(data), self.device
                )
                diagnostics = self.sampler.last_diagnostics
                totals["sgld_score_gain"] += diagnostics.get("score_gain", 0.0)
                totals["sgld_displacement_l2"] += diagnostics.get(
                    "displacement_l2", 0.0
                )
                totals["sgld_boundary_fraction"] += diagnostics.get(
                    "boundary_fraction", 0.0
                )
            loss, terms = self.model.mixed_loss(
                positives, labels, negatives, self.cfg.alpha
            )
            if self.cfg.alpha > 0 and self.cfg.energy_l2 > 0:
                pos_score = self.model.marginal_score(positives)
                neg_score = self.model.marginal_score(negatives.detach())
                loss = loss + self.cfg.energy_l2 * (
                    pos_score.square().mean() + neg_score.square().mean()
                )
            self.optim.zero_grad()
            loss.backward()
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optim.step()
            totals["loss"] += loss.item()
            totals["dis"] += terms["dis_loss"].item()
            totals["px_cd"] += terms["px_cd_loss"].item()
            totals["joint_cd"] += terms["joint_cd_loss"].item()
            count += 1
        return {k: v / max(count, 1) for k, v in totals.items()}

    def train(self, callback=None, output_dir: Path | None = None) -> None:
        self.optim = optimizer(self.model.parameters(), self.cfg.optimizer)
        patience = 0
        for epoch in tqdm(range(1, self.cfg.max_epoch + 1), desc="JEM", unit="ep"):
            start = time.perf_counter()
            train = self._train_epoch()
            if self.cfg.alpha > 0:
                valid = evaluate_jem(
                    self.model,
                    self.datahandler.classification["valid"],
                    self.validation_sampler,
                    self.cfg.alpha,
                    self.validation_sampler_cfg,
                    self.device,
                    epoch,
                )
            else:
                valid = evaluate_classifier(
                    self.model, self.datahandler.classification["valid"], self.device
                )
                valid.update(
                    {
                        "px_cd_loss": float("nan"),
                        "joint_cd_loss": float("nan"),
                        "gen_loss": float("nan"),
                        "mixed_loss": valid["dis_loss"],
                    }
                )
            metrics = {
                "nll/train": train["loss"],
                "dis_loss/train": train["dis"],
                "px_cd_loss/train": train["px_cd"],
                "joint_cd_loss/train": train["joint_cd"],
                "gen_loss/train": train["joint_cd"],
                "dis_loss/valid": valid["dis_loss"],
                "px_cd_loss/valid": valid["px_cd_loss"],
                "joint_cd_loss/valid": valid["joint_cd_loss"],
                "gen_loss/valid": valid["joint_cd_loss"],
                "mixed_loss/valid": valid["mixed_loss"],
                "acc/valid": valid["acc"],
                "time/epoch": time.perf_counter() - start,
            }
            if self.cfg.alpha > 0 and self.sampler.cfg.track_diagnostics:
                metrics.update(
                    {
                        "sgld_score_gain/train": train["sgld_score_gain"],
                        "sgld_displacement_l2/train": train[
                            "sgld_displacement_l2"
                        ],
                        "sgld_boundary_fraction/train": train[
                            "sgld_boundary_fraction"
                        ],
                    }
                )
            for key in (
                "positive_marginal_score",
                "negative_marginal_score",
                "sgld_score_start",
                "sgld_score_gain",
                "sgld_displacement_l2",
                "sgld_boundary_fraction",
            ):
                if key in valid:
                    metrics[f"{key}/valid"] = valid[key]
            if callback:
                callback(epoch, metrics)
            if self.cfg.stop_crit == "acc":
                improved = valid["acc"] > self.best["acc"]
            elif self.cfg.stop_crit in {"dis_loss", "mixed_loss", "px_cd_loss"}:
                improved = valid[self.cfg.stop_crit] < self.best[self.cfg.stop_crit]
            else:
                raise ValueError(
                    "stop_crit must be acc, dis_loss, mixed_loss or px_cd_loss"
                )
            if improved:
                self.best = valid
                self.best_epoch = epoch
                self.best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.best_buffer_state = {
                    "data": self.sampler.buffer.data.detach().cpu().clone()
                }
                patience = 0
            else:
                patience += 1
            if patience > self.cfg.patience:
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if self.best_buffer_state is not None:
            self.sampler.buffer.load_state_dict(self.best_buffer_state)
        if self.cfg.save and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(
                output_dir / "model.pt",
                best=self.best,
                best_epoch=self.best_epoch,
                replay_buffer=self.sampler.buffer.state_dict(),
                sampler_config=asdict(self.sampler.cfg),
            )


class AdversarialTrainer:
    def __init__(self, model, cfg, datahandler, device):
        self.model: JEMMLP = model
        self.cfg: AdversarialTrainerConfig = cfg
        self.datahandler = datahandler
        self.device = device
        self.best = {"acc": 0.0, "rob": float("-inf")}
        self.best_state = None

    def train(self, callback=None, output_dir: Path | None = None) -> None:
        optim = optimizer(self.model.parameters(), self.cfg.optimizer)
        attack_cfg = PGDConfig(
            epsilon=self.cfg.epsilon,
            num_steps=self.cfg.attack_steps,
            step_size=self.cfg.attack_step_size,
        )
        patience = 0
        for epoch in tqdm(range(1, self.cfg.max_epoch + 1), desc="MLP-AT", unit="ep"):
            self.model.train()
            running, count = 0.0, 0
            for data, labels in self.datahandler.classification["train"]:
                data, labels = data.to(self.device), labels.to(self.device)
                self.model.eval()
                adv = pgd_classification(self.model, data, labels, attack_cfg)
                self.model.train()
                adv_loss = F.cross_entropy(self.model(adv), labels)
                clean_loss = F.cross_entropy(self.model(data), labels)
                loss = (
                    (1.0 - self.cfg.clean_weight) * adv_loss
                    + self.cfg.clean_weight * clean_loss
                )
                optim.zero_grad()
                loss.backward()
                if self.cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                optim.step()
                running += loss.item()
                count += 1
            valid = evaluate_classifier(
                self.model, self.datahandler.classification["valid"], self.device
            )
            metrics = {
                "at/train": running / max(count, 1),
                "dis_loss/valid": valid["dis_loss"],
                "acc/valid": valid["acc"],
            }
            rob = None
            if epoch % self.cfg.eval_rob_freq == 0:
                robust_correct, robust_total = 0, 0
                self.model.eval()
                for data, labels in self.datahandler.classification["valid"]:
                    data, labels = data.to(self.device), labels.to(self.device)
                    adv = pgd_classification(self.model, data, labels, attack_cfg)
                    with torch.no_grad():
                        robust_correct += (
                            self.model(adv).argmax(1) == labels
                        ).sum().item()
                        robust_total += len(labels)
                rob = robust_correct / robust_total
                metrics["rob/valid"] = rob
            if callback:
                callback(epoch, metrics)
            if rob is not None and rob > self.best["rob"]:
                self.best = {**valid, "rob": rob}
                self.best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience = 0
            elif rob is not None:
                patience += 1
            if patience > self.cfg.patience:
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if self.cfg.save and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(output_dir / "model.pt", best=self.best)
