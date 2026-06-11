"""NLL trainer for ConditionalBornMachine (discriminative, generative, or mixed)."""

import math
import time
import torch
from tqdm import tqdm
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from src.utils.train import OptimizerConfig, NormRegularizer, eval_metrics, optimizer
from src.datahandler import DataHandler
from src.model import ConditionalBornMachine

import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def _nullctx():
    yield


_LOSS_METRICS = {"dis_loss", "gen_loss"}
_ACC_METRICS = {"acc", "rob"}
_VALID_STOP_CRIT = {"dis_loss", "gen_loss", "acc", "rob"}


@dataclass
class NormControlConfig:
    log_target: Optional[Union[float, str]] = 0.0
    hard_every: int = 1
    soft_strength: float = 0.0
    debug: bool = False


@dataclass
class NLLConfig:
    alpha: float = 0.0
    max_epoch: int = 100
    batch_size: int = 64
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    stop_crit: str = "acc"
    patience: int = 250
    norm_control: NormControlConfig = field(default_factory=NormControlConfig)
    save: bool = False


class NLLTrainer:
    """
    Unified NLL trainer for ConditionalBornMachine.

    alpha=0  → pure discriminative (-log p(c|x))
    alpha>0  → mixed NLL; alpha=1 → pure generative (-log p(x,c))
    Both dis_loss and gen_loss are always reported on the validation set.
    """

    def __init__(
        self,
        cbm: ConditionalBornMachine,
        train_cfg: NLLConfig,
        datahandler: DataHandler,
        device: torch.device,
    ):
        self.cbm = cbm
        self.datahandler = datahandler
        self.device = device
        self.train_cfg = train_cfg

        if getattr(self.datahandler, "classification", None) is None:
            self.datahandler.get_classification_loaders(batch_size=train_cfg.batch_size)

        self._nc = train_cfg.norm_control
        self.norm_regularizer: NormRegularizer | None = None
        self._nc_log_target: float | None = None

        self._init_best()
        self.best_tensors = [t.cpu().clone().detach() for t in cbm.tensors]
        self.step = 0
        self._collapsed = False

    def _init_best(self):
        self.best = {"gen_loss": float("inf"), "dis_loss": float("inf"), "acc": 0.0}
        self.stopping_criterion_name = self.train_cfg.stop_crit
        if self.stopping_criterion_name not in _VALID_STOP_CRIT:
            raise ValueError(
                f"Invalid stop_crit '{self.stopping_criterion_name}'. "
                f"Must be one of: {sorted(_VALID_STOP_CRIT)}"
            )

    def _resolve_nc_log_target(self) -> float:
        import math as _math
        raw = self._nc.log_target

        if raw is None:
            with torch.no_grad():
                log_Z0 = self.cbm.log_partition_function()
            log_target = log_Z0.item()
            logger.info(f"NormControl: log_target (pretrained) = {log_target:.6g}")
            return log_target

        if isinstance(raw, str):
            n_features = self.cbm.n_features
            data_dim = self.datahandler.data_dim
            in_dim = self.cbm.in_dim
            out_dim = self.cbm.out_dim
            bond_dim = self.cbm.bond_dim
            _ns = {
                "__builtins__": {},
                "n_features": n_features,
                "data_dim": data_dim,
                "in_dim": in_dim,
                "out_dim": out_dim,
                "bond_dim": bond_dim,
                "sqrt": _math.sqrt,
                "log": _math.log,
                "exp": _math.exp,
            }
            try:
                result = eval(raw, _ns)  # noqa: S307
            except Exception as exc:
                raise ValueError(
                    f"NormControl: could not evaluate log_target expression "
                    f"{raw!r} (n_features={n_features}, data_dim={data_dim}, "
                    f"in_dim={in_dim}, out_dim={out_dim}, bond_dim={bond_dim}): {exc}"
                ) from exc
            log_target = float(result)
            if not _math.isfinite(log_target):
                raise ValueError(
                    f"NormControl: log_target expression {raw!r} evaluated to "
                    f"{log_target}, but log_target must be finite."
                )
            logger.info(
                f"NormControl: log_target (expression {raw!r}) = {log_target:.6g} "
                f"[n_features={n_features}, data_dim={data_dim}, "
                f"in_dim={in_dim}, out_dim={out_dim}, bond_dim={bond_dim}]"
            )
            return log_target

        return float(raw)

    def _diagnostics(self, data: torch.Tensor) -> Dict[str, float]:
        """Compute log_Z and log|amp|² stats. No-grad; safe to call mid-training."""
        result: Dict[str, float] = {}
        _tiny = float(torch.finfo(torch.float32).tiny)
        with torch.no_grad():
            try:
                self.cbm.reset()
                log_Z = self.cbm.log_partition_function()
                result["log_Z"] = log_Z.item()
            except Exception:
                result["log_Z"] = float("nan")
            try:
                amp = self.cbm.amplitudes(data)
                log_abs_sq = 2.0 * torch.log(amp.abs().clamp(min=_tiny))
                finite_mask = torch.isfinite(log_abs_sq)
                finite = log_abs_sq[finite_mask]
                result["log_amp_sq_mean"] = finite.mean().item() if finite.numel() else float("nan")
                result["log_amp_sq_min"] = finite.min().item() if finite.numel() else float("nan")
                result["log_amp_sq_max"] = finite.max().item() if finite.numel() else float("nan")
                result["amp_nonfinite_frac"] = (~finite_mask).float().mean().item()
            except Exception:
                result["log_amp_sq_mean"] = float("nan")
                result["log_amp_sq_min"] = float("nan")
                result["log_amp_sq_max"] = float("nan")
                result["amp_nonfinite_frac"] = float("nan")
        return result

    def _nan_loss_diagnostics(
        self, data: torch.Tensor, labels: torch.Tensor, alpha: float
    ) -> str:
        """Decompose mixed_nll terms under no-grad to locate the NaN source."""
        _tiny = float(torch.finfo(torch.float32).tiny)
        parts = []
        with torch.no_grad():
            try:
                amp = self.cbm.amplitudes(data)           # (B, C)
                B = data.shape[0]

                # term1: -2 log |ψ(x, c_true)|
                a1 = amp[torch.arange(B), labels].abs().clamp(min=_tiny)
                t1 = -2.0 * torch.log(a1)
                nan1 = torch.isnan(t1) | torch.isinf(t1)
                parts.append(
                    f"term1(-2·log|ψ(x,c)|): "
                    f"mean={t1[~nan1].mean().item():.4g} "
                    f"max={t1[~nan1].max().item():.4g} "
                    f"nonfinite={nan1.float().mean().item():.1%}"
                )

                # term2: (1-α) log Σ_c |ψ|²  — zero for alpha=1
                if alpha < 1.0:
                    abs_sq_sum = amp.abs().pow(2).sum(dim=-1).clamp(min=_tiny)
                    t2 = (1.0 - alpha) * torch.log(abs_sq_sum)
                    nan2 = torch.isnan(t2) | torch.isinf(t2)
                    parts.append(
                        f"term2((1-α)·log Σ|ψ|²): "
                        f"mean={t2[~nan2].mean().item():.4g} "
                        f"nonfinite={nan2.float().mean().item():.1%}"
                    )
                else:
                    parts.append("term2=0 (α=1)")

                # term3: α·log_Z
                if alpha > 0.0:
                    log_Z = self.cbm.log_partition_function()
                    parts.append(f"term3(α·log_Z): {alpha * log_Z.item():.4g}")

                # combined
                t2_full = (
                    torch.zeros(B, device=data.device)
                    if alpha >= 1.0
                    else (1.0 - alpha) * torch.log(amp.abs().pow(2).sum(dim=-1).clamp(min=_tiny))
                )
                log_Z_val = self.cbm.log_partition_function() if alpha > 0.0 else torch.tensor(0.0)
                loss_recomputed = (t1 + t2_full + alpha * log_Z_val).mean()
                parts.append(f"recomputed_loss={loss_recomputed.item():.4g}")

            except Exception as exc:
                parts.append(f"(diagnostic failed: {exc})")
        return " | ".join(parts)

    def _format_diagnostics(self, d: Dict[str, float]) -> str:
        parts = []
        log_Z = d.get("log_Z", float("nan"))
        if not math.isfinite(log_Z):
            tag = "overflow" if log_Z > 0 else ("underflow" if log_Z < 0 else "nan")
            parts.append(f"norm {tag} (log_Z={log_Z:.4g})")
        else:
            parts.append(f"log_Z={log_Z:.4g}")
        mean_ = d.get("log_amp_sq_mean", float("nan"))
        min_ = d.get("log_amp_sq_min", float("nan"))
        max_ = d.get("log_amp_sq_max", float("nan"))
        nf = d.get("amp_nonfinite_frac", 0.0)
        if not math.isfinite(mean_):
            parts.append("amplitudes non-finite")
        else:
            s = f"log|amp|² mean={mean_:.4g} min={min_:.4g} max={max_:.4g}"
            if nf > 0:
                tag = "overflow" if mean_ > 80 else "underflow"
                s += f" ({nf:.1%} non-finite → {tag})"
            parts.append(s)
        return "; ".join(parts)

    def _train_epoch(self):
        losses = []
        self._collapsed = False
        self._debug_diags: list = []
        self.cbm.train()

        for data, labels in self.datahandler.classification["train"]:
            data, labels = data.to(self.device), labels.to(self.device)
            self.step += 1

            try:
                ctx = torch.autograd.detect_anomaly() if self._nc.debug else _nullctx()
                with ctx:
                    loss = self.cbm.mixed_nll(data, labels, self.train_cfg.alpha)
            except RuntimeError as e:
                msg = str(e)
                if "out of memory" in msg.lower():
                    logger.error(f"CUDA OOM at step {self.step} — re-raising.")
                    raise
                logger.warning(f"Training stopped at step {self.step}: {msg}")
                self._collapsed = True
                break

            if torch.isnan(loss) or torch.isinf(loss):
                self.cbm.reset()
                loss = self.cbm.mixed_nll(data, labels, self.train_cfg.alpha)
                if torch.isnan(loss) or torch.isinf(loss):
                    diag = self._diagnostics(data)
                    term_info = self._nan_loss_diagnostics(data, labels, self.train_cfg.alpha)
                    logger.warning(
                        f"NaN/inf loss at step {self.step} (also after cbm.reset()) — "
                        f"{self._format_diagnostics(diag)}\n"
                        f"  term breakdown (no-grad): {term_info}"
                    )
                    self._collapsed = True
                    break
                logger.warning(
                    f"NaN/inf loss at step {self.step} recovered after cbm.reset(); continuing."
                )

            if self.norm_regularizer is not None:
                loss = loss + self.norm_regularizer(self.cbm)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self._nc.hard_every > 0 and (self.step % self._nc.hard_every == 0):
                self.cbm.renormalize_(log_target=self._nc_log_target)

            if self._nc.debug:
                diag = self._diagnostics(data)
                logger.debug(f"step={self.step}: {self._format_diagnostics(diag)}")
                self._debug_diags.append(diag)

            losses.append(loss.detach().cpu().item())

        self._train_loss = sum(losses) / len(losses) if losses else float("nan")

    def _update(self):
        current_value = self.valid_perf.get(self.stopping_criterion_name)

        if current_value is None or not math.isfinite(current_value):
            return

        former_best = self.best.get(
            self.stopping_criterion_name,
            0.0 if self.stopping_criterion_name in _ACC_METRICS else float("inf"),
        )

        if self.stopping_criterion_name in _ACC_METRICS:
            improved = current_value > former_best
        elif self.stopping_criterion_name in _LOSS_METRICS:
            improved = current_value < former_best
        else:
            raise ValueError(f"Unknown stopping criterion: {self.stopping_criterion_name}")

        if improved:
            self.best = dict(self.valid_perf)
            self.best_tensors = [t.clone().detach() for t in self.cbm.tensors]
            self.best_epoch = self.epoch
            self.patience_counter = 0
        else:
            self.patience_counter += 1

    def _summarise_training(self, output_dir: Optional[Path]):
        self.cbm.initialize(tensors=self.best_tensors)
        self.cbm.reset()
        self.cbm.to("cpu")

        if hasattr(self, "valid_perf"):
            del self.valid_perf

        best_stop = self.best.get(self.stopping_criterion_name)
        if best_stop is not None and not math.isfinite(best_stop):
            logger.warning(
                f"Best {self.stopping_criterion_name} is {best_stop} (non-finite); skipping model save."
            )
        elif self.train_cfg.save and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.cbm.save(str(output_dir / "model"))

        logger.info("NLL training finished.")

    def train(
        self,
        on_epoch_end: Optional[Callable[[int, Dict], None]] = None,
        output_dir: Optional[Path] = None,
    ):
        """Run the NLL training loop."""
        self.step, self.patience_counter = 0, 0
        self.epoch = 0
        self.best_epoch = 0
        self.epoch_times = []
        self._collapsed = False

        self.cbm.prepare(device=self.device)
        self._nc_log_target = self._resolve_nc_log_target()
        if self._nc.soft_strength > 0.0:
            self.norm_regularizer = NormRegularizer(
                strength=self._nc.soft_strength, log_target=self._nc_log_target
            )
        self.optimizer = optimizer(self.cbm.parameters(), self.train_cfg.optimizer)

        logger.info(f"NLL training begins (alpha={self.train_cfg.alpha:.3g}).")

        pbar = tqdm(range(self.train_cfg.max_epoch), desc="NLL", unit="ep", dynamic_ncols=True)
        for epoch in pbar:
            epoch_start = time.perf_counter()
            self.epoch = epoch + 1

            self._train_epoch()

            if self._collapsed:
                logger.info("Ending training early (see warning above).")
                break

            dis_loss, acc, gen_loss = eval_metrics(
                self.cbm, self.datahandler.classification["valid"], self.device
            )
            self.valid_perf = {"dis_loss": dis_loss, "acc": acc, "gen_loss": gen_loss}

            pbar.set_postfix(
                loss=f"{self._train_loss:.4f}",
                dis=f"{dis_loss:.4f}",
                acc=f"{acc:.4f}",
            )

            if on_epoch_end is not None:
                metrics = {
                    "nll/train":      self._train_loss,
                    "dis_loss/valid": dis_loss,
                    "gen_loss/valid": gen_loss,
                    "acc/valid":      acc,
                }
                if self._nc.debug and self._debug_diags:
                    log_Zs = [d["log_Z"] for d in self._debug_diags if math.isfinite(d.get("log_Z", float("nan")))]
                    means  = [d["log_amp_sq_mean"] for d in self._debug_diags if math.isfinite(d.get("log_amp_sq_mean", float("nan")))]
                    mins   = [d["log_amp_sq_min"] for d in self._debug_diags if math.isfinite(d.get("log_amp_sq_min", float("nan")))]
                    if log_Zs:
                        metrics["norm/log_Z_mean"] = sum(log_Zs) / len(log_Zs)
                        metrics["norm/log_Z_min"]  = min(log_Zs)
                        metrics["norm/log_Z_max"]  = max(log_Zs)
                    if means:
                        metrics["norm/log_amp_sq_mean"] = sum(means) / len(means)
                    if mins:
                        metrics["norm/log_amp_sq_min"] = min(mins)
                    logger.info(
                        f"[debug] epoch {self.epoch}: "
                        + self._format_diagnostics({
                            "log_Z":         metrics.get("norm/log_Z_mean", float("nan")),
                            "log_amp_sq_mean": metrics.get("norm/log_amp_sq_mean", float("nan")),
                            "log_amp_sq_min":  metrics.get("norm/log_amp_sq_min", float("nan")),
                            "amp_nonfinite_frac": 0.0,
                        })
                    )
                on_epoch_end(self.epoch, metrics)

            self._update()
            self.epoch_times.append(time.perf_counter() - epoch_start)

            if self.patience_counter > self.train_cfg.patience:
                logger.info(f"Early stopping after epoch {self.epoch}.")
                break

        self._summarise_training(output_dir)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    import torch
    from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig
    from src.datahandler import DataHandler
    from src.datahandler import DatasetConfig, DataGenDowConfig

    device = torch.device("cpu")

    def _make_cbm():
        cfg = CBMConfig(
            embedding="legendre",
            init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3),
        )
        return ConditionalBornMachine(cfg=cfg, data_dim=2, num_classes=2, device=device)

    ds_cfg = DatasetConfig(
        name="spirals",
        gen_dow_kwargs=DataGenDowConfig(name="spirals", size=32, seed=42, noise=0.1),
        overwrite=True,
    )
    dh = DataHandler(ds_cfg)
    dh.load()
    dh.split_and_rescale(_make_cbm())

    _EXPECTED_KEYS = {"dis_loss/valid", "gen_loss/valid", "acc/valid", "nll/train"}

    for alpha in (0.0, 1.0):
        cbm = _make_cbm()
        train_cfg = NLLConfig(max_epoch=5, batch_size=4, patience=250, alpha=alpha)
        trainer = NLLTrainer(cbm=cbm, train_cfg=train_cfg, datahandler=dh, device=device)

        logged = []
        trainer.train(on_epoch_end=lambda ep, m: logged.append((ep, m)))

        assert len(logged) == 5, f"alpha={alpha}: expected 5 epochs, got {len(logged)}"
        last_ep, last_m = logged[-1]
        missing = _EXPECTED_KEYS - last_m.keys()
        assert not missing, f"alpha={alpha}: missing keys {missing}"
        print(
            f"  alpha={alpha}  epoch={last_ep}"
            f"  dis_loss/valid={last_m['dis_loss/valid']:.4f}"
            f"  gen_loss/valid={last_m['gen_loss/valid']:.4f}"
            f"  acc/valid={last_m['acc/valid']:.4f}"
        )

    print("\nnll.py smoke test passed.")
