"""NLL trainer for ConditionalBornMachine (discriminative, generative, or mixed)."""

import math
import time
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from src.utils.train import OptimizerConfig, NormRegularizer, eval_metrics, optimizer
from src.data.handler import DataHandler
from src.model import ConditionalBornMachine

import logging
logger = logging.getLogger(__name__)

_LOSS_METRICS = {"dis_loss", "gen_loss"}
_ACC_METRICS = {"acc", "rob"}
_VALID_STOP_CRIT = {"dis_loss", "gen_loss", "acc", "rob"}


@dataclass
class NormControlConfig:
    target: Optional[Union[float, str]] = 1.0
    hard_every: int = 1
    soft_strength: float = 0.0


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
        self._nc_target: float | None = None

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

    def _resolve_nc_target(self) -> float:
        import math as _math
        raw = self._nc.target

        if raw is None:
            with torch.no_grad():
                log_Z0 = self.cbm.log_partition_function()
            target = torch.exp(log_Z0).item()
            logger.info(f"NormControl: target Z (pretrained) = {target:.6g}")
            return target

        if isinstance(raw, str):
            n_features = self.cbm.n_features
            data_dim = self.datahandler.data_dim
            _ns = {
                "__builtins__": {},
                "n_features": n_features,
                "data_dim": data_dim,
                "sqrt": _math.sqrt,
                "log": _math.log,
                "exp": _math.exp,
            }
            try:
                result = eval(raw, _ns)  # noqa: S307
            except Exception as exc:
                raise ValueError(
                    f"NormControl: could not evaluate target expression "
                    f"{raw!r} (n_features={n_features}, data_dim={data_dim}): {exc}"
                ) from exc
            target = float(result)
            if target <= 0:
                raise ValueError(
                    f"NormControl: target expression {raw!r} evaluated to "
                    f"{target}, but Z must be positive."
                )
            logger.info(
                f"NormControl: target Z (expression {raw!r}) = {target:.6g} "
                f"[n_features={n_features}, data_dim={data_dim}]"
            )
            return target

        return float(raw)

    def _train_epoch(self):
        losses = []
        self._collapsed = False
        self.cbm.train()

        for data, labels in self.datahandler.classification["train"]:
            data, labels = data.to(self.device), labels.to(self.device)
            self.step += 1

            try:
                loss = self.cbm.mixed_nll(data, labels, self.train_cfg.alpha)
            except RuntimeError as e:
                logger.warning(f"Norm or amplitude over-/underflow ({e}). Stopping early.")
                self._collapsed = True
                break

            if torch.isnan(loss):
                logger.warning("NaN loss detected — aborting epoch.")
                self._collapsed = True
                break

            if self.norm_regularizer is not None:
                loss = loss + self.norm_regularizer(self.cbm)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self._nc.hard_every > 0 and (self.step % self._nc.hard_every == 0):
                self.cbm.renormalize_(target=self._nc_target)

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
        self._nc_target = self._resolve_nc_target()
        if self._nc.soft_strength > 0.0:
            self.norm_regularizer = NormRegularizer(
                strength=self._nc.soft_strength, target=self._nc_target
            )
        self.optimizer = optimizer(self.cbm.parameters(), self.train_cfg.optimizer)

        logger.info(f"NLL training begins (alpha={self.train_cfg.alpha:.3g}).")

        for epoch in range(self.train_cfg.max_epoch):
            epoch_start = time.perf_counter()
            self.epoch = epoch + 1

            self._train_epoch()

            if self._collapsed:
                logger.info("Norm collapsed or NaN loss; ending training.")
                break

            dis_loss, acc, gen_loss = eval_metrics(
                self.cbm, self.datahandler.classification["valid"], self.device
            )
            self.valid_perf = {"dis_loss": dis_loss, "acc": acc, "gen_loss": gen_loss}

            if on_epoch_end is not None:
                on_epoch_end(self.epoch, {
                    "nll/train":      self._train_loss,
                    "dis_loss/valid": dis_loss,
                    "gen_loss/valid": gen_loss,
                    "acc/valid":      acc,
                })

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
    from src.data.handler import DataHandler
    from src.data.gen_n_load import DatasetConfig, DataGenDowConfig

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
