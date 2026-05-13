"""
Generative trainer for BornMachine using NLL minimization.

Trains p(x|c) by minimizing negative log-likelihood.
"""

import math
from pathlib import Path
import time
import torch
from typing import Dict
from src.utils import schemas, get
import wandb
from src.tracking import record, log_grads
from src.data.handler import DataHandler
from src.models import BornMachine
from src.utils.criterions import NormRegularizer
from src.trainer.eval import eval_dis, eval_gen

import logging
logger = logging.getLogger(__name__)

_LOSS_METRICS = {"dis_loss", "gen_loss"}
_ACC_METRICS = {"acc", "rob"}
_VALID_STOP_CRIT = {"dis_loss", "gen_loss", "acc", "rob"}


class Trainer:
    """
    Generative trainer for BornMachine using NLL minimization.

    Trains the generator view of the BornMachine by minimizing negative
    log-likelihood of p(x|c).

    Attributes:
        best: Dict of best metric values achieved during training.
        best_tensors: Tensors from the best-performing epoch.
    """

    def __init__(
            self,
            bornmachine: BornMachine,
            cfg: schemas.Config,
            datahandler: DataHandler,
            device: torch.device
    ):
        self.datahandler = datahandler
        self.device = device
        self.cfg = cfg
        self.stage = "gen"
        self.train_cfg = cfg.trainer.generative

        if getattr(self.datahandler, "classification", None) is None:
            self.datahandler.get_classification_loaders(batch_size=self.train_cfg.batch_size)

        self.criterion = get.criterion("generative", self.train_cfg.criterion)

        wandb.define_metric(f"{self.stage}/train/loss", summary="none")

        self._nc = self.train_cfg.norm_control
        self.norm_regularizer: NormRegularizer | None = None
        self._nc_target: float | None = None
        if self._nc.soft_strength > 0.0:
            wandb.define_metric(f"{self.stage}/train/norm_reg", summary="none")

        self._init_best()

        self.bornmachine = bornmachine
        self.best_tensors = [t.cpu().clone().detach() for t in self.bornmachine.generator.tensors]

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
                log_Z0 = self.bornmachine.generator.log_partition_function()
            target = torch.exp(log_Z0).item()
            logger.info(f"NormControl: target Z (pretrained) = {target:.6g}")
            return target

        if isinstance(raw, str):
            n_features = self.bornmachine.generator.n_features
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
        self._norm_collapsed = False

        for data, labels in self.datahandler.classification["train"]:
            data, labels = data.to(self.device), labels.to(self.device)
            self.step += 1

            try:
                nll_loss = self.criterion(self.bornmachine, data, labels)
            except RuntimeError as e:
                logger.warning(f"Norm collapse during training ({e}). Stopping early.")
                self._norm_collapsed = True
                break

            if self.norm_regularizer is not None:
                reg_loss = self.norm_regularizer(self.bornmachine)
                total_loss = nll_loss + reg_loss
            else:
                reg_loss = None
                total_loss = nll_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            log_grads(bm_view=self.bornmachine.generator, watch_freq=self.train_cfg.watch_freq,
                      step=self.step, stage=self.stage)
            self.optimizer.step()

            if self._nc.hard_every > 0 and (self.step % self._nc.hard_every == 0):
                self.bornmachine.generator.renormalize_(target=self._nc_target)

            losses.append(nll_loss.detach().cpu().item())

            log_payload = {f"{self.stage}/train/loss": sum(losses) / len(losses)}
            if reg_loss is not None:
                log_payload[f"{self.stage}/train/norm_reg"] = reg_loss.detach().cpu().item()
            wandb.log(log_payload)

    def _update(self):
        current_value = self.valid_perf.get(self.stopping_criterion_name)

        if current_value is None or not math.isfinite(current_value):
            return

        former_best = self.best.get(
            self.stopping_criterion_name,
            0.0 if self.stopping_criterion_name in _ACC_METRICS else float("inf")
        )

        if self.stopping_criterion_name in _ACC_METRICS:
            improved = current_value > former_best
        elif self.stopping_criterion_name in _LOSS_METRICS:
            improved = current_value < former_best
        else:
            raise ValueError(f"Unknown stopping criterion: {self.stopping_criterion_name}")

        goal_key = list(self.goal.keys())[0] if self.goal else None
        reached_goal = False
        if goal_key is not None:
            goal_val = self.valid_perf.get(goal_key)
            if goal_val is not None:
                if goal_key in _ACC_METRICS:
                    reached_goal = goal_val > self.goal[goal_key]
                else:
                    reached_goal = goal_val < self.goal[goal_key]

        if improved or reached_goal:
            self.best = dict(self.valid_perf)
            self.best_tensors = [t.clone().detach() for t in self.bornmachine.generator.tensors]
            self.best_epoch = self.epoch
            self.patience_counter = 0
            if reached_goal:
                self.patience_counter = self.train_cfg.patience + 1
                logger.info("Goal reached.")
        else:
            self.patience_counter += 1

    def _summarise_training(self):
        """Restore best tensors and clean up. No test eval."""
        self.bornmachine.generator.reset()
        self.bornmachine.generator.initialize(tensors=self.best_tensors)
        self.bornmachine.sync_tensors(after="generation", verify=True)
        self.bornmachine.to(self.device)

        self.bornmachine.reset()
        self.bornmachine.to("cpu")
        if hasattr(self, "valid_perf"):
            del self.valid_perf

        best_stop = self.best.get(self.stopping_criterion_name)
        if best_stop is not None and not math.isfinite(best_stop):
            logger.warning(
                f"Best {self.stopping_criterion_name} is {best_stop} (non-finite); skipping model save."
            )
        elif self.train_cfg.save:
            import hydra
            run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            folder = run_dir / "models"
            folder.mkdir(parents=True, exist_ok=True)
            self.bornmachine.save(path=str(folder / "gen"))
            if wandb.run is not None and not wandb.run.disabled:
                wandb.log_model(str(folder / "gen"))

        logger.info("Generative-Trainer finished.")

    def train(self, goal: Dict[str, float] | None = None):
        """Run the generative training loop."""
        self.step, self.patience_counter, self.goal = 0, 0, goal
        self.epoch = 0
        self.best_epoch = 0
        self.epoch_times = []
        self._norm_collapsed = False

        self.bornmachine.generator.reset()
        self.bornmachine.generator.out_features = []
        self.bornmachine.to(self.device)

        self._nc_target = self._resolve_nc_target()
        if self._nc.soft_strength > 0.0:
            self.norm_regularizer = NormRegularizer(
                strength=self._nc.soft_strength, target=self._nc_target
            )

        self.optimizer = get.optimizer(self.bornmachine.parameters(mode="generator"),
                                       self.train_cfg.optimizer)

        logger.info("Generative training begins.")
        for epoch in range(self.train_cfg.max_epoch):
            epoch_start = time.perf_counter()
            self.epoch = epoch + 1

            self._train_epoch()

            if self._norm_collapsed:
                logger.info("Norm collapsed; ending training.")
                break

            # Sync to classifier view for eval_dis
            self.bornmachine.sync_tensors(after="generation", verify=False)

            gen_loss = eval_gen(
                self.bornmachine, self.datahandler.classification["valid"], self.device
            )
            dis_loss, acc = eval_dis(
                self.bornmachine, self.datahandler.classification["valid"], self.device
            )
            self.valid_perf = {"gen_loss": gen_loss, "dis_loss": dis_loss, "acc": acc}
            record(results=self.valid_perf, stage=self.stage, set="valid")

            self._update()
            self.epoch_times.append(time.perf_counter() - epoch_start)

            if self.patience_counter > self.train_cfg.patience:
                logger.info(f"Early stopping after epoch {self.epoch}.")
                break

        self._summarise_training()
