import time
import torch
from typing import *
import src.utils.schemas as schemas
import src.utils.get as get
import wandb
from src.tracking import log_grads, record
from src.data.handler import DataHandler
from src.models import BornMachine
from src.trainer.eval import eval_dis

import logging
logger = logging.getLogger(__name__)

_LOSS_METRICS = {"dis_loss"}
_ACC_METRICS = {"acc", "rob"}
_VALID_STOP_CRIT = {"dis_loss", "gen_loss", "acc", "rob"}


class Trainer:
    """
    Classification trainer for BornMachine discriminative training.

    Trains the MPS as a classifier using the Born rule. Supports early stopping
    and checkpoint saving.

    Attributes:
        best: Dict of best metric values achieved during training.
        best_tensors: Tensors from the best-performing epoch.
    """

    def __init__(
            self,
            bornmachine: BornMachine,
            cfg: schemas.Config,
            stage: str,
            datahandler: DataHandler,
            device: torch.device
    ):
        self.datahandler = datahandler
        self.device = device
        self.cfg, self.stage = cfg, stage

        if stage == "pre":
            self.train_cfg = cfg.trainer.discriminative
        else:
            raise ValueError(f"Stage '{stage}' not recognised. Use 'pre'.")

        if self.datahandler.classification is None:
            self.datahandler.get_classification_loaders(batch_size=self.train_cfg.batch_size)

        wandb.define_metric(f"{stage}/train/loss", summary="none")
        self._init_best()

        self.bornmachine = bornmachine
        self.best_tensors = [t.cpu().clone().detach() for t in self.bornmachine.classifier.tensors]

    def _init_best(self):
        self.best = {"dis_loss": float("inf"), "acc": 0.0}
        self.stopping_criterion_name = self.train_cfg.stop_crit
        if self.stopping_criterion_name not in _VALID_STOP_CRIT:
            raise ValueError(
                f"Invalid stop_crit '{self.stopping_criterion_name}'. "
                f"Must be one of: {sorted(_VALID_STOP_CRIT)}"
            )

    def _train_epoch(self):
        losses = []
        self.bornmachine.classifier.train()
        for data, labels in self.datahandler.classification["train"]:
            data, labels = data.to(self.device), labels.to(self.device)
            self.step += 1

            probs = self.bornmachine.class_probabilities(data)
            loss: torch.Tensor = self.criterion(probs, labels)

            if torch.isnan(loss):
                logger.warning("NaN loss detected — aborting epoch.")
                self._nan_detected = True
                break

            self.optimizer.zero_grad()
            loss.backward()
            log_grads(bm_view=self.bornmachine.classifier, watch_freq=self.train_cfg.watch_freq,
                      step=self.step, stage=self.stage)
            self.optimizer.step()

            losses.append(loss.detach().cpu().item())

            wandb.log({f"{self.stage}/train/loss": sum(losses) / len(losses)})

    def _update(self):
        """Check if valid_perf improved; update best tensors and patience counter."""
        current_value = self.valid_perf.get(self.stopping_criterion_name)
        if current_value is None:
            return

        former_best = self.best.get(
            self.stopping_criterion_name,
            0.0 if self.stopping_criterion_name in _ACC_METRICS else float("inf")
        )

        if self.stopping_criterion_name in _ACC_METRICS:
            improved = current_value > former_best
        else:
            improved = current_value < former_best

        # Optional early exit on goal
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
            self.best_tensors = [t.clone().detach() for t in self.bornmachine.classifier.tensors]
            self.best_epoch = self.epoch
            self.patience_counter = 0
            if reached_goal:
                self.patience_counter = self.train_cfg.patience + 1
                logger.info("Goal reached.")
        else:
            self.patience_counter += 1

    def _summarise_training(self):
        """Restore best tensors and clean up. No test eval."""
        self.bornmachine.classifier.prepare(tensors=self.best_tensors, device=self.device,
                                            train_cfg=self.train_cfg)
        self.bornmachine.sync_tensors(after="classification", verify=True)
        self.bornmachine.to(self.device)

        self.bornmachine.reset()
        self.bornmachine.to("cpu")
        if hasattr(self, "valid_perf"):
            del self.valid_perf

        if self.train_cfg.save:
            import hydra
            from pathlib import Path
            run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            folder = run_dir / "models"
            folder.mkdir(parents=True, exist_ok=True)
            self.bornmachine.save(path=str(folder / "cls"))
            if wandb.run is not None and not wandb.run.disabled:
                wandb.log_model(str(folder / "cls"))

        logger.info(f"Classification-Trainer for {self.stage}-training finished.")

    def train(self, goal: Dict[str, float] | None = None):
        """Run the classification training loop."""
        self.step, self.patience_counter, self.goal = 0, 0, goal
        self.epoch = 0
        self.best_epoch = 0
        self.epoch_times = []
        self._nan_detected = False

        self.bornmachine.classifier.prepare(device=self.device, train_cfg=self.train_cfg)
        self.criterion = get.criterion("classification", self.train_cfg.criterion)
        self.optimizer = get.optimizer(self.bornmachine.classifier.parameters(),
                                       self.train_cfg.optimizer)

        logger.info("Classification training begins.")
        for epoch in range(self.train_cfg.max_epoch):
            epoch_start = time.perf_counter()
            self.epoch = epoch + 1
            self._train_epoch()

            if self._nan_detected:
                logger.warning("NaN loss — stopping training. Reporting dis_loss=inf to HPO.")
                self.best["dis_loss"] = float("inf")
                break

            dis_loss, acc = eval_dis(
                self.bornmachine, self.datahandler.classification["valid"], self.device
            )
            self.valid_perf = {"dis_loss": dis_loss, "acc": acc}
            record(results=self.valid_perf, stage=self.stage, set="valid")

            self._update()
            self.epoch_times.append(time.perf_counter() - epoch_start)
            if self.patience_counter > self.train_cfg.patience:
                logger.info(f"Early stopping after epoch {self.epoch}.")
                break

        self._summarise_training()
