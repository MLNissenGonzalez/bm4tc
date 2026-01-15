"""
Generative trainer for BornMachine using NLL minimization.

Trains p(x|c) by minimizing negative log-likelihood.
Simpler than GANStyleTrainer - no critic, no retraining.
"""

import hydra
from pathlib import Path
import time
import torch
from typing import Dict
import src.utils.schemas as schemas
import src.utils.get as get
import wandb
from src.tracking import PerformanceEvaluator, record, log_grads
from src.data.handler import DataHandler
from src.models import BornMachine
from src.utils.generative_losses import GenerativeNLL

import logging
logger = logging.getLogger(__name__)


class Trainer:
    """
    Generative trainer for BornMachine using NLL minimization.

    Trains the generator view of the BornMachine by minimizing negative
    log-likelihood of p(x|c). Follows the same pattern as ClassificationTrainer
    with early stopping, metric tracking, and checkpoint saving.

    The criterion must be a subclass of GenerativeNLL with user-implemented
    normalization.

    Attributes:
        best: Dict of best metric values achieved during training.
        best_tensors: Tensors from the best-performing epoch.
    """

    def __init__(
            self,
            bornmachine: BornMachine,
            cfg: schemas.Config,
            datahandler: DataHandler,
            criterion: GenerativeNLL,
            device: torch.device
    ):
        """
        Initialize the generative trainer.

        Args:
            bornmachine: BornMachine instance to train.
            cfg: Complete configuration object.
            datahandler: DataHandler with loaded datasets.
            criterion: User-implemented GenerativeNLL subclass.
            device: Torch device for training.
        """
        self.datahandler = datahandler
        if self.datahandler.classification is None:
            self.datahandler.get_classification_loaders()

        self.device = device
        self.cfg = cfg
        self.stage = "gen"
        self.train_cfg = cfg.trainer.generative

        # User-provided criterion (must be GenerativeNLL subclass)
        self.criterion = criterion

        wandb.define_metric(f"{self.stage}/train/loss", summary="none")
        metrics = self.train_cfg.metrics
        self.evaluator = PerformanceEvaluator(cfg, self.datahandler, self.train_cfg, self.device)
        self._best_perf_factory(metrics)

        # Trainer modifies weights of generator
        self.bornmachine = bornmachine
        self.best_tensors = [t.cpu().clone().detach() for t in self.bornmachine.generator.tensors]

    def _best_perf_factory(self, metrics: Dict[str, int]):
        """Initialize tracking for best metric values."""
        self.best = dict.fromkeys(metrics.keys())
        for metric_name in self.best.keys():
            if metric_name in ["acc", "rob"]:
                self.best[metric_name] = 0.0
            elif metric_name in ["loss", "fid"]:
                self.best[metric_name] = float("Inf")

        self.stopping_criterion_name = self.train_cfg.stop_crit

    def _train_epoch(self):
        """Execute one training epoch: forward pass, loss, backward, optimizer step."""
        losses = []

        for data, labels in self.datahandler.classification["train"]:
            data, labels = data.to(self.device), labels.to(self.device)
            self.step += 1

            # Generative NLL (criterion takes bornmachine, not just probs)
            loss = self.criterion(self.bornmachine, data, labels)

            self.optimizer.zero_grad()
            loss.backward()
            log_grads(bm_view=self.bornmachine.generator, watch_freq=self.train_cfg.watch_freq,
                      step=self.step, stage=self.stage)
            self.optimizer.step()

            losses.append(loss.detach().cpu().item())

            wandb.log({f"{self.stage}/train/loss": sum(losses) / len(losses)})

    def _update(self):
        """
        Epoch-wise check whether model performs better than previous best on validation set.

        - If better: update best performance values and tensors. Reset patience counter.
        - If not: Increase patience counter.
        """
        current_value = self.valid_perf[self.stopping_criterion_name]
        former_best = self.best[self.stopping_criterion_name]

        # Check whether the monitored metric improved
        if self.stopping_criterion_name in ["acc", "rob"]:
            improved = current_value > former_best
        elif self.stopping_criterion_name in ["loss", "fid"]:
            improved = current_value < former_best
        else:
            raise ValueError(f"Unknown stopping criterion: {self.stopping_criterion_name}")

        # Check if we reached target (optional shortcut)
        goal_key = list(self.goal.keys())[0] if self.goal else None
        if self.goal is None or goal_key not in self.valid_perf:
            reached_goal = False
        else:
            if goal_key in ["acc", "rob"]:
                reached_goal = (self.valid_perf[goal_key] > self.goal[goal_key])
            elif goal_key in ["loss", "fid"]:
                reached_goal = (self.valid_perf[goal_key] < self.goal[goal_key])
            else:
                reached_goal = False

        isBetter = improved or reached_goal

        if isBetter:
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
        """
        1. Override generator tensors to best tensors.
        2. Sync to classifier view.
        3. Evaluate on test set.
        4. Record results.
        5. Optionally save model.
        """
        # Override generator with best tensors
        self.bornmachine.generator.reset()
        self.bornmachine.generator.initialize(tensors=self.best_tensors)
        self.bornmachine.sync_tensors(after="generation", verify=True)
        self.bornmachine.to(self.device)

        # Evaluate on test
        test_results = self.evaluator.evaluate(self.bornmachine, "test", self.epoch)

        # Summarise training for wandb
        for metric_name in ["fid", "loss"]:
            if metric_name in test_results.keys():
                wandb.summary[f"{self.stage}/test/{metric_name}"] = test_results[metric_name]
            if metric_name in self.best:
                wandb.summary[f"{self.stage}/valid/{metric_name}"] = self.best[metric_name]

        wandb.summary[f"{self.stage}/epoch/best"] = self.best_epoch
        wandb.summary[f"{self.stage}/epoch/last"] = self.epoch
        if self.epoch_times:
            wandb.summary[f"{self.stage}/avg_epoch_time_s"] = sum(self.epoch_times) / len(self.epoch_times)

        self.bornmachine.reset()
        self.bornmachine.to("cpu")
        del self.valid_perf

        # Save model
        if self.train_cfg.save:
            run_dir = Path(
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

            folder = run_dir / "models"
            folder.mkdir(parents=True, exist_ok=True)

            optimizer_cfg = self.train_cfg.optimizer
            lr = optimizer_cfg.kwargs.lr or ""
            weight_decay = optimizer_cfg.kwargs.weight_decay or ""

            filename_components = [
                f"{self.cfg.dataset.name}",
                f"{self.stage}_mps_bd{self.cfg.born.init_kwargs.bond_dim}",
                f"{self.cfg.born.embedding}{self.cfg.born.init_kwargs.in_dim}",
                f"{self.train_cfg.max_epoch}{optimizer_cfg.name}lr{lr}wd{weight_decay}"
            ]

            filename = "_".join(filename_components)

            save_path = folder / filename
            self.bornmachine.save(path=str(save_path))
            wandb.log_model(str(save_path))

        logger.info(f"Generative-Trainer finished.")

    def train(self, goal: Dict[str, float] | None = None):
        """
        Run the generative training loop.

        Args:
            goal: Optional target metrics to reach early (e.g., {"fid": 10.0}).
                  If reached, training stops regardless of patience.
        """
        self.step, self.patience_counter, self.goal = 0, 0, goal
        self.epoch_times = []

        # Prepare generator and optimizer
        self.bornmachine.generator.reset()
        self.bornmachine.to(self.device)
        self.optimizer = get.optimizer(self.bornmachine.parameters(mode="generator"),
                                       self.train_cfg.optimizer)

        logger.info("Generative training begins.")
        for epoch in range(self.train_cfg.max_epoch):
            epoch_start = time.perf_counter()
            self.epoch = epoch + 1

            self._train_epoch()

            # Sync to classifier for evaluation metrics
            self.bornmachine.sync_tensors(after="generation", verify=False)
            self.valid_perf = self.evaluator.evaluate(self.bornmachine, "valid", epoch)
            record(results=self.valid_perf, stage=self.stage, set="valid")

            self._update()
            self.epoch_times.append(time.perf_counter() - epoch_start)

            if self.patience_counter > self.train_cfg.patience:
                logger.info(f"Early stopping after epoch {self.epoch}.")
                break

        self._summarise_training()
