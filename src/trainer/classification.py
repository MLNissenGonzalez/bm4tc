import hydra
from pathlib import Path
import time
import torch
from typing import *
import src.utils.schemas as schemas
import src.utils.get as get
import wandb
from src.tracking import *
from src.data.handler import DataHandler
from src.models import BornMachine

import logging
logger = logging.getLogger(__name__)

class Trainer:
    """
    Classification trainer for BornMachine discriminative training.

    Trains the MPS as a classifier using the Born rule. Supports early stopping,
    metric tracking via W&B, and checkpoint saving. Records average epoch time.

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
        """
        Initialize the classification trainer.

        Args:
            bornmachine: BornMachine instance to train.
            cfg: Complete configuration object.
            stage: Training stage - "pre" for pretraining, "re" for retraining.
            datahandler: DataHandler with loaded datasets.
            device: Torch device for training.
        """
        self.datahandler = datahandler
        if self.datahandler.classification == None:
            self.datahandler.get_classification_loaders()
            
        self.device = device
        self.cfg, self.stage = cfg, stage # needed in summary of training
    
        if stage == "pre": # TODO: I am using train_cfg only here. think about maybe have early stopping only for classification, retrain and not for ganstyle training.
            self.train_cfg = cfg.trainer.classification
        elif stage == "re":
            self.train_cfg = cfg.trainer.ganstyle.retrain
        else: raise f"{stage} not recognised."

        wandb.define_metric(f"{stage}/train/loss", summary="none")
        metrics = self.train_cfg.metrics
        self.evaluator = PerformanceEvaluator(cfg, self.datahandler, self.train_cfg, self.device)
        self._best_perf_factory(metrics)

        # Trainer modifies weights of classifier. 
        self.bornmachine = bornmachine
        self.best_tensors = [t.cpu().clone().detach() for t in self.bornmachine.classifier.tensors]

    def _best_perf_factory(self, metrics: Dict[str, int]):
        self.best = dict.fromkeys(metrics.keys())
        for metric_name in self.best.keys():
            if metric_name in ["acc", "rob"]:
                self.best[metric_name] = 0.0
            elif metric_name in ["clsloss", "genloss", "fid"]:
                self.best[metric_name] = float("Inf")

        self.stopping_criterion_name = self.train_cfg.stop_crit
        valid_criteria = ["clsloss", "genloss", "acc", "fid", "rob"]
        if self.stopping_criterion_name not in valid_criteria:
            raise ValueError(
                f"Invalid stop_crit '{self.stopping_criterion_name}'. "
                f"Must be one of: {valid_criteria}"
            )
        # Always track rob for averaging (even if not in metrics explicitly)
        if self.stopping_criterion_name == "rob" and "rob" not in self.best:
            self.best["rob"] = 0.0

    def _train_epoch(self):
        """Execute one training epoch: forward pass, loss, backward, optimizer step."""
        losses = []
        self.bornmachine.classifier.train()
        for data, labels in self.datahandler.classification["train"]:
            data, labels = data.to(self.device), labels.to(self.device)
            self.step += 1

            probs = self.bornmachine.class_probabilities(data)
            loss : torch.Tensor = self.criterion(probs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            log_grads(bm_view=self.bornmachine.classifier, watch_freq=self.train_cfg.watch_freq,
                      step=self.step, stage=self.stage)
            self.optimizer.step()

            losses.append(loss.detach().cpu().item())
        
            wandb.log({f"{self.stage}/train/loss":
               sum(losses)/len(losses)})
        
    def _update(self):
        """
        Epoch-wise check whether model performs better than previous best on validation set.

        - If better: update best performance values and tensors. Reset patience counter.
        - If not: Increase patience counter.

        For rob metric: averages all rob/{strength} values since robustness is evaluated
        at multiple perturbation strengths.
        """
        # Handle rob specially: average all rob/{strength} values
        if self.stopping_criterion_name == "rob":
            rob_values = [v for k, v in self.valid_perf.items()
                         if k.startswith("rob/") and isinstance(v, (int, float))]
            current_value = sum(rob_values) / len(rob_values) if rob_values else None
        else:
            current_value = self.valid_perf.get(self.stopping_criterion_name)

        if current_value is None:
            return

        former_best = self.best.get(
            self.stopping_criterion_name,
            0.0 if self.stopping_criterion_name in ["acc", "rob"] else float("Inf")
        )

        # Check whether the monitored metric improved
        if self.stopping_criterion_name in ["acc", "rob"]:
            improved = current_value > former_best
        elif self.stopping_criterion_name in ["clsloss", "genloss", "fid"]:
            improved = current_value < former_best
        else:
            raise ValueError(f"Unknown stopping criterion: {self.stopping_criterion_name}")

        # Check if we reached target (optional shortcut)
        goal_key = list(self.goal.keys())[0] if self.goal else None
        if self.goal is None:
            reached_goal = False
        elif goal_key == "rob":
            # Handle rob goal by averaging rob/* values
            rob_values = [v for k, v in self.valid_perf.items()
                         if k.startswith("rob/") and isinstance(v, (int, float))]
            goal_value = sum(rob_values) / len(rob_values) if rob_values else 0.0
            reached_goal = goal_value > self.goal["rob"]
        elif goal_key in ["acc"]:
            reached_goal = self.valid_perf.get(goal_key, 0.0) > self.goal[goal_key]
        elif goal_key in ["clsloss", "genloss", "fid"]:
            reached_goal = self.valid_perf.get(goal_key, float("Inf")) < self.goal[goal_key]
        else:
            reached_goal = False

        isBetter = improved or reached_goal

        if isBetter:
            self.best = dict(self.valid_perf)  # make a shallow copy
            # Store averaged rob value if using rob as stopping criterion
            if self.stopping_criterion_name == "rob":
                self.best["rob"] = current_value
            self.best_tensors = [t.clone().detach() for t in self.bornmachine.classifier.tensors]
            self.best_epoch = self.epoch
            self.patience_counter = 0

            if reached_goal:
                self.patience_counter = self.train_cfg.patience + 1
                logger.info("Goal reached.")
        else:
            self.patience_counter += 1

    def _summarise_training(self): 
        """
        1. Override classifier tensors to best tensors. 
        2. Evaluate this best model on the test set.
        3. Record best and test results. 
        4. Delete last entry of validation set.
        5. Optionally: Save model weights. 
        """
        # Overide classifier to best tensors
        self.bornmachine.classifier.prepare(tensors=self.best_tensors, device=self.device, train_cfg=self.train_cfg)
        self.bornmachine.sync_tensors(after="classification", verify=True)
        self.bornmachine.to(self.device)
        # Evaluate on test
        test_results = self.evaluator.evaluate(self.bornmachine, "test", self.epoch)
        # Summarise training for wandb
        for metric_name in ["acc", "rob"]:
            if metric_name in test_results.keys():
                wandb.summary[f"{self.stage}/test/{metric_name}"] = test_results[metric_name]
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

            # Models subfolder inside it
            folder = run_dir / "models"
            folder.mkdir(parents=True, exist_ok=True)
            # Filename construction
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

            # Saving
            save_path = folder / filename
            self.bornmachine.save(path=str(save_path))
            wandb.log_model(str(save_path))
        logger.info(f"Classification-Trainer for {self.stage}-training finished.")

    def train(self, goal: Dict[str, float] | None = None):
        """
        Run the classification training loop.

        Args:
            goal: Optional target metrics to reach early (e.g., {"acc": 0.95}).
                  If reached, training stops regardless of patience.
        """
        self.step, self.patience_counter, self.goal = 0, 0, goal
        self.best_epoch = 0
        self.epoch_times = []

        # Prepare classifier, then instantiate criterion and optimizer.
        self.bornmachine.classifier.prepare(device=self.device, train_cfg=self.train_cfg)
        self.criterion = get.criterion("classification", self.train_cfg.criterion)
        self.optimizer = get.optimizer(self.bornmachine.classifier.parameters(), self.train_cfg.optimizer)


        logger.info("Categorisation training begins.")
        for epoch in range(self.train_cfg.max_epoch):
            epoch_start = time.perf_counter()
            # Train and evaluate for one epoch
            self.epoch = epoch + 1
            self._train_epoch()
            # TODO: Write a prior check if generative eval-metrics are requested. If not, skip sync_tensors here.
            self.bornmachine.sync_tensors(after="classification", verify=False) # needed for generative-performance eval-metrics
            self.valid_perf = self.evaluator.evaluate(self.bornmachine, "valid", epoch)
            record(results=self.valid_perf, stage=self.stage, set="valid")

            # Updating and early stopping.
            self._update() # self.best is updated here
            self.epoch_times.append(time.perf_counter() - epoch_start)
            if self.patience_counter > self.train_cfg.patience:
                logger.info(f"Early stopping after epoch {self.epoch}.")
                break

        self._summarise_training()