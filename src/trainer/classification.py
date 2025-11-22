import hydra
from pathlib import Path
import torch
from typing import *
import src.utils.schemas as schemas
import src.utils.getters as get
import wandb
from src.tracking import *
from src.data.handler import DataHandler
from src.models import BornMachine

import logging
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
            self,
            bornmachine: BornMachine,
            cfg: schemas.Config,
            datahandler: DataHandler,
            device: torch.device
                 ):
        """
        Parameters
        ----------
        classifier: BornClassifier (subclass of MPSLayer)
        cfg: Complete configuration. 
        loaders: train, valid, and test loaders, loading preprocessed, but unembedded data. 
        device: torch.device
        """
        self.cfg = cfg
        self.datahandler = datahandler
        self.device = device
    
        self._best_perf_factory()

        # Trainer modifies weights of classifier. 
        self.bornmachine = bornmachine
        self.best_tensors = [t.cpu().clone().detach() for t in self.bornmachine.classifier.tensors]

    def _best_perf_factory(self):
        self.best = {}
        self.best.keys() = self.cfg.tracking.metrics.keys()
        for metric_name in self.best.keys():
            if metric_name in ["acc", "rob"]:
                self.best[metric_name] = 0.0
            elif metric_name  in ["loss", "fid"]:
                self.best[metric_name] = float("Inf")

        self.stopping_criterion_name = self.train_cfg.stop_crit

    def _train_epoch(self,):
        losses = []
        self.bornmachine.classifier.train()
        for data, labels in self.datahandler.classification["train"]:
            data, labels = data.to(self.device), labels.to(self.device)
            self.step += 1

            probs = self.bornmachine.class_probabilities(data)
            loss : torch.Tensor = self.criterion(probs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            log_grads(mps=self.bornmachine.classifier, watch_freq=self.train_cfg.watch_freq,
                      step=self.step, stage=self.stage)
            self.optimizer.step()

            losses.append(loss.detach().cpu().item())
        
            wandb.log({f"{self.stage}_mps/train/loss":
               sum(losses)/len(losses)})
        
    def _update(self):
        """
        Epoch-wise check whether model performs better than previous best on validation set. Then:
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

        # Check if we reached target accuracy (optional shortcut)
        reached_goal = (
            self.goal_acc is not None
            and "acc" in self.valid_perf
            and self.valid_perf["acc"] >= self.goal_acc
    )

        isBetter = improved or reached_goal

        if isBetter:
            self.best = dict(self.valid_perf)  # make a shallow copy
            self.best_tensors = [t.clone().detach() for t in self.bornmachine.classifier.tensors]
            self.best_epoch = self.epoch
            self.patience_counter = 0

            if reached_goal:
                # Bump up goal_acc. 
                self.goal_acc = self.valid_perf["acc"]
            
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
        # Evaluate on test
        test_results = self.evaluator.evaluate(self.bornmachine, "test", self.epoch)
        # Summarise training for wandb
        for metric_name in ["acc", "rob"]:
            if metric_name in test_results.keys():
                wandb.summary[f"{self.stage}_mps/test/{metric_name}"] = test_results[metric_name]
                wandb.summary[f"{self.stage}_mps/valid/{metric_name}"] = self.best[metric_name]
        wandb.summary[f"{self.stage}_mps/epoch/best"] = self.best_epoch
        wandb.summary[f"{self.stage}_mps/epoch/last"] = self.epoch
        
        self.bornmachine.reset()
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
                f"{self.stage}_mps_bd{self.cfg.models.born.init_kwargs.bond_dim}",
                f"{self.cfg.models.born.embedding}{self.cfg.models.born.init_kwargs.in_dim}",
                f"{self.train_cfg.max_epoch}{optimizer_cfg.name}lr{lr}wd{weight_decay}"
            ]
            

            filename = "_".join(filename_components)

            # Saving
            save_path = folder / filename
            self.bornmachine.save(path=str(save_path))
            wandb.log_model(str(save_path))

    def train(
            self, stage : str, goal_acc: float | None = None
    ):
        if stage == "pre":
            self.train_cfg = self.cfg.trainer.classification
        elif stage == "re":
            self.train_cfg = self.cfg.trainer.ganstyle.retrain
        else:
            raise ValueError(f"Unknown stage '{stage}' for Trainer.")

        self.evaluator = PerformanceEvaluator(self.cfg, self.datahandler, stage, self.device)
        wandb.define_metric(f"{stage}_mps/train/loss", summary="none")
        self.step, self.patience_counter = 0, 0
        self.goal_acc, self.stage = goal_acc, stage

        # Prepare classifier, then instantiate criterion and optimizer.
        self.bornmachine.classifier.prepare(device=self.device, train_cfg=self.train_cfg)
        self.criterion = get.criterion("classification", self.train_cfg.criterion)
        self.optimizer = get.optimizer(self.bornmachine.classifier.parameters(), self.train_cfg.optimizer)
        

        logger.info("Categorisation training begins.")        
        for epoch in range(self.train_cfg.max_epoch):
            # Train and evaluate for one epoch
            self.epoch = epoch + 1
            self._train_epoch()
            self.valid_perf = self.evaluator.evaluate(self.bornmachine.classifier, "valid", epoch)
            record(self.valid_perf)

            # Updating and early stopping.
            self._update()
            if self.patience_counter > self.train_cfg.patience:
                logger.info(f"Early stopping after epoch {self.epoch}.")
                break

        self._summarise_training()