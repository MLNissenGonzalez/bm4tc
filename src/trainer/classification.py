import hydra
from pathlib import Path
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
    def __init__(
            self,
            bornmachine: BornMachine,
            cfg: schemas.Config,
            stage: str,
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
        self.evaluator = PerformanceEvaluator(cfg, self.datahandler, self.train_cfg, metrics, self.device)
        self._best_perf_factory(metrics)

        # Trainer modifies weights of classifier. 
        self.bornmachine = bornmachine
        self.best_tensors = [t.cpu().clone().detach() for t in self.bornmachine.classifier.tensors]

    def _best_perf_factory(self, metrics: Dict[str, int]):
        self.best = dict.fromkeys(metrics.keys())
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
            log_grads(bm_view=self.bornmachine.classifier, watch_freq=self.train_cfg.watch_freq,
                      step=self.step, stage=self.stage)
            self.optimizer.step()

            losses.append(loss.detach().cpu().item())
        
            wandb.log({f"{self.stage}/train/loss":
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
        if self.goal is None or self.goal.keys()[0] not in self.valid_perf:
            reached_goal = False

        else:
            if self.goal.keys()[0] == "acc":
                reached_goal = (self.valid_perf["acc"]>self.goal["acc"])
            elif self.goal.keys()[0] == "loss":
                reached_goal = (self.valid_perf["loss"]<self.goal["loss"])
            else: raise KeyError(f"{self.goal.keys()=} not recognised")

        isBetter = improved or reached_goal

        if isBetter:
            self.best = dict(self.valid_perf)  # make a shallow copy
            self.best_tensors = [t.clone().detach() for t in self.bornmachine.classifier.tensors]
            self.best_epoch = self.epoch
            self.patience_counter = 0

            if reached_goal: # do I want to train after reaching the goal acc? No!
                self.patience_counter = self.train_cfg.patience + 1
                logger.info("Goal acc reached.")
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

    def train(
            self, goal: Dict[str, float] | None = None
    ):
     
        self.step, self.patience_counter, self.goal = 0, 0, goal

        # Prepare classifier, then instantiate criterion and optimizer.
        self.bornmachine.classifier.prepare(device=self.device, train_cfg=self.train_cfg)
        self.criterion = get.criterion("classification", self.train_cfg.criterion)
        self.optimizer = get.optimizer(self.bornmachine.classifier.parameters(), self.train_cfg.optimizer)
        

        logger.info("Categorisation training begins.")        
        for epoch in range(self.train_cfg.max_epoch):
            # Train and evaluate for one epoch
            self.epoch = epoch + 1
            self._train_epoch()
            # TODO: Write a prior check if generative eval-metrics are requested. If not, skip sync_tensors here.
            self.bornmachine.sync_tensors(after="classification", verify=False) # needed for generative-performance eval-metrics
            self.valid_perf = self.evaluator.evaluate(self.bornmachine, "valid", epoch)
            record(results=self.valid_perf, stage=self.stage, set="valid")

            # Updating and early stopping.
            self._update() # self.best is updated here
            if self.patience_counter > self.train_cfg.patience:
                logger.info(f"Early stopping after epoch {self.epoch}.")
                break

        self._summarise_training()