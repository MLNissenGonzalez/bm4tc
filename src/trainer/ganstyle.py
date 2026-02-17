import time
import torch
import hydra
from pathlib import Path
from typing import *
import src.utils.schemas as schemas
import logging
from .classification import Trainer as ClassificationTrainer
from src.tracking import *
from src.data.handler import DataHandler
from src.models import Critic, BornMachine
import src.utils.get as get
import wandb
logger = logging.getLogger(__name__)


class Trainer:
    """
    GAN-style trainer for improving BornMachine generative capabilities.

    Alternates between training a critic to distinguish real/synthetic samples
    and training the generator to fool the critic. Triggers classification
    retraining when discriminative accuracy drops below tolerance.

    Records two timing metrics:
    - avg_epoch_time_total_s: Average time per epoch including retraining.
    - avg_epoch_time_no_retrain_s: Average time per epoch excluding retraining.
    """

    def __init__(
            self,
            bornmachine: BornMachine,
            cfg: schemas.Config,
            datahandler: DataHandler,
            critic: "Critic",
            device: torch.device,
            best: Dict[str, float] = {"acc": 1.0, "clsloss": 0.0}
    ):
        """
        Initialize the GAN-style trainer.

        Args:
            bornmachine: BornMachine instance to train.
            cfg: Complete configuration object.
            datahandler: DataHandler with loaded datasets.
            critic: Critic network for discriminating real vs synthetic.
            device: Torch device for training.
            best: Target metrics from pretraining (triggers retraining if dropped).
        """
        # Save some attributes
        self.cfg = cfg
        self.device= device
        self.train_cfg = self.cfg.trainer.ganstyle
        self.goal = {
            self.train_cfg.retrain_crit: best[self.train_cfg.retrain_crit]}

        # Assign datahandler
        self.datahandler = datahandler
        if self.datahandler.discrimination == None:
            # Adapt batch size of dataloader of natural data to fit sampling cfg
            num_spc_nat = int(round(self.train_cfg.r_real *
                              self.train_cfg.sampling.num_spc))
            self.datahandler.get_discrimination_loaders(batch_size=num_spc_nat)

        # Assign bornmachine
        self.bornmachine = bornmachine
        self.bornmachine.reset()

        # Assign critic
        self.critic = critic
        if not self.critic.pretrained:
            self.critic.ganstyle_pretrain(datahandler, bornmachine, device)

        # Initialize best metrics tracking (for HPO)
        self.stopping_criterion = self.train_cfg.stop_crit
        valid_criteria = ["clsloss", "genloss", "acc", "fid", "rob"]
        if self.stopping_criterion not in valid_criteria:
            raise ValueError(
                f"Invalid stop_crit '{self.stopping_criterion}'. "
                f"Must be one of: {valid_criteria}"
            )
        # Ensure stop_crit is in metrics (evaluated every epoch if not already)
        metrics_for_best = dict(self.train_cfg.metrics)
        if self.stopping_criterion not in metrics_for_best and self.stopping_criterion != "rob":
            logger.warning(f"stop_crit '{self.stopping_criterion}' not in metrics, adding with freq=1")
            metrics_for_best[self.stopping_criterion] = 1
        self._best_perf_factory(metrics_for_best)
        self.best_epoch = 0

    def _best_perf_factory(self, metrics: Dict[str, int]):
        """Initialize tracking for best metric values (similar to ClassificationTrainer)."""
        self.best = dict.fromkeys(metrics.keys())
        for metric_name in self.best.keys():
            if metric_name in ["acc", "rob"]:
                self.best[metric_name] = 0.0
            else:
                self.best[metric_name] = float("Inf")
        # Always track rob for averaging (even if not in metrics explicitly)
        if self.stopping_criterion == "rob" and "rob" not in self.best:
            self.best["rob"] = 0.0

    def _update_best(self, validation_metrics: Dict[str, float]):
        """
        Update best metrics if current epoch improves on the stopping criterion.

        When the stopping criterion improves, ALL metrics are updated to current values.
        This ensures self.best reflects the metrics of the best model according to
        the stopping criterion.

        For rob metric: averages all rob/{strength} values since robustness is evaluated
        at multiple perturbation strengths.
        """
        criterion = self.stopping_criterion

        # Handle rob specially: average all rob/{strength} values
        if criterion == "rob":
            rob_values = [v for k, v in validation_metrics.items()
                        if k.startswith("rob/") and isinstance(v, (int, float))]
            current_value = sum(rob_values) / len(rob_values) if rob_values else None
        else:
            current_value = validation_metrics.get(criterion)

        if current_value is None:
            return

        # Check if improved based on criterion type
        improved = False
        if criterion in ["acc", "rob"]:
            if current_value > self.best.get(criterion, 0.0):
                improved = True
        else:
            if current_value < self.best.get(criterion, float("Inf")):
                improved = True

        if improved:
            # Update ALL metrics to current values (this is the new best model)
            # Use shallow copy to include all keys (including rob/* variants)
            self.best = dict(validation_metrics)
            # Store averaged rob value for HPO access
            if criterion == "rob":
                self.best["rob"] = current_value
            self.best_epoch = self.epoch

    def _toRetrain(self, validation_metrics):
        """
        Checks whether bornmachine should be retrained given goal metric and tolerance. Returns boolean. 
        
        :param validation_metrics: Accuracy and loss of BornClassifier evaluated on validation set.
        """
        goal_key = list(self.goal.keys())[0]
        toRetrain = (
            (goal_key == "acc" and
             ((self.goal["acc"]-validation_metrics["acc"]) > self.train_cfg.tolerance))
            or (goal_key == "clsloss" and
                (validation_metrics["clsloss"]-self.goal["clsloss"])/self.goal["clsloss"] > self.train_cfg.tolerance)
        )
        return toRetrain


    def _summarise_training(self):
        """
        Evaluate BornMachine on test set and save it if wanted. 
        """
        test_results = self.evaluator.evaluate(self.bornmachine, "test", self.epoch)
        # Summarise training for wandb
        for metric_name in ["acc", "rob"]:
            if metric_name in test_results.keys():
                wandb.summary[f"gan/test/{metric_name}"] = test_results[metric_name]
        if self.epoch_times_total:
            wandb.summary["gan/avg_epoch_time_total_s"] = sum(self.epoch_times_total) / len(self.epoch_times_total)
        if self.epoch_times_no_retrain:
            wandb.summary["gan/avg_epoch_time_no_retrain_s"] = sum(self.epoch_times_no_retrain) / len(self.epoch_times_no_retrain)
        
        self.bornmachine.reset()
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
                f"gan_mps_bd{self.cfg.models.born.init_kwargs.bond_dim}",
                f"{self.cfg.models.born.embedding}{self.cfg.models.born.init_kwargs.in_dim}",
                f"{self.train_cfg.max_epoch}{optimizer_cfg.name}lr{lr}wd{weight_decay}"
            ]
            

            filename = "_".join(filename_components)

            # Saving
            save_path = folder / filename
            self.bornmachine.save(path=str(save_path))
            if wandb.run is not None and not wandb.run.disabled:
                wandb.log_model(str(save_path))
    
    def train(self):
        """
        Run the GAN-style training loop.

        For each epoch:
        1. Train critic on real vs synthetic samples (inner loop).
        2. Train generator to fool the critic (single step).
        3. Evaluate on validation set.
        4. Retrain classifier if accuracy dropped below tolerance.
        """
        # Initialize optimizer, evaluator, and retrainer
        self.bornmachine.to(device=self.device)
        self.optimizer = get.optimizer(
            self.bornmachine.parameters(), config=self.train_cfg.optimizer)
        self.evaluator = PerformanceEvaluator(
            self.cfg, self.datahandler, self.train_cfg, self.device)
        self.retrainer = ClassificationTrainer(
            self.bornmachine, self.cfg, "re", self.datahandler, self.device)
        self.critic.to(device=self.device)
        
        # Other Configs
        max_epoch = self.train_cfg.max_epoch
        sampling_cfg = self.train_cfg.sampling
        self.epoch_times_total = []
        self.epoch_times_no_retrain = []

        # Outer loop: GANStyle training
        logger.info("Starting GANStyle training.")
        for epoch in range(max_epoch):
            epoch_start = time.perf_counter()
            self.epoch = epoch + 1
            g_losses, d_losses = [], []
            for naturals in self.datahandler.discrimination["train"]:
                # Dynamically adapt number of synth sampled generated for varying batch sizes
                sampling_cfg.num_spc = int(
                    round(naturals.shape[0] / self.train_cfg.r_real))
                
                naturals = naturals.to(self.device)
                # Inner training loop for discriminator / critic
                with torch.no_grad():
                    generated = self.bornmachine.sample(sampling_cfg).to(self.device)
                for mini_epoch in range(self.train_cfg.critic.discrimination.max_epoch_gan):
                    d_losses.append(
                        self.critic.train_step(naturals, generated))

                # Training step for generator
                generated = self.bornmachine.sample(sampling_cfg).to(self.device)
                self.optimizer.zero_grad()
                loss = self.critic.generator_loss(generated)
                loss.backward()
                self.optimizer.step()
                g_losses.append(loss.cpu().detach().item())

            g_loss = sum(g_losses) / len(g_losses)
            d_loss = sum(d_losses) / len(d_losses)
            train_metrics = {"gan/train/g_loss": g_loss,
                             "gan/train/d_loss": d_loss}
            wandb.log(train_metrics)

            self.bornmachine.reset()
            self.bornmachine.sync_tensors(after="gan")

            # evaluation of gan training vs retraining
            validation_metrics = self.evaluator.evaluate(
                self.bornmachine, "valid", epoch)
            record(validation_metrics, "gan", "valid")
            self._update_best(validation_metrics)
            self.epoch_times_no_retrain.append(time.perf_counter() - epoch_start)
            if self._toRetrain(validation_metrics):
                logger.info("Retraining.")
                self.retrainer.train(self.goal)
                # Move model back to device after retraining (retrainer moves to CPU)
                self.bornmachine.to(self.device)
                logger.info("Finished retraining.")
            self.epoch_times_total.append(time.perf_counter() - epoch_start)
            # End of loop

        # result on test set of metrics, number of retrainings (maybe epoch number aswell)
        self._summarise_training()
        logger.info("GANStyle training completed.")