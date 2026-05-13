"""
Adversarial Training for Born Machine classifiers.

Implements two adversarial training methods:
- PGD-AT (Madry et al.): Train on adversarial examples
- TRADES (Zhang et al.): Clean loss + KL regularization for robustness
"""

import time
import torch
from pathlib import Path
from typing import Callable, Dict, Optional
import src.utils.schemas as schemas
import src.utils.get as get
from src.data.handler import DataHandler
from src.models import BornMachine
from src.utils.evasion import ProjectedGradientDescent, FastGradientMethod
from src.trainer.eval import eval_dis, eval_rob

import logging
logger = logging.getLogger(__name__)

_LOSS_METRICS = {"dis_loss", "gen_loss"}
_ACC_METRICS = {"acc", "rob"}
_VALID_STOP_CRIT = {"dis_loss", "gen_loss", "acc", "rob"}


class Trainer:
    """
    Adversarial training trainer for BornMachine classifiers.

    Supports two methods:
    - PGD-AT: Replace inputs with adversarial examples, minimize L(x_adv, y)
    - TRADES: Minimize L(x, y) + beta * KL(p(x) || p(x_adv))
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
        self.cfg = cfg
        self.stage = stage
        self.train_cfg = cfg.trainer.adversarial

        if self.datahandler.classification is None:
            self.datahandler.get_classification_loaders(batch_size=self.train_cfg.batch_size)

        if self.train_cfg.method not in ["pgd_at", "trades"]:
            raise ValueError(f"Unknown adversarial training method: {self.train_cfg.method}")

        self._init_best()

        self.bornmachine = bornmachine
        self.best_tensors = [t.cpu().clone().detach() for t in self.bornmachine.classifier.tensors]
        self._init_attack()

    def _init_attack(self):
        evasion = self.train_cfg.evasion

        if evasion.method == "PGD":
            self.attack = ProjectedGradientDescent(
                norm=evasion.norm,
                criterion=evasion.criterion,
                num_steps=evasion.num_steps,
                step_size=evasion.step_size,
                random_start=evasion.random_start
            )
        elif evasion.method == "FGM":
            self.attack = FastGradientMethod(
                norm=evasion.norm,
                criterion=evasion.criterion
            )
        else:
            raise ValueError(f"Unknown attack method: {evasion.method}")

        range_size = self.bornmachine.input_range[1] - self.bornmachine.input_range[0]
        self.range_size = range_size
        self.base_epsilon = (evasion.strengths[0] if evasion.strengths else 0.1) * range_size
        self._abs_curriculum_start = self.train_cfg.curriculum_start * range_size

    def _init_best(self):
        self.best = {"dis_loss": float("inf"), "acc": 0.0}
        if self.train_cfg.eval_rob_freq > 0:
            self.best["rob"] = 0.0
        self.stopping_criterion_name = self.train_cfg.stop_crit
        if self.stopping_criterion_name not in _VALID_STOP_CRIT:
            raise ValueError(
                f"Invalid stop_crit '{self.stopping_criterion_name}'. "
                f"Must be one of: {sorted(_VALID_STOP_CRIT)}"
            )

    def _get_epsilon(self, epoch: int) -> float:
        if not self.train_cfg.curriculum:
            return self.base_epsilon
        end_epoch = self.train_cfg.curriculum_end_epoch or self.train_cfg.max_epoch
        progress = min(1.0, epoch / end_epoch)
        return self._abs_curriculum_start + progress * (self.base_epsilon - self._abs_curriculum_start)

    def _generate_adversarial(self, data, labels, epsilon):
        return self.attack.generate(
            born=self.bornmachine,
            naturals=data,
            labels=labels,
            strength=epsilon,
            device=self.device
        )

    def _compute_kl_divergence(self, clean_probs, adv_probs, eps=1e-12):
        clean_probs = clean_probs.clamp(min=eps)
        adv_probs = adv_probs.clamp(min=eps)
        return (clean_probs * (clean_probs.log() - adv_probs.log())).sum(dim=1).mean()

    def _train_epoch_pgd_at(self, epsilon: float):
        losses = []
        self.bornmachine.classifier.train()

        for data, labels in self.datahandler.classification["train"]:
            data, labels = data.to(self.device), labels.to(self.device)
            self.step += 1

            self.bornmachine.classifier.eval()
            adv_data = self._generate_adversarial(data, labels, epsilon)
            self.bornmachine.classifier.train()

            adv_probs = self.bornmachine.class_probabilities(adv_data)
            adv_loss = self.criterion(adv_probs, labels)

            if self.train_cfg.clean_weight > 0:
                clean_probs = self.bornmachine.class_probabilities(data)
                clean_loss = self.criterion(clean_probs, labels)
                loss = (1 - self.train_cfg.clean_weight) * adv_loss + \
                       self.train_cfg.clean_weight * clean_loss
            else:
                loss = adv_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach().cpu().item())

        self._train_loss = sum(losses) / len(losses) if losses else float("nan")

    def _train_epoch_trades(self, epsilon: float):
        total_losses = []
        self.bornmachine.classifier.train()
        beta = self.train_cfg.trades_beta

        for data, labels in self.datahandler.classification["train"]:
            data, labels = data.to(self.device), labels.to(self.device)
            self.step += 1

            clean_probs = self.bornmachine.class_probabilities(data)
            clean_loss = self.criterion(clean_probs, labels)

            self.bornmachine.classifier.eval()
            adv_data = self._generate_adversarial(data, labels, epsilon)
            self.bornmachine.classifier.train()

            adv_probs = self.bornmachine.class_probabilities(adv_data)
            kl_loss = self._compute_kl_divergence(clean_probs.detach(), adv_probs)
            loss = clean_loss + beta * kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_losses.append(loss.detach().cpu().item())

        self._train_loss = sum(total_losses) / len(total_losses) if total_losses else float("nan")

    def _train_epoch(self, epsilon: float):
        if self.train_cfg.method == "pgd_at":
            self._train_epoch_pgd_at(epsilon)
        elif self.train_cfg.method == "trades":
            self._train_epoch_trades(epsilon)

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

        if improved:
            self.best = dict(self.valid_perf)
            self.best_tensors = [t.clone().detach() for t in self.bornmachine.classifier.tensors]
            self.best_epoch = self.epoch
            self.patience_counter = 0
        else:
            self.patience_counter += 1

    def _summarise_training(self, output_dir: Optional[Path]):
        """Restore best tensors and clean up. No test eval."""
        self.bornmachine.classifier.prepare(tensors=self.best_tensors, device=self.device,
                                            train_cfg=self.train_cfg)
        self.bornmachine.sync_tensors(after="classification", verify=True)
        self.bornmachine.to(self.device)

        self.bornmachine.reset()
        self.bornmachine.to("cpu")
        del self.valid_perf

        if self.train_cfg.save and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.bornmachine.save(path=str(output_dir / "adv"))

        logger.info(f"Adversarial Trainer ({self.train_cfg.method}) finished.")

    def train(
            self,
            on_epoch_end: Optional[Callable[[int, Dict], None]] = None,
            output_dir: Optional[Path] = None,
    ):
        """Run the adversarial training loop."""
        self.step = 0
        self.patience_counter = 0
        self.best_epoch = 0
        self.epoch_times = []

        self.bornmachine.classifier.prepare(device=self.device, train_cfg=self.train_cfg)
        self.criterion = get.criterion("classification", self.train_cfg.criterion)
        self.optimizer = get.optimizer(self.bornmachine.classifier.parameters(),
                                       self.train_cfg.optimizer)

        rob_freq = self.train_cfg.eval_rob_freq

        logger.info(f"Adversarial training ({self.train_cfg.method}) begins.")

        for epoch in range(self.train_cfg.max_epoch):
            epoch_start = time.perf_counter()
            self.epoch = epoch + 1

            epsilon = self._get_epsilon(self.epoch)
            self._train_epoch(epsilon)

            self.bornmachine.sync_tensors(after="classification", verify=False)
            dis_loss, acc = eval_dis(
                self.bornmachine, self.datahandler.classification["valid"], self.device
            )
            self.valid_perf = {"dis_loss": dis_loss, "acc": acc}

            if rob_freq and (self.epoch % rob_freq == 0):
                rob = eval_rob(
                    self.bornmachine, self.datahandler.classification["valid"],
                    self.attack, self.base_epsilon, self.device
                )
                self.valid_perf["rob"] = rob

            if on_epoch_end is not None:
                metrics = {
                    "dis_loss/train": self._train_loss,
                    "epsilon/train":  epsilon,
                    "dis_loss/valid": dis_loss,
                    "acc/valid":      acc,
                }
                if "rob" in self.valid_perf:
                    metrics["rob/valid"] = self.valid_perf["rob"]
                on_epoch_end(self.epoch, metrics)

            self._update()
            self.epoch_times.append(time.perf_counter() - epoch_start)

            if self.patience_counter > self.train_cfg.patience:
                logger.info(f"Early stopping after epoch {self.epoch}.")
                break

        self._summarise_training(output_dir)
