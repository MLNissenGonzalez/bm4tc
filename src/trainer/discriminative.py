import time
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional
import src.utils.get as get
from src.utils.get import OptimizerConfig
from src.utils.get import CriterionConfig
from src.data.handler import DataHandler
from src.models import BornMachine
from src.trainer.utils import eval_metrics

# TODO: merge with generative. only use mixed nll. distinguish using alpha. this file corresponds to alpha=0. 
# softmax get's its own trainer. no need/possibility to report alpha=1 on valid for softmax.
# three trainers: nlll.py, adversarial.py, softmax.py simplest up to the fact that inference is different. 
# adversarial could theoretically use mixed loss aswell. most complex.

@dataclass
class DiscriminativeConfig:
    max_epoch: int = 100
    batch_size: int = 64
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    criterion: CriterionConfig = field(default_factory=CriterionConfig)
    stop_crit: str = "acc"
    patience: int = 250
    save: bool = False
    auto_stack: bool = True
    auto_unbind: bool = False

import logging
logger = logging.getLogger(__name__)

_LOSS_METRICS = {"dis_loss"}
_ACC_METRICS = {"acc", "rob"}
_VALID_STOP_CRIT = {"dis_loss", "gen_loss", "acc", "rob"}


class DiscriminativeTrainer:
    """Discriminative trainer for BornMachine classification."""

    def __init__(
            self,
            bornmachine: BornMachine,
            train_cfg: DiscriminativeConfig,
            datahandler: DataHandler,
            device: torch.device
    ):
        self.datahandler = datahandler
        self.device = device
        self.train_cfg = train_cfg

        if self.datahandler.classification is None:
            self.datahandler.get_classification_loaders(batch_size=self.train_cfg.batch_size)

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
            self.optimizer.step()

            losses.append(loss.detach().cpu().item())

        self._train_loss = sum(losses) / len(losses) if losses else float("nan")

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
        if hasattr(self, "valid_perf"):
            del self.valid_perf

        if self.train_cfg.save and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.bornmachine.save(path=str(output_dir / "cls"))

        logger.info("Discriminative training finished.")

    def train(
            self,
            on_epoch_end: Optional[Callable[[int, Dict], None]] = None,
            output_dir: Optional[Path] = None,
    ):
        """Run the classification training loop."""
        self.step, self.patience_counter = 0, 0
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

            self.bornmachine.sync_tensors(after="classification", verify=False)
            dis_loss, acc, _ = eval_metrics(
                self.bornmachine, self.datahandler.classification["valid"], self.device
            )
            self.valid_perf = {"dis_loss": dis_loss, "acc": acc}

            if on_epoch_end is not None:
                on_epoch_end(self.epoch, {
                    "dis_loss/train": self._train_loss,
                    "dis_loss/valid": dis_loss,
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
    from src.models.born import BornMachine, BornMachineConfig, MPSInitConfig
    from src.data.handler import DataHandler
    from src.data.gen_n_load import DatasetConfig, DataGenDowConfig

    device = torch.device("cpu")
    bm = BornMachine(
        cfg=BornMachineConfig(embedding="legendre", init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3)),
        data_dim=2, num_classes=2, device=device,
    )
    ds_cfg = DatasetConfig(
        name="spirals",
        gen_dow_kwargs=DataGenDowConfig(name="spirals", size=32, seed=42, noise=0.1),
        overwrite=True,
    )
    dh = DataHandler(ds_cfg)
    dh.load()
    dh.split_and_rescale(bm)

    train_cfg = DiscriminativeConfig(max_epoch=10, batch_size=4, patience=250)
    trainer = DiscriminativeTrainer(bornmachine=bm, train_cfg=train_cfg, datahandler=dh, device=device)

    logged = []
    trainer.train(on_epoch_end=lambda ep, m: logged.append((ep, m)))

    assert len(logged) == 10, f"Expected 10 epochs, got {len(logged)}"
    last_ep, last_m = logged[-1]
    assert "dis_loss/valid" in last_m and "acc/valid" in last_m
    print(f"  epoch={last_ep}  dis_loss/valid={last_m['dis_loss/valid']:.4f}  acc/valid={last_m['acc/valid']:.4f}")
    print("discriminative.py smoke test passed.")
