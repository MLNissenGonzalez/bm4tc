import time
import logging
from pathlib import Path
from typing import Callable, Dict, Optional
import torch
from torch import nn
from src.utils.train import optimizer

logger = logging.getLogger(__name__)

_LOSS_METRICS = {"softmax_loss"}
_ACC_METRICS = {"acc"}
_VALID_STOP_CRIT = {"softmax_loss", "acc"}


class ClassificationSoftmaxNLL(nn.Module):
    """
    Thin wrapper around nn.CrossEntropyLoss for softmax-based MPS classifiers.

    Expects raw MPS amplitudes ψ (signed, unnormalized) as logits — NOT
    Born-rule probabilities. Delegates to nn.CrossEntropyLoss, which applies
    log-softmax internally (numerically stable fused kernel).
    """
    def __init__(self):
        super().__init__()
        self._loss = nn.CrossEntropyLoss()

    def forward(self, amplitudes: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._loss(amplitudes, t)


class SoftmaxTrainer:
    """
    Standalone CBM trainer using raw amplitudes as softmax logits.

    Trains via ClassificationSoftmaxNLL (CrossEntropyLoss on raw amplitudes)
    and validates with eval_softmax from src.utils.train. Supports early
    stopping on 'acc' or 'softmax_loss'.
    """

    def __init__(self, cbm, train_cfg, datahandler, device: torch.device):
        self.cbm = cbm
        self.train_cfg = train_cfg
        self.datahandler = datahandler
        self.device = device

        if self.datahandler.classification is None:
            self.datahandler.get_classification_loaders(batch_size=self.train_cfg.batch_size)

        self.stopping_criterion_name = self.train_cfg.stop_crit
        if self.stopping_criterion_name not in _VALID_STOP_CRIT:
            raise ValueError(
                f"Invalid stop_crit '{self.stopping_criterion_name}'. "
                f"Must be one of: {sorted(_VALID_STOP_CRIT)}"
            )
        self.best = {
            "softmax_loss": float("inf"),
            "acc": 0.0,
        }

    def _train_epoch(self):
        losses = []
        self.cbm.train()
        for data, labels in self.datahandler.classification["train"]:
            data, labels = data.to(self.device), labels.to(self.device)
            self.step += 1

            amps = self.cbm.amplitudes(data)
            loss: torch.Tensor = self.criterion(amps, labels)

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
        current_value = self.valid_perf.get(self.stopping_criterion_name)
        if current_value is None:
            return

        former_best = self.best.get(
            self.stopping_criterion_name,
            0.0 if self.stopping_criterion_name in _ACC_METRICS else float("inf"),
        )

        if self.stopping_criterion_name in _ACC_METRICS:
            improved = current_value > former_best
        else:
            improved = current_value < former_best

        if improved:
            self.best = dict(self.valid_perf)
            self.best_tensors = [t.cpu().clone().detach() for t in self.cbm.tensors]
            self.best_epoch = self.epoch
            self.patience_counter = 0
        else:
            self.patience_counter += 1

    def _summarise_training(self, output_dir: Optional[Path]):
        self.cbm.initialize(tensors=self.best_tensors)
        self.cbm.prepare(device=self.device)
        self.cbm.to("cpu")
        if hasattr(self, "valid_perf"):
            del self.valid_perf
        if self.train_cfg.save and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.cbm.save(str(output_dir / "model.pt"))
        logger.info("Softmax training finished.")

    def train(
        self,
        on_epoch_end: Optional[Callable[[int, Dict], None]] = None,
        output_dir: Optional[Path] = None,
    ):
        """Run the softmax training loop."""
        from src.utils.train import eval_softmax

        self.step, self.patience_counter = 0, 0
        self.epoch = 0
        self.best_epoch = 0
        self.epoch_times = []
        self._nan_detected = False
        self.best_tensors = [t.cpu().clone().detach() for t in self.cbm.tensors]

        self.cbm.prepare(device=self.device)
        self.criterion = ClassificationSoftmaxNLL()
        self.optimizer = optimizer(self.cbm.parameters(), self.train_cfg.optimizer)

        logger.info("Softmax training begins.")
        for epoch in range(self.train_cfg.max_epoch):
            epoch_start = time.perf_counter()
            self.epoch = epoch + 1
            self._train_epoch()

            if self._nan_detected:
                logger.warning("NaN loss — stopping training.")
                self.best["softmax_loss"] = float("inf")
                break

            softmax_loss, acc = eval_softmax(
                self.cbm, self.datahandler.classification["valid"], self.device
            )
            self.valid_perf = {"softmax_loss": softmax_loss, "acc": acc}

            if on_epoch_end is not None:
                on_epoch_end(self.epoch, {
                    "softmax_loss/train": self._train_loss,
                    "softmax_loss/valid": softmax_loss,
                    "acc/valid": acc,
                })

            self._update()
            self.epoch_times.append(time.perf_counter() - epoch_start)
            if self.patience_counter > self.train_cfg.patience:
                logger.info(f"Early stopping after epoch {self.epoch}.")
                break

        self._summarise_training(output_dir)


if __name__ == "__main__":
    import sys
    import torch
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from torch.utils.data import DataLoader, TensorDataset
    from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig
    from src.train.nll import NLLConfig

    criterion = ClassificationSoftmaxNLL()
    logits = torch.randn(8, 2)
    labels = torch.randint(0, 2, (8,))
    loss = criterion(logits, labels)
    assert loss.isfinite(), f"ClassificationSoftmaxNLL non-finite: {loss}"
    print(f"  ClassificationSoftmaxNLL loss={loss.item():.4f}")

    device = torch.device("cpu")
    cfg = CBMConfig(
        embedding="legendre",
        init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3),
    )
    cbm = ConditionalBornMachine(cfg=cfg, data_dim=2, num_classes=2, device=device)

    # Minimal DataHandler substitute using raw DataLoaders
    data = torch.zeros(16, 2)
    labels_t = torch.randint(0, 2, (16,))
    ds = TensorDataset(data, labels_t)
    loader = DataLoader(ds, batch_size=4)

    class _FakeDH:
        classification = {"train": loader, "valid": loader}
        def get_classification_loaders(self, **_): pass

    train_cfg = NLLConfig(max_epoch=3, batch_size=4, patience=250)
    trainer = SoftmaxTrainer(cbm=cbm, train_cfg=train_cfg, datahandler=_FakeDH(), device=device)

    logged = []
    trainer.train(on_epoch_end=lambda ep, m: logged.append((ep, m)))
    assert len(logged) == 3, f"Expected 3 epoch callbacks, got {len(logged)}"
    assert "acc/valid" in logged[-1][1]
    print(f"  SoftmaxTrainer 3 epochs OK, best_acc={trainer.best.get('acc', 'n/a'):.4f}")

    print("\nsoftmax.py smoke test passed.")
