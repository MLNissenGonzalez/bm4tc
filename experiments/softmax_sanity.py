"""
Softmax sanity-check experiment entry point.

Trains the MPS classifier using raw amplitudes as softmax logits,
matching tutorial MPS implementations. Used to sanity-check that the
MPS architecture can learn when trained with a standard softmax loss.

Two variants (configured via +experiments=tests/softmax/<config>):
  - legendre_mnist  : Legendre d3D10 embedding, MNIST
  - simp_mnist      : SimpEmbedding (1, x, 1-x) d3D10, MNIST

Run:
    python -m experiments.softmax_sanity +experiments=tests/softmax/legendre_mnist
    python -m experiments.softmax_sanity +experiments=tests/softmax/simp_mnist
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import hydra, torch, logging
from src.tracking.wandb_utils import init_wandb
from src.tracking import log_dataset_viz
from src.utils import schemas, set_seed
from src.data import DataHandler
from src.models import BornMachine
from src.trainer import ClassificationTrainer

logger = logging.getLogger(__name__)


class SoftmaxSanityTrainer(ClassificationTrainer):
    """
    Variant of ClassificationTrainer that feeds raw MPS amplitudes to the
    loss criterion instead of Born-rule probabilities.

    Only _train_epoch is overridden. All other logic (early stopping,
    evaluation, checkpointing) is inherited unchanged from ClassificationTrainer.

    Note: evaluation still uses Born-rule probabilities (bm.class_probabilities)
    because the PerformanceEvaluator is unchanged. Accuracy reported during
    training therefore reflects Born-rule, not softmax, discrimination.
    """

    def _train_epoch(self):
        losses = []
        self.bornmachine.classifier.train()
        for data, labels in self.datahandler.classification["train"]:
            data, labels = data.to(self.device), labels.to(self.device)
            self.step += 1

            embs = self.bornmachine.classifier.embed(data)
            amplitudes = self.bornmachine.classifier.amplitudes(embs)
            loss: torch.Tensor = self.criterion(amplitudes, labels)

            if torch.isnan(loss):
                logger.warning("NaN loss detected — aborting epoch.")
                self._nan_detected = True
                break

            self.optimizer.zero_grad()
            loss.backward()
            from src.tracking import log_grads
            log_grads(bm_view=self.bornmachine.classifier,
                      watch_freq=self.train_cfg.watch_freq,
                      step=self.step, stage=self.stage)
            self.optimizer.step()

            losses.append(loss.detach().cpu().item())
            import wandb
            wandb.log({f"{self.stage}/train/loss": sum(losses) / len(losses)})


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: schemas.Config):
    run = init_wandb(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datahandler = DataHandler(cfg.dataset)
    datahandler.load()

    set_seed(cfg.tracking.seed)

    bornmachine = BornMachine(cfg.born, datahandler.data_dim, datahandler.num_cls, device)

    datahandler.split_and_rescale(bornmachine)
    log_dataset_viz(datahandler)

    trainer = SoftmaxSanityTrainer(bornmachine, cfg, "pre", datahandler, device)
    trainer.train()

    run.finish()

    stop_crit = cfg.trainer.classification.stop_crit
    objective = trainer.best.get(stop_crit, float("inf"))
    if stop_crit in ["acc", "rob"]:
        objective = -objective
    return objective


if __name__ == "__main__":
    main()
