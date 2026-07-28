"""PGD Adversarial Training for ConditionalBornMachine classifiers."""

import time
import torch
from tqdm import tqdm
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional
from src.utils.train import (
    OptimizerConfig,
    NormControlConfig,
    NormRegularizer,
    NormTracker,
    eval_metrics,
    eval_rob,
    eval_split,
    optimizer,
    resolve_log_target,
)
from src.utils.evasion import EvasionConfig, ProjectedGradientDescent, FastGradientMethod
from src.datahandler import DataHandler
from src.model import ConditionalBornMachine


@dataclass
class AdversarialConfig:
    """Configuration for :class:`AdversarialTrainer`.

    ``gen_on_clean`` selects between two training objectives at alpha > 0:

    * ``False`` (default, backward compatible) — the whole mixed NLL is fitted to
      adversarial examples: ``(1-cw)*mixed_nll(x_adv) + cw*mixed_nll(x)``.
    * ``True`` — the adversarial signal enters the *discriminative* half only,
      the generative half sees clean data:
      ``(1-a)*[(1-cw)*L_dis(x_adv) + cw*L_dis(x)] + a*L_gen(x)``.
      ``clean_weight`` therefore mixes inside the discriminative term (mixing the
      generative term with itself would be a no-op).

    With ``gen_on_clean=True`` validation switches to :func:`eval_split`, which
    runs *every* ``eval_rob_freq`` epochs and nothing in between. ``patience`` is
    then counted in validation events, not epochs: ``eval_rob_freq=5`` with
    ``patience=200`` means 1000 epochs without improvement.
    """
    max_epoch: int = 100
    batch_size: int = 64
    alpha: float = 0.0  # mixed-NLL weight for the training objective: alpha*gen + (1-alpha)*dis
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    evasion: EvasionConfig = field(default_factory=EvasionConfig)
    stop_crit: str = "acc"
    patience: int = 250
    eval_rob_freq: int = 5
    clean_weight: float = 0.0
    gen_on_clean: bool = False  # adversarial signal in the discriminative term only
    acc_floor: Optional[float] = None  # min clean valid-acc for a model to be selectable
    curriculum: bool = False
    curriculum_start: float = 0.0
    curriculum_end_epoch: Optional[int] = None
    # Norm control is opt-in for AT (defaults off): hard_every=0 + soft_strength=0
    # is a complete no-op, preserving the small-lr / normalized-start regime.
    norm_control: NormControlConfig = field(
        default_factory=lambda: NormControlConfig(hard_every=0)
    )
    save: bool = False

import logging
logger = logging.getLogger(__name__)

_LOSS_METRICS = {"dis_loss", "gen_loss", "mixed_loss"}
_ACC_METRICS = {"acc", "rob"}
_VALID_STOP_CRIT = {"dis_loss", "gen_loss", "mixed_loss", "acc", "rob"}

# Constant (not the run seed) so every seed in a sweep evaluates robustness on
# the same validation samples, keeping cross-seed rob comparisons clean.
_ADV_SUBSET_SEED = 0


class AdversarialTrainer:
    """PGD adversarial training for ConditionalBornMachine classifiers."""

    def __init__(
            self,
            cbm: ConditionalBornMachine,
            train_cfg: AdversarialConfig,
            datahandler: DataHandler,
            device: torch.device
    ):
        self.datahandler = datahandler
        self.device = device
        self.train_cfg = train_cfg

        if self.datahandler.classification is None:
            self.datahandler.get_classification_loaders(batch_size=self.train_cfg.batch_size)

        self._init_best()

        self.cbm = cbm
        self.best_tensors = [t.cpu().clone().detach() for t in self.cbm.tensors]
        self._init_attack()

        self._nc = train_cfg.norm_control
        self.norm_regularizer: NormRegularizer | None = None
        self._nc_log_target: float | None = None

        self.adv_indices: set[int] = set()
        self._n_rob_valid: int | None = None
        if train_cfg.gen_on_clean:
            self._init_split()

    def _init_split(self):
        """Validate the split objective and draw the fixed valid attack subset.

        The subset holds ``(1 - clean_weight) * n_valid`` positions in the valid
        loader's iteration order (stable: non-train splits are not shuffled). It
        is drawn once, from a constant seed, so the rob curve within a run is not
        perturbed by resampling and is comparable across seeds.
        """
        cfg = self.train_cfg

        if cfg.eval_rob_freq < 1:
            raise ValueError(
                "gen_on_clean=True requires eval_rob_freq >= 1: it is the cadence "
                "of the whole validation pass, not just of the robustness eval."
            )
        if cfg.alpha >= 1.0:
            logger.warning(
                "gen_on_clean=True with alpha=1: the discriminative term vanishes, "
                "so training reduces to clean generative NLL and the generated "
                "adversarial examples are discarded."
            )
        if cfg.clean_weight >= 1.0:
            logger.warning(
                "gen_on_clean=True with clean_weight=1: no adversarial examples "
                "enter the objective and no valid samples are attacked, so 'rob' "
                "is never reported."
            )

        n = len(self.datahandler.classification["valid"].dataset)
        k = min(n, max(0, int(round((1.0 - cfg.clean_weight) * n))))
        gen = torch.Generator().manual_seed(_ADV_SUBSET_SEED)
        self.adv_indices = set(torch.randperm(n, generator=gen)[:k].tolist())
        logger.info(
            f"Split objective: attacking {k}/{n} valid samples every "
            f"{cfg.eval_rob_freq} epoch(s); patience={cfg.patience} valid events "
            f"(~{cfg.patience * cfg.eval_rob_freq} epochs)."
        )

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

        range_size = self.cbm.input_range[1] - self.cbm.input_range[0]
        self.range_size = range_size
        self.base_epsilon = (evasion.strengths[0] if evasion.strengths else 0.1) * range_size
        self._abs_curriculum_start = self.train_cfg.curriculum_start * range_size

    def _init_best(self):
        self.best = {
            "dis_loss": float("inf"), "gen_loss": float("inf"),
            "mixed_loss": float("inf"), "acc": 0.0,
        }
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
            born=self.cbm,
            naturals=data,
            labels=labels,
            strength=epsilon,
            device=self.device
        )

    def _split_nll(self, adv_data, data, labels, tracker: NormTracker):
        """Adversarial signal in the discriminative term only (``gen_on_clean``).

            L = (1-a)*[(1-cw)*L_dis(x_adv) + cw*L_dis(x)] + a*L_gen(x)

        ``mixed_nll(x, y, a) = (1-a)*L_dis + a*L_gen`` decomposes exactly, so both
        clean terms fold into a single call at a rescaled alpha: with
        ``s = (1-a)*cw + a`` and ``a' = a/s``,

            s * mixed_nll(x, y, a') = (1-a)*cw*L_dis(x) + a*L_gen(x)

        which keeps the step at two forwards (one log_Z) rather than three. At
        least one of the two weights is always positive, so the sum is never an
        empty tensor-free float.
        """
        alpha = self.train_cfg.alpha
        cw = self.train_cfg.clean_weight
        adv_w = (1.0 - alpha) * (1.0 - cw)
        s = (1.0 - alpha) * cw + alpha

        terms = []
        if adv_w > 0.0:
            terms.append(adv_w * self.cbm.mixed_nll(adv_data, labels, alpha=0.0))
            # Amplitudes explode on the adversarial batch; record before the clean
            # forward overwrites the cache.
            tracker.record_amp(self.cbm)
        if s > 0.0:
            terms.append(s * self.cbm.mixed_nll(data, labels, alpha=alpha / s))
            if adv_w <= 0.0:
                tracker.record_amp(self.cbm)

        return terms[0] if len(terms) == 1 else terms[0] + terms[1]

    def _train_epoch(self, epsilon: float):
        losses, nll_losses, reg_losses = [], [], []
        tracker = NormTracker()
        self.cbm.train()

        for data, labels in self.datahandler.classification["train"]:
            data, labels = data.to(self.device), labels.to(self.device)
            self.step += 1

            self.cbm.eval()
            adv_data = self._generate_adversarial(data, labels, epsilon)
            self.cbm.train()

            if self.train_cfg.gen_on_clean:
                nll = self._split_nll(adv_data, data, labels, tracker)
            else:
                adv_loss = self.cbm.mixed_nll(adv_data, labels, alpha=self.train_cfg.alpha)
                # Track amplitude stats on the adversarial batch (where amplitudes
                # explode), before the optional clean forward overwrites the cache.
                tracker.record_amp(self.cbm)

                if self.train_cfg.clean_weight > 0:
                    clean_loss = self.cbm.mixed_nll(data, labels, alpha=self.train_cfg.alpha)
                    nll = (1 - self.train_cfg.clean_weight) * adv_loss + \
                          self.train_cfg.clean_weight * clean_loss
                else:
                    nll = adv_loss

            if self.norm_regularizer is not None:
                reg = self.norm_regularizer(self.cbm)
                loss = nll + reg
            else:
                reg = None
                loss = nll

            # log_Z is cached by mixed_nll (alpha>0) or the regularizer (soft);
            # read it before optimizer.step() invalidates the cache.
            tracker.record_logZ(self.cbm)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # The soft penalty reads the with-grad log_Z via recompute=False; the
            # in-place optimizer step left that cache pointing at a freed graph, so
            # drop it before the next step (mirrors NLLTrainer). Only needed when a
            # regularizer is active — renormalize_() invalidates the cache itself.
            if self.norm_regularizer is not None:
                self.cbm._invalidate_log_Z_cache()

            if self._nc.hard_every > 0 and (self.step % self._nc.hard_every == 0):
                self.cbm.renormalize_(log_target=self._nc_log_target)

            nll_losses.append(nll.detach().cpu().item())
            reg_losses.append(reg.detach().cpu().item() if reg is not None else 0.0)
            losses.append(loss.detach().cpu().item())

        n = len(losses)
        self._train_loss = sum(losses)     / n if losses else float("nan")
        self._train_nll  = sum(nll_losses) / n if nll_losses else float("nan")
        self._train_reg  = sum(reg_losses) / n if reg_losses else float("nan")
        self._norm_stats = tracker.finalize(self.cbm)

    def _update(self):
        """Check if valid_perf improved; update best tensors and patience counter."""
        current_value = self.valid_perf.get(self.stopping_criterion_name)
        if current_value is None:
            return

        # Clean-accuracy floor: never select a model whose clean acc has collapsed.
        # Sub-floor epochs count as non-improvements. If no epoch ever meets the
        # floor, `best` keeps its init (e.g. rob=0.0) and `best_tensors` stays the
        # starting (pretrained) model — HPO then sees the worst objective, and a
        # saved seed_sweep run falls back to the clean pretrained model.
        floor = self.train_cfg.acc_floor
        if floor is not None and self.valid_perf.get("acc", 1.0) < floor:
            self.patience_counter += 1
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
            self.best_tensors = [t.clone().detach() for t in self.cbm.tensors]
            self.best_epoch = self.epoch
            self.patience_counter = 0
        else:
            self.patience_counter += 1

    def _summarise_training(self, output_dir: Optional[Path]):
        """Restore best tensors and clean up. No test eval."""
        self.cbm.initialize(tensors=self.best_tensors)
        self.cbm.reset()
        self.cbm.to("cpu")

        if hasattr(self, "valid_perf"):
            del self.valid_perf

        if self.train_cfg.save and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.cbm.save(str(output_dir / "model"))

        logger.info(f"Adversarial Trainer ({self.train_cfg.evasion.method}) finished.")

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

        self.cbm.prepare(device=self.device)
        self._nc_log_target = resolve_log_target(self.cbm, self.datahandler, self._nc)
        if self._nc.soft_strength > 0.0:
            self.norm_regularizer = NormRegularizer(
                strength=self._nc.soft_strength, log_target=self._nc_log_target
            )
        self.optimizer = optimizer(self.cbm.parameters(), self.train_cfg.optimizer)

        rob_freq = self.train_cfg.eval_rob_freq
        split = self.train_cfg.gen_on_clean

        logger.info(f"Adversarial training ({self.train_cfg.evasion.method}) begins.")

        pbar = tqdm(range(self.train_cfg.max_epoch), desc="ADV", unit="ep", dynamic_ncols=True)
        for epoch in pbar:
            epoch_start = time.perf_counter()
            self.epoch = epoch + 1

            epsilon = self._get_epsilon(self.epoch)
            self._train_epoch(epsilon)

            # Split mode runs one combined clean+robust pass every rob_freq epochs
            # and nothing in between, so patience is counted in valid events.
            do_valid = (not split) or (self.epoch % rob_freq == 0)

            if do_valid and split:
                self.valid_perf = eval_split(
                    self.cbm, self.datahandler.classification["valid"],
                    self.attack, self.base_epsilon, self.device,
                    alpha=self.train_cfg.alpha,
                    clean_weight=self.train_cfg.clean_weight,
                    adv_indices=self.adv_indices,
                )
                # Kept out of valid_perf so it never leaks into `best`.
                self._n_rob_valid = self.valid_perf.pop("n_rob")
            elif do_valid:
                dis_loss, acc, gen_loss = eval_metrics(
                    self.cbm, self.datahandler.classification["valid"], self.device
                )
                alpha = self.train_cfg.alpha
                mixed_loss = alpha * gen_loss + (1 - alpha) * dis_loss
                self.valid_perf = {
                    "dis_loss": dis_loss, "gen_loss": gen_loss,
                    "mixed_loss": mixed_loss, "acc": acc,
                }

                if rob_freq and (self.epoch % rob_freq == 0):
                    rob = eval_rob(
                        self.cbm, self.datahandler.classification["valid"],
                        self.attack, self.base_epsilon, self.device
                    )
                    self.valid_perf["rob"] = rob

            postfix = {"loss": f"{self._train_loss:.4f}"}
            if do_valid:
                postfix["acc"] = f"{self.valid_perf['acc']:.4f}"
            postfix["logZ"] = f"{self._norm_stats.get('norm/log_Z_mean', float('nan')):.3g}"
            if do_valid and "rob" in self.valid_perf:
                postfix["rob"] = f"{self.valid_perf['rob']:.4f}"
            pbar.set_postfix(**postfix)

            if on_epoch_end is not None:
                metrics = {
                    "dis_loss/train":   self._train_nll,
                    "reg/train":        self._train_reg,
                    "epsilon/train":    epsilon,
                }
                metrics.update(self._norm_stats)
                if do_valid:
                    metrics.update({
                        "dis_loss/valid":   self.valid_perf["dis_loss"],
                        "gen_loss/valid":   self.valid_perf["gen_loss"],
                        "mixed_loss/valid": self.valid_perf["mixed_loss"],
                        "acc/valid":        self.valid_perf["acc"],
                    })
                    if "rob" in self.valid_perf:
                        metrics["rob/valid"] = self.valid_perf["rob"]
                    if self._n_rob_valid is not None:
                        # rob/valid is a (1-clean_weight) subset estimator in split
                        # mode; log its size so that stays recoverable from the run.
                        metrics["n_rob_valid"] = self._n_rob_valid
                on_epoch_end(self.epoch, metrics)

            if do_valid:
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
    from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig
    from src.datahandler import DataHandler
    from src.datahandler import DatasetConfig, DataGenDowConfig
    from src.utils.evasion import EvasionConfig

    device = torch.device("cpu")
    cbm = ConditionalBornMachine(
        cfg=CBMConfig(
            embedding="legendre",
            init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3),
        ),
        data_dim=2, num_classes=2, device=device,
    )
    ds_cfg = DatasetConfig(
        name="spirals",
        gen_dow_kwargs=DataGenDowConfig(name="spirals", size=32, seed=42, noise=0.1),
        overwrite=True,
    )
    dh = DataHandler(ds_cfg)
    dh.load()
    dh.split_and_rescale(cbm)

    # epsilon=0.05 as fraction of legendre range (2.0) → 0.1 absolute
    evasion_cfg = EvasionConfig(method="PGD", num_steps=3, strengths=[0.05])
    train_cfg = AdversarialConfig(
        max_epoch=10, batch_size=4, patience=250,
        evasion=evasion_cfg, eval_rob_freq=2,
    )
    trainer = AdversarialTrainer(cbm=cbm, train_cfg=train_cfg, datahandler=dh, device=device)

    logged = []
    trainer.train(on_epoch_end=lambda ep, m: logged.append((ep, m)))

    assert len(logged) == 10, f"Expected 10 epochs, got {len(logged)}"
    rob_epochs = [m for _, m in logged if "rob/valid" in m]
    assert len(rob_epochs) == 5, f"Expected 5 rob evals (every 2 epochs), got {len(rob_epochs)}"
    last_ep, last_m = logged[-1]
    print(f"  epoch={last_ep}  dis_loss/valid={last_m['dis_loss/valid']:.4f}  acc/valid={last_m['acc/valid']:.4f}")
    print(f"  rob evals at epochs: {[ep for ep, m in logged if 'rob/valid' in m]}")
    print("adversarial.py smoke test passed.")
