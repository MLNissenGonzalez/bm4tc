"""
Uncertainty Quantification (UQ) evaluation for Born Machines.

Born Machines learn the joint distribution p(x,c), enabling computation of
the marginal input likelihood p(x) = sum_c p(x,c). This provides two defense
mechanisms against adversarial examples:

1. **Detection**: Reject inputs whose likelihood falls below a threshold tau
2. **Purification**: For rejected inputs, find a nearby point x* maximizing
   likelihood within a perturbation ball, then classify x* instead

This module provides tools to evaluate both defenses by:
- Computing log p(x) on clean and adversarial data
- Calibrating detection thresholds from clean data percentiles
- Purifying adversarial examples and measuring accuracy recovery

Budget convention (see "Budget vocabulary" in CLAUDE.md): ``UQConfig`` is authored
entirely in *relative* fractions of the input domain (``eps_rel`` for the attacker,
``delta_rel`` for purification). :func:`evaluate_uq` converts them once, up front, and
everything below that point is absolute (``eps_abs`` / ``delta_abs``). Result dicts and
metric keys are keyed by the *relative* values.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import logging

logger = logging.getLogger(__name__)


@dataclass
class UQConfig:
    """Configuration for UQ evaluation.

    All budgets are RELATIVE — fractions of the input domain width ``hi - lo``.
    :func:`evaluate_uq` converts them to absolute model-domain values once.

    Attributes:
        norm: Lp norm for purification perturbation ball.
        num_steps: Gradient descent iterations for purification.
        step_size: Step size per iteration (None = auto).
        delta_rel: Purification radii to evaluate, as fractions of the input domain.
        percentiles: Percentiles of clean log p(x) for threshold candidates.
        attack_method: Attack method for generating adversarial inputs.
        eps_rel: Attack budgets, as fractions of the input domain.
        attack_num_steps: PGD steps for attack generation.
        random_start: Random start for purification.
    """
    # Purification params
    norm: int | str = "inf"
    num_steps: int = 20
    step_size: float | None = None
    # On legendre (width 2.0) these are absolute radii 0.1 / 0.2 / 0.3.
    delta_rel: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15])
    random_start: bool = False

    # Threshold params
    percentiles: List[float] = field(default_factory=lambda: [1, 5, 10, 20])

    # Attack params
    attack_method: str = "PGD"
    # On legendre (width 2.0) these are absolute epsilons 0.1 / 0.2 / 0.3.
    eps_rel: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15])
    attack_num_steps: int = 20

    # Gibbs purification params
    run_gibbs: bool = False
    gibbs_n_sweeps: List[int] = field(default_factory=lambda: [1, 3, 5])
    gibbs_num_bins: int = 200
    gibbs_batch_size: int = 8
    # Per-sweep L∞ step, as a fraction of the input range; None = unrestricted.
    # NOT a global budget: the window re-centres each sweep, so after k sweeps the
    # envelope is k*gibbs_step_delta_rel*(hi-lo). Strength is set by gibbs_n_sweeps.
    gibbs_step_delta_rel: Optional[float] = 0.1
    # Gibbs is ~99% of UQ cost and reduces to a mean over the test set, so it runs on a
    # fixed random subsample (cheap metrics keep the full set). None = full set.
    gibbs_subsample: Optional[int] = None
    gibbs_subsample_seed: int = 0  # fixed ⇒ same samples across model-seeds/alphas (paired)

    # Memory control
    eval_batch_size: Optional[int] = None  # chunk size for forwards; None = loader batch


def _recover_after_failure(born) -> None:
    """Restore a clean state after a failed/aborted block so later blocks are unaffected.

    A mid-contraction OOM leaves the tk network's data nodes dirty; reset() clears them,
    and empty_cache() releases the freed memory back to the allocator for the next block.
    """
    try:
        born.reset()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _batched_forward(fn, x: torch.Tensor, batch_size: Optional[int], device) -> torch.Tensor:
    """Apply a no-grad model forward `fn` over `x` in chunks, returning a CPU tensor.

    Keeps peak GPU memory bounded for full-test-set forwards (e.g. on MNIST, where a
    single forward over the whole test split would OOM). `batch_size=None` falls back
    to a single pass over all of `x`.
    """
    bs = batch_size if batch_size is not None else len(x)
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            outs.append(fn(x[i:i + bs].to(device)).cpu())
    return torch.cat(outs)


def compute_log_px(
    born,
    loader: DataLoader,
    device: torch.device,
    desc: str = "log p(x)",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute marginal log p(x) for all samples in a loader.

    Args:
        born: ConditionalBornMachine instance.
        loader: DataLoader yielding (data, labels) tuples.
        device: Torch device.
        desc: Label for the progress bar.

    Returns:
        Tuple of (log_px, labels) tensors concatenated over all batches.
    """
    all_log_px = []
    all_labels = []

    born.to(device)

    with torch.no_grad():
        for batch_data, batch_labels in tqdm(
            loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True
        ):
            batch_data = batch_data.to(device)
            log_px = born.marginal_log_probability(batch_data)
            all_log_px.append(log_px.cpu())
            all_labels.append(batch_labels)

    return torch.cat(all_log_px), torch.cat(all_labels)


def compute_thresholds(
    born,
    clean_loader: DataLoader,
    percentiles: List[float],
    device: torch.device,
) -> Tuple[Dict[float, float], torch.Tensor]:
    """Compute percentile-based detection thresholds from clean data.

    Standard approach from OOD detection literature: thresholds are set
    at percentiles of the clean data's log p(x) distribution.

    Args:
        born: ConditionalBornMachine instance.
        clean_loader: DataLoader for clean (in-distribution) data.
        percentiles: List of percentile values (e.g., [1, 5, 10, 20]).
        device: Torch device.

    Returns:
        Tuple of:
            - Dict mapping percentile -> threshold value.
            - Tensor of all clean log p(x) values.
    """
    clean_log_px, _ = compute_log_px(born, clean_loader, device)

    thresholds = {}
    for p in percentiles:
        thresholds[p] = float(np.percentile(clean_log_px.numpy(), p))

    return thresholds, clean_log_px


@dataclass
class PurificationMetrics:
    """Metrics for a single (eps_rel, delta_rel) purification evaluation.

    Attributes:
        accuracy_after_purify: Classification accuracy on purified samples.
        recovery_rate: Fraction of correctly classified after purification
            among those misclassified before purification.
        mean_log_px_before: Mean log p(x) of adversarial inputs.
        mean_log_px_after: Mean log p(x) of purified inputs.
        rejection_rate: Fraction of inputs below threshold after purification.
    """
    accuracy_after_purify: float
    recovery_rate: float
    mean_log_px_before: float
    mean_log_px_after: float
    rejection_rate: float


@dataclass
class UQResults:
    """Complete UQ evaluation results.

    All dicts are keyed by RELATIVE budgets (fractions of the input domain), matching
    the emitted metric keys.

    Attributes:
        clean_log_px: Log p(x) values for clean test data.
        clean_accuracy: Clean classification accuracy.
        thresholds: Dict mapping percentile -> threshold value.
        adv_log_px: Dict mapping eps_rel -> log p(x) values for adversarial data.
        adv_accuracies: Dict mapping eps_rel -> adversarial accuracy.
        detection_rates: Dict mapping (percentile, eps_rel) -> detection rate.
        purification_results: Dict mapping (eps_rel, delta_rel) -> PurificationMetrics.
    """
    clean_log_px: np.ndarray
    clean_accuracy: float
    thresholds: Dict[float, float]
    adv_log_px: Dict[float, np.ndarray]
    adv_accuracies: Dict[float, float]
    detection_rates: Dict[Tuple[float, float], float]
    purification_results: Dict[Tuple[float, float], PurificationMetrics]
    gibbs_purification_results: Dict[Tuple[float, int], PurificationMetrics] = field(
        default_factory=dict
    )
    clean_purification_results: Dict[float, PurificationMetrics] = field(
        default_factory=dict
    )
    clean_gibbs_purification_results: Dict[int, PurificationMetrics] = field(
        default_factory=dict
    )
    err_rate_detected: Dict[Tuple[float, float], float] = field(default_factory=dict)
    err_rate_passed: Dict[Tuple[float, float], float] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a formatted summary of UQ evaluation results."""
        lines = [
            "=" * 60,
            "Uncertainty Quantification Results",
            "=" * 60,
            f"Clean Accuracy: {self.clean_accuracy:.4f}",
            f"Clean log p(x): mean={self.clean_log_px.mean():.2f}, "
            f"std={self.clean_log_px.std():.2f}",
            "",
            "--- Detection Thresholds ---",
        ]
        for pct, tau in sorted(self.thresholds.items()):
            lines.append(f"  {pct}th percentile: tau = {tau:.4f}")

        lines.extend(["", "--- Adversarial Results ---"])
        for eps_rel in sorted(self.adv_accuracies.keys()):
            adv_lp = self.adv_log_px[eps_rel]
            lines.append(
                f"  eps_rel={eps_rel}: acc={self.adv_accuracies[eps_rel]:.4f}, "
                f"mean log p(x)={adv_lp.mean():.2f}"
            )

        lines.extend(["", "--- Detection Rates ---"])
        for (pct, eps_rel), rate in sorted(self.detection_rates.items()):
            err_det = self.err_rate_detected.get((pct, eps_rel), float("nan"))
            err_pas = self.err_rate_passed.get((pct, eps_rel), float("nan"))
            err_det_s = f"{err_det:.2%}" if not np.isnan(err_det) else "nan"
            err_pas_s = f"{err_pas:.2%}" if not np.isnan(err_pas) else "nan"
            lines.append(
                f"  tau={pct}th pct, eps_rel={eps_rel}: {rate:.2%} detected, "
                f"err_if_detected={err_det_s}, err_if_passed={err_pas_s}"
            )

        lines.extend(["", "--- Purification Results ---"])
        for (eps_rel, delta_rel), metrics in sorted(self.purification_results.items()):
            lines.append(
                f"  eps_rel={eps_rel}, delta_rel={delta_rel}: "
                f"acc={metrics.accuracy_after_purify:.4f}, "
                f"recovery={metrics.recovery_rate:.2%}, "
                f"log p(x) {metrics.mean_log_px_before:.2f} -> {metrics.mean_log_px_after:.2f}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


class UQEvaluation:
    """Main class for running UQ evaluation.

    Evaluates both detection and purification defenses against adversarial
    examples, using the Born Machine's marginal likelihood p(x).

    Example:
        >>> uq_eval = UQEvaluation(uq_config)
        >>> results = uq_eval.evaluate(born, test_loader, device)
        >>> print(results.summary())
    """

    def __init__(self, config: Optional[UQConfig] = None):
        """Initialize UQ evaluation.

        Args:
            config: UQ evaluation configuration. Uses defaults if None.
        """
        self.config = config or UQConfig()

    def evaluate(
        self,
        born,
        clean_loader: DataLoader,
        device: torch.device,
    ) -> UQResults:
        """Run the full UQ evaluation pipeline.

        Steps:
        0. Convert every relative budget in the config to absolute, once
        1. Cache log Z on the Born Machine
        2. Compute clean log p(x) and derive detection thresholds
        3. For each attack eps_rel: generate adversarial examples,
           compute log p(x_adv), detection rate
        4. For each (eps_rel, delta_rel): purify adversarial examples,
           classify, compute metrics
        5. Package into UQResults

        Results are keyed by the relative budgets; the absolute values exist only
        inside this method.

        Args:
            born: ConditionalBornMachine instance.
            clean_loader: DataLoader for clean test data.
            device: Torch device.

        Returns:
            UQResults with all evaluation metrics.
        """
        from src.utils.evasion import RobustnessEvaluation
        from src.utils.train import CriterionConfig
        from src.analysis.purification import LikelihoodPurification
        from src.utils.embeddings import range_size_of, rel_to_abs

        cfg = self.config
        born.to(device)

        # 0. The rel -> abs boundary. Below this point every budget is absolute;
        #    the relative values survive only as dict/metric keys.
        range_size = range_size_of(born)
        eps_abs_of = {r: rel_to_abs(r, range_size) for r in cfg.eps_rel}
        delta_abs_of = {r: rel_to_abs(r, range_size) for r in cfg.delta_rel}

        # Re-batch to a memory-safe chunk size so the gradient path (attack /
        # purification) and the per-batch forwards stay bounded on large inputs.
        if cfg.eval_batch_size is not None:
            clean_loader = DataLoader(
                clean_loader.dataset, batch_size=cfg.eval_batch_size, shuffle=False
            )

        # 1. Cache log Z
        logger.info("Computing partition function...")
        born.cache_log_Z()

        # 2. Compute clean log p(x) and thresholds
        logger.info("Computing clean log p(x) and thresholds...")
        thresholds, clean_log_px_tensor = compute_thresholds(
            born, clean_loader, cfg.percentiles, device
        )
        clean_log_px = clean_log_px_tensor.numpy()

        # Compute clean accuracy
        clean_correct = 0
        clean_total = 0
        with torch.no_grad():
            for batch_data, batch_labels in tqdm(
                clean_loader, desc="clean acc", unit="batch", leave=False, dynamic_ncols=True
            ):
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                probs = born.class_probabilities(batch_data)
                preds = probs.argmax(dim=1)
                clean_correct += (preds == batch_labels).sum().item()
                clean_total += len(batch_labels)
        clean_accuracy = clean_correct / clean_total
        logger.info(f"Clean accuracy: {clean_accuracy:.4f}")

        # 3. Generate adversarial examples and evaluate detection
        attack = RobustnessEvaluation(
            method=cfg.attack_method,
            norm=cfg.norm,
            criterion=CriterionConfig(name="nll", kwargs=None),
            eps_rel=cfg.eps_rel,
            num_steps=cfg.attack_num_steps,
            random_start=True,
        )

        adv_log_px: Dict[float, np.ndarray] = {}
        adv_accuracies: Dict[float, float] = {}
        detection_rates: Dict[Tuple[float, float], float] = {}
        err_rate_detected: Dict[Tuple[float, float], float] = {}
        err_rate_passed: Dict[Tuple[float, float], float] = {}
        # Store adversarial examples for purification
        adv_examples_cache: Dict[float, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

        for eps_rel in tqdm(
            cfg.eps_rel, desc="UQ attack", unit="eps", dynamic_ncols=True
        ):
            eps_abs = eps_abs_of[eps_rel]
            logger.info(
                f"Generating adversarial examples (eps_rel={eps_rel}, eps_abs={eps_abs})..."
            )
            try:
                all_adv_log_px = []
                all_adv_correct_list = []
                all_adv_correct = 0
                all_adv_total = 0
                adv_batches = []

                for batch_data, batch_labels in tqdm(
                    clean_loader, desc=f"attack eps_rel={eps_rel}", unit="batch",
                    leave=False, dynamic_ncols=True,
                ):
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)

                    # Generate adversarial examples
                    adv_data = attack.generate(
                        born, batch_data, batch_labels, eps_abs, device
                    )

                    # Classify adversarial examples
                    with torch.no_grad():
                        adv_probs = born.class_probabilities(adv_data)
                        adv_preds = adv_probs.argmax(dim=1)
                        correct_batch = adv_preds == batch_labels
                        all_adv_correct += correct_batch.sum().item()
                        all_adv_correct_list.append(correct_batch.cpu())
                        all_adv_total += len(batch_labels)

                        # Compute log p(x_adv)
                        log_px_adv = born.marginal_log_probability(adv_data)
                        all_adv_log_px.append(log_px_adv.cpu())

                    adv_batches.append((adv_data.detach().cpu(), batch_labels.cpu()))

                adv_log_px_arr = torch.cat(all_adv_log_px).numpy()
                adv_log_px[eps_rel] = adv_log_px_arr
                adv_accuracies[eps_rel] = all_adv_correct / all_adv_total
                adv_examples_cache[eps_rel] = adv_batches
                misclf_arr = ~torch.cat(all_adv_correct_list).numpy()

                logger.info(
                    f"  eps_rel={eps_rel}: adv_acc={adv_accuracies[eps_rel]:.4f}, "
                    f"mean log p(x_adv)={adv_log_px_arr.mean():.2f}"
                )

                # Detection rates and conditional error rates at each threshold
                for pct, tau in thresholds.items():
                    det_mask = adv_log_px_arr < tau
                    pas_mask = ~det_mask
                    detection_rates[(pct, eps_rel)] = float(det_mask.mean())
                    err_rate_detected[(pct, eps_rel)] = (
                        float(misclf_arr[det_mask].mean()) if det_mask.any() else float("nan")
                    )
                    err_rate_passed[(pct, eps_rel)] = (
                        float(misclf_arr[pas_mask].mean()) if pas_mask.any() else float("nan")
                    )
            except Exception as e:
                logger.warning(f"Detection/attack failed (eps_rel={eps_rel}): {e}; skipping")
                _recover_after_failure(born)

        # 4. Purification
        purifier = LikelihoodPurification(
            norm=cfg.norm,
            num_steps=cfg.num_steps,
            step_size=cfg.step_size,
            random_start=cfg.random_start,
        )

        purification_results: Dict[Tuple[float, float], PurificationMetrics] = {}

        for eps_rel in tqdm(
            cfg.eps_rel, desc="UQ purify", unit="eps", dynamic_ncols=True
        ):
            for delta_rel in cfg.delta_rel:
                delta_abs = delta_abs_of[delta_rel]
                logger.info(f"Purifying (eps_rel={eps_rel}, delta_rel={delta_rel})...")
                try:
                    all_purified_correct = 0
                    all_recovered = 0
                    all_misclassified_before = 0
                    all_log_px_before = []
                    all_log_px_after = []
                    all_total = 0
                    all_below_threshold = 0

                    # Use median threshold for rejection rate
                    median_pct = cfg.percentiles[len(cfg.percentiles) // 2]
                    tau = thresholds[median_pct]

                    for adv_data_cpu, labels_cpu in tqdm(
                        adv_examples_cache[eps_rel],
                        desc=f"purify eps_rel={eps_rel} d={delta_rel}", unit="batch",
                        leave=False, dynamic_ncols=True,
                    ):
                        adv_data = adv_data_cpu.to(device)
                        labels = labels_cpu.to(device)

                        # Log p(x) before purification
                        with torch.no_grad():
                            log_px_before = born.marginal_log_probability(adv_data)
                            # Classify before purification
                            adv_probs = born.class_probabilities(adv_data)
                            adv_preds = adv_probs.argmax(dim=1)
                            misclassified = (adv_preds != labels)

                        # Purify
                        purified, log_px_after = purifier.purify(
                            born, adv_data, delta_abs, device
                        )

                        # Classify after purification
                        with torch.no_grad():
                            pur_probs = born.class_probabilities(purified)
                            pur_preds = pur_probs.argmax(dim=1)
                            correct_after = (pur_preds == labels)

                        # Recovery: correctly classified after purification
                        # among those misclassified before
                        recovered = (misclassified & correct_after).sum().item()

                        all_purified_correct += correct_after.sum().item()
                        all_recovered += recovered
                        all_misclassified_before += misclassified.sum().item()
                        all_log_px_before.append(log_px_before.cpu())
                        all_log_px_after.append(log_px_after.cpu())
                        all_total += len(labels)
                        all_below_threshold += (log_px_after.cpu() < tau).sum().item()

                    acc_after = all_purified_correct / all_total
                    recovery = (
                        all_recovered / all_misclassified_before
                        if all_misclassified_before > 0
                        else 1.0
                    )
                    mean_before = torch.cat(all_log_px_before).mean().item()
                    mean_after = torch.cat(all_log_px_after).mean().item()
                    rejection_rate = all_below_threshold / all_total

                    purification_results[(eps_rel, delta_rel)] = PurificationMetrics(
                        accuracy_after_purify=acc_after,
                        recovery_rate=recovery,
                        mean_log_px_before=mean_before,
                        mean_log_px_after=mean_after,
                        rejection_rate=rejection_rate,
                    )

                    logger.info(
                        f"  eps_rel={eps_rel}, d={delta_rel}: "
                        f"acc={acc_after:.4f}, recovery={recovery:.2%}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Gradient purification failed (eps_rel={eps_rel}, "
                        f"delta_rel={delta_rel}): {e}; skipping"
                    )
                    _recover_after_failure(born)

        # 5. Clean purification (natural examples, no attack)
        clean_purification_results: Dict[float, PurificationMetrics] = {}
        for delta_rel in tqdm(
            cfg.delta_rel, desc="UQ clean purify", unit="delta", dynamic_ncols=True
        ):
            delta_abs = delta_abs_of[delta_rel]
            logger.info(f"Clean purification (delta_rel={delta_rel})...")
            try:
                all_correct = 0
                all_total = 0
                all_log_px_before = []
                all_log_px_after = []

                for batch_data, batch_labels in tqdm(
                    clean_loader, desc=f"clean purify d={delta_rel}", unit="batch",
                    leave=False, dynamic_ncols=True,
                ):
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)

                    with torch.no_grad():
                        log_px_before = born.marginal_log_probability(batch_data)

                    purified, log_px_after = purifier.purify(born, batch_data, delta_abs, device)

                    with torch.no_grad():
                        preds = born.class_probabilities(purified).argmax(dim=1)
                        all_correct += (preds == batch_labels).sum().item()
                        all_total += len(batch_labels)

                    all_log_px_before.append(log_px_before.cpu())
                    all_log_px_after.append(log_px_after.cpu())

                acc = all_correct / all_total
                clean_purification_results[delta_rel] = PurificationMetrics(
                    accuracy_after_purify=acc,
                    recovery_rate=float("nan"),
                    mean_log_px_before=torch.cat(all_log_px_before).mean().item(),
                    mean_log_px_after=torch.cat(all_log_px_after).mean().item(),
                    rejection_rate=0.0,
                )
                logger.info(f"  delta_rel={delta_rel}: clean_purify_acc={acc:.4f}")
            except Exception as e:
                logger.warning(
                    f"Clean purification failed (delta_rel={delta_rel}): {e}; skipping"
                )
                _recover_after_failure(born)

        # 6. Gibbs purification
        gibbs_purification_results: Dict[Tuple[float, int], PurificationMetrics] = {}
        clean_gibbs_purification_results: Dict[int, PurificationMetrics] = {}

        if cfg.run_gibbs:
            from src.analysis.purification import GibbsPurification

            gibbs_purifier = GibbsPurification(
                num_bins=cfg.gibbs_num_bins,
                gibbs_batch_size=cfg.gibbs_batch_size,
                step_delta_rel=cfg.gibbs_step_delta_rel,
            )
            sweep_points = sorted(set(cfg.gibbs_n_sweeps))

            def _gibbs_subsample(*tensors):
                """Take a fixed random subsample (shared across model-seeds) of inputs.

                Gibbs is ~99% of cost and only feeds a mean over the test set; estimating
                that mean on a fixed ~1k subsample keeps the statistic within ~±1.5%.
                """
                n = len(tensors[0])
                if cfg.gibbs_subsample is None or cfg.gibbs_subsample >= n:
                    return tensors
                rng = np.random.default_rng(cfg.gibbs_subsample_seed)
                idx = torch.from_numpy(rng.permutation(n)[: cfg.gibbs_subsample])
                return tuple(t[idx] for t in tensors)

            for eps_rel in tqdm(
                cfg.eps_rel, desc="Gibbs purify", unit="eps", dynamic_ncols=True
            ):
                try:
                    all_adv = torch.cat([b[0] for b in adv_examples_cache[eps_rel]])
                    all_labels = torch.cat([b[1] for b in adv_examples_cache[eps_rel]])
                    all_adv, all_labels = _gibbs_subsample(all_adv, all_labels)

                    # Recompute misclassification + mean log p(x) on the SAME subsample so
                    # accuracy/recovery/log-px are internally consistent.
                    adv_preds = _batched_forward(
                        born.class_probabilities, all_adv, cfg.eval_batch_size, device
                    ).argmax(dim=1)
                    misclassified = adv_preds != all_labels
                    mean_log_px_before = float(
                        _batched_forward(
                            born.marginal_log_probability, all_adv, cfg.eval_batch_size, device
                        ).mean()
                    )

                    snapshots = gibbs_purifier.purify_snapshots(
                        born, all_adv, sweep_points, device
                    )
                except Exception as e:
                    logger.warning(f"Gibbs failed (eps_rel={eps_rel}): {e}; skipping")
                    _recover_after_failure(born)
                    continue

                for n_sw, (x_purified, log_px_after) in snapshots.items():
                    try:
                        pur_preds = _batched_forward(
                            born.class_probabilities, x_purified, cfg.eval_batch_size, device
                        ).argmax(dim=1)
                        correct_after = pur_preds == all_labels
                        acc_after = correct_after.float().mean().item()
                        misclassified_before = misclassified.sum().item()
                        recovered = (misclassified & correct_after).sum().item()
                        recovery = (
                            recovered / misclassified_before
                            if misclassified_before > 0
                            else 1.0
                        )
                        gibbs_purification_results[(eps_rel, n_sw)] = PurificationMetrics(
                            accuracy_after_purify=acc_after,
                            recovery_rate=recovery,
                            mean_log_px_before=mean_log_px_before,
                            mean_log_px_after=float(log_px_after.mean()),
                            rejection_rate=0.0,
                        )
                        logger.info(
                            f"  eps_rel={eps_rel}, sweeps={n_sw}: "
                            f"acc={acc_after:.4f}, recovery={recovery:.2%}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Gibbs scoring failed (eps_rel={eps_rel}, n_sweeps={n_sw}): "
                            f"{e}; skipping"
                        )
                        _recover_after_failure(born)

            # Clean Gibbs purification
            all_clean = torch.cat([b for b, _ in clean_loader])
            all_clean_labels = torch.cat([lb for _, lb in clean_loader])
            all_clean, all_clean_labels = _gibbs_subsample(all_clean, all_clean_labels)
            try:
                clean_mean_log_px_before = float(
                    _batched_forward(
                        born.marginal_log_probability, all_clean, cfg.eval_batch_size, device
                    ).mean()
                )
                clean_snapshots = gibbs_purifier.purify_snapshots(
                    born, all_clean, sweep_points, device
                )
            except Exception as e:
                logger.warning(f"Clean Gibbs failed: {e}; skipping")
                _recover_after_failure(born)
                clean_snapshots = {}
                clean_mean_log_px_before = float("nan")

            for n_sw, (x_purified, log_px_after) in clean_snapshots.items():
                try:
                    pur_preds = _batched_forward(
                        born.class_probabilities, x_purified, cfg.eval_batch_size, device
                    ).argmax(dim=1)
                    acc = (pur_preds == all_clean_labels).float().mean().item()
                    clean_gibbs_purification_results[n_sw] = PurificationMetrics(
                        accuracy_after_purify=acc,
                        recovery_rate=float("nan"),
                        mean_log_px_before=clean_mean_log_px_before,
                        mean_log_px_after=float(log_px_after.mean()),
                        rejection_rate=0.0,
                    )
                    logger.info(f"  sweeps={n_sw}: clean_gibbs_acc={acc:.4f}")
                except Exception as e:
                    logger.warning(
                        f"Clean Gibbs scoring failed (n_sweeps={n_sw}): {e}; skipping"
                    )
                    _recover_after_failure(born)

        return UQResults(
            clean_log_px=clean_log_px,
            clean_accuracy=clean_accuracy,
            thresholds=thresholds,
            adv_log_px=adv_log_px,
            adv_accuracies=adv_accuracies,
            detection_rates=detection_rates,
            err_rate_detected=err_rate_detected,
            err_rate_passed=err_rate_passed,
            purification_results=purification_results,
            gibbs_purification_results=gibbs_purification_results,
            clean_purification_results=clean_purification_results,
            clean_gibbs_purification_results=clean_gibbs_purification_results,
        )
