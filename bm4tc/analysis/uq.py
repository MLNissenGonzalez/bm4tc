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
- Calibrating detection thresholds from percentiles of clean log p(x) on a held-out
  calibration split (validation), never on the evaluated split (D2)
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
from functools import partial

from bm4tc.core.graphs import GraphCaptureError, Graphs
from bm4tc.core.interface import class_probabilities, log_px

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

    # Sweep purification (Gibbs for the MPS by default; JEM passes its SGLD
    # purifier to evaluate()): snapshots after each count in `sweeps`
    run_sweeps: bool = False
    sweeps: List[int] = field(default_factory=lambda: [1, 3, 5])
    gibbs_num_bins: int = 200
    gibbs_batch_size: int = 8
    # Per-sweep L∞ step, as a fraction of the input range; None = unrestricted.
    # NOT a global budget: the window re-centres each sweep, so after k sweeps the
    # envelope is k*gibbs_step_delta_rel*(hi-lo). Strength is set by sweeps.
    gibbs_step_delta_rel: Optional[float] = 0.1
    # Sweeps are ~99% of UQ cost and reduce to a mean over the test set, so it runs on a
    # fixed random subsample (cheap metrics keep the full set). None = full set.
    sweep_subsample: Optional[int] = None
    sweep_subsample_seed: int = 0  # fixed ⇒ same samples across model-seeds/betas (paired)

    # Memory control
    eval_batch_size: Optional[int] = None  # chunk size for forwards; None = loader batch


def _recover_after_failure(model) -> None:
    """Restore a clean state after a failed/aborted block so later blocks are unaffected.

    A mid-contraction OOM leaves the tk network's data nodes dirty; reset() clears them,
    and empty_cache() releases the freed memory back to the allocator for the next block.
    """
    try:
        model.reset()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _BatchWork:
    """The UQ evaluation's work on one batch: log p(x), the predicted class, the
    attack and the likelihood purification, each replayed from a CUDA graph when
    ``graphs`` are enabled (D92), one graph per batch shape. The radii are device
    scalars there (float64, as in training), so one graph serves every budget;
    eager, they stay floats, as before.
    """

    def __init__(self, model, attack, purifier, graphs: Graphs, device):
        self.device = device
        self._captured = graphs.enabled

        def marginal(x):
            with torch.no_grad():
                return log_px(model, x)

        def predicted_class(x):
            with torch.no_grad():
                return class_probabilities(model, x).argmax(dim=1)

        self.log_px = graphs.wrap(marginal)
        self.predict = graphs.wrap(predicted_class)
        self._attack = graphs.wrap(
            lambda x, labels, radius: attack.generate(model, x, labels, radius, device))
        self._purify = graphs.wrap(
            lambda x, radius: purifier.purify(model, x, radius, device))

    def _radius(self, radius: float):
        if not self._captured:
            return radius
        return torch.tensor(radius, dtype=torch.float64, device=self.device)

    def attack(self, x, labels, eps_abs: float) -> torch.Tensor:
        return self._attack(x, labels, self._radius(eps_abs))

    def purify(self, x, delta_abs: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """(purified x, its log p(x))."""
        return self._purify(x, self._radius(delta_abs))


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
    model,
    loader: DataLoader,
    device: torch.device,
    desc: str = "log p(x)",
    batch_log_px=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute marginal log p(x) for all samples in a loader.

    Args:
        model: ConditionalBornMachine instance.
        loader: DataLoader yielding (data, labels) tuples.
        device: Torch device.
        desc: Label for the progress bar.
        batch_log_px: log p(x) of one batch (a graphed one, :class:`_BatchWork`);
            None is the eager ``log_px(model, ·)``.

    Returns:
        Tuple of (log_px, labels) tensors concatenated over all batches.
    """
    all_log_px = []
    all_labels = []
    batch_log_px = batch_log_px or partial(log_px, model)

    model.to(device)

    with torch.no_grad():
        for batch_data, batch_labels in tqdm(
            loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True
        ):
            batch_data = batch_data.to(device)
            all_log_px.append(batch_log_px(batch_data).cpu())
            all_labels.append(batch_labels)

    return torch.cat(all_log_px), torch.cat(all_labels)


def compute_thresholds(
    model,
    clean_loader: DataLoader,
    percentiles: List[float],
    device: torch.device,
    batch_log_px=None,
) -> Tuple[Dict[float, float], torch.Tensor]:
    """Compute percentile-based detection thresholds from clean data.

    Standard approach from OOD detection literature: thresholds are set
    at percentiles of the clean data's log p(x) distribution.

    Args:
        model: ConditionalBornMachine instance.
        clean_loader: DataLoader for clean (in-distribution) data.
        percentiles: List of percentile values (e.g., [1, 5, 10, 20]).
        device: Torch device.
        batch_log_px: as for :func:`compute_log_px`.

    Returns:
        Tuple of:
            - Dict mapping percentile -> threshold value.
            - Tensor of all clean log p(x) values.
    """
    clean_log_px, _ = compute_log_px(model, clean_loader, device, batch_log_px=batch_log_px)

    thresholds = {}
    for p in percentiles:
        thresholds[p] = float(np.percentile(clean_log_px.numpy(), p))

    return thresholds, clean_log_px


@dataclass
class DetectionMetrics:
    """Detection outcome at a single threshold percentile.

    Attributes:
        detection_rate: Fraction of adversarial inputs flagged. ``nan`` when any
            score is non-finite -- see :func:`detection_metrics`.
        err_rate_detected: Misclassification rate among flagged inputs; ``nan`` if
            nothing was flagged.
        err_rate_passed: Misclassification rate among passed inputs; ``nan`` if
            nothing passed.
        chance: The rate a signal-free detector produces, ``percentile / 100``: the
            clean flag rate on the calibration split by construction. On the evaluated
            split the clean flag rate is measured (``UQResults.clean_flagged``).
        lift: ``detection_rate - chance``. The quantity that actually says whether the
            detector did anything.
        n_nonfinite: Count of non-finite scores in ``adv_log_px``.
        degenerate: True when every finite score is identical, so the threshold cannot
            separate anything.
    """
    detection_rate: float
    err_rate_detected: float
    err_rate_passed: float
    chance: float
    lift: float
    n_nonfinite: int
    degenerate: bool


def detection_metrics(
    adv_log_px: np.ndarray,
    misclassified: np.ndarray,
    thresholds: Dict[float, float],
) -> Dict[float, DetectionMetrics]:
    """Score adversarial log-densities against calibrated thresholds.

    An input is flagged when ``log p(x_adv) < tau``. The comparison is deliberately
    **strict**: a score exactly equal to the threshold is *not* flagged. That choice
    is what turns a degenerate (constant) score into a detection rate of exactly 0.0
    at every percentile, which is why ``degenerate`` is reported alongside.

    Non-finite scores are not silently treated as "not detected" (``nan < tau`` is
    ``False``, which would read as a confident 0.0). They are counted, and the rate
    is returned as ``nan`` so the failure is visible rather than plausible.

    Args:
        adv_log_px: log p(x) of the adversarial inputs, shape (N,).
        misclassified: Boolean array, True where the model got the adversarial input
            wrong, shape (N,).
        thresholds: Mapping percentile -> tau, as returned by :func:`compute_thresholds`.

    Returns:
        Dict mapping percentile -> :class:`DetectionMetrics`.
    """
    adv_log_px = np.asarray(adv_log_px)
    misclassified = np.asarray(misclassified, dtype=bool)

    finite = np.isfinite(adv_log_px)
    n_nonfinite = int((~finite).sum())
    degenerate = bool(finite.any() and np.ptp(adv_log_px[finite]) == 0.0)

    out: Dict[float, DetectionMetrics] = {}
    for pct, tau in thresholds.items():
        det_mask = adv_log_px < tau
        pas_mask = ~det_mask
        chance = float(pct) / 100.0
        rate = float("nan") if n_nonfinite else float(det_mask.mean())
        out[pct] = DetectionMetrics(
            detection_rate=rate,
            err_rate_detected=(
                float(misclassified[det_mask].mean()) if det_mask.any() else float("nan")
            ),
            err_rate_passed=(
                float(misclassified[pas_mask].mean()) if pas_mask.any() else float("nan")
            ),
            chance=chance,
            lift=rate - chance,
            n_nonfinite=n_nonfinite,
            degenerate=degenerate,
        )
    return out


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
        clean_flagged: Dict mapping percentile -> fraction of clean test data flagged
            (the false-positive rate of the threshold calibrated on validation).
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
    clean_flagged: Dict[float, float]
    adv_log_px: Dict[float, np.ndarray]
    adv_accuracies: Dict[float, float]
    detection_rates: Dict[Tuple[float, float], float]
    purification_results: Dict[Tuple[float, float], PurificationMetrics]
    sweep_purification_results: Dict[Tuple[float, int], PurificationMetrics] = field(
        default_factory=dict
    )
    clean_purification_results: Dict[float, PurificationMetrics] = field(
        default_factory=dict
    )
    clean_sweep_purification_results: Dict[int, PurificationMetrics] = field(
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
        >>> results = uq_eval.evaluate(model, test_loader, device)
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
        model,
        clean_loader: DataLoader,
        device: torch.device,
        *,
        calib_loader: DataLoader,
        sweep_purifier=None,
        graphs: Optional[Graphs] = None,
    ) -> UQResults:
        """Run the full UQ evaluation pipeline.

        Steps:
        0. Convert every relative budget in the config to absolute, once
        1. Cache log Z on the Born Machine
        2. Derive detection thresholds from clean log p(x) on ``calib_loader``;
           compute clean log p(x) on ``clean_loader``
        3. For each attack eps_rel: generate adversarial examples,
           compute log p(x_adv), detection rate
        4. For each (eps_rel, delta_rel): purify adversarial examples,
           classify, compute metrics
        5. Package into UQResults

        Results are keyed by the relative budgets; the absolute values exist only
        inside this method.

        Args:
            model: ConditionalBornMachine instance.
            clean_loader: DataLoader for clean test data.
            device: Torch device.
            calib_loader: DataLoader for the clean calibration split (validation),
                which sets the detection thresholds (D2).
            graphs: replay the per-batch work (attack, likelihood purification,
                log p(x), prediction) from CUDA graphs if enabled (D92); None is
                eager. The sweep purification stays eager.

        Returns:
            UQResults with all evaluation metrics.
        """
        from bm4tc.core.attacks import EvasionConfig, build_attack
        from bm4tc.analysis.purification import LikelihoodPurification
        from bm4tc.core.embeddings import range_size_of, rel_to_abs

        cfg = self.config
        model.to(device)

        # 0. The rel -> abs boundary. Below this point every budget is absolute;
        #    the relative values survive only as dict/metric keys.
        range_size = range_size_of(model)
        eps_abs_of = {r: rel_to_abs(r, range_size) for r in cfg.eps_rel}
        delta_abs_of = {r: rel_to_abs(r, range_size) for r in cfg.delta_rel}

        # Re-batch to a memory-safe chunk size so the gradient path (attack /
        # purification) and the per-batch forwards stay bounded on large inputs.
        if cfg.eval_batch_size is not None:
            clean_loader = DataLoader(
                clean_loader.dataset, batch_size=cfg.eval_batch_size, shuffle=False
            )

        # 1. Cache log Z
        logger.info("Computing the normalizer (MPS: log Z)...")
        model.log_normalizer()

        # 2. Thresholds from the calibration split; clean log p(x) on the test split
        logger.info("Calibrating thresholds and computing clean log p(x)...")
        if cfg.eval_batch_size is not None:
            calib_loader = DataLoader(
                calib_loader.dataset, batch_size=cfg.eval_batch_size, shuffle=False
            )
        # 3. and 4. build the attack and the purifier; their per-batch work, and
        #    log p(x) and the prediction, go through `work`.
        attack = build_attack(EvasionConfig(
            method=cfg.attack_method,
            norm=cfg.norm,
            num_steps=cfg.attack_num_steps,
            random_start=True,
        ))
        likelihood_purifier = LikelihoodPurification(
            norm=cfg.norm,
            num_steps=cfg.num_steps,
            step_size=cfg.step_size,
            random_start=cfg.random_start,
        )
        work = _BatchWork(model, attack, likelihood_purifier,
                          graphs if graphs is not None else Graphs(enabled=False), device)

        thresholds, _ = compute_thresholds(model, calib_loader, cfg.percentiles, device,
                                           batch_log_px=work.log_px)
        clean_log_px = compute_log_px(model, clean_loader, device,
                                      batch_log_px=work.log_px)[0].numpy()
        clean_flagged = {p: float((clean_log_px < tau).mean()) for p, tau in thresholds.items()}

        # Compute clean accuracy
        clean_correct = 0
        clean_total = 0
        with torch.no_grad():
            for batch_data, batch_labels in tqdm(
                clean_loader, desc="clean acc", unit="batch", leave=False, dynamic_ncols=True
            ):
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                preds = work.predict(batch_data)
                clean_correct += (preds == batch_labels).sum().item()
                clean_total += len(batch_labels)
        clean_accuracy = clean_correct / clean_total
        logger.info(f"Clean accuracy: {clean_accuracy:.4f}")

        # 3. Generate adversarial examples and evaluate detection

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
                    adv_data = work.attack(batch_data, batch_labels, eps_abs)

                    # Classify adversarial examples
                    adv_preds = work.predict(adv_data)
                    correct_batch = adv_preds == batch_labels
                    all_adv_correct += correct_batch.sum().item()
                    all_adv_correct_list.append(correct_batch.cpu())
                    all_adv_total += len(batch_labels)

                    # Compute log p(x_adv)
                    all_adv_log_px.append(work.log_px(adv_data).cpu())

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
                det = detection_metrics(adv_log_px_arr, misclf_arr, thresholds)
                for pct, m in det.items():
                    detection_rates[(pct, eps_rel)] = m.detection_rate
                    err_rate_detected[(pct, eps_rel)] = m.err_rate_detected
                    err_rate_passed[(pct, eps_rel)] = m.err_rate_passed

                first = next(iter(det.values()))
                if first.n_nonfinite:
                    logger.warning(
                        f"  eps_rel={eps_rel}: {first.n_nonfinite}/{len(adv_log_px_arr)} "
                        "non-finite log p(x_adv); detection reported as nan"
                    )
                if first.degenerate:
                    logger.warning(
                        f"  eps_rel={eps_rel}: log p(x_adv) is constant; detection is "
                        "0.0 at every threshold by the strict < convention, not by signal"
                    )
                for pct, m in sorted(det.items()):
                    logger.info(
                        f"    tau={pct}pct: det={m.detection_rate:.4f} "
                        f"(chance {m.chance:.2f}, lift {m.lift:+.4f})"
                    )
            except GraphCaptureError:
                raise   # fatal for the process (graphs.py): no budget would succeed
            except Exception as e:
                logger.warning(f"Detection/attack failed (eps_rel={eps_rel}): {e}; skipping")
                _recover_after_failure(model)

        # 4. Purification
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

                        # Log p(x) and the class before purification
                        log_px_before = work.log_px(adv_data)
                        misclassified = work.predict(adv_data) != labels

                        purified, log_px_after = work.purify(adv_data, delta_abs)

                        # Classify after purification
                        correct_after = work.predict(purified) == labels

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
                except GraphCaptureError:
                    raise
                except Exception as e:
                    logger.warning(
                        f"Gradient purification failed (eps_rel={eps_rel}, "
                        f"delta_rel={delta_rel}): {e}; skipping"
                    )
                    _recover_after_failure(model)

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

                    log_px_before = work.log_px(batch_data)
                    purified, log_px_after = work.purify(batch_data, delta_abs)
                    preds = work.predict(purified)
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
            except GraphCaptureError:
                raise
            except Exception as e:
                logger.warning(
                    f"Clean purification failed (delta_rel={delta_rel}): {e}; skipping"
                )
                _recover_after_failure(model)

        # 6. Sweep purification: anything with purify_snapshots(model, x, sweep_points,
        #    device) -> {k: (x_k, log_px_k)}; Gibbs (MPS) unless one is given
        sweep_purification_results: Dict[Tuple[float, int], PurificationMetrics] = {}
        clean_sweep_purification_results: Dict[int, PurificationMetrics] = {}

        if cfg.run_sweeps:
            from bm4tc.analysis.purification import GibbsPurification

            purifier = sweep_purifier or GibbsPurification(
                num_bins=cfg.gibbs_num_bins,
                gibbs_batch_size=cfg.gibbs_batch_size,
                step_delta_rel=cfg.gibbs_step_delta_rel,
            )
            sweep_points = sorted(set(cfg.sweeps))

            def _sweep_subsample(*tensors):
                """Take a fixed random subsample (shared across model-seeds) of inputs.

                Sweeps are ~99% of cost and only feed a mean over the test set; estimating
                that mean on a fixed ~1k subsample keeps the statistic within ~±1.5%.
                """
                n = len(tensors[0])
                if cfg.sweep_subsample is None or cfg.sweep_subsample >= n:
                    return tensors
                rng = np.random.default_rng(cfg.sweep_subsample_seed)
                idx = torch.from_numpy(rng.permutation(n)[: cfg.sweep_subsample])
                return tuple(t[idx] for t in tensors)

            for eps_rel in tqdm(
                cfg.eps_rel, desc="sweep purify", unit="eps", dynamic_ncols=True
            ):
                try:
                    all_adv = torch.cat([b[0] for b in adv_examples_cache[eps_rel]])
                    all_labels = torch.cat([b[1] for b in adv_examples_cache[eps_rel]])
                    all_adv, all_labels = _sweep_subsample(all_adv, all_labels)

                    # Recompute misclassification + mean log p(x) on the SAME subsample so
                    # accuracy/recovery/log-px are internally consistent.
                    adv_preds = _batched_forward(
                        partial(class_probabilities, model), all_adv, cfg.eval_batch_size, device
                    ).argmax(dim=1)
                    misclassified = adv_preds != all_labels
                    mean_log_px_before = float(
                        _batched_forward(
                            partial(log_px, model), all_adv, cfg.eval_batch_size, device
                        ).mean()
                    )

                    snapshots = purifier.purify_snapshots(
                        model, all_adv, sweep_points, device
                    )
                except Exception as e:
                    logger.warning(f"Sweep purification failed (eps_rel={eps_rel}): {e}; skipping")
                    _recover_after_failure(model)
                    continue

                for n_sw, (x_purified, log_px_after) in snapshots.items():
                    try:
                        pur_preds = _batched_forward(
                            partial(class_probabilities, model), x_purified, cfg.eval_batch_size, device
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
                        sweep_purification_results[(eps_rel, n_sw)] = PurificationMetrics(
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
                            f"Sweep scoring failed (eps_rel={eps_rel}, n_sweeps={n_sw}): "
                            f"{e}; skipping"
                        )
                        _recover_after_failure(model)

            # Clean sweep purification
            all_clean = torch.cat([b for b, _ in clean_loader])
            all_clean_labels = torch.cat([lb for _, lb in clean_loader])
            all_clean, all_clean_labels = _sweep_subsample(all_clean, all_clean_labels)
            try:
                clean_mean_log_px_before = float(
                    _batched_forward(
                        partial(log_px, model), all_clean, cfg.eval_batch_size, device
                    ).mean()
                )
                clean_snapshots = purifier.purify_snapshots(
                    model, all_clean, sweep_points, device
                )
            except Exception as e:
                logger.warning(f"Clean sweep purification failed: {e}; skipping")
                _recover_after_failure(model)
                clean_snapshots = {}
                clean_mean_log_px_before = float("nan")

            for n_sw, (x_purified, log_px_after) in clean_snapshots.items():
                try:
                    pur_preds = _batched_forward(
                        partial(class_probabilities, model), x_purified, cfg.eval_batch_size, device
                    ).argmax(dim=1)
                    acc = (pur_preds == all_clean_labels).float().mean().item()
                    clean_sweep_purification_results[n_sw] = PurificationMetrics(
                        accuracy_after_purify=acc,
                        recovery_rate=float("nan"),
                        mean_log_px_before=clean_mean_log_px_before,
                        mean_log_px_after=float(log_px_after.mean()),
                        rejection_rate=0.0,
                    )
                    logger.info(f"  sweeps={n_sw}: clean_sweep_acc={acc:.4f}")
                except Exception as e:
                    logger.warning(
                        f"Clean sweep scoring failed (n_sweeps={n_sw}): {e}; skipping"
                    )
                    _recover_after_failure(model)

        return UQResults(
            clean_log_px=clean_log_px,
            clean_accuracy=clean_accuracy,
            thresholds=thresholds,
            clean_flagged=clean_flagged,
            adv_log_px=adv_log_px,
            adv_accuracies=adv_accuracies,
            detection_rates=detection_rates,
            err_rate_detected=err_rate_detected,
            err_rate_passed=err_rate_passed,
            purification_results=purification_results,
            sweep_purification_results=sweep_purification_results,
            clean_purification_results=clean_purification_results,
            clean_sweep_purification_results=clean_sweep_purification_results,
        )
