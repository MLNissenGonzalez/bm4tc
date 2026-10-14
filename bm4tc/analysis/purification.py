"""Likelihood-based purification for Born Machines.

Budget convention (see "Budget vocabulary" in CLAUDE.md):
    ``delta_rel``  authored fraction of the embedding domain width ``hi - lo``.
    ``delta_abs``  model-domain radius, ``delta_rel * (hi - lo)``. The purify methods
                   take ``delta_abs`` — conversion happens in the caller via
                   ``rel_to_abs``.

``delta`` is the *defense* budget (how far purification may move an input), as opposed
to ``eps``, the *attacker* budget in ``bm4tc/core/attacks.py``.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

from tqdm.auto import tqdm

from bm4tc.core.attacks import normalizing, project, random_in_ball
from bm4tc.core.graphs import Graphs
from bm4tc.core.interface import log_px


@dataclass
class PurificationConfig:
    norm: int | str = "inf"
    num_steps: int = 20
    step_size: Optional[float] = None
    random_start: bool = False
    # Relative: fractions of the input domain width. On legendre (width 2.0) these
    # are absolute radii 0.1 / 0.2 / 0.3.
    delta_rel: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15])


class LikelihoodPurification:
    """
    Purify adversarial examples by maximizing marginal log-likelihood
    within a perturbation ball around the input.

    Uses gradient descent on the negative marginal log-probability
    (i.e., ascent on log p(x)) with projection back onto the Lp ball,
    analogous to PGD but in reverse direction.
    """

    def __init__(
            self,
            norm: int | str = "inf",
            num_steps: int = 20,
            step_size: float | None = None,
            random_start: bool = False,
    ):
        """
        Initialize purification.

        Args:
            norm: Lp norm for perturbation ball ("inf" or int >= 1).
            num_steps: Number of gradient descent iterations.
            step_size: Step size per iteration. If None, defaults to
                2.5 * delta_abs / num_steps.
            random_start: Whether to start from random point within
                the ball.
        """
        self.norm = norm
        self.num_steps = num_steps
        self.step_size = step_size
        self.random_start = random_start

    def purify(
            self,
            model,
            data: torch.Tensor,
            delta_abs: float,
            device: torch.device | str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Purify inputs by gradient descent on marginal NLL within an Lp ball.

        Moves the input towards higher likelihood regions of the learned
        distribution, staying within ``delta_abs`` of the original input.

        Args:
            model: a GenerativeClassifier (bm4tc.core.interface).
            data: Input tensor of shape (batch_size, data_dim).
            delta_abs: Maximum perturbation radius, absolute in model-domain units
                (not a fraction — convert with ``rel_to_abs`` in the caller).
            device: Torch device.

        Returns:
            Tuple of:
                - purified: Purified inputs, shape (batch_size, data_dim).
                - log_px: Marginal log-probabilities of purified inputs,
                  shape (batch_size,).
        """
        model.to(device)
        return self.graphed(model, Graphs(enabled=False))(data.to(device).detach(), delta_abs)

    def _step(self, model, data: torch.Tensor, delta: torch.Tensor, delta_abs) -> torch.Tensor:
        """One descent step on the marginal NLL from ``delta``, projected."""
        step_size = self.step_size if self.step_size is not None else 2.5 * delta_abs / self.num_steps
        lo, hi = model.input_range
        delta = delta.detach().requires_grad_(True)
        x_tilde = (data + delta).clamp(lo, hi)
        nll = -log_px(model, x_tilde).mean()
        # The input gradient only, as in PGD: no parameter gradients are
        # written, so the step can be captured as a CUDA graph (D92).
        (grad,) = torch.autograd.grad(nll, delta)
        # Gradient descent on NLL (subtract, not add), back into the Lp ball
        delta = delta.detach() - step_size * normalizing(grad, norm=self.norm)
        return project(delta, self.norm, delta_abs)

    def _finish(self, model, data: torch.Tensor, delta: torch.Tensor):
        """The purified inputs, clamped to the input range, and their log p(x)."""
        lo, hi = model.input_range
        purified = (data + delta).clamp(lo, hi).detach()
        with torch.no_grad():
            return purified, log_px(model, purified)

    def graphed(self, model, graphs: Graphs) -> "GraphedPurification":
        """Purification of ``model``'s inputs with each step, and the final
        log p(x), replayed from CUDA graphs of ``graphs``; see
        :class:`GraphedPurification`."""
        return GraphedPurification(self, model, graphs)


class GraphedPurification:
    """:class:`LikelihoodPurification` on one model, its step and its final
    log p(x) each replayed from a CUDA graph; the loop stays in Python, as in
    :class:`bm4tc.core.attacks.GraphedPGD` (D105). Disabled graphs make this the
    eager purification, number for number."""

    def __init__(self, purifier: LikelihoodPurification, model, graphs: Graphs):
        self.purifier = purifier
        self.model = model
        self.step = graphs.wrap(
            lambda data, delta, delta_abs: purifier._step(model, data, delta, delta_abs))
        self.finish = graphs.wrap(
            lambda data, delta: purifier._finish(model, data, delta))

    def __call__(self, data: torch.Tensor, delta_abs) -> Tuple[torch.Tensor, torch.Tensor]:
        """(purified inputs, their log p(x)) for ``data`` on the model's device;
        ``delta_abs`` a float, or a device scalar when the graphs are enabled."""
        p = self.purifier
        if p.random_start:
            delta = random_in_ball(data.shape, p.norm, delta_abs, data.device)
        else:
            delta = torch.zeros_like(data)
        for _ in range(p.num_steps):
            delta = self.step(data, delta, delta_abs)
        return self.finish(data, delta)


"""Gibbs-sampling purification for ConditionalBornMachine (class-marginalized)."""

from bm4tc.core.embeddings import rel_to_abs


class GibbsPurification:
    """Purify adversarial examples via class-marginalized Gibbs sampling.

    For each feature in turn, resample it from the conditional distribution
    p(x_i | x_{-i}), marginalizing over class label, on a discrete grid of
    candidate values.  One sweep is one ``born.sample(in_condition=x)``
    (tensorkrowch's sampler, D100, D101): O(features) per sweep, each candidate
    a quadratic form on the site's reduced density matrix.  Multiple sweeps
    produce a sample more consistent with the model's learned distribution.

    **Attack-radius agnostic.**  ``step_delta_rel`` is a *per-sweep* L∞ step, not a
    global perturbation budget: the restriction window is re-centred on the
    sweep-start snapshot at the beginning of every sweep, so after ``k`` sweeps a
    coordinate can have travelled up to ``k · step_delta_rel · (hi - lo)`` from where
    it started.  Purification strength is therefore controlled by ``n_sweeps``
    alone, which means the defense never has to be told the attacker's budget.
    Keep ``step_delta_rel`` small (a local move) and vary ``n_sweeps``.

    Args:
        num_bins: Number of candidate values evaluated per feature.  With a
            ``step_delta_rel`` these all fall *inside* the window (see below), so
            this is the resolution of the window, not of the full input range.
        progress: Show progress bars.
        gibbs_batch_size: Number of adversarial samples swept together.  Bounds
            memory (a few MiB per sample at MNIST 12×12, r40); time per sweep
            hardly grows with it.
        step_delta_rel: Per-sweep L∞ step as a fraction of the input range size
            (hi - lo).  Feature i is resampled on a grid spanning
            [x̄_i ± delta_abs] ∩ input_range, where delta_abs =
            step_delta_rel * (hi - lo) and x̄ is the sweep-start snapshot.
            If None, samples from the full input range (unrestricted).

    Note:
        With a ``step_delta_rel`` the candidate grid is built *locally*, per sample:
        ``linspace(x̄_i - delta, x̄_i + delta, num_bins)`` clamped to the input
        range.  This spends every candidate evaluation inside the window (a global
        grid masked down to a 10%-wide window would discard ~90% of them) and gives
        full ``num_bins`` resolution there.  The price is that the proposal support
        moves each sweep, so this is a local Metropolis-within-Gibbs-style move
        rather than exact Gibbs on one fixed discretization — the k→∞ limit is not
        the model's discretized marginal.  That is the intended behaviour for a
        local purification walk; ``step_delta_rel=None`` is the fixed-grid path.
    """

    def __init__(
        self,
        num_bins: int,
        gibbs_batch_size: int = 8,
        step_delta_rel: Optional[float] = 0.1,
        progress: bool = True,
    ):
        self.progress = progress   # off inside an attack loop (bm4tc.analysis.adaptive)
        self.num_bins = num_bins
        self.gibbs_batch_size = gibbs_batch_size
        self.step_delta_rel = step_delta_rel

    def purify(
        self,
        born,
        x_adv: torch.Tensor,
        n_sweeps: int,
        device: torch.device | str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Purify adversarial examples using Gibbs sampling.

        Thin wrapper over :meth:`purify_snapshots` for the single-sweep-count case.

        Args:
            born: ConditionalBornMachine instance.
            x_adv: Adversarial inputs, shape (n_samples, data_dim).
            n_sweeps: Number of full sweeps over all features.  This, not
                ``step_delta_rel``, is the purification-strength knob.
            device: Torch device.

        Returns:
            Tuple of:
                - x_purified: Purified inputs on CPU, shape (n_samples, data_dim).
                - log_px_after: Marginal log p(x) of purified inputs on CPU,
                  shape (n_samples,).
        """
        return self.purify_snapshots(born, x_adv, [n_sweeps], device)[n_sweeps]

    def purify_snapshots(
        self,
        born,
        x_adv: torch.Tensor,
        sweep_points: List[int],
        device: torch.device | str,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Purify via Gibbs sampling, snapshotting state at several sweep counts.

        Runs ``max(sweep_points)`` sweeps once and records the purified state after
        each requested sweep count. A 5-sweep run thus yields the 1-, 3-, and
        5-sweep results in a single pass instead of three independent reruns — the
        snapshot at sweep ``s`` is bit-identical to a dedicated ``s``-sweep run under
        the same RNG state.

        Args:
            born: ConditionalBornMachine instance.
            x_adv: Adversarial inputs, shape (n_samples, data_dim).
            sweep_points: Sweep counts to snapshot (e.g. ``[1, 3, 5]``).
            device: Torch device.

        Returns:
            Dict mapping each sweep count to ``(x_purified, log_px_after)``, both on CPU.
        """
        born.to(device)
        born.eval()

        sweep_points = sorted({int(s) for s in sweep_points})
        max_sweeps = max(sweep_points)

        n_samples = len(x_adv)
        lo, hi = born.input_range

        # Unrestricted path: one fixed grid over the whole input range, shared by
        # every sample. Restricted path builds its grid per sample, per feature.
        input_space = (
            torch.linspace(lo, hi, self.num_bins, device=device)
            if self.step_delta_rel is None
            else None
        )
        # Interpolation weights for the local windows, (num_bins,) in [0, 1].
        unit_grid = (
            torch.linspace(0.0, 1.0, self.num_bins, device=device)
            if self.step_delta_rel is not None
            else None
        )

        # The rel -> abs boundary for this defense: everything below is model-domain.
        delta_abs: Optional[float] = (
            rel_to_abs(self.step_delta_rel, hi - lo)
            if self.step_delta_rel is not None
            else None
        )

        # The candidates' embedding, (*domain.shape, phys_dim) in the model's dtype.
        def embed(values: torch.Tensor) -> torch.Tensor:
            return born.embedding(values).to(born.dtype).reshape(*values.shape, -1)

        # Per-snapshot accumulation of purified batches (concatenated after the loop).
        snap_results: Dict[int, List[torch.Tensor]] = {s: [] for s in sweep_points}
        n_batches = (n_samples + self.gibbs_batch_size - 1) // self.gibbs_batch_size
        for batch_start in tqdm(
            range(0, n_samples, self.gibbs_batch_size),
            total=n_batches,
            desc="Gibbs",
            unit="batch",
            dynamic_ncols=True,
            disable=not self.progress,
        ):
            x_cur = x_adv[batch_start : batch_start + self.gibbs_batch_size].to(device)

            # No reset: every sweep and batch reuses the nodes of the first
            # (SamplingMPS, D103); the log p(x) forward below removes them.
            for s in range(1, max_sweeps + 1):
                # Candidate values per feature. Unrestricted: the shared global
                # grid. Restricted: per sample, a grid spanning its window
                # [x̄_k ± delta] ∩ [lo, hi] around the sweep-start value x̄, so every
                # candidate is admissible and none of the num_bins resolution is
                # wasted outside the window. Re-centring each sweep is what makes
                # the total budget k*delta rather than a single global ball.
                if delta_abs is None:
                    domain = input_space                                # (bins,)
                else:
                    w_lo = (x_cur - delta_abs).clamp(lo, hi)            # (bs, D)
                    w_hi = (x_cur + delta_abs).clamp(lo, hi)
                    windows = w_lo[..., None] + (w_hi - w_lo)[..., None] * unit_grid
                    domain = list(windows.unbind(1))                    # D × (bs, bins)

                # One sweep: feature k is drawn given the new values left of it
                # and x_cur right of it, the class summed out.
                x_cur = born.sample(in_condition=born.embed(x_cur).to(born.dtype),
                                    domain=domain, embedding=embed,
                                    renormalize=True)

                if s in snap_results:
                    snap_results[s].append(x_cur.cpu().clone())

        out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for s in sweep_points:
            x_purified = torch.cat(snap_results[s], dim=0)
            out[s] = (x_purified, self._chunked_log_px(born, x_purified, device))
        return out

    def _chunked_log_px(
        self,
        born,
        x_purified: torch.Tensor,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Chunk the final log p(x) forward by gibbs_batch_size.

        Keeps this within the same memory budget as the sweep (a single forward over
        all samples would OOM on large inputs, e.g. MNIST's full test split).
        """
        born.reset()
        log_px_chunks = []
        n_chunks = (len(x_purified) + self.gibbs_batch_size - 1) // self.gibbs_batch_size
        with torch.no_grad():
            for i in tqdm(
                range(0, len(x_purified), self.gibbs_batch_size),
                total=n_chunks,
                desc="Gibbs log p(x)",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
                disable=not self.progress,
            ):
                chunk = x_purified[i:i + self.gibbs_batch_size].to(device)
                log_px_chunks.append(born.marginal_log_probability(chunk).cpu())
        return torch.cat(log_px_chunks)
