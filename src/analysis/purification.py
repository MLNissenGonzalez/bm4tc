"""Likelihood-based purification for Born Machines.

Budget convention (see "Budget vocabulary" in CLAUDE.md):
    ``delta_rel``  authored fraction of the embedding domain width ``hi - lo``.
    ``delta_abs``  model-domain radius, ``delta_rel * (hi - lo)``. The purify methods
                   take ``delta_abs`` — conversion happens in the caller via
                   ``rel_to_abs``.

``delta`` is the *defense* budget (how far purification may move an input), as opposed
to ``eps``, the *attacker* budget in ``src/utils/evasion.py``.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

from tqdm.auto import tqdm


@dataclass
class PurificationConfig:
    norm: int | str = "inf"
    num_steps: int = 20
    step_size: Optional[float] = None
    random_start: bool = False
    # Relative: fractions of the input domain width. On legendre (width 2.0) these
    # are absolute radii 0.1 / 0.2 / 0.3.
    delta_rel: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15])


def normalizing(x: torch.FloatTensor, norm: int | str):
    """
    Normalize a tensor of shape (batch size, data dim)
    along the data dim (flattened).
    """
    if norm == "inf":
        normalized = x.sign()

    elif isinstance(norm, int):
        if norm < 1:
            raise ValueError("Only accept p >= 1.")
        x_norm = x.norm(p=norm, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=1e-12)
        normalized = x / x_norm

    else:
        raise ValueError(f"{norm=}, but expected to be int or 'inf'.")

    return normalized


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

    def _project(self, perturbation: torch.Tensor, delta_abs: float) -> torch.Tensor:
        """Project perturbation back into the Lp ball."""
        if self.norm == "inf":
            return perturbation.clamp(-delta_abs, delta_abs)
        elif isinstance(self.norm, int):
            norms = perturbation.norm(p=self.norm, dim=1, keepdim=True)
            scale = torch.clamp(norms / delta_abs, min=1.0)
            return perturbation / scale
        else:
            raise ValueError(f"{self.norm=}, but expected int or 'inf'.")

    def _random_init(self, shape: torch.Size, delta_abs: float, device: torch.device) -> torch.Tensor:
        """Initialize random perturbation within the Lp ball."""
        if self.norm == "inf":
            return (2 * torch.rand(shape, device=device) - 1) * delta_abs
        elif isinstance(self.norm, int):
            delta = torch.randn(shape, device=device)
            delta = normalizing(delta, self.norm) * delta_abs * torch.rand(shape[0], 1, device=device)
            return delta
        else:
            raise ValueError(f"{self.norm=}, but expected int or 'inf'.")

    def purify(
            self,
            born,
            data: torch.Tensor,
            delta_abs: float,
            device: torch.device | str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Purify inputs by gradient descent on marginal NLL within an Lp ball.

        Moves the input towards higher likelihood regions of the learned
        distribution, staying within ``delta_abs`` of the original input.

        Args:
            born: ConditionalBornMachine instance (must have marginal_log_probability).
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
        born.to(device)
        data = data.to(device).detach()
        input_range = born.input_range

        step_size = self.step_size if self.step_size is not None else 2.5 * delta_abs / self.num_steps

        # Initialize perturbation
        if self.random_start:
            delta = self._random_init(data.shape, delta_abs, device)
        else:
            delta = torch.zeros_like(data)

        # Iterative gradient descent on NLL (= gradient ascent on log p(x))
        for _ in range(self.num_steps):
            delta.requires_grad_(True)
            x_tilde = (data + delta).clamp(input_range[0], input_range[1])

            nll = -born.marginal_log_probability(x_tilde).mean()

            born.zero_grad()
            if delta.grad is not None:
                delta.grad.zero_()

            nll.backward()

            grad = delta.grad.detach()
            normalized_gradient = normalizing(grad, norm=self.norm)

            # Gradient descent on NLL (subtract, not add)
            delta = delta.detach() - step_size * normalized_gradient
            # Project back into Lp ball
            delta = self._project(delta, delta_abs)

        # Final purified samples, clamped to input range
        purified = (data + delta).clamp(input_range[0], input_range[1]).detach()

        # Compute final log p(x) for the purified samples
        with torch.no_grad():
            log_px = born.marginal_log_probability(purified)

        return purified, log_px


"""Gibbs-sampling purification for ConditionalBornMachine (class-marginalized)."""

import torch
from typing import Optional, Tuple

from src.model import draw_from_grid_log
from src.utils.embeddings import rel_to_abs


class GibbsPurification:
    """Purify adversarial examples via class-marginalized Gibbs sampling.

    For each feature in turn, resample it from the conditional distribution
    p(x_i | x_{-i}), marginalizing over class label, by evaluating the CBM
    joint amplitudes over a discrete grid.  Multiple sweeps produce a sample
    more consistent with the model's learned distribution.

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
        gibbs_batch_size: Number of adversarial samples processed per batch.
            Controls memory: gibbs_batch_size × num_bins inputs per forward pass.
            At bs=8, bins=200: 1600 forward evals per feature per sweep.
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
    ):
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
        data_dim = x_adv.shape[1]
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

        # Per-snapshot accumulation of purified batches (concatenated after the loop).
        snap_results: Dict[int, List[torch.Tensor]] = {s: [] for s in sweep_points}
        n_batches = (n_samples + self.gibbs_batch_size - 1) // self.gibbs_batch_size
        for batch_start in tqdm(
            range(0, n_samples, self.gibbs_batch_size),
            total=n_batches,
            desc="Gibbs",
            unit="batch",
            dynamic_ncols=True,
        ):
            batch = x_adv[batch_start : batch_start + self.gibbs_batch_size].to(device)
            bs = len(batch)
            x_cur = batch.clone()

            # Clear any data nodes left over from a previous batch. The candidate
            # forward runs through the eager accumulate path (log_amp_sq), which
            # resets around itself, so this is only hygiene between batches.
            born.reset()

            for s in range(1, max_sweeps + 1):
                # Snapshot at sweep start; the restriction window is centred on these
                # values, not on within-sweep updated values. Re-centring each sweep is
                # what makes the total budget k*delta rather than a single global ball.
                x_bar = x_cur.clone() if delta_abs is not None else None

                for k in tqdm(
                    range(data_dim),
                    desc=f"sweep {s}/{max_sweeps}",
                    unit="feat",
                    leave=False,
                    dynamic_ncols=True,
                ):
                    # Candidate values for feature k. Unrestricted: the shared global
                    # grid. Restricted: a per-sample grid spanning this sample's window
                    # [x̄_k ± delta] ∩ [lo, hi], so every candidate is admissible by
                    # construction and none of the num_bins resolution is wasted
                    # outside the window.
                    if delta_abs is None:
                        grid_k = input_space                              # (bins,)
                        flat_grid = input_space.repeat(bs)                # (bs*bins,)
                    else:
                        lo_k = (x_bar[:, k] - delta_abs).clamp(lo, hi)    # (bs,)
                        hi_k = (x_bar[:, k] + delta_abs).clamp(lo, hi)    # (bs,)
                        grid_k = lo_k[:, None] + (hi_k - lo_k)[:, None] * unit_grid[None, :]
                        flat_grid = grid_k.reshape(-1)                    # (bs*bins,)

                    # Build (bs × num_bins) candidate inputs: x_cur with x[:, k] = grid.
                    x_cand = (
                        x_cur.unsqueeze(1)
                        .expand(bs, self.num_bins, -1)
                        .reshape(bs * self.num_bins, -1)
                        .clone()
                    )
                    x_cand[:, k] = flat_grid

                    with torch.no_grad():
                        # Sum |ψ(x,c)|² over classes → unnormalized p(x_i | x_{-i}),
                        # computed entirely in log space. Each candidate is a full
                        # chain contraction over every site, so the linear |ψ|²
                        # overflows on long chains; log_amp_sq is the norm-
                        # accumulating (always overflow-safe) contraction, and
                        # logsumexp does the class sum without leaving log space.
                        las = born.log_amp_sq(x_cand)                      # (bs*bins, C)
                        log_p = torch.logsumexp(las, dim=-1).view(bs, self.num_bins)

                    # No masking needed: the grid *is* the window, so every bin is
                    # admissible and no row can be entirely -inf.
                    x_cur[:, k] = draw_from_grid_log(log_p, grid_k)

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
            ):
                chunk = x_purified[i:i + self.gibbs_batch_size].to(device)
                log_px_chunks.append(born.marginal_log_probability(chunk).cpu())
        return torch.cat(log_px_chunks)


if __name__ == "__main__":
    import sys
    import torch
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig
    from src.utils.embeddings import range_size_of

    device = torch.device("cpu")
    cbm = ConditionalBornMachine(
        cfg=CBMConfig(embedding="legendre", init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3)),
        data_dim=2, num_classes=2, device=device,
    )
    cbm.prepare(device=device)
    cbm.eval()
    cbm.cache_log_Z()

    x_adv = torch.zeros(4, 2)
    # Authored relative; converted once, as every caller must.
    delta_rel = 0.05
    delta_abs = rel_to_abs(delta_rel, range_size_of(cbm))  # legendre: 0.05 * 2.0 = 0.1

    purifier = LikelihoodPurification(norm="inf", num_steps=5)
    x_pur, log_px = purifier.purify(cbm, x_adv, delta_abs=delta_abs, device=device)
    assert x_pur.shape == x_adv.shape, "LikelihoodPurification: shape mismatch"
    assert log_px.isfinite().all(), "LikelihoodPurification: non-finite log_px"
    assert (x_pur - x_adv).abs().max().item() <= delta_abs + 1e-6, "purify: left the ball"
    print(f"  LikelihoodPurification  shape={tuple(x_pur.shape)}  log_px_mean={log_px.mean().item():.4f}")

    gibbs = GibbsPurification(num_bins=20, gibbs_batch_size=4, step_delta_rel=delta_rel)
    x_g, log_px_g = gibbs.purify(cbm, x_adv, n_sweeps=1, device=device)
    assert x_g.shape == x_adv.shape, "GibbsPurification: shape mismatch"
    print(f"  GibbsPurification       shape={tuple(x_g.shape)}  log_px_mean={log_px_g.mean().item():.4f}")

    print("purification.py smoke test passed.")
