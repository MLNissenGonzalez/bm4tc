# Likelihood-based purification for Born Machines

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
    radii: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])


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
                2.5 * radius / num_steps.
            random_start: Whether to start from random point within
                the radius ball.
        """
        self.norm = norm
        self.num_steps = num_steps
        self.step_size = step_size
        self.random_start = random_start

    def _project(self, perturbation: torch.Tensor, radius: float) -> torch.Tensor:
        """Project perturbation back into the Lp ball."""
        if self.norm == "inf":
            return perturbation.clamp(-radius, radius)
        elif isinstance(self.norm, int):
            norms = perturbation.norm(p=self.norm, dim=1, keepdim=True)
            scale = torch.clamp(norms / radius, min=1.0)
            return perturbation / scale
        else:
            raise ValueError(f"{self.norm=}, but expected int or 'inf'.")

    def _random_init(self, shape: torch.Size, radius: float, device: torch.device) -> torch.Tensor:
        """Initialize random perturbation within the Lp ball."""
        if self.norm == "inf":
            return (2 * torch.rand(shape, device=device) - 1) * radius
        elif isinstance(self.norm, int):
            delta = torch.randn(shape, device=device)
            delta = normalizing(delta, self.norm) * radius * torch.rand(shape[0], 1, device=device)
            return delta
        else:
            raise ValueError(f"{self.norm=}, but expected int or 'inf'.")

    def purify(
            self,
            born,
            data: torch.Tensor,
            radius: float,
            device: torch.device | str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Purify inputs by gradient descent on marginal NLL within an Lp ball.

        Moves the input towards higher likelihood regions of the learned
        distribution, staying within a radius of the original input.

        Args:
            born: ConditionalBornMachine instance (must have marginal_log_probability).
            data: Input tensor of shape (batch_size, data_dim).
            radius: Maximum perturbation radius.
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

        step_size = self.step_size if self.step_size is not None else 2.5 * radius / self.num_steps

        # Initialize perturbation
        if self.random_start:
            delta = self._random_init(data.shape, radius, device)
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
            delta = self._project(delta, radius)

        # Final purified samples, clamped to input range
        purified = (data + delta).clamp(input_range[0], input_range[1]).detach()

        # Compute final log p(x) for the purified samples
        with torch.no_grad():
            log_px = born.marginal_log_probability(purified)

        return purified, log_px


"""Gibbs-sampling purification for ConditionalBornMachine (class-marginalized)."""

import torch
from typing import Optional, Tuple

from src.model import draw_from_grid


class GibbsPurification:
    """Purify adversarial examples via class-marginalized Gibbs sampling.

    For each feature in turn, resample it from the conditional distribution
    p(x_i | x_{-i}), marginalizing over class label, by evaluating the CBM
    joint amplitudes over a discrete grid.  Multiple sweeps produce a sample
    more consistent with the model's learned distribution.

    Args:
        num_bins: Resolution of the discrete grid over the input range.
        gibbs_batch_size: Number of adversarial samples processed per batch.
            Controls memory: gibbs_batch_size × num_bins inputs per forward pass.
            At bs=8, bins=200: 1600 forward evals per feature per sweep.
        radius: Perturbation budget as a fraction of the input range size
            (b - a).  Each sweep restricts resampling of feature i to
            [x̄_i ± delta_abs] where delta_abs = radius * (hi - lo).
            The same budget applies regardless of n_sweeps.  Intervals are
            clamped to input_range.  If None, samples from the full input range.
    """

    def __init__(
        self,
        num_bins: int,
        gibbs_batch_size: int = 8,
        radius: Optional[float] = 0.1,
    ):
        self.num_bins = num_bins
        self.gibbs_batch_size = gibbs_batch_size
        self.radius = radius

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
            born: ConditionalBornMachine instance (must be prepared and in eval mode).
            x_adv: Adversarial inputs, shape (n_samples, data_dim).
            n_sweeps: Number of full sweeps over all features.
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
            born: ConditionalBornMachine instance (must be prepared and in eval mode).
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

        input_space = torch.linspace(lo, hi, self.num_bins, device=device)

        # Convert relative radius to absolute.
        delta_abs: Optional[float] = (
            self.radius * (hi - lo) if self.radius is not None else None
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

            # The candidate forward shape (bs*num_bins) is constant across all features
            # and sweeps of this batch and never changes core roles, so a single reset
            # here suffices — it re-traces only when bs changes (the final partial batch).
            born.reset()

            for s in range(1, max_sweeps + 1):
                # Snapshot at sweep start; restriction intervals are centred on these
                # values, not on within-sweep updated values.
                x_bar = x_cur.clone() if delta_abs is not None else None

                for k in tqdm(
                    range(data_dim),
                    desc=f"sweep {s}/{max_sweeps}",
                    unit="feat",
                    leave=False,
                    dynamic_ncols=True,
                ):
                    # Build (bs × num_bins) candidate inputs: x_cur with x[:, k] = grid.
                    x_cand = (
                        x_cur.unsqueeze(1)
                        .expand(bs, self.num_bins, -1)
                        .reshape(bs * self.num_bins, -1)
                        .clone()
                    )
                    x_cand[:, k] = input_space.repeat(bs)

                    with torch.no_grad():
                        # Sum |ψ(x,c)|² over classes → unnormalized p(x_i | x_{-i}).
                        abs_sq = born.abs_square(born.amplitudes(x_cand))  # (bs*bins, C)
                        p = abs_sq.sum(dim=-1).view(bs, self.num_bins)     # (bs, bins)

                    if delta_abs is not None:
                        lo_k = (x_bar[:, k] - delta_abs).clamp(lo, hi).unsqueeze(1)
                        hi_k = (x_bar[:, k] + delta_abs).clamp(lo, hi).unsqueeze(1)
                        mask = (input_space[None, :] >= lo_k) & (input_space[None, :] <= hi_k)
                        p = p * mask

                    x_cur[:, k] = draw_from_grid(p, input_space)

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

    device = torch.device("cpu")
    cbm = ConditionalBornMachine(
        cfg=CBMConfig(embedding="legendre", init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3)),
        data_dim=2, num_classes=2, device=device,
    )
    cbm.prepare(device=device)
    cbm.eval()
    cbm.cache_log_Z()

    x_adv = torch.zeros(4, 2)
    radius = 0.1

    purifier = LikelihoodPurification(norm="inf", num_steps=5)
    x_pur, log_px = purifier.purify(cbm, x_adv, radius=radius, device=device)
    assert x_pur.shape == x_adv.shape, "LikelihoodPurification: shape mismatch"
    assert log_px.isfinite().all(), "LikelihoodPurification: non-finite log_px"
    print(f"  LikelihoodPurification  shape={tuple(x_pur.shape)}  log_px_mean={log_px.mean().item():.4f}")

    gibbs = GibbsPurification(num_bins=20, gibbs_batch_size=4, radius=0.1)
    x_g, log_px_g = gibbs.purify(cbm, x_adv, n_sweeps=1, device=device)
    assert x_g.shape == x_adv.shape, "GibbsPurification: shape mismatch"
    print(f"  GibbsPurification       shape={tuple(x_g.shape)}  log_px_mean={log_px_g.mean().item():.4f}")

    print("purification.py smoke test passed.")
