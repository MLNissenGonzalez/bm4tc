"""SGLD purification for JEM (D72): the counterpart of the MPS's Gibbs purification.

One sweep is ``steps`` SGLD transitions on the JEM score, projected onto the L∞
ball of radius ``step_rel·(hi-lo)`` around the state at the start of the sweep;
the next sweep is re-centred on its result. Like Gibbs, the radius is a per-sweep
step, not a budget: the strength is the number of sweeps. Step size and noise are
fixed for every model, not the model's training-sampler settings.
"""
from typing import Dict, List, Tuple

import torch

from bm4tc.core.interface import log_px
from bm4tc.core.jem.sampler import SGLDConfig, SGLDSampler


class SGLDPurification:
    def __init__(self, step_rel: float, steps: int = 20, step_size: float = 0.01,
                 noise_std: float = 0.005, batch_size: int = 256):
        self.step_rel, self.batch_size = step_rel, batch_size
        # refine() reads only these; the buffer is not used for purification.
        self.sampler = SGLDSampler(SGLDConfig(num_steps=steps, step_size=step_size,
                                              noise_std=noise_std), buffer=None)

    def purify_snapshots(self, model, data: torch.Tensor, sweep_points: List[int],
                         device) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """``{k: (x after k sweeps, log p(x) of it)}`` on CPU, for k in sweep_points."""
        points = sorted({int(k) for k in sweep_points})
        if not points or points[0] < 1:
            raise ValueError("sweep_points must contain positive integers")
        lo, hi = model.input_range
        radius = self.step_rel * (hi - lo)
        model.to(device)
        snapshots: Dict[int, List[torch.Tensor]] = {k: [] for k in points}
        for i in range(0, len(data), self.batch_size):
            current = data[i:i + self.batch_size].to(device)
            for sweep in range(1, points[-1] + 1):
                current = self.sampler.refine(model, current, center=current, radius=radius)
                if sweep in snapshots:
                    snapshots[sweep].append(current.cpu())
        out = {}
        for k, chunks in snapshots.items():
            x = torch.cat(chunks)
            with torch.no_grad():
                scores = torch.cat([log_px(model, x[i:i + self.batch_size].to(device)).cpu()
                                    for i in range(0, len(x), self.batch_size)])
            out[k] = (x, scores)
        return out
