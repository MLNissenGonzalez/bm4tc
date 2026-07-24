"""Persistent SGLD sampling used for JEM training and generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import JEMMLP


@dataclass
class SGLDConfig:
    num_steps: int = 20
    step_size: float = 0.01
    noise_std: float = 0.005
    reinit_probability: float = 0.05
    buffer_size: int = 10000


class ReplayBuffer:
    def __init__(
        self,
        size: int,
        data_dim: int,
        input_range: tuple[float, float],
        seed: int = 0,
    ):
        self.size = int(size)
        self.data_dim = int(data_dim)
        self.input_range = tuple(input_range)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        lo, hi = self.input_range
        self.data = torch.empty(self.size, self.data_dim).uniform_(
            lo, hi, generator=generator
        )

    def initial(
        self,
        batch_size: int,
        reinit_probability: float,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        indices = torch.randint(0, self.size, (batch_size,))
        samples = self.data[indices].clone()
        reinit = torch.rand(batch_size) < reinit_probability
        if reinit.any():
            lo, hi = self.input_range
            samples[reinit].uniform_(lo, hi)
        return samples.to(device), indices

    def update(self, indices: torch.LongTensor, samples: torch.Tensor) -> None:
        self.data[indices.cpu()] = samples.detach().cpu()

    def state_dict(self) -> dict:
        return {"data": self.data}

    def load_state_dict(self, state: dict) -> None:
        data = state["data"]
        if data.shape != self.data.shape:
            raise ValueError(
                f"Replay buffer shape mismatch: {tuple(data.shape)} != {tuple(self.data.shape)}"
            )
        self.data.copy_(data)


class SGLDSampler:
    def __init__(self, cfg: SGLDConfig, buffer: ReplayBuffer):
        self.cfg = cfg
        self.buffer = buffer

    def refine(
        self,
        model: JEMMLP,
        initial: torch.Tensor,
        *,
        class_idx: int | torch.Tensor | None = None,
        num_steps: int | None = None,
        step_size: float | None = None,
        noise_std: float | None = None,
        center: torch.Tensor | None = None,
        radius: float | None = None,
    ) -> torch.Tensor:
        """Run SGLD on marginal or class-conditional JEM score.

        ``center`` and ``radius`` enable projected SGLD purification.
        """
        steps = self.cfg.num_steps if num_steps is None else int(num_steps)
        eta = self.cfg.step_size if step_size is None else float(step_size)
        sigma = self.cfg.noise_std if noise_std is None else float(noise_std)
        lo, hi = model.input_range
        x = initial.detach().clone()
        fixed_center = center.detach() if center is not None else None

        was_training = model.training
        model.eval()
        for _ in range(steps):
            x.requires_grad_(True)
            logits = model(x)
            if class_idx is None:
                score = torch.logsumexp(logits, dim=-1)
            elif isinstance(class_idx, int):
                score = logits[:, class_idx]
            else:
                labels = class_idx.to(x.device)
                score = logits.gather(1, labels[:, None]).squeeze(1)
            grad = torch.autograd.grad(score.sum(), x, only_inputs=True)[0]
            with torch.no_grad():
                x = x + 0.5 * eta * grad
                if sigma > 0:
                    x = x + sigma * torch.randn_like(x)
                if fixed_center is not None and radius is not None:
                    delta = (x - fixed_center).clamp(-radius, radius)
                    x = fixed_center + delta
                x.clamp_(lo, hi)
        model.train(was_training)
        return x.detach()

    def sample_training(
        self, model: JEMMLP, batch_size: int, device: torch.device | str
    ) -> torch.Tensor:
        initial, indices = self.buffer.initial(
            batch_size, self.cfg.reinit_probability, device
        )
        samples = self.refine(model, initial)
        self.buffer.update(indices, samples)
        return samples

    def sample_fresh(
        self,
        model: JEMMLP,
        n: int,
        device: torch.device | str,
        *,
        class_idx: int | None = None,
        num_steps: int = 500,
        batch_size: int = 256,
    ) -> torch.Tensor:
        lo, hi = model.input_range
        chunks = []
        for start in range(0, n, batch_size):
            size = min(batch_size, n - start)
            initial = torch.empty(size, model.data_dim, device=device).uniform_(lo, hi)
            chunks.append(
                self.refine(model, initial, class_idx=class_idx, num_steps=num_steps).cpu()
            )
        return torch.cat(chunks)
