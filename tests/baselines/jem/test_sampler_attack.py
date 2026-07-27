import torch
from torch.utils.data import DataLoader, TensorDataset

from baselines.jem.attacks import (
    PGDConfig,
    pgd_classification,
    pgd_likelihood_aware,
)
from baselines.jem.model import JEMMLP, JEMMLPConfig
from baselines.jem.purification import (
    PurificationConfig,
    gradient_purify,
    sgld_purify,
    sgld_purify_snapshots,
)
from baselines.jem.sampler import ReplayBuffer, SGLDConfig, SGLDSampler
from baselines.jem.trainer import ValidationSamplerConfig, evaluate_jem


def make_small():
    return JEMMLP(
        JEMMLPConfig(input_dim=4, hidden_dims=(8, 8), num_classes=2)
    )


def test_sgld_stays_in_input_range():
    model = make_small()
    buffer = ReplayBuffer(32, 4, model.input_range, seed=0)
    sampler = SGLDSampler(
        SGLDConfig(num_steps=3, step_size=0.01, noise_std=0.01, buffer_size=32),
        buffer,
    )
    samples = sampler.sample_training(model, 8, "cpu")
    assert samples.shape == (8, 4)
    assert samples.min() >= -1
    assert samples.max() <= 1


def test_pgd_respects_domain_and_budget():
    model = make_small()
    x = torch.empty(8, 4).uniform_(-1, 1)
    y = torch.randint(0, 2, (8,))
    adv = pgd_classification(
        model, x, y, PGDConfig(epsilon=0.2, num_steps=3, restarts=2)
    )
    assert (adv - x).abs().max() <= 0.200001
    assert adv.min() >= -1
    assert adv.max() <= 1


def test_adaptive_pgd_respects_domain_and_budget():
    model = make_small()
    x = torch.empty(8, 4).uniform_(-1, 1)
    y = torch.randint(0, 2, (8,))
    adv = pgd_likelihood_aware(
        model, x, y, PGDConfig(epsilon=0.2, num_steps=3)
    )
    assert (adv - x).abs().max() <= 0.200001
    assert adv.min() >= -1
    assert adv.max() <= 1


def test_gradient_purification_respects_radius():
    model = make_small()
    x = torch.empty(8, 4).uniform_(-0.7, 0.7)
    purified = gradient_purify(
        model, x, PurificationConfig(radius=0.2, num_steps=3)
    )
    assert (purified - x).abs().max() <= 0.200001
    assert purified.min() >= -1
    assert purified.max() <= 1


def test_sgld_purification_respects_radius():
    model = make_small()
    buffer = ReplayBuffer(32, 4, model.input_range, seed=0)
    sampler = SGLDSampler(
        SGLDConfig(num_steps=3, step_size=0.01, noise_std=0.01, buffer_size=32),
        buffer,
    )
    x = torch.empty(8, 4).uniform_(-0.7, 0.7)
    purified = sgld_purify(
        model, sampler, x, PurificationConfig(radius=0.2, num_steps=3)
    )
    assert (purified - x).abs().max() <= 0.200001
    assert purified.min() >= -1
    assert purified.max() <= 1


def test_sgld_purification_recenters_between_sweeps():
    class LinearScore(torch.nn.Module):
        input_range = (-1.0, 1.0)

        def forward(self, data):
            return data.sum(dim=1, keepdim=True)

    model = LinearScore()
    buffer = ReplayBuffer(4, 2, model.input_range, seed=0)
    sampler = SGLDSampler(
        SGLDConfig(num_steps=1, step_size=2.0, noise_std=0.0, buffer_size=4),
        buffer,
    )
    initial = torch.zeros(1, 2)
    snapshots = sgld_purify_snapshots(
        model,
        sampler,
        initial,
        PurificationConfig(
            radius=0.2,
            num_steps=1,
            step_size=2.0,
            sgld_noise_std=0.0,
        ),
        (1, 3, 5),
    )

    assert torch.allclose(snapshots[1], torch.full_like(initial, 0.2))
    assert torch.allclose(snapshots[3], torch.full_like(initial, 0.6))
    assert torch.allclose(snapshots[5], torch.full_like(initial, 1.0))
    assert (snapshots[5] - initial).abs().max() > 0.2


def test_standardized_validation_mixed_loss_decomposition():
    model = make_small()
    x = torch.empty(16, 4).uniform_(-0.7, 0.7)
    y = torch.randint(0, 2, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    buffer = ReplayBuffer(32, 4, model.input_range, seed=123)
    sampler = SGLDSampler(
        SGLDConfig(num_steps=2, step_size=0.01, noise_std=0.0, buffer_size=32),
        buffer,
    )
    cfg = ValidationSamplerConfig(
        num_steps=2,
        step_size=0.01,
        noise_std=0.0,
        buffer_size=32,
        batch_size=8,
        num_batches=2,
    )
    metrics = evaluate_jem(model, loader, sampler, 0.5, cfg, "cpu", epoch=1)
    assert abs(
        metrics["mixed_loss"]
        - (metrics["dis_loss"] + 0.5 * metrics["px_cd_loss"])
    ) < 1e-8
    assert abs(
        metrics["joint_cd_loss"]
        - (metrics["dis_loss"] + metrics["px_cd_loss"])
    ) < 1e-8
