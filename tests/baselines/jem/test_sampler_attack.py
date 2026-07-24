import torch

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
)
from baselines.jem.sampler import ReplayBuffer, SGLDConfig, SGLDSampler


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
