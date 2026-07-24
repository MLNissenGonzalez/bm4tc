import torch
from torch.nn import functional as F

from baselines.jem.model import (
    JEMMLP,
    JEMMLPConfig,
    mps_parameter_count,
    nearest_uniform_width,
)


def make_model():
    return JEMMLP(
        JEMMLPConfig(input_dim=144, hidden_dims=(347, 347), num_classes=10)
    )


def test_parameter_match_r20():
    model = make_model()
    target = mps_parameter_count(144, 3, 20, 10)
    assert target == 174_520
    assert model.count_parameters() == 174_551
    assert nearest_uniform_width(target) == (347, 31)


def test_alpha_zero_is_exact_cross_entropy():
    torch.manual_seed(0)
    model = make_model()
    x = torch.randn(8, 144)
    y = torch.randint(0, 10, (8,))
    mixed, terms = model.mixed_loss(x, y, negatives=None, alpha=0.0)
    expected = F.cross_entropy(model(x), y)
    assert torch.allclose(mixed, expected)
    assert torch.allclose(terms["dis_loss"], expected)


def test_alpha_endpoints_are_convex_interpolation():
    torch.manual_seed(0)
    model = make_model()
    x = torch.randn(8, 144)
    neg = torch.randn(8, 144)
    y = torch.randint(0, 10, (8,))
    alpha = 0.2
    mixed, terms = model.mixed_loss(x, y, neg, alpha)
    expected = (1 - alpha) * terms["dis_loss"] + alpha * terms["gen_loss"]
    assert torch.allclose(mixed, expected)


def test_probabilities_normalize():
    probs = make_model().class_probabilities(torch.randn(4, 144))
    assert probs.shape == (4, 10)
    assert torch.allclose(probs.sum(1), torch.ones(4), atol=1e-6)
