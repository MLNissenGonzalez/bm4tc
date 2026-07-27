import torch
from torch.nn import functional as F

from baselines.jem.model import (
    JEMMLP,
    JEMMLPConfig,
    mps_parameter_count,
    nearest_uniform_width,
)
from baselines.jem.device import resolve_device


def make_model():
    return JEMMLP(
        JEMMLPConfig(input_dim=144, hidden_dims=(550, 480), num_classes=10)
    )


def test_parameter_match_r20():
    model = make_model()
    complex_parameters = mps_parameter_count(144, 3, 20, 10)
    real_degrees_of_freedom = 2 * complex_parameters
    assert complex_parameters == 174_520
    assert model.count_parameters() == real_degrees_of_freedom == 349_040
    assert nearest_uniform_width(real_degrees_of_freedom) == (518, 102)


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


def test_auto_device_has_mps_training_policy():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto").type == expected
