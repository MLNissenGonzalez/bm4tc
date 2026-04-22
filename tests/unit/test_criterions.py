import pytest
import torch
import torch.nn as nn
from src.utils.criterions import (
    ClassificationNLL,
    ClassificationBrier,
    ClassificationSoftmaxNLL,
)

BATCH = 16
NUM_CLASSES = 3


@pytest.fixture
def probs():
    p = torch.rand(BATCH, NUM_CLASSES)
    return p / p.sum(dim=1, keepdim=True)


@pytest.fixture
def targets():
    return torch.randint(0, NUM_CLASSES, (BATCH,))


# ---- ClassificationNLL ----

def test_nll_correct_class_formula(probs, targets):
    loss_fn = ClassificationNLL()
    loss = loss_fn(probs, targets)
    expected = -torch.log(probs[torch.arange(BATCH), targets]).mean()
    assert loss.item() == pytest.approx(expected.item(), rel=1e-4)


def test_nll_uniform_distribution(targets):
    p = torch.full((BATCH, NUM_CLASSES), 1.0 / NUM_CLASSES)
    loss = ClassificationNLL()(p, targets)
    import math
    assert loss.item() == pytest.approx(math.log(NUM_CLASSES), rel=1e-4)


def test_nll_perfect_prediction():
    p = torch.eye(NUM_CLASSES)[:BATCH % NUM_CLASSES + 1]
    t = torch.arange(p.shape[0])
    loss = ClassificationNLL()(p, t)
    assert loss.item() < 1e-5


def test_nll_output_is_scalar(probs, targets):
    loss = ClassificationNLL()(probs, targets)
    assert loss.ndim == 0


def test_nll_gradient_flows(targets):
    p = torch.rand(BATCH, NUM_CLASSES, requires_grad=True)
    p_norm = p / p.sum(dim=1, keepdim=True)
    loss = ClassificationNLL()(p_norm, targets)
    loss.backward()
    assert p.grad is not None


def test_nll_eps_prevents_log_zero(targets):
    p = torch.zeros(BATCH, NUM_CLASSES)
    p[:, 0] = 1.0
    t = torch.zeros(BATCH, dtype=torch.long)
    loss = ClassificationNLL(eps=1e-12)(p, t)
    assert torch.isfinite(loss)


# ---- ClassificationBrier ----

def test_brier_range_0_to_2(probs, targets):
    loss = ClassificationBrier()(probs, targets)
    assert 0.0 <= loss.item() <= 2.0 + 1e-5


def test_brier_perfect_prediction():
    t = torch.arange(NUM_CLASSES)
    p = torch.eye(NUM_CLASSES)
    loss = ClassificationBrier()(p, t)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_brier_output_is_scalar(probs, targets):
    loss = ClassificationBrier()(probs, targets)
    assert loss.ndim == 0


def test_brier_gradient_flows(targets):
    p = torch.rand(BATCH, NUM_CLASSES, requires_grad=True)
    p_norm = p / p.sum(dim=1, keepdim=True)
    loss = ClassificationBrier()(p_norm, targets)
    loss.backward()
    assert p.grad is not None


# ---- ClassificationSoftmaxNLL ----

def test_softmax_nll_shape(targets):
    logits = torch.randn(BATCH, NUM_CLASSES)
    loss = ClassificationSoftmaxNLL()(logits, targets)
    assert loss.ndim == 0


def test_softmax_nll_gradient_flows(targets):
    logits = torch.randn(BATCH, NUM_CLASSES, requires_grad=True)
    loss = ClassificationSoftmaxNLL()(logits, targets)
    loss.backward()
    assert logits.grad is not None


def test_softmax_nll_decreases_on_correct(targets):
    logits_good = torch.zeros(BATCH, NUM_CLASSES)
    logits_good[torch.arange(BATCH), targets] = 10.0
    logits_bad = torch.zeros(BATCH, NUM_CLASSES)
    fn = ClassificationSoftmaxNLL()
    assert fn(logits_good, targets).item() < fn(logits_bad, targets).item()


def test_softmax_nll_equals_cross_entropy(targets):
    logits = torch.randn(BATCH, NUM_CLASSES)
    loss_ours = ClassificationSoftmaxNLL()(logits, targets)
    loss_ref = nn.CrossEntropyLoss()(logits, targets)
    assert loss_ours.item() == pytest.approx(loss_ref.item(), rel=1e-5)
