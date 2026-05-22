import pytest
import torch
import torch.nn as nn
from src.train.softmax import ClassificationSoftmaxNLL

BATCH = 16
NUM_CLASSES = 3


@pytest.fixture
def targets():
    return torch.randint(0, NUM_CLASSES, (BATCH,))


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
