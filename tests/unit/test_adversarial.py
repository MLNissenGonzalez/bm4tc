"""Unit tests for AdversarialTrainer model selection (constrained by clean-acc floor).

`_update` is exercised in isolation via __new__ + manual attribute setup, so these
tests don't construct a DataHandler or PGD attack — they target the selection logic
(`acc_floor` gating, patience accounting, best-tensor bookkeeping) directly.
"""

import torch

from src.train.adversarial import AdversarialTrainer, AdversarialConfig


def _make_trainer(cbm, *, stop_crit="rob", acc_floor=None):
    """Build a trainer with just the attributes `_update` touches."""
    t = AdversarialTrainer.__new__(AdversarialTrainer)
    t.train_cfg = AdversarialConfig(stop_crit=stop_crit, acc_floor=acc_floor)
    t.cbm = cbm
    t.stopping_criterion_name = stop_crit
    t.best = {"dis_loss": float("inf"), "acc": 0.0, "rob": 0.0}
    t.best_tensors = [tt.cpu().clone().detach() for tt in cbm.tensors]
    t.patience_counter = 0
    t.best_epoch = 0
    t.epoch = 1
    return t


def test_floor_blocks_selection(cbm):
    """rob improved but clean acc < floor => no best update, patience increments."""
    t = _make_trainer(cbm, acc_floor=0.9)
    initial_tensors = t.best_tensors
    t.valid_perf = {"dis_loss": 0.5, "acc": 0.8, "rob": 0.5}

    t._update()

    assert t.best["rob"] == 0.0          # best untouched
    assert t.best_tensors is initial_tensors
    assert t.best_epoch == 0
    assert t.patience_counter == 1       # counted as non-improvement


def test_floor_allows_selection(cbm):
    """rob improved and clean acc >= floor => best updates, patience resets."""
    t = _make_trainer(cbm, acc_floor=0.9)
    t.patience_counter = 3
    t.valid_perf = {"dis_loss": 0.5, "acc": 0.95, "rob": 0.5}

    t._update()

    assert t.best["rob"] == 0.5
    assert t.best == dict(t.valid_perf)
    assert t.best_epoch == 1
    assert t.patience_counter == 0


def test_no_floor_ignores_acc(cbm):
    """acc_floor=None reproduces prior behavior: clean acc is irrelevant."""
    t = _make_trainer(cbm, acc_floor=None)
    t.valid_perf = {"dis_loss": 0.5, "acc": 0.1, "rob": 0.5}

    t._update()

    assert t.best["rob"] == 0.5          # selected despite low clean acc


def test_all_subfloor_keeps_initial_model(cbm):
    """If every epoch is sub-floor, best stays at init and tensors stay the start model."""
    t = _make_trainer(cbm, acc_floor=0.9)
    initial_tensors = t.best_tensors

    for epoch in range(1, 6):
        t.epoch = epoch
        t.valid_perf = {"dis_loss": 0.5, "acc": 0.5, "rob": 0.1 * epoch}
        t._update()

    assert t.best["rob"] == 0.0
    assert t.best_tensors is initial_tensors
    assert t.best_epoch == 0
    assert t.patience_counter == 5


def test_missing_rob_metric_is_skipped(cbm):
    """Non-rob epochs (no 'rob' in valid_perf) return early, before floor/patience logic."""
    t = _make_trainer(cbm, acc_floor=0.9)
    t.valid_perf = {"dis_loss": 0.5, "acc": 0.95}  # rob not evaluated this epoch

    t._update()

    assert t.patience_counter == 0       # untouched: nothing to compare against
    assert t.best["rob"] == 0.0


def test_mixed_loss_stop_crit_selection(cbm):
    """mixed_loss is a loss (lower-is-better): improves on decrease, not on increase."""
    t = _make_trainer(cbm, stop_crit="mixed_loss")
    t.valid_perf = {"dis_loss": 1.0, "gen_loss": 2.0, "mixed_loss": 1.5, "acc": 0.9}
    t._update()
    assert t.best["mixed_loss"] == 1.5
    assert t.best_epoch == 1
    assert t.patience_counter == 0

    t.epoch = 2  # worse (higher) mixed_loss => no improvement
    t.valid_perf = {"dis_loss": 1.0, "gen_loss": 3.0, "mixed_loss": 2.0, "acc": 0.9}
    t._update()
    assert t.best["mixed_loss"] == 1.5   # unchanged
    assert t.patience_counter == 1


def test_alpha_threads_into_training_objective():
    """cfg.alpha is passed to mixed_nll for both the adversarial and clean terms."""
    import torch
    from src.train.adversarial import AdversarialTrainer, AdversarialConfig

    captured = []
    param = torch.nn.Parameter(torch.zeros(1))

    class StubCBM:
        def train(self): pass
        def eval(self): pass
        def mixed_nll(self, data, labels, alpha):
            captured.append(alpha)
            return param.sum() + 1.0  # leaf-dependent so backward() works

    batch = (torch.zeros(2, 4), torch.zeros(2, dtype=torch.long))

    t = AdversarialTrainer.__new__(AdversarialTrainer)
    t.train_cfg = AdversarialConfig(alpha=0.5, clean_weight=0.3)
    t.cbm = StubCBM()
    t.device = torch.device("cpu")
    t.step = 0
    t.optimizer = torch.optim.SGD([param], lr=0.0)
    t.datahandler = type("DH", (), {"classification": {"train": [batch]}})()
    t._generate_adversarial = lambda data, labels, eps: data

    t._train_epoch(epsilon=0.1)

    # adv term + clean term (clean_weight > 0), both at the configured alpha
    assert captured == [0.5, 0.5]
