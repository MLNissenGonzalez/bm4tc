"""Unit tests for AdversarialTrainer model selection (constrained by clean-acc floor).

`_update` is exercised in isolation via __new__ + manual attribute setup, so these
tests don't construct a DataHandler or PGD attack — they target the selection logic
(`acc_floor` gating, patience accounting, best-tensor bookkeeping) directly.
"""

import math
import torch
from unittest.mock import patch
from torch.utils.data import DataLoader, TensorDataset

from src.train.adversarial import AdversarialTrainer, AdversarialConfig
from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig
from src.utils.train import (
    NormControlConfig, NormRegularizer, NormTracker, eval_at, eval_metrics,
    eval_rob, eval_split,
)


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
    t._nc = t.train_cfg.norm_control  # off by default → no-op
    t.norm_regularizer = None

    t._train_epoch(eps_abs=0.1)

    # adv term + clean term (clean_weight > 0), both at the configured alpha
    assert captured == [0.5, 0.5]


# ── Norm control ────────────────────────────────────────────────────────────

def _tiny_cbm():
    cfg = CBMConfig(
        embedding="fourier",
        init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3),
    )
    return ConditionalBornMachine(cfg=cfg, data_dim=2, num_classes=2)


class _FakeDataHandler:
    """Minimal DataHandler substitute that skips file I/O."""
    data_dim = 2

    def __init__(self, n=16, batch_size=4):
        ds = TensorDataset(torch.rand(n, 2), torch.randint(0, 2, (n,)))
        loader = DataLoader(ds, batch_size=batch_size)
        self.classification = {"train": loader, "valid": loader}

    def get_classification_loaders(self, batch_size=4):
        pass


def _bare_epoch_trainer(cbm, dh, train_cfg, *, norm_regularizer=None, log_target=0.0):
    """Trainer wired with just the attributes `_train_epoch` touches.

    Bypasses `_init_attack` (no PGD) by stubbing `_generate_adversarial` to the
    identity, so these tests exercise the norm-control plumbing in isolation.
    """
    t = AdversarialTrainer.__new__(AdversarialTrainer)
    t.train_cfg = train_cfg
    t.cbm = cbm
    t.datahandler = dh
    t.device = torch.device("cpu")
    t.step = 0
    t._nc = train_cfg.norm_control
    t.norm_regularizer = norm_regularizer
    t._nc_log_target = log_target
    t.optimizer = torch.optim.Adam(cbm.parameters(), lr=1e-3)
    t._generate_adversarial = lambda data, labels, eps: data
    return t


def test_adversarial_config_norm_control_defaults():
    """AT norm control default: hard renormalization off, soft log_Z penalty on."""
    nc = AdversarialConfig().norm_control
    assert nc.hard_every == 0
    assert nc.soft_strength == 0.1


def test_norm_control_off_skips_renormalize():
    cbm = _tiny_cbm()
    cbm.prepare(device=torch.device("cpu"))
    dh = _FakeDataHandler()
    cfg = AdversarialConfig(
        norm_control=NormControlConfig(hard_every=0, soft_strength=0.0),
    )
    t = _bare_epoch_trainer(cbm, dh, cfg)

    with patch.object(cbm, "renormalize_", wraps=cbm.renormalize_) as m_renorm:
        t._train_epoch(eps_abs=0.1)

    m_renorm.assert_not_called()
    assert t._train_reg == 0.0


def test_hard_renorm_called_every_step():
    cbm = _tiny_cbm()
    cbm.prepare(device=torch.device("cpu"))
    dh = _FakeDataHandler(n=16, batch_size=4)  # 4 batches → 4 steps
    cfg = AdversarialConfig(
        norm_control=NormControlConfig(hard_every=1, soft_strength=0.0, log_target=0.0),
    )
    t = _bare_epoch_trainer(cbm, dh, cfg, log_target=0.0)

    with patch.object(cbm, "renormalize_", wraps=cbm.renormalize_) as m_renorm:
        t._train_epoch(eps_abs=0.1)

    assert m_renorm.call_count == 4


def test_soft_norm_control_multistep_backward():
    """alpha=0 + soft norm control must train across multiple steps.

    The NormRegularizer reads the with-grad log_Z via recompute=False; without
    per-step cache invalidation the second step backwards through the first
    step's freed graph. Mirrors the NLLTrainer regression guard.
    """
    cbm = _tiny_cbm()
    cbm.prepare(device=torch.device("cpu"))
    dh = _FakeDataHandler(n=16, batch_size=4)
    cfg = AdversarialConfig(
        alpha=0.0,
        norm_control=NormControlConfig(hard_every=0, soft_strength=1.0, log_target=0.0),
    )
    t = _bare_epoch_trainer(
        cbm, dh, cfg,
        norm_regularizer=NormRegularizer(strength=1.0, log_target=0.0),
        log_target=0.0,
    )

    # Must complete without "Trying to backward through the graph a second time".
    t._train_epoch(eps_abs=0.1)

    assert t.step >= 2
    assert t._train_reg > 0.0
    assert cbm._log_Z_cache is None  # invalidated after the final step


def test_norm_stats_populated_after_epoch():
    """norm/* training stats are tracked every epoch by default (no debug flag)."""
    cbm = _tiny_cbm()
    cbm.prepare(device=torch.device("cpu"))
    dh = _FakeDataHandler()
    cfg = AdversarialConfig(
        norm_control=NormControlConfig(hard_every=0, soft_strength=0.0),  # off
    )
    t = _bare_epoch_trainer(cbm, dh, cfg)

    t._train_epoch(eps_abs=0.1)

    # alpha=0 / no soft → log_Z via the end-of-epoch snapshot; amp from the adv forward.
    for key in ("norm/log_Z_mean", "norm/log_Z_max", "norm/log_amp_sq_mean"):
        assert key in t._norm_stats, f"missing {key}"
    assert math.isfinite(t._norm_stats["norm/log_Z_mean"])
    assert math.isfinite(t._norm_stats["norm/log_amp_sq_mean"])


# ── gen_on_clean: split training objective ──────────────────────────────────

# Distinct per-batch values so a test can tell the adversarial batch (tag 1.0)
# from the clean one (tag 0.0) purely from the returned loss.
_L_DIS = {0.0: 2.0, 1.0: 3.0}
_L_GEN = {0.0: 7.0, 1.0: 8.0}


class _DecompStubCBM:
    """Stub whose mixed_nll decomposes exactly like the real one.

    ``mixed_nll(x, y, a) = (1-a)*L_dis(x) + a*L_gen(x)`` — the identity the split
    objective relies on (src/model.py mixed_nll) — with L_dis/L_gen keyed off a
    per-batch tag, so the weighting can be checked in closed form.
    """

    def __init__(self):
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def train(self): pass
    def eval(self): pass

    def mixed_nll(self, data, labels, alpha):
        tag = float(data[0, 0])
        self.calls.append((tag, alpha))
        # param keeps the result a graph leaf so backward() works in _train_epoch
        return self.param.sum() + (1 - alpha) * _L_DIS[tag] + alpha * _L_GEN[tag]


def _split_trainer(cbm, cfg):
    """Trainer wired with just the attributes `_split_nll` / `_train_epoch` touch."""
    from src.train.adversarial import AdversarialTrainer

    t = AdversarialTrainer.__new__(AdversarialTrainer)
    t.train_cfg = cfg
    t.cbm = cbm
    t.device = torch.device("cpu")
    t.step = 0
    t._nc = cfg.norm_control
    t.norm_regularizer = None
    # tag 0.0 = clean batch, tag 1.0 = adversarial batch
    clean = (torch.zeros(4, 2), torch.zeros(4, dtype=torch.long))
    t.datahandler = type("DH", (), {"classification": {"train": [clean]}})()
    t._generate_adversarial = lambda data, labels, eps: torch.ones_like(data)
    t.optimizer = torch.optim.SGD([cbm.param], lr=0.0)
    return t


def _naive_split_loss(alpha, cw):
    """The three-term form the two-call implementation must reproduce."""
    return (1 - alpha) * ((1 - cw) * _L_DIS[1.0] + cw * _L_DIS[0.0]) + alpha * _L_GEN[0.0]


def test_gen_on_clean_defaults_off():
    """Default must reproduce today's objective for the existing AT configs."""
    assert AdversarialConfig().gen_on_clean is False


def test_split_objective_matches_naive_three_term_form():
    """Two rescaled mixed_nll calls == (1-a)[(1-cw)L_dis(adv)+cw L_dis(clean)] + a L_gen(clean)."""
    for alpha, cw in [(0.5, 0.3), (0.1, 0.0), (0.9, 0.7), (0.25, 1.0), (0.0, 0.4)]:
        cbm = _DecompStubCBM()
        cfg = AdversarialConfig(alpha=alpha, clean_weight=cw, gen_on_clean=True)
        t = _split_trainer(cbm, cfg)

        loss = t._split_nll(
            torch.ones(4, 2), torch.zeros(4, 2), torch.zeros(4, dtype=torch.long),
            NormTracker(),
        )

        assert abs(loss.item() - _naive_split_loss(alpha, cw)) < 1e-6, (alpha, cw)


def test_split_uses_two_forwards_with_rescaled_alpha():
    """alpha=0.5, cw=0.3 → mixed_nll(adv, 0) and mixed_nll(clean, alpha/s), nothing more."""
    alpha, cw = 0.5, 0.3
    cbm = _DecompStubCBM()
    t = _split_trainer(cbm, AdversarialConfig(alpha=alpha, clean_weight=cw, gen_on_clean=True))

    t._split_nll(torch.ones(4, 2), torch.zeros(4, 2),
                 torch.zeros(4, dtype=torch.long), NormTracker())

    s = (1 - alpha) * cw + alpha
    assert len(cbm.calls) == 2, cbm.calls
    assert cbm.calls[0] == (1.0, 0.0)           # adversarial batch, discriminative
    assert cbm.calls[1][0] == 0.0               # clean batch
    assert abs(cbm.calls[1][1] - alpha / s) < 1e-9


def test_split_at_alpha0_is_a_noop():
    """alpha=0: the split cannot differ from unsplit — same single call, same loss."""
    cw = 0.0
    split_cbm = _DecompStubCBM()
    t_split = _split_trainer(split_cbm, AdversarialConfig(
        alpha=0.0, clean_weight=cw, gen_on_clean=True))
    loss_split = t_split._split_nll(
        torch.ones(4, 2), torch.zeros(4, 2), torch.zeros(4, dtype=torch.long), NormTracker())

    assert split_cbm.calls == [(1.0, 0.0)]
    assert abs(loss_split.item() - _L_DIS[1.0]) < 1e-6


def test_split_at_alpha1_drops_the_adversarial_term():
    """alpha=1: only the clean generative call survives (attacks are discarded)."""
    cbm = _DecompStubCBM()
    t = _split_trainer(cbm, AdversarialConfig(alpha=1.0, clean_weight=0.3, gen_on_clean=True))

    loss = t._split_nll(torch.ones(4, 2), torch.zeros(4, 2),
                        torch.zeros(4, dtype=torch.long), NormTracker())

    assert cbm.calls == [(0.0, 1.0)]
    assert abs(loss.item() - _L_GEN[0.0]) < 1e-6


def test_split_train_epoch_routes_through_split_nll():
    """_train_epoch honours the flag end to end (2 calls/step, adv first)."""
    cbm = _DecompStubCBM()
    t = _split_trainer(cbm, AdversarialConfig(
        alpha=0.5, clean_weight=0.3, gen_on_clean=True,
        norm_control=NormControlConfig(hard_every=0, soft_strength=0.0),
    ))

    t._train_epoch(eps_abs=0.1)

    assert len(cbm.calls) == 2  # one training step in the stub loader
    assert cbm.calls[0][1] == 0.0
    assert abs(t._train_nll - _naive_split_loss(0.5, 0.3)) < 1e-6


# ── gen_on_clean: split validation (eval_split) ─────────────────────────────

class _ShiftAttack:
    """Deterministic stand-in for PGD; records which samples it was handed."""

    def __init__(self, shift=0.05):
        self.shift = shift
        self.seen = []

    def generate(self, born, naturals, labels, eps_abs, device):
        self.seen.append(naturals.detach().clone())
        return (naturals + self.shift).detach()


def _valid_loader(n=20, batch_size=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.rand(n, 2, generator=g),
                       torch.randint(0, 2, (n,), generator=g))
    # shuffle=False mirrors DataHandler's non-train splits: positional indices stable
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def test_eval_split_clean_metrics_match_eval_metrics():
    """acc/dis_loss/gen_loss are clean and over the full set, as in eval_metrics."""
    cbm = _tiny_cbm()
    cbm.prepare(device=torch.device("cpu"))
    # Evenly-dividing batches: eval_metrics averages per-batch means while
    # eval_split averages per sample, so the two only coincide exactly when every
    # batch is the same size.
    loader = _valid_loader(n=20, batch_size=5)

    out = eval_split(cbm, loader, _ShiftAttack(), 0.1, torch.device("cpu"),
                     alpha=0.5, clean_weight=0.3, adv_indices={0, 1, 2})
    dis_loss, acc, gen_loss = eval_metrics(cbm, loader, torch.device("cpu"))

    assert abs(out["acc"] - acc) < 1e-9
    assert abs(out["dis_loss"] - dis_loss) < 1e-5
    assert abs(out["gen_loss"] - gen_loss) < 1e-5


def test_eval_split_attacks_only_the_given_subset():
    """Exactly the adv_indices samples are handed to the attack, once each."""
    cbm = _tiny_cbm()
    cbm.prepare(device=torch.device("cpu"))
    loader = _valid_loader(n=20, batch_size=6)
    all_x = torch.cat([x for x, _ in loader])
    adv_indices = {1, 5, 6, 13, 19}
    attack = _ShiftAttack()

    out = eval_split(cbm, loader, attack, 0.1, torch.device("cpu"),
                     alpha=0.5, clean_weight=0.75, adv_indices=adv_indices)

    assert out["n_rob"] == len(adv_indices)
    attacked = torch.cat(attack.seen)
    assert attacked.shape[0] == len(adv_indices)
    expected = all_x[sorted(adv_indices)]
    assert torch.allclose(attacked, expected)


def test_eval_split_rob_absent_when_no_samples_attacked():
    """clean_weight=1 => empty subset => 'rob' omitted rather than nan."""
    cbm = _tiny_cbm()
    cbm.prepare(device=torch.device("cpu"))

    out = eval_split(cbm, _valid_loader(), _ShiftAttack(), 0.1, torch.device("cpu"),
                     alpha=0.5, clean_weight=1.0, adv_indices=set())

    assert "rob" not in out
    assert out["n_rob"] == 0


def test_eval_split_at_loss_matches_hand_computed_reference():
    """at_loss reproduces the split training objective, sample by sample."""
    device = torch.device("cpu")
    cbm = _tiny_cbm()
    cbm.prepare(device=device)
    loader = _valid_loader(n=20, batch_size=6)
    alpha, cw, shift = 0.4, 0.35, 0.05
    adv_indices = {0, 3, 4, 9, 11, 15, 17}

    out = eval_split(cbm, loader, _ShiftAttack(shift), 0.1, device,
                     alpha=alpha, clean_weight=cw, adv_indices=adv_indices)

    # Reference: one sample at a time, no batching, no subset bookkeeping.
    xs = torch.cat([x for x, _ in loader])
    ys = torch.cat([y for _, y in loader])
    with torch.no_grad():
        log_Z = cbm.log_partition_function()

    def _dis(x, y):
        with torch.no_grad():
            las = cbm._log_amp_sq(x.unsqueeze(0))
        return (torch.logsumexp(las, dim=1) - las[0, y]).item()

    dis_adv = [_dis(xs[i] + shift, ys[i]) for i in sorted(adv_indices)]
    dis_cln = [_dis(xs[i], ys[i]) for i in range(len(xs)) if i not in adv_indices]
    with torch.no_grad():
        las_all = cbm._log_amp_sq(xs)
    gen_all = (log_Z - las_all[range(len(ys)), ys]).mean().item()

    ref = (1 - alpha) * (
        (1 - cw) * sum(dis_adv) / len(dis_adv) + cw * sum(dis_cln) / len(dis_cln)
    ) + alpha * gen_all

    assert abs(out["at_loss"] - ref) < 1e-4
    # mixed_loss is the CLEAN alpha-mix, not the objective: it never sees x_adv,
    # so it stays comparable to a NAT run's mixed_loss/valid.
    clean_ref = (1 - alpha) * out["dis_loss"] + alpha * out["gen_loss"]
    assert abs(out["mixed_loss"] - clean_ref) < 1e-9
    assert abs(out["mixed_loss"] - out["at_loss"]) > 1e-6


# ── default objective: combined validation (eval_at) ────────────────────────

def test_eval_at_clean_metrics_match_eval_metrics():
    """acc/dis_loss/gen_loss are clean and over the full set, as in eval_metrics."""
    cbm = _tiny_cbm()
    cbm.prepare(device=torch.device("cpu"))
    # Evenly-dividing batches: eval_metrics averages per-batch means while eval_at
    # averages per sample, so the two coincide exactly only at equal batch sizes.
    loader = _valid_loader(n=20, batch_size=5)

    out = eval_at(cbm, loader, _ShiftAttack(), 0.1, torch.device("cpu"),
                  alpha=0.5, clean_weight=0.3)
    dis_loss, acc, gen_loss = eval_metrics(cbm, loader, torch.device("cpu"))

    assert abs(out["acc"] - acc) < 1e-9
    assert abs(out["dis_loss"] - dis_loss) < 1e-5
    assert abs(out["gen_loss"] - gen_loss) < 1e-5


def test_eval_at_rob_matches_eval_rob_over_the_full_set():
    """rob keeps eval_rob's meaning: every sample attacked, no subset estimator."""
    device = torch.device("cpu")
    cbm = _tiny_cbm()
    cbm.prepare(device=device)
    loader = _valid_loader(n=20, batch_size=6)
    attack = _ShiftAttack()

    out = eval_at(cbm, loader, attack, 0.1, device, alpha=0.5, clean_weight=0.3)
    ref = eval_rob(cbm, loader, _ShiftAttack(), 0.1, device)

    assert abs(out["rob"] - ref) < 1e-9
    assert torch.cat(attack.seen).shape[0] == 20   # whole set, once each


def test_eval_at_at_loss_matches_hand_computed_reference():
    """at_loss reproduces (1-cw)*mixed_nll(x_adv) + cw*mixed_nll(x), sample by sample."""
    device = torch.device("cpu")
    cbm = _tiny_cbm()
    cbm.prepare(device=device)
    loader = _valid_loader(n=20, batch_size=6)
    alpha, cw, shift = 0.5, 0.3, 0.05

    out = eval_at(cbm, loader, _ShiftAttack(shift), 0.1, device,
                  alpha=alpha, clean_weight=cw)

    # Reference: one sample at a time, no batching.
    xs = torch.cat([x for x, _ in loader])
    ys = torch.cat([y for _, y in loader])
    with torch.no_grad():
        log_Z = cbm.log_partition_function()

    def _terms(x, y):
        with torch.no_grad():
            las = cbm._log_amp_sq(x.unsqueeze(0))
        return ((torch.logsumexp(las, dim=1) - las[0, y]).item(),
                (log_Z - las[0, y]).item())

    def _mixed(shifted):
        pairs = [_terms(xs[i] + shifted, ys[i]) for i in range(len(ys))]
        dis = sum(d for d, _ in pairs) / len(pairs)
        gen = sum(g for _, g in pairs) / len(pairs)
        return (1 - alpha) * dis + alpha * gen

    # The generative half of the adversarial term is L_gen(x_adv), not L_gen(x):
    # this mirrors the objective, which puts the whole mixed NLL on x_adv.
    ref = (1 - cw) * _mixed(shift) + cw * _mixed(0.0)

    assert abs(out["at_loss"] - ref) < 1e-4


def test_eval_at_reduces_to_adversarial_dis_loss_at_alpha0_cw0():
    """The anchor case: pure PGD-AT selects on the discriminative loss on x_adv."""
    device = torch.device("cpu")
    cbm = _tiny_cbm()
    cbm.prepare(device=device)
    loader = _valid_loader(n=20, batch_size=5)
    shift = 0.05

    out = eval_at(cbm, loader, _ShiftAttack(shift), 0.1, device,
                  alpha=0.0, clean_weight=0.0)

    # Same loader, shifted once up front: eval_at's adversarial dis term.
    xs = torch.cat([x for x, _ in loader]) + shift
    ys = torch.cat([y for _, y in loader])
    shifted = DataLoader(TensorDataset(xs, ys), batch_size=5, shuffle=False)
    ref_dis, _, _ = eval_metrics(cbm, shifted, device)

    assert abs(out["at_loss"] - ref_dis) < 1e-5


def test_eval_at_reduces_to_clean_mixed_loss_at_cw1():
    """clean_weight=1 => no adversarial signal in the criterion."""
    device = torch.device("cpu")
    cbm = _tiny_cbm()
    cbm.prepare(device=device)

    out = eval_at(cbm, _valid_loader(), _ShiftAttack(), 0.1, device,
                  alpha=0.5, clean_weight=1.0)

    assert abs(out["at_loss"] - out["mixed_loss"]) < 1e-9
    # rob is still measured and reported at cw=1 -- selection stops using the
    # attack, evaluation does not.
    assert math.isfinite(out["rob"])


# ── at_loss as a stopping criterion ─────────────────────────────────────────

def test_at_loss_is_minimized_not_maximized(cbm):
    """at_loss is a loss: lower wins. Guards against an acc-style comparison."""
    t = _make_trainer(cbm, stop_crit="at_loss")
    t.best["at_loss"] = 0.5

    t.valid_perf = {"acc": 0.9, "at_loss": 0.7}
    t._update()
    assert t.best["at_loss"] == 0.5          # worse (higher) => rejected
    assert t.patience_counter == 1

    t.valid_perf = {"acc": 0.9, "at_loss": 0.3}
    t._update()
    assert t.best["at_loss"] == 0.3          # better (lower) => selected
    assert t.patience_counter == 0


def test_at_loss_requires_positive_eval_rob_freq():
    """at_loss comes from the attack pass, so eval_rob_freq=0 never produces it."""
    import pytest
    cfg = AdversarialConfig(
        stop_crit="at_loss", eval_rob_freq=0,
        norm_control=NormControlConfig(hard_every=0, soft_strength=0.0),
    )
    with pytest.raises(ValueError, match="eval_rob_freq >= 1"):
        AdversarialTrainer(cbm=_tiny_cbm(), train_cfg=cfg,
                           datahandler=_FakeDataHandler(n=20, batch_size=5),
                           device=torch.device("cpu"))


# ── gen_on_clean: validation cadence and patience accounting ────────────────

def _split_run_trainer(*, eval_rob_freq, clean_weight=0.5, max_epoch=9, patience=250):
    cbm = _tiny_cbm()
    dh = _FakeDataHandler(n=20, batch_size=5)
    cfg = AdversarialConfig(
        alpha=0.5, clean_weight=clean_weight, gen_on_clean=True,
        max_epoch=max_epoch, eval_rob_freq=eval_rob_freq, patience=patience,
        stop_crit="rob",
        norm_control=NormControlConfig(hard_every=0, soft_strength=0.0),
    )
    t = AdversarialTrainer(cbm=cbm, train_cfg=cfg, datahandler=dh,
                           device=torch.device("cpu"))
    t.attack = _ShiftAttack()          # cheap stand-in for PGD
    t._generate_adversarial = lambda data, labels, eps_abs: data + 0.05
    return t


def test_split_valid_subset_is_fixed_across_epochs():
    """The attacked valid subset is drawn once, sized (1-cw)*n_valid."""
    t = _split_run_trainer(eval_rob_freq=3, clean_weight=0.4)
    assert len(t.adv_indices) == round(0.6 * 20)

    again = _split_run_trainer(eval_rob_freq=3, clean_weight=0.4)
    assert t.adv_indices == again.adv_indices  # constant seed, not the run seed


def test_split_requires_positive_eval_rob_freq():
    """eval_rob_freq is the whole validation cadence in split mode."""
    import pytest
    with pytest.raises(ValueError, match="eval_rob_freq >= 1"):
        _split_run_trainer(eval_rob_freq=0)


def test_split_validates_only_every_eval_rob_freq_epochs():
    """Valid metrics appear on eval epochs only; patience counts valid events."""
    t = _split_run_trainer(eval_rob_freq=3, max_epoch=9)

    logged = []
    t.train(on_epoch_end=lambda ep, m: logged.append((ep, m)))

    valid_epochs = [ep for ep, m in logged if "acc/valid" in m]
    assert valid_epochs == [3, 6, 9]
    for ep, m in logged:
        # train-side metrics are still emitted every epoch
        assert "dis_loss/train" in m
        if ep in valid_epochs:
            # Robustness is keyed by the run's relative budget (see AdversarialTrainer
            # .rob_metric_key), so the key states what was measured.
            assert {t.rob_metric_key, "mixed_loss/valid", "n_rob_valid"} <= set(m)
            assert m["n_rob_valid"] == len(t.adv_indices)
    # 3 valid events => patience can have ticked at most 3 times
    assert t.patience_counter <= 3


def test_unsplit_still_validates_every_epoch():
    """Regression guard: the default path is untouched by the cadence change."""
    cbm = _tiny_cbm()
    dh = _FakeDataHandler(n=20, batch_size=5)
    cfg = AdversarialConfig(
        alpha=0.0, max_epoch=4, eval_rob_freq=2, stop_crit="acc",
        norm_control=NormControlConfig(hard_every=0, soft_strength=0.0),
    )
    t = AdversarialTrainer(cbm=cbm, train_cfg=cfg, datahandler=dh,
                           device=torch.device("cpu"))
    t.attack = _ShiftAttack()
    t._generate_adversarial = lambda data, labels, eps_abs: data + 0.05

    logged = []
    t.train(on_epoch_end=lambda ep, m: logged.append((ep, m)))

    assert [ep for ep, m in logged if "acc/valid" in m] == [1, 2, 3, 4]
    assert [ep for ep, m in logged if t.rob_metric_key in m] == [2, 4]
    assert not any("n_rob_valid" in m for _, m in logged)
    # at_loss needs the attack, so it appears on rob epochs only; mixed_loss is
    # clean and therefore emitted every epoch.
    assert [ep for ep, m in logged if "at_loss/valid" in m] == [2, 4]
    assert [ep for ep, m in logged if "mixed_loss/valid" in m] == [1, 2, 3, 4]
