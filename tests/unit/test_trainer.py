"""Unit tests for the Trainer (NAT and AT), its objective, validation and selection."""

import math
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from bm4tc.pipeline.metrics import flatten_epoch, key
from bm4tc.core.model import ConditionalBornMachine, CBMConfig, MPSInitConfig
from bm4tc.core.graphs import Graphed
from bm4tc.core.train import Trainer, TrainConfig
from bm4tc.core.objective import (
    NORM_STATISTICS, NormControlConfig, NormRegularizer, NormTracker, eval_rob, evaluate,
    mix, norm_statistics,
)

CPU = torch.device("cpu")
PGD = {"method": "PGD", "eps_rel": [0.1]}
NO_NORM = NormControlConfig(soft_strength=0.0)


# ── Helpers ────────────────────────────────────────────────────────────────

def _tiny_cbm():
    cfg = CBMConfig(
        embedding="fourier",
        init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3),
    )
    return ConditionalBornMachine(cfg=cfg, data_dim=2, num_classes=2)


def _loaders(n=16, batch_size=4):
    """The same tiny random loader as train and valid (no file I/O)."""
    ds = TensorDataset(torch.rand(n, 2), torch.randint(0, 2, (n,)))
    loader = DataLoader(ds, batch_size=batch_size)
    return loader, loader


class _ShiftAttack:
    """Deterministic stand-in for PGD; records which samples it was handed."""

    def __init__(self, shift=0.05):
        self.shift = shift
        self.seen = []

    def generate(self, model, naturals, labels, eps_abs, device):
        self.seen.append(naturals.detach().clone())
        return (naturals + self.shift).detach()


def _trainer(cfg, *, n=16, batch_size=4, stub_attack=True):
    """A real Trainer on a tiny model; PGD replaced by a cheap shift."""
    t = Trainer(_tiny_cbm(), cfg, *_loaders(n=n, batch_size=batch_size), CPU)
    if stub_attack and t.attack is not None:
        t.attack = _ShiftAttack()
    return t


def _ready(t, *, norm_regularizer=None, log_target=0.0):
    """Do what train() does before the epoch loop, so _train_epoch can run alone."""
    t.cbm.prepare(device=CPU)
    t.step = 0
    t._nc_log_target = log_target
    t.norm_regularizer = norm_regularizer
    t.optimizer = torch.optim.Adam(t.cbm.parameters(), lr=1e-3)
    return t


# ── Config and construction ────────────────────────────────────────────────

def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.beta == 0.0
    assert cfg.evasion is None  # NAT
    assert cfg.eval_every == 1
    assert cfg.patience == 250
    assert cfg.max_epoch == 100
    assert cfg.batch_size == 64
    assert cfg.save is False
    assert cfg.cuda_graph is True  # D90; ignored on CPU
    assert not hasattr(cfg, "stop_crit")  # selection is a fixed rule (D8)
    assert not hasattr(cfg, "acc_floor")  # D40
    assert not hasattr(cfg, "gen_on_clean")  # the split objective is the only one (D18)


def test_norm_control_config_defaults():
    nc = NormControlConfig()
    assert nc.log_target == 0.0
    assert not hasattr(nc, "hard_every")  # hard renormalisation is gone (D90)
    assert nc.soft_strength == 0.1


def test_nat_trainer_constructs_without_attack():
    t = _trainer(TrainConfig())
    assert t.attack is None
    assert t.adv_indices == set()
    assert t.norm_regularizer is None
    assert len(t.best_tensors) == len(t.cbm.tensors)


def test_eval_every_must_be_positive():
    with pytest.raises(ValueError, match="eval_every"):
        _trainer(TrainConfig(eval_every=0))


def test_unknown_evasion_key_fails_at_construction():
    with pytest.raises(Exception, match="eps"):
        _trainer(TrainConfig(evasion={"method": "PGD", "eps": 0.1}))


def test_attack_budget_and_valid_subset():
    """The subset is (1-cw)·n_valid positions, drawn from a constant seed."""
    cfg = TrainConfig(evasion=PGD, clean_weight=0.4)
    t = _trainer(cfg, n=20, batch_size=5)
    assert t.eps_rel == 0.1
    assert len(t.adv_indices) == round(0.6 * 20)
    again = _trainer(cfg, n=20, batch_size=5)
    assert t.adv_indices == again.adv_indices  # constant seed, not the run seed


def test_curriculum_reaches_the_full_radius_at_its_end_fraction():
    t = _trainer(TrainConfig(evasion=PGD, curriculum=True, curriculum_end=0.5, max_epoch=10))
    assert t._eps_abs(0) == pytest.approx(0.0)
    assert t._eps_abs(1) == pytest.approx(t.eps_abs / 5)
    assert t._eps_abs(5) == pytest.approx(t.eps_abs)
    assert t._eps_abs(9) == pytest.approx(t.eps_abs)
    flat = _trainer(TrainConfig(evasion=PGD, max_epoch=10))
    assert flat._eps_abs(1) == flat.eps_abs


# ── NormRegularizer ────────────────────────────────────────────────────────

def test_norm_regularizer_zero_at_target():
    log_target = math.log(2.5)
    reg = NormRegularizer(strength=1.0, log_target=log_target)
    cbm = MagicMock(n_features=5)
    cbm.log_Z.return_value = torch.tensor(log_target)
    penalty = reg(cbm)
    cbm.log_Z.assert_called_once_with(recompute=False)
    assert penalty.item() == pytest.approx(0.0, abs=1e-6)


def test_norm_regularizer_nonzero_off_target():
    """strength·(log Z − target)²/N: per site (D86)."""
    reg = NormRegularizer(strength=3.0, log_target=0.0)
    cbm = MagicMock(n_features=4)
    cbm.log_Z.return_value = torch.tensor(2.0)
    assert reg(cbm).item() == pytest.approx(3.0 * 2.0 ** 2 / 4, rel=1e-5)


def test_norm_regularizer_invalid_target():
    with pytest.raises(ValueError, match="log_target must be finite"):
        NormRegularizer(strength=1.0, log_target=float("inf"))


def test_log_target_resolves_float_expression_and_pretrained():
    """The three forms of NormControlConfig.log_target: a number, an expression in
    the model's sizes (the pilot's n·ln d / 2), and None = the model's own log Z."""
    from bm4tc.core.objective import resolve_log_target
    cbm = _tiny_cbm()                                  # 2 data sites + the class site
    assert resolve_log_target(cbm, NormControlConfig(log_target=0.0)) == 0.0
    half = resolve_log_target(cbm, NormControlConfig(log_target="n_features * log(in_dim) / 2"))
    assert half == pytest.approx(cbm.n_features * math.log(cbm.in_dim) / 2)
    assert cbm.n_features == 3
    own = resolve_log_target(cbm, NormControlConfig(log_target=None))
    assert own == pytest.approx(cbm.log_partition_function().item())
    with pytest.raises(ValueError, match="could not evaluate"):
        resolve_log_target(cbm, NormControlConfig(log_target="n_sites * 2"))


# ── Collapse diagnostics ───────────────────────────────────────────────────

def test_amp_nonfinite_count_always_tagged_overflow():
    """2·log(|amp|.clamp(min=tiny)) floors underflow, so a non-finite count is
    overflow, also when the mean log|amp|² is small."""
    s = Trainer._format_diagnostics({
        "log_Z": 0.0,
        "log_amp_sq_mean": 1.0, "log_amp_sq_min": 0.0, "log_amp_sq_max": 2.0,
        "amp_nonfinite_count": 3,
    })
    assert "3 non-finite → overflow" in s
    assert "underflow" not in s


def test_format_diagnostics_surfaces_overflow_headroom():
    assert "overflow headroom=77.45" in Trainer._format_diagnostics(
        {"log_Z": 100.0, "log_Z_headroom": 77.45})
    assert "headroom" not in Trainer._format_diagnostics({"log_Z": 100.0})


def test_format_diagnostics_leaves_out_a_log_Z_the_step_did_not_form():
    report = Trainer._format_diagnostics({"log_amp_sq_mean": 1.0, "log_amp_sq_min": 0.0,
                                          "log_amp_sq_max": 2.0, "amp_nonfinite_count": 0})
    assert "log_Z" not in report and "norm" not in report


def test_log_Z_overflow_ceiling_matches_float32():
    """The float32/complex64 overflow ceiling is 2·log(finfo.max) ≈ 177.45."""
    ceiling = 2.0 * math.log(torch.finfo(torch.float32).max)
    assert ceiling == pytest.approx(177.45, abs=0.1)


@pytest.mark.parametrize("evasion", [None, PGD], ids=["nat", "at"])
def test_nonfinite_loss_ends_the_epoch_without_retry(evasion):
    """Both regimes: the first non-finite loss ends the epoch, and the warning
    reports the failing forward's norm statistics."""
    t = _ready(_trainer(TrainConfig(evasion=evasion, norm_control=NO_NORM)))
    nan = torch.tensor(float("nan"), requires_grad=True)
    log_amp_sq = torch.tensor([[0.0, float("inf")]] * 4)
    with patch.object(t.cbm, "mixed_nll", return_value=(nan, log_amp_sq)) as m, \
         patch("bm4tc.core.train.logger") as m_logger:
        t._train_epoch(eps_abs=0.0)
    assert t._collapsed
    assert t.step == 1
    # beta=0, cw=0: one forward per objective (AT: the adversarial one only)
    assert m.call_count == 1
    warning = m_logger.warning.call_args.args[0]
    assert "NaN/inf loss at step 1" in warning and "4 non-finite → overflow" in warning


def test_runtime_error_collapses_but_oom_is_raised():
    t = _ready(_trainer(TrainConfig(evasion=PGD, norm_control=NO_NORM)))
    with patch.object(t.cbm, "mixed_nll", side_effect=RuntimeError("svd failed")):
        t._train_epoch(eps_abs=0.0)
    assert t._collapsed
    with patch.object(t.cbm, "mixed_nll", side_effect=RuntimeError("CUDA out of memory")):
        with pytest.raises(RuntimeError, match="out of memory"):
            t._train_epoch(eps_abs=0.0)


# ── Norm control in the epoch loop ─────────────────────────────────────────

def test_beta0_skips_per_step_log_partition_function():
    """NAT beta=0 without norm control contracts the norm once per epoch: the
    NormTracker's end-of-epoch snapshot, not once per step."""
    t = _ready(_trainer(TrainConfig(beta=0.0, norm_control=NO_NORM)))  # 4 steps
    with patch.object(t.cbm, "log_partition_function",
                      wraps=t.cbm.log_partition_function) as mock_logZ:
        t._train_epoch(eps_abs=0.0)
    assert mock_logZ.call_count == 1
    assert math.isfinite(t._norm_stats["norm/log_Z_mean"])


@pytest.mark.parametrize("evasion", [None, PGD], ids=["nat", "at"])
def test_beta0_soft_norm_control_multistep_backward(evasion):
    """Regression: beta=0 + soft norm control trains across several steps.

    The regularizer reads the with-grad log Z via recompute=False and mixed_nll
    never refreshes it at beta=0; without per-step invalidation the second step
    backwards through the first step's freed graph."""
    nc = NormControlConfig(soft_strength=1.0, log_target=0.0)
    t = _ready(_trainer(TrainConfig(beta=0.0, evasion=evasion, norm_control=nc)),
               norm_regularizer=NormRegularizer(strength=1.0, log_target=0.0))
    t._train_epoch(eps_abs=0.1)
    assert not t._collapsed
    assert t.step >= 2
    assert t._train_penalty > 0.0


def test_no_penalty_without_soft_norm_control():
    t = _ready(_trainer(TrainConfig(evasion=PGD, norm_control=NO_NORM)))
    t._train_epoch(eps_abs=0.1)
    assert t._train_penalty == 0.0


def test_norm_stats_populated_after_epoch():
    t = _ready(_trainer(TrainConfig(evasion=PGD, norm_control=NO_NORM)))
    t._train_epoch(eps_abs=0.1)
    # beta=0 / no soft → log_Z via the end-of-epoch snapshot; amp from the adv forward.
    for k in ("norm/log_Z_mean", "norm/log_Z_max", "norm/log_amp_sq_mean"):
        assert math.isfinite(t._norm_stats[k]), k


# ── NormTracker ─────────────────────────────────────────────────────────────

class _FakeNormCBM:
    """Minimal cbm exposing what NormTracker.finalize reads: dtype, n_features
    (1: the per-site values equal the absolute ones) and the log Z snapshot."""
    def __init__(self, dtype=torch.complex64, snapshot=3.0, n_features=1):
        self.n_features = n_features
        self.dtype = dtype
        self._snapshot = snapshot

    def log_partition_function(self):
        return torch.tensor(self._snapshot)


def _statistics(mean, high, low, log_Z=float("nan")):
    """One step's norm statistics, in NORM_STATISTICS order."""
    return torch.tensor([mean, low, high, 0.0, 0.0, log_Z])


def test_norm_statistics_reads_the_finite_entries():
    log_amp_sq = torch.tensor([[1.0, float("inf")], [3.0, float("nan")]])
    stats = dict(zip(NORM_STATISTICS, norm_statistics(log_amp_sq, None).tolist()))
    assert (stats["log_amp_sq_mean"], stats["log_amp_sq_min"], stats["log_amp_sq_max"]) == (2.0, 1.0, 3.0)
    assert (stats["amp_nonfinite_count"], stats["amp_nan_count"]) == (2.0, 1.0)
    assert math.isnan(stats["log_Z"])
    assert norm_statistics(log_amp_sq, torch.tensor(5.0))[-1] == 5.0


def test_norm_tracker_aggregates_mean_max_min():
    t = NormTracker()
    cbm = _FakeNormCBM()
    t.add(_statistics(-2.0, -1.0, -3.0, log_Z=1.0))
    t.add(_statistics(-4.0, 0.0, -6.0, log_Z=5.0))
    out = t.finalize(cbm)

    assert out["norm/log_Z_mean"] == pytest.approx(3.0)
    assert out["norm/log_Z_max"] == 5.0
    assert out["norm/log_Z_min"] == 1.0
    assert out["norm/log_amp_sq_mean"] == pytest.approx(-3.0)
    assert out["norm/log_amp_sq_max"] == 0.0
    assert out["norm/log_amp_sq_min"] == -6.0
    ceiling = 2.0 * math.log(torch.finfo(torch.complex64).max)
    assert out["norm/log_Z_headroom"] == pytest.approx(ceiling - 5.0)


def test_norm_tracker_copies_what_it_is_given():
    """A captured step overwrites its outputs on every replay (D90)."""
    t = NormTracker()
    statistics = _statistics(-2.0, -1.0, -3.0, log_Z=1.0)
    t.add(statistics)
    statistics.fill_(100.0)
    assert t.finalize(_FakeNormCBM())["norm/log_Z_mean"] == 1.0


def test_norm_tracker_reports_per_site_but_headroom_absolute():
    """norm/log_Z_* and norm/log_amp_sq_* are divided by N; the headroom is not
    (overflow happens at an absolute log Z, D86)."""
    t = NormTracker()
    cbm = _FakeNormCBM(n_features=4)
    t.add(_statistics(-2.0, -1.0, -3.0, log_Z=8.0))
    out = t.finalize(cbm)
    assert out["norm/log_Z_mean"] == out["norm/log_Z_max"] == out["norm/log_Z_min"] == 2.0
    assert (out["norm/log_amp_sq_mean"], out["norm/log_amp_sq_max"],
            out["norm/log_amp_sq_min"]) == (-0.5, -0.25, -0.75)
    ceiling = 2.0 * math.log(torch.finfo(torch.complex64).max)
    assert out["norm/log_Z_headroom"] == pytest.approx(ceiling - 8.0)


def test_norm_tracker_logZ_snapshot_fallback():
    t = NormTracker()
    cbm = _FakeNormCBM(snapshot=3.0)
    t.add(_statistics(-2.0, -1.0, -3.0))  # no log Z formed
    out = t.finalize(cbm)
    assert out["norm/log_Z_mean"] == out["norm/log_Z_max"] == out["norm/log_Z_min"] == 3.0
    assert out["norm/log_amp_sq_mean"] == pytest.approx(-2.0)


def test_norm_tracker_ignores_nonfinite():
    t = NormTracker()
    cbm = _FakeNormCBM()
    t.add(_statistics(float("nan"), float("inf"), -5.0, log_Z=float("inf")))
    out = t.finalize(cbm)
    assert out["norm/log_Z_mean"] == 3.0          # from snapshot, not inf
    assert out["norm/log_amp_sq_min"] == -5.0
    assert "norm/log_amp_sq_mean" not in out


# ── The training objective ──────────────────────────────────────────────────

# Distinct per-batch values so a test can tell the adversarial batch (tag 1.0)
# from the clean one (tag 0.0) purely from the returned loss.
_L_DIS = {0.0: 2.0, 1.0: 3.0}
_L_GEN = {0.0: 7.0, 1.0: 8.0}
_N = 3   # modelled variables of the stub


class _DecompStubCBM:
    """Stub whose mixed_nll decomposes exactly like the real one:
    ``mixed_nll(x, y, b) = (1-b)*L_dis(x) + (b/N)*L_gen(x)``, with L_dis/L_gen keyed
    off a per-batch tag, so the weighting can be checked in closed form."""
    n_features = _N
    dtype = torch.float32

    def __init__(self):
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def train(self): pass
    def eval(self): pass
    def log_Z(self, recompute=False): return torch.tensor(0.0)

    def mixed_nll(self, data, labels, beta, debug=False):
        tag = float(data[0, 0])
        self.calls.append((tag, beta))
        # param keeps the result a graph leaf so backward() works in _train_epoch
        objective = self.param.sum() + (1 - beta) * _L_DIS[tag] + beta / _N * _L_GEN[tag]
        return objective, torch.zeros(len(data), 2)


class _OnesAttack:
    def generate(self, model, naturals, labels, eps_abs, device):
        return torch.ones_like(naturals)


def _stub_trainer(beta, cw, *, attack=True):
    """Trainer wired with just what _objective / _train_epoch touch."""
    cbm = _DecompStubCBM()
    t = Trainer.__new__(Trainer)
    t.cfg = TrainConfig(beta=beta, clean_weight=cw, norm_control=NO_NORM)
    t.cbm = cbm
    t.device = CPU
    t.step = 0
    t._nc = t.cfg.norm_control
    t.norm_regularizer = None
    t.attack = _OnesAttack() if attack else None
    t.clean_weight = cw if attack else 1.0
    clean = (torch.zeros(4, 2), torch.zeros(4, dtype=torch.long))  # tag 0.0
    t.train_loader = [clean]
    t.optimizer = torch.optim.SGD([cbm.param], lr=0.0)
    t._graphed_train_step = Graphed(t._train_step, enabled=False)
    return t, cbm


def _objective(t):
    objective, _ = t._objective(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long), 0.1)
    return objective


def _naive_at_loss(beta, cw):
    """The three-term form the two-call implementation must reproduce."""
    return (1 - beta) * ((1 - cw) * _L_DIS[1.0] + cw * _L_DIS[0.0]) + beta / _N * _L_GEN[0.0]


def test_at_objective_matches_naive_three_term_form():
    for beta, cw in [(0.5, 0.3), (0.1, 0.0), (0.9, 0.7), (0.25, 1.0), (0.0, 0.4)]:
        t, _ = _stub_trainer(beta, cw)
        assert _objective(t).item() == pytest.approx(_naive_at_loss(beta, cw), abs=1e-6)


def test_at_objective_uses_two_forwards_with_rescaled_beta():
    beta, cw = 0.5, 0.3
    t, cbm = _stub_trainer(beta, cw)
    _objective(t)
    s = (1 - beta) * cw + beta
    assert len(cbm.calls) == 2, cbm.calls
    assert cbm.calls[0] == (1.0, 0.0)           # adversarial batch, discriminative
    assert cbm.calls[1][0] == 0.0               # clean batch
    assert cbm.calls[1][1] == pytest.approx(beta / s)


def test_at_objective_at_beta0_cw0_is_adversarial_dis_loss():
    t, cbm = _stub_trainer(0.0, 0.0)
    assert _objective(t).item() == pytest.approx(_L_DIS[1.0])
    assert cbm.calls == [(1.0, 0.0)]


def test_at_objective_at_beta1_drops_the_adversarial_term():
    t, cbm = _stub_trainer(1.0, 0.3)
    assert _objective(t).item() == pytest.approx(_L_GEN[0.0] / _N)
    assert cbm.calls == [(0.0, 1.0)]


@pytest.mark.parametrize("beta", [0.0, 0.01, 0.5, 1.0])
def test_nat_objective_is_one_mixed_nll_at_beta(beta):
    """No attack: exactly mixed_nll(x, beta), not a rescaled call."""
    t, cbm = _stub_trainer(beta, 0.3, attack=False)
    assert _objective(t).item() == pytest.approx((1 - beta) * 2.0 + beta * 7.0 / _N)
    assert cbm.calls == [(0.0, beta)]


def test_train_epoch_routes_through_the_objective():
    t, cbm = _stub_trainer(0.5, 0.3)
    t._train_epoch(eps_abs=0.1)
    assert len(cbm.calls) == 2  # one training step in the stub loader
    assert cbm.calls[0][1] == 0.0
    assert t._train_objective == pytest.approx(_naive_at_loss(0.5, 0.3))


# ── evaluate ────────────────────────────────────────────────────────────────

def _valid_loader(n=20, batch_size=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.rand(n, 2, generator=g),
                       torch.randint(0, 2, (n,), generator=g))
    # shuffle=False mirrors DataHandler's non-train splits: positional indices stable
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _ready_cbm():
    cbm = _tiny_cbm()
    cbm.prepare(device=CPU)
    return cbm


def test_evaluate_is_per_sample_not_per_batch():
    """A short last batch is not over-weighted: batching does not change the means."""
    cbm = _ready_cbm()
    whole = evaluate(cbm, _valid_loader(n=20, batch_size=20), CPU, beta=0.5)
    ragged = evaluate(cbm, _valid_loader(n=20, batch_size=6), CPU, beta=0.5)
    for k in ("loss_dis", "loss_x", "acc", "objective"):
        assert ragged[k] == pytest.approx(whole[k], rel=1e-5), k


def test_evaluate_without_attack_is_the_clean_mix():
    cbm = _ready_cbm()
    out = evaluate(cbm, _valid_loader(), CPU, beta=0.3)
    assert set(out) == {"objective", "loss_dis", "loss_x", "acc"}
    assert out["objective"] == pytest.approx(_clean_mix(out, 0.3, n=2), rel=1e-6)


def _clean_mix(out, beta, n):
    """(1-s)·loss_dis + s·loss_x, s = β·n/N: the objective as a weighted mean of
    nats per label and nats per feature (D86)."""
    s = beta * n / (n + 1)
    return (1 - s) * out["loss_dis"] + s * out["loss_x"]


def test_evaluate_loss_x_is_the_marginal_nll_per_feature():
    cbm = _ready_cbm()
    loader = _valid_loader()
    out = evaluate(cbm, loader, CPU)
    xs = torch.cat([x for x, _ in loader])
    with torch.no_grad():
        ref = -cbm.marginal_log_probability(xs).mean().item() / xs.shape[1]
    assert out["loss_x"] == pytest.approx(ref, abs=1e-5)


def test_evaluate_clean_metrics_do_not_depend_on_the_attack():
    """acc/loss_dis/loss_x are clean and over the full set."""
    cbm = _ready_cbm()
    loader = _valid_loader()
    out = evaluate(cbm, loader, CPU, beta=0.5, attack=_ShiftAttack(), eps_abs=0.1,
                   clean_weight=0.3, adv_indices={0, 1, 2})
    clean = evaluate(cbm, loader, CPU, beta=0.5)
    for k in ("loss_dis", "loss_x", "acc"):
        assert out[k] == pytest.approx(clean[k], rel=1e-6), k


def test_evaluate_attacks_only_the_given_subset():
    cbm = _ready_cbm()
    loader = _valid_loader(n=20, batch_size=6)
    all_x = torch.cat([x for x, _ in loader])
    adv_indices = {1, 5, 6, 13, 19}
    attack = _ShiftAttack()
    out = evaluate(cbm, loader, CPU, beta=0.5, attack=attack, eps_abs=0.1,
                   clean_weight=0.75, adv_indices=adv_indices)
    assert out["n_rob"] == len(adv_indices)
    attacked = torch.cat(attack.seen)
    assert torch.allclose(attacked, all_x[sorted(adv_indices)])


def test_evaluate_rob_absent_when_no_samples_attacked():
    """clean_weight=1 => empty subset => 'rob' omitted rather than nan."""
    out = evaluate(_ready_cbm(), _valid_loader(), CPU, beta=0.5, attack=_ShiftAttack(),
                   eps_abs=0.1, clean_weight=1.0, adv_indices=set())
    assert "rob" not in out
    assert out["n_rob"] == 0


def test_evaluate_rob_matches_eval_rob_when_every_sample_is_attacked():
    cbm = _ready_cbm()
    loader = _valid_loader(n=20, batch_size=6)
    out = evaluate(cbm, loader, CPU, attack=_ShiftAttack(), eps_abs=0.1,
                   clean_weight=0.0, adv_indices=set(range(20)))
    assert out["rob"] == pytest.approx(eval_rob(cbm, loader, _ShiftAttack(), 0.1, CPU))


def test_evaluate_objective_matches_hand_computed_reference():
    """The objective reproduces the AT training objective, sample by sample."""
    cbm = _ready_cbm()
    loader = _valid_loader(n=20, batch_size=6)
    beta, cw, shift = 0.4, 0.35, 0.05
    adv_indices = {0, 3, 4, 9, 11, 15, 17}

    out = evaluate(cbm, loader, CPU, beta=beta, attack=_ShiftAttack(shift),
                   eps_abs=0.1, clean_weight=cw, adv_indices=adv_indices)

    xs = torch.cat([x for x, _ in loader])
    ys = torch.cat([y for _, y in loader])
    with torch.no_grad():
        log_Z = cbm.log_partition_function()

    def _dis(x, y):
        with torch.no_grad():
            las = cbm.log_amp_sq(x.unsqueeze(0))
        return (torch.logsumexp(las, dim=1) - las[0, y]).item()

    dis_adv = [_dis(xs[i] + shift, ys[i]) for i in sorted(adv_indices)]
    dis_cln = [_dis(xs[i], ys[i]) for i in range(len(xs)) if i not in adv_indices]
    with torch.no_grad():
        las_all = cbm.log_amp_sq(xs)
    gen_all = (log_Z - las_all[range(len(ys)), ys]).mean().item()
    ref = (1 - beta) * (
        (1 - cw) * sum(dis_adv) / len(dis_adv) + cw * sum(dis_cln) / len(dis_cln)
    ) + beta * gen_all / (xs.shape[1] + 1)

    assert out["objective"] == pytest.approx(ref, abs=1e-4)
    # The clean beta-mix is not the objective: it never sees x_adv.
    assert abs(out["objective"] - _clean_mix(out, beta, n=xs.shape[1])) > 1e-6


# ── Selection ───────────────────────────────────────────────────────────────

def _selector():
    t = Trainer.__new__(Trainer)
    t.cbm = _tiny_cbm()
    t.best = {"objective": float("inf")}
    t.best_tensors = [tt.cpu().clone().detach() for tt in t.cbm.tensors]
    t.patience_counter = 0
    t.best_epoch = 0
    t.epoch = 1
    return t


def test_objective_is_minimized():
    """The objective is a loss: lower wins, a higher value counts against patience."""
    t = _selector()
    t._update({"objective": 1.5, "acc": 0.9})
    assert (t.best["objective"], t.best_epoch, t.patience_counter) == (1.5, 1, 0)

    t.epoch = 2
    t._update({"objective": 2.0, "acc": 0.9})
    assert (t.best["objective"], t.patience_counter) == (1.5, 1)

    t.epoch = 3
    t._update({"objective": 0.3, "acc": 0.1})  # clean acc plays no part (D40)
    assert (t.best["objective"], t.best_epoch, t.patience_counter) == (0.3, 3, 0)


def test_nonfinite_objective_is_never_selected():
    t = _selector()
    initial = t.best_tensors
    t._update({"objective": float("nan"), "acc": 0.9})
    assert t.best_tensors is initial
    assert t.patience_counter == 1


# ── Validation cadence ──────────────────────────────────────────────────────

@pytest.mark.parametrize("evasion", [None, PGD], ids=["nat", "at"])
def test_validates_every_eval_every_epochs(evasion):
    """Valid metrics appear on eval epochs only; patience counts valid events."""
    t = _trainer(TrainConfig(beta=0.5, evasion=evasion, clean_weight=0.5, max_epoch=9,
                             eval_every=3, norm_control=NO_NORM), n=20, batch_size=5)
    logged = []
    t.train(on_epoch_end=lambda ep, m: logged.append((ep, flatten_epoch(m))))

    assert [ep for ep, m in logged if "objective/valid" in m] == [3, 6, 9]
    assert all("objective/train" in m for _, m in logged)
    assert t.patience_counter <= 3
    for ep, m in logged:
        if "objective/valid" not in m:
            continue
        if evasion is None:
            assert not any(k.startswith(("rob/", "n_rob/", "loss_adv/")) for k in m)
        else:
            rob = key("rob", "valid", t.eps_rel)
            assert {rob, "loss_adv/valid", "n_rob/valid", "eps_rel/train"} <= set(m)
            assert m["n_rob/valid"] == len(t.adv_indices)


def test_nat_logs_norm_metrics_every_epoch():
    t = _trainer(TrainConfig(max_epoch=2, norm_control=NO_NORM))
    logged = []
    t.train(on_epoch_end=lambda ep, m: logged.append(flatten_epoch(m)))
    assert len(logged) == 2
    for k in ("norm/log_Z_mean", "norm/log_Z_max", "norm/log_Z_min",
              "norm/log_Z_headroom", "norm/log_amp_sq_mean"):
        assert k in logged[-1], f"missing {k}"


# ── Micro-batches (D79) ────────────────────────────────────────────────────

@pytest.mark.parametrize("evasion", [None, {"method": "PGD", "eps_rel": [0.1],
                                            "num_steps": 3, "random_start": False}])
def test_micro_batches_take_the_full_batch_step(evasion):
    """A batch computed in chunks takes the same optimizer step as the batch at
    once, up to float rounding: the gradient of the batch mean, the norm penalty added once (NAT and
    real PGD, which must leave the accumulated parameter gradients alone)."""
    torch.manual_seed(0)
    loaders = _loaders(n=16, batch_size=8)
    tensors = []
    for micro in (None, 2):
        torch.manual_seed(1)
        cfg = TrainConfig(beta=0.5, evasion=evasion, micro_batch_size=micro,
                          norm_control=NO_NORM)
        t = Trainer(_tiny_cbm(), cfg, *loaders, CPU)
        _ready(t, norm_regularizer=NormRegularizer(strength=1e-2, log_target=0.0))
        # SGD carries the gradient over one to one; Adam's first step g/(|g|+eps)
        # would blow float rounding up into a visible difference on tiny entries.
        t.optimizer = torch.optim.SGD(t.cbm.parameters(), lr=1e-6)
        t._train_epoch(eps_abs=0.05 if evasion else 0.0)
        assert not t._collapsed and t.step == 2
        tensors.append([x.detach().clone() for x in t.cbm.tensors])
    for full, chunked in zip(*tensors):
        torch.testing.assert_close(chunked, full, rtol=1e-5, atol=1e-7)


def test_pgd_leaves_parameter_gradients_alone():
    from bm4tc.core.attacks import ProjectedGradientDescent
    torch.manual_seed(0)
    cbm = _tiny_cbm()
    cbm.prepare(device=CPU)
    for p in cbm.parameters():
        p.grad = torch.ones_like(p)
    x, y = torch.rand(4, 2), torch.randint(0, 2, (4,))
    ProjectedGradientDescent(norm="inf", num_steps=2).generate(cbm, x, y, 0.05, CPU)
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in cbm.parameters())
