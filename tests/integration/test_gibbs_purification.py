import pytest
import torch
from tests.conftest import DATA_DIM
from src.analysis.purification import GibbsPurification

pytestmark = pytest.mark.slow

NUM_BINS = 10
BATCH_SIZE = 4


@pytest.fixture
def x_adv():
    return torch.rand(BATCH_SIZE, DATA_DIM)


def test_purify_output_shape(cbm, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE)
    purified, _ = purifier.purify(cbm, x_adv, n_sweeps=1, device="cpu")
    assert purified.shape == (BATCH_SIZE, DATA_DIM)


def test_purify_log_px_shape(cbm, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE)
    _, log_px = purifier.purify(cbm, x_adv, n_sweeps=1, device="cpu")
    assert log_px.shape == (BATCH_SIZE,)


def test_purify_in_input_range(cbm, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE)
    purified, _ = purifier.purify(cbm, x_adv, n_sweeps=1, device="cpu")
    lo, hi = cbm.input_range
    assert (purified >= lo - 1e-5).all()
    assert (purified <= hi + 1e-5).all()


def test_purify_log_px_finite(cbm, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE)
    _, log_px = purifier.purify(cbm, x_adv, n_sweeps=1, device="cpu")
    assert torch.isfinite(log_px).all()


def test_purify_one_sweep(cbm, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE)
    purified, log_px = purifier.purify(cbm, x_adv, n_sweeps=1, device="cpu")
    assert purified.shape[0] == BATCH_SIZE


def test_purify_three_sweeps(cbm, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE)
    purified, log_px = purifier.purify(cbm, x_adv, n_sweeps=3, device="cpu")
    assert purified.shape[0] == BATCH_SIZE


def test_purify_batch_size_one(cbm):
    x = torch.rand(1, DATA_DIM)
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=1)
    purified, _ = purifier.purify(cbm, x, n_sweeps=1, device="cpu")
    assert purified.shape == (1, DATA_DIM)


def test_purify_partial_batch(cbm):
    n_samples = 5
    x = torch.rand(n_samples, DATA_DIM)
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=3)
    purified, _ = purifier.purify(cbm, x, n_sweeps=1, device="cpu")
    assert purified.shape == (n_samples, DATA_DIM)


# --- Restricted Gibbs (step_radius) ---
# step_radius is a PER-SWEEP L-inf step, not a global budget: the window re-centres
# at the start of every sweep, so the k-sweep envelope is k*step_radius*(hi-lo).

def test_restricted_purify_output_shape(cbm, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE, step_delta_rel=0.3)
    purified, _ = purifier.purify(cbm, x_adv, n_sweeps=1, device="cpu")
    assert purified.shape == (BATCH_SIZE, DATA_DIM)


def test_restricted_purify_stays_in_input_range(cbm, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE, step_delta_rel=0.3)
    purified, _ = purifier.purify(cbm, x_adv, n_sweeps=1, device="cpu")
    lo, hi = cbm.input_range
    assert (purified >= lo - 1e-5).all()
    assert (purified <= hi + 1e-5).all()


def test_restricted_purify_stays_near_start(cbm):
    # After ONE sweep the window is centred on x_adv, so every purified value must
    # lie in [x_adv_k ± delta], and — since the candidate grid IS the window — must
    # be one of the num_bins grid points of that window, not merely inside it.
    torch.manual_seed(0)
    step_radius = 0.1
    x_adv = torch.full((BATCH_SIZE, DATA_DIM), 0.5)  # well inside input_range [0,1]
    purifier = GibbsPurification(
        num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE, step_delta_rel=step_radius
    )
    purified, _ = purifier.purify(cbm, x_adv, n_sweeps=1, device="cpu")
    lo, hi = cbm.input_range
    delta = step_radius * (hi - lo)

    lo_bound = (x_adv - delta).clamp(lo, hi)
    hi_bound = (x_adv + delta).clamp(lo, hi)
    assert (purified >= lo_bound - 1e-5).all(), "Purified values below lower restriction bound"
    assert (purified <= hi_bound + 1e-5).all(), "Purified values above upper restriction bound"

    # Every sample lands exactly on its own window's local grid.
    expected = lo_bound.unsqueeze(-1) + (hi_bound - lo_bound).unsqueeze(-1) * torch.linspace(
        0.0, 1.0, NUM_BINS
    )
    on_grid = (purified.unsqueeze(-1) - expected).abs().min(dim=-1).values
    assert (on_grid < 1e-5).all(), "Purified value is not a point of its local window grid"


def test_local_grid_uses_full_resolution_inside_window(cbm):
    """The window is discretized with num_bins points, not sliced out of a global grid.

    The old implementation masked a global linspace(lo, hi, num_bins) down to the
    window, leaving only ~2*step_radius*num_bins admissible values and throwing the
    rest of every forward pass away. With a local grid the window carries all
    num_bins values, so the reachable set is strictly larger than the old one.
    """
    import math

    torch.manual_seed(0)
    step_radius = 0.1
    num_bins = 40
    x_adv = torch.full((64, DATA_DIM), 0.5)
    purifier = GibbsPurification(
        num_bins=num_bins, gibbs_batch_size=64, step_delta_rel=step_radius
    )
    purified, _ = purifier.purify(cbm, x_adv, n_sweeps=1, device="cpu")

    # Bins the old global-grid-plus-mask scheme could have offered inside the window.
    old_admissible = math.ceil(2 * step_radius * num_bins) + 1
    distinct = len(torch.unique(purified[:, 0]))
    assert distinct > old_admissible, (
        f"only {distinct} distinct values reachable, no better than the "
        f"{old_admissible} a masked global grid would allow"
    )


def test_k_sweeps_widen_the_envelope(cbm):
    """Purification strength comes from n_sweeps — this is the attack-agnostic property.

    step_radius bounds a single sweep's move; k sweeps compose to at most
    k*step_radius*(hi-lo). Pins BOTH directions: the k-sweep bound holds, and the
    1-sweep bound does NOT (otherwise step_radius would be a global budget and the
    number of sweeps would buy nothing).
    """
    torch.manual_seed(0)
    step_radius = 0.1
    k = 6
    x0 = torch.full((32, DATA_DIM), 0.5)
    purifier = GibbsPurification(
        num_bins=NUM_BINS, gibbs_batch_size=32, step_delta_rel=step_radius
    )
    purified, _ = purifier.purify(cbm, x0, n_sweeps=k, device="cpu")
    lo, hi = cbm.input_range
    delta = step_radius * (hi - lo)
    drift = (purified - x0).abs().max()

    assert drift <= k * delta + 1e-5, "drift exceeded the k-sweep envelope"
    assert drift > delta + 1e-5, (
        "drift never exceeded a single step — the window is not re-centring per "
        "sweep, so n_sweeps is not acting as the strength knob"
    )


def test_restricted_purify_log_px_finite(cbm, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE, step_delta_rel=0.3)
    _, log_px = purifier.purify(cbm, x_adv, n_sweeps=3, device="cpu")
    assert torch.isfinite(log_px).all()


# --- Snapshot purification (one max-sweep run yields several sweep counts) ---

def test_purify_snapshots_keys_and_shapes(cbm):
    n = 6
    x = torch.rand(n, DATA_DIM)
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE)
    snaps = purifier.purify_snapshots(cbm, x, [1, 3], device="cpu")
    assert sorted(snaps.keys()) == [1, 3]
    for k in (1, 3):
        xp, lp = snaps[k]
        assert xp.shape == (n, DATA_DIM)
        assert lp.shape == (n,)
        assert torch.isfinite(lp).all()


def test_purify_snapshots_single_batch_matches_purify(cbm):
    # With one batch (gibbs_batch_size >= n), the RNG stream is identical to dedicated
    # per-sweep runs, so each snapshot is bit-for-bit equal to purify(n_sweeps=k).
    n = 6
    x = torch.rand(n, DATA_DIM)
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=64)
    torch.manual_seed(777)
    snaps = purifier.purify_snapshots(cbm, x, [1, 3], device="cpu")
    for k in (1, 3):
        torch.manual_seed(777)
        xp, lp = purifier.purify(cbm, x, n_sweeps=k, device="cpu")
        assert torch.equal(snaps[k][0], xp)
        assert torch.equal(snaps[k][1], lp)


def test_purify_snapshots_partial_batch_valid(cbm):
    # Across multiple batches the per-sample RNG drifts, but every snapshot must still be
    # a valid k-sweep purification: right shape, finite log p(x), inside the input range.
    n = 5
    x = torch.rand(n, DATA_DIM)
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=3)  # partial final batch
    snaps = purifier.purify_snapshots(cbm, x, [1, 2], device="cpu")
    lo, hi = cbm.input_range
    for k in (1, 2):
        xp, lp = snaps[k]
        assert xp.shape == (n, DATA_DIM)
        assert torch.isfinite(lp).all()
        assert (xp >= lo - 1e-5).all() and (xp <= hi + 1e-5).all()


@pytest.mark.parametrize("step_radius", [None, 0.2])
def test_gibbs_stable_when_amplitudes_overflow(step_radius):
    """Gibbs weights stay correct when the raw |ψ|² would overflow to inf.

    The sweep evaluates a full-chain contraction per candidate, so on a long
    chain raw amplitudes() overflows; the old linear path then produced p=inf and
    draw_from_grid mapped posinf->0, silently zeroing the HIGHEST-probability bins
    so sampling ran backwards. The log-domain path (log_amp_sq + logsumexp) has no
    such regime. Guards against a *backwards* sampler, not just NaN.
    """
    from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig

    data_dim = 60
    cbm = ConditionalBornMachine(
        cfg=CBMConfig(embedding="fourier",
                      init_kwargs=MPSInitConfig(in_dim=2, bond_dim=4, std=0.3)),
        data_dim=data_dim, num_classes=2, device=torch.device("cpu"))
    with torch.no_grad():
        for node in cbm._mats_env:
            node.tensor.data.mul_(8.0)
    cbm.prepare(device=torch.device("cpu")); cbm.eval(); cbm.cache_log_Z()

    lo, hi = cbm.input_range
    # precondition: the linear path this replaced really is broken here
    x_probe = lo + (hi - lo) * torch.rand(2, data_dim)
    assert not torch.isfinite(cbm.amplitudes(x_probe)).all(), \
        "fixture no longer overflows; the regime this guards is untested"

    torch.manual_seed(0)
    x_adv = lo + (hi - lo) * torch.rand(6, data_dim)
    # step_delta_rel=None also keeps the global fixed-grid path under test.
    purifier = GibbsPurification(num_bins=8, gibbs_batch_size=4, step_delta_rel=step_radius)
    torch.manual_seed(1)
    # n_sweeps=1 so the restriction (centred on the sweep-start snapshot) is
    # well-defined relative to x_adv; across k sweeps x_cur may drift up to
    # k*step_radius from the original, so a single-step bound would not hold.
    xp, lp = purifier.purify(cbm, x_adv, n_sweeps=1, device="cpu")

    assert xp.shape == x_adv.shape
    assert torch.isfinite(xp).all()
    assert (xp >= lo).all() and (xp <= hi).all()
    assert len(torch.unique(xp)) > 1, "sampler collapsed to a single bin"
    if step_radius is not None:
        assert (xp - x_adv).abs().max() <= step_radius * (hi - lo) + 1e-6


def test_gibbs_numeric_regression():
    """Pin Gibbs output so future changes can't silently alter it.

    Checked for both a batch size that divides n_samples and one that leaves a partial
    final batch.

    Goldens re-pinned 2026-07-27 for the local-grid change: the restricted sweep now
    discretizes each sample's own [x̄ ± delta] window with num_bins points instead of
    masking a global linspace(lo, hi, num_bins) down to it. That intentionally changes
    which values are reachable, so the previous goldens
    ({3: (-4.5714287758, -10.9865303040), 4: (-2.0, -9.3423662186)}) no longer apply.
    """
    from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig

    def make_cbm():
        torch.manual_seed(0)
        m = ConditionalBornMachine(
            cfg=CBMConfig(embedding="legendre",
                          init_kwargs=MPSInitConfig(in_dim=2, bond_dim=3, std=1e-1)),
            data_dim=4, num_classes=3, device=torch.device("cpu"),
        )
        m.prepare(device=torch.device("cpu")); m.eval(); m.cache_log_Z()
        return m

    golden = {3: (-0.4685872495, -13.2247095108), 4: (-2.3235769272, -13.2589607239)}
    for bs, (xp_sum, lp_sum) in golden.items():
        cbm = make_cbm()
        torch.manual_seed(123)
        x_adv = -1 + 2 * torch.rand(6, 4)
        purifier = GibbsPurification(num_bins=8, gibbs_batch_size=bs, step_delta_rel=0.2)
        torch.manual_seed(777)
        xp, lp = purifier.purify(cbm, x_adv, n_sweeps=2, device="cpu")
        assert xp.sum().item() == pytest.approx(xp_sum, abs=1e-5)
        assert lp.sum().item() == pytest.approx(lp_sum, abs=1e-4)
