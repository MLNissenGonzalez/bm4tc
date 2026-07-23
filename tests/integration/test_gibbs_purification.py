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


# --- Restricted Gibbs (radius) ---

def test_restricted_purify_output_shape(cbm, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE, radius=0.3)
    purified, _ = purifier.purify(cbm, x_adv, n_sweeps=1, device="cpu")
    assert purified.shape == (BATCH_SIZE, DATA_DIM)


def test_restricted_purify_stays_in_input_range(cbm, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE, radius=0.3)
    purified, _ = purifier.purify(cbm, x_adv, n_sweeps=1, device="cpu")
    lo, hi = cbm.input_range
    assert (purified >= lo - 1e-5).all()
    assert (purified <= hi + 1e-5).all()


def test_restricted_purify_stays_near_start(cbm):
    # With a small radius, purified values must be within radius of x_adv
    # (per feature, since each feature is sampled from [x_adv_k ± delta]).
    torch.manual_seed(0)
    radius = 0.1
    x_adv = torch.full((BATCH_SIZE, DATA_DIM), 0.5)  # well inside input_range [0,1]
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE, radius=radius)
    purified, _ = purifier.purify(cbm, x_adv, n_sweeps=1, device="cpu")
    lo, hi = cbm.input_range
    delta = radius * (hi - lo)
    # Each feature must stay within [x_adv_k - delta, x_adv_k + delta] ∩ [lo, hi]
    lo_bound = (x_adv - delta).clamp(lo, hi)
    hi_bound = (x_adv + delta).clamp(lo, hi)
    assert (purified >= lo_bound - 1e-5).all(), "Purified values below lower restriction bound"
    assert (purified <= hi_bound + 1e-5).all(), "Purified values above upper restriction bound"


def test_restricted_purify_log_px_finite(cbm, x_adv):
    purifier = GibbsPurification(num_bins=NUM_BINS, gibbs_batch_size=BATCH_SIZE, radius=0.3)
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


@pytest.mark.parametrize("radius", [None, 0.2])
def test_gibbs_stable_when_amplitudes_overflow(radius):
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
    purifier = GibbsPurification(num_bins=8, gibbs_batch_size=4, radius=radius)
    torch.manual_seed(1)
    # n_sweeps=1 so the radius restriction (centred on the sweep-start snapshot)
    # is well-defined relative to x_adv; across multiple sweeps x_cur may drift up
    # to n_sweeps*radius from the original, so a single-radius bound wouldn't hold.
    xp, lp = purifier.purify(cbm, x_adv, n_sweeps=1, device="cpu")

    assert xp.shape == x_adv.shape
    assert torch.isfinite(xp).all()
    assert (xp >= lo).all() and (xp <= hi).all()
    assert len(torch.unique(xp)) > 1, "sampler collapsed to a single bin"
    if radius is not None:
        assert (xp - x_adv).abs().max() <= radius * (hi - lo) + 1e-6


def test_gibbs_numeric_regression():
    """Pin Gibbs output so future changes (e.g. the reset hoist) can't silently alter it.

    Golden values captured from the per-feature-reset implementation; reproduced exactly
    by the current per-batch-reset code for both a batch size that divides n_samples and
    one that leaves a partial final batch.
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

    golden = {3: (-4.5714287758, -10.9865303040), 4: (-2.0000000000, -9.3423662186)}
    for bs, (xp_sum, lp_sum) in golden.items():
        cbm = make_cbm()
        torch.manual_seed(123)
        x_adv = -1 + 2 * torch.rand(6, 4)
        purifier = GibbsPurification(num_bins=8, gibbs_batch_size=bs, radius=0.2)
        torch.manual_seed(777)
        xp, lp = purifier.purify(cbm, x_adv, n_sweeps=2, device="cpu")
        assert xp.sum().item() == pytest.approx(xp_sum, abs=1e-5)
        assert lp.sum().item() == pytest.approx(lp_sum, abs=1e-4)
