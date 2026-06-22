import pytest
import torch
from src.model import CBMConfig, ConditionalBornMachine, MPSInitConfig

pytestmark = pytest.mark.slow

DATA_DIM = 4
NUM_CLASSES = 2


@pytest.fixture(scope="session")
def cbm():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "embedding": "fourier",
        "model_path": None,
        "init_kwargs": {
            "in_dim": 4,
            "bond_dim": 2,
            "out_position": 2,
            "boundary": "obc",
            "init_method": "randn",
            "dtype": "complex64",
            "n_features": None,
            "out_dim": None,
            "std": 1e-3,
        },
    })
    model = ConditionalBornMachine(cfg=cfg, data_dim=DATA_DIM, num_classes=NUM_CLASSES, device="cpu")
    model.cache_log_Z()
    return model


@pytest.fixture
def x_batch():
    return torch.rand(8, DATA_DIM)


@pytest.fixture
def y_batch():
    return torch.randint(0, NUM_CLASSES, (8,))


# ── Structural ─────────────────────────────────────────────────────────────

def test_cbm_dtype_complex64(cbm):
    for t in cbm.tensors:
        assert t.dtype == torch.complex64


def test_cbm_input_range(cbm):
    lo, hi = cbm.input_range
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(1.0)


def test_cbm_embedding_name(cbm):
    assert cbm.embedding_name == "fourier"


def test_norm_net_tensors_shared(cbm):
    for i, (aux, main) in enumerate(zip(cbm.norm_net._mats_env, cbm._mats_env)):
        assert aux.tensor is main.tensor, f"norm_net tensor {i} not shared with main"


# ── Amplitudes / probabilities ────────────────────────────────────────────

def test_amplitudes_shape(cbm, x_batch):
    amps = cbm.amplitudes(x_batch)
    assert amps.shape == (x_batch.shape[0], NUM_CLASSES)


def test_class_probabilities_shape(cbm, x_batch):
    probs = cbm.class_probabilities(x_batch)
    assert probs.shape == (x_batch.shape[0], NUM_CLASSES)


def test_class_probabilities_sum_to_one(cbm, x_batch):
    probs = cbm.class_probabilities(x_batch)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(x_batch.shape[0]), atol=1e-5)


def test_class_probabilities_nonneg(cbm, x_batch):
    probs = cbm.class_probabilities(x_batch)
    assert (probs >= 0).all()


# ── Partition function ────────────────────────────────────────────────────

def test_log_partition_function_finite(cbm):
    cbm.reset()
    log_Z = cbm.log_partition_function()
    assert log_Z.isfinite()


def test_log_partition_product_state_analytic():
    """bond_dim=1 product state: Z = Π_i ‖t_i‖²  →  log Z = Σ_i log‖t_i‖².

    Independent analytic value for the zip-up contraction (bonds are trivial,
    so the chain factorises over sites)."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "embedding": "fourier", "model_path": None,
        "init_kwargs": {
            "in_dim": 2, "bond_dim": 1, "out_position": 1, "boundary": "obc",
            "init_method": "randn", "dtype": "float32", "n_features": None,
            "out_dim": None, "std": 1.0,
        },
    })
    m = ConditionalBornMachine(cfg=cfg, data_dim=2, num_classes=2, device="cpu")
    m.reset()
    log_Z = m.log_partition_function()
    expected = sum(t.abs().square().sum().log() for t in m.tensors)
    assert torch.allclose(log_Z, expected, atol=1e-5)


def test_log_partition_reentrant_stable_complex(cbm):
    """Calling log_partition_function repeatedly without reset() (the training
    path) must be stable for complex models. Regression for the reuse-path
    conj bug that computed Σψ² instead of Σ|ψ|² after the first call."""
    cbm.reset()
    ref = cbm.log_partition_function()        # reset → create-copies path
    again = cbm.log_partition_function()       # no reset → reuse path
    third = cbm.log_partition_function()
    assert torch.allclose(again, ref, atol=1e-5)
    assert torch.allclose(third, ref, atol=1e-5)


def test_cache_log_Z_attribute_finite(cbm):
    assert cbm._log_Z is not None
    assert torch.isfinite(torch.tensor(cbm._log_Z))


# ── Marginal log prob ──────────────────────────────────────────────────────

def test_marginal_log_prob_shape(cbm, x_batch):
    log_px = cbm.marginal_log_probability(x_batch)
    assert log_px.shape == (x_batch.shape[0],)


def test_marginal_log_prob_finite(cbm, x_batch):
    log_px = cbm.marginal_log_probability(x_batch)
    assert log_px.isfinite().all()


# ── mixed_nll ─────────────────────────────────────────────────────────────

def test_mixed_nll_alpha0_finite(cbm, x_batch, y_batch):
    cbm.reset()
    loss = cbm.mixed_nll(x_batch, y_batch, alpha=0.0)
    assert loss.ndim == 0 and loss.isfinite()


def test_mixed_nll_alpha05_finite(cbm, x_batch, y_batch):
    cbm.reset()
    loss = cbm.mixed_nll(x_batch, y_batch, alpha=0.5)
    assert loss.ndim == 0 and loss.isfinite()


def test_mixed_nll_alpha1_finite(cbm, x_batch, y_batch):
    cbm.reset()
    loss = cbm.mixed_nll(x_batch, y_batch, alpha=1.0)
    assert loss.ndim == 0 and loss.isfinite()


# ── renormalize_ ──────────────────────────────────────────────────────────

def test_renormalize_norm_near_zero(cbm):
    cbm.reset()
    cbm.renormalize_(log_target=0.0)
    cbm.reset()
    log_Z = cbm.log_partition_function()
    assert log_Z.abs() < 0.1, f"log_Z after renormalize_ should be ≈ 0, got {log_Z.item():.4f}"


# ── save / load ───────────────────────────────────────────────────────────

def test_save_load_roundtrip(cbm, x_batch, tmp_path):
    path = str(tmp_path / "cbm_test.pt")
    cbm.save(path)
    loaded = ConditionalBornMachine.load(path)
    probs_orig = cbm.class_probabilities(x_batch)
    probs_loaded = loaded.class_probabilities(x_batch)
    assert torch.allclose(probs_orig, probs_loaded, atol=1e-5)


def test_save_load_no_regime_field(cbm, tmp_path):
    path = str(tmp_path / "cbm_regime.pt")
    cbm.save(path)
    ckpt = torch.load(path, weights_only=False)
    assert "regime" not in ckpt


def test_save_load_preserves_dtype(cbm, tmp_path):
    path = str(tmp_path / "cbm_dtype.pt")
    cbm.save(path)
    loaded = ConditionalBornMachine.load(path)
    assert loaded.dtype == torch.complex64


# ── Device / mode ──────────────────────────────────────────────────────────

def test_to_cpu(cbm):
    cbm.to("cpu")


def test_eval_mode(cbm):
    cbm.eval()


def test_train_mode(cbm):
    cbm.train()
