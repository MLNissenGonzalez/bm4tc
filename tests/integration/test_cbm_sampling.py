import pytest
import torch
from src.model import CBMConfig, ConditionalBornMachine, MPSInitConfig

pytestmark = pytest.mark.slow

DATA_DIM = 4
NUM_CLASSES = 2
N_SAMPLES = 6
NUM_BINS = 20


@pytest.fixture(scope="session")
def cbm():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "embedding": "fourier",
        "model_path": None,
        "init_kwargs": {
            "in_dim": 4,
            "bond_dim": 3,
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
    model.prepare(device="cpu")
    model.eval()
    return model


# ── condition_on_class ──────────────────────────────────────────────────────

def test_condition_on_class_returns_data_dim_tensors(cbm):
    tensors = cbm.condition_on_class(0)
    assert len(tensors) == DATA_DIM


def test_condition_on_class_tensors_are_3d(cbm):
    tensors = cbm.condition_on_class(0)
    for i, t in enumerate(tensors):
        assert t.ndim == 3, f"tensor {i} has {t.ndim} dims, expected 3"


def test_condition_on_class_does_not_mutate_mats_env(cbm):
    snapshots = [node.tensor.clone() for node in cbm._mats_env]
    cbm.condition_on_class(0)
    cbm.condition_on_class(1)
    for i, (snap, node) in enumerate(zip(snapshots, cbm._mats_env)):
        assert torch.allclose(snap, node.tensor), f"_mats_env[{i}] was mutated"


# ── sample ──────────────────────────────────────────────────────────────────

def test_sample_shape(cbm):
    samples = cbm.sample(class_idx=0, n=N_SAMPLES, num_bins=NUM_BINS)
    assert samples.shape == (N_SAMPLES, DATA_DIM)


def test_sample_dtype_float(cbm):
    samples = cbm.sample(class_idx=0, n=N_SAMPLES, num_bins=NUM_BINS)
    assert samples.is_floating_point()


def test_sample_in_range(cbm):
    lo, hi = cbm.input_range
    for c in range(NUM_CLASSES):
        samples = cbm.sample(class_idx=c, n=N_SAMPLES, num_bins=NUM_BINS)
        assert (samples >= lo).all() and (samples <= hi).all(), (
            f"class {c}: samples outside [{lo}, {hi}]"
        )


def test_sample_finite(cbm):
    for c in range(NUM_CLASSES):
        samples = cbm.sample(class_idx=c, n=N_SAMPLES, num_bins=NUM_BINS)
        assert torch.isfinite(samples).all(), f"class {c}: non-finite samples"


def test_sample_batch_larger_than_batch_size(cbm):
    # n > batch_size forces chunked sampling path
    samples = cbm.sample(class_idx=0, n=10, num_bins=NUM_BINS, batch_size=3)
    assert samples.shape == (10, DATA_DIM)
    assert torch.isfinite(samples).all()


def test_sample_does_not_mutate_mats_env(cbm):
    snapshots = [node.tensor.clone() for node in cbm._mats_env]
    cbm.sample(class_idx=0, n=N_SAMPLES, num_bins=NUM_BINS)
    cbm.sample(class_idx=1, n=N_SAMPLES, num_bins=NUM_BINS)
    for i, (snap, node) in enumerate(zip(snapshots, cbm._mats_env)):
        assert torch.allclose(snap, node.tensor), f"_mats_env[{i}] was mutated by sample()"


# ── sample_all_classes ──────────────────────────────────────────────────────

def test_sample_all_classes_shapes(cbm):
    n_per = 5
    samples, labels = cbm.sample_all_classes(n_per_class=n_per, num_bins=NUM_BINS)
    assert samples.shape == (n_per * NUM_CLASSES, DATA_DIM)
    assert labels.shape == (n_per * NUM_CLASSES,)


def test_sample_all_classes_label_values(cbm):
    n_per = 4
    _, labels = cbm.sample_all_classes(n_per_class=n_per, num_bins=NUM_BINS)
    for c in range(NUM_CLASSES):
        assert (labels == c).sum() == n_per, f"class {c} count mismatch"


def test_sample_all_classes_in_range(cbm):
    lo, hi = cbm.input_range
    samples, _ = cbm.sample_all_classes(n_per_class=4, num_bins=NUM_BINS)
    assert (samples >= lo).all() and (samples <= hi).all()
