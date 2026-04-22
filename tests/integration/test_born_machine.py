import pytest
import torch
from tests.conftest import DATA_DIM, NUM_CLASSES


pytestmark = pytest.mark.slow


def test_bm_has_classifier_and_generator(born_machine):
    assert hasattr(born_machine, "classifier")
    assert hasattr(born_machine, "generator")


def test_bm_dtype_complex64(born_machine):
    for t in born_machine.classifier.tensors:
        assert t.dtype == torch.complex64


def test_bm_input_range_fourier(born_machine):
    lo, hi = born_machine.input_range
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(1.0)


def test_bm_embedding_name(born_machine):
    assert born_machine.embedding_name == "fourier"


def test_cache_log_z_attribute_finite(born_machine):
    assert born_machine._log_Z is not None
    assert torch.isfinite(torch.tensor(born_machine._log_Z))


def test_marginal_log_prob_shape(born_machine, x_batch):
    log_px = born_machine.marginal_log_probability(x_batch)
    assert log_px.shape == (x_batch.shape[0],)


def test_marginal_log_prob_finite(born_machine, x_batch):
    log_px = born_machine.marginal_log_probability(x_batch)
    assert torch.isfinite(log_px).all()


def test_class_probs_shape(born_machine, x_batch):
    probs = born_machine.class_probabilities(x_batch)
    assert probs.shape == (x_batch.shape[0], NUM_CLASSES)


def test_class_probs_sum_to_one(born_machine, x_batch):
    probs = born_machine.class_probabilities(x_batch)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(x_batch.shape[0]), atol=1e-5)


def test_class_probs_nonnegative(born_machine, x_batch):
    probs = born_machine.class_probabilities(x_batch)
    assert (probs >= 0).all()


def test_class_probs_in_01(born_machine, x_batch):
    probs = born_machine.class_probabilities(x_batch)
    assert (probs >= 0).all() and (probs <= 1 + 1e-5).all()


def test_sync_tensors_classification(born_machine):
    born_machine.sync_tensors(after="classification")
    for t_cls, t_gen in zip(born_machine.classifier.tensors, born_machine.generator.tensors):
        assert torch.allclose(t_cls, t_gen, atol=1e-6)


def test_save_load_roundtrip(born_machine, tmp_path, x_batch):
    path = str(tmp_path / "bm_test.pt")
    born_machine.save(path)
    from src.models.born import BornMachine
    loaded = BornMachine.load(path)
    probs_orig = born_machine.class_probabilities(x_batch)
    probs_loaded = loaded.class_probabilities(x_batch)
    assert torch.allclose(probs_orig, probs_loaded, atol=1e-5)


def test_save_load_preserves_dtype(born_machine, tmp_path):
    path = str(tmp_path / "bm_dtype.pt")
    born_machine.save(path)
    from src.models.born import BornMachine
    loaded = BornMachine.load(path)
    assert loaded.dtype == torch.complex64


def test_to_cpu(born_machine):
    born_machine.to("cpu")


def test_eval_mode(born_machine):
    born_machine.eval()


def test_train_mode(born_machine):
    born_machine.train()
