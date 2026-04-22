import pytest
import torch
from tests.conftest import DATA_DIM, NUM_CLASSES
import src.utils.schemas as schemas

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def gen(born_machine):
    return born_machine.generator


def test_cls_pos_matches_classifier_out_position(born_machine):
    assert born_machine.generator.cls_pos == born_machine.classifier.out_position


def test_n_features(born_machine):
    assert born_machine.generator.n_features == DATA_DIM + 1


def test_prepare_does_not_raise(gen):
    gen.prepare()


def test_sequential_marginal_shape(gen):
    gen.prepare()
    num_bins = 10
    bs = 4
    input_space = torch.linspace(0.0, 1.0, num_bins)
    in_emb = gen.embedding(input_space)  # (num_bins, in_dim)
    in_emb_batch = in_emb[None, :, :].expand(bs, -1, -1).reshape(bs * num_bins, -1)
    cls_pos = gen.cls_pos
    # Provide all data sites except the last data site to trigger marginalisation
    data_sites = [s for s in range(gen.n_features) if s != cls_pos]
    embs = {}
    for s in data_sites[:-1]:  # omit the last data site → it is marginalised out
        embs[s] = in_emb_batch
    p = gen.sequential(embs)
    # sequential returns flat (bs*num_bins,) which _single_class reshapes to (bs, num_bins)
    assert p.shape == (bs * num_bins,)


def test_sequential_raises_on_tensor_input(gen):
    gen.prepare()
    with pytest.raises(TypeError):
        gen.sequential(torch.rand(4, 4))


def test_virtual_mps_shares_parameters(born_machine):
    gen = born_machine.generator
    for vmain, vvirt in zip(gen._mats_env, gen.virtual_mps._mats_env):
        assert vmain.tensor.data_ptr() == vvirt.tensor.data_ptr()


def test_sample_single_class_shape(born_machine):
    cfg = schemas.SamplingConfig(method="multinomial", num_spc=4, num_bins=10, batch_spc=8)
    samples = born_machine.sample(cfg, cls=0)
    assert samples.shape == (4, DATA_DIM)


def test_sample_single_class_in_range(born_machine):
    cfg = schemas.SamplingConfig(method="multinomial", num_spc=4, num_bins=10, batch_spc=8)
    samples = born_machine.sample(cfg, cls=0)
    lo, hi = born_machine.input_range
    assert (samples >= lo - 1e-5).all()
    assert (samples <= hi + 1e-5).all()


def test_two_sequential_calls_with_prepare(gen):
    num_bins = 8
    bs = 2
    input_space = torch.linspace(0.0, 1.0, num_bins)
    in_emb = gen.embedding(input_space)
    in_emb_batch = in_emb[None, :, :].expand(bs, -1, -1).reshape(bs * num_bins, -1)
    cls_pos = gen.cls_pos
    embs = {s: in_emb_batch for s in range(gen.n_features) if s != cls_pos}
    for _ in range(2):
        gen.prepare()
        p = gen.sequential(embs)
        assert p.shape[0] == bs * num_bins
