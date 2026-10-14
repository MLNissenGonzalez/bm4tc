"""SamplingMPS (D100): tensorkrowch develop's ``condition`` and ``sample`` on 1.1.6.

The first part is develop's own tests (``tests/models/test_mps.py`` at ``2e23da9``),
ported: every local distribution the sampler draws from is checked against the exact
one, enumerated over all configurations. ``torch.multinomial`` is replaced by argmax
that records its input, so a test sees each distribution and the path is
deterministic. The second part checks it on the ConditionalBornMachine: the class
is summed out or fixed, and ``sample(in_condition=x)`` is one sweep of
GibbsPurification on a fixed grid.
"""
import itertools

import pytest
import torch

import tensorkrowch as tk

from bm4tc.core.sampling import SamplingMPS

BOUNDARY_CASES = ['obc', 'pbc']


@pytest.fixture(autouse=True)
def _seed():
    # The MPS are random and the checks float32: over 600 random draws the worst
    # conditional was 9.4e-6 off, so an unseeded run can now and then cross the
    # 1e-5 tolerance (it did once, in the full suite). Seeded, every run is the same.
    torch.manual_seed(0)


def _basis_data_for_configs(mps, configs, phys_dims):
    data = [
        tk.embeddings.basis(configs[:, i], dim=dim)
        .to(mps.mats_env[0].dtype)
        .to(mps.mats_env[0].device)
        for i, dim in enumerate(phys_dims)
    ]
    if len(set(phys_dims)) == 1:
        return torch.stack(data, dim=1)
    return data


def _exact_input_distribution(mps):
    phys_dims = [mps.phys_dim[i] for i in mps.in_features]
    configs = torch.tensor(
        list(itertools.product(*[range(dim) for dim in phys_dims])),
        device=mps.mats_env[0].device,
        dtype=torch.long)

    data = _basis_data_for_configs(mps, configs, phys_dims)
    result = mps(data, marginalize_output=bool(mps.out_features))

    if mps.out_features:
        probs = torch.diagonal(result).real
    else:
        probs = result.abs().pow(2)
    probs = probs / probs.sum()

    mps.reset()
    if mps.data_nodes:
        mps.unset_data_nodes()

    return configs, probs


def _conditional_probs_from_exact(configs, probs, step, prefix, phys_dim,
                                  condition=None):
    mask = torch.ones(configs.shape[0], device=configs.device,
                      dtype=torch.bool)
    for i, value in enumerate(prefix):
        mask &= configs[:, i] == value

    if condition is not None:
        for i in range(step + 1, configs.shape[1]):
            mask &= configs[:, i] == condition[i]

    expected = []
    for value in range(phys_dim):
        value_mask = mask & (configs[:, step] == value)
        expected.append(probs[value_mask].sum())
    expected = torch.stack(expected)
    return expected / expected.sum()


@pytest.fixture
def captured(monkeypatch):
    """The input of every ``torch.multinomial`` call; draws become argmax."""
    calls = []

    def fake_multinomial(input, num_samples, replacement=False, *,
                         generator=None, out=None):
        calls.append(input.detach().clone())
        return input.argmax(dim=1, keepdim=True)

    monkeypatch.setattr(torch, 'multinomial', fake_multinomial)
    return calls


# ── develop's tests ──────────────────────────────────────────────────────────

@pytest.mark.parametrize('boundary', BOUNDARY_CASES)
@pytest.mark.parametrize('dtype', [None, torch.complex64])
@pytest.mark.parametrize('renormalize', [True, False])
def test_sample_matches_exact_local_probabilities(captured, boundary, dtype,
                                                  renormalize):
    kwargs = {} if dtype is None else {'dtype': dtype}
    phys_dim = [2, 3, 2]
    bond_dim = [2, 3] if boundary == 'obc' else [2, 3, 4]
    mps = SamplingMPS(n_features=3, phys_dim=phys_dim, bond_dim=bond_dim,
                      boundary=boundary, **kwargs)

    configs, probs = _exact_input_distribution(mps)
    _, indices = mps.sample(n_samples=1, renormalize=renormalize,
                            return_indices=True)

    prefix = []
    for step, local_probs in enumerate(captured):
        expected = _conditional_probs_from_exact(
            configs, probs, step, prefix, phys_dim[step])
        assert torch.allclose(local_probs[0], expected, atol=1e-5, rtol=1e-5)
        prefix.append(indices[0, step].item())


@pytest.mark.parametrize('boundary', BOUNDARY_CASES)
def test_sample_marginalizes_output_nodes_exactly(captured, boundary):
    mps = SamplingMPS(n_features=3, phys_dim=2, bond_dim=2,
                      boundary=boundary, out_features=[1])

    configs, probs = _exact_input_distribution(mps)
    _, indices = mps.sample(n_samples=1, return_indices=True)

    prefix = []
    for step, local_probs in enumerate(captured):
        expected = _conditional_probs_from_exact(configs, probs, step,
                                                 prefix, 2)
        assert torch.allclose(local_probs[0], expected, atol=1e-5, rtol=1e-5)
        prefix.append(indices[0, step].item())


@pytest.mark.parametrize('boundary', BOUNDARY_CASES)
def test_sample_with_in_condition_matches_exact_conditionals(captured,
                                                             boundary):
    mps = SamplingMPS(n_features=3, phys_dim=2, bond_dim=2,
                      boundary=boundary)
    condition_ids = torch.tensor([[1, 0, 1],
                                  [0, 1, 0]])
    condition = tk.embeddings.basis(condition_ids, dim=2)\
        .to(mps.mats_env[0].dtype)

    configs, probs = _exact_input_distribution(mps)
    _, indices = mps.sample(in_condition=condition, return_indices=True)

    prefixes = [[] for _ in range(condition_ids.shape[0])]
    for step, local_probs in enumerate(captured):
        for batch in range(condition_ids.shape[0]):
            expected = _conditional_probs_from_exact(
                configs, probs, step, prefixes[batch], 2,
                condition=condition_ids[batch])
            assert torch.allclose(local_probs[batch], expected,
                                  atol=1e-5, rtol=1e-5)
            prefixes[batch].append(indices[batch, step].item())


def test_sample_canonical_matches_right_environments(captured):
    mps = SamplingMPS(n_features=4, phys_dim=2, bond_dim=2, boundary='obc')
    mps.canonicalize(oc=0, mode='svd')

    mps.sample(n_samples=2, canonical=False)
    environment_probs = [probs.clone() for probs in captured]
    captured.clear()
    mps.sample(n_samples=2, canonical=True)
    canonical_probs = [probs.clone() for probs in captured]

    assert len(environment_probs) == len(canonical_probs)
    for probs_env, probs_canonical in zip(environment_probs, canonical_probs):
        assert torch.allclose(probs_env, probs_canonical, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    'boundary,sample_kwargs',
    [('pbc', {}),
     ('obc', {'in_condition': torch.ones(1, 3, 2)}),
     ('obc', {'embedding_matrices': torch.eye(2)}),
     ('obc', {'build_matrices': True})])
def test_sample_ignores_incompatible_canonical(boundary, sample_kwargs):
    mps = SamplingMPS(n_features=3, phys_dim=2, bond_dim=2, boundary=boundary)
    with pytest.warns(UserWarning, match='`canonical`.*ignored'):
        samples = mps.sample(n_samples=2, canonical=True, **sample_kwargs)
    assert samples.shape == (2, 3)


def test_sample_rejects_mps_without_input_nodes():
    mps = SamplingMPS(n_features=3, phys_dim=2, bond_dim=2, in_features=[])
    with pytest.raises(ValueError, match='no input nodes'):
        mps.sample(n_samples=2)


def test_sample_build_matrices_matches_basis_metric(captured):
    mps = SamplingMPS(n_features=3, phys_dim=2, bond_dim=2, boundary='obc')
    domain = torch.arange(2)

    mps.sample(n_samples=1, domain=domain, embedding=tk.embeddings.basis,
               build_matrices=False)
    identity_probs = [probs.clone() for probs in captured]
    captured.clear()
    mps.sample(n_samples=1, domain=domain, embedding=tk.embeddings.basis,
               build_matrices=True)
    built_probs = [probs.clone() for probs in captured]

    assert len(identity_probs) == len(built_probs)
    for probs_id, probs_built in zip(identity_probs, built_probs):
        assert torch.allclose(probs_id, probs_built)


def test_sample_with_continuous_embedding():
    mps = SamplingMPS(n_features=2, phys_dim=2, bond_dim=2, boundary='obc')
    domain = torch.linspace(0, 1, 16)
    samples = mps.sample(n_samples=8, domain=domain,
                         embedding=tk.embeddings.unit, build_matrices=True)
    assert samples.shape == (8, 2)
    assert torch.isin(samples, domain).all()


def test_sample_keeps_mps_usable():
    mps = SamplingMPS(n_features=3, phys_dim=2, bond_dim=2, boundary='obc')
    _ = mps.sample(n_samples=4)
    assert not mps.resultant_nodes
    assert not mps.data_nodes

    data = tk.embeddings.basis(torch.randint(0, 2, (5, 3)), dim=2)\
        .to(mps.mats_env[0].dtype)
    result = mps(data)
    assert result.shape == (5,)


@pytest.mark.parametrize('out_position', [0, 1, 3])
def test_condition_matches_fixed_output(out_position):
    # develop compares against MPSLayer, whose 1.1.6 version has no condition.
    label = 1
    phys_dim = [2] * 4
    phys_dim[out_position] = 3
    mps = SamplingMPS(n_features=4, phys_dim=phys_dim, bond_dim=2,
                      boundary='obc', out_features=[out_position])
    label_data = tk.embeddings.basis(torch.tensor(label), dim=3)\
        .to(mps.mats_env[0].dtype)

    cond_mps = mps.condition(label_data)
    configs = torch.tensor(list(itertools.product(range(2), repeat=3)))
    data = tk.embeddings.basis(configs, dim=2).to(mps.mats_env[0].dtype)

    assert isinstance(cond_mps, SamplingMPS)
    assert torch.allclose(cond_mps(data), mps(data)[:, label])


@pytest.mark.parametrize('boundary', BOUNDARY_CASES)
def test_condition_multiple_output_nodes(boundary):
    mps = SamplingMPS(n_features=5, phys_dim=2, bond_dim=2,
                      boundary=boundary, out_features=[1, 3])
    out_values = [1, 0]
    out_data = [tk.embeddings.basis(torch.tensor(value), dim=2)
                .to(mps.mats_env[0].dtype) for value in out_values]

    cond_mps = mps.condition(out_data)
    configs = torch.tensor(list(itertools.product(range(2), repeat=3)))
    data = tk.embeddings.basis(configs, dim=2).to(mps.mats_env[0].dtype)

    original_output = mps(data)[:, out_values[0], out_values[1]]
    assert torch.allclose(cond_mps(data), original_output)


def test_condition_pbc_output_regions_at_both_ends():
    mps = SamplingMPS(n_features=5, phys_dim=2, bond_dim=2, boundary='pbc',
                      out_features=[0, 3, 4])
    out_values = [1, 0, 1]
    out_data = [tk.embeddings.basis(torch.tensor(value), dim=2)
                .to(mps.mats_env[0].dtype) for value in out_values]

    cond_mps = mps.condition(out_data)
    configs = torch.tensor(list(itertools.product(range(2), repeat=2)))
    data = tk.embeddings.basis(configs, dim=2).to(mps.mats_env[0].dtype)

    original_output = mps(data)[:, out_values[0], out_values[1],
                                out_values[2]]
    assert cond_mps.boundary == 'pbc'
    assert torch.allclose(cond_mps(data), original_output)


@pytest.mark.parametrize(
    'layout',
    ['restricted_tensor', 'full_tensor', 'restricted_tensor_batch',
     'full_tensor_batch', 'full_list', 'restricted_list_batch',
     'full_list_batch'])
def test_condition_accepts_data_layouts(layout):
    mps = SamplingMPS(n_features=4, phys_dim=2, bond_dim=2,
                      out_features=[1, 3])
    out_values = torch.tensor([1, 0])
    all_values = torch.tensor([0, 1, 1, 0])

    if layout == 'restricted_tensor':
        condition_data = tk.embeddings.basis(out_values, dim=2)
    elif layout == 'full_tensor':
        condition_data = tk.embeddings.basis(all_values, dim=2)
    elif layout == 'restricted_tensor_batch':
        condition_data = tk.embeddings.basis(out_values.unsqueeze(0), dim=2)
    elif layout == 'full_tensor_batch':
        condition_data = tk.embeddings.basis(all_values.unsqueeze(0), dim=2)
    else:
        values = out_values if layout == 'restricted_list_batch' \
            else all_values
        batch = layout.endswith('_batch')
        condition_data = [
            tk.embeddings.basis(value.unsqueeze(0) if batch else value, dim=2)
            for value in values]

    if isinstance(condition_data, torch.Tensor):
        condition_data = condition_data.to(mps.mats_env[0].dtype)
    else:
        condition_data = [tensor.to(mps.mats_env[0].dtype)
                          for tensor in condition_data]

    cond_mps = mps.condition(condition_data)
    configs = torch.tensor(list(itertools.product(range(2), repeat=2)))
    data = tk.embeddings.basis(configs, dim=2).to(mps.mats_env[0].dtype)

    original_output = mps(data)[:, out_values[0], out_values[1]]
    assert torch.allclose(cond_mps(data), original_output)


@pytest.mark.parametrize('data', [torch.randn(2, 2, 2),
                                  [torch.randn(2, 2), torch.randn(1, 2)]])
def test_condition_rejects_batch_size_greater_than_one(data):
    mps = SamplingMPS(n_features=4, phys_dim=2, bond_dim=2,
                      out_features=[1, 3])
    with pytest.raises(ValueError, match='batch size 1'):
        mps.condition(data)


def test_condition_then_sample_matches_exact_distribution(captured):
    mps = SamplingMPS(n_features=3, phys_dim=2, bond_dim=2, boundary='obc',
                      out_features=[1])
    data = tk.embeddings.basis(torch.tensor(1), dim=2)\
        .to(mps.mats_env[0].dtype)
    cond_mps = mps.condition(data)

    configs, probs = _exact_input_distribution(cond_mps)
    _, indices = cond_mps.sample(n_samples=1, return_indices=True)

    prefix = []
    for step, local_probs in enumerate(captured):
        expected = _conditional_probs_from_exact(configs, probs, step,
                                                 prefix, 2)
        assert torch.allclose(local_probs[0], expected, atol=1e-5, rtol=1e-5)
        prefix.append(indices[0, step].item())


def test_sample_uses_trapezoidal_point_masses(captured):
    mps = SamplingMPS(n_features=1, phys_dim=1, bond_dim=2, boundary='obc')
    domain = torch.tensor([0., 1., 3.])

    def embedding(data):
        return torch.ones(data.shape[0], 1, device=data.device)

    mps.sample(n_samples=1, domain=domain, embedding=embedding,
               build_matrices=True)
    expected = torch.tensor([0.5, 1.5, 1.]) / 3
    assert torch.allclose(captured[0][0], expected)


# ── on the ConditionalBornMachine ────────────────────────────────────────────

def _cbm(dtype, out_position, data_dim=6, num_classes=3):
    from bm4tc.core.model import CBMConfig, ConditionalBornMachine, MPSInitConfig
    cfg = CBMConfig(embedding="legendre",
                    init_kwargs=MPSInitConfig(in_dim=3, bond_dim=5, std=0.3,
                                              init_method="randn", dtype=dtype,
                                              out_position=out_position))
    torch.manual_seed(0)
    cbm = ConditionalBornMachine(cfg=cfg, data_dim=data_dim,
                                 num_classes=num_classes, device="cpu")
    cbm.prepare("cpu")
    cbm.eval()
    return cbm


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("out_position", [0, 3, 6])
def test_cbm_in_condition_is_one_gibbs_sweep(captured, dtype, out_position):
    """One ``sample(in_condition=x)`` draws every feature from the same
    distributions as one sweep of GibbsPurification on the full-range grid."""
    from bm4tc.analysis.purification import GibbsPurification

    cbm = _cbm(dtype, out_position)
    lo, hi = cbm.input_range
    x = lo + (hi - lo) * torch.rand(4, 6)
    bins = 12

    gibbs = GibbsPurification(num_bins=bins, gibbs_batch_size=4,
                              step_delta_rel=None, progress=False)
    x_gibbs, _ = gibbs.purify(cbm, x, n_sweeps=1, device="cpu")
    gibbs_probs = [p / p.sum(-1, keepdim=True) for p in captured]
    captured.clear()

    grid = torch.linspace(lo, hi, bins)
    x_tk = SamplingMPS.sample(
        cbm, in_condition=cbm.embed(x).to(cbm.dtype), domain=grid,
        embedding=lambda v: cbm.embedding(v).to(cbm.dtype).reshape(len(v), -1),
        renormalize=True)

    assert len(captured) == len(gibbs_probs) == 6
    for p_tk, p_gibbs in zip(captured, gibbs_probs):
        assert torch.allclose(p_tk.double(), p_gibbs.double(), atol=1e-6)
    assert torch.equal(x_tk, x_gibbs)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("out_position", [0, 3, 6])
def test_cbm_condition_on_class_matches_amplitudes(dtype, out_position):
    """``condition`` on a one-hot class gives ψ(x, c) as a plain MPS."""
    cbm = _cbm(dtype, out_position)
    x = 2 * torch.rand(5, 6) - 1
    c = 1
    label = tk.embeddings.basis(torch.tensor(c), dim=cbm.out_dim).to(cbm.dtype)
    cond_mps = cbm.condition(label)
    psi_c = cond_mps(cbm.embed(x).to(cbm.dtype))
    las = 2 * psi_c.abs().log()
    assert torch.allclose(las, cbm.log_amp_sq(x)[:, c], atol=1e-4)


def test_cbm_unchanged_by_sampling():
    cbm = _cbm("complex64", 3)
    x = 2 * torch.rand(5, 6) - 1
    las, log_z = cbm.log_amp_sq(x), cbm.log_partition_function()
    SamplingMPS.sample(
        cbm, in_condition=cbm.embed(x).to(cbm.dtype),
        domain=torch.linspace(-1, 1, 8),
        embedding=lambda v: cbm.embedding(v).to(cbm.dtype).reshape(len(v), -1))
    cbm.condition(tk.embeddings.basis(torch.tensor(0), dim=3).to(cbm.dtype))
    assert torch.equal(las, cbm.log_amp_sq(x))
    assert torch.equal(log_z, cbm.log_partition_function())


# ── per-sample domains (D100, step 2a) ───────────────────────────────────────

@pytest.mark.parametrize('condition', [False, True])
def test_per_sample_domain_equal_rows_matches_shared(captured, condition):
    mps = SamplingMPS(n_features=4, phys_dim=3, bond_dim=3, boundary='obc',
                      out_features=[2])
    domain = torch.linspace(-1, 1, 5)
    kwargs = dict(n_samples=3, embedding=tk.embeddings.poly)
    if condition:
        kwargs['in_condition'] = tk.embeddings.poly(
            2 * torch.rand(3, 3) - 1, degree=2).to(mps.mats_env[0].dtype)

    shared, _ = mps.sample(domain=domain, return_indices=True, **kwargs)
    shared_probs = [p.clone() for p in captured]
    captured.clear()
    per_sample, _ = mps.sample(domain=domain.expand(3, -1),
                               return_indices=True, **kwargs)

    assert len(captured) == len(shared_probs) == 3
    for p, q in zip(captured, shared_probs):
        assert torch.allclose(p, q, atol=1e-6)
    assert torch.equal(per_sample, shared)


def test_per_sample_domain_values_come_from_own_row(captured):
    mps = SamplingMPS(n_features=3, phys_dim=2, bond_dim=2, boundary='obc')
    domain = torch.arange(3 * 4, dtype=torch.float).view(3, 4) / 12
    samples = mps.sample(n_samples=3, domain=domain,
                         embedding=tk.embeddings.unit)
    for row in range(3):
        assert torch.isin(samples[row], domain[row]).all()


def test_per_sample_domain_rejects_wrong_rows_and_build_matrices():
    mps = SamplingMPS(n_features=3, phys_dim=2, bond_dim=2, boundary='obc')
    with pytest.raises(ValueError, match='`n_samples` rows'):
        mps.sample(n_samples=3, domain=torch.rand(2, 4),
                   embedding=tk.embeddings.unit)
    with pytest.raises(ValueError, match='`build_matrices`'):
        mps.sample(n_samples=2, domain=torch.rand(2, 4).sort(-1).values,
                   embedding=tk.embeddings.unit, build_matrices=True)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("out_position", [0, 3, 6])
def test_cbm_windowed_in_condition_is_one_windowed_gibbs_sweep(
        captured, dtype, out_position):
    """With a window per sample and feature, ``sample(in_condition=x)`` is one
    sweep of GibbsPurification as the analysis runs it (``step_delta_rel``)."""
    from bm4tc.analysis.purification import GibbsPurification

    cbm = _cbm(dtype, out_position)
    lo, hi = cbm.input_range
    x = lo + (hi - lo) * torch.rand(4, 6)
    bins, step = 12, 0.1
    delta = step * (hi - lo)

    gibbs = GibbsPurification(num_bins=bins, gibbs_batch_size=4,
                              step_delta_rel=step, progress=False)
    x_gibbs, _ = gibbs.purify(cbm, x, n_sweeps=1, device="cpu")
    gibbs_probs = [p / p.sum(-1, keepdim=True) for p in captured]
    captured.clear()

    w_lo, w_hi = (x - delta).clamp(lo, hi), (x + delta).clamp(lo, hi)
    unit = torch.linspace(0, 1, bins)
    windows = w_lo[..., None] + (w_hi - w_lo)[..., None] * unit   # (B, D, bins)
    x_tk = SamplingMPS.sample(
        cbm, in_condition=cbm.embed(x).to(cbm.dtype),
        domain=list(windows.unbind(1)),
        embedding=lambda v: cbm.embedding(v).to(cbm.dtype),
        renormalize=True)

    assert len(captured) == len(gibbs_probs) == 6
    for p_tk, p_gibbs in zip(captured, gibbs_probs):
        assert torch.allclose(p_tk.double(), p_gibbs.double(), atol=1e-6)
    assert torch.equal(x_tk, x_gibbs)
