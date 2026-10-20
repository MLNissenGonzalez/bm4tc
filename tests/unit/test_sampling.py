"""SamplingMPS (D100): tensorkrowch develop's ``condition`` and ``sample`` on 1.1.6.

The first part is develop's own tests (``tests/models/test_mps.py`` at ``2e23da9``),
ported: every local distribution the sampler draws from is checked against the exact
one, enumerated over all configurations. ``torch.multinomial`` is replaced by argmax
that records its input, so a test sees each distribution and the path is
deterministic. The second part checks it on the ConditionalBornMachine: the class
is summed out or fixed, and ``sample(in_condition=x)`` is one Gibbs sweep, checked
against the full-chain sweep that GibbsPurification ran before D101. The third part
checks per-sample domains and the uniform fallback (D101), the fourth the nodes kept
between calls (D103).
"""
import itertools
import warnings

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


@pytest.mark.parametrize('out_features', [[], [0], [1]])
def test_sample_canonical_matches_right_environments(captured, out_features):
    mps = SamplingMPS(n_features=4, phys_dim=2, bond_dim=2, boundary='obc',
                      out_features=out_features)
    mps.canonicalize(oc=0, mode='svd')

    mps.sample(n_samples=2, canonical=False)
    environment_probs = [probs.clone() for probs in captured]
    captured.clear()
    mps.sample(n_samples=2, canonical=True)
    canonical_probs = [probs.clone() for probs in captured]

    assert len(environment_probs) == len(canonical_probs)
    for probs_env, probs_canonical in zip(environment_probs, canonical_probs):
        assert torch.allclose(probs_env, probs_canonical, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('boundary', BOUNDARY_CASES)
@pytest.mark.parametrize('conditioned', [False, True])
def test_sample_many_marginalized_sites(captured, boundary, conditioned):
    # 20 output sites of dimension 3 around 4 inputs. Summing each output
    # inside |psi|^2 joins ket and bra there; a single layer would carry one
    # vector per combination of output values (3**20).
    in_features = [0, 7, 16, 23]
    mps = SamplingMPS(n_features=24,
                      phys_dim=[2 if site in in_features else 3
                                for site in range(24)],
                      bond_dim=3, boundary=boundary,
                      out_features=[site for site in range(24)
                                    if site not in in_features],
                      dtype=torch.float64)
    condition_ids = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 0]])
    kwargs = dict(n_samples=1)
    if conditioned:
        kwargs = dict(in_condition=tk.embeddings.basis(condition_ids, dim=2)
                      .to(torch.float64))

    configs, probs = _exact_input_distribution(mps)
    _, indices = mps.sample(renormalize=True, return_indices=True, **kwargs)

    assert len(captured) == 4
    for row in range(indices.shape[0]):
        prefix = []
        for step, local_probs in enumerate(captured):
            expected = _conditional_probs_from_exact(
                configs, probs, step, prefix, 2,
                condition=condition_ids[row] if conditioned else None)
            assert torch.allclose(local_probs[row], expected.to(local_probs),
                                  atol=1e-5, rtol=1e-5)
            prefix.append(indices[row, step].item())


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
    # develop: no nodes left after sample. Here sample keeps its nodes (D103);
    # forward removes them first.
    mps = SamplingMPS(n_features=3, phys_dim=2, bond_dim=2, boundary='obc')
    data = tk.embeddings.basis(torch.randint(0, 2, (5, 3)), dim=2)\
        .to(mps.mats_env[0].dtype)
    expected = mps(data)
    mps.reset()
    mps.unset_data_nodes()

    _ = mps.sample(n_samples=4)
    assert mps.resultant_nodes
    result = mps(data)
    assert result.shape == (5,)
    assert torch.equal(result, expected)


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


def _full_chain_sweep(cbm, x, domains):
    """One Gibbs sweep as GibbsPurification ran it before D101: every candidate is
    a full-chain contraction (``log_amp_sq``), the class summed in log space, the
    features drawn left to right with ``torch.multinomial``. ``domains``: one
    ``(bins,)`` or ``(batch, bins)`` tensor per feature."""
    x = x.clone()
    batch, features = x.shape
    for k, candidates in enumerate(domains):
        candidates = candidates.expand(batch, -1)                 # (batch, bins)
        bins = candidates.shape[1]
        trial = x.unsqueeze(1).expand(batch, bins, features).clone()
        trial[:, :, k] = candidates
        log_p = torch.logsumexp(cbm.log_amp_sq(trial.reshape(-1, features)), dim=-1)
        log_p = log_p.view(batch, bins)
        weights = (log_p - log_p.amax(dim=1, keepdim=True)).exp()
        ids = torch.multinomial(weights, 1).squeeze(1)
        x[:, k] = candidates.gather(1, ids[:, None]).squeeze(1)
    return x


def _embedding(cbm):
    return lambda v: cbm.embedding(v).to(cbm.dtype).reshape(*v.shape, -1)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("out_position", [0, 3, 6])
def test_cbm_in_condition_is_one_gibbs_sweep(captured, dtype, out_position):
    """One ``sample(in_condition=x)`` draws every feature from the same
    distributions as one full-chain Gibbs sweep on the full-range grid."""
    cbm = _cbm(dtype, out_position)
    lo, hi = cbm.input_range
    x = lo + (hi - lo) * torch.rand(4, 6)
    grid = torch.linspace(lo, hi, 12)

    x_ref = _full_chain_sweep(cbm, x, [grid] * 6)
    ref_probs = [p / p.sum(-1, keepdim=True) for p in captured]
    captured.clear()

    x_tk = SamplingMPS.sample(
        cbm, in_condition=cbm.embed(x).to(cbm.dtype), domain=grid,
        embedding=_embedding(cbm), renormalize=True)

    assert len(captured) == len(ref_probs) == 6
    for p_tk, p_ref in zip(captured, ref_probs):
        assert torch.allclose(p_tk.double(), p_ref.double(), atol=1e-6)
    assert torch.equal(x_tk, x_ref)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("out_position", [0, 3, 6])
def test_cbm_windowed_in_condition_is_one_windowed_gibbs_sweep(
        captured, dtype, out_position):
    """With a window per sample and feature, ``sample(in_condition=x)`` is one
    full-chain Gibbs sweep over the same windows."""
    cbm = _cbm(dtype, out_position)
    lo, hi = cbm.input_range
    x = lo + (hi - lo) * torch.rand(4, 6)
    delta = 0.1 * (hi - lo)
    w_lo, w_hi = (x - delta).clamp(lo, hi), (x + delta).clamp(lo, hi)
    windows = w_lo[..., None] + (w_hi - w_lo)[..., None] * torch.linspace(0, 1, 12)
    domains = list(windows.unbind(1))                              # 6 × (4, 12)

    x_ref = _full_chain_sweep(cbm, x, domains)
    ref_probs = [p / p.sum(-1, keepdim=True) for p in captured]
    captured.clear()

    x_tk = SamplingMPS.sample(
        cbm, in_condition=cbm.embed(x).to(cbm.dtype), domain=domains,
        embedding=_embedding(cbm), renormalize=True)

    assert len(captured) == len(ref_probs) == 6
    for p_tk, p_ref in zip(captured, ref_probs):
        assert torch.allclose(p_tk.double(), p_ref.double(), atol=1e-6)
    assert torch.equal(x_tk, x_ref)


def test_gibbs_purification_sweeps_are_sample_calls(captured):
    """GibbsPurification's restricted sweep is one ``sample`` over windows
    re-centred on the sweep-start values: two sweeps match two reference sweeps."""
    from bm4tc.analysis.purification import GibbsPurification

    cbm = _cbm("float32", 3)
    lo, hi = cbm.input_range
    x = lo + (hi - lo) * torch.rand(4, 6)
    bins, step = 12, 0.1
    delta = step * (hi - lo)

    x_ref = x
    for _ in range(2):
        w_lo, w_hi = (x_ref - delta).clamp(lo, hi), (x_ref + delta).clamp(lo, hi)
        windows = w_lo[..., None] + (w_hi - w_lo)[..., None] * torch.linspace(0, 1, bins)
        x_ref = _full_chain_sweep(cbm, x_ref, list(windows.unbind(1)))

    gibbs = GibbsPurification(num_bins=bins, gibbs_batch_size=4,
                              step_delta_rel=step, progress=False)
    x_gibbs, _ = gibbs.purify(cbm, x, n_sweeps=2, device="cpu")
    assert torch.equal(x_gibbs, x_ref)


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
        embedding=_embedding(cbm))
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


# ── rows with no usable weight (D101) ────────────────────────────────────────

def test_unusable_rows_become_uniform_and_are_counted():
    probs = torch.tensor([[0., 0., 0.],
                          [1., float('nan'), 2.],
                          [float('inf'), 1., 0.],
                          [1., 0., 3.]])
    clean, n_unusable = SamplingMPS._probabilities_from_amplitudes(probs)
    assert n_unusable == 3
    assert torch.allclose(clean[:3], torch.full((3, 3), 1 / 3))
    assert torch.allclose(clean[3], torch.tensor([0.25, 0., 0.75]))


def test_huge_finite_weights_keep_their_ratios():
    # Each weight is finite, but a row sum would overflow float32.
    probs = torch.tensor([[3e38, 3e38, 0.], [1e-40, 3e-40, 0.]])
    clean, n_unusable = SamplingMPS._probabilities_from_amplitudes(probs)
    assert n_unusable == 0
    assert torch.allclose(clean, torch.tensor([[0.5, 0.5, 0.], [0.25, 0.75, 0.]]))


def test_complex_weights_use_the_real_part():
    probs = torch.tensor([[1 + 0j, 3 + 0j], [0j, 0j]])
    clean, n_unusable = SamplingMPS._probabilities_from_amplitudes(probs)
    assert n_unusable == 1
    assert torch.allclose(clean, torch.tensor([[0.25, 0.75], [0.5, 0.5]]))


def test_negative_weights_still_raise():
    with pytest.raises(ValueError, match='negative'):
        SamplingMPS._probabilities_from_amplitudes(torch.tensor([[1., -0.5]]))


def test_sample_with_no_usable_weight_warns_once_with_the_count():
    mps = SamplingMPS(n_features=3, phys_dim=2, bond_dim=2, boundary='obc')
    for node in mps.mats_env:
        node.tensor = torch.zeros_like(node.tensor)
    with pytest.warns(UserWarning, match='12 of 12 draws') as record:
        samples = mps.sample(n_samples=4)
    assert len([w for w in record if 'usable' in str(w.message)]) == 1
    assert samples.shape == (4, 3)


# ── nodes kept between calls (D103) ──────────────────────────────────────────

def _draws(mps, fresh, calls):
    """Each call's draws under its own seed; ``fresh`` resets before each, so
    every call builds its nodes, otherwise later calls reuse the first's."""
    out = []
    for i, kwargs in enumerate(calls):
        if fresh:
            mps.reset()
        torch.manual_seed(i)
        out.append(mps.sample(**kwargs))
    return out


def _sweep_calls(mps, batches):
    calls = []
    for batch in batches:
        x = torch.rand(batch, len(mps.in_features))
        windows = (x[..., None] + 0.1 * torch.linspace(-1, 1, 7)).clamp(0, 1)
        calls.append(dict(in_condition=tk.embeddings.unit(x),
                          domain=list(windows.unbind(1)),
                          embedding=tk.embeddings.unit, renormalize=True))
    return calls


@pytest.mark.parametrize('boundary', BOUNDARY_CASES)
@pytest.mark.parametrize('out_features', [[], [2]])
def test_repeated_sweeps_reuse_nodes_and_draw_as_fresh(boundary, out_features):
    mps = SamplingMPS(n_features=5, phys_dim=2, bond_dim=3, boundary=boundary,
                      out_features=out_features)
    calls = _sweep_calls(mps, [4, 4, 6, 2])
    fresh = _draws(mps, True, calls)
    mps.reset()

    torch.manual_seed(0)
    reused = [mps.sample(**calls[0])]
    names = set(mps.nodes)
    for i, kwargs in enumerate(calls[1:], start=1):
        torch.manual_seed(i)
        reused.append(mps.sample(**kwargs))
        assert set(mps.nodes) == names          # no node built after the first call
    for a, b in zip(fresh, reused):
        assert torch.equal(a, b)


@pytest.mark.parametrize('boundary', BOUNDARY_CASES)
@pytest.mark.parametrize('canonical', [False, True])
@pytest.mark.parametrize('out_features', [[], [1]])
def test_repeated_unconditioned_samples_reuse_nodes_and_draw_as_fresh(
        boundary, canonical, out_features):
    mps = SamplingMPS(n_features=4, phys_dim=2, bond_dim=3, boundary=boundary,
                      out_features=out_features)
    if canonical and boundary == 'obc':
        mps.canonicalize(oc=0, renormalize=True)
    kwargs = dict(n_samples=5, canonical=canonical and boundary == 'obc',
                  renormalize=True)
    calls = [kwargs, kwargs, dict(kwargs, n_samples=3)]
    fresh = _draws(mps, True, calls)
    mps.reset()
    reused = _draws(mps, False, calls)
    for a, b in zip(fresh, reused):
        assert torch.equal(a, b)


def test_another_structure_rebuilds_without_warning():
    mps = SamplingMPS(n_features=4, phys_dim=2, bond_dim=3, boundary='obc')
    sweep = _sweep_calls(mps, [3])[0]
    shared = dict(sweep, domain=torch.linspace(0, 1, 5))
    calls = [sweep, shared, dict(n_samples=3), sweep]
    fresh = _draws(mps, True, calls)
    mps.reset()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        reused = _draws(mps, False, calls)
    for a, b in zip(fresh, reused):
        assert torch.equal(a, b)


def test_failed_sample_leaves_the_mps_reset():
    mps = SamplingMPS(n_features=3, phys_dim=2, bond_dim=2, boundary='obc')
    _ = mps.sample(n_samples=2)
    with pytest.raises(ValueError, match='negative'):
        mps.sample(n_samples=2, embedding_matrices=-torch.eye(2))
    assert not mps.resultant_nodes and not mps.data_nodes
    assert mps.sample(n_samples=2).shape == (2, 3)


# ── contract(zipup_input=True) ───────────────────────────────────────────────
# The data nodes contracted into the running result site by site, instead of
# first with every input node: the same tensor, to rounding, for every layout
# of output nodes, either boundary and dtype, with and without renormalization;
# the same gradients; and, from a boundary vector, a fraction of what autograd
# keeps. With ``inline_input`` it raises.

ZIPUP_LAYOUTS = [[], [3], [0], [6], [1, 4], [0, 6]]


def _zipup_pair(mps, data, **kwargs):
    """``mps(data, **kwargs)`` without and with ``zipup_input``."""
    outs = []
    for zipup in (False, True):
        mps.reset()
        outs.append(mps(data, zipup_input=zipup, **kwargs))
    mps.reset()
    return outs


@pytest.mark.parametrize('boundary', BOUNDARY_CASES)
@pytest.mark.parametrize('out_features', ZIPUP_LAYOUTS)
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_zipup_input_contracts_the_same_tensor(boundary, out_features, dtype):
    mps = SamplingMPS(n_features=7, phys_dim=3, bond_dim=4, boundary=boundary,
                      out_features=out_features, init_method='randn',
                      std=0.5, dtype=dtype)
    n_in = 7 - len(out_features)
    data = torch.randn(5, n_in, 3, dtype=dtype)
    default, zipped = _zipup_pair(mps, data)
    assert zipped.shape == default.shape
    assert torch.allclose(zipped, default, rtol=1e-10, atol=1e-12)
    # Renormalized the same way as the inline default: each step divides the
    # same running result by its norm.
    default, zipped = _zipup_pair(mps, data, renormalize=True)
    mps.reset()
    inline = mps(data, inline_mats=True, renormalize=True)
    assert torch.allclose(zipped, inline, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('boundary', BOUNDARY_CASES)
def test_zipup_input_gives_the_same_gradients(boundary):
    mps = SamplingMPS(n_features=7, phys_dim=3, bond_dim=4, boundary=boundary,
                      out_features=[3], init_method='randn', std=0.5,
                      dtype=torch.float64)
    mps.auto_stack = False     # as the CBM: stacking replaces the parameters
    data = torch.randn(5, 6, 3, dtype=torch.float64)
    grads = []
    for zipup in (False, True):
        mps.reset()
        x = data.clone().requires_grad_(True)
        out = mps(x, zipup_input=zipup)
        # by name: reset reorders parameters()
        params = [p for _, p in sorted(mps.named_parameters())]
        grads.append(torch.autograd.grad(out.square().sum(), params + [x]))
    mps.reset()
    for a, b in zip(*grads):
        assert torch.allclose(a, b, rtol=1e-10, atol=1e-12)


def test_zipup_input_keeps_vectors_not_matrices():
    """From a boundary vector, autograd keeps (batch, bond) vectors per site, not
    (batch, bond, bond) matrices."""
    mps = SamplingMPS(n_features=21, phys_dim=2, bond_dim=16, boundary='obc',
                      out_features=[10], init_method='randn', std=0.2)
    data = torch.randn(32, 20, 2)

    def saved_bytes(zipup):
        mps.reset()
        saved = []

        def pack(t):
            saved.append(t.numel() * t.element_size())
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            mps(data, zipup_input=zipup).sum()
        return sum(saved)

    default, zipped = saved_bytes(False), saved_bytes(True)
    mps.reset()
    assert zipped < default / 4


def test_zipup_input_does_not_marginalize_yet():
    mps = SamplingMPS(n_features=4, phys_dim=2, bond_dim=2, out_features=[1])
    with pytest.raises(NotImplementedError, match='marginalize_output'):
        mps(torch.randn(3, 3, 2), zipup_input=True, marginalize_output=True)
    mps.reset()


def test_zipup_input_with_inline_input_raises():
    """Exclusive options raise rather than one being ignored (José's rule)."""
    mps = SamplingMPS(n_features=4, phys_dim=2, bond_dim=2, out_features=[1])
    with pytest.raises(ValueError, match='inline_input'):
        mps(torch.randn(3, 3, 2), zipup_input=True, inline_input=True)
    mps.reset()
