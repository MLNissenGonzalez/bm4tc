"""CUDA graphs (D90, D92): bm4tc.core.graphs, the Trainer's captured step and
validation, and the analysis stage's captured per-batch work.

The tests that capture need a GPU and are skipped without one (the seams run on
CPU and never capture): run them on the cluster or with the eGPU,
``pytest -q -p no:logging tests/unit/test_graphs.py``.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from bm4tc.analysis.purification import LikelihoodPurification
from bm4tc.analysis.uq import UQConfig, UQEvaluation, _BatchWork
from bm4tc.core.attacks import EvasionConfig, build_attack
from bm4tc.core.graphs import Graphed, Graphs
from bm4tc.core.model import CBMConfig, ConditionalBornMachine, MPSInitConfig
from bm4tc.core.objective import NormControlConfig, OptimizerConfig
from bm4tc.core.train import TrainConfig, Trainer
from bm4tc.pipeline.metrics import flatten_epoch

REPO = Path(__file__).resolve().parents[2]
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CUDA = torch.device("cuda")
PGD_FIXED_START = {"method": "PGD", "eps_rel": [0.1], "num_steps": 3, "random_start": False}
PGD_RANDOM_START = {"method": "PGD", "eps_rel": [0.1], "num_steps": 3, "random_start": True}


def test_disabled_graphed_is_the_function():
    calls = []
    graphed = Graphs(enabled=False).wrap(lambda x: calls.append(x) or x + 1)
    assert graphed(torch.ones(2)).tolist() == [2.0, 2.0]
    assert len(calls) == 1


@needs_cuda
def test_graphed_replays_the_function_per_input_shape():
    """Warm-up calls, then one capture per shape; every call does the work once."""
    total, reference = torch.zeros((), device=CUDA), torch.zeros((), device=CUDA)

    def accumulate(state, x):
        state.add_(x.sum())
        return state * 2

    graphed = Graphs(enabled=True).wrap(lambda x: accumulate(total, x), warmup_calls=2)
    sizes = [4] * 5 + [2] * 4 + [4] * 2
    for i, n in enumerate(sizes):
        x = torch.arange(n, dtype=torch.float32, device=CUDA) + i
        assert torch.equal(graphed(x), accumulate(reference, x))
    assert len(graphed._captures) == 2


@needs_cuda
def test_graphs_share_one_pool_and_return_copies():
    """Two graphs called alternately: each output stays valid after later calls,
    so callers may keep what a call returns, although the second capture may
    reuse what the first freed: the graphs of one Graphs share a pool (D107)."""
    graphs = Graphs(enabled=True)
    squares = graphs.wrap(lambda x: (x * x).cumsum(0), warmup_calls=1)
    shifted = graphs.wrap(lambda x: (x + 1).cumsum(0), warmup_calls=1)
    kept = []
    for i in range(5):
        x = torch.arange(64, dtype=torch.float32, device=CUDA) + i
        kept.append((x, squares(x), shifted(x)))
    for x, square_sums, shifted_sums in kept:
        assert torch.equal(square_sums, (x * x).cumsum(0))
        assert torch.equal(shifted_sums, (x + 1).cumsum(0))
    (square_capture,), (shifted_capture,) = squares._captures.values(), shifted._captures.values()
    assert square_capture.graph.pool() == shifted_capture.graph.pool() == graphs.pool


@needs_cuda
def test_warm_up_calls_reuse_one_streams_memory():
    """Three warm-up calls reserve one call's memory, not three (the allocator
    caches per stream). One call peaks at 128 MiB: the repeat and its double,
    64 MiB each, are alive together; a new stream per call reserved 384 MiB."""
    graphs = Graphs(enabled=True)
    doubled = graphs.wrap(lambda x: (x.repeat(16, 1) * 2).sum(0), warmup_calls=3)
    x = torch.ones(2**20, device=CUDA)               # 4 MiB; the repeat is 64 MiB
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_reserved()
    for _ in range(3):
        doubled(x)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_reserved() - before < 2 * 128 * 2**20


@pytest.fixture
def capture_streams(monkeypatch):
    """Per captured function, the streams its capture allocated on (the
    allocator's history), the default stream left out: the static inputs are
    copied there before capture. One stream means every buffer the graph uses
    came from the graphs' pool."""
    def trace():
        return torch.cuda.memory._snapshot()["device_traces"][torch.cuda.current_device()]

    streams = {}
    capture = Graphed._capture

    def recording(self, inputs):
        start = len(trace())
        output = capture(self, inputs)
        streams[self.function.__name__] = {
            e["stream"] for e in trace()[start:] if e["action"] == "alloc"} - {0}
        return output

    monkeypatch.setattr(Graphed, "_capture", recording)
    torch.cuda.memory._record_memory_history(max_entries=200_000, stacks="python")
    yield streams
    torch.cuda.memory._record_memory_history(enabled=None)


@needs_cuda
def test_a_capture_allocates_only_in_its_graphs_pool(capture_streams):
    """Autograd accumulates a parameter's gradients on the stream where its
    AccumulateGrad node was made. State that keeps an earlier call's autograd
    graph alive (here ``held``; in bm4tc tensorkrowch's resultant nodes and the
    log Z cache) keeps that node from the eager warm-up. Captured on another
    stream, the backward's buffers came from the general cache, which later
    eager calls reuse and ``empty_cache`` releases while the graph still writes
    them: D92's illegal memory access."""
    weight = torch.randn(64, 64, dtype=torch.complex64, device=CUDA, requires_grad=True)
    held = []

    def step(x):
        y = x @ weight
        held[:] = [y]                                  # outlives the call
        # Two gradients to sum, one a conjugate view: summed out of place, in a
        # new buffer (as tensorkrowch's complex log Z does).
        (y.abs().sum() + (x @ weight.conj()).abs().sum()).backward()
        return weight.grad.clone()

    graphed = Graphs(enabled=True).wrap(step, warmup_calls=1)
    x = torch.randn(32, 64, dtype=torch.complex64, device=CUDA)
    for _ in range(3):
        graphed(x)
    assert len(capture_streams["step"]) == 1


@needs_cuda
def test_the_captured_training_step_allocates_only_in_its_pool(capture_streams):
    """The same through the Trainer, with log Z in the step (β > 0 and the norm
    penalty), whose norm_net keeps its resultant nodes between calls."""
    _run(True)
    assert len(capture_streams["_train_step"]) == 1
    assert len(capture_streams["_clean_batch"]) == 1


HOST_SYNC_CAPTURE = """
import torch
from bm4tc.core.graphs import GraphCaptureError, Graphs
graphed = Graphs(enabled=True).wrap(lambda x: x * x.sum().item(), warmup_calls=0)
try:
    graphed(torch.ones(3, device="cuda"))
except GraphCaptureError:
    raise SystemExit(0)
raise SystemExit("no GraphCaptureError")
"""


@needs_cuda
def test_a_host_sync_fails_capture_with_its_own_error():
    """In a child process: a failed capture leaves the process's CUDA allocator and
    generator in capture mode (torch 2.1), so every later capture would fail."""
    child = subprocess.run([sys.executable, "-c", HOST_SYNC_CAPTURE], cwd=REPO,
                           capture_output=True, text=True, timeout=300)
    assert child.returncode == 0, child.stderr[-3000:]


def _run(cuda_graph: bool, evasion=None, micro_batch_size=None, seed=0):
    """Three epochs of a tiny complex MPS on CUDA, validated every epoch (two
    batches of one shape: three warm-up calls, captured in the second
    validation); the AT radius follows the curriculum (a new device scalar each
    epoch). Returns the logged records and the best tensors."""
    torch.manual_seed(seed)
    cbm = ConditionalBornMachine(
        CBMConfig(embedding="legendre", init_kwargs=MPSInitConfig(in_dim=3, bond_dim=4)),
        data_dim=6, num_classes=3)
    lo, hi = cbm.input_range
    g = torch.Generator().manual_seed(1)

    def loader(n, drop_last):
        data = lo + (hi - lo) * torch.rand(n, 6, generator=g)
        return DataLoader(TensorDataset(data, torch.randint(0, 3, (n,), generator=g)),
                          batch_size=8, drop_last=drop_last)

    cfg = TrainConfig(beta=0.5, max_epoch=3, eval_every=1, batch_size=8,
                      micro_batch_size=micro_batch_size, evasion=evasion,
                      curriculum=evasion is not None, curriculum_eps_start_rel=0.01,
                      norm_control=NormControlConfig(soft_strength=1e-2, log_target=0.0),
                      optimizer=OptimizerConfig(kwargs={"lr": 1e-3}), cuda_graph=cuda_graph)
    trainer = Trainer(cbm, cfg, loader(40, True), loader(16, False), CUDA)
    records = []
    trainer.train(on_epoch_end=lambda epoch, record: records.append(flatten_epoch(record)))
    assert len(trainer._graphed_train_step._captures) == (1 if cuda_graph else 0)
    assert len(trainer._evaluation._clean_sums._captures) == (1 if cuda_graph else 0)
    assert len(trainer._evaluation._adversarial_sums._captures) == \
        (1 if cuda_graph and evasion is not None else 0)
    if evasion is not None:   # the attack's step: one graph per chunk shape (D105)
        assert len(trainer._graphed_attack.step._captures) == (1 if cuda_graph else 0)
    return records, [t.detach().cpu() for t in trainer.cbm.tensors]


@needs_cuda
@pytest.mark.parametrize("evasion,micro_batch_size", [(None, None), (None, 4),
                                                      (PGD_FIXED_START, None),
                                                      (PGD_FIXED_START, 4)],
                         ids=["nat", "nat-micro", "at-fixed-start", "at-micro"])
def test_captured_training_is_bit_identical_to_eager(evasion, micro_batch_size):
    eager_records, eager_tensors = _run(False, evasion, micro_batch_size)
    graph_records, graph_tensors = _run(True, evasion, micro_batch_size)
    assert graph_records == eager_records
    assert all(torch.equal(e, g) for e, g in zip(eager_tensors, graph_tensors))


def _run_at_e2as_shape(cuda_graph: bool):
    """MNIST12 d3r40, batch 512, β 0.5, as E2a trains it, for three epochs:
    validation (four batches) captured in the first epoch, then training
    replays again."""
    torch.manual_seed(0)
    cbm = ConditionalBornMachine(CBMConfig(init_kwargs=MPSInitConfig(in_dim=3, bond_dim=40)),
                                 data_dim=144, num_classes=10)
    lo, hi = cbm.input_range
    g = torch.Generator().manual_seed(1)

    def loader(n, shuffle):
        data = lo + (hi - lo) * torch.rand(n, 144, generator=g)
        return DataLoader(TensorDataset(data, torch.randint(0, 10, (n,), generator=g)),
                          batch_size=512, shuffle=shuffle)

    cfg = TrainConfig(beta=0.5, max_epoch=3, eval_every=1, batch_size=512,
                      norm_control=NormControlConfig(soft_strength=0.1, log_target=0.0),
                      optimizer=OptimizerConfig(kwargs={"lr": 1e-3}), cuda_graph=cuda_graph)
    trainer = Trainer(cbm, cfg, loader(5 * 512, True), loader(4 * 512, False), CUDA)
    records = []
    trainer.train(on_epoch_end=lambda epoch, record: records.append(flatten_epoch(record)))
    assert len(records) == 3 and not trainer._collapsed
    assert len(trainer._evaluation._clean_sums._captures) == (1 if cuda_graph else 0)
    torch.cuda.synchronize()
    return records, [t.detach().cpu() for t in trainer.cbm.tensors]


@needs_cuda
def test_captured_training_and_validation_at_e2as_shape_match_eager():
    """At this size D92's bug crashed the next training replay after validation
    was captured (an illegal memory access): the backward's stray buffers sat
    in segments that validation's capture released. The tiny models above kept
    them in shared segments and passed. Number for number, not only no crash."""
    eager_records, eager_tensors = _run_at_e2as_shape(False)
    graph_records, graph_tensors = _run_at_e2as_shape(True)
    assert graph_records == eager_records
    assert all(torch.equal(e, g) for e, g in zip(eager_tensors, graph_tensors))


def _run_at_at_e2as_model(cuda_graph: bool):
    """AT PGD-5 (fixed start) on MNIST12 d3r40, batch 256 in micro-batches of
    128, validation in 2.5 chunks, for five epochs: every graph is captured
    (training step and attack step at 128; clean sums, attack step and
    adversarial sums at 128 and 64), then replays. Returns the records, the
    tensors and the pools of the captures."""
    torch.manual_seed(0)
    cbm = ConditionalBornMachine(CBMConfig(init_kwargs=MPSInitConfig(in_dim=3, bond_dim=40)),
                                 data_dim=144, num_classes=10)
    lo, hi = cbm.input_range
    g = torch.Generator().manual_seed(1)

    def loader(n, batch_size, shuffle):
        data = lo + (hi - lo) * torch.rand(n, 144, generator=g)
        return DataLoader(TensorDataset(data, torch.randint(0, 10, (n,), generator=g)),
                          batch_size=batch_size, shuffle=shuffle)

    cfg = TrainConfig(beta=0.5, max_epoch=5, eval_every=1, batch_size=256,
                      micro_batch_size=128,
                      evasion={**PGD_FIXED_START, "num_steps": 5}, curriculum=True,
                      curriculum_eps_start_rel=0.01,
                      norm_control=NormControlConfig(soft_strength=0.1, log_target=0.0),
                      optimizer=OptimizerConfig(kwargs={"lr": 1e-3}), cuda_graph=cuda_graph)
    trainer = Trainer(cbm, cfg, loader(4 * 256, 256, True), loader(2 * 128 + 64, 128, False),
                      CUDA)
    records = []
    trainer.train(on_epoch_end=lambda epoch, record: records.append(flatten_epoch(record)))
    torch.cuda.synchronize()
    evaluation = trainer._evaluation
    captures = [c for graphed in (trainer._graphed_train_step, trainer._graphed_attack.step,
                                  evaluation._clean_sums, evaluation._attack.step,
                                  evaluation._adversarial_sums)
                for c in graphed._captures.values()]
    assert len(captures) == (8 if cuda_graph else 0)
    return (records, [t.detach().cpu() for t in trainer.cbm.tensors],
            {c.graph.pool() for c in captures})


@needs_cuda
def test_captured_at_in_one_pool_matches_eager():
    """The eight graphs of micro-batched AT and its validation in one pool
    (D107): each capture may reuse what earlier ones freed, so a graph that kept
    state there between replays would show here. Number for number."""
    eager_records, eager_tensors, _ = _run_at_at_e2as_model(False)
    graph_records, graph_tensors, pools = _run_at_at_e2as_model(True)
    assert len(pools) == 1
    assert graph_records == eager_records
    assert all(torch.equal(e, g) for e, g in zip(eager_tensors, graph_tensors))


@needs_cuda
def test_captured_at_with_random_start_is_reproducible():
    """Its draws differ from eager AT's (other RNG offsets), but not run to run."""
    first, _ = _run(True, PGD_RANDOM_START)
    second, _ = _run(True, PGD_RANDOM_START)
    assert first == second


# ── The analysis stage (D92) ────────────────────────────────────────────────

def _analysed_cbm():
    """A tiny MPS on CUDA, as the analysis loads it, with its log Z cached."""
    torch.manual_seed(0)
    cbm = ConditionalBornMachine(
        CBMConfig(embedding="legendre", init_kwargs=MPSInitConfig(in_dim=3, bond_dim=4)),
        data_dim=6, num_classes=3)
    cbm.prepare(device=CUDA)
    cbm.log_normalizer()
    return cbm


def _batches(cbm, n_batches, batch_size=8):
    lo, hi = cbm.input_range
    g = torch.Generator().manual_seed(2)
    return [((lo + (hi - lo) * torch.rand(batch_size, 6, generator=g)).to(CUDA),
             torch.randint(0, 3, (batch_size,), generator=g).to(CUDA))
            for _ in range(n_batches)]


@needs_cuda
@pytest.mark.parametrize("method", ["PGD", "JOINT_PGD"])
def test_captured_analysis_batch_work_is_bit_identical_to_eager(method):
    """log p(x), the prediction, the attack (fixed start) and the likelihood
    purification, replayed from graphs, against the eager routines: three
    warm-up batches, then capture and replays."""
    cbm = _analysed_cbm()
    attack = build_attack(EvasionConfig(method=method, num_steps=3, random_start=False))
    purifier = LikelihoodPurification(num_steps=3)
    eager = _BatchWork(cbm, attack, purifier, Graphs(enabled=False), CUDA)
    captured = _BatchWork(cbm, attack, purifier, Graphs(enabled=True), CUDA)
    for data, labels in _batches(cbm, 6):
        assert torch.equal(captured.log_px(data), eager.log_px(data))
        assert torch.equal(captured.predict(data), eager.predict(data))
        for eps_abs in (0.1, 0.3):           # one graph serves every radius
            adversarials = captured.attack(data, labels, eps_abs)
            assert torch.equal(adversarials, eager.attack(data, labels, eps_abs))
            purified, purified_log_px = captured.purify(adversarials, 0.05)
            eager_purified, eager_log_px = eager.purify(adversarials, 0.05)
            assert torch.equal(purified, eager_purified)
            assert torch.equal(purified_log_px, eager_log_px)
    for routine in (captured.log_px, captured.predict, captured._attack.step,
                    captured._purify.step, captured._purify.finish):
        assert len(routine._captures) == 1


def _uq_run(captured: bool):
    cbm = _analysed_cbm()
    data, labels = (torch.cat(parts).cpu() for parts in zip(*_batches(cbm, 5)))
    loader = DataLoader(TensorDataset(data, labels), batch_size=8)   # 5 batches a pass
    cfg = UQConfig(eps_rel=[0.1, 0.2], delta_rel=[0.1], percentiles=[10],
                   attack_num_steps=3, num_steps=3)
    torch.manual_seed(5)
    return UQEvaluation(cfg).evaluate(cbm, loader, CUDA, calib_loader=loader,
                                      graphs=Graphs(enabled=captured))


@needs_cuda
def test_captured_uq_evaluation_is_reproducible():
    """The whole UQ evaluation on graphs: every budget and purification
    completes, and the attack's random start (other RNG offsets than eager)
    gives the same results run to run. Clean metrics match eager exactly."""
    first, second, eager = _uq_run(True), _uq_run(True), _uq_run(False)
    assert set(first.adv_accuracies) == {0.1, 0.2}
    assert set(first.purification_results) == {(0.1, 0.1), (0.2, 0.1)}
    assert first.adv_accuracies == second.adv_accuracies
    assert first.purification_results == second.purification_results
    assert (first.adv_log_px[0.2] == second.adv_log_px[0.2]).all()
    assert first.clean_accuracy == eager.clean_accuracy
    assert (first.clean_log_px == eager.clean_log_px).all()
    assert first.thresholds == eager.thresholds
