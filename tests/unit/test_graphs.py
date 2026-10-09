"""CUDA graphs (D90): bm4tc.core.graphs.Graphed and the Trainer's captured step.

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

from bm4tc.core.graphs import Graphed
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
    graphed = Graphed(lambda x: calls.append(x) or x + 1, enabled=False)
    assert graphed(torch.ones(2)).tolist() == [2.0, 2.0]
    assert len(calls) == 1


@needs_cuda
def test_graphed_replays_the_function_per_input_shape():
    """Warm-up calls, then one capture per shape; every call does the work once."""
    total, reference = torch.zeros((), device=CUDA), torch.zeros((), device=CUDA)

    def accumulate(state, x):
        state.add_(x.sum())
        return state * 2

    graphed = Graphed(lambda x: accumulate(total, x), enabled=True, warmup_calls=2)
    sizes = [4] * 5 + [2] * 4 + [4] * 2
    for i, n in enumerate(sizes):
        x = torch.arange(n, dtype=torch.float32, device=CUDA) + i
        assert torch.equal(graphed(x), accumulate(reference, x))
    assert len(graphed._captures) == 2


HOST_SYNC_CAPTURE = """
import torch
from bm4tc.core.graphs import GraphCaptureError, Graphed
graphed = Graphed(lambda x: x * x.sum().item(), enabled=True, warmup_calls=0)
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
    """Three epochs of a tiny complex MPS on CUDA, validated every epoch (eager
    validation between replays); the AT radius follows the curriculum (a new
    device scalar each epoch). Returns the logged records and the best tensors."""
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
    return records, [t.detach().cpu() for t in trainer.cbm.tensors]


@needs_cuda
@pytest.mark.parametrize("evasion,micro_batch_size", [(None, None), (None, 4),
                                                      (PGD_FIXED_START, None)],
                         ids=["nat", "nat-micro", "at-fixed-start"])
def test_captured_training_is_bit_identical_to_eager(evasion, micro_batch_size):
    eager_records, eager_tensors = _run(False, evasion, micro_batch_size)
    graph_records, graph_tensors = _run(True, evasion, micro_batch_size)
    assert graph_records == eager_records
    assert all(torch.equal(e, g) for e, g in zip(eager_tensors, graph_tensors))


@needs_cuda
def test_captured_at_with_random_start_is_reproducible():
    """Its draws differ from eager AT's (other RNG offsets), but not run to run."""
    first, _ = _run(True, PGD_RANDOM_START)
    second, _ = _run(True, PGD_RANDOM_START)
    assert first == second
