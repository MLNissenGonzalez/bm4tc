"""CUDA graphs (D90): a function of tensors captured once per input shape and replayed.

One MPS training step is thousands of tiny kernels, each dispatched from Python:
a unit keeps a CPU core busy and the GPU mostly idle. Captured as a CUDA graph,
the same kernels replay with one launch (≈ 9-10x per unit on G21G01, bit-identical
to the eager step), and with NVIDIA MPS several units share a GPU.

:class:`Graphed` hides what capture needs: eager warm-up calls on a side stream,
one graph per tuple of input shapes and dtypes, and static input buffers.
:class:`Graphs` groups the graphed functions of one task (a training step and its
validation; an analysis part), D92. Each graph has a memory pool of its own:
sharing one between the training step and its validation corrupted the training
graph (an illegal memory access at its next replay, on G21G01).
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import torch
from torch.utils._pytree import tree_map


class GraphCaptureError(Exception):
    """Capturing failed (the function is not capturable). Not a RuntimeError, so
    a training loop that stops on runtime errors does not mistake it for a
    collapse. Fatal for the process: a failed capture leaves the CUDA allocator
    and generator in capture mode (torch 2.1), so later captures fail too."""


@dataclass
class _Capture:
    graph: "torch.cuda.CUDAGraph"
    inputs: Tuple[torch.Tensor, ...]   # static buffers the graph reads
    outputs: Any                        # static tensors the graph writes


class Graphs:
    """The CUDA graphs of one task's functions; all eager when disabled (on CPU,
    or with ``trainer.cuda_graph`` off).

    The eager warm-up calls all run on one side stream: the allocator caches
    freed memory per stream, so a new stream per call would reserve the
    function's peak memory once per call (three warm-up steps held three steps'
    memory, D92). Each graph captures into a memory pool of its own.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._warmup_stream = None  # created on the first warm-up call

    def wrap(self, function: Callable, warmup_calls: int = 3) -> "Graphed":
        return Graphed(function, self, warmup_calls)


class Graphed:
    """``function(*inputs)`` replayed from a CUDA graph of its :class:`Graphs`;
    eager when they are disabled.

    The inputs are CUDA tensors. The function must be capturable: no host syncs
    (``.item()``, boolean-mask indexing, Python branches on tensor values), the
    same operations for the same input shapes, and state that lives across calls
    (parameters, gradients, optimizer state) updated in place. A replayed call
    returns copies of the graph's outputs, which the caller may keep (the next
    replay overwrites the graph's own).

    Per input shape, the first ``warmup_calls`` calls run eagerly on a side
    stream (capture needs the library handles and autograd's streams set up),
    the next one captures the graph and replays it, and every later call
    replays it. Every call does the function's work exactly once, so the warm-up
    calls are real calls (training steps, for a training step).
    """

    def __init__(self, function: Callable, graphs: Graphs, warmup_calls: int = 3):
        self.function = function
        self.graphs = graphs
        self.warmup_calls = warmup_calls
        self._eager_calls: Dict[tuple, int] = {}
        self._captures: Dict[tuple, _Capture] = {}

    @property
    def enabled(self) -> bool:
        return self.graphs.enabled

    def __call__(self, *inputs: torch.Tensor):
        if not self.enabled:
            return self.function(*inputs)
        key = tuple((tuple(x.shape), x.dtype) for x in inputs)
        capture = self._captures.get(key)
        if capture is None:
            calls = self._eager_calls.get(key, 0)
            if calls < self.warmup_calls:
                self._eager_calls[key] = calls + 1
                return self._warm_up(inputs)
            capture = self._captures[key] = self._capture(inputs)
        for static, x in zip(capture.inputs, inputs):
            static.copy_(x)
        capture.graph.replay()
        return tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x,
                        capture.outputs)

    def _warm_up(self, inputs):
        if self.graphs._warmup_stream is None:
            self.graphs._warmup_stream = torch.cuda.Stream()
        stream = self.graphs._warmup_stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            outputs = self.function(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        return outputs

    def _capture(self, inputs) -> _Capture:
        """Record the function on static copies of ``inputs``. Capture runs no
        kernels: the caller replays the graph to do this call's work."""
        static_inputs = tuple(x.clone() for x in inputs)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                outputs = self.function(*static_inputs)
        except RuntimeError as e:
            raise GraphCaptureError(f"CUDA graph capture failed: {e}") from e
        return _Capture(graph, static_inputs, outputs)
