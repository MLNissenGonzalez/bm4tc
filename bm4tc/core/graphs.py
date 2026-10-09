"""CUDA graphs (D90): a function of tensors captured once per input shape and replayed.

One MPS training step is thousands of tiny kernels, each dispatched from Python:
a unit keeps a CPU core busy and the GPU mostly idle. Captured as a CUDA graph,
the same kernels replay with one launch (≈ 9-10x per unit on G21G01, bit-identical
to the eager step), and with NVIDIA MPS several units share a GPU.

:class:`Graphed` hides what capture needs: eager warm-up calls on a side stream,
one graph per tuple of input shapes and dtypes, static input buffers, and one
memory pool shared by the graphs (they never run at the same time).
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import torch


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


class Graphed:
    """``function(*inputs)`` replayed from a CUDA graph; eager when disabled.

    The inputs are CUDA tensors. The function must be capturable: no host syncs
    (``.item()``, boolean-mask indexing, Python branches on tensor values), the
    same operations for the same input shapes, and state that lives across calls
    (parameters, gradients, optimizer state) updated in place. Its outputs are
    static: each call overwrites them, so copy what must outlive the next call.

    Per input shape, the first ``warmup_calls`` calls run eagerly on a side
    stream (capture needs the library handles and autograd's streams set up),
    the next one captures the graph and replays it, and every later call
    replays it. Every call does the function's work exactly once, so the warm-up
    calls are real calls (training steps, for a training step).
    """

    def __init__(self, function: Callable, enabled: bool, warmup_calls: int = 3):
        self.function = function
        self.enabled = enabled
        self.warmup_calls = warmup_calls
        self._eager_calls: Dict[tuple, int] = {}
        self._captures: Dict[tuple, _Capture] = {}
        self._pool = None

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
        return capture.outputs

    def _warm_up(self, inputs):
        stream = torch.cuda.Stream()
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
            with torch.cuda.graph(graph, pool=self._pool):
                outputs = self.function(*static_inputs)
        except RuntimeError as e:
            raise GraphCaptureError(f"CUDA graph capture failed: {e}") from e
        if self._pool is None:
            self._pool = graph.pool()
        return _Capture(graph, static_inputs, outputs)
