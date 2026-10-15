"""CUDA graphs (D90): a function of tensors captured once per input shape and replayed.

One MPS training step is thousands of tiny kernels, each dispatched from Python:
a unit keeps a CPU core busy and the GPU mostly idle. Captured as a CUDA graph,
the same kernels replay with one launch (≈ 9-10x per unit on G21G01, bit-identical
to the eager step), and with NVIDIA MPS several units share a GPU.

:class:`Graphed` hides what capture needs: eager warm-up calls on a side stream,
one graph per tuple of input shapes and dtypes, and static input buffers.
:class:`Graphs` groups the graphed functions of one task (a training step and its
validation; an analysis part), D92. The graphs of one :class:`Graphs` share
one memory pool (D107).

Warm-ups and captures run on one side stream per :class:`Graphs` (D94). Autograd
runs a parameter's gradient accumulation on the stream where its AccumulateGrad
node was made, and tensorkrowch's resultant nodes and the log Z cache keep the
warm-ups' autograd graphs, hence those nodes, alive. Captured on another stream,
the backward's summing buffers came from the general cache instead of the graph's
pool; later eager calls reused them and the next capture's ``empty_cache``
released them while the graph still wrote there (D92's illegal memory access).

One pool for a task's graphs (D107): with a pool per graph, the reserved memory
was the sum of every graph's peak (AT at MNIST12 batch 256: 8 graphs, 5.3 GiB
reserved where the largest needs 1.4), although between replays a graph's pool
holds only its outputs. Sharing is safe because the graphs of one task replay
one at a time on one stream, each replay's outputs are copied out and its
static outputs are never freed, and no graph reads another's memory (inputs are
copied into static buffers allocated outside the pool). What a capture frees is
then scratch that a later capture of the task may reuse. A graph must therefore
not carry state from one replay to the next in memory it allocated during its
capture: such state lives in tensors made outside capture (parameters,
gradients and optimizer state from the eager warm-ups) or in its outputs.

Warm up everything, then capture (D110). Eager code cannot use the pool's free
memory, so an eager warm-up that runs after a capture adds its peak to the
pool's: a run peaked near twice a chunk's eager memory (full MNIST d3r40, chunk
256: out of memory on 8 GB in validation's first warm-up, the training step's
capture having left ≈ 5.3 GiB of free pool memory). A :class:`Graphs` made with
``warm_up_first`` runs every call eagerly until :meth:`Graphs.start_capture`;
the Trainer calls it after epoch 1, once every function has run at every shape.
The pool then fills only after the eager work ends, and the peak is about the
larger of the two (full-MNIST AT, micro 128: 5.92 GiB reserved, eager 5.88).
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

    The eager warm-up calls and the captures all run on one side stream: the
    allocator caches freed memory per stream, so a new stream per call would
    reserve the function's peak memory once per call (three warm-up steps held
    three steps' memory, D92); and a capture must run where the warm-ups made
    their autograd nodes (module docstring, D94). Every graph captures into one
    memory pool, so the task reserves about its largest graph's memory, not the
    sum (module docstring, D107).

    With ``warm_up_first``, every call runs eagerly (a warm-up) until
    :meth:`start_capture`; from then on a function captures at its next call
    at any shape it has run at once, and a shape it has not seen warms up
    ``warmup_calls`` times first, as without the phase (module docstring,
    D110). Without it, each function and shape warms up on its own.
    """

    def __init__(self, enabled: bool, warm_up_first: bool = False):
        self.enabled = enabled
        self._side_stream = None
        self._pool = None
        self.capturing = not warm_up_first

    def start_capture(self):
        """End the warm-up phase: from the next call on, the functions capture
        and replay."""
        self.capturing = True

    @property
    def side_stream(self) -> "torch.cuda.Stream":
        if self._side_stream is None:
            self._side_stream = torch.cuda.Stream()
        return self._side_stream

    @property
    def pool(self):
        """The memory pool every graph of this task captures into."""
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        return self._pool

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
    replays it; in a phased :class:`Graphs`, every call before
    :meth:`Graphs.start_capture` is eager and one is enough. Every call does the function's work exactly once, so the warm-up
    calls are real calls (training steps, for a training step).
    """

    def __init__(self, function: Callable, graphs: Graphs, warmup_calls: int = 3):
        self.function = function
        self.graphs = graphs
        self.warmup_calls = warmup_calls
        self._eager_calls: Dict[tuple, int] = {}
        self._warmed_in_phase: set = set()   # shapes run before start_capture (D110)
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
            ready = calls >= self.warmup_calls or key in self._warmed_in_phase
            if not (self.graphs.capturing and ready):
                self._eager_calls[key] = calls + 1
                if not self.graphs.capturing:
                    self._warmed_in_phase.add(key)
                return self._warm_up(inputs)
            capture = self._captures[key] = self._capture(inputs)
        for static, x in zip(capture.inputs, inputs):
            static.copy_(x)
        capture.graph.replay()
        return tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x,
                        capture.outputs)

    def _warm_up(self, inputs):
        stream = self.graphs.side_stream
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
            with torch.cuda.graph(graph, pool=self.graphs.pool, stream=self.graphs.side_stream):
                outputs = self.function(*static_inputs)
        except RuntimeError as e:
            raise GraphCaptureError(f"CUDA graph capture failed: {e}") from e
        return _Capture(graph, static_inputs, outputs)
