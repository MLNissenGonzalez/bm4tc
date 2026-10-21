"""Sampling for tensorkrowch MPS (D100).

tensorkrowch's ``develop`` branch (checked out at ``../tensorkrowch``) adds
``MPS.condition`` and ``MPS.sample``; tensorkrowch 1.1.6, which bm4tc pins, has
neither. :class:`SamplingMPS` carries them over to 1.1.6, in tensorkrowch's
style, so the code can go back upstream as a patch. ``condition`` is
``develop``'s (at ``2e23da9``); ``sample`` keeps its interface and its tests, and
changes how it contracts.

For 1.1.6: ``n_sites`` is ``n_features``; nodes have ``rank``, not ``ndim``;
``develop``'s ``super().set_data_nodes`` inside ``MPS`` is
``TensorNetwork.set_data_nodes`` here; ``condition`` returns a
:class:`SamplingMPS`.

``sample``, changed from ``develop``:

- **One domain per sample.** A domain may be ``(n_samples, domain_size)``, e.g. a
  window around each sample's current value (Gibbs purification).
- **The site's weights first.** At each sampled site everything but the site's
  physical edge is contracted first, so the cost does not depend on the number
  of candidates: with both environments single-layer, one amplitude per
  candidate (the weight is its squared modulus); otherwise the site's reduced
  density matrix (``phys_dim`` × ``phys_dim`` per sample) and one quadratic form
  per candidate. ``develop`` contracted every candidate with the environments,
  so time and memory grew with ``domain_size`` × bond dimension².
- **Double layers only where something is summed** (Martin's rule). An
  environment is double-layer (ket and bra contracted together) only if it
  contains a summed site: an output node, or, without ``in_condition``, an input
  to the right of the drawn site. Elsewhere it is the ket alone. So:

  - *Gibbs refinement* (``in_condition``, no output node): single-layer
    throughout;
  - *normal sampling* (no ``in_condition``, no output node): single left, double
    right environments without batch edge (bond dimension² once, not per
    sample);
  - *sampling from a marginal* (an output node, e.g. the class for Gibbs
    purification): the environments that contain it are double; with
    ``in_condition`` and the class in the middle, exactly one of the two at each
    site, and the right ones kept for the sweep cost batch × bond dimension²
    each, left of the class.
- **Fixed sampling nodes.** Each step of the sweep (a right environment absorbing
  a site, the left one absorbing a drawn or summed site, a site's weights) is a
  small fixed network of virtual leaves built at its first use; later sites only
  set the leaves' tensors, so tensorkrowch reuses the steps' result nodes
  (successors) and only recomputes tensors. Memory does not grow with the number
  of sites, except for the right environments kept for the sweep. Nothing is
  copied from the MPS and no edge of its nodes is touched.
- **No usable weight, uniform draw.** A row of candidate probabilities that is
  all zero or not finite is drawn uniformly, with one warning per call that
  counts such draws; ``develop`` raises.
- **Nodes kept between calls.** The steps' nodes stay in the MPS instead of
  being reset, so a later call (a further Gibbs sweep, any batch size) only sets
  tensors and repeats the same operations: no new nodes, no edges connected, no
  reset. ``reset``, and ``forward`` through it, removes them; ``develop`` built
  and reset every call.
- **Float32 precision (docs/evidence.md, "Sampling" step 6).** A double layer is
  positive semi-definite in exact arithmetic, not after float32 rounding: its
  error grows like the square of the contraction's cancellation (one layer:
  linearly), so long chains with improbable candidate windows can give negative
  weights, which raise past tolerance.

Norms are discarded (``renormalize`` divides each environment by one scalar per
sample): sampling needs only ratios between candidates, and no gradients.

:class:`SamplingMPS` also carries a contraction option for tensorkrowch's
``contract``, not in ``develop`` either: ``zipup_input`` (D112) contracts the
data into the running result site by site, so autograd keeps vectors instead of
one matrix per site (``contract``'s docstring says when it is better). The
default is tensorkrowch's ``contract``, unchanged. The zip-up is its own branch
of ``contract`` and its own method, ``_contract_envs_zipup`` (José's review,
D118), as on the fork's branch ``zipup-input`` for tensorkrowch ``develop``,
where ``marginalize_output`` works with it too; here it raises, as nothing in
bm4tc needs it. Both inline chains (``_contract_envs_zipup`` and
``_inline_contraction``) renormalize through ``_renormalize_inline``, which a
subclass that keeps the norm overrides. It has no tensorkrowch counterpart: it
goes when tensorkrowch's ``renormalize`` keeps the norm (plan.md).
"""
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Text, Tuple, Union

import torch

import tensorkrowch as tk
from tensorkrowch.components import AbstractNode, Node
from tensorkrowch.embeddings import basis


class SamplingMPS(tk.models.MPS):
    """:class:`tensorkrowch.models.MPS` with ``condition`` and ``sample`` (see
    the module docstring)."""

    # The contraction whose nodes the network keeps between calls (``sample``'s,
    # or a subclass's own; None: none kept), and the leaves of ``sample``'s
    # steps, by step (see ``_sample_kernel``).
    _nodes_key: Optional[tuple] = None
    _sample_nodes: dict = {}

    def reset(self):
        """:meth:`~tensorkrowch.TensorNetwork.reset`, which also removes the
        nodes kept between calls (see ``_nodes_key``)."""
        super().reset()
        if self._nodes_key is not None:
            self._nodes_key = None
            self._sample_nodes = {}
            self.unset_data_nodes()

    def forward(self, data=None, *args, **kwargs):
        """:meth:`~tensorkrowch.TensorNetwork.forward`, after removing the nodes
        kept by :meth:`sample` (their operations would be replayed)."""
        if self._nodes_key is not None:
            self.reset()
        return super().forward(data, *args, **kwargs)

    def _absorb_in_results_in_out_regions(self,
                                          in_results: List[Node]
                                          ) -> List[AbstractNode]:
        """Absorbs contracted input regions into the output regions."""
        nodes_out_env = []
        out_first = self.out_regions[0][0] == 0
        out_last = self.out_regions[-1][-1] == (self._n_features - 1)
        
        for i, region in enumerate(self.out_regions):
            aux_out_env = [self._mats_env[j] for j in region]
            
            if (i == 0) and out_first:
                if self._boundary == 'obc':
                    aux_out_env[0] = self._left_node @ aux_out_env[0]
            else:
                aux_out_env[0] = in_results[i - out_first] @ aux_out_env[0]
            nodes_out_env += aux_out_env

        if out_last:
            if self._boundary == 'obc':
                nodes_out_env[-1] = nodes_out_env[-1] @ self._right_node
        else:
            nodes_out_env[-1] = nodes_out_env[-1] @ in_results[-1]
        
        return nodes_out_env

    def contract(self,
                 inline_input: bool = False,
                 inline_mats: bool = False,
                 renormalize: bool = False,
                 marginalize_output: bool = False,
                 embedding_matrices: Optional[
                                        Union[torch.Tensor,
                                              Sequence[torch.Tensor]]] = None,
                 mpo: Optional[tk.models.MPO] = None,
                 zipup_input: bool = False) -> Node:
        """
        :meth:`tensorkrowch.models.MPS.contract`, with one more option,
        ``zipup_input``. Without it (the default) this is tensorkrowch's
        ``contract``, unchanged.

        By default, every input node is first contracted with its ``data`` node,
        giving one matrix per site (with a batch edge), and the matrices are
        then multiplied. All of them exist at once, and autograd keeps them for
        the backward, so memory grows like ``batch_size`` × number of sites ×
        bond dimension².

        With ``zipup_input``, the input regions are contracted the way
        ``"zip-up"`` contracts two MPS: a single running result absorbs one
        input node, then that node's ``data`` node, then the next input node,
        and so on. Starting from a boundary vector (``obc``), the result stays a
        vector per batch element, so what is kept grows like ``batch_size`` ×
        number of sites × bond dimension (× physical dimension, if gradients
        with respect to the data are needed). Starting from an input node
        (``pbc``, or an input region between two output regions), the result is
        a matrix, and nothing is saved. The number of operations is about the
        same, but they are done one site at a time, which is slower when they
        are small.

        So ``zipup_input`` is better for long MPS with large bond dimension and
        batch size relative to the physical dimension, where the matrices of the
        default would not fit in memory or take most of the time. The default is
        better for short or narrow MPS, small batches, or large physical
        dimensions, where it does fewer, larger operations.

        Parameters
        ----------
        zipup_input : bool
            Boolean indicating whether input ``data`` nodes should be contracted
            into the running result of each input region, one site at a time
            (see above), instead of with the MPS input nodes before the regions
            are contracted. ``inline_input`` must then be ``False`` (it
            raises otherwise), and the input regions are always contracted
            inline, whatever ``inline_mats``. ``marginalize_output`` (and so
            ``embedding_matrices`` and ``mpo``) is not supported with it yet.

        The other parameters are those of
        :meth:`tensorkrowch.models.MPS.contract`.

        Returns
        -------
        Node
        """
        if not zipup_input:
            return super().contract(inline_input=inline_input,
                                    inline_mats=inline_mats,
                                    renormalize=renormalize,
                                    marginalize_output=marginalize_output,
                                    embedding_matrices=embedding_matrices,
                                    mpo=mpo)
        if inline_input:
            raise ValueError(
                '`zipup_input` and `inline_input` cannot be both True: with '
                '`zipup_input`, input nodes are contracted with their `data` '
                'nodes one at a time, while each region of input nodes is '
                'contracted')
        if marginalize_output:
            raise NotImplementedError(
                '`zipup_input` does not support `marginalize_output` yet')

        # Input nodes stay connected to their data nodes, which are
        # contracted region by region in ``_contract_envs_zipup``
        in_env = self.in_env
        in_results = []
        for region in self.in_regions:
            mats_env = in_env[:len(region)]
            in_env = in_env[len(region):]
            data_env = [node.neighbours('input') for node in mats_env]
            result = self._contract_envs_zipup(mats_env=mats_env,
                                               data_env=data_env,
                                               renormalize=renormalize)
            in_results.append(result)

        if not self.out_regions:
            # If there is only input region, in_results has only 1 node
            result = in_results[0]
        else:
            nodes_out_env = self._absorb_in_results_in_out_regions(in_results)
            result = self._inline_contraction(mats_env=nodes_out_env,
                                              renormalize=renormalize)

        # As tensorkrowch's ``contract`` ends: contract periodic edge, put
        # batch edges in first positions
        if result.is_connected_to(result):
            result @= result

        batch_edges = []
        other_edges = []
        for i, edge in enumerate(result.edges):
            if edge.is_batch():
                batch_edges.append(i)
            else:
                other_edges.append(i)

        all_edges = batch_edges + other_edges
        if all_edges != list(range(len(all_edges))):
            result = tk.permute(result, tuple(all_edges))

        return result

    def _inline_contraction(self,
                            mats_env: List[AbstractNode],
                            renormalize: bool = False,
                            from_left: bool = True) -> Node:
        """tensorkrowch's ``_inline_contraction``, renormalizing through
        ``_renormalize_inline``."""
        if from_left:
            result_node = mats_env[0]
            for node in mats_env[1:]:
                result_node @= node

                if renormalize:
                    right_axes = []
                    for ax_name in result_node.axes_names:
                        if 'right' in ax_name:
                            right_axes.append(ax_name)
                    if right_axes:
                        result_node = self._renormalize_inline(result_node,
                                                               right_axes)

            return result_node

        else:
            result_node = mats_env[-1]
            for node in mats_env[-2::-1]:
                result_node = node @ result_node

                if renormalize:
                    left_axes = []
                    for ax_name in result_node.axes_names:
                        if 'left' in ax_name:
                            left_axes.append(ax_name)
                    if left_axes:
                        result_node = self._renormalize_inline(result_node,
                                                               left_axes)

            return result_node

    def _contract_envs_zipup(self,
                             mats_env: List[AbstractNode],
                             data_env: List[AbstractNode],
                             renormalize: bool = False) -> Node:
        """Contracts nodes environments inline, contracting each ``data`` node
        right after its MPS input node (zip-up): tensorkrowch's (fork branch
        ``zipup-input``), renormalizing through ``_renormalize_inline``."""
        left_node = None
        right_node = None
        if self._boundary == 'obc':
            if mats_env[0].neighbours('left') is self._left_node:
                left_node = self._left_node
            if mats_env[-1].neighbours('right') is self._right_node:
                right_node = self._right_node

        if right_node is None:
            # From left
            if left_node is None:
                result_node = data_env[0] @ mats_env[0]
                mats_env = mats_env[1:]
                data_env = data_env[1:]
            else:
                result_node = left_node

            for node, data_node in zip(mats_env, data_env):
                result_node @= node
                # Data first, as in ``_input_contraction``, so that batch
                # edges come first
                result_node = data_node @ result_node

                if renormalize:
                    right_axes = []
                    for ax_name in result_node.axes_names:
                        if 'right' in ax_name:
                            right_axes.append(ax_name)
                    if right_axes:
                        result_node = self._renormalize_inline(result_node,
                                                               right_axes)

            return result_node

        else:
            # From right
            result_node = right_node
            for node, data_node in zip(mats_env[::-1], data_env[::-1]):
                result_node = node @ result_node
                result_node = data_node @ result_node

                if renormalize:
                    left_axes = []
                    for ax_name in result_node.axes_names:
                        if 'left' in ax_name:
                            left_axes.append(ax_name)
                    if left_axes:
                        result_node = self._renormalize_inline(result_node,
                                                               left_axes)

            if left_node is not None:
                result_node = left_node @ result_node

            return result_node

    @staticmethod
    def _renormalize_inline(node: AbstractNode,
                            axes: List[Text]) -> Node:
        """One renormalization step of ``_contract_envs_zipup`` and
        ``_inline_contraction``. A subclass that keeps the norm overrides it."""
        return node.renormalize(axis=axes)

    def condition(self,
                  data: Union[torch.Tensor, Sequence[torch.Tensor]]) -> 'MPS':
        """
        Conditions output nodes on embedded data and returns a new MPS.

        Each conditioned node is contracted with its data vector. Consecutive
        conditioned nodes are then absorbed into the closest input node.
        
        If there are ``resultant`` nodes in the MPS, it will be first
        :meth:`~tensorkrowch.TensorNetwork.reset`.

        Embedded data can be passed as a single tensor with one of the
        following layouts:

        * ``(phys_dim,)`` when there is only one output node.
        * ``(len(out_features), phys_dim)`` with no batch dimension.
        * ``(n_features, phys_dim)`` with no batch dimension. Only the entries
          corresponding to ``out_features`` are used.
        * Either of the two previous layouts with an initial batch dimension
          of size 1.

        It can also be passed as a list or tuple containing either
        ``len(out_features)`` or ``n_features`` tensors. Each tensor should
        have shape ``(phys_dim,)`` or ``(1, phys_dim)``. In the latter case,
        the initial dimension is a batch dimension of size 1. For a sequence
        with ``n_features`` elements, only those corresponding to
        ``out_features`` are used.

        Thus, all accepted layouts represent a single configuration; batches
        with more than one element are not accepted.

        Parameters
        ----------
        data : torch.Tensor, list[torch.Tensor] or tuple[torch.Tensor]
            Embedded data used to condition the output nodes, with one of the
            layouts described above.

        Returns
        -------
        MPS

        Examples
        --------
        >>> mps = SamplingMPS(n_features=4,
        ...                          in_dim=2,
        ...                          out_dim=3,
        ...                          bond_dim=5)
        >>> label = torch.tensor(1)
        >>> embedded_label = tk.embeddings.basis(label, dim=3).float()
        >>> embedded_label.shape
        torch.Size([3])
        >>> cond_mps = mps.condition(embedded_label)
        >>> cond_mps.n_features
        3

        >>> mps = SamplingMPS(n_features=4, phys_dim=2, bond_dim=5,
        ...                     out_features=[1, 3])
        >>> all_data = tk.embeddings.basis(
        ...     torch.tensor([[0, 1, 0, 1]]), dim=2).float()
        >>> cond_mps = mps.condition(all_data)
        >>> samples = cond_mps.sample(n_samples=10)
        """
        
        if self._resultant_nodes:
            if self._nodes_key is None:
                warnings.warn(
                    'Resultant nodes will be removed before conditioning the TN')
            self.reset()
        if self._data_nodes:
            self.unset_data_nodes()

        if not self._out_features:
            raise ValueError('Cannot condition an MPS with no output nodes')
        if not self._in_features:
            raise ValueError('Conditioning all nodes would not return an MPS')

        # Remove the optional batch dimension and select output data.
        if isinstance(data, torch.Tensor):
            if data.ndim == 3:
                if data.shape[0] != 1:
                    raise ValueError('`data` should have batch size 1')
                data = data.squeeze(0)
            elif data.ndim == 1:
                data = data.unsqueeze(0)
            elif data.ndim != 2:
                raise ValueError(
                    '`data` should be provided without batch dimension or '
                    'with batch size 1')

            if data.shape[-2] == self._n_features:
                data = data[self._out_features]
        else:
            if len(data) == self._n_features:
                data = [data[site] for site in self._out_features]

            data = list(data)
            for i, tensor in enumerate(data):
                if tensor.ndim == 2:
                    if tensor.shape[0] != 1:
                        raise ValueError('`data` should have batch size 1')
                    data[i] = tensor.squeeze(0)
                elif tensor.ndim != 1:
                    raise ValueError(
                        '`data` should be provided without batch dimension or '
                        'with batch size 1')

        # Temporarily turn conditioned output nodes into inputs with data.
        self.in_features = self.out_features
        # develop's super() inside MPS: TensorNetwork's set_data_nodes.
        tk.TensorNetwork.set_data_nodes(
            self,
            input_edges=[node['input'] for node in self.in_env],
            num_batch_edges=0)
        self.add_data(data)

        mats_in_env = self._input_contraction(
            nodes_env=self.in_env,
            input_nodes=[node.neighbours('input') for node in self.in_env],
            inline_input=True)
        
        in_results = []
        for region in self.in_regions:
            result = self._contract_envs_inline(mats_env=mats_in_env[:len(region)])
            mats_in_env = mats_in_env[len(region):]
            in_results.append(result)

        nodes_out_env = self._absorb_in_results_in_out_regions(in_results)
        
        cond_mps = SamplingMPS(tensors=[node.tensor for node in nodes_out_env])
        
        self.reset()
        self.unset_data_nodes()
        self.in_features = self.out_features
        
        return cond_mps

    @staticmethod
    def _renormalize_sample_env(node: AbstractNode) -> AbstractNode:
        """Divides a sweep environment by its norm over all non-batch axes.

        That is one scalar per sample, common to every candidate value, so no
        sampling probability changes; it only keeps long chains in range.
        """
        axes = [axis.name for axis in node.axes if not axis.is_batch()]
        if axes:
            node = node.renormalize(axis=axes)
        return node

    @staticmethod
    def _trapezoidal_weights(domain: torch.Tensor) -> torch.Tensor:
        """Returns the point weights induced by the trapezoidal rule."""
        if (domain.ndim != 1) or (domain.numel() <= 1) or \
                (not torch.is_floating_point(domain)):
            return torch.ones(domain.shape[0],
                              device=domain.device,
                              dtype=torch.get_default_dtype())

        grid_basis = torch.eye(domain.numel(),
                               device=domain.device,
                               dtype=domain.dtype)
        return torch.trapezoid(grid_basis, x=domain, dim=0)

    @staticmethod
    def _probabilities_from_amplitudes(probs: torch.Tensor) -> torch.Tensor:
        """Cleans finite Born weights: their real part, clamped at zero, each
        row divided by its maximum (which keeps the row's sum finite). A row
        without weight stays zero."""
        if torch.is_complex(probs):
            scale = probs.abs().max().detach()
            imag = probs.imag.abs().max().detach()
            if scale > 0:
                if imag > (1e-5 * scale):
                    warnings.warn(
                        'Probabilities have a non-negligible imaginary part. '
                        'Only their real part will be used')
            probs = probs.real

        if not torch.is_floating_point(probs):
            probs = probs.float()

        finfo = torch.finfo(probs.dtype)
        scale = probs.abs().max().detach().clamp_min(1)
        tol = 1000 * finfo.eps * scale
        if (probs < -tol).any():
            raise ValueError(
                'Some probabilities are negative beyond numerical tolerance')

        probs = probs.clamp_min(0)
        row_max = probs.amax(dim=1, keepdim=True)
        return probs / row_max.clamp_min(finfo.tiny)

    def _sample_apply_embedding(self,
                                domain: torch.Tensor,
                                embedding: Optional[Callable],
                                phys_dim: int,
                                reference: torch.Tensor) -> torch.Tensor:
        """Embeds one sampling domain: ``(domain_size,)`` to
        ``(domain_size, phys_dim)``, or a per-sample ``(n_samples, domain_size)``
        to ``(n_samples, domain_size, phys_dim)``."""
        if embedding is None:
            if torch.is_floating_point(domain):
                rounded = domain.round()
                if not torch.equal(domain, rounded):
                    raise ValueError(
                        'Default basis embedding requires integer domain values')
                domain = rounded.long()
            embedded = basis(domain.long(), dim=phys_dim).to(
                dtype=reference.dtype)
        else:
            embedded = embedding(domain)

        if not isinstance(embedded, torch.Tensor):
            raise TypeError('`embedding` should return torch.Tensor')

        if domain.ndim == 1:
            if embedded.ndim >= 3 and embedded.shape[-2] == 1:
                embedded = embedded.squeeze(-2)

            if (embedded.ndim != 2) or (embedded.shape[-1] != phys_dim):
                raise ValueError(
                    'Embedded domains should have shape (domain_size, phys_dim)')
        elif embedded.shape != (*domain.shape, phys_dim):
            raise ValueError(
                'Embedded per-sample domains should have shape '
                '(n_samples, domain_size, phys_dim)')

        if not torch.is_floating_point(embedded) and \
                not torch.is_complex(embedded):
            embedded = embedded.to(dtype=reference.dtype)

        return embedded

    def _prepare_sample_domains(self,
                                domain,
                                embedding,
                                embedding_matrices,
                                build_matrices: bool,
                                n_samples: int
                                ) -> Tuple[List[torch.Tensor],
                                           List[torch.Tensor],
                                           List[torch.Tensor],
                                           List[torch.Tensor]]:
        """Prepares domains, embedded domains and physical metrics."""
        n_inputs = len(self._in_features)

        # Normalize site-wise arguments in input-feature order.
        if domain is None:
            first_site = self._in_features[0]
            same_default_domain = all(
                (self._phys_dim[site] == self._phys_dim[first_site]) and
                (self._mats_env[site].device ==
                 self._mats_env[first_site].device)
                for site in self._in_features)
            if same_default_domain:
                shared_domain = torch.arange(
                    self._phys_dim[first_site],
                    device=self._mats_env[first_site].device)
                domains = [shared_domain] * n_inputs
            else:
                domains = [None] * n_inputs
        elif isinstance(domain, torch.Tensor):
            domains = [domain] * n_inputs
        elif isinstance(domain, Sequence):
            if len(domain) == self._n_features:
                domains = [domain[site] for site in self._in_features]
            elif len(domain) == n_inputs:
                domains = list(domain)
            else:
                raise ValueError(
                    '`domain` should have either `n_features` elements or as '
                    'many elements as input nodes are sampled')
        else:
            raise TypeError('`domain` should be torch.Tensor type')

        if embedding is None or callable(embedding):
            embeddings_arg = [embedding] * n_inputs
        elif isinstance(embedding, Sequence):
            if len(embedding) == self._n_features:
                embeddings_arg = [embedding[site]
                                  for site in self._in_features]
            elif len(embedding) == n_inputs:
                embeddings_arg = list(embedding)
            else:
                raise ValueError(
                    '`embedding` should have either `n_features` elements or '
                    'as many elements as input nodes are sampled')
        else:
            raise TypeError('`embedding` should be callable type')

        if embedding_matrices is None or \
                isinstance(embedding_matrices, torch.Tensor):
            matrices_arg = [embedding_matrices] * n_inputs
        elif isinstance(embedding_matrices, Sequence):
            if len(embedding_matrices) == self._n_features:
                matrices_arg = [embedding_matrices[site]
                                for site in self._in_features]
            elif len(embedding_matrices) == n_inputs:
                matrices_arg = list(embedding_matrices)
            else:
                raise ValueError(
                    '`embedding_matrices` should have either `n_features` '
                    'elements or as many elements as input nodes are sampled')
        else:
            raise TypeError(
                '`embedding_matrices` should be torch.Tensor type')

        for i, site in enumerate(self._in_features):
            if domains[i] is None:
                domains[i] = torch.arange(
                    self._phys_dim[site], device=self._mats_env[site].device)
            elif not isinstance(domains[i], torch.Tensor):
                raise TypeError('`domain` should be torch.Tensor type')
            if domains[i].ndim not in (1, 2):
                raise ValueError(
                    'Each element of `domain` should be a rank-1 tensor, or '
                    'a rank-2 tensor with one domain per sample')
            if domains[i].ndim == 2:
                if domains[i].shape[0] != n_samples:
                    raise ValueError(
                        'Per-sample domains should have `n_samples` rows')
                if build_matrices:
                    raise ValueError(
                        '`build_matrices` needs one domain shared by all '
                        'samples: a per-sample metric would make the right '
                        'environments per sample')
            if domains[i].numel() < 1:
                raise ValueError('`domain` tensors cannot be empty')

        shared_embedding = all(aux_domain is domains[0]
                               for aux_domain in domains) and \
            all(aux_embedding is embeddings_arg[0]
                for aux_embedding in embeddings_arg) and \
            all(self._phys_dim[site] == self._phys_dim[self._in_features[0]]
                for site in self._in_features)

        if shared_embedding:
            embedded = self._sample_apply_embedding(
                domain=domains[0],
                embedding=embeddings_arg[0],
                phys_dim=self._phys_dim[self._in_features[0]],
                reference=self._mats_env[self._in_features[0]].tensor)
            embeddings = [embedded] * n_inputs
        else:
            embeddings = [
                self._sample_apply_embedding(
                    domain=aux_domain,
                    embedding=aux_embedding,
                    phys_dim=self._phys_dim[site],
                    reference=self._mats_env[site].tensor)
                for site, aux_domain, aux_embedding in zip(
                    self._in_features, domains, embeddings_arg)
            ]

        matrices = []
        weights = []
        shared_metric = shared_embedding and \
            all(aux_matrix is matrices_arg[0]
                for aux_matrix in matrices_arg)
        for i, (site, aux_domain, aux_embedding, aux_matrix) in enumerate(zip(
                self._in_features, domains, embeddings, matrices_arg)):
            if shared_metric and i > 0:
                matrices.append(matrices[0])
                weights.append(weights[0])
                continue

            phys_dim = self._phys_dim[site]
            if aux_matrix is not None:
                if not isinstance(aux_matrix, torch.Tensor):
                    raise TypeError(
                        '`embedding_matrices` should be torch.Tensor type')
                if aux_matrix.shape != (phys_dim, phys_dim):
                    raise ValueError(
                        '`embedding_matrices` should have shape '
                        '(phys_dim, phys_dim)')
                aux_weights = torch.ones(aux_domain.shape[-1],
                                         device=aux_domain.device,
                                         dtype=torch.get_default_dtype())
            elif build_matrices:
                if torch.is_floating_point(aux_domain) and \
                        (aux_domain.numel() > 1):
                    if not (aux_domain[1:] > aux_domain[:-1]).all():
                        raise ValueError(
                            'Continuous `domain` tensors should be strictly '
                            'increasing')
                    integrand = torch.einsum('ds,dt->dst',
                                              aux_embedding,
                                              aux_embedding.conj())
                    aux_matrix = torch.trapezoid(integrand,
                                                  x=aux_domain,
                                                  dim=0)
                    aux_weights = self._trapezoidal_weights(aux_domain)
                else:
                    aux_matrix = torch.einsum('ds,dt->st',
                                              aux_embedding,
                                              aux_embedding.conj())
                    aux_weights = torch.ones(
                        aux_domain.shape[0],
                        device=aux_domain.device,
                        dtype=torch.get_default_dtype())
            else:
                reference = self._mats_env[site]
                aux_matrix = torch.eye(phys_dim,
                                       device=reference.device,
                                       dtype=reference.dtype)
                aux_weights = torch.ones(
                    aux_domain.shape[-1],
                    device=aux_domain.device,
                    dtype=reference.tensor.real.dtype)

            matrices.append(aux_matrix)
            weights.append(aux_weights)

        return domains, embeddings, matrices, weights

    def _prepare_in_condition(self,
                              in_condition,
                              n_samples: Optional[int]
                              ) -> Tuple[Optional[Union[torch.Tensor,
                                                        List[torch.Tensor]]], int]:
        """Normalizes embedded input conditions in input-feature order."""
        if in_condition is None:
            return None, 1 if n_samples is None else n_samples

        if isinstance(in_condition, torch.Tensor):
            if in_condition.ndim != (self._n_batches + 2):
                raise ValueError(
                    '`in_condition` should have `n_batches` batch dimensions '
                    'followed by an input-feature and a physical dimension')
            if in_condition.shape[-2] != len(self._in_features):
                raise ValueError(
                    'The penultimate dimension of `in_condition` should be '
                    'the number of input nodes')
            if any(in_condition.shape[-1] != self._phys_dim[site]
                   for site in self._in_features):
                raise ValueError(
                    'The last dimension of `in_condition` should match the '
                    'physical dimensions of all input nodes')

            batch_size = in_condition.shape[:-2].numel()
            in_condition = in_condition.reshape(
                batch_size, len(self._in_features), in_condition.shape[-1])

        elif isinstance(in_condition, Sequence):
            if len(in_condition) != len(self._in_features):
                raise ValueError(
                    '`in_condition` should have as many elements as input nodes')
            conditions = list(in_condition)

            for condition in conditions:
                if not isinstance(condition, torch.Tensor):
                    raise TypeError(
                        'Elements of `in_condition` should be torch.Tensor type')
                if condition.ndim != (self._n_batches + 1):
                    raise ValueError(
                        'Elements of `in_condition` should have `n_batches` '
                        'batch dimensions followed by a physical dimension')

            batch_shape = None
            for site, condition in zip(self._in_features, conditions):
                if condition.shape[-1] != self._phys_dim[site]:
                    raise ValueError(
                        'The last dimension of each `in_condition` tensor '
                        'should match the physical dimension of its input node')
                if batch_shape is None:
                    batch_shape = condition.shape[:-1]
                elif condition.shape[:-1] != batch_shape:
                    raise ValueError(
                        'All `in_condition` tensors should have the same '
                        'batch shape')

            batch_size = torch.Size(batch_shape).numel()
            in_condition = [
                condition.reshape(batch_size, condition.shape[-1])
                for condition in conditions]
        else:
            raise TypeError(
                '`in_condition` should be torch.Tensor, tuple[torch.Tensor] '
                'or list[torch.Tensor] type')

        if n_samples is None:
            n_samples = batch_size
        elif batch_size not in [1, n_samples]:
            raise ValueError(
                '`n_samples` should match the batch size of `in_condition`')

        if batch_size == 1 and n_samples > 1:
            if isinstance(in_condition, torch.Tensor):
                in_condition = in_condition.expand(n_samples, -1, -1)
            else:
                in_condition = [condition.expand(n_samples, -1)
                                for condition in in_condition]

        return in_condition, n_samples

    def _sample_kernel(self,
                       kernels: Dict,
                       key: Tuple,
                       leaves: Dict[Text, Tuple[Tuple[Text, ...],
                                                torch.Tensor]],
                       edges: Sequence[Tuple[Tuple[Text, Text],
                                             Tuple[Text, Text]]]
                       ) -> Dict[Text, Node]:
        """The leaves of one step of the sampling sweep: a small fixed network.

        Built at the step's first use, from ``leaves`` (name: axes and tensor)
        connected by ``edges`` (pairs of (leaf, axis)). Later uses only set the
        leaves' tensors, so the step's operations reuse their result nodes
        (successors) and only recompute tensors: the number of nodes, and the
        memory they hold, do not grow with the number of sites.
        """
        nodes = kernels.get(key)
        if nodes is None:
            nodes = {name: Node(tensor=tensor,
                                axes_names=axes,
                                name='virtual_result_sample_' + name.lower(),
                                network=self,
                                virtual=True)
                     for name, (axes, tensor) in leaves.items()}
            for (node1, axis1), (node2, axis2) in edges:
                nodes[node1][axis1] ^ nodes[node2][axis2]
            kernels[key] = nodes
        else:
            for name, (_, tensor) in leaves.items():
                nodes[name]._direct_set_tensor(tensor)
        return nodes

    def _sample_env_axes(self,
                         bond: Text,
                         double: bool,
                         batched: bool = True,
                         bra: bool = False) -> Tuple[Text, ...]:
        """Axis names of a sweep environment: its open bond (``right`` for a
        left environment, ``left`` for a right one), the closing bond ``loop``
        with PBC, and their ``_copy`` on the bra layer when it is double (or
        when it is the bra of a single one)."""
        names = [bond] + (['loop'] if self._boundary == 'pbc' else [])
        if bra:
            names = [name + '_copy' for name in names]
        elif double:
            names = names + [name + '_copy' for name in names]
        return (('batch',) if batched else ()) + tuple(names)

    def _sample_env_tensor(self,
                           node: AbstractNode,
                           axes: Tuple[Text, ...],
                           renormalize: bool) -> torch.Tensor:
        """The tensor of a new environment, its axes in the order ``axes``."""
        if renormalize:
            node = self._renormalize_sample_env(node)
        return node.tensor.permute([node.get_axis_num(axis) for axis in axes])

    def _sample_right_step(self,
                           kernels: Dict,
                           site: int,
                           right: torch.Tensor,
                           double: bool,
                           batched: bool,
                           data: Optional[torch.Tensor],
                           matrix: Optional[torch.Tensor],
                           renormalize: bool) -> torch.Tensor:
        """Absorbs ``site`` into a right environment (moving left).

        With ``data`` (a given input) each layer contracts it into the site
        first, and a single environment stays single. Otherwise the site is
        summed (through ``matrix``, its physical metric, for an input; directly
        for an output), which makes the environment double.
        """
        tensor = self._mats_env[site].tensor
        site_axes, bra_axes = ('left', 'input', 'right'), \
            ('left_copy', 'input_copy', 'right_copy')
        env_axes = self._sample_env_axes('left', double, batched)
        leaves = {'A': (site_axes, tensor), 'R': (env_axes, right)}
        edges = [(('A', 'right'), ('R', 'left'))]
        summed = data is None
        new_double = double or summed
        if double or summed:
            leaves['Ab'] = (bra_axes, tensor.conj())
            if double:
                edges.append((('Ab', 'right_copy'), ('R', 'left_copy')))
            else:
                leaves['Rb'] = (self._sample_env_axes('left', False, batched,
                                                      bra=True),
                                right.conj())
                edges.append((('Ab', 'right_copy'), ('Rb', 'left_copy')))
        if summed:
            if matrix is None:
                edges.append((('A', 'input'), ('Ab', 'input_copy')))
            else:
                leaves['M'] = (('input', 'input_copy'), matrix)
                edges += [(('A', 'input'), ('M', 'input')),
                          (('M', 'input_copy'), ('Ab', 'input_copy'))]
        else:
            leaves['D'] = (('batch', 'feature'), data)
            edges.append((('A', 'input'), ('D', 'feature')))
            if double:
                leaves['Db'] = (('batch', 'feature'), data.conj())
                edges.append((('Ab', 'input_copy'), ('Db', 'feature')))

        key = ('right', tensor.shape, double, batched, summed,
               matrix is None)
        n = self._sample_kernel(kernels, key, leaves, edges)
        if not summed:
            if double:
                result = ((n['A'] @ n['D']) @ n['R']) @ (n['Ab'] @ n['Db'])
            else:
                result = (n['A'] @ n['R']) @ n['D']
        else:
            half = n['A'] @ n['R']
            if matrix is not None:
                half = half @ n['M']
            if double:
                result = half @ n['Ab']
            else:
                result = half @ (n['Ab'] @ n['Rb'])
        return self._sample_env_tensor(
            result, self._sample_env_axes('left', new_double, batched),
            renormalize)

    def _sample_left_step(self,
                          kernels: Dict,
                          site: int,
                          left: torch.Tensor,
                          double: bool,
                          selected: Optional[torch.Tensor],
                          renormalize: bool) -> torch.Tensor:
        """Absorbs ``site`` into the left environment (moving right): the
        drawn values ``selected`` of an input, contracted on each layer, or an
        output, summed, which makes the environment double."""
        tensor = self._mats_env[site].tensor
        site_axes, bra_axes = ('left', 'input', 'right'), \
            ('left_copy', 'input_copy', 'right_copy')
        leaves = {'A': (site_axes, tensor),
                  'L': (self._sample_env_axes('right', double), left)}
        edges = [(('L', 'right'), ('A', 'left'))]
        summed = selected is None
        if double or summed:
            leaves['Ab'] = (bra_axes, tensor.conj())
            if double:
                edges.append((('L', 'right_copy'), ('Ab', 'left_copy')))
            else:
                leaves['Lb'] = (self._sample_env_axes('right', False,
                                                      bra=True),
                                left.conj())
                edges.append((('Lb', 'right_copy'), ('Ab', 'left_copy')))
        if summed:
            edges.append((('A', 'input'), ('Ab', 'input_copy')))
        else:
            leaves['S'] = (('batch', 'feature'), selected)
            edges.append((('A', 'input'), ('S', 'feature')))
            if double:
                leaves['Sb'] = (('batch', 'feature'), selected.conj())
                edges.append((('Ab', 'input_copy'), ('Sb', 'feature')))

        key = ('left', tensor.shape, double, summed)
        n = self._sample_kernel(kernels, key, leaves, edges)
        if not summed:
            if double:
                result = (n['L'] @ (n['A'] @ n['S'])) @ (n['Ab'] @ n['Sb'])
            else:
                result = (n['L'] @ n['A']) @ n['S']
        elif double:
            result = (n['L'] @ n['A']) @ n['Ab']
        else:
            result = (n['L'] @ n['A']) @ (n['Lb'] @ n['Ab'])
        return self._sample_env_tensor(
            result, self._sample_env_axes('right', double or summed),
            renormalize)

    def _sample_site_weights(self,
                             kernels: Dict,
                             site: int,
                             left: torch.Tensor,
                             left_double: bool,
                             right: torch.Tensor,
                             right_double: bool,
                             right_batched: bool,
                             embedded: torch.Tensor) -> torch.Tensor:
        """The Born weight of every candidate value of ``site`` given the
        rest, ``(n_samples, domain_size)``.

        Everything but the site's physical edge is contracted first, so the
        cost does not depend on the number of candidates. With both
        environments single (nothing summed) that is one amplitude per
        candidate, whose squared modulus is the weight. Otherwise it is the
        site's reduced density matrix, and each weight is a quadratic form on
        it.
        """
        tensor = self._mats_env[site].tensor
        pbc = self._boundary == 'pbc'
        per_sample = embedded.ndim == 3
        candidate_axes = ('batch', 'domain_batch', 'feature') if per_sample \
            else ('domain_batch', 'feature')
        leaves = {
            'A': (('left', 'input', 'right'), tensor),
            'L': (self._sample_env_axes('right', left_double), left),
            'R': (self._sample_env_axes('left', right_double, right_batched),
                  right),
            'C': (candidate_axes, embedded)}
        edges = [(('L', 'right'), ('A', 'left')),
                 (('A', 'right'), ('R', 'left')),
                 (('A', 'input'), ('C', 'feature'))]
        if pbc:
            edges.append((('L', 'loop'), ('R', 'loop')))

        density = left_double or right_double
        if density:
            leaves['Ab'] = (('left_copy', 'input_copy', 'right_copy'),
                            tensor.conj())
            leaves['Cb'] = (candidate_axes, embedded.conj())
            edges.append((('Ab', 'input_copy'), ('Cb', 'feature')))
            # Each layer's bonds go to the environment that holds that layer.
            if left_double:
                bra_left = ('L', 'right_copy'), ('L', 'loop_copy')
            else:
                leaves['Lb'] = (self._sample_env_axes('right', False,
                                                      bra=True),
                                left.conj())
                bra_left = ('Lb', 'right_copy'), ('Lb', 'loop_copy')
            if right_double:
                bra_right = ('R', 'left_copy'), ('R', 'loop_copy')
            else:
                leaves['Rb'] = (self._sample_env_axes('left', False,
                                                      right_batched,
                                                      bra=True),
                                right.conj())
                bra_right = ('Rb', 'left_copy'), ('Rb', 'loop_copy')
            edges += [(bra_left[0], ('Ab', 'left_copy')),
                      (('Ab', 'right_copy'), bra_right[0])]
            if pbc:
                edges.append((bra_left[1], bra_right[1]))

        key = ('site', tensor.shape, left_double, right_double,
               right_batched, per_sample)
        n = self._sample_kernel(kernels, key, leaves, edges)
        if not density:
            amplitudes = ((n['L'] @ n['A']) @ n['R']) @ n['C']
            weights = amplitudes.tensor
            weights = weights.real ** 2 + weights.imag ** 2 \
                if torch.is_complex(weights) else weights ** 2
            probs_node = amplitudes
        else:
            if left_double and right_double:
                rho = ((n['L'] @ n['A']) @ n['R']) @ n['Ab']
            elif left_double:
                rho = (n['L'] @ (n['A'] @ n['R'])) @ (n['Ab'] @ n['Rb'])
            else:
                rho = ((n['L'] @ n['A']) @ n['R']) @ (n['Lb'] @ n['Ab'])
            probs_node = (rho @ n['C']) @ n['Cb']
            weights = probs_node.tensor

        domain_axis = probs_node.get_axis_num('domain_batch')
        weights = weights.movedim(domain_axis, -1)
        return weights.reshape(-1, embedded.shape[-2])

    @torch.no_grad()
    def sample(self,
               n_samples: Optional[int] = None,
               domain: Optional[Union[torch.Tensor,
                                      Sequence[torch.Tensor]]] = None,
               embedding: Optional[Union[Callable,
                                         Sequence[Callable]]] = None,
               embedding_matrices: Optional[Union[torch.Tensor,
                                                  Sequence[torch.Tensor]]] = None,
               build_matrices: bool = False,
               in_condition: Optional[Union[torch.Tensor,
                                            Sequence[torch.Tensor]]] = None,
               canonical: bool = False,
               renormalize: bool = False,
               return_indices: bool = False,
               generator: Optional[torch.Generator] = None
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Samples input configurations from the Born distribution of the MPS.

        The probability of a configuration is proportional to the squared
        modulus of the MPS amplitude. Output nodes are always marginalized.
        If ``in_condition`` is provided, its embedded values are used as a
        right context while all input sites are resampled from left to right.
        Thus, it does not fix any feature. To sample the remaining features
        conditional on fixed values, use :meth:`condition` first and then call
        :meth:`sample` on the returned MPS.

        An environment is double-layer (ket and bra contracted together) only
        if it contains a summed site: an output node, or, without
        ``in_condition``, an input to the right of the drawn site. Elsewhere
        it is the ket alone. So without output nodes, ``in_condition`` fixes
        every site but the one being drawn (a Gibbs refinement, e.g. after a
        first draw from this MPS or from one returned by :meth:`condition`)
        and every environment is single-layer; unconditioned, the right
        environments are double-layer (identities when ``canonical``). With
        an output node, the environments that contain it are double-layer,
        which costs more than a draw without one: memory bond_dim² instead of
        bond_dim per sample and site for the right environments kept during
        the sweep. A sweep reuses a few fixed nodes from site to site, so the
        other memory does not grow with the number of sites.

        When ``build_matrices`` is ``True`` and a floating-point domain is
        given, numerical marginalization uses the trapezoidal rule over that
        domain. The same quadrature weights are included in the categorical
        probabilities used to sample each domain point.

        The nodes this method builds stay in the network: a later call sets
        their tensors and repeats the same operations, which tensorkrowch then
        computes from their successors without building nodes.
        :meth:`~tensorkrowch.TensorNetwork.reset`, and ``forward`` through it,
        removes them.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to draw. If ``None`` and ``in_condition`` has a
            batch dimension, that batch size is used. Otherwise the default is
            1.
        domain : torch.Tensor, list[torch.Tensor] or tuple[torch.Tensor], optional
            Values from which each input node is sampled. A single tensor is
            shared by all input nodes. A sequence can be ordered either by the
            sampled input nodes, with ``len(in_features)`` elements, or by all
            MPS sites, with ``n_features`` elements. In the latter case,
            entries at output sites are ignored. If ``None``, each input node
            uses ``torch.arange(phys_dim)``. Each tensor is either rank-1,
            ``(domain_size,)``, one domain for all samples, or rank-2,
            ``(n_samples, domain_size)``, one domain per sample (e.g. a window
            around each sample's current value). Per-sample domains cannot be
            combined with ``build_matrices``.
        embedding : callable, list[callable] or tuple[callable], optional
            Embedding applied to the domains before contraction. The callable
            should return tensors with shape ``(domain_size, phys_dim)``, or
            ``(n_samples, domain_size, phys_dim)`` for a per-sample domain. If
            ``None``, domains are embedded with
            :func:`~tensorkrowch.embeddings.basis`.
        embedding_matrices : torch.Tensor, list[torch.Tensor] or tuple[torch.Tensor], optional
            Physical metric matrices used to marginalize input nodes. If not
            provided, identity matrices are used unless ``build_matrices`` is
            ``True``.
        build_matrices : bool
            Boolean indicating whether metric matrices should be approximated
            numerically from ``domain`` and ``embedding``. Floating-point
            domains use :func:`torch.trapezoid`; non-floating domains are
            treated as discrete sums.
        in_condition : torch.Tensor, list[torch.Tensor] or tuple[torch.Tensor], optional
            Embedded values for the input nodes used as a right context during
            the sampling sweep. At site ``i``, only values of input nodes to
            its right affect its distribution; every input node is eventually
            sampled again. Output nodes are always marginalized, never
            conditioned. Like data passed to
            :meth:`~tensorkrowch.TensorNetwork.forward`, it can be a tensor
            with shape ``(*batch, len(in_features), phys_dim)`` or a sequence
            with one ``(*batch, phys_dim_i)`` tensor per input node.
        canonical : bool
            If ``True``, the MPS is assumed to be in right-canonical form with
            the orthogonality center at the leftmost site, and right
            environments can be replaced by identities. For PBC, an
            ``in_condition`` context or non-identity physical metrics, this
            option is ignored with a warning and right environments are
            explicitly constructed.
        renormalize : bool
            Boolean indicating whether intermediate environments should be
            normalized during the sweep.
        return_indices : bool
            If ``True``, returns both sampled domain values and sampled domain
            indices.
        generator : torch.Generator, optional
            Random generator passed to :func:`torch.multinomial`.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            If ``return_indices`` is ``False``, returns a tensor with shape
            ``(n_samples, len(in_features))`` containing sampled domain values.
            Otherwise returns ``(samples, indices)``.

        Examples
        --------
        >>> mps = SamplingMPS(n_features=4,
        ...                     phys_dim=2,
        ...                     bond_dim=5)
        >>> samples = mps.sample(n_samples=10)
        >>> samples.shape
        torch.Size([10, 4])

        >>> domain = torch.linspace(0, 1, 32)
        >>> samples = mps.sample(n_samples=5,
        ...                      domain=domain,
        ...                      embedding=tk.embeddings.unit,
        ...                      build_matrices=True)
        >>> samples.shape
        torch.Size([5, 4])

        >>> mps.out_features = [1]
        >>> fixed_data = tk.embeddings.basis(torch.tensor([0]), dim=2).float()
        >>> cond_mps = mps.condition(fixed_data)
        >>> samples = cond_mps.sample(n_samples=5)
        >>> samples.shape
        torch.Size([5, 3])
        """
        if not isinstance(build_matrices, bool):
            raise TypeError('`build_matrices` should be bool type')
        if not isinstance(canonical, bool):
            raise TypeError('`canonical` should be bool type')
        if not isinstance(renormalize, bool):
            raise TypeError('`renormalize` should be bool type')
        if not isinstance(return_indices, bool):
            raise TypeError('`return_indices` should be bool type')
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError('`generator` should be torch.Generator type')
        if n_samples is not None:
            if not isinstance(n_samples, int):
                raise TypeError('`n_samples` should be int type')
            if n_samples <= 0:
                raise ValueError('`n_samples` should be greater than 0')

        if not self._in_features:
            raise ValueError('Cannot sample an MPS with no input nodes')

        if canonical:
            incompatible = (self._boundary != 'obc') or \
                (in_condition is not None) or \
                (embedding_matrices is not None) or build_matrices
            if incompatible:
                warnings.warn(
                    '`canonical` will be ignored and right environments will '
                    'be explicitly contracted because its assumptions are '
                    'not satisfied')
                canonical = False

        in_condition, n_samples = self._prepare_in_condition(
            in_condition=in_condition,
            n_samples=n_samples)

        domains, embeddings, matrices, quadrature_weights = \
            self._prepare_sample_domains(
            domain=domain,
            embedding=embedding,
            embedding_matrices=embedding_matrices,
            build_matrices=build_matrices,
            n_samples=n_samples)

        # The steps of an earlier call are reused (any batch and domain sizes:
        # batch edges); each kind of step builds its nodes at its first use.
        if self._nodes_key != ('sample',):
            if self._resultant_nodes or self._nodes_key is not None:
                if self._nodes_key is None:
                    warnings.warn('Resultant nodes will be removed before '
                                  'sampling the MPS')
                self.reset()
            if self._data_nodes:
                self.unset_data_nodes()
            self._nodes_key = ('sample',)
            self._sample_nodes = {}

        try:
            samples, sample_indices, n_unusable = self._sample_sweep(
                in_condition=in_condition,
                n_samples=n_samples,
                domains=domains,
                embeddings=embeddings,
                matrices=matrices,
                quadrature_weights=quadrature_weights,
                canonical=canonical,
                renormalize=renormalize,
                generator=generator)
        except BaseException:
            self.reset()
            raise

        if n_unusable:
            warnings.warn(
                f'{n_unusable} of {samples.numel()} draws had no usable '
                'weight (all zero or non-finite); drawn uniformly')

        if return_indices:
            return samples, sample_indices
        return samples

    def _sample_sweep(self,
                      in_condition,
                      n_samples: int,
                      domains: List[torch.Tensor],
                      embeddings: List[torch.Tensor],
                      matrices: List[torch.Tensor],
                      quadrature_weights: List[torch.Tensor],
                      canonical: bool,
                      renormalize: bool,
                      generator: Optional[torch.Generator]
                      ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """The body of :meth:`sample`: ``(samples, indices, unusable draws)``."""
        # An environment is double-layer (ket and bra contracted together)
        # only if it contains a summed site: an output node, or, without
        # ``in_condition``, an input to the right of the drawn site. Everywhere
        # else (drawn or given sites) it is the ket alone, one vector per
        # sample.
        n_features = self._n_features
        conditioned = in_condition is not None
        out_sites = [site for site in range(n_features)
                     if site not in self._in_features]
        first_out = min(out_sites, default=n_features)
        last_out = max(out_sites, default=-1)
        last_in = max(self._in_features)

        def left_double(site):
            return site > first_out

        def right_double(site):
            return canonical or (site < last_out) or \
                ((not conditioned) and (site < last_in))

        reference = self._mats_env[0].tensor
        pbc = self._boundary == 'pbc'
        if pbc:
            loop_dim = self._mats_env[0].shape[0]
            left = torch.eye(loop_dim, device=reference.device,
                             dtype=reference.dtype)
            right = left
        else:
            left = self._left_node.tensor
            right = self._right_node.tensor
        left = left.expand(n_samples, *left.shape)
        if conditioned:
            right = right.expand(n_samples, *right.shape)

        data = [None] * n_features
        matrix = [None] * n_features
        for idx, site in enumerate(self._in_features):
            if conditioned:
                data[site] = in_condition[idx] \
                    if isinstance(in_condition, list) \
                    else in_condition[:, idx]
            else:
                matrix[site] = matrices[idx]

        # Right environments of the drawn sites, built right to left (or
        # identities in right-canonical form). Only those are kept.
        kernels = self._sample_nodes
        right_envs = {}
        if canonical:
            for site in self._in_features:
                bond = self._mats_env[site].shape[2]
                eye = torch.eye(bond, device=reference.device,
                                dtype=reference.dtype)
                right_envs[site] = eye
        else:
            right_envs[n_features - 1] = right
            for site in range(n_features - 1, 0, -1):
                right = self._sample_right_step(
                    kernels, site, right,
                    double=right_double(site),
                    batched=conditioned,
                    data=data[site],
                    matrix=matrix[site],
                    renormalize=renormalize)
                if (site - 1) in self._in_features:
                    right_envs[site - 1] = right

        # Sweep from left to right, sampling one input node at a time.
        samples = []
        sample_indices = []
        input_idx = 0
        n_unusable = 0

        for site in range(n_features):
            if site not in self._in_features:
                left = self._sample_left_step(
                    kernels, site, left, left_double(site), None,
                    renormalize)
                continue

            per_sample = domains[input_idx].ndim == 2
            probs = self._sample_site_weights(
                kernels, site, left, left_double(site),
                right_envs.pop(site), right_double(site),
                right_batched=conditioned and not canonical,
                embedded=embeddings[input_idx])

            probs = probs * quadrature_weights[input_idx].reshape(1, -1)
            # A row with no usable weight (a value non-finite, or all zero) is
            # drawn uniformly instead of raising, so that one sample does not
            # stop the others. Counted on the device, warned about at the end
            unusable = ~torch.isfinite(probs).all(dim=1)
            probs = torch.where(unusable[:, None], torch.zeros_like(probs),
                                probs)
            probs = self._probabilities_from_amplitudes(probs)
            unusable = unusable | (probs.amax(dim=1) <= 0)
            probs = torch.where(unusable[:, None], torch.ones_like(probs),
                                probs)
            probs = probs / probs.sum(dim=1, keepdim=True)
            n_unusable = n_unusable + unusable.sum()
            ids = torch.multinomial(probs,
                                    num_samples=1,
                                    replacement=True,
                                    generator=generator).squeeze(1)

            sample_indices.append(ids)
            if per_sample:
                rows = torch.arange(n_samples, device=ids.device)
                samples.append(domains[input_idx][rows, ids])
                selected_embedding = embeddings[input_idx][rows, ids]
            else:
                samples.append(domains[input_idx][ids])
                selected_embedding = embeddings[input_idx][ids]

            input_idx += 1
            if site == last_in:
                break
            left = self._sample_left_step(
                kernels, site, left, left_double(site), selected_embedding,
                renormalize)

        # Collect values in the order of ``in_features``.
        samples = torch.stack(samples, dim=1)
        sample_indices = torch.stack(sample_indices, dim=1)
        return samples, sample_indices, int(n_unusable)
