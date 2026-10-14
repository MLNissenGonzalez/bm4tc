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
- **Candidates through a density matrix.** At each sampled site everything but
  the site's physical edge is contracted first: ket and bra halves joined through
  the right environment give the site's reduced density matrix (``phys_dim`` ×
  ``phys_dim`` per sample), and each candidate costs one quadratic form on it.
  ``develop`` contracted every candidate with the environments, so time and
  memory grew with ``domain_size`` × bond dimension².
- **One layer for a sweep.** With ``in_condition`` every variable but the one
  being sampled is fixed (the sites to its left already drawn, those to its
  right given). The ket alone then suffices, contracted environment first: one
  vector per sample and value of the open output edges, so memory per sample is
  linear in the bond dimension. Each site's density is that ket joined with its
  conjugate, ρ = Σ_c w_c w_c†, which sums the outputs out. Unconditioned
  sampling sums the right part out: there a bra layer is needed, the right
  environments are double-layer but have no batch edge (bond dimension² once,
  not per sample), and the drawn left part stays two single-layer chains.
- **No edge surgery.** ``canonical`` joins the two right bonds on views, and pbc
  rings stay connected, so nothing is disconnected and restored.
- **The bra layer, when needed, is built in this network.** ``develop`` copied the MPS into a
  new network and moved it here; moving renames nodes in time quadratic in the
  number of sites.
- **No usable weight, uniform draw.** A row of candidate probabilities that is
  all zero or not finite is drawn uniformly, with one warning per call that
  counts such draws; ``develop`` raises.
- **Nodes kept between calls.** Every tensorkrowch operation caches its result
  node as a successor of its operands: called again on the same nodes, it only
  recomputes the tensor. ``sample`` keeps its nodes in the MPS instead of
  resetting, so a later call with the same structure (a further Gibbs sweep)
  refills the candidate and condition nodes, any batch size, and repeats the
  same operations: no new nodes, no edges connected, no reset. ``reset``, and
  ``forward`` through it, removes them; ``develop`` built and reset every call.

Norms are discarded (``renormalize`` divides each environment by one scalar per
sample): sampling needs only ratios between candidates, and no gradients.
"""
import warnings
from typing import Callable, List, Optional, Sequence, Text, Tuple, Union

import torch

import tensorkrowch as tk
from tensorkrowch.components import AbstractNode, Node
from tensorkrowch.embeddings import basis


class SamplingMPS(tk.models.MPS):
    """:class:`tensorkrowch.models.MPS` with ``condition`` and ``sample`` (see
    the module docstring)."""

    # The contraction whose nodes the network keeps between calls (a ``sample``
    # structure, or a subclass's own; None: none kept), and those of the
    # sampling nodes that are not results of operations.
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

    def _sample_node(self,
                     role: Text,
                     site: int,
                     tensor: torch.Tensor,
                     axes_names: Sequence[Text]) -> Tuple[Node, bool]:
        """The node ``sample`` keeps for ``role`` at ``site``, holding
        ``tensor``, and whether it was just created (its edges still to be
        connected). Virtual, so ``reset`` deletes it and ``add_data`` skips it."""
        node = self._sample_nodes.get((role, site))
        if node is not None:
            node.tensor = tensor
            return node, False
        node = Node(tensor=tensor,
                    axes_names=axes_names,
                    name='virtual_result_sample_' + role,
                    network=self,
                    virtual=True)
        self._sample_nodes[(role, site)] = node
        return node, True

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

    def _copy_sample_mps(self) -> Tuple[List[AbstractNode],
                                        Optional[AbstractNode],
                                        Optional[AbstractNode]]:
        """Builds a virtual, tensor-sharing copy of the MPS in this network,
        connected like the MPS (the bra layer).

        develop copies the whole MPS into a new network and moves it here;
        moving renames every node, which tensorkrowch does in time quadratic in
        the number of sites (3 s of a 6.6 s sweep at 400 sites). Building the
        copies here avoids it.
        """
        nodes = list(self._mats_env)
        if self._boundary == 'obc':
            nodes = [self._left_node] + nodes + [self._right_node]

        copied = []
        for node in nodes:
            copied_node = node.__class__(shape=node._shape,
                                         axes_names=node.axes_names,
                                         name='virtual_result_sample_copy',
                                         network=self,
                                         virtual=True)
            copied_node.set_tensor_from(node)
            copied.append(copied_node)

        for left, right in zip(copied[:-1], copied[1:]):
            left['right'] ^ right['left']
        if self._boundary == 'pbc':
            copied[-1]['right'] ^ copied[0]['left']
            return copied, None, None
        return copied[1:-1], copied[0], copied[-1]

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

    def _contract_sample_site(self,
                              node: AbstractNode,
                              copied_node: AbstractNode,
                              env: Optional[AbstractNode],
                              matrix_node: Optional[AbstractNode] = None,
                              renormalize: bool = False) -> Node:
        """Absorbs one double-layer (marginalized) site into a right
        environment: ket site, physical metric, then bra site."""
        result = node if env is None else node @ env
        if matrix_node is not None:
            result = result @ matrix_node
        result = result @ copied_node
        if renormalize:
            result = self._renormalize_sample_env(result)
        return result

    @staticmethod
    def _sample_candidate(node: AbstractNode, axis_name: Text) -> Node:
        """A view of ``node`` whose physical edge is its own, open and named
        ``axis_name``. Bond edges stay shared with the environments, so the
        view contracts with them. A repeated call returns the same view."""
        candidate = node.permute(tuple(range(node.rank)))
        if axis_name in candidate.axes_names:
            return candidate
        candidate.reattach_edges(axes=['input'])
        candidate.disconnect('input')
        candidate.get_axis('input').name = axis_name
        return candidate

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
    def _probabilities_from_amplitudes(probs: torch.Tensor
                                       ) -> Tuple[torch.Tensor, int]:
        """Cleans and row-normalizes Born probabilities.

        A row with no usable weight (all zero, or any value non-finite) is
        made uniform; returns the probabilities and the number of such rows.
        """
        unusable = ~torch.isfinite(probs).all(dim=1)
        probs = torch.where(unusable[:, None], torch.zeros_like(probs), probs)

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

        # Dividing by the row maximum first keeps the sum finite.
        probs = probs.clamp_min(0)
        row_max = probs.amax(dim=1, keepdim=True)
        unusable = unusable | (row_max.squeeze(1) <= 0)
        probs = torch.where(unusable[:, None], torch.ones_like(probs),
                            probs / row_max.clamp_min(finfo.tiny))
        n_unusable = int(unusable.sum())
        return probs / probs.sum(dim=1, keepdim=True), n_unusable

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

    def _build_sample_right_chain(self,
                                  data_nodes: List[Optional[Node]],
                                  right_ket: Optional[AbstractNode],
                                  renormalize: bool
                                  ) -> List[Optional[AbstractNode]]:
        """Right environments of a sweep, built right to left.

        Every input node to the right is fixed by its data, so the environment
        is one single-layer chain: a vector per sample (and per value of the
        open output edges), contracted environment first, so it never holds
        more than bond × physical dimension per sample. Entry ``site`` holds the
        environment of the sites right of ``site``.
        """
        right_envs = [None] * self._n_features
        right_envs[-1] = right_ket

        for site in range(self._n_features - 1, 0, -1):
            node = self._mats_env[site]
            right_ket = node if right_ket is None else node @ right_ket
            if data_nodes[site] is not None:
                right_ket = right_ket @ data_nodes[site]
            if renormalize:
                right_ket = self._renormalize_sample_env(right_ket)
            right_envs[site - 1] = right_ket

        return right_envs

    def _build_sample_right_envs(self,
                                 bra_nodes: List[AbstractNode],
                                 matrix_nodes: List[Optional[Node]],
                                 right_env: Optional[AbstractNode],
                                 renormalize: bool) -> List[Optional[Node]]:
        """Right environments for unconditioned sampling, built right to left.

        The sites to the right are marginalized, so each is absorbed as a
        double layer through its physical metric. Nothing here depends on the
        samples: the environments have no batch edge and are built once.
        Entry ``site`` holds the environment of the sites right of ``site``.
        """
        right_envs = [None] * self._n_features
        right_envs[-1] = right_env

        for site in range(self._n_features - 1, 0, -1):
            right_env = self._contract_sample_site(
                node=self._mats_env[site],
                copied_node=bra_nodes[site],
                env=right_env,
                matrix_node=matrix_nodes[site],
                renormalize=renormalize)
            right_envs[site - 1] = right_env

        return right_envs

    @staticmethod
    def _sample_sweep_density(half: AbstractNode,
                              right_env: Optional[AbstractNode]) -> Node:
        """The reduced density matrix of one site in a sweep.

        ``half`` is the left environment contracted with the site's candidate
        view; with the right environment it gives the ket ``w``: one vector over
        the candidate edge per sample and value of the open output edges. The
        density is ``w`` joined with its conjugate over the output edges,
        ρ = Σ_c w_c w_c†, which sums the outputs out.
        """
        if right_env is None:
            ket = half.permute(tuple(range(half.rank)))
        else:
            ket = half @ right_env
        open_axes = [axis.name for axis in ket.axes
                     if (not axis.is_batch()) and (axis.name != 'candidate')]
        bra = ket.conj()
        if 'candidate_copy' not in bra.axes_names:
            bra.reattach_edges(axes=open_axes + ['candidate'])
            bra.get_axis('candidate').name = 'candidate_copy'
            if open_axes:
                ket.reattach_edges(axes=open_axes)
                for name in open_axes:
                    ket[name] ^ bra[name]
        return SamplingMPS._join_layers(ket, bra)

    @staticmethod
    def _sample_density(half: AbstractNode,
                        bra_half: AbstractNode,
                        right_env,
                        canonical: bool) -> Node:
        """The reduced density matrix of one site's physical edge.

        ``half`` is the left environment contracted with the site's candidate
        view (ket), ``bra_half`` its conjugate. Joined through the right
        environment they give a node with axes ``candidate`` and
        ``candidate_copy`` (and ``batch``): the Born weights of every value of
        the site, given the rest, as a quadratic form. Joining ket and bra also
        closes the output edges between them, which marginalizes the outputs.
        """
        if canonical:
            # Right-canonical form: the right environment is the identity,
            # i.e. the two right bonds joined directly (on views, so the
            # halves keep their bonds to the next site).
            half = half.permute(tuple(range(half.rank)))
            bra_half = bra_half.permute(tuple(range(bra_half.rank)))
            if half['right'] is not bra_half['right']:
                for aux_half in (half, bra_half):
                    aux_half.reattach_edges(axes=['right'])
                    aux_half.disconnect('right')
                half['right'] ^ bra_half['right']
        elif isinstance(right_env, tuple):
            right_ket, right_bra = right_env
            if right_ket is not None:
                half = half @ right_ket
                bra_half = bra_half @ right_bra
        elif right_env is not None:
            half = half @ right_env
        return SamplingMPS._join_layers(half, bra_half)

    @staticmethod
    def _join_layers(ket: AbstractNode, bra: AbstractNode) -> Node:
        """Contracts a ket node with a bra node. When nothing connects them
        (no output edge, no bond left open: a product of the two layers), only
        their batch edges are shared, which ``@`` does not accept; ``einsum``
        then takes the product, batch edge by batch edge."""
        if ket.is_connected_to(bra):
            return ket @ bra
        letters = iter('abcdefghijklmnopqrstuvwxyz')
        index = {}
        strings = []
        for node in (ket, bra):
            string = ''
            for axis in node.axes:
                key = axis.name if axis.is_batch() else (id(node), axis.name)
                if key not in index:
                    index[key] = next(letters)
                string += index[key]
            strings.append(string)
        out = ''.join(dict.fromkeys(strings[0] + strings[1]))
        return tk.einsum(f'{strings[0]},{strings[1]}->{out}', ket, bra)

    def _sample_probabilities(self,
                              site: int,
                              density: AbstractNode,
                              embedded: torch.Tensor,
                              per_sample: bool) -> torch.Tensor:
        """Evaluates the density's quadratic form at every embedded candidate:
        ``(rows, domain_size)``, with one row, or one per sample."""
        axes_names = ('batch', 'domain_batch', 'feature') if per_sample \
            else ('domain_batch', 'feature')
        candidate_data, new = self._sample_node('candidate', site, embedded,
                                                axes_names)
        copied_candidate_data, _ = self._sample_node('candidate_copy', site,
                                                     embedded, axes_names)
        copied_candidate_data = copied_candidate_data.conj()
        if new:
            copied_candidate_data.reattach_edges(axes=['feature'])
            density.reattach_edges(axes=['candidate', 'candidate_copy'])
            candidate_data['feature'] ^ density['candidate']
            copied_candidate_data['feature'] ^ density['candidate_copy']
        probs_node = (candidate_data @ density) @ copied_candidate_data

        domain_axis = probs_node.get_axis_num('domain_batch')
        probs = probs_node.tensor.movedim(domain_axis, -1)
        return probs.reshape(-1, embedded.shape[-2])

    def _sample_absorb(self,
                       site: int,
                       half: AbstractNode,
                       axis_name: Text,
                       embedded: torch.Tensor,
                       conj: bool,
                       renormalize: bool) -> Node:
        """Closes a half's open physical edge with the drawn values: the next
        left environment of the ket (or, with ``conj``, of the bra)."""
        selected_data, new = self._sample_node(
            'selected_copy' if conj else 'selected', site, embedded,
            ('batch', 'feature'))
        if conj:
            selected_data = selected_data.conj()
        if new:
            if conj:
                selected_data.reattach_edges(axes=['feature'])
            half.reattach_edges(axes=[axis_name])
            selected_data['feature'] ^ half[axis_name]
        result = half @ selected_data
        if renormalize:
            result = self._renormalize_sample_env(result)
        return result

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

        When ``build_matrices`` is ``True`` and a floating-point domain is
        given, numerical marginalization uses the trapezoidal rule over that
        domain. The same quadrature weights are included in the categorical
        probabilities used to sample each domain point.

        The nodes this method builds (environments, candidates, and for
        unconditioned sampling a bra layer) stay in the network: a later call
        with the same structure refills them and repeats the same operations,
        which tensorkrowch then computes from their successors without
        building nodes. :meth:`~tensorkrowch.TensorNetwork.reset`, and
        ``forward`` through it, removes them.

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

        # The nodes of an earlier call are reused when the structure is the
        # same; the batch and domain sizes may differ (batch edges).
        sweep = in_condition is not None
        key = (sweep, canonical, renormalize, tuple(self._in_features),
               isinstance(in_condition, torch.Tensor),
               tuple(aux_domain.ndim for aux_domain in domains))
        build = key != self._nodes_key
        if build:
            if self._resultant_nodes or self._nodes_key is not None:
                if self._nodes_key is None:
                    warnings.warn('Resultant nodes will be removed before '
                                  'sampling the MPS')
                self.reset()
            if self._data_nodes:
                self.unset_data_nodes()
            self._nodes_key = key
            self._sample_nodes = {}

        try:
            samples, sample_indices, n_unusable = self._sample_sweep(
                build=build,
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
                      build: bool,
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
        """The body of :meth:`sample`: ``(samples, indices, unusable draws)``.

        With ``build`` the nodes are created and connected; otherwise the kept
        ones are refilled and the same operations repeated.
        """
        # With ``in_condition`` every variable but the one being sampled is
        # fixed (a sweep): the ket alone suffices, one vector per sample and
        # value of the open output edges, and each site's density is the ket
        # joined with its own conjugate. Otherwise the sites to the right are
        # summed out, which needs a bra layer (a conjugated copy) in this
        # network; output nodes join ket and bra through their physical edges,
        # so contracting the two layers marginalizes them.
        sweep = in_condition is not None
        if sweep:
            if build:
                tk.TensorNetwork.set_data_nodes(
                    self,
                    input_edges=[node['input'] for node in self.in_env],
                    num_batch_edges=1)
            self.add_data(in_condition)
            data_nodes = [None] * self._n_features
            for site, data_node in zip(self._in_features,
                                       self.data_nodes.values()):
                data_nodes[site] = data_node

            left_ket, left_bra = None, None
            if self._boundary == 'obc':
                left_ket = self._left_node
            right_envs = self._build_sample_right_chain(
                data_nodes=data_nodes,
                right_ket=self._right_node if self._boundary == 'obc' else None,
                renormalize=renormalize)
        else:
            if build:
                self._sample_nodes['bra'] = self._copy_sample_mps()
            copied_nodes, bra_left, bra_right = self._sample_nodes['bra']
            bra_nodes = [node.conj() for node in copied_nodes]
            if self._boundary == 'obc':
                bra_left = bra_left.conj()
                bra_right = bra_right.conj()

            # Physical metrics join ket and bra at input nodes.
            matrix_nodes = [None] * self._n_features
            input_idx = 0
            for site, (node, bra_node) in enumerate(zip(self._mats_env,
                                                        bra_nodes)):
                if build:
                    bra_node.reattach_edges(axes=['input'])
                if site in self._in_features:
                    matrix_node, _ = self._sample_node(
                        'mat', site, matrices[input_idx],
                        ('input', 'input_copy'))
                    if build:
                        node['input'] ^ matrix_node['input']
                        matrix_node['input_copy'] ^ bra_node['input']
                    matrix_nodes[site] = matrix_node
                    input_idx += 1
                elif build:
                    node['input'] ^ bra_node['input']

            # Right environments: one batch-free double layer, or none when
            # canonical.
            left_ket, left_bra = None, None
            if self._boundary == 'obc':
                left_ket, left_bra = self._left_node, bra_left
            if canonical:
                right_envs = [None] * self._n_features
            else:
                right_envs = self._build_sample_right_envs(
                    bra_nodes=bra_nodes,
                    matrix_nodes=matrix_nodes,
                    right_env=(self._right_node % bra_right
                               if self._boundary == 'obc' else None),
                    renormalize=renormalize)

        # Sweep from left to right, sampling one input node at a time. The left
        # environment is always single-layer: every site to the left is fixed.
        samples = []
        sample_indices = []
        input_idx = 0
        n_unusable = 0

        for site in range(self._n_features):
            node = self._mats_env[site]
            if site in self._in_features:
                # Contract everything but the site's physical edge, then
                # evaluate the candidates on that small density matrix.
                half = self._sample_candidate(node, 'candidate')
                if left_ket is not None:
                    half = left_ket @ half
                if sweep:
                    density = self._sample_sweep_density(half,
                                                         right_envs[site])
                else:
                    bra_half = self._sample_candidate(bra_nodes[site],
                                                      'candidate_copy')
                    if left_bra is not None:
                        bra_half = left_bra @ bra_half
                    density = self._sample_density(half, bra_half,
                                                   right_envs[site],
                                                   canonical)
                per_sample = domains[input_idx].ndim == 2
                probs = self._sample_probabilities(
                    site, density, embeddings[input_idx], per_sample)
                if probs.shape[0] == 1 and n_samples > 1:
                    probs = probs.expand(n_samples, -1)

                probs = probs * quadrature_weights[input_idx].reshape(1, -1)
                probs, unusable = self._probabilities_from_amplitudes(probs)
                n_unusable += unusable
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

                # Absorb the drawn values into the left environments.
                left_ket = self._sample_absorb(site, half, 'candidate',
                                               selected_embedding,
                                               conj=False,
                                               renormalize=renormalize)
                if not sweep:
                    left_bra = self._sample_absorb(site, bra_half,
                                                   'candidate_copy',
                                                   selected_embedding,
                                                   conj=True,
                                                   renormalize=renormalize)
                input_idx += 1

            else:
                # An output node: its physical edge stays open (to the bra,
                # or dangling in a sweep) until a density closes it.
                left_ket = node if left_ket is None else left_ket @ node
                if renormalize:
                    left_ket = self._renormalize_sample_env(left_ket)
                if not sweep:
                    bra_node = bra_nodes[site]
                    left_bra = bra_node if left_bra is None \
                        else left_bra @ bra_node
                    if renormalize:
                        left_bra = self._renormalize_sample_env(left_bra)

        # Collect values in the order of ``in_features``.
        samples = torch.stack(samples, dim=1)
        sample_indices = torch.stack(sample_indices, dim=1)
        return samples, sample_indices, n_unusable
