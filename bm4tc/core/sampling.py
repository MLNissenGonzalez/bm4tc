"""Sampling for tensorkrowch MPS (D100).

tensorkrowch's ``develop`` branch (checked out at ``../tensorkrowch``) adds
``MPS.condition`` and ``MPS.sample``; tensorkrowch 1.1.6, which bm4tc pins, has
neither. :class:`SamplingMPS` carries them over to 1.1.6, so the code can go back
upstream as a patch: the methods are ``develop``'s (at ``2e23da9``), in
tensorkrowch's style, with only what 1.1.6 needs changed:

- ``n_sites`` is ``n_features`` in 1.1.6 (``_n_features``, docstrings);
- nodes have ``rank``, not ``ndim``;
- 1.1.6's ``MPS.copy`` does not copy the obc boundary vectors (``develop``'s
  does), so ``_copy_sample_mps`` shares them itself;
- ``condition`` returns a :class:`SamplingMPS`, so the result can be sampled.
"""
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch

import tensorkrowch as tk
from tensorkrowch.components import AbstractNode, Node
from tensorkrowch.embeddings import basis


class SamplingMPS(tk.models.MPS):
    """:class:`tensorkrowch.models.MPS` with ``develop``'s ``condition`` and
    ``sample``."""

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
        """Moves a tensor-sharing virtual MPS copy to the current network."""
        copied_mps = self.copy(share_tensors=True)
        copied_nodes = copied_mps.mats_env[:]
        copied_boundaries = []
        if self._boundary == 'obc':
            # 1.1.6's copy leaves fresh boundary vectors; share this MPS's own,
            # as develop's copy does.
            copied_mps.left_node.tensor = self._left_node.tensor
            copied_mps.right_node.tensor = self._right_node.tensor
            copied_boundaries = [copied_mps.left_node,
                                 copied_mps.right_node]

        # Mark the copied MPS as temporary before moving the whole component.
        for node in copied_nodes + copied_boundaries:
            node.name = 'virtual_result_sample_copy'
            node.change_type(virtual=True)
        copied_nodes[0].move_to_network(self)

        if self._boundary == 'obc':
            copied_left = copied_boundaries[0]
            copied_right = copied_boundaries[1]
        else:
            copied_left = None
            copied_right = None

        return copied_nodes, copied_left, copied_right

    @staticmethod
    def _contract_sample_site(node: AbstractNode,
                              copied_node: AbstractNode,
                              env: Optional[AbstractNode],
                              matrix_node: Optional[AbstractNode] = None,
                              data_nodes: Optional[Tuple[AbstractNode,
                                                         AbstractNode]] = None,
                              from_left: bool = False,
                              renormalize: bool = False) -> Node:
        """Absorbs one double-layer MPS site into a sweep environment."""
        if data_nodes is not None:
            data_node, copied_data_node = data_nodes
            if env is None:
                pair_node = Node(
                    tensor=torch.ones(1,
                                      device=data_node.device,
                                      dtype=data_node.dtype),
                    axes_names=('pair',),
                    name='virtual_result_sample_pair',
                    network=data_node.network,
                    virtual=True)
                copied_pair_node = Node(
                    tensor=torch.ones(1,
                                      device=copied_data_node.device,
                                      dtype=copied_data_node.dtype),
                    axes_names=('pair',),
                    name='virtual_result_sample_pair_copy',
                    network=data_node.network,
                    virtual=True)
                pair_node['pair'] ^ copied_pair_node['pair']
                data_node = data_node % pair_node
                copied_data_node = copied_data_node % copied_pair_node

            node = data_node @ node
            copied_node = copied_data_node @ copied_node

        if env is None:
            if matrix_node is not None:
                result = (node @ matrix_node) @ copied_node
            elif node.is_connected_to(copied_node):
                result = node @ copied_node
            else:
                raise ValueError(
                    'The two sample layers should be connected before '
                    'contracting a site')
        elif from_left:
            result = env @ node
            if matrix_node is not None:
                result = result @ matrix_node
            result = result @ copied_node
        else:
            result = node @ env
            if matrix_node is not None:
                result = result @ matrix_node
            result = result @ copied_node

        if renormalize:
            axes = [axis.name for axis in result.axes
                    if not axis.is_batch()]
            if axes:
                result = result.renormalize(axis=axes)

        return result

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
        """Cleans and row-normalizes Born probabilities."""
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

        if not torch.isfinite(probs).all():
            raise ValueError('Probabilities contain non-finite values')

        finfo = torch.finfo(probs.dtype)
        scale = probs.abs().max().detach().clamp_min(1)
        tol = 1000 * finfo.eps * scale
        if (probs < -tol).any():
            raise ValueError(
                'Some probabilities are negative beyond numerical tolerance')

        probs = probs.clamp_min(0)
        norm = probs.sum(dim=1, keepdim=True)
        if (norm <= 0).any() or (not torch.isfinite(norm).all()):
            raise ValueError(
                'Cannot sample from a zero or non-finite probability row')

        return probs / norm

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

    def _build_sample_right_envs(self,
                                 copied_nodes: List[Node],
                                 matrix_nodes: List[Optional[Node]],
                                 condition_nodes: List[Optional[
                                     Tuple[Node, Node]]],
                                 right_env: Optional[Node],
                                 renormalize: bool) -> List[Node]:
        """Builds reusable right environments with a right-to-left zip-up."""
        right_envs = [None] * self._n_features
        right_envs[-1] = right_env

        for site in range(self._n_features - 1, 0, -1):
            right_env = self._contract_sample_site(
                node=self._mats_env[site],
                copied_node=copied_nodes[site],
                env=right_env,
                matrix_node=matrix_nodes[site],
                data_nodes=condition_nodes[site],
                from_left=False,
                renormalize=renormalize)
            right_envs[site - 1] = right_env

        return right_envs

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

        This method internally calls
        :meth:`~tensorkrowch.TensorNetwork.reset`, as sampling constructs a
        temporary double-layer tensor network and contraction environments.

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

        # Sampling builds a temporary double layer in the current network.
        if self._resultant_nodes:
            warnings.warn(
                'Resultant nodes will be removed before sampling the MPS')
            self.reset()
        if self._data_nodes:
            self.unset_data_nodes()

        copied_nodes, copied_left, copied_right = self._copy_sample_mps()

        # For PBC, insert an identity on each periodic edge. The combined
        # identity is the initial left environment and keeps the ring connected.
        if self._boundary == 'pbc':
            self._mats_env[-1]['right'].disconnect()
            copied_nodes[-1]['right'].disconnect()

            closure = Node(
                tensor=torch.eye(self._mats_env[-1]['right'].size(),
                                 device=self._mats_env[-1].device,
                                 dtype=self._mats_env[-1].dtype),
                axes_names=('right', 'left'),
                name='virtual_result_sample_closure',
                network=self,
                virtual=True)
            copied_closure = Node(
                shape=closure.shape,
                axes_names=closure.axes_names,
                name='virtual_result_sample_closure_copy',
                network=self,
                virtual=True)
            copied_closure.set_tensor_from(closure)

            self._mats_env[-1]['right'] ^ closure['right']
            closure['left'] ^ self._mats_env[0]['left']
            copied_nodes[-1]['right'] ^ copied_closure['right']
            copied_closure['left'] ^ copied_nodes[0]['left']

            copied_nodes = [node.conj() for node in copied_nodes]
            copied_closure = copied_closure.conj()
            left_env = closure % copied_closure
            right_env = None
        else:
            if canonical:
                for i in range(self._n_features - 1):
                    self._mats_env[i]['right'].disconnect()
                    copied_nodes[i]['right'].disconnect()
                self._mats_env[-1]['right'].disconnect()
                copied_nodes[-1]['right'].disconnect()

            copied_nodes = [node.conj() for node in copied_nodes]
            copied_left = copied_left.conj()
            copied_right = copied_right.conj()
            left_env = self._left_node % copied_left
            right_env = None if canonical \
                else self._right_node % copied_right

        for copied_node in copied_nodes:
            copied_node.reattach_edges(axes=['input'])

        # Connect physical metrics or right-context data to both layers.
        matrix_nodes = [None] * self._n_features
        condition_nodes = [None] * self._n_features
        if in_condition is not None:
            tk.TensorNetwork.set_data_nodes(
                self,
                input_edges=[node['input'] for node in self.in_env],
                num_batch_edges=1)
            self.add_data(in_condition)
            data_nodes = list(self.data_nodes.values())

            for site, data_node in zip(self._in_features, data_nodes):
                copied_data = Node(
                    tensor=data_node.tensor,
                    axes_names=data_node.axes_names,
                    name='sample_data_copy',
                    network=self,
                    data=True)
                copied_data = copied_data.conj()
                copied_data.reattach_edges(axes=['feature'])
                copied_data['feature'] ^ copied_nodes[site]['input']
                condition_nodes[site] = (data_node, copied_data)

        input_idx = 0
        for site, (node, copied_node) in enumerate(
                zip(self._mats_env, copied_nodes)):
            if site in self._in_features:
                if condition_nodes[site] is None:
                    matrix_node = Node(
                        tensor=matrices[input_idx],
                        axes_names=('input', 'input_copy'),
                        name='virtual_result_sample_mat',
                        network=self,
                        virtual=True)
                    node['input'] ^ matrix_node['input']
                    matrix_node['input_copy'] ^ copied_node['input']
                    matrix_nodes[site] = matrix_node
                input_idx += 1
            else:
                node['input'] ^ copied_node['input']

        if canonical:
            right_envs = [None] * self._n_features
        else:
            right_envs = self._build_sample_right_envs(
                copied_nodes=copied_nodes,
                matrix_nodes=matrix_nodes,
                condition_nodes=condition_nodes,
                right_env=right_env,
                renormalize=renormalize)

        # Sweep from left to right, sampling one input node at a time.
        samples = []
        sample_indices = []
        input_idx = 0

        for site in range(self._n_features):
            if site in self._in_features:
                node = self._mats_env[site]
                copied_node = copied_nodes[site]
                right_env = right_envs[site]

                if canonical:
                    copied_node.reattach_edges(axes=['right'])
                    right_env = Node(
                        tensor=torch.eye(node['right'].size(),
                                         device=node.device,
                                         dtype=node.dtype),
                        axes_names=('right', 'right_copy'),
                        name='virtual_result_sample_right_env',
                        network=self,
                        virtual=True)
                    node['right'] ^ right_env['right']
                    copied_node['right'] ^ right_env['right_copy']

                # Replace only the physical edges to evaluate all candidates;
                # bond edges remain inherited from the stored environments.
                candidate = node.permute(tuple(range(node.rank)))
                copied_candidate = copied_node.permute(
                    tuple(range(copied_node.rank)))
                candidate.reattach_edges(axes=['input'])
                copied_candidate.reattach_edges(axes=['input'])
                candidate.disconnect('input')
                copied_candidate.disconnect('input')

                # A per-sample domain shares the samples' batch edge.
                per_sample = domains[input_idx].ndim == 2
                candidate_data = Node(
                    tensor=embeddings[input_idx],
                    axes_names=(('batch', 'domain_batch', 'feature')
                                if per_sample
                                else ('domain_batch', 'feature')),
                    name='sample_candidate_data',
                    network=self,
                    data=True)
                copied_candidate_data = Node(
                    tensor=candidate_data.tensor,
                    axes_names=candidate_data.axes_names,
                    name='sample_candidate_data_copy',
                    network=self,
                    data=True)
                copied_candidate_data = copied_candidate_data.conj()
                copied_candidate_data.reattach_edges(axes=['feature'])
                candidate_data['feature'] ^ candidate['input']
                copied_candidate_data['feature'] ^ \
                    copied_candidate['input']

                candidate = candidate_data @ candidate
                copied_candidate = copied_candidate_data @ copied_candidate
                probs_node = left_env @ candidate
                if right_env is not None:
                    probs_node = probs_node @ right_env
                probs_node = probs_node @ copied_candidate
                if probs_node.is_connected_to(probs_node):
                    probs_node @= probs_node

                domain_axis = probs_node.get_axis_num('domain_batch')
                probs = probs_node.tensor.movedim(domain_axis, -1)
                probs = probs.reshape(-1, embeddings[input_idx].shape[-2])
                if probs.shape[0] == 1 and n_samples > 1:
                    probs = probs.expand(n_samples, -1)

                probs = probs * quadrature_weights[input_idx].reshape(1, -1)
                probs = self._probabilities_from_amplitudes(probs)
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

                # Absorb the selected values into the reusable left environment.
                selected = node.permute(node.axes_names)
                copied_selected = copied_node.permute(copied_node.axes_names)
                selected.reattach_edges(axes=['input'])
                copied_selected.reattach_edges(axes=['input'])
                selected.disconnect('input')
                copied_selected.disconnect('input')

                selected_data = Node(
                    tensor=selected_embedding,
                    axes_names=('batch', 'feature'),
                    name='sample_selected_data',
                    network=self,
                    data=True)
                copied_selected_data = Node(
                    tensor=selected_data.tensor,
                    axes_names=selected_data.axes_names,
                    name='sample_selected_data_copy',
                    network=self,
                    data=True)
                copied_selected_data = copied_selected_data.conj()
                copied_selected_data.reattach_edges(axes=['feature'])
                selected_data['feature'] ^ selected['input']
                copied_selected_data['feature'] ^ copied_selected['input']

                left_env = self._contract_sample_site(
                    node=selected,
                    copied_node=copied_selected,
                    env=left_env,
                    data_nodes=(selected_data, copied_selected_data),
                    from_left=True,
                    renormalize=renormalize)
                input_idx += 1

            else:
                left_env = self._contract_sample_site(
                    node=self._mats_env[site],
                    copied_node=copied_nodes[site],
                    env=left_env,
                    from_left=True,
                    renormalize=renormalize)

            if canonical and site < (self._n_features - 1):
                right_axes = [axis.name for axis in left_env.axes
                              if ('right' in axis.name) and
                              (not axis.is_batch())]
                left_env.reattach_edges(axes=right_axes)
                for axis in right_axes:
                    left_env.disconnect(axis)

                copied_nodes[site + 1].reattach_edges(axes=['left'])
                left_env[right_axes[0]] ^ self._mats_env[site + 1]['left']
                left_env[right_axes[1]] ^ copied_nodes[site + 1]['left']

        # Collect values in the order of ``in_features``.
        samples = torch.stack(samples, dim=1)
        sample_indices = torch.stack(sample_indices, dim=1)

        self.reset()
        self.unset_data_nodes()
        if self._boundary == 'pbc':
            self._mats_env[-1]['right'] ^ self._mats_env[0]['left']
        elif canonical:
            for i in range(self._n_features - 1):
                self._mats_env[i]['right'] ^ self._mats_env[i + 1]['left']
            self._mats_env[-1]['right'] ^ self._right_node['left']

        if return_indices:
            return samples, sample_indices
        return samples
