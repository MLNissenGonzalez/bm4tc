import torch
import tensorkrowch as tk
from src.models.generator.differential_sampling import main as diff_sampling
from typing import *
import src.utils.schemas as schemas
import logging
logger = logging.getLogger(__name__)

# TODO: Add docstrings for API versions of sampling method.
class BornGenerator(tk.models.MPS):
    """
    Generator view of `BornMachine`. 
    
    Implements Born rule by purely sequential contraction, such that 
    the model paramaters as handled by `tensorkrowch` keep there identity. This essential as the Born rule is
    implemented `data_dim` amount of times to obtain a sample only after which backpropagation in the GAN-style 
    training regime is issued. If model paramaters would change with every computation of a conditional probability,
    gradients would not propagate.

    Implements sampling algorithms on the basis of the sequential contraction of the tensor network, either for one class or 
    all of them, using a differentiable sampling algorithm, that is specified in the sampling config parameter.
    """
    def __init__(
            self,
            tensors: List[torch.Tensor],
            embedding: Callable[[torch.Tensor, int], torch.Tensor],
            cls_pos: int,
            in_dim: int,
            num_cls: int,
            input_range: Tuple[float, float],
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
            ):
        super().__init__(
            tensors=tensors, device=device, dtype=dtype
        )
        self.embedding = embedding
        self.input_range = input_range
        self.input_space: torch.FloatTensor | None = None
        self.cls_pos, self.in_dim, self.num_cls = cls_pos, in_dim, num_cls
        self.cls_embs = []
        for cls in range(self.num_cls):
            emb = tk.embeddings.basis(torch.tensor(cls), self.num_cls).float()
            self.cls_embs.append(emb.to(device))
        self.device = device

        # Initialise the virtual tensorkrowch MPS for computing the partition function
        # Needs to share the exact same tensors as self
        # But we need a seperate instance to avoid problems whith the computational graph
        # Since mps.norm() does a full contraction with itself, which is achieved by changing the 
        # virtual node structure. THis is another virtual node structure than the one used for 
        # A simple forward pass to compute amplitudes.
        # Maybe it is better to copy like it is done inside .norm()?

        self.virtual_mps = self.copy(share_tensors=True)

    def prepare(self):
        """Reset MPS state and clear data nodes for a fresh sampling pass."""
        self.reset()
        self.unset_data_nodes()

    def to(self, device):
        """Move generator and its embeddings to the specified device."""
        super().to(device)
        self.device = device
        self.cls_embs = [emb.to(device) for emb in self.cls_embs]
        # Also move input_space to the device if it exists
        if self.input_space is not None:
            self.input_space = self.input_space.to(device)
        return self
    
    def reset(self):
        """Reset MPS state, ensuring input_space is on the correct device."""
        super().reset()
        # If input_space exists but is on a different device, move it to self.device
        if self.input_space is not None and self.input_space.device != self.device:
            self.input_space = self.input_space.to(self.device)

    def sequential(
            self, 
            embs: Dict[int, torch.Tensor] | torch.Tensor
            ) -> torch.Tensor:  # very flexible, not parallizable), but expects input to be embedded (ok.)
        
        """
        Sequential contraction of an MPS with embedded input and computation of probabilities
        using the Born rule. Used primarily for **sampling**, since it supports marginalization
        over unobserved variables.

        Behavior
        --------
        - **Partial inputs (marginalized case):**
        When not all sites are included in `embs`, i.e. p({x_i}_I | {x_j}_J),
        the function marginalizes over missing sites by computing a reduced density
        matrix ρ. The diagonal entries of ρ represent unnormalized marginal probabilities.

        - **Complete inputs (full contraction):**
        When all sites are present, the function computes the amplitude ψ(x₁,…,x_D)
        and returns its squared magnitude |ψ|² as the unnormalized joint distribution.

        Parameters
        ----------
        mps : tk.models.MPS
            The tensor network representing the probability amplitude ψ(x).
        embs : dict[int, torch.Tensor]
            A mapping from site index → embedded tensor of shape (batch_size, phys_dim).
            The keys specify which input sites are provided. Must be a dict, not a tensor.

        Returns
        -------
        torch.Tensor
            The unnormalized probability distribution(s):
            - (batch_size, num_bins) if marginalized,
            - (batch_size, 1) if all variables included.

        Raises
        ------
        TypeError
            If `embs` is not a dictionary.
        """

        if not isinstance(embs, dict):
            raise TypeError(
                "embs input must be a dictionary with site indices as keys and "
                "embedded tensors as values."
            )

        # Identify which features are fed into the MPS, including class site.
        self.in_features = [i for i in embs.keys()]
        in_tensors = [embs[i] for i in self.in_features]

        # Case 1: Not all variables appear, thus marginalize_output=True and one has to take the diagonal.
        if len(in_tensors) < self.n_features:
            rho = self.forward(in_tensors, marginalize_output=True,
                            inline_input=True, inline_mats=True)  # density matrix
            p = torch.diagonal(rho)
        # Case 2: All variables appear.
        else:
            amplitude = self.forward(
                data=in_tensors, inline_input=True, inline_mats=True)  # prob. amplitude
            p = torch.square(amplitude)
        return p
    
    def _single_class(
            self,
            method: str,
            cls_emb: torch.Tensor,
            num_spc: int,
            num_bins: int
            ) -> torch.Tensor:
        """
        Core class-conditional sampling routine for a single class embedding.

        This function sequentially samples each non-class site of an MPS according to
        its Born probabilities, conditioned on a fixed class embedding at position
        `cls_pos`. Sampling at each step is handled by `diff_sampling(p, input_space, method)`,
        which allows configurable sampling strategies (e.g., secant).

        Parameters
        ----------
        method : str
            Sampling method identifier passed to `diff_sampling` (e.g., "secant").
        cls_emb : torch.Tensor
            Precomputed embedding of the target class; shape `(num_cls,)`.
        num_bins : int
            Number of candidate discrete values for each feature (resolution of `input_space`).
        num_spc : int
            Number of samples to generate for this class.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(num_spc, data_dim)` where `data_dim = mps.n_features - 1`.
            Each row is a sampled point in the original real-valued input space.
        """
        data_dim = self.n_features - 1
        
        if num_spc == 0:
            # Return empty tensor with correct feature dimension
            logger.info(
                f"No real examples of class {cls_emb.argmax()} in this batch.")
            return torch.empty((0, data_dim), device=self.device)

        samples = []
        embs = {}  # best way to save embeddings and their position

        embs[self.cls_pos] = cls_emb[None, :].expand(num_spc * num_bins, -1).to(self.device)

        # Input embedding TODO: Could move outside, as class and batch independent
        in_emb = self.embedding(self.input_space, self.in_dim)  # [num_bins, in_dim]
        # [num_samples, num_bins, in_dim]
        in_emb = in_emb[None, :, :].expand(num_spc, -1, -1)
        # [num_samples * num_bins, phys_dim]
        in_emb = in_emb.reshape(num_spc * num_bins, self.in_dim).to(self.device)

        for site in range(data_dim + 1):
            if site == self.cls_pos:
                continue# TODO: Add docstrings for API versions of sampling method.

            # This missing caused a silent bug.
            embs[site] = in_emb

            # This reset has to be performed as born_sequential assigns new in features everytime new embedding is added
            self.prepare()

            # Compute marginal probability over the bins for the current site for each sample.
            p = self.sequential(embs=embs).view(
                num_spc, num_bins)  # shape (num_spc, num_bins)

            # Draw one bin value per sample and site from p, TODO: Make configable
            feature = diff_sampling(p, self.input_space, method)
            samples.append(feature)  # shape (num_spc,)

            # Embed drawn samples and use for conditioning for the next site.
            embs[site] = self.embedding(samples[-1], self.in_dim
                                )[:, None, :].expand(-1, num_bins, -1
                                                        ).reshape(num_spc*num_bins, -1)

        # Stack sampled per-site arrays across sites
        samples = torch.stack(tensors=samples, dim=1)  # shape (num_spc, data_dim)
        return samples

    def sample_single_class(self, cls: int, cfg: schemas.SamplingConfig) -> torch.Tensor:
        """
        Sample from the generator for a single class.

        Args:
            cls: Class index to sample from.
            cfg: Sampling configuration with num_bins, num_spc, batch_spc, method.

        Returns:
            Tensor of shape (num_spc, data_dim) with sampled points.
        """
        # Create the 1D grid (input_space) and instanciate the embedding callable
        if (self.input_space is None) or (not self.input_space.shape[0] == cfg.num_bins):
            self.input_space = torch.linspace(self.input_range[0], self.input_range[1], cfg.num_bins, device=self.device)
        cls_emb = self.cls_embs[cls]

        isBatched = (cfg.num_spc > cfg.batch_spc)
        if not isBatched:
            samples = self._single_class(method=cfg.method, cls_emb=cls_emb, 
                                         num_spc=cfg.num_spc, num_bins=cfg.num_bins).cpu()
        else:
            samples = []
            num_batches = (cfg.num_spc + cfg.batch_spc - 1) // cfg.batch_spc
            batches = []
            for _ in range(num_batches):
                batch = self._single_class(method=cfg.method, cls_emb=cls_emb, 
                                           num_spc=cfg.batch_spc, num_bins=cfg.num_bins).cpu()
                batches.append(batch)
            samples = torch.cat(batches, dim=0)
            batches.clear()
            samples = samples[:cfg.num_spc, :]
        return samples
    
    def _all_classes(
            self,
            method: str,
            num_spc: int,
            num_bins: int) -> torch.Tensor:
        """
        Internal helper to sample multiple classes in a single call, assuming
        precomputed embeddings and input grid.

        Loops over a list of class embeddings, invoking `_single_class` for each,
        and stacks the results along the class dimension.

        Parameters
        ----------
        method : str
            Sampling method passed to `diff_sampling`.
        num_spc: int
            Number of samples to be sampled per class.
        num_bins: int
            Number of candidate discrete values for each feature (resolution of `input_space`).

        Returns
        -------
        torch.Tensor
            Tensor of shape `(num_spc, num_cls, data_dim)`.
        """

        samples = []
        for cls_emb in self.cls_embs:
            cls_samples = self._single_class(method, cls_emb, num_spc, num_bins)  # shape (num_spc, data_dim)
            samples.append(cls_samples)

        # shape: (n_spc, num_cls, data_dim)
        samples = torch.stack(tensors=samples, dim=1)
        return samples

    
    def sample_all_classes(self, cfg: schemas.SamplingConfig) -> torch.Tensor:
        """
        Sample from the generator for all classes.

        Args:
            cfg: Sampling configuration with num_bins, num_spc, batch_spc, method.

        Returns:
            Tensor of shape (num_spc, num_classes, data_dim) with sampled points.
        """
        # Create the 1D grid (input_space) and instanciate the embedding callable, and send it to device.
        if self.input_space is None or not self.input_space.shape[0] == cfg.num_bins:
            self.input_space = torch.linspace(self.input_range[0], self.input_range[1], cfg.num_bins, device=self.device)
        
        # Perform sampling batched, if desired number of samples per classs is higher than batch_size per class. 
        isBatched = (cfg.batch_spc < cfg.num_spc)
        if not isBatched:
            samples = self._all_classes(method=cfg.method, num_spc=cfg.num_spc, 
                                        num_bins=cfg.num_bins).cpu()

        else:
            num_batches = (cfg.num_spc + cfg.batch_spc - 1) // cfg.batch_spc
            batches = []
            for _ in range(num_batches):  # could compute these batches in parallel, not to bad
                batch = self._all_classes(method=cfg.method, num_spc=cfg.batch_spc, 
                                          num_bins=cfg.num_bins).cpu()
                batches.append(batch)

            # (batch_size x num_batches, num_cls, data.dim)
            samples = torch.concat(tensors=batches, dim=0)
            batches.clear()
            samples = samples[:cfg.num_spc, :, :]    
        
        
        return samples
    
    def log_partition_function(self) -> torch.FloatTensor:
        """
        Computes the log partition function of the BornMachine. (without embedding matrix).
        Is basically a copied from tensorkrowch.MPS.norm() method with log_scale=True
        and without the sqrt. Also, removed the option of complex tensors.
        
        Think about setting up the contraction network (copying of nodes etc) once at the start.
        Such that we init the virtual structure only once.
        
        This method internally removes all data nodes in the MPS, if any, and
        contracts the nodes with themselves. Therefore, this may alter the
        usual behaviour of :meth:`contract` if the MPS is not
        :meth:`~tensorkrowch.TensorNetwork.reset` afterwards. Also, if the MPS
        was contracted before with other arguments, it should be ``reset``
        before calling ``norm`` to avoid undesired behaviour.
        
        Since the norm is computed by contracting the MPS, it means one can
        take gradients of it with respect to the MPS tensors, if it is needed.
        
        """
        if self.virtual_mps._data_nodes:
            self.virtual_mps.unset_data_nodes()
        
        # All nodes belong to the output region
        all_nodes = self.virtual_mps.mats_env[:]
        
        if self.virtual_mps._boundary == 'obc':
            all_nodes[0] = self.virtual_mps._left_node @ all_nodes[0]
            all_nodes[-1] = all_nodes[-1] @ self.virtual_mps._right_node
        
        # Check if nodes are already connected to copied nodes
        create_copies = []
        for node in all_nodes:
            neighbour = node.neighbours('input')
            if neighbour is None:
                create_copies.append(True)
            else:
                if 'virtual_result_copy' not in neighbour.name:
                    raise ValueError(
                        f'Node {node} is already connected to another node '
                        'at axis "input". Disconnect the node or reset the '
                        'network before calling `norm`')
                else:
                    create_copies.append(False)
        
        if any(create_copies) and not all(create_copies):
            raise ValueError(
                'There are some nodes connected and some disconnected at axis '
                '"input". Disconnect all of them before calling `norm`')
        
        create_copies = any(create_copies)
        
        # Copy output nodes sharing tensors
        if create_copies:
            copied_nodes = []
            for node in all_nodes:
                copied_node = node.__class__(shape=node._shape,
                                             axes_names=node.axes_names,
                                             name='virtual_result_copy',
                                             network=self.virtual_mps,
                                             virtual=True)
                copied_node.set_tensor_from(node)
                copied_nodes.append(copied_node)
                
                # Change batch names so that they not coincide with
                # original batches, which gives dupliicate output batches
                for ax in copied_node.axes:
                    if ax._batch:
                        ax.name = ax.name + '_copy'
            
            # Connect copied nodes with neighbours
            for i in range(len(copied_nodes)):
                if (i == 0) and (self.virtual_mps._boundary == 'pbc'):
                    if all_nodes[i - 1].is_connected_to(all_nodes[i]):
                        copied_nodes[i - 1]['right'] ^ copied_nodes[i]['left']
                elif i > 0:
                    copied_nodes[i - 1]['right'] ^ copied_nodes[i]['left']
            
            # Reattach input edges of resultant output nodes and connect
            # with copied nodes
            for node, copied_node in zip(all_nodes, copied_nodes):
                # Reattach input edges
                node.reattach_edges(axes=['input'])
                
                # Connect copies directly to output nodes
                copied_node['input'] ^ node['input']
        else:
            copied_nodes = []
            for node in all_nodes:
                copied_nodes.append(node.neighbours('input'))
            
        # Contract output nodes with copies
        mats_out_env = self.virtual_mps._input_contraction(
            nodes_env=all_nodes,
            input_nodes=copied_nodes,
            inline_input=True)
        
        # Contract resultant matrices (inline mats explicitly here)
        log_Z = 0
        result_node = mats_out_env[0]
        log_Z += result_node.norm().log()
        result_node = result_node.renormalize()
                
        for node in mats_out_env[1:]:
            result_node @= node
            log_Z += result_node.norm().log()
            result_node = result_node.renormalize()
        
        # Contract periodic edge
        if result_node.is_connected_to(result_node):
            result_node @= result_node
            log_Z += result_node.norm().log()
            result_node = result_node.renormalize()

        return log_Z
    
    def unnormalized_prob(
            self,
            data: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.FloatTensor:
        """
        Compute log(|psi(x,c)|^2) for each sample.

        This should use the generator's sequential method with full contraction
        (because on internal tensorkrowch stuff) to compute the squared amplitude for each (data, label) pair.

        Args:
            bornmachine: BornMachine instance with generator view.
            data: Input tensor of shape (batch_size, data_dim).
            labels: Class labels of shape (batch_size,).

        Returns:
            Tensor of shape (batch_size,) with log unnormalized probabilities.
        """
        data_embs = self.embedding(data, self.in_dim)
        class_embs = tk.embeddings.basis(labels, self.num_cls).float()

        # start with data embeddings as a list
        embs = [data_embs[:, i, :] for i in range(data_embs.shape[1])]

        # insert class embedding at cls_pos
        embs.insert(self.cls_pos, class_embs)

        # Compute amplitude
        amplitude = self.forward(
                data=embs, inline_input=True, inline_mats=True)  # prob. amplitude

        return torch.square(amplitude)