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
            self.cls_embs.append(tk.embeddings.basis(
                torch.tensor(cls), self.num_cls).float())
        self.device = device

    def prepare(self):
        self.reset()
        self.unset_data_nodes()

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

        # Identify which features are fed into the MPS.
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

    # Created a batched version of this
    def sample_single_class(self, cls: int, cfg: schemas.SamplingConfig):
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

    
    def sample_all_classes(self, cfg: schemas.SamplingConfig):
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
    
