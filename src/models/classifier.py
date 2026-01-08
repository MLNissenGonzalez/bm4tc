from typing import *
import torch
import tensorkrowch as tk
import src.utils.schemas as schemas

# As init, take either basic config or tensors (optional). 
class BornClassifier(tk.models.MPSLayer):
    def __init__(
            self,
            embedding: Callable[[torch.Tensor, int], torch.Tensor],
            n_features: Optional[int] = None,
            in_dim: Optional[Union[int, Sequence[int]]] = None,
            out_dim: Optional[int] = None,
            bond_dim: Optional[Union[int, Sequence[int]]] = None,
            out_position: Optional[int] = None,
            boundary: Text = 'obc',
            tensors: Optional[Sequence[torch.Tensor]] = None,
            n_batches: int = 1,
            init_method: Text = 'randn',
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            **kwargs
            ):
        super().__init__(
            n_features=n_features, in_dim=in_dim, out_dim=out_dim,
            bond_dim=bond_dim, out_position=out_position, 
            boundary=boundary, tensors=tensors, n_batches=n_batches,
            init_method=init_method, device=device, dtype=dtype, **kwargs
        )
        self.embedding = embedding

    def embed(self,
              data: torch.Tensor):
        in_dim = self.in_dim[0]
        return self.embedding(data, in_dim)
    
    def prepare(self, 
                tensors: List[torch.Tensor] | None = None, 
                device: torch.device | None = None,
                train_cfg: schemas.ClassificationConfig | None = None):
        """
        Preperation of tensor network for classification as implemented in `tensorkrowch` tutorials.
        """

        self.unset_data_nodes(), self.reset()
        if tensors is not None:
            self.initialize(tensors=tensors)
        self.to(device)
        if train_cfg is not None:
            self.auto_stack, self._auto_unbind = train_cfg.auto_stack, train_cfg.auto_unbind
        self.trace(torch.zeros(1, len(self.in_features),
                        self.in_dim[0]).to(device))
        
    def parallel(
            self,  # could use MPSLayer class for this one actually
            embs: torch.Tensor
            ) -> torch.Tensor:
        """
        Parallel contraction of an MPS with embedded input and computation of Born probabilities.
        Used primarily for **classification**.
        The function computes unnormalized conditional probabilities p(c | x₁,…,x_D) and normalizes
        them across classes.

        Parameters
        ----------
        embs : torch.Tensor
            Embedded inputs with shape (batch_size, D, phys_dim).

        Returns
        -------
        torch.Tensor
            (batch_size, num_cls): Unnormalized conditional probabilities p(c | x)

        Raises
        ------
        TypeError
            If `embs` is not a `torch.Tensor`.
        """

        # embs=tensor (parallizable), assume mps.out_feature = [cls_pos] globally??
        if not isinstance(embs, torch.Tensor):
            raise TypeError(
                "embs input must be a tensor of shape (batch_size, D, phys_dim).")

        
        p = torch.square(self.forward(data=embs))
        return p / p.sum(dim=-1, keepdim=True)
        
    
    def probabilities(self, data: torch.Tensor):
        "Returns class probability given unembedded input."
        embs = self.embed(data)
        return self.parallel(embs)

    def eval(self):
        super().eval()