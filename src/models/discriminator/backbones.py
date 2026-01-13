import torch
from torch import nn
from typing import *
from math import ceil

# TODO: Fit pretrained Bornmachine in this.

# TODO: Number of classes not needed. feature dim kind of ugly. 

# Backbone could be MLP of some type or BornMachine. Abstract class below
class BackBone(nn.Module):
    def __init__(self, data_dim: int, pretrained: bool):
        super().__init__()
        self.data_dim = data_dim # D
        self.out_dim = None # F, assigned by the specific architecture
        self.pretrained = pretrained

    def reset(self):
        """
        For a backbone with virtual state, like a model based on tensorkrowch, the reset method is necessary to save the statedict.
        """
        return NotImplementedError
    
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        
    def forward(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """
        Parameters
        ----------
        data: Tensor
            shape = (N*C, D)

        Returns
        -------
        features: Tensor
            shape = (N*C, F)
        """
        return NotImplementedError


class MLP(BackBone):
    def __init__(self, data_dim: int,
                 hidden_multipliers: List[float], nonlinearity: str, 
                 negative_slope: float | None = None, pretrained: bool = False):
        super().__init__(data_dim, pretrained)

        # Determine activation
        act = nonlinearity.replace(" ", "").lower()
        if act == "relu":
            def get_activation(): return nn.ReLU()
        elif act == "leakyrelu":
            if negative_slope is None:
                raise ValueError("LeakyReLU needs negative_slope parameter.")

            def get_activation(): return nn.LeakyReLU(negative_slope)
        else:
            raise ValueError(
                f"{nonlinearity} not recognised. Try ReLU or LeakyReLU.")

        # Determine hidden_dims
        if hidden_multipliers is None or len(hidden_multipliers) == 0:
            raise ValueError(
                "hidden_multipliers must be provided as a list of floats.")

        hidden_dims = [max(1, ceil(mult * self.data_dim))
                       for mult in hidden_multipliers]
        self.out_dim = hidden_dims[-1]

        # Build layers: (N, D) -> (N, F), # F = self.out_dim
        layers = [nn.Linear(self.data_dim, hidden_dims[0])]
        for i in range(len(hidden_dims) - 1):
            layers.append(get_activation())
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.stack = nn.Sequential(*layers)

    def reset(self):
        pass

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.stack(x)


_ARCHITECTURE_MAPPING = {
    "mlp": MLP,
    #"conv": ConvBackBone,
    #"ae": AEBackBone,
    # "born": BornMachineBackbone initialize from trained bornmachine
}

def get_backbone(name: str, data_dim: int, 
                 pretrained: bool, model_kwargs: dict) -> BackBone:
    name=name.lower().replace("-", "").replace(" ", "")
    return _ARCHITECTURE_MAPPING[name](data_dim=data_dim, **model_kwargs, pretrained=pretrained)