import torch
from torch import nn
from typing import *
from math import ceil
import ast


# TODO: Fit pretrained Bornmachine in this.

# TODO: Number of classes not needed. feature dim kind of ugly. 

# Backbone could be MLP of some type or BornMachine. Abstract class below
class BackBone(nn.Module):
    def __init__(self, data_dim: int, pretrained: bool, device: torch.device):
        super().__init__()
        self.data_dim = data_dim # D
        self.out_dim = None # F, assigned by the specific architecture
        self.pretrained = pretrained
        self.device = device
    
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
    def __init__(self, data_dim: int, device: torch.device,
                 hidden_multipliers: List[float], nonlinearity: str, 
                 negative_slope: float | None = None, pretrained: bool = False):
        super().__init__(data_dim, pretrained, device)

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
        hidden_dims = [max(1, ceil(mult * self.data_dim))
                       for mult in hidden_multipliers]
        self.out_dim = hidden_dims[-1]

        # Build layers: (N, D) -> (N, F), # F = self.out_dim
        layers = [nn.Linear(self.data_dim, hidden_dims[0])]
        for i in range(len(hidden_dims) - 1):
            layers.append(get_activation())
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.stack = nn.Sequential(*layers)
        self.stack.to(device)

    def reset(self):
        pass

    def to(self, device):
        self.stack.to(device)
        self.device = device

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.stack(x)


_ARCHITECTURE_MAPPING = {
    "mlp": MLP,
    #"conv": ConvBackBone,
    #"ae": AEBackBone,
    # "born": BornMachineBackbone initialize from trained bornmachine
}

def get_backbone(name: str, data_dim: int, device: torch.device,
                 pretrained: bool, model_kwargs: dict) -> BackBone:
    name=name.lower().replace("-", "").replace(" ", "")
    return _ARCHITECTURE_MAPPING[name](data_dim=data_dim, device=device, **model_kwargs, pretrained=pretrained)