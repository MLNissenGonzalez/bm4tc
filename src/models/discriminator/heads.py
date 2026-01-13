import torch
from torch import nn
from typing import *
from math import ceil

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#---------AWARE HEADS FOR GAN style TRAINING---------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

class AwareHead(nn.Module):
    def __init__(self, feature_dim: int, num_cls: int, device: torch.device):
        super().__init__()
        self.feature_dim, self.num_cls = feature_dim, num_cls
        self.device = device

    def forward(self, features: torch.FloatTensor) -> torch.FloatTensor:
        """
        Class Aware. 

        Parameters
        ----------
        feat: Tensor
            shape (N, C, F)

        Returns
        -------
        logits: Tensor
            shape (N, C)
        """
        return NotImplementedError

class AwareLinearHead(AwareHead):
    def __init__(self, feature_dim: int, num_cls: int, device: torch.device):
        super().__init__(feature_dim, num_cls, device)
        self.weight = nn.Parameter(torch.randn(self.num_cls, self.feature_dim))
        self.bias = nn.Parameter(torch.zeros(self.num_cls))
        self.weight.to(device)
        self.bias.to(device)

    def forward(self, features: torch.FloatTensor):
        # features (N, C , F)
        logits = torch.einsum("ncf, cf -> nc", features, self.weight) + self.bias
        return logits
    
    def to(self, device):
        self.weight.to(device)
        self.bias.to(device)
        self.device = device
    
_AWARE_HEADS = {
    "linear": AwareLinearHead
}
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#---------AWARE HEADS FOR GANSTYLE AND ADVERSARIAL TRAINING------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

# If used in adversarial training, these give probability of being natural vs adversarial example
class AgnosticHead(nn.Module):
    def __init__(self, feature_dim: int, device: torch.device):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device

    def forward(self, features: torch.FloatTensor) -> torch.FloatTensor:
        """
        Unaware of class identity. 

        Parameters
        ----------
        feat: Tensor
            shape (N, F)
        
        Returns
        -------
        logits: Tensor
            shape (N,)
        """
        return NotImplementedError
    
class AgnosticProjectionHead(nn.Module):
    def __init__(self, feature_dim: int, device: torch.device):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(feature_dim))
        self.bias = nn.Parameter(torch.zeros(1))
        self.weight.to(device)
        self.bias.to(device)
        self.device = device

    def forward(self, features: torch.FloatTensor):
        # feat: (N, F)
        return (features * self.weight).sum(dim=1) + self.bias # returns: (N,)
    
    def to(self, device):
        self.weight.to(device)
        self.bias.to(device)
        self.device = device
    
class AgnosticMLPHead(AgnosticHead):
    def __init__(self, feature_dim: int, device: torch.device, 
                 hidden_multipliers: List[float], nonlinearity: str, 
                 negative_slope: float | None = None):
        super().__init__(feature_dim, device)

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

        hidden_dims = [max(1, ceil(mult * self.feature_dim))
                       for mult in hidden_multipliers]

        # Build layers: (N, F) -> (N)
        layers = [nn.Linear(self.feature_dim, hidden_dims[0])]
        for i in range(len(hidden_dims) - 1):
            layers.append(get_activation())
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        layers.append(get_activation())
        layers.append(nn.Linear(hidden_dims[-1], 1))  

        self.stack = nn.Sequential(*layers)
        self.stack.to(device)

    def forward(self, features: torch.FloatTensor) -> torch.FloatTensor:
        logits : torch.FloatTensor = self.stack(features) # x: (N, F)
        return logits.squeeze(-1) # returns: (N,)
    
    def to(self, device):
        self.stack.to(device)
        self.device = device

_AGNOSTIC_HEADS = {
    "linear": AgnosticProjectionHead,
    "projection": AgnosticProjectionHead,
    "mlp": AgnosticMLPHead
}
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#---------GetterFunction-----------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

def get_head(class_aware: bool, name: str, model_kwargs: dict, 
             feature_dim: int, num_cls: int, device: torch.device) -> AwareHead | AgnosticHead:
    name=name.lower().replace("-", "").replace(" ", "")
    if class_aware:
        return _AWARE_HEADS[name](feature_dim, num_cls, device, **model_kwargs)
    else:
        return _AGNOSTIC_HEADS[name](feature_dim, device, **model_kwargs)