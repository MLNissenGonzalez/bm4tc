# Minimal implementation of evasion attacks

import torch
from torch.utils.data import DataLoader
from typing import List
import src.utils.get as get
from src.models import *

# TODO: Implement the below (maybe) with the BornMachine class
def normalizing(x: torch.FloatTensor, norm: int | str):
    """
    Normalize a tensor of shape (batch size, data dim)
    along the data dim (flattened).
    """
    if norm == "inf":
        normalized = x.sign()

    elif isinstance(norm, int):
        if norm < 1:
            raise ValueError("Only accept p >= 1.")
        x_norm = x.norm(p=norm, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=1e-12)
        normalized = x / x_norm

    else:
        raise ValueError(f"{norm=}, but expected to be int or 'inf'.")
    
    return normalized


class FastGradientMethod:
    def __init__(
            self,
            norm: int | str = "inf",
            criterion: str = "crossentropy"
    ):
        """
        Initialize FGM with chosen norm and loss function.
        """
        self.norm = norm
        self.criterion = get.criterion(criterion)

    def generate(
            self,
            classifier: BornClassifier,
            naturals: torch.Tensor,
            labels: torch.LongTensor,
            strength: float = 0.1,
            device: torch.device | str = "cpu"
    ):
        """
        Generate adversarial examples from natural examples
        based on a given criterion.
        """
        classifier.to(device)
        naturals = naturals.to(device).detach().clone().requires_grad_(True)
        labels = labels.to(device)

        # Forward and backward pass
        predictions = classifier(naturals)
        loss = self.criterion(predictions, labels)

        classifier.zero_grad()
        if naturals.grad is not None:
            naturals.grad.zero_()

        loss.backward()

        grad = naturals.grad.detach()  # shape: (batch size, data dim)
        normalized_gradient = normalizing(grad, norm=self.norm)

        ad_examples = naturals + strength * normalized_gradient
        ad_examples = ad_examples.detach()

        return ad_examples


_METHOD_MAP = {
    "FGM": FastGradientMethod
}


class RobustnessEvaluation:
    def __init__(
            self,
            method: str = "FGM",
            norm: int | str = "inf",
            criterion: str = "crossentropy",
            strengths: List[float] = [0.1, 0.3]
    ):
        self.strengths = strengths
        self.method = _METHOD_MAP[method](
            norm=norm,
            criterion=criterion
        )

    def generate(
            self,
            classifier: BornClassifier,
            naturals: torch.Tensor,
            labels: torch.LongTensor,
            strength: float,
            device: torch.device | str = "cpu"
    ):
        return self.method.generate(
            classifier, naturals, labels, strength, device
        )

    def evaluate(
            self,
            classifier: BornClassifier,
            loader: DataLoader,
            device: torch.device | str = "cpu"
    ):
        """
        Evaluate robustness of a classifier over multiple perturbation strengths.
        """
        classifier.to(device)
        classifier.eval()

        strength_acc = []

        for strength in self.strengths:
            batch_acc = []

            for naturals, labels in loader:
                ad_examples = self.generate(
                    classifier, naturals, labels, strength, device
                )

                with torch.no_grad():
                    ad_output = classifier(ad_examples)
                    ad_pred = torch.argmax(ad_output, dim=1)
                    acc = (ad_pred == labels.to(device)).float().mean().item()
                    batch_acc.append(acc)

            mean_acc = sum(batch_acc) / len(batch_acc)
            strength_acc.append(mean_acc)

        return strength_acc

