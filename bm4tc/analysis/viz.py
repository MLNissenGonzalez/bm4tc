from typing import Optional, Tuple
import torch
import matplotlib.pyplot as plt
import numpy as np


def visualise_samples(
    samples: torch.FloatTensor,
    labels: Optional[torch.LongTensor] = None,
    gen_viz: Optional[int] = None,
    input_range: Optional[Tuple[float, float]] = None,
):
    """
    Visualise real or synthesised samples.
    If labels is None, samples are synthetic with shape (n, num_classes, data_dim).
    """
    if labels is None:
        n, num_classes, data_dim = samples.shape
        samples = samples.reshape(n * num_classes, data_dim)
        labels = torch.arange(num_classes).repeat(n)

    if samples.shape[1] == 2:
        return create_2d_scatter(X=samples, t=labels, input_range=input_range)
    else:
        if gen_viz is None:
            gen_viz = samples.shape[0]
        raise ValueError("Higher data dimension not yet implemented.")


def create_2d_scatter(
    X: torch.FloatTensor,
    t: torch.LongTensor,
    title=None,
    ax=None,
    show_legend=True,
    input_range: Optional[Tuple[float, float]] = None,
):
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    if torch.is_tensor(t):
        t = t.detach().cpu().numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    classes = np.unique(t)
    for cls in classes:
        idx = (t == cls)
        ax.scatter(X[idx, 0], X[idx, 1], s=5, label=f'Class {cls}')

    if input_range is not None:
        lo, hi = input_range
    else:
        margin = 0.05 * (X.max() - X.min())
        lo, hi = X.min() - margin, X.max() + margin

    if title:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.grid(True)
    if show_legend:
        ax.legend(title="Class")
    return ax
