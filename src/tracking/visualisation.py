from typing import  *
import torch
import matplotlib.pyplot as plt
import numpy as np

# TODO: Implement this also for other visualisable datatypes (ADDED AS ISSUE)
def visualise_samples(
    samples: torch.FloatTensor,
    labels: Optional[torch.LongTensor] = None,
    gen_viz: Optional[int] = None,
    input_range: Optional[Tuple[float, float]] = None,
):
    """
    Visualise real or synthetised samples.
    If t==None, then samples are synthesised and
    expected to be of the shape (n, num classes, data dim),
    else data is real and of shape (N, data dim).
    gen_viz tells us how many samples should be visualised for higher dimensional cases (MNIST).

    Parameters
    ----------
    samples : torch.FloatTensor
        Data to visualise.
    labels : torch.LongTensor, optional
        Class labels. If None, samples are treated as synthetic with shape
        (n, num_classes, data_dim).
    gen_viz : int, optional
        Number of samples to visualise for higher-dimensional data.
    input_range : tuple of (float, float), optional
        Axis limits (lo, hi) matching the embedding domain.
        If None, inferred from data.

    Returns
    -------
    ax
        axis object of matplotlib (either image or scatter plot)
    """
    if labels is None:
        n, num_classes, data_dim = samples.shape
        # (n*num_classes, data_dim)
        samples = samples.reshape(n*num_classes, data_dim)
        labels = torch.arange(num_classes).repeat(n)    # (n*num_classes,)

    if samples.shape[1] == 2:
        return create_2d_scatter(X=samples, t=labels, input_range=input_range)
    else:
        if gen_viz is None:
            # Can be used to visualise only a limited amount of examples
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
    """
    Create a 2D scatter plot that handles both numpy arrays and torch tensors
    and can be embedded in larger figures.

    Parameters
    ----------
    input_range : tuple of (float, float), optional
        Axis limits (lo, hi). If None, inferred from data with a small margin.
    """
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