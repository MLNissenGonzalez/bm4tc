from typing import  * 
import torch
import matplotlib.pyplot as plt
import numpy as np

# TODO: Implement this also for other visualisable datatypes (ADDED AS ISSUE)
def visualise_samples(samples: torch.FloatTensor, labels: Optional[torch.LongTensor] = None, gen_viz: Optional[int] = None):
    """
    Visualise real or synthetised samples. 
    If t==None, then samples are synthesised and 
    expected to be of the shape (n, num classes, data dim), 
    else data is real and of shape (N, data dim).
    gen_viz tells us how many samples should be visualised for higher dimensional cases (MNIST).

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
        return create_2d_scatter(X=samples, t=labels)
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
    show_legend=True
):
    """
    Create a 2D scatter plot that handles both numpy arrays and torch tensors
    and can be embedded in larger figures.
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

    if title:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect('equal')
    ax.grid(True)
    if show_legend:
        ax.legend(title="Class")
    return ax