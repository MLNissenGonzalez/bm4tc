# utils that are practical for experiments
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from typing import Optional, Dict, Any
from schemas import CriterionConfig
from pathlib import Path
import hydra

def save_model(model: torch.nn.Module, run_name: str, model_type: str):
    """
    Save model inside the Hydra run's output directory:
    ${hydra:run.dir}/models/{model_type}_{run_name}.pt
    """
    assert model_type in ["pre_mps", "pre_dis", "gan_mps", "gan_dis"], \
        f"Invalid model_type {model_type}"

    # Hydra's current run dir
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Models subfolder inside it
    folder = run_dir / "models"
    folder.mkdir(parents=True, exist_ok=True)

    # Save path
    filename = f"{model_type}_{run_name}.pt"
    save_path = folder / filename

    # Save state dict
    torch.save(model.state_dict(), save_path)

    return str(save_path)


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------Criterion-----------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Using classes instead of functions in case I want to you use loss functions with more hyperparameters and/or learnable parameters

# TODO: Add documentation

class MPSNLLL(nn.Module):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        p = p.clamp(min=self.eps)
        return -torch.log(p[torch.arange(p.size(0)), t]).mean()

_LOSS_MAP = {
    "nll": MPSNLLL,
    "nlll": MPSNLLL,
    "negativeloglikelihood": MPSNLLL,
    "negloglikelihood": MPSNLLL,
    "bce": nn.BCELoss, # BCE loss expects probabilities and target is float between 0 and 1
    "binarycrossentropy": nn.BCELoss,
    "vanilla": nn.BCELoss # TODO: Has to be adapted to the swapped logarithm
}

def get_criterion(cfg: CriterionConfig):
    key = cfg.name.replace(" ", "").replace("-", "").lower()
    if key not in _LOSS_MAP:
        raise ValueError(f"Loss '{cfg.name}' not recognised")
    # use empty dict if kwargs is None
    return _LOSS_MAP[key](**(cfg.kwargs or {}))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------Optimizer-----------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

# TODO: Add documentation

from schemas import OptimizerConfig

_OPTIMIZER_MAP = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
    "adamax": optim.Adamax,
    "nadam": optim.NAdam,
}

def get_optimizer(params, config: OptimizerConfig) -> optim.Optimizer:
    """
    Select and instantiate a PyTorch optimizer.

    Parameters
    ----------
    params : iterable
        Parameters to optimize, e.g. model.parameters()
    config.name : str
        Name of the optimizer, e.g. "adam"
    config.kwargs : dict, optional
        Extra arguments passed to the optimizer, e.g. {"lr": 1e-3}

    Returns
    -------
    optim.Optimizer
        Instantiated optimizer.
    """
    key = config.name.replace("-", "").replace("_", "").lower()
    try:
        optimizer_cls = _OPTIMIZER_MAP[key]
    except KeyError:
        raise ValueError(f"Optimizer {config.name} not recognised. "
                         f"Available: {list(_OPTIMIZER_MAP.keys())}")

    return optimizer_cls(params, **config.kwargs)


def _class_wise_dataset_size(t: torch.LongTensor, num_cls: int) -> list:
    return torch.bincount(input=t, minlength=num_cls).tolist()


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------VISUALISATIONS-----------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

def visualise_samples(samples: torch.FloatTensor, labels: Optional[torch.LongTensor] = None, gen_viz: Optional[int]=None):
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
        samples = samples.reshape(-1, data_dim)                # (n*num_classes, data_dim)
        labels = torch.arange(num_classes).repeat(n)    # (n*num_classes,)

    if samples.shape[1]==2:
        return create_2d_scatter(X=samples, t=labels)
    else:
        if gen_viz is None:
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





# TODO: Learn how to document examples
# example usage
# fig, axes = plt.subplots(1, 3, figsize=(15,19))

# for i, title in enumerate(samples_train.keys()):
#     create_2d_scatter(samples_train[title],
#                       labels_train[title],
#                       title=title,
#                       ax=axes[i],
#                       show_legend=True)
# plt.tight_layout()
# plt.show()


# Visiualisation of training
def _epoch_wise_loss_averaging(train_loss: list, 
                               epochs: int) -> list:
    """
    Running average over epoch of minbatch-wise loss

    Parameters
    ----------
    train_loss: list of floats
        minibatch-wise training loss
    epochs: int
        number of epochs of the training

    Returns
    -------
    list of floats
    """
    mini = len(train_loss) // epochs
    train_loss_average = [sum(train_loss[(i*mini) : ((i+1)*mini)]) / mini for i in range(epochs)]
    return train_loss_average

# TODO: Add option to plot other loss curves and accuracies. 
# MAYBE TODO: Add highlighter for chosen epoch

def plot_train_test_curves(
    train_loss: list,
    test_accuracy: list,
    epochs: int,
    ax: tuple | None = None,
    title: str  | None = None
):
    """
    Plot averaged train loss and test accuracy.

    Parameters
    ----------
    train_loss: list of floats
        list of loss values (flattened over batches and classes)
    test_accuracy: list of floats 
        val accuracy values per epoch
    num_batches: int 
        number of batches per epoch per class
    ax: tuple of matplotlib axes, optional
    title: str, optional 
        title for plots, names of datasets

    Returns
    -------
    axes: tuple of matplotlib axes
    """
    averaged_loss = _epoch_wise_loss_averaging(train_loss, epochs)

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    assert ax is not None
    # Plot loss
    ax[0].plot(range(1, epochs + 1), averaged_loss)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title(f"Train Loss Curve{f' ({title})' if title else ''}")

    # Plot accuracy
    ax[1].plot(range(1, len(test_accuracy) + 1), test_accuracy)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy (%)")
    ax[1].set_title(f"Test Accuracy Curve{f' ({title})' if title else ''}")

    plt.tight_layout()
    return ax

# TODO: Maybe use other curves like loss on validation set
# TODO: Also instead of component plot nothing or gradient plot
# TODO: Think about doing something like a multiplot option like above
# TODO: And using running average instead of raw loss plots.
def ad_train_results(cat_acc: list,
                     d_losses: list,
                     l_t_comps: list):
    """
    Plotting function of logged adversarial training.

    Parameters
    ----------
    cat_acc: list of floats
        epoch-wise categorisation accuracy
    d_losses: list of floats
        minibatch-wise loss of discriminator
    gradient_flow_metric

    Returns
    -------
    Figure with three subplots to judge adversarial training. 
    """
    # X-axis for accuracy: one point per epoch (or per accuracy check)
    epochs = range(len(cat_acc))

    # X-axis for losses and tensor values: one point per batch
    batches = range(len(d_losses))  # assuming l_t_comps is same length

    # Create figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot 1: Classification Accuracy
    axs[0].plot(epochs, cat_acc, marker='o', color='green')
    axs[0].set_title("Cat. Accuracy on Validation")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_ylim(0, 1.05)

    # Plot 2: Discriminator Loss
    axs[1].plot(batches, d_losses, color='blue')
    axs[1].set_title("BCE Loss DNet")
    axs[1].set_xlabel("Batches")
    axs[1].set_ylabel("Loss")

    # Plot 3: L-Tensor (0, 0) Component
    axs[2].plot(batches, l_t_comps, color='red')
    axs[2].set_title("(0,0) Component of L-Tensor")
    axs[2].set_xlabel("Batches")
    axs[2].set_ylabel("Value")

    # Improve spacing
    plt.tight_layout()
    plt.show()