# utils that are practical for experiments
import os
import logging
import matplotlib.pyplot as plt
import torch
import numpy as np


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------VISUALISATIONS-----------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

# TODO: Add type hinting
def create_2d_scatter(X, t, title=None, ax=None, show_legend=True):
    """
    Create a 2D scatter plot that handles both numpy arrays and torch tensors
    And can be embedded in larger figures

    Parameters
    ----------
    X : tensor or array
        2D samples. Shape (N, 2)
    t : tensor or array
        class labels of samples, shape: (N,)
    title : str, optional
        title of dataset
    ax : matplotlib axis, optional 
        for subplot usage
    show_legend : bool
        whether to show legend with class labels

    Returns:
    ax : matplotlib axis
    """

    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    if torch.is_tensor(t):
        t = t.detach().cpu().numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(5,5))

    # Plot each class separately for legend
    classes = np.unique(t)
    for cls in classes:
        idx = (t == cls)
        ax.scatter(X[idx, 0], X[idx, 1], s=5, label=f'Class {cls}', cmap='managua')

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

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------LOGGIN AND OTHER FUNCTIONS----------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

# TODO: Think of using MLFlow instead of logging. 

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger