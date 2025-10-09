# utils that are practical for experiments
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from typing import Optional, Dict, Any
import schemas
from pathlib import Path
from datetime import datetime
import random 
import os
import hydra
import logging
import wandb
import omegaconf
from torch.autograd import Function
import scipy.linalg

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

def get_criterion(cfg: schemas.CriterionConfig) -> nn.Module:
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

logger = logging.getLogger(__name__)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------wandb tracking setup-----------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------

def init_wandb(cfg: schemas.Config):
    # Convert only loggable types
    wandb_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    
    # Job Info
    runtime_cfg = hydra.core.hydra_config.HydraConfig.get()
    run_dir = Path(runtime_cfg.runtime.output_dir)
    job_num = int(runtime_cfg.job.get("num", 0)) + 1

    # Job Mode
    mode = runtime_cfg.mode.value
    total_num = 1
    if mode == 1: # single run
        now = datetime.now().strftime("%d%b%y_%I%p%M")
    else: # multirun
        now = datetime.now().strftime("%d%b%y")
        params = runtime_cfg.sweeper.params
        for group in params.values():
            # handle both list and string forms
            if isinstance(group, (list, tuple, omegaconf.ListConfig)):
                options = group
            elif isinstance(group, str):
                options = group.split(",")
            else:
                raise TypeError(f"Unexpected sweeper param type: {type(group)} ({group})")
            total_num *= len(options)

    # Group (folder) and run name 
    group_key = f"{cfg.experiment}_{cfg.dataset.name}_{now}"
    run_name = f"job{job_num}/{total_num}_D{cfg.model.mps.init_kwargs.bond_dim}-d{cfg.model.mps.init_kwargs.in_dim}-pre{cfg.pretrain.mps.max_epoch}-gan{cfg.gantrain.max_epoch}"
    
    # Initializing the wandb object
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        dir=str(run_dir),
        config=wandb_cfg,
        group=group_key,
        name=run_name,
        mode=cfg.wandb.mode,
        reinit="finish_previous"
    )
    return run



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------Model weights save and transfer-----------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

# TODO: MPS pretraining models should be saved carefully.

# General saving function
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

# Verification of model transfer
def verify_tensors(model1, model2, name1="Original", name2="Copy"):
    logger.info("Verifying tensor transfer...")
    for i, (t1, t2) in enumerate(zip(model1.tensors, model2.tensors)):
        if not torch.allclose(t1, t2):
            logger.error(f"Tensor {i} mismatch between {name1} and {name2}")
            logger.error(f"Max difference: {(t1 - t2).abs().max().item()}")
        else:
            logger.info(f"Tensor {i} successfully transferred")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------VISUALISATIONS-----------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------


# TODO: Implement this also for other visualisable datatypes
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
        samples = samples.reshape(n*num_classes, data_dim)                # (n*num_classes, data_dim)
        labels = torch.arange(num_classes).repeat(n)    # (n*num_classes,)

    if samples.shape[1]==2:
        return create_2d_scatter(X=samples, t=labels)
    else:
        if gen_viz is None:
            gen_viz = samples.shape[0] # Can be used to visualise only a limited amount of examples
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

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------Random Seed -----------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Ensure deterministic behavior in cuDNN (can slow things down!)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set PYTHONHASHSEED environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------Metrics for generative capabilities -----------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

# --- Matrix square root implementation for pytorch, 
# from https://github.com/steveli/pytorch-sqrtm/blob/master/sqrtm.py ---

class MatrixSquareRoot(Function):
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float64)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.detach().cpu().numpy().astype(np.float64)
            gm = grad_output.detach().cpu().numpy().astype(np.float64)
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)
            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply

def mean_n_cov(data: torch.FloatTensor):
    """
    data: shape (N, d)
    """
    mu = data.mean(dim=0)
    cov = torch.cov(data.T)
    return mu, cov

# --- FID-like metric ---
class FIDLike(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def lazy_forward(self, mu_r, cov_r, generated):
        mu_g, cov_g = mean_n_cov(generated)
        
        # Regularize for numerical stability
        eye = torch.eye(cov_r.shape[0], device=cov_r.device)
        cov_r = cov_r + self.eps * eye
        cov_g = cov_g + self.eps * eye

        # Mean difference
        diff_mu = torch.sum((mu_r - mu_g) ** 2)

        # Matrix square root term using custom function
        covmean = sqrtm(cov_r @ cov_g)

        # Ensure real part (numerical precision can introduce tiny imaginary parts)
        if torch.is_complex(covmean):
            covmean = covmean.real

        diff_cov = torch.trace(cov_r + cov_g - 2 * covmean)

        return diff_mu + diff_cov
    
    def forward(self, real, generated):
        mu_r, cov_r = mean_n_cov(real)
        return self.lazy_forward(mu_r, cov_r, generated)
        

def sample_quality_control(synths: torch.FloatTensor, 
                           upper: float, lower: float):
    
    bad_idx = (
        (synths.abs() > upper) | (synths < lower)
        ).nonzero(as_tuple=False)  # (sample_idx, class_idx, feat_idx)
    logger.info(f"bad positions count = {bad_idx.shape[0]}")
    if bad_idx.shape[0] > 0:
        # show first few offending indices and their values
        for i in range(min(200, bad_idx.shape[0])):
            s,c,f = bad_idx[i].tolist()
            val = synths[s, c, f].item()
            logger.info(f"BAD value at sample={s}, class={c}, feat={f}: {val}")
        # show global per-dim & per-class stats
        logger.info("per-class-per-dim max abs:")
        for c in range(synths.shape[1]):
            for f in range(synths.shape[2]):
                m = synths[:, c, f].abs().max().item()
                logger.info(f" class={c}, feat={f}, max_abs={m:.4g}, mean={synths[:,c,f].mean().item():.4g}")
    else:
        logger.info("No values above threshold found.")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------Dead and unused code-----------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------

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