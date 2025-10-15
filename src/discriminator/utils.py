# Building the discriminator class out of a predefined MPS.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Dict
from schemas import DisConfig, PretrainDisConfig
from _utils import get_criterion, get_optimizer, _class_wise_dataset_size
import wandb
from math import ceil
import copy

import logging
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------´
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# --------------------DISCRIMINIATOR INITIALIZATION-----------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------


class MLP(nn.Module):
    """
    Flexible discriminator-style multilayer perceptron (MLP).

    The hidden layer widths are determined dynamically as a scaled multiple 
    of the input dimensionality. This allows proportional model scaling across
    datasets of varying feature sizes. Supports ReLU and LeakyReLU activations.

    Parameters
    ----------
    cfg : DisConfig
        Discriminator configuration containing:
        - `nonlinearity`: str — activation name ("relu" or "leakyrelu").
        - `negative_slope`: float or None — required if using LeakyReLU.
        - `hidden_multipliers`: list of float — each value scales the previous
          layer’s output size by `ceil(multiplier * input_dim)`.
    input_dim : int
        Dimensionality of the input feature vector.

    Attributes
    ----------
    stack : nn.Sequential
        Sequential container of linear and activation layers, ending with 
        a single-unit output layer (no sigmoid applied).

    Forward Input
    -------------
    x : torch.Tensor
        Shape `(batch_size, input_dim)`, dtype `torch.float32`.

    Forward Output
    --------------
    torch.Tensor
        Shape `(batch_size, 1)`, representing raw (unnormalized) logits
        for binary classification tasks.

    Raises
    ------
    ValueError
        If activation type is unsupported, if `negative_slope` is missing for 
        LeakyReLU, or if `hidden_multipliers` is empty.
    """

    def __init__(self, cfg: DisConfig, input_dim: int):
        super().__init__()

        # Determine activation
        act = cfg.nonlinearity.replace(" ", "").lower()
        if act == "relu":
            def get_activation(): return nn.ReLU()
        elif act == "leakyrelu":
            if cfg.negative_slope is None:
                raise ValueError("LeakyReLU needs negative_slope parameter.")

            def get_activation(): return nn.LeakyReLU(cfg.negative_slope)
        else:
            raise ValueError(
                f"{cfg.nonlinearity} not recognised. Try ReLU or LeakyReLU.")

        # Determine hidden_dims
        if cfg.hidden_multipliers is None or len(cfg.hidden_multipliers) == 0:
            raise ValueError(
                "hidden_multipliers must be provided as a list of floats.")

        hidden_dims = [max(1, ceil(mult * input_dim))
                       for mult in cfg.hidden_multipliers]

        # Build layers
        layers = [nn.Linear(input_dim, hidden_dims[0])]
        for i in range(len(hidden_dims) - 1):
            layers.append(get_activation())
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        layers.append(get_activation())
        layers.append(nn.Linear(hidden_dims[-1], 1))  # binary classification

        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.stack(x)

# ADDED AS ISSUE.
# TODO: Add other discriminator types, e.g. MPS with MLP module
# TODO: Add the discriminator class taking an MPS as input
#       and returning a pytorch module that is the MPS with an
#       MLP at the end to discriminate real from fake inputs to the MPS


def init_discriminator(cfg: DisConfig,
                       input_dim: int,
                       num_classes: int,
                       device: torch.device) -> Dict[Any, nn.Module]:
    """
    Initialize one or more discriminator networks based on configuration.

    Depending on the discriminator mode, this function constructs either:
    - a single shared discriminator (for unconditional setups), or
    - an ensemble of class-specific discriminators (for conditional setups).

    Parameters
    ----------
    cfg : DisConfig
        Discriminator configuration containing at least:
        - `mode`: str — either "single" or "ensemble".
        - `nonlinearity`, `negative_slope`, and `hidden_multipliers` fields
          used for building each `MLP`.
    input_dim : int
        Number of input features per sample passed to each discriminator.
    num_classes : int
        Number of label classes. Used only when `cfg.mode == "ensemble"`.
    device : torch.device
        Target device on which to place the discriminator module(s).

    Returns
    -------
    dict of (str or int) -> nn.Module
        Mapping from discriminator identifier to initialized model:
        - If `cfg.mode == "single"`:
            {"single": MLP(...)}  
        - If `cfg.mode == "ensemble"`:
            {0: MLP(...), 1: MLP(...), ..., num_classes-1: MLP(...)}  
        All returned modules are moved to the specified `device`.

    Raises
    ------
    KeyError
        If `cfg.mode` is not one of {"single", "ensemble"}.
    """

    if cfg.mode == "single":
        return {"single": MLP(cfg, input_dim).to(device)}
    elif cfg.mode == "ensemble":
        assert num_classes > 0, f"{num_classes=}"
        return {c: MLP(cfg, input_dim).to(device) for c in range(num_classes)}
    else:
        raise KeyError(f"{cfg.mode} not recognised.")

# ------------------------------------------------------------------------------------------------´
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# --------------------DISCRIMINIATOR PRETRAINING----------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------


def pretrain_dataset(X_real: torch.FloatTensor,
                     c_real: torch.LongTensor,
                     X_synth: torch.FloatTensor,
                     mode="single"):
    """
    Construct a dataset (or per-class datasets) for discriminator pretraining,
    combining real and synthesised samples into a binary classification task
    (real = 1, synthetic = 0).

    Parameters
    ----------
    X_real : torch.FloatTensor
        Real samples of shape (N, data_dim).
    c_real : torch.LongTensor
        Class labels for the real samples of shape (N,).
    X_synth : torch.FloatTensor
        Synthesised samples of shape (n_samples_per_class, n_classes, data_dim),
        typically from `sampling.batched`.
    mode : str, default="single"
        - "single": return one dataset containing all classes pooled together.
        - "ensemble": return a dictionary mapping each class index to its
          corresponding dataset.

    Returns
    -------
    torch.utils.data.TensorDataset or Dict[int, torch.utils.data.TensorDataset]
        - If `mode="single"`, a single dataset of real/synthetic pairs across all classes.
        - If `mode="ensemble"`, a dictionary keyed by class index, each containing a dataset
          of real/synthetic pairs for that class only.

    Raises
    ------
    KeyError
        If `mode` is not "single" or "ensemble".
    """
    num_cls = X_synth.shape[1]
    num_spc = _class_wise_dataset_size(c_real, num_cls)
    if mode == "single":
        synths = []
        for c in range(num_cls):
            X_synth_c = X_synth[:num_spc[c], c, :]
            synths.append(X_synth_c)

        X_fake = torch.concat(synths, dim=0)
        t_fake = torch.zeros(len(X_fake), dtype=torch.long)  # fake
        t_real = torch.ones(len(X_real), dtype=torch.long)      # real

        logger.debug(f"{_class_wise_dataset_size(t_real, num_cls)=}")
        logger.debug(f"{_class_wise_dataset_size(t_fake, num_cls)=}")

        X = torch.cat([X_real, X_fake.cpu()], dim=0)
        t = torch.cat([t_real, t_fake], dim=0)
        return TensorDataset(X, t)

    elif mode == "ensemble":
        datasets = {}
        for c in range(num_cls):
            X_real_c = X_real[c_real == c]                # select class c
            # synthetic for class c
            X_synth_c = X_synth[:num_spc[c], c, :].cpu()
            t_real = torch.ones(len(X_real_c), dtype=torch.long)
            t_synth = torch.zeros(len(X_synth_c), dtype=torch.long)

            X = torch.cat([X_real_c, X_synth_c], dim=0)
            t = torch.cat([t_real, t_synth], dim=0)
            datasets[c] = TensorDataset(X, t)
        return datasets

    else:
        raise KeyError(f"{mode} has to be either single or ensemble.")


def pretrain_loader(X_real: torch.FloatTensor,
                    c_real: torch.LongTensor,
                    X_synth: torch.FloatTensor,
                    mode: str,
                    batch_size: int,
                    split: str) -> Dict[Any, DataLoader]:
    """
    Wrap the pretraining datasets into PyTorch DataLoaders for a given split.

    Parameters
    ----------
    X_real : torch.FloatTensor
        Real samples of shape (N, data_dim).
    c_real : torch.LongTensor
        Class labels for the real samples of shape (N,).
    X_synth : torch.FloatTensor
        Synthesised samples of shape (n_samples_per_class, n_classes, data_dim).
    mode : str
        - "single": return one DataLoader with all classes pooled.
        - "ensemble": return a dictionary of DataLoaders keyed by class index.
    batch_size : int
        Batch size for each DataLoader.
    split : {"train", "valid", "test"}
        Which split this loader belongs to. Used to set shuffling and drop_last.

    Returns
    -------
    Dict[str, torch.utils.data.DataLoader] or Dict[int, torch.utils.data.DataLoader]
        - If `mode="single"`, returns {"single": DataLoader(...)}.
        - If `mode="ensemble"`, returns {class_idx: DataLoader(...), ...}.

    Raises
    ------
    KeyError
        If `split` is not one of {"train", "valid", "test"}.
    """
    if split not in ["train", "valid", "test"]:
        raise KeyError(f"{split} not recognised.")

    dataset = pretrain_dataset(X_real, c_real, X_synth, mode=mode)
    if mode == "single":
        return {"single": DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), drop_last=(split == "train"))}
    elif mode == "ensemble":
        loaders = {c: DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), drop_last=(split == "train"))
                   for c, ds in dataset.items()}
        return loaders


def eval(dis: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    """
    Evaluate a discriminator on a binary classification task.

    Parameters
    ----------
    dis : nn.Module
        The discriminator model.
    loader : torch.utils.data.DataLoader
        DataLoader providing batches of (X, t), where t is the real/synthetic label.
    criterion : nn.Module
        Loss function (e.g. binary cross entropy).

    Returns
    -------
    acc : float
        Classification accuracy over the dataset.
    avg_loss : float
        Average loss over the dataset.
    """
    dis.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    dis.to(device)
    with torch.no_grad():
        for X, t in loader:
            X, t = X.to(device), t.to(device)
            logit = dis(X)
            prob = torch.sigmoid(logit.squeeze())
            preds = (prob >= 0.5).long()
            loss = criterion(prob, t.float())
            total_loss += loss.item() * t.size(0)
            correct += (preds == t).sum().item()
            total += t.size(0)

        avg_loss = total_loss / total
        acc = correct / total
    return acc, avg_loss


def _define_metric(key: str | int):
    """
    Define Weights & Biases (W&B) metrics for discriminator pretraining.

    Creates metric keys for tracking training and validation losses during
    the discriminator pretraining phase. Each discriminator (identified by
    `key`) has its own namespace to support both single and ensemble setups.

    Parameters
    ----------
    key : str or int
        Identifier for the discriminator instance. Typically `"single"` for
        shared models or an integer index (e.g., 0, 1, 2, ...) for ensemble
        members.

    Defines
    --------
    - `"pre_dis/{key}/train/loss"` : epoch-wise training loss.
    - `"pre_dis/{key}/valid/loss"` : epoch-wise validation loss.

    Notes
    -----
    The metrics are defined with `summary="none"`, meaning no automatic
    aggregation will be performed by W&B. Summaries (e.g., best accuracy)
    are manually added after pretraining completes.
    """
    wandb.define_metric(f"pre_dis/{key}/train/loss", summary="none")
    wandb.define_metric(f"pre_dis/{key}/valid/loss", summary="none")


def _log(key: int | str,
         train_loss: float,
         valid_acc: float,
         valid_loss: float):
    """
    Log discriminator pretraining metrics to Weights & Biases (W&B).

    Logs epoch-wise training and validation statistics for a specific
    discriminator instance. Works seamlessly for both single and ensemble
    modes, using `key` to namespace the logged values.

    Parameters
    ----------
    key : int or str
        Discriminator identifier matching the W&B metric namespace.
        - `"single"` for a shared discriminator.
        - Integer index for ensemble members.
    train_loss : float
        Mean training loss for the current epoch.
    valid_acc : float
        Validation accuracy for the current epoch.
    valid_loss : float
        Mean validation loss for the current epoch.

    Logs
    ----
    - `"pre_dis/{key}/train/loss"` : Training loss value.
    - `"pre_dis/{key}/valid/acc"`  : Validation accuracy.
    - `"pre_dis/{key}/valid/loss"` : Validation loss value.

    Notes
    -----
    These metrics correspond to those defined in `_define_metric()`.
    Final summary values (best and test accuracies) are recorded separately
    in W&B summaries after pretraining.
    """
    wandb.log({
        f"pre_dis/{key}/train/loss": train_loss,
        f"pre_dis/{key}/valid/acc": valid_acc,
        f"pre_dis/{key}/valid/loss": valid_loss
    })


def pretraining(dis: nn.Module,
                cfg: PretrainDisConfig,
                loaders: Dict[str, DataLoader],
                key: Any,
                device: torch.device
                ):
    """
    Supervised pretraining routine for a discriminator network.

    This function trains a discriminator (`dis`) using labeled data before
    adversarial training. It supports both single and ensemble setups, logs
    intermediate metrics to Weights & Biases (W&B), and employs early stopping
    based on validation accuracy or loss.

    Parameters
    ----------
    dis : nn.Module
        Discriminator network to be trained. Expects inputs of shape
        `(batch_size, n_features)` and outputs logits of shape `(batch_size, 1)`.
    cfg : PretrainDisConfig
        Pretraining configuration containing:
        - `max_epoch` : int — maximum number of training epochs.
        - `optimizer` : OptimizerConfig — optimizer setup used by `get_optimizer()`.
        - `criterion` : CriterionConfig — defines the loss (e.g., "bce").
        - `stop_crit` : {"acc", "loss"} — early stopping criterion.
        - `patience` : int — allowed number of non-improving epochs.
        - `info_freq` : int — interval for logging training progress.
    loaders : dict of str -> DataLoader
        Data loaders for `"train"`, `"valid"`, and `"test"` splits.
        Each batch yields `(X, t)` pairs where:
        - `X` : torch.Tensor, shape `(batch_size, n_features)`, dtype float32
        - `t` : torch.Tensor, shape `(batch_size,)`, dtype float32, binary labels in {0, 1}.
    key : Any
        Identifier for the discriminator instance. Used in W&B metric keys
        (e.g., `"pre_dis/{key}/train/loss"`).
    device : torch.device
        Computation device to which the model and data are moved.

    Returns
    -------
    nn.Module
        The pretrained discriminator with restored weights corresponding
        to the best validation performance.

    Training Details
    ----------------
    - Uses `nn.BCELoss`, so logits are converted to probabilities via
      `torch.sigmoid()`.
    - Optimizer is created via `get_optimizer(dis.parameters(), cfg.optimizer)`.
    - Early stopping is triggered if validation metric fails to improve for
      `cfg.patience` consecutive epochs.

    Raises
    ------
    ValueError
        If the provided loss or optimizer type in `cfg` is invalid.
    KeyError
        If `cfg.stop_crit` is not one of {"acc", "loss"}.

    Notes
    -----
    The function restores the model weights corresponding to the best
    validation epoch before returning the final pretrained discriminator.
    """
    # Early stopping quantities
    best_loss, best_acc, patience_counter = float("inf"), 0.0, 0
    best_model_state = copy.deepcopy(dis.state_dict())
    _define_metric(key)

    # Initialize optimizer and loss function
    optimizer = get_optimizer(dis.parameters(), cfg.optimizer)
    criterion = get_criterion(cfg.criterion)
    dis.to(device)

    # Loop over epochs
    for epoch in range(cfg.max_epoch):
        losses = []
        dis.train()
        # Loop over batches
        for X, t in loaders["train"]:
            # Prediction
            X, t = X.to(device), t.to(device)
            logit = dis(X)
            prob = torch.sigmoid(logit.squeeze())
            loss = criterion(prob, t.float())

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Save losses
            losses.append(loss.detach().cpu().item())

        # Evaluate epoch-wise performance
        train_loss = sum(losses) / len(losses)
        losses.clear()
        valid_acc, valid_loss = eval(dis, loaders["valid"], criterion, device)

        # Log epoch-wise performance
        _log(key, train_loss, valid_acc, valid_loss)

        # Info
        if ((epoch+1) % cfg.info_freq) == 0:
            logger.info(f"Epoch {epoch+1}: val_accuracy={valid_acc:.4f}")

        # Model update and early stopping
        if cfg.stop_crit == "acc":
            if valid_acc > best_acc:
                best_acc, patience_counter = valid_acc, 0
                best_model_state = copy.deepcopy(dis.state_dict())
            else:
                patience_counter += 1
        elif cfg.stop_crit == "loss":
            if valid_loss < best_loss:
                best_loss, patience_counter = valid_loss, 0
                best_model_state = copy.deepcopy(dis.state_dict())
            else:
                patience_counter += 1

        # Early stopping
        if patience_counter > cfg.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # If mode == ensemble, then this would be only one member
    dis.load_state_dict(best_model_state)

    # Final evaluation of discriminator performance after pretraining
    test_accuracy, _ = eval(dis, loaders["test"], criterion, device)
    logger.info(f"{test_accuracy=}")
    wandb.summary[f"pre_dis/{key}/test/acc"] = test_accuracy
    wandb.summary[f"pre_dis/{key}/valid/acc"] = best_acc

    return dis
