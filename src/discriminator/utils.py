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
    Flexible discriminator MLP where hidden layers scale with input dimension.
    """

    def __init__(self, cfg: DisConfig, input_dim: int):
        super().__init__()

        # Determine activation
        act = cfg.nonlinearity.replace(" ", "").lower()
        if act == "relu":
            get_activation = lambda: nn.ReLU()
        elif act == "leakyrelu":
            if cfg.negative_slope is None:
                raise ValueError("LeakyReLU needs negative_slope parameter.")
            get_activation = lambda: nn.LeakyReLU(cfg.negative_slope)
        else:
            raise ValueError(f"{cfg.nonlinearity} not recognised. Try ReLU or LeakyReLU.")

        # Determine hidden_dims
        if cfg.hidden_multipliers is None or len(cfg.hidden_multipliers) == 0:
            raise ValueError("hidden_multipliers must be provided as a list of floats.")

        hidden_dims = [max(1, ceil(mult * input_dim)) for mult in cfg.hidden_multipliers]

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


# TODO: Add other discriminator types, e.g. MPS with MLP module


def init_discriminator(cfg: DisConfig,
                       input_dim: int,
                       num_classes: int,
                       device: torch.device) -> Dict[Any, nn.Module]:
    if cfg.mode == "single":
        return {"single": MLP(cfg, input_dim).to(device)}
    elif cfg.mode == "ensemble":
        return {c: MLP(cfg, input_dim).to(device) for c in range(num_classes)}
    else:
        raise KeyError(f"{cfg.mode} not recognised.")

# TODO: Add the discriminator class taking an MPS as input
#       and returning a pytorch module that is the MPS with an
#       MLP at the end to discriminate real from fake inputs to the MPS

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


def pretraining(dis: nn.Module,
                cfg: PretrainDisConfig,
                loaders: Dict[str, DataLoader],
                key: Any,
                device: torch.device
                ):
    # TODO: Provide updated docstring

    # Early stopping quantities
    patience_counter = 0
    best_loss = float("inf")
    best_acc = 0.0

    optimizer = get_optimizer(dis.parameters(), cfg.optimizer)
    criterion = get_criterion(cfg.criterion)

    dis.to(device)
    optimizer.zero_grad()
    for epoch in range(cfg.max_epoch):
        losses = []
        dis.train()
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
        acc, avg_valid_loss = eval(dis, loaders["valid"], criterion, device)

        # Log epoch-wise performance
        wandb.log({
            f"pre_dis/{key}/train/loss": train_loss,
            f"pre_dis/{key}/valid/acc": acc,
            f"pre_dis/{key}/valid/loss": avg_valid_loss
        })

        # Info
        if ((epoch+1)%cfg.info_freq) == 0:
            logger.info(f"Epoch {epoch+1}: val_accuracy={acc:.4f}")

        # Model update and early stopping
        if cfg.stop_crit=="acc":
            if acc > best_acc:
                best_acc, patience_counter = acc, 0
                best_model_state = copy.deepcopy(dis.state_dict())
            else: patience_counter += 1
        elif cfg.stop_crit=="loss":
            if avg_valid_loss < best_loss:
                best_loss, patience_counter = avg_valid_loss, 0
                best_model_state = copy.deepcopy(dis.state_dict())
            else: patience_counter += 1
            
        # Early stopping
        if patience_counter > cfg.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # If mode == ensemble, then this would be only one member
    dis.load_state_dict(best_model_state)
    dis.to(device)
    test_accuracy, _ = eval(dis, loaders["test"], criterion, device)
    logger.info(f"{test_accuracy=}")
    wandb.summary[f"pre_dis/{key}/test/acc"] = test_accuracy

    return dis
