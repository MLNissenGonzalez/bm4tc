# Building the discriminator class out of a predefined MPS.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Dict
from schemas import DisConfig, PretrainDisConfig
from _utils import get_criterion, get_optimizer, _class_wise_dataset_size

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
    Discriminator for GAN training. This architecture is just a fully connected feed forward network with single logit output. 
    One could use this class in a single discriminator setup (one for all classes of samples) or for the ensemble approach. 
    """

    def __init__(self,
                 cfg: DisConfig,
                 input_dim: int):
        super().__init__()

        nonlinearity = cfg.nonlinearity.replace(" ", "").lower()

        if nonlinearity == "relu":
            def get_activation(): return nn.ReLU()
        elif nonlinearity == "leakyrelu":
            if cfg.negative_slope == None:
                raise ValueError("LeakyReLU needs negative_slope parameter.")

            def get_activation(): return nn.LeakyReLU(cfg.negative_slope)
        else:
            raise ValueError(
                f"{nonlinearity} not recognised. Try ReLU or LeakyReLU instead.")

        if not cfg.hidden_dims:
            raise ValueError(
                "hidden_dims must contain at least one layer size.")

        layers = []
        layers.append(nn.Linear(input_dim, cfg.hidden_dims[0]))
        for i in range(len(cfg.hidden_dims)-1):
            layers.append(get_activation())
            layers.append(nn.Linear(cfg.hidden_dims[i], cfg.hidden_dims[i+1]))
        layers.append(get_activation())
        # always binary classification
        layers.append(nn.Linear(cfg.hidden_dims[-1], 1))

        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.stack(x)  # returns logit

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
    logger.debug(f"class wise dataset size = {num_spc}")
    if mode == "single":
        synths = []
        for c in range(num_cls):
            X_synth_c = X_synth[:num_spc[c], c, :]
            synths.append(X_synth_c)

        X_synth = torch.concat(synths, dim=0)
        t_synth = torch.zeros(len(X_synth), dtype=torch.long)  # fake
        t_real = torch.ones(len(X_real), dtype=torch.long)      # real

        logger.debug(f"{_class_wise_dataset_size(t_real, num_cls)=}")
        logger.debug(f"{_class_wise_dataset_size(t_synth, num_cls)=}")

        X = torch.cat([X_real, X_synth], dim=0)
        t = torch.cat([t_real, t_synth], dim=0)
        return TensorDataset(X, t)

    elif mode == "ensemble":
        datasets = {}
        for c in range(num_cls):
            X_real_c = X_real[c_real == c]                # select class c
            # synthetic for class c
            X_synth_c = X_synth[:num_spc[c], c, :]
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


def eval(dis: nn.Module, loader: DataLoader, criterion: nn.Module):
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
    with torch.no_grad():
        for X, t in loader:
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
                loaders: Dict[str, DataLoader]):
    """
    Pretrain a single discriminator in a binary classification setting.
    Real samples are labeled 1, synthetic samples are labeled 0.
    Synthesised samples are assumed to be precomputed.

    Parameters
    ----------
    dis : nn.Module
        The discriminator model to pretrain.
    cfg : PretrainDisConfig
        Configuration containing optimizer, criterion, max_epoch, and patience.
    loaders : Dict[str, DataLoader]
        Dictionary of DataLoaders with keys:
        - "train": DataLoader for training data.
        - "valid": DataLoader for validation data.
        - "test": DataLoader for test data.

    Returns
    -------
    result : dict
        A dictionary containing:
        - "model": the best-performing discriminator (nn.Module).
        - "train loss": list of floats, minibatch-wise training loss.
        - "valid loss": list of floats, epoch-wise validation loss.
        - "valid accuracy": list of floats, epoch-wise validation accuracy.
        - "test accuracy": float, final accuracy on the test set.

    Notes
    -----
    - Early stopping is applied based on validation loss with patience `cfg.patience`.
    - Only supports pretraining of a *single* discriminator. For ensembles,
      call this function separately per member.
    """

    train_loss = []
    valid_loss = []
    valid_accuracy = []
    patience_counter = 0
    best_loss = float('inf')

    optimizer = get_optimizer(dis.parameters(), cfg.optimizer)
    criterion = get_criterion(cfg.criterion)

    optimizer.zero_grad()
    for epoch in range(cfg.max_epoch):
        dis.train()
        for X, t in loaders["train"]:
            logit = dis(X)
            prob = torch.sigmoid(logit.squeeze())
            loss = criterion(prob, t.float())
            train_loss.append(loss.item())

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        dis.eval()
        acc, avg_valid_loss = eval(dis, loaders["valid"], criterion)
        valid_loss.append(avg_valid_loss)
        valid_accuracy.append(acc)

        # Progress tracking and best model update
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            patience_counter = 0
            best_model_state = dis.state_dict()
        else:
            patience_counter += 1
        # Early stopping
        if patience_counter > cfg.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # If mode == ensemble, then this would be only one member
    dis.load_state_dict(best_model_state)
    test_accuracy, _ = eval(dis, loaders["test"], criterion)
    logger.info(f"{test_accuracy=}")

    result = {
        "model": dis,
        "train loss": train_loss,
        "valid loss": valid_loss,
        "valid accuracy": valid_accuracy,
        "test accuracy": test_accuracy
    }

    return result
