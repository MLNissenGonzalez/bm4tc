import torch
import tensorkrowch as tk
from typing import Callable, Dict
from torch.utils.data import TensorDataset, DataLoader
from schemas import PretrainMPSConfig
from _utils import get_optimizer, get_criterion
from mps.utils import get_embedding, born_parallel

import logging
logger = logging.getLogger(__name__)

def mps_cat_loader( X: torch.Tensor, 
                    t: torch.Tensor, 
                    embedding: str, 
                    batch_size: int, 
                    phys_dim: int,
                    split: str) -> DataLoader:
    """
    Create DataLoaders for multiple datasets and splits

    Parameters
    ----------
    X: torch.Tensor, shape: (batch_size, n_feat)
        The preprocessed, non-embedded features.
    t: torch.Tensor, shape (batch_size,)
        The labels of the features X.
    batch_size: int
        The number of examples of features for all classes (not per class!)
    embedding: str
        name of embedding for X
    phys_dim: int, 
        Dimension of embedding space

    Returns
    -------
    DataLoader
        Dataloader for supervised classification.
    """
    embedding = get_embedding(embedding)
    X = embedding(X, phys_dim)
    dataset = TensorDataset(X, t)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split in ['train', 'valid']),
        drop_last=(split == 'train')
    )
    return loader

def mps_acc_eval(   mps: tk.models.MPSLayer, # relies on mps.out_features = [cls_pos]
                    loader: DataLoader,
                    device: torch.device) -> float:
    """
    Computes the prediction accuracy of a MPS Born machine on a dataset.

    Parameters
    ----------
    mps: MPS model with central tensor
    loader: DataLoader
    device: torch.device

    Returns
    -------
    float
        prediction accuracy
    """
    mps.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, t in loader:
            X, t = X.to(device), t.to(device)
            p = born_parallel(mps, X)
            _, preds = torch.max(p, 1)

            # Counting correct predictions
            correct += (preds == t).sum().item()
            total += t.size(0)
    acc = correct / total
    return acc


def _mps_cls_train_step(mps: tk.models.MPSLayer, # expect mps.out_features = [cls_pos]
                       loader: DataLoader,
                       device: torch.device,
                       optimizer: torch.optim.Optimizer,
                       loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] # expects score, not probs. switch for str parameter
                       ) -> list:
    """
    Single epoch classification training. Returns minibatch-wise train_loss.

    Parameters
    ----------
    mps: MPS model with central tensor
    loader: DataLoader
    optimizer: Optimizer
    loss_fn: LossFunction
        takes probabilities and class labels and returns float

    Returns
    -------
    list of floats
        minibatch-wise trainloss 
    """
    train_loss = []
    mps.train()
    for X, t in loader:
        X, t = X.to(device), t.to(device)

        # Compute probs and loss
        p = born_parallel(mps, X)
        loss = loss_fn(p, t)

        # Backpropagate and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    return train_loss

# TODO: Think about logging different or more quantities
# TODO: Consider removing title parameter
def disr_train_mps( mps: tk.models.MPSLayer,
                    loaders: Dict[str, DataLoader], #loads embedded inputs and labels
                    cfg: PretrainMPSConfig,
                    device: torch.device,
                    title: str | None = None,
                    ):
    """
    Full classification loop for MPS with early stopping based on validation accuracy.

    The training loop uses patience stopping and optional goal accuracy stopping. 
    Loss and optimizer are instantiated from the provided config objects. 
    Model-specific flags (`auto_stack`, `auto_unbind`) and device placement are set automatically.

    Parameters
    ----------
    mps : tk.models.MPS or tk.models.MPSLayer
        The MPS model to train.
    loaders : dict[str, DataLoader]
        Dictionary containing 'train' and 'valid' DataLoaders with embedded inputs and labels.
    cfg : PretrainMPSConfig
        Configuration object containing all training hyperparameters and sub-configs:
        
        Fields:
        - optimizer_cfg : OptimizerConfig
            Name and kwargs for optimizer, e.g., {"name": "adam", "optimizer_kwargs": {"lr": 1e-4}}
        - criterion_cfg : CriterionConfig
            Name and kwargs for loss function, e.g., {"name": "nlll", "kwargs": {"eps": 1e-12}}
        - max_epochs : int
            Maximum number of training epochs.
        - patience : int
            Number of epochs without improvement on validation accuracy before stopping.
        - auto_stack : bool
        - auto_unbind : bool
        - goal_acc : Optional[float]
            If provided, training stops early once validation accuracy reaches this value.
        - print_early_stop : bool
            If True, prints early stopping information.
        - print_updates : bool
            If True, prints epoch-wise validation accuracy.
    cls_pos : int
        Position of the central/output tensor in the MPS.
    device : torch.device
        Device to place model and inputs on (CPU or GPU).
    title : str, optional
        Optional title for printing progress messages.

    Returns
    -------
    best_tensors : list[torch.Tensor]
        The tensors of the MPS corresponding to the best validation accuracy.
    train_loss : list[float]
        Mini-batch-wise training loss collected during training.
    val_accuracy : list[float]
        Epoch-wise validation accuracy.
    """
    logger.info("Categorisation training begins.")
    val_accuracy = []
    train_loss = []
    best_acc = 0.0
    best_tensors = []  # Container for best model parameters

    # Prepare MPS for training
    mps.to(device)
    mps.auto_stack = cfg.auto_stack
    mps.auto_unbind = cfg.auto_unbind
    mps.unset_data_nodes()
    mps.reset()
    mps.trace(torch.zeros(1, len(mps.in_features), mps.in_dim[0]).to(device))
    logger.debug("MPS resetted and ready.")

    # Instantiate criterion and optimizer
    criterion = get_criterion(cfg.criterion)
    optimizer = get_optimizer(mps.parameters(), cfg.optimizer)

    patience_counter = 0
    for epoch in range(cfg.max_epoch):
        # Training step
        train_loss = train_loss + _mps_cls_train_step(mps, loaders['train'], 
                                  device, optimizer, criterion)

        # Validation step
        acc = mps_acc_eval(mps, loaders['valid'], device)
        val_accuracy.append(acc)

        # Progress tracking and best model update
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            best_tensors = mps.tensors
        else:
            patience_counter += 1

        # Early stopping via patience
        if patience_counter > cfg.patience:
            if cfg.print_early_stop:
                logger.info(f"Early stopping via patience at epoch {epoch}")
            break
        # Optional early stopping via goal accuracy
        if (cfg.goal_acc is not None):
            if cfg.goal_acc < best_acc:
                if cfg.print_early_stop:
                    logger.info(f"Early stopping via goal acc at epoch {epoch}")
            break
        # Update report
        if (epoch == 0) and (title is not None):
            logger.info(f'\nTraining of {title} dataset:\n')
        elif cfg.print_updates and ((epoch+1) % 10 == 0):
            logger.info(f"Epoch {epoch+1}: val_accuracy={acc:.4f}")

    return best_tensors, train_loss, val_accuracy
