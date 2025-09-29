import torch
import tensorkrowch as tk
from typing import Callable, Dict, Tuple, List
from torch.utils.data import TensorDataset, DataLoader
from schemas import PretrainMPSConfig
from _utils import get_optimizer, get_criterion
import mps.utils as mps
import wandb

import logging
logger = logging.getLogger(__name__)


def loader_creator(X: torch.Tensor,
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
    embedding = mps.get_embedding(embedding)
    X = embedding(X, phys_dim)
    dataset = TensorDataset(X, t)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split in ['train', 'valid']),
        drop_last=(split == 'train')
    )
    logger.debug("DataLoader for categorisation intialized.")
    return loader

# TODO: Add tensor shape description


def eval(classifier: tk.models.MPSLayer,  # relies on mps.out_features = [cls_pos]
         loader: DataLoader,
         criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
         device: torch.device) -> Tuple[float, float]:
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
    classifier.eval()
    correct = 0
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for X, t in loader:
            X, t = X.to(device), t.to(device)
            p = mps.born_parallel(classifier, X)

            # Batchwise loss and predictions
            total_loss += criterion(p, t).item() * t.size(0)
            _, preds = torch.max(input=p, dim=-1)

            # Counting correct predictions
            correct += (preds == t).sum().item()

            total += t.size(0)
    acc = correct / total
    avg_loss = total_loss / total
    return acc, avg_loss

# TODO: Add tensor shape description


def _train_step(classifier: tk.models.MPSLayer,  # expect mps.out_features = [cls_pos]
                loader: DataLoader,
                device: torch.device,
                optimizer: torch.optim.Optimizer,
                # expects score, not probs. switch for str parameter
                loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                stage: str,
                watch_freq: int,
                step: int
                ) -> int:
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

    classifier.train()
    for X, t in loader:
        X, t = X.to(device), t.to(device)
        step += 1
        # Compute probs and loss
        p = mps.born_parallel(classifier, X)
        loss = loss_fn(p, t)

        # Backpropagate and update parameters
        optimizer.zero_grad()
        loss.backward()
        mps.log_grads(mps=classifier, watch_freq=watch_freq,
                      step=step, stage="pre")
        optimizer.step()

        wandb.log({f"{stage}_mps/train/loss": loss})
        return step


# TODO: Think about logging different or more quantities
# TODO: Consider removing title parameter


def train(classifier: tk.models.MPSLayer,
          loaders: Dict[str, DataLoader],  # loads embedded inputs and labels
          cfg: PretrainMPSConfig,
          device: torch.device,
          stage: str,
          title: str | None = None,
          goal_acc: float | None = None
          ) -> Tuple[dict, List[torch.FloatTensor], float]:
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
        check_counter = 0

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
    goal_acc: float, optional

    Returns
    -------
    best_tensors : list[torch.Tensor]
        The tensors of the MPS corresponding to the best validation accuracy.
    train_loss : list[float]
        Mini-batch-wise training loss collected during training.
    val_accuracy : list[float]
        Epoch-wise validation accuracy.
    test_accuracy : float
        Final test accuracy.
    """
    logger.info("Categorisation training begins.")
    best_acc = 0.0
    best_loss = float("inf")
    patience_counter = 0
    best_tensors = []  # Container for best model parameters
    step = 0

    # Prepare MPS for training
    classifier.to(device)
    classifier.auto_stack, classifier._auto_unbind = cfg.auto_stack, cfg.auto_unbind
    classifier.unset_data_nodes()
    classifier.reset()
    classifier.trace(torch.zeros(1, len(classifier.in_features),
                     classifier.in_dim[0]).to(device))

    # Instantiate criterion and optimizer
    criterion = get_criterion(cfg.criterion)
    optimizer = get_optimizer(classifier.parameters(), cfg.optimizer)

    for epoch in range(cfg.max_epoch):
        # Training step
        step = _train_step(
            classifier=classifier, loader=loaders['train'], device=device,
            optimizer=optimizer, loss_fn=criterion, stage=stage,
            watch_freq=cfg.watch_freq, step=step
        )

        # Validation step
        acc, val_loss = eval(classifier, loaders['valid'], criterion, device)
        wandb.log({
            f"{stage}_mps/valid/acc": acc,
            f"{stage}_mps/valid/loss": val_loss
        })

        # Model update and early stopping
        if cfg.stop_crit == "acc":
            if acc > best_acc:
                best_acc, patience_counter = acc, 0
                best_tensors = [t.clone().detach() for t in classifier.tensors]
            else:
                patience_counter += 1
        elif cfg.stop_crit == "loss":
            if val_loss < best_loss:
                best_loss, patience_counter = val_loss, 0
                best_tensors = [t.clone().detach() for t in classifier.tensors]
            else:
                patience_counter += 1
        if patience_counter > cfg.patience:
            logger.info(f"Early stopping via patience at epoch {epoch}")
            break

        # Optional early stopping via goal accuracy
        if (goal_acc is not None):
            if goal_acc < best_acc:
                logger.info(f"Early stopping via goal acc at epoch {epoch}")
            break
        # Update report
        if (epoch+1) % cfg.update_freq == 0:
            logger.info(f"Epoch {epoch+1}: val_accuracy={acc:.4f}")
        elif (epoch == 0) and (title is not None):
            logger.info(f'\nTraining of {title} dataset:\n')

    # Finish with evaluation on test set
    classifier.reset()
    classifier.initialize(tensors=best_tensors)
    classifier.auto_stack, classifier.auto_unbind = cfg.auto_stack, cfg.auto_unbind
    classifier.trace(torch.zeros(1, len(classifier.in_features),
                     classifier.in_dim[0]).to(device))
    test_accuracy, _ = eval(classifier, loaders["test"], criterion, device)
    logger.info(f"{test_accuracy=}")

    # Summarise training
    wandb.summary[f"{stage}_mps/test/acc"] = test_accuracy
    wandb.summary[f"{stage}_mps/best/acc"] = best_acc
    classifier.reset()

    return best_tensors, best_acc
