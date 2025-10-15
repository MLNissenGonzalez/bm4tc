import hydra
from pathlib import Path
import matplotlib.pyplot as plt
import mps.sampling as sampling
from src._utils import visualise_samples, FIDLike, mean_n_cov
import torch
import tensorkrowch as tk
from typing import Callable, Dict, Tuple, List
from torch.utils.data import TensorDataset, DataLoader
import schemas
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
        Input embedded using embedding, X_embedded.size()=(batch_size, n_feat, phys_dim)
    """
    embedding = mps.get_embedding(embedding)
    X_embedded = embedding(X, phys_dim)  # shape (batch_size, n_feat, phys_dim)
    dataset = TensorDataset(X_embedded, t)
    dataset.dim = X.shape[1]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split in ['train', 'valid']),
        drop_last=(split == 'train')
    )
    logger.debug("DataLoader for categorisation intialized.")
    return loader


def eval(classifier: tk.models.MPSLayer,  # relies on mps.out_features = [cls_pos]
         loader: DataLoader,
         criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
         device: torch.device) -> Tuple[float, float]:
    """
    Evaluate an MPS-based classifier on a dataset.

    Performs a forward pass through the MPSLayer without computing gradients,
    calculates batchwise loss using the provided criterion, and aggregates
    accuracy and average loss over the entire dataset. Probabilities are
    computed via the Born rule using `born_parallel`.

    Parameters
    ----------
    classifier : tk.models.MPSLayer
        MPS model expected to have `out_features = [cls_pos]`.
        Receives embedded inputs of shape `(batch_size, n_feat, phys_dim)`.
    loader : torch.utils.data.DataLoader
        DataLoader yielding batches `(X, t)` where:
        - `X`: FloatTensor of shape `(batch_size, n_feat, phys_dim)`
        - `t`: LongTensor of shape `(batch_size,)` with class labels
    criterion : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Loss function accepting predicted probabilities and targets.
        Returns a scalar tensor representing batch loss.
    device : torch.device
        Target device for evaluation.

    Returns
    -------
    acc : float
        Classification accuracy over the dataset.
    avg_loss : float
        Mean loss over all samples in the dataset.

    Notes
    -----
    - Uses `mps.born_parallel(classifier, X)` to compute predicted probabilities:
      - Shape `(batch_size, num_cls)` for conditional class probabilities.
    - Predicted classes are obtained via `torch.argmax(p, dim=-1)`.
    - Suitable for validation or test evaluation.
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
    Perform a single training epoch for an MPS-based classifier.

    Iterates over all batches in `loader`, computes probabilities using
    the Born rule (`born_parallel`), evaluates the loss, backpropagates
    gradients, updates model parameters, and logs average training loss.

    Parameters
    ----------
    classifier : tk.models.MPSLayer
        MPS model with `out_features = [cls_pos]`.
        Receives input tensors of shape `(batch_size, n_feat, phys_dim)`.
    loader : torch.utils.data.DataLoader
        DataLoader yielding `(X, t)` batches:
        - `X`: FloatTensor `(batch_size, n_feat, phys_dim)` — embedded input
        - `t`: LongTensor `(batch_size,)` — class labels
    device : torch.device
        Device to place model and data.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Loss function accepting predicted probabilities `p` and targets `t`.
    stage : str
        Training stage name, used as a prefix for logging (e.g., `"pre"`).
    watch_freq : int
        Step interval for gradient logging via `mps.log_grads()`.
    step : int
        Global step counter, incremented per batch.

    Returns
    -------
    step : int
        Updated global step count after processing all batches.

    Notes
    -----
    - Probabilities `p` are computed via `mps.born_parallel(classifier, X)`:
      - Shape `(batch_size, num_cls)` for conditional probabilities
    - Loss is computed per batch and detached to avoid memory leaks.
    - Gradients are zeroed before each backward pass.
    - Gradients can be logged to W&B using `mps.log_grads`.
    - Average epoch loss is logged as `f"{stage}_mps/train/loss"` to W&B.

    Shape Summary
    --------------
    X : (batch_size, n_feat, phys_dim)
    t : (batch_size,)
    p : (batch_size, num_cls)
    """

    log = []
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
        log.append(loss.detach().cpu().item())

    wandb.log({f"{stage}_mps/train/loss":
               sum(log)/len(log)})
    return step


# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# --------GEN CAPABILITIES---------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------


def sample_from_classifier(classifier: tk.models.MPSLayer,
                           device: torch.device,
                           cfg: schemas.SamplingConfig) -> torch.FloatTensor:
    """
    Generate synthetic samples from a pretrained MPS-based classifier.

    Constructs a temporary MPS generator from the classifier tensors
    and draws samples in batches to limit peak memory usage. Uses
    the `batched` function for parallelized Born-rule sampling.

    Parameters
    ----------
    classifier : tk.models.MPSLayer
        Pretrained MPS classifier. Must have dynamically added attribute:
        - `embedding` : str - embedding identifier (e.g., "fourier", "legendre")
    device : torch.device
        Device to place temporary generator and input embeddings.
    cfg : schemas.SamplingConfig
        Sampling configuration, with fields:
        - `num_spc` : int - total samples per class
        - `num_bins` : int - discretization bins for input space
        - `batch_spc` : int - samples per batch to control memory usage
        - `method` : str - name of sampling method, e.g. 'secant'
    Returns
    -------
    torch.FloatTensor
        Generated samples with shape `(num_spc, num_cls, data_dim)`:
        - `num_spc` : requested number of samples per class
        - `num_cls` : number of classes inferred from MPS
        - `data_dim` : dimensionality of the embedded input space

    Notes
    -----
    - A temporary MPS instance is created via `tk.models.MPS(tensors=classifier.tensors)`.
    - Memory is cleared after sampling using `torch.cuda.empty_cache()`.
    - This function does not perform gradient tracking (`torch.no_grad()`).
    """
    generator = tk.models.MPS(tensors=classifier.tensors, device=device)
    cls_pos = classifier.out_position
    with torch.no_grad():
        synths = sampling.batched(
            mps=generator,
            embedding=classifier.embedding,
            cls_pos=cls_pos, num_spc=cfg.num_spc,
            num_bins=cfg.num_bins,
            batch_spc=cfg.batch_spc,
            method=cfg.method,
            device=device).cpu()
    torch.cuda.empty_cache()
    return synths


fid_like = FIDLike()

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# --------SECONDARY TRAINING PERFORMANCE METRICS---------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

_METRICS = ["loss", "acc", "fid"]


def _performance_check(crit: str,
                       current: Dict[str, float],
                       best: Dict[str, float]) -> bool:
    """
    Check whether current performance is better than the best so far.

    Supports metrics: "loss", "acc", and "fid".

    Parameters
    ----------
    crit : str
        Metric to evaluate improvement. Must be one of "loss", "acc", "fid".
    current : dict of str -> float
        Current epoch metrics, e.g., {"loss": 0.5, "acc": 0.8, "fid": 12.0}.
    best : dict of str -> float
        Best metrics observed so far.

    Returns
    -------
    isBetter : bool
        True if `current[crit]` improves over `best[crit]`:
        - "acc": improvement if larger
        - "loss" or "fid": improvement if smaller
        - For "fid", None values are ignored

    Notes
    -----
    - Used internally for early stopping and metric tracking.
    """
    isBetter = False
    isBetter = (
        ((crit == "acc") and (current[crit] > best[crit])) or
        ((crit == "loss") and (current[crit] < best[crit]))
    )
    if crit == "fid" and current["fid"] is not None:
        isBetter = (current[crit] < best[crit])

    return isBetter


def _update(crit: str,
            current: Dict[str, float],
            best: Dict[str, float],
            patience_counter: int) -> Tuple[int, bool]:
    """
    Update best metrics and patience counter based on current performance.

    Checks if the current performance is better than the best for the
    specified criterion, updates the best metrics if improved, and
    increments or resets the patience counter accordingly.

    Parameters
    ----------
    crit : str
        Metric for performance comparison ("loss", "acc", or "fid").
    current : dict of str -> float
        Current epoch metrics.
    best : dict of str -> float
        Best metrics observed so far. Will be updated in-place if improvement.
    patience_counter : int
        Number of consecutive epochs without improvement.

    Returns
    -------
    patience_counter : int
        Updated patience counter (reset to 0 if improvement, incremented otherwise)
    isBetter : bool
        True if current performance improved over best for `crit`.

    Notes
    -----
    - Updates all metrics in `_METRICS` when improvement occurs.
    - Used for early stopping and monitoring secondary performance metrics.
    """
    # Condition check
    isBetter = _performance_check(crit=crit,
                                  current=current,
                                  best=best)

    # Update step
    if isBetter:
        for metric in _METRICS:
            best[metric] = current[metric]
        patience_counter = 0
    else:
        patience_counter += 1

    return patience_counter, isBetter

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# --------TRAINING LOOP---------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------


def train(classifier: tk.models.MPSLayer,
          loaders: Dict[str, DataLoader],  # loads embedded inputs and labels
          cfg: schemas.PretrainMPSConfig,
          samp_cfg: schemas.SamplingConfig,
          device: torch.device,
          stage: str,
          goal_acc: float | None = None,
          stat_r: Dict[int, Tuple[torch.FloatTensor,
                                  torch.FloatTensor]] | None = None
          ) -> Tuple[List[torch.Tensor], float]:
    """
    Pretrain an MPS-based classifier with optional generative evaluation.

    This function trains the provided MPS classifier on embedded input
    data using mini-batch gradient descent. It supports early stopping
    based on a chosen criterion (accuracy, loss, or FID), optional
    goal accuracy, and logging to Weights & Biases (W&B). Additionally,
    generative capabilities can be monitored via sampling and FID-like
    evaluation.

    Parameters
    ----------
    classifier : tk.models.MPSLayer
        MPS classifier with attributes:
        - `tensors` : list of torch.Tensor — model tensors for training
        - `in_features` : list — input site indices
        - `in_dim` : tuple — local site dimension
        - `embedding` : str (dynamically added) — embedding name for sampling
        - `out_position` : int — index of output tensor
    loaders : dict of str -> DataLoader
        Dictionary containing DataLoaders for `"train"`, `"valid"`, and `"test"`.
        Each loader yields batches `(X, t)`:
        - `X`: FloatTensor `(batch_size, n_feat, phys_dim)` — embedded inputs
        - `t`: LongTensor `(batch_size,)` — class labels
    cfg : schemas.PretrainMPSConfig
        Training configuration including:
        - `max_epoch` : int — maximum epochs
        - `stop_crit` : str — early stopping criterion ("acc", "loss", "fid")
        - `optimizer`, `criterion` : optimizer and loss configs
        - `auto_stack`, `auto_unbind` : MPS stacking/unbinding flags
        - `watch_freq`, `update_freq` : logging frequencies
        - `patience` : int — patience for early stopping
        - `toViz` : bool — whether to visualize generated samples
    samp_cfg : schemas.SamplingConfig
        Configuration for sample generation (num_spc, num_bins, batch_spc)
    device : torch.device
        Device to place model and data.
    stage : str
        Stage name used for W&B logging (e.g., `"pre"` or `"fine"`).
    goal_acc : float, optional
        Optional target accuracy for early stopping.
    stat_r : dict of int -> (torch.FloatTensor, torch.FloatTensor), optional
        Per-class mean and covariance for FID-like computation on low-dimensional data.

    Returns
    -------
    best_tensors : list of torch.Tensor
        MPS tensors corresponding to the epoch with the best observed performance.
    best_acc : float
        Best validation accuracy achieved during training.

    Notes
    -----
    - Training uses `_train_step` for batch-wise forward/backward passes.
    - Evaluation uses `eval` on validation batches and optionally computes
      FID-like metrics using `sample_from_classifier` and `FIDLike`.
    - Early stopping occurs if:
        1. The chosen criterion (`cfg.stop_crit`) does not improve over `cfg.patience` epochs
        2. Optional `goal_acc` is reached
    - Logging:
        - Metrics (`loss`, `acc`, `fid`) are logged per epoch to W&B.
        - Sample visualizations are optionally logged if `cfg.toViz` is True or low-dimensional data (<1000 features).
    - Tensor shapes:
        - `X` : `(batch_size, n_feat, phys_dim)`
        - `t` : `(batch_size,)`
        - Generated samples : `(num_spc, num_cls, data_dim)`
    - After training, the classifier is reset, and best tensors are restored
      for final evaluation on the test set.
    """

    logger.info("Categorisation training begins.")
    best = {}
    wandb.define_metric(f"{stage}_mps/train/loss", summary="none")
    for crit in _METRICS:
        if crit == "acc":
            best[crit] = 0.0
        else:
            best[crit] = float("inf")

    # TODO: Implement something more general and configurable
    # fid_like metric too expensive higher dim data, in the current implementation
    if loaders["train"].dataset.dim < 1e3:
        mu_r, cov_r = {}, {}
        for c in stat_r.keys():
            mu_r[c], cov_r[c] = stat_r[c]

    best_epoch, step, patience_counter = 0, 0, 0
    best_tensors = [t.clone().detach() for t in classifier.tensors]

    # Prepare MPS for training
    classifier.to(device)
    classifier.auto_stack, classifier._auto_unbind = cfg.auto_stack, cfg.auto_unbind
    classifier.unset_data_nodes(), classifier.reset()
    classifier.trace(torch.zeros(1, len(classifier.in_features),
                     classifier.in_dim[0]).to(device))

    # Instantiate criterion and optimizer
    criterion = get_criterion(cfg.criterion)
    optimizer = get_optimizer(classifier.parameters(), cfg.optimizer)

    for epoch in range(cfg.max_epoch):
        # Training step
        step = _train_step(
            classifier=classifier, loader=loaders["train"], device=device,
            optimizer=optimizer, loss_fn=criterion, stage=stage,
            watch_freq=cfg.watch_freq, step=step)

        # Performance evaluation
        current = {}
        current["acc"], current["loss"] = eval(classifier,
                                               loaders["valid"],
                                               criterion, device)
        if cfg.stop_crit == "fid":
            fid_values = []
            synths = sample_from_classifier(classifier=classifier,
                                            cfg=samp_cfg, device=device)
            for c in range(synths.shape[1]):
                gen = synths[:, c, :]
                fid_val = fid_like.lazy_forward(mu_r[c], cov_r[c], gen)
                fid_values.append(fid_val)
            current["fid"] = torch.mean(torch.stack(fid_values)).item()
        else:
            current["fid"] = None

        # Report to stream
        if (epoch+1) % cfg.update_freq == 0:
            logger.info(f"Epoch {epoch+1}: valid acc = {current['acc']}")

            # Reporting generative capabilities
            if cfg.toViz or loaders["train"].dataset.dim < 1000:
                synths = sample_from_classifier(classifier=classifier,
                                                cfg=samp_cfg, device=device)
                if loaders["train"].dataset.dim < 1000:
                    for c in range(synths.shape[1]):
                        gen = synths[:, c, :]
                        current[f"fid/{c}"] = fid_like.lazy_forward(
                            mu_r[c], cov_r[c], gen)
                if cfg.toViz:
                    ax = visualise_samples(
                        samples=synths, gen_viz=samp_cfg.batch_spc)
                    wandb.log({f"samples/{stage}": wandb.Image(ax.figure)})
                    plt.close(ax.figure)

        # Log performance
        performance_log = {}
        for crit in _METRICS:
            performance_log[f"{stage}_mps/valid/{crit}"] = current[crit]
        wandb.log(performance_log)

        # Update
        patience_counter, isBetter = _update(crit=cfg.stop_crit, current=current, best=best,
                                             patience_counter=patience_counter)
        if isBetter:
            best_epoch = epoch + 1
            best_tensors = [t.clone().detach() for t in classifier.tensors]

        # Early stopping via patience
        if patience_counter > cfg.patience:
            logger.info(f"Early stopping via patience at epoch {epoch}")
            break
        # Optional early stopping via goal accuracy
        if (goal_acc is not None):
            if goal_acc < best["acc"]:
                logger.info(f"Early stopping via goal acc at epoch {epoch}")
                break

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
    for metric in _METRICS:
        wandb.summary[f"{stage}_mps/valid/{metric}"] = best[metric]
    wandb.summary[f"{stage}_mps/epoch/best"] = best_epoch
    wandb.summary[f"{stage}_mps/epoch/last"] = epoch + 1
    classifier.reset()

    return best_tensors, best["acc"]


# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# --------SAVING MODEL WEIGHTS AFTER PRETRAINING---------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------


# Saving MPS after pretrainin -> save data set, architecture, embedding, weight decay, stopping criterion, max epoch

def save(model: tk.models.MPSLayer, cfg: schemas.Config):
    """
    Save a pretrained MPSLayer to disk with a descriptive filename.

    The function resets the model, constructs a folder inside Hydra's
    current run directory, and saves the model's state dictionary. The
    filename encodes the dataset, bond dimension, input dimension, embedding,
    optimizer, stopping criterion, weight decay, and maximum epoch.

    Parameters
    ----------
    model : tk.models.MPSLayer
        Pretrained MPS model whose state_dict will be saved.
        The model is reset before saving to ensure a clean state.
    cfg : schemas.Config
        Configuration object containing:
        - `dataset.name` : str — dataset identifier
        - `model.mps.embedding` : str — embedding used for the MPS
        - `pretrain.mps.optimizer.name` : str — optimizer name
        - `pretrain.mps.optimizer.kwargs.weight_decay` : float, optional
        - `pretrain.mps.stop_crit` : str — early stopping criterion
        - `pretrain.mps.max_epoch` : int — maximum number of epochs

    Returns
    -------
    filename : str
        Generated descriptive filename (without path).
    save_path : str
        Full path to the saved model state dictionary.

    Notes
    -----
    - The model is saved as a PyTorch state dictionary using `torch.save`.
    - The save folder is `<hydra_run_dir>/models/`.
    - The filename format is:
      `"{dataset}_bd{bond_dim}_in{in_dim}_{embedding}_{optimizer}_{stop_crit}_wd{weight_decay}_ep{max_epoch}"`
    - If `weight_decay` is not specified in the optimizer kwargs, it defaults to 0.
    - Example filename:
      `"mnist_bd10_in2_fourier_sgd_acc_wd0_ep50"`
    """

    model.reset()

    # Hydra's current run dir
    run_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Models subfolder inside it
    folder = run_dir / "models"
    folder.mkdir(parents=True, exist_ok=True)

    # File name
    (dataset, bond_dim, in_dim,
     embedding, optimizer,
     stopping_criterion, max_epoch) = (cfg.dataset.name, model.bond_dim[0],
                                       model.in_dim[0],
                                       cfg.model.mps.embedding,
                                       cfg.pretrain.mps.optimizer.name,
                                       cfg.pretrain.mps.stop_crit,
                                       cfg.pretrain.mps.max_epoch)
    if "weight_decay" in cfg.pretrain.mps.optimizer.kwargs.keys():
        weight_decay = cfg.pretrain.mps.optimizer.kwargs.weight_decay
    else:
        weight_decay = 0
    filename_components = [
        dataset, f"bd{bond_dim}", f"in{in_dim}", embedding, optimizer,
        stopping_criterion, f"wd{weight_decay}", f"ep{max_epoch}"
    ]
    filename = "_".join(filename_components)

    # Saving
    save_path = folder / filename
    torch.save(model.state_dict(), save_path)

    return filename, str(save_path)
