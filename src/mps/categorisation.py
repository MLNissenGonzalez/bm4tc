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
    X_embedded = embedding(X, phys_dim) # shape (batch_size, n_feat, phys_dim)
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
    
    # TODO: Update docstring (ADDED AS ISSUE)
    # TODO: Add tensor shape description (ADDED AS ISSUE)


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
    
    # TODO: Add tensor shape description (ADDED AS ISSUE)
    # TODO: Update docstring (ADDED AS ISSUE)

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


#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#--------GEN CAPABILITIES---------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
from src._utils import visualise_samples, FIDLike, mean_n_cov
import mps.sampling as sampling
import matplotlib.pyplot as plt

def sample_from_classifier(classifier: tk.models.MPSLayer,
                           device: torch.device,
                           cfg: schemas.SamplingConfig):
    # TODO: Add docstring (ADDED AS ISSUE)
    generator = tk.models.MPS(tensors=classifier.tensors, device=device)
    cls_pos = classifier.out_position
    with torch.no_grad():
        synths = sampling.batched(
            mps=generator,
            embedding=classifier.embedding,
            cls_pos=cls_pos, num_spc=cfg.num_spc,
            num_bins=cfg.num_bins,
            batch_spc=cfg.batch_spc,
            device=device).cpu()
    torch.cuda.empty_cache()
    return synths

fid_like = FIDLike()

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#--------SECONDARY TRAINING PERFORMANCE METRICS---------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

_METRICS = ["loss", "acc", "fid"]

def _performance_check(crit: str, 
                   current: Dict[str, float],
                   best: Dict[str, float]):
    isBetter = False
    isBetter = (
        ((crit=="acc") and (current[crit]>best[crit])) or
        ((crit=="loss") and (current[crit]<best[crit]))
        )
    if crit == "fid" and current["fid"] is not None:
        isBetter = (current[crit]<best[crit])

    return isBetter

def _update(crit: str, 
            current: Dict[str, float], 
            best: Dict[str, float],
            patience_counter: int):
    
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

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#--------TRAINING LOOP---------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

def train(classifier: tk.models.MPSLayer,
          loaders: Dict[str, DataLoader],  # loads embedded inputs and labels
          cfg: schemas.PretrainMPSConfig,
          samp_cfg: schemas.SamplingConfig,
          device: torch.device,
          stage: str,
          goal_acc: float | None = None,
          stat_r: Dict[int, Tuple[torch.FloatTensor, torch.FloatTensor]] | None = None
          ) -> Tuple[List[torch.Tensor], float]:
    # TODO: Update docstring (ADDED AS ISSUE)

    logger.info("Categorisation training begins.")
    best = {}
    wandb.define_metric(f"{stage}_mps/train/loss", summary="none")
    for crit in _METRICS:
        if crit=="acc": best[crit]=0.0
        else: best[crit]=float("inf")
    
    # TODO: Implement something more general and configurable
    if loaders["train"].dataset.dim < 1e3: # fid_like metric too expensive higher dim data, in the current implementation
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
                        current[f"fid/{c}"] = fid_like.lazy_forward(mu_r[c], cov_r[c], gen)
                if cfg.toViz:
                    ax = visualise_samples(samples=synths, gen_viz=samp_cfg.batch_spc)
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


#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#--------SAVING MODEL WEIGHTS AFTER PRETRAINING---------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

from pathlib import Path
import hydra
# Saving MPS after pretrainin -> save data set, architecture, embedding, weight decay, stopping criterion, max epoch
def save(model: tk.models.MPSLayer, cfg: schemas.Config):
    model.reset()

    # Hydra's current run dir
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

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