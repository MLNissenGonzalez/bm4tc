import torch
import math
import tensorkrowch as tk
from typing import Callable, Dict, List, Any, Tuple
import mps.categorisation as mps_cat
import mps.sampling as sampling
import mps.utils as mps
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import schemas
from _utils import get_criterion, get_optimizer, _class_wise_dataset_size
import wandb

import logging
logger = logging.getLogger(__name__)


def real_loader(X: torch.FloatTensor,
                c: torch.LongTensor,
                n_real: int, # corresponds roughly to num_cls * batch_spc
                split: str) -> DataLoader:
    """
    Construct a DataLoader for real, unembedded data samples.

    This is used to provide real samples (with class indices) during
    adversarial training of the discriminator/generator.

    Parameters
    ----------
    X : torch.FloatTensor
        Tensor of preprocessed, non-embedded input features. Shape: (N, ...).
    c : torch.LongTensor
        Tensor of integer class labels for the samples. Shape: (N,).
    n_real : int
        Batch size for the DataLoader (number of real samples per step).
    split : {'train', 'valid', 'test'}
        Dataset split type. For 'train' and 'valid' the DataLoader is shuffled.
        If 'train', batches are dropped if incomplete (drop_last=True).

    Returns
    -------
    DataLoader
        A DataLoader over the real dataset with the requested batching and split behavior.
    """
    dataset = TensorDataset(X, c)
    loader = DataLoader(dataset,
                        batch_size=n_real,
                        shuffle=(split in ['train', 'valid']),
                        drop_last=(split == 'train')
                        )
    return loader


def _batch_constructor(generator: tk. models.MPS,
                       # Input from iteratable
                       X_real: torch.FloatTensor,
                       c_real: torch.LongTensor,
                       # Hyperparameters
                       cls_pos: int,
                       num_bins: int,
                       r_real: int,
                       # Initialized in outer loop
                       in_dim: int,
                       num_cls: int,
                       cls_embs: List[torch.FloatTensor],
                       embedding: Callable[[torch.FloatTensor, int], torch.FloatTensor],
                       input_space: torch.FloatTensor,
                       device: torch.device
                       ) -> Tuple[List[torch.FloatTensor], List[torch.LongTensor]]:
    """
    Construct mixed batches of real and synthetic samples for adversarial training.

    For each class, the function extracts the subset of real samples,
    generates a proportional number of synthetic samples from the MPS model,
    assigns binary labels (1 for real, 0 for synthetic), and shuffles the
    combined set before returning.

    Parameters
    ----------
    mps : tk.models.MPS
        The MPS generative model used to sample synthetic data.
    X_real : torch.FloatTensor
        Tensor of real input samples (preprocessed, non-embedded).
    c_real : torch.LongTensor
        Corresponding integer class labels for the real samples.
    cls_pos : int
        Index position of the class label within the MPS input dimension.
    num_bins : int
        Number of discretization bins used in sampling.
    r_synth : float
        Ratio controlling how many synthetic samples to generate per class.
    in_dim : int
        Input dimensionality of the MPS model.
    cls_embs : List[torch.Tensor]
        Class embeddings (one per class).
    embedding : Callable[[torch.FloatTensor, int], torch.FloatTensor]
        Function mapping input data to the embedded space expected by the MPS.
    input_space : torch.FloatTensor
        Input discretization space used by the sampler.
    device : torch.device
        Torch device to perform computation on.

    Returns
    -------
    X : list of torch.FloatTensor
        Per-class tensors of mixed real + synthetic samples.
    t : list of torch.LongTensor
        Per-class binary target labels (1=real, 0=synthetic).
    X_synth : list of torch.FloatTensor
        Per-class synthetic-only samples generated for generator updates.
    """
    n_real_pc = _class_wise_dataset_size(c_real, num_cls)
    X, t, X_synth = [], [], []
    # maybe replace this with dis.keys() and make this mode universal
    for c, cls_emb in enumerate(cls_embs):
        # Creation of dataset for discriminator
        X_real_c = X_real[c_real == c]
        n_synth_c = math.ceil(n_real_pc[c] / r_real)
        X_synth_c = sampling._single_class(mps=generator,
                                           embedding=embedding,
                                           cls_pos=cls_pos,
                                           cls_emb=cls_emb,
                                           in_dim=in_dim,
                                           num_bins=num_bins,
                                           input_space=input_space,
                                           num_spc=n_synth_c,
                                           device=device)

        t_real_c = torch.ones(len(X_real_c))
        t_synth_c = torch.zeros(n_synth_c)
        t_real_c.to(device=device, dtype=torch.long)
        t_synth_c.to(device=device, dtype=torch.long)
        X_c = torch.cat([X_real_c, X_synth_c.detach()], dim=0)
        t_c = torch.cat([t_real_c, t_synth_c], dim=0)
        perm = torch.randperm(X_c.shape[0])
        X_c, t_c = X_c[perm], t_c[perm]
        X.append(X_c)
        t.append(t_c)
        X_synth.append(X_synth_c)
    return X, t, X_synth


def _step(generator: tk.models.MPS,
          dis: Dict[int, nn.Module],

          # Input to the loop
          X_real: torch.FloatTensor,
          c_real: torch.LongTensor,

          # Hyperparameters
          r_real: float,
          cls_pos: int,
          num_bins: int,

          # Initialized in outer loop
          in_dim: int,
          num_cls: int,
          embedding: Callable[[torch.FloatTensor, int], torch.FloatTensor],
          cls_embs: List[torch.FloatTensor],
          input_space: torch.FloatTensor,
          d_optimizer: Dict[Any, torch.optim.Optimizer],
          g_optimizer: torch.optim.Optimizer,
          d_criterion: Callable[[torch.FloatTensor, torch.Tensor], torch.FloatTensor],
          g_criterion: Callable[[torch.FloatTensor, torch.Tensor], torch.FloatTensor],
          watch_freq: int,
          step: int,
          device: torch.device
          ) -> Tuple[Dict[str, float], int]:
    # TODO: Add docstring
    
    X, t, X_synth = _batch_constructor(
        generator=generator, X_real=X_real, c_real=c_real,
        cls_pos=cls_pos, num_bins=num_bins, r_real=r_real,
        in_dim=in_dim, num_cls=num_cls, cls_embs=cls_embs,
        embedding=embedding, input_space=input_space, device=device)

    if len(dis.keys()) == 1:  # i.e. single discriminator instead ensemble of discriminators
        X = torch.concatenate(X, dim=0)
        t = torch.concatenate(t, dim=0)
        X_synth = torch.concatenate(X_synth, dim=0)

        perm = torch.randperm(X.shape[0])
        X, t, X_synth = {"single": X[perm]}, {
            "single": t[perm]}, {"single": X_synth}

    d_loss, g_loss, d_acc, log = {}, {}, {}, {}
    for c in dis.keys():
        step += 1
        # Discriminator prediction
        d = dis[c]
        d_logit = d(X[c])
        d_prob = torch.sigmoid(d_logit.squeeze())
        d_target = t[c].to(device=device, dtype=torch.float32)
        d_loss[c] = d_criterion(d_prob, d_target)
        d_pred = (d_prob >= 0.5).float()
        d_acc[c] = (d_pred == d_target).float().mean()
        
        # Discriminator update step
        d_optimizer[c].zero_grad()
        d_loss[c].backward()
        d_optimizer[c].step()

        # Log discriminator performance, to expensive, not informative
        log[f"gan_dis/{c}/loss"] = d_loss[c].detach().cpu().item()
        log[f"gan_dis/{c}/acc"] = d_acc[c].detach().cpu().item()

        # Generator performance against discriminator
        g_logit = d(X_synth[c])
        g_prob = torch.sigmoid(g_logit.squeeze())
        # TODO: Target kind of determines what loss is below
        g_target = torch.ones_like(g_prob).to(
            device=device, dtype=torch.float32)
        g_loss[c] = g_criterion(g_prob, g_target)

        # Update step and gradient tracking of generator
        g_optimizer.zero_grad()
        g_loss[c].backward() #
        mps.log_grads(generator, step=step, watch_freq=watch_freq, stage="gan")
        # TODO: This optimizer step is only minimising currently
        g_optimizer.step()

        # Log generator performance
        log[f"gan_mps/{c}/loss"] = g_loss[c].detach().cpu().item()

    return log, step

from mps.categorisation import sample_from_classifier
from _utils import mean_n_cov, FIDLike, visualise_samples
import matplotlib.pyplot as plt
fid_like = FIDLike()

def check_and_retrain(generator: tk.models.MPS,
                      loaders: Dict[str, DataLoader],
                      g_optimizer: torch.optim.Optimizer,
                      cfg: schemas.PretrainMPSConfig,
                      cls_pos: int, epoch: int, toViz: bool,
                      samp_cfg: schemas.SamplingConfig,
                      trigger_accuracy: float,
                      device: torch.device,
                      cfg_g_optimizer: schemas.OptimizerConfig,
                      stat_r: Dict[int, Tuple[torch.FloatTensor, torch.FloatTensor]] | None = None
                      ) -> torch.optim.Optimizer:
    # TODO: Add updated docstring
    log = {}
    generator.reset()
    classifier = tk.models.MPSLayer(
        tensors=generator.tensors, out_position=cls_pos, device=device)
    classifier.embedding = generator.embedding
    classifier.trace(torch.zeros(0, len(classifier.in_features),
                     classifier.in_dim[0]).to(device))
    classifier.to(device), 
    
    # Log generative capabilities, if wanted
    if loaders["train"].dataset.dim < 1e3 or toViz:
        synths = sample_from_classifier(classifier=classifier, device=device, 
                                        cfg=samp_cfg)
        # TODO: Implement something more general and configable
        if loaders["train"].dataset.dim < 1e3: # fid_like metric too expensive higher dim data, in the current implementation
            fid_values = []
            for c in stat_r.keys():
                mu_r, cov_r = stat_r[c]
                gen = synths[:, c, :]
                fid_val = fid_like.lazy_forward(mu_r, cov_r, gen)
                fid_values.append(fid_val)
            log[f"gan_mps/fid"] = torch.mean(torch.stack(fid_values)).item()
        if toViz:
            ax = visualise_samples(synths, gen_viz=samp_cfg.batch_spc)
            log[f"samples/gan"] = wandb.Image(ax.figure)
            plt.close(ax.figure)

    # Evaluate classification performance on validation set
    criterion = get_criterion(cfg.criterion)
    acc, avg_loss = mps_cat.eval(classifier=classifier, loader=loaders["valid"],
                                 criterion=criterion, device=device)
    log["gan_mps/valid/acc"] = acc
    log["gan_mps/valid/loss"] = avg_loss

    wandb.log(log)
    log.clear()

    if acc < trigger_accuracy:
        # Retrain mps as a classifier
        logger.info(f'Starting to retrain at epoch {epoch+1}')
        best_tensors, best_acc = mps_cat.train(
            classifier=classifier, stat_r=stat_r,
            loaders=loaders,cfg=cfg,device=device,
            stage="gan",samp_cfg=samp_cfg)
        
        # Log generative capabilities again, if wanted
        if loaders["train"].dataset.dim < 1e3 or toViz:
            synths = sample_from_classifier(classifier=classifier, device=device, 
                                            cfg=samp_cfg)
            if loaders["train"].dataset.dim < 1e3: # fid_like metric too expensive higher dim data, in the current implementation
                fid_values = []
                for c in stat_r.keys():
                    mu_r, cov_r = stat_r[c]
                    gen = synths[:, c, :]
                    fid_val = fid_like.lazy_forward(mu_r, cov_r, gen)
                    fid_values.append(fid_val)
                log[f"gan_mps/fid"] = torch.mean(torch.stack(fid_values)).item()
            if toViz:
                ax = visualise_samples(synths, gen_viz=samp_cfg.batch_spc)
                log[f"samples/gan"] = wandb.Image(ax.figure)
                plt.close(ax.figure)
            wandb.log(log)

        # Update generator and g_optimizer (new params)
        generator.initialize(tensors=best_tensors, device=device)
        g_optimizer = get_optimizer(generator.parameters(), cfg_g_optimizer)
        logger.info("Retraining finished.")
    # g_optimizer could be a new object
    return g_optimizer

def _init_logs(dis: Dict[Any, nn.Module]) -> Dict[str, list]:
    logs = {}
    for c in dis.keys():
        logs[f"gan_mps/{c}/loss"] = []
        logs[f"gan_dis/{c}/loss"] = []
        logs[f"gan_dis/{c}/acc"] = []
    return logs

def _define_metrics(dis: Dict[Any, nn.Module]):
    for c in dis.keys():
        wandb.define_metric(f"gan_mps/{c}/loss", summary="none")
        wandb.define_metric(f"gan_dis/{c}/loss", summary="none")
        wandb.define_metric(f"gan_dis/{c}/acc")
    wandb.define_metric("gan_mps/valid/loss", summary="none")

# TODO: Add checkpoints
def loop(generator: tk.models.MPS,
         dis: Dict[Any, nn.Module],
         real_loaders: Dict[str, DataLoader],
         cfg: schemas.GANStyleConfig,
         samp_cfg: schemas.SamplingConfig,
         cls_pos: int,
         embedding: str,

         # Retraining specifics
         best_acc: float,
         cat_loaders: Dict[str, DataLoader],

         device: torch.device,

         # Optional kwargs
         stat_r: Dict[int, Tuple[torch.FloatTensor, torch.FloatTensor]] | None = None
         ) -> None:
    
    # TODO: Add updated docstring

    trigger_accuracy = max(best_acc-cfg.acc_drop_tol, 0.0)
    logger.info(f"{trigger_accuracy=}")
    step = 0
    _define_metrics(dis)

    generator.unset_data_nodes()
    generator.reset()
    generator.to(device)

    # Initialize gantraining specific objects
    d_optimizer = {}
    for c in dis.keys():
        d_optimizer[c] = get_optimizer(dis[c].parameters(), cfg.d_optimizer)
    g_optimizer = get_optimizer(generator.parameters(), cfg.g_optimizer)
    d_criterion = get_criterion(cfg.d_criterion)
    g_criterion = get_criterion(cfg.g_criterion)

    # Initialize other objects needed for sampling
    input_space = mps._embedding_to_range(embedding)
    input_space = torch.linspace(input_space[0], input_space[1], samp_cfg.num_bins)
    input_space = input_space.to(device, dtype=torch.float32)
    embedding = mps.get_embedding(embedding)
    in_dim, num_cls = mps._get_indim_and_ncls(generator)
    cls_embs = []
    for c in range(num_cls):
        cls_emb = tk.embeddings.basis(torch.tensor(c), num_cls)
        cls_emb = cls_emb.to(device=device, dtype=torch.float32)
        cls_embs.append(cls_emb)
    # Epoch loop
    for epoch in range(cfg.max_epoch):
        logs = _init_logs(dis)
        if ((epoch+1) % cfg.info_freq) == 0:
            logger.info(f"GAN training at epoch={epoch+1}")

        # Actual GAN-style training
        for X_real, c_real in real_loaders["train"]:
            X_real, c_real = X_real.to(device), c_real.to(device)
            log, step = _step(generator=generator, dis=dis, X_real=X_real, c_real=c_real,
                         r_real=cfg.r_real, cls_pos=cls_pos, step=step,
                         num_bins=samp_cfg.num_bins, in_dim=in_dim,
                         num_cls=num_cls, embedding=embedding,
                         cls_embs=cls_embs, input_space=input_space,
                         d_optimizer=d_optimizer, g_optimizer=g_optimizer,
                         d_criterion=d_criterion, g_criterion=g_criterion,
                         watch_freq=cfg.watch_freq, device=device)
            # Saving batchwise training performance as list
            for k in logs.keys():
                logs[k].append(log[k])
        # Logging the epoch-wise train loss average
        for k in logs.keys():
            logs[k] = sum(logs[k]) / len(logs[k])
        logs[f"gan_mps/epoch"] = epoch
        wandb.log(logs)

        # Checking classification accuracy and retraining if necessary
        if (epoch+1) % cfg.check_freq == 0:
            g_optimizer = check_and_retrain(generator=generator, loaders=cat_loaders,
                                            cfg=cfg.retrain, cls_pos=cls_pos, samp_cfg=samp_cfg,
                                            g_optimizer=g_optimizer, epoch=epoch, device=device,
                                            trigger_accuracy=trigger_accuracy, toViz=cfg.toViz,
                                            cfg_g_optimizer=cfg.g_optimizer, stat_r=stat_r)