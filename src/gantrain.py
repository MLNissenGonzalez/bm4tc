import torch
import math
import tensorkrowch as tk
from typing import Callable, Dict, List, Any, Tuple
import mps.categorisation as mps_cat
import mps.sampling as sampling
from mps.utils import get_embedding, _embedding_to_range, _get_indim_and_ncls, log_mps_grads
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import schemas
from _utils import get_criterion, get_optimizer, _class_wise_dataset_size
import wandb

import logging
logger = logging.getLogger(__name__)


def real_loader(X: torch.FloatTensor,
                c: torch.LongTensor,
                n_real: int,
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
                       r_synth: int,
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
    n_npc = _class_wise_dataset_size(c_real, num_cls)
    X, t, X_synth = [], [], []
    # maybe replace this with dis.keys() and make this mode universal
    for c, cls_emb in enumerate(cls_embs):
        # Creation of dataset for discriminator
        X_real_c = X_real[c_real == c]
        n_synth_c = math.ceil(r_synth * n_npc[c])
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
          r_synth: float,
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
          device: torch.device
          ) -> None:
    """
    Perform one adversarial training step for the discriminator(s) and generator (MPS).

    This function builds batches of real and synthetic samples per class,
    updates each discriminator on its corresponding data, then updates
    the generator (MPS) to fool the discriminators.

    Parameters
    ----------
    mps : tk.models.MPS
        Generator model producing synthetic samples.
    dis : dict[int, nn.Module]
        Dictionary of discriminators, keyed by class index.
    X_real : torch.FloatTensor, shape (B, D_in)
        Real samples for this step.
    c_real : torch.LongTensor, shape (B,)
        Class labels for real samples.
    r_synth : float
        Ratio of synthetic samples to real samples per class.
    cls_pos : int
        Position of the class label within the MPS input dimension.
    num_bins : int
        Number of discretization bins for sampling.
    in_dim : int
        Input dimensionality of the MPS.
    embedding : callable
        Embedding function mapping real data to the MPS input space.
    cls_embs : list of torch.Tensor
        List of class embeddings, length = C.
    input_space : torch.FloatTensor, shape (num_bins,)
        Discretized input space for MPS.
    d_optimizer : dict[int, torch.optim.Optimizer]
        Optimizers for each discriminator.
    g_optimizer : torch.optim.Optimizer
        Optimizer for the generator (MPS).
    d_criterion : callable
        Loss function for discriminator training.
    g_criterion : callable
        Loss function for generator training.
    device : torch.device
        Torch device for computation.

    Returns
    -------
    d_losses : torch.FloatTensor, shape (1, C) or (1, 1)
        Discriminator losses for this step, per class (row vector).
    g_losses : torch.FloatTensor, shape (1, C) or (1, 1)
        Generator losses for this step, per class (row vector).
    """

    X, t, X_synth = _batch_constructor(
        generator=generator, X_real=X_real, c_real=c_real,
        cls_pos=cls_pos, num_bins=num_bins, r_synth=r_synth,
        in_dim=in_dim, num_cls=num_cls, cls_embs=cls_embs,
        embedding=embedding, input_space=input_space, device=device)

    if len(dis.keys()) == 1:  # i.e. single discriminator instead ensemble of discriminators
        X = torch.concatenate(X, dim=0)
        t = torch.concatenate(t, dim=0)
        X_synth = torch.concatenate(X_synth, dim=0)

        perm = torch.randperm(X.shape[0])
        X, t, X_synth = {"single": X[perm]}, {
            "single": t[perm]}, {"single": X_synth}

    for c in dis.keys():
        # Discriminator update
        d = dis[c]
        d_optimizer[c].zero_grad()
        d_logit = d(X[c])
        d_prob = torch.sigmoid(d_logit.squeeze())
        d_target = t[c].to(device=device, dtype=torch.float32)
        d_loss = d_criterion(d_prob, d_target)
        d_loss.backward()
        d_optimizer[c].step()

        # Convert probabilities to binary predictions
        d_pred = (d_prob >= 0.5).float()
        d_acc = (d_pred == d_target).float().mean()

        # Generator update
        g_optimizer.zero_grad()
        g_logit = d(X_synth[c])
        g_prob = torch.sigmoid(g_logit.squeeze())
        g_target = torch.ones_like(g_prob).to(
            device=device, dtype=torch.float32)
        g_loss = g_criterion(g_prob, g_target)
        g_loss.backward()
        log_mps_grads(generator, watch_freq=watch_freq, stage="gan")
        g_optimizer.step()

        wandb.log({
            f"gan_dis/{c}/loss": d_loss.detach(),
            f"gan_mps/loss": g_loss.detach(),
            f"gan_dis/{c}/acc": d_acc.detach()
            })


def check_and_retrain(generator: tk.models.MPS,
                      loaders: Dict[str, DataLoader],
                      g_optimizer: torch.optim.Optimizer,
                      cfg: schemas.PretrainMPSConfig,
                      cls_pos: int,
                      epoch: int,
                      trigger_accuracy: float,
                      device: torch.device) -> tk.models.MPS:
    """
    Evaluate the generator (MPS) on validation data and optionally retrain it.

    If validation accuracy falls below the trigger threshold, retraining
    is performed using the provided retraining configuration.

    Parameters
    ----------
    mps : tk.models.MPS
        Current generator model.
    loaders : dict[str, DataLoader]
        Dataloaders for training and validation.
    cfg : schemas.PretrainMPSConfig
        Configuration for retraining the MPS.
    cls_pos : int
        Position of the class label in the MPS input dimension.
    epoch : int
        Current training epoch (for logging).
    trigger_accuracy : float
        Accuracy threshold that triggers retraining.
    device : torch.device
        Torch device for computation.

    Returns
    -------
    mps : tk.models.MPS
        Updated (possibly retrained) MPS generator.
    accuracy : list of float
        Validation accuracies before and after retraining.
    loss : list of float
        Validation losses before and after retraining.
    """
    generator.reset()
    classifier = tk.models.MPSLayer(
        tensors=generator.tensors, out_position=cls_pos, device=device)
    classifier.trace(torch.zeros(0, len(classifier.in_features), classifier.in_dim[0]).to(device))
    classifier.to(device)

    criterion = get_criterion(cfg.criterion)

    acc, avg_loss = mps_cat.eval(mps=classifier, loader=loaders["valid"],
                                 criterion=criterion, device=device)

    wandb.log({
        "gan_mps/valid/acc": acc,
        "gan_mps/valid/loss": avg_loss
    })

    if acc < trigger_accuracy:
        # retrain
        logger.info(f'Retraining after epoch {epoch+1}')
        best_state_dict, _, best_acc = mps_cat.train(
            mps=classifier,
            loaders=loaders,
            cfg=cfg,
            device=device,
            stage="gan"
        )
        generator.load_state_dict(best_state_dict)
        g_optimizer = get_optimizer(generator.parameters(), cfg.g_optimizer)

    return generator.state_dict(), g_optimizer


# TODO: Add safety saves
def loop(generator: tk.models.MPS,
         dis: Dict[Any, nn.Module],
         real_loaders: Dict[str, DataLoader],
         cfg: schemas.GANStyleConfig,
         cls_pos: int,
         embedding: str,

         # Retraining specifics
         best_acc: float,
         cat_loaders: Dict[str, DataLoader],

         device: torch.device
         ) -> None:
    """
    Main training loop for GAN-style training with MPS generator and discriminators.

    Alternates between discriminator updates and generator updates, and
    periodically evaluates/retrains the generator to maintain classification
    accuracy.

    Parameters
    ----------
    mps : tk.models.MPS
        Generator model (tensor network).
    dis : dict[Any, nn.Module]
        Dictionary of discriminators (per-class or single).
    real_loaders : dict[str, DataLoader]
        DataLoaders providing real training and validation data.
    cfg : schemas.GANStyleConfig
        Configuration object for GAN training (optimizers, loss functions, etc.).
    cls_pos : int
        Position of the class label within the MPS input dimension.
    best_acc : float
        Best validation accuracy observed before GAN training (used for retraining trigger).
    cat_loaders : dict[str, DataLoader]
        DataLoaders used for classification evaluation/retraining.
    device : torch.device
        Torch device for training.

    Returns
    -------
    d_losses : torch.FloatTensor, shape (N, C) or (N, 1)
        Discriminator losses per training step across epochs.
    g_losses : torch.FloatTensor, shape (N, C) or (N, 1)
        Generator losses per training step across epochs.
    valid_acc : list of float
        Validation accuracies at retraining checkpoints
        (1 element if no retraining occurred at a checkpoint, 2 if retraining occurred).
    valid_loss : list of float
        Validation losses at retraining checkpoints
        (same convention as `valid_acc`).
    """

    trigger_accuracy = min(best_acc-cfg.acc_drop_tol, 0.0)

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
    input_space = _embedding_to_range(embedding)
    input_space = torch.linspace(input_space[0], input_space[1], cfg.num_bins)
    input_space = input_space.to(device, dtype=torch.float32)
    embedding = get_embedding(embedding)
    in_dim, num_cls = _get_indim_and_ncls(generator)
    cls_embs = []
    for c in range(num_cls):
        cls_emb = tk.embeddings.basis(torch.tensor(c), num_cls)
        cls_emb = cls_emb.to(device=device, dtype=torch.float32)
        cls_embs.append(cls_emb)

    # Epoch loop
    for epoch in range(cfg.max_epoch):
        if (epoch+1) % cfg.info_freq == 0:
            logger.info(f"GAN training at epoch={epoch+1}")

        # Actual GAN-style training
        for X_real, c_real in real_loaders["train"]:
            X_real, c_real = X_real.to(device), c_real.to(device)
            _step(mps=generator, dis=dis, X_real=X_real, c_real=c_real,
                  r_synth=cfg.r_synth, cls_pos=cls_pos,
                  num_bins=cfg.num_bins, in_dim=in_dim,
                  num_cls=num_cls, embedding=embedding,
                  cls_embs=cls_embs, input_space=input_space,
                  d_optimizer=d_optimizer, g_optimizer=g_optimizer,
                  d_criterion=d_criterion, g_criterion=g_criterion,
                  watch_freq=cfg.watch_freq, device=device)

        # Checking classification accuracy and retraining if necessary
        if (epoch+1) % cfg.check_freq == 0:
            best_state_dict = check_and_retrain(generator=generator, loaders=cat_loaders,
                                    cfg=cfg.retrain_cfg, cls_pos=cls_pos,
                                    g_optimizer=g_optimizer, epoch=epoch, device=device,
                                    trigger_accuracy=trigger_accuracy)

            generator.load_state_dict(best_state_dict)
