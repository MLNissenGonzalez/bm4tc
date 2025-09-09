import torch
import tensorkrowch as tk
from typing import Union, Sequence, Callable, Dict
from mps.utils import mps_sampling, mps_acc_eval, disr_train_mps
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from schemas import OptimizerConfig, CriterionConfig, PretrainMPSConfig
from _utils import get_criterion, get_optimizer

import logging
logger = logging.getLogger(__name__)

"""
Currently, this script contains only three functions. A function performing the adversarial training (GAN-style) step,
and another performing the loop with a check on the classification accuracy of the MPS. 
"""

def real_loader(X: torch.Tensor,
                n_real: int,
                split: str) -> DataLoader:
    """
    Loader for real, but unlabelled data. Used in ad. train.
    
    Parameters
    ----------
    X: tensor
        whole batch of preprocessed, non-embedded data features
    n_real: int
    split: str in {'train', 'valid'}
        adtrain needs only train for training and valid for evaluation

    Returns
    -------
    DataLoader
    """
    dataset = TensorDataset(X)
    loader = DataLoader(dataset,
                        batch_size=n_real,
                        shuffle=(split in ['train', 'valid']),
                        drop_last=(split == 'train')
                        )
    return loader

# TODO: Think about how to implement configuration. .yml lend themselves well to strings (e.g. 'BCE'), 
#       but arguments to functions in my scripts are currently often functions (e.g. nn.BCEloss).


# TODO: Add other checks for gradients
def _adversarial_training_step(dis, # TODO: define abstract discriminator class
                                mps: tk.models.MPS,
                                n_synth: int,
                                real_data_loader: DataLoader, # train version
                                d_optimizer: torch.optim.Optimizer,
                                g_optimizer: torch.optim.Optimizer,
                                input_space: torch.Tensor | Sequence[torch.Tensor],
                                embedding: Union[Callable[[torch.Tensor, int], torch.Tensor], Sequence[Callable[[torch.Tensor, int], torch.Tensor]]],
                                cls_pos: int,
                                cls_embs: Sequence[torch.Tensor],
                                check: bool,
                                device: torch.device,
                                d_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
                                g_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
                                ):
    """
    Standard step in adversarial training with half the samples real, the other samples being synthesised by the generator.
    
    Parameters
    ----------
    dis: nn.Module
        pretrained discriminator distinguishing real from synthesised samples
    mps: tk.MPS
        pretrained (via classification) Matrix Product State-type generator
    batch_size: int
        (number of synthesised samples per step) = batch_size x num_cls
    real_data_loader: DataLoader
        loads the minibatch of real examples
    d_optimizer: Optimizer
        optimizer for the discriminator
    g_optimizer: Optimizer
        optimizer for the generator
    input_space: tensor
        interval I, expecting observed data space to be of type I^d
    embedding: tk.embeddings
        embedding of every single input feature to a larger space with dimension phys_dim
    cls_pos: int
        position of central tensor (assumes single MPS)
    check: Boolean
        If true, save metrics on gradient flow to MPS
    device: torch.device
    d_loss_fn: Callable
        loss function for discriminator, usually BCE loss
    g_loss_fn: Callable
        loss function for generator

    Returns
    -------
    d_loss: float
        step-wise classification loss of discriminator
    g_loss: float
        step-wise loss of generator (MPS)
    gradient_flow_metric: float, optional
        metric for gradient flow to MPS
    """
    
    # 1. Get a batch of real data
    real_X = next(iter(real_data_loader))[0]
    real_t = torch.ones(real_X.size(0), dtype=torch.float32)
    

    # 2. Generate a batch of synthetic data
    synth_X = mps_sampling( mps=mps,
                            input_space=input_space,
                            embedding=embedding,
                            num_samples=n_synth, # returns num_cls x batch_size
                            cls_pos=cls_pos,
                            cls_embs=cls_embs,
                            device=device
                            )
    
    synth_t = torch.zeros(synth_X.size(0), dtype=torch.float32)
    
    # 3. Concatenate and shuffle for discriminator update
    # detach very important!! Not just efficiency but also in order not to reuse computational graph twice
    X = torch.cat([real_X, synth_X.detach()], dim=0) 
    t = torch.cat([real_t, synth_t], dim=0)
    perm = torch.randperm(X.size(0))
    X, t = X[perm], t[perm]

    # --- Discriminator update ---
    d_optimizer.zero_grad()
    d_logit = dis(X)
    d_prob = torch.sigmoid(d_logit.squeeze())
    d_loss = d_loss_fn(d_prob, t)
    d_loss.backward()
    d_optimizer.step()

    # --- Generator update ---
    g_optimizer.zero_grad()
    # Use only synthetic samples, want discriminator to predict "real" (label=1)
    g_logit = dis(synth_X)
    g_prob = torch.sigmoid(g_logit.squeeze())
    g_loss = g_loss_fn(g_prob, torch.ones_like(synth_t))
    g_loss.backward()
    g_optimizer.step()

    # Saving one tensor for later check
    if check:
        l_tensor_comp = mps.tensors[0][0, 0].detach().numpy() # save only one tensor for later
        return d_loss.item(), g_loss.item(), l_tensor_comp
    
    else:
        return d_loss.item(), g_loss.item()
    

# TODO: Return loss on validation set as a more sensitive measure of change
# TODO: Use different metric for gradient flow to MPS parameters
# TODO: Reorder and maybe rename function parameters for better readability
# TODO: Reinitialize cat optimizer with each retraining? Bring up to speed with dis.train changes.
#       same with generator optimizer. should these be reinitialised with each retraining? 
 
def ad_train_loop(  dis: nn.Module, # TODO: define abstract discriminator class,
                    mps: tk.models.MPS,
                    real_data_loader: DataLoader,
                    n_synth: int,
                    d_optimizer: OptimizerConfig,
                    d_criterion: CriterionConfig,
                    g_optimizer: OptimizerConfig,
                    g_criterion: CriterionConfig,
                    ad_max_epoch: int,
                    
                    input_space: torch.Tensor | Sequence[torch.Tensor],
                    embedding: str, # only one embedding for all input sites, given by cfg.model.mps.embedding, for sampling
                    cls_pos: int, # In the outside script
                    cls_embs: Sequence[torch.Tensor], # one embedding per class. Just use basis embedding, right?

                    acc_drop_tol: float,
                    recompute_best_acc: bool, # allowed drop in accuracy, might be zero
                    num_acc_check: int, # number of accuracy checks per epoch (going through the original dataste)
                    
                    device: torch.device,
                    
                    mps_cat_loader: Dict[str, DataLoader],
                    retrain_cfg: PretrainMPSConfig,
                    title: str) -> tuple[list[float], list[float], list[float], list[int], list[float]]:
    """
    Full adversarial training loop with retraining of MPS if classification accuracy drops (too low).

    Parameters
    ----------
    dis: MLP model
        pretrained discriminator distinguishing real from synthesised samples
    mps: MPS model
        pretrained (as a classifier) MPS that acts as generator in the adversarial training framework
    batch_size: int
        (batch_size x num classes) = num of samples MPS generates during each ad. train. step. 
    real_data_loader: DataLoader
        loads real samples. its minibatch size is equal to (batch_size x num classes) in default config
    d_optimizer: Optimizer
        optimizer for discriminator in ad. train.
    g_optimizer: Optimizer
        optimizer for MPS (generator) in ad. train.
    cat_optimizer: Optimizer
        optimizer for MPS during classification retraining
    input_space: tensor
        preprocessed inputspace
    embedding: Callable
    cls_pos: int
        position of central tensor of MPS. Currently assumes central tensor setup of MPS
    cls_embs: list of tensors
        class embeddings, one-hot vectors
    acc_drop_tol: float
        tolerated drop in prediction accuracy on the validation set of the MPS as classifier
    num_acc_check: int
        number of epochs without checking MPS classification accuracy
    cat_patience: int
        patience parameter for early stopping in retraining of MPS as classifier
    d_loss_fn: Callable
        loss function for discriminator in ad. train.
    g_loss_fn: Callable
        loss function for MPS as generator in ad. train.
    cat_loss_fn: Callable
        loss function for MPS as classifier in retraining
    title: str, optional
        title of dataset being used
    recompute_best_acc: Boolean
        If true, best accuracy to measure drop against is the one after retraining, not the one before ad. train.

    Returns
    -------
    d_losses: list of floats
        batch-wise loss of discriminator during ad train
    g_losses: list of floats
        batch-wise loss of MPS as generator during ad train
    cat_acc: epoch-wise classification accuracy of MPS
    retrain_epochs: list of ints
        epochs where retraining of MPS as classifier was performed
    gradient_flow_metric: list of floats
    """
    d_losses, g_losses, cat_acc, retrain_epochs, l_t_comps = [], [], [], [], []

    mps.unset_data_nodes()
    mps.reset()
    
    # Rethink this.  
    mps.out_features = [cls_pos]
    phys_dim = mps.phys_dim[cls_pos-1] # assume cls_pos not equal to 0

    # Computing best accuracy
    mps.trace(torch.zeros(0, len(mps.in_features), phys_dim))
    mps.to(device)
    best_acc = mps_acc_eval(mps, mps_cat_loader, 'valid', device)
    print(f"Initial classification accuracy: {best_acc:.4f}")

    # Initializing optimizer and loss_functions
    d_criterion = get_criterion(d_criterion)
    d_optimizer = get_optimizer(params=dis.parameters(), config=d_optimizer)
    g_criterion = get_criterion(g_criterion)
    g_optimizer = get_optimizer(params=mps.parameters(), config=g_optimizer)


    check_counter = 0
    for epoch in range(ad_max_epoch):
        print(f'Starting with epoch {epoch+1}')
        for _ in range(len(real_data_loader)):
            # TODO: check l_t_comp shape
            d_loss, g_loss, *l_t_comp = _adversarial_training_step(
                dis=dis,
                mps=mps,
                n_synth=n_synth, # interacts with batch_size of real_data_loader
                real_data_loader=real_data_loader,
                d_optimizer=d_optimizer,
                g_optimizer=g_optimizer,
                input_space=input_space,
                embedding=embedding,
                cls_pos=cls_pos, # these I could compute once within this ad_train_loop
                cls_embs=cls_embs, # these I could compute once within this ad_train_loop
                check=True,
                d_loss_fn=d_criterion,
                g_loss_fn=g_criterion,
                device=device
            )

            d_losses.append(d_loss)
            g_losses.append(g_loss)
            l_t_comps.append(l_t_comp)

        if check_counter == num_acc_check:

            mps.unset_data_nodes()
            mps.reset()
            mps.out_features = [cls_pos]
            mps.trace(torch.zeros(0, len(mps.in_features), phys_dim))
            mps.to(device)
            
            acc = mps_acc_eval(mps, mps_cat_loader, 'valid', device)
            cat_acc.append(acc)

            if acc < (best_acc - acc_drop_tol):
                # retrain
                print(f'Retraining after epoch {epoch+1}')
                retrain_epochs.append(epoch)
                # not interested in training dynamics
                (best_tensors, _, 
                 val_accuracy) = disr_train_mps(mps=mps,
                                                loaders=mps_cat_loader,
                                                cfg=retrain_cfg,
                                                cls_pos=cls_pos,
                                                phys_dim=phys_dim,
                                                device=device,
                                                title=title)
                
                mps = tk.models.MPS(tensors=best_tensors)
                cat_acc.append(acc)

                # Recompute best accuracy?
                if recompute_best_acc:
                    best_acc = max(val_accuracy)
                check_counter = 0
                
            else:
                # check_counter not resetted to retrain whenever the first drop occurs.
                continue

        else: 
            check_counter += 1

    return d_losses, g_losses, cat_acc, retrain_epochs, l_t_comps