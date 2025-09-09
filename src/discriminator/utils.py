# Building the discriminator class out of a predefined MPS. 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Union, Sequence, Callable, Dict
from schemas import DisConfig, PretrainDisConfig
from _utils import get_criterion, get_optimizer

import logging
logger = logging.getLogger(__name__)
#------------------------------------------------------------------------------------------------´
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#--------------------DISCRIMINIATOR INITIALIZATION-----------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# TODO: Add ensemble method
class MLPdis(nn.Module):
    """
    Discriminator for GAN training. This architecture is just a fully connected feed forward network with single logit output for BCE loss. 
    One could use this class in a single discriminator setup (one for all classes of samples) or for the ensemble approach. 
    """
    def __init__(self, 
                 cfg: DisConfig):
        super().__init__()

        nonlinearity = cfg.nonlinearity.replace(" ", "").lower()

        if nonlinearity == "relu":
            def get_activation(): return nn.ReLU()
        elif nonlinearity == "leakyrelu":
            if cfg.negative_slope == None:
                raise ValueError("LeakyReLU needs negative_slope parameter.")
            def get_activation(): return nn.LeakyReLU(cfg.negative_slope)
        else:
            raise ValueError(f"{nonlinearity} not recognised. Try ReLU or LeakyReLU instead.")
        
        if not cfg.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size.")

        if cfg.mode == "single":
            layers = []
            layers.append(nn.Linear(cfg.input_dim, cfg.hidden_dims[0]))
            for i in range(len(cfg.hidden_dims)-1):
                layers.append(get_activation())
                layers.append(nn.Linear(cfg.hidden_dims[i], cfg.hidden_dims[i+1]))
            layers.append(get_activation())
            layers.append(nn.Linear(cfg.hidden_dims[-1], 1)) # always binary classification

            self.stack = nn.Sequential(*layers)
        elif cfg.mode == "ensemble":
            raise KeyError(f"ensemble mode not implemented yet")
        else:
            raise KeyError(f"{cfg.mode} not recognised.")

    def forward(self, x):
        return self.stack(x) # returns logit

# TODO: Add the discriminator class taking an MPS as input
#       and returning a pytorch module that is the MPS with an
#       MLP at the end to discriminate real from fake inputs to the MPS

# TODO: Classical discriminator initialisation based on predefined MPS or data.
# TODO: Optimizer initialisation

#------------------------------------------------------------------------------------------------´
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#--------------------DISCRIMINIATOR PRETRAINING----------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# (Class-wise) dataset sizes should be controlled/logged (e.g. via config)
def _class_wise_dataset_size(t: torch.LongTensor):
    _, cwds = t.unique(return_counts=True)
    return cwds


def dis_pretrain_dataset(X_real: torch.FloatTensor, 
                         c_real: torch.LongTensor, 
                         X_synth: torch.FloatTensor, 
                         mode="single"):
    num_spc, num_cls, data_dim = X_synth.shape

    if mode == "single":
        X_synth = X_synth.reshape(num_spc * num_cls, data_dim)
        t_synth = torch.zeros(len(X_synth), dtype=torch.long)  # fake
        t_real = torch.ones(len(X_real), dtype=torch.long)      # real
        
        X = torch.cat([X_real, X_synth], dim=0)
        t = torch.cat([t_real, t_synth], dim=0)
        return TensorDataset(X, t)

    elif mode == "ensemble":
        datasets = {}
        for c in range(num_cls):
            X_real_c = X_real[c_real == c]                # select class c
            X_synth_c = X_synth[:, c, :]                  # synthetic for class c
            t_real = torch.ones(len(X_real_c), dtype=torch.long)
            t_synth = torch.zeros(len(X_synth_c), dtype=torch.long)
            
            X = torch.cat([X_real_c, X_synth_c], dim=0)
            t = torch.cat([t_real, t_synth], dim=0)
            datasets[c] = TensorDataset(X, t)
        return datasets

    else:
        raise KeyError(f"{mode} has to be either single or ensemble.")

def dis_pretrain_loader(X_real: torch.FloatTensor, 
                        c_real: torch.LongTensor, 
                        X_synth: torch.FloatTensor, 
                        mode: str,
                        batch_size: int,
                        split: str):
    if split not in ["train", "valid", "test"]:
        raise KeyError(f"{split} not recognised.")
    
    dataset = dis_pretrain_dataset(X_real, c_real, X_synth, mode=mode)
    if mode == "single":
        return {"all": DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), drop_last=(split=="train"))}
    elif mode == "ensemble":
        loaders = {c: DataLoader(ds, batch_size=batch_size, shuffle=(split=="train"), drop_last=(split=="train")) 
                   for c, ds in dataset.items()}
        return loaders


def dis_eval(dis: nn.Module, loader: DataLoader, criterion: nn.Module):
    dis.eval()
    total_test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, t in loader:
            logit = dis(X)
            prob = torch.sigmoid(logit.squeeze())
            preds = (prob >= 0.5).long()
            loss = criterion(prob, t.float())
            total_test_loss += loss.item() * t.size(0)
            correct += (preds == t).sum().item()
            total += t.size(0)

        avg_test_loss = total_test_loss / total
        acc = correct / total
    return acc, avg_test_loss

# TODO: Rename test loader to validation loader as it is used for hyperparameter optimization (early stopping)
# TODO: Why this i? Remove it if it is unnesessary to log. 


def discriminator_pretraining(dis: nn.Module,
                              cfg: PretrainDisConfig,
                              loader_train: DataLoader,
                              loader_test: DataLoader):
    """
    Pretraining of discriminator in binary classification setting. Synthesised samples are precomputed.

    Parameters
    ----------
    dis: MLP model
        discriminator
    max_epoch: int
    patience: int
        patience method for early stopping
    optimizer: Optimizer
    loss_fn: Callable
        loss function for classification problem. Default config: BCE loss
    loader_train: DataLoader
    loader_test: DataLoader

    Returns
    -------
    dis: MLP model 
        best performing discriminator
    train_loss: list of floats
        minibatch-wise train loss
    test_loss: list of floats
        epoch-wise validation loss
    test_accuracy: list of floats
        epoch-wise classification accuracy of discriminator
    i: int
        last epoch with training
    """
        
    train_loss = []
    test_loss = []
    test_accuracy = []
    patience_counter = 0
    best_loss = float('inf')

    optimizer = get_optimizer(dis.parameters(), cfg.optimizer)
    criterion = get_criterion(cfg.criterion)

    optimizer.zero_grad()
    for epoch in range(cfg.max_epoch):
        dis.train()
        for X, t in loader_train:
            logit = dis(X)
            prob = torch.sigmoid(logit.squeeze())
            loss = criterion(prob, t.float())
            train_loss.append(loss.item())

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        acc, avg_test_loss = dis_eval(dis, loader_test, criterion)
        test_loss.append(avg_test_loss)
        test_accuracy.append(acc)

        # Progress tracking and best model update
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience_counter = 0
            best_model_state = dis.state_dict()
        else:
            patience_counter += 1
        # Early stopping
        if patience_counter > cfg.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    dis.load_state_dict(best_model_state)
    
    return dis, train_loss, test_loss, test_accuracy, epoch