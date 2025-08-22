# Building the discriminator class out of a predefined MPS. 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Union, Sequence, Callable, Dict

#------------------------------------------------------------------------------------------------´
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#--------------------DISCRIMINIATOR INITIALIZATION-----------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------


# TODO: Add the discriminator class taking an MPS as input
#       and returning a pytorch module that is the MPS with an
#       MLP at the end to discriminate real from fake inputs to the MPS

class MLPdis(nn.Module):
    def __init__(self, 
                 hidden_dims: list,
                 nonlinearity=nn.ReLU,
                 input_dim=2):
        super().__init__()

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dims[0]))

        for i in range(len(hidden_dims)-1):
            layers.append(nonlinearity())
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

        layers.append(nonlinearity())
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.stack(x) # returns logit

# TODO: Classical discriminator initialisation based on predefined MPS or data.
# TODO: Optimizer initialisation
# TODO: Add discriminator ensemble, one for each class. 


#------------------------------------------------------------------------------------------------´
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#--------------------DISCRIMINIATOR PRETRAINING----------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------


# Data loader for the discrimination between real and synthesied examples
# The construction below could be used for a single discriminator for all classes
# or for a ensemble of discriminators, one for each class

def dis_pre_train_loader(   samples_train: torch.Tensor, 
                            samples_test: torch.Tensor, 
                            dataset_train: torch.Tensor,
                            dataset_test: torch.Tensor,
                            batch_size: int) -> tuple[DataLoader, DataLoader]:
    """
    DataLoader constructor for discriminator pretraining. Could be class dependent or not.

    Parameters
    ----------
    samples_train: tensor, shape=(num_samples_train, n_features)
        batch of unlabelled and by the MPS generated samples for training
    sampeles_test: tensor, shape=(num_samples_test, n_features)
    dataset_train: tensor, shape=(num_samples_train, n_features)
        batch of unlabelled natural samples for training
    dataset_test: tensor, shape=(num_samples_test, n_features)
    batch_size: int
        number of samples in the final loader (synthetic mixed with natural)

    Returns
    -------
    tuple[DataLoader, DataLoader]
        the train and test dataloader for pretraining of the discriminator
    """        
        
    X_train = torch.concat([dataset_train, samples_train])
    t_train = torch.concat([torch.ones(len(dataset_train)), 
                torch.zeros(len(samples_train), dtype=torch.long)])
    train_data = TensorDataset(X_train, t_train)
    loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    X_test = torch.concat([dataset_test, samples_test])
    t_test = torch.concat([torch.ones(len(dataset_test)), 
                torch.zeros(len(samples_test), dtype=torch.long)])
    test_data = TensorDataset(X_test, t_test)
    loader_test = DataLoader(test_data, batch_size=batch_size)
    
    return loader_train, loader_test 

# TODO: Allow for other losses
# TODO: Rename test loader to validation loader as it is used for hyperparameter optimization (early stopping)
# TODO: Why this i? Remove it if it is unnesessary to log. 

bce_loss = torch.nn.BCELoss() 
def discriminator_pretraining(dis,
                              max_epoch: int,
                              patience: int,
                              optimizer: torch.optim.Optimizer,
                              loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
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
    optimizer.zero_grad()
    for i in range(max_epoch):
        dis.train()
        for X, t in loader_train:
            logit = dis(X)
            prob = torch.sigmoid(logit.squeeze())
            loss = loss_fn(prob, t.float())
            train_loss.append(loss.item())

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        dis.eval()
        total_test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, t in loader_test:
                logit = dis(X)
                prob = torch.sigmoid(logit.squeeze())
                preds = (prob >= 0.5).long()
                loss = bce_loss(prob, t.float())
                total_test_loss += loss.item() * t.size(0)
                correct += (preds == t).sum().item()
                total += t.size(0)

            avg_test_loss = total_test_loss / total
            acc = correct / total
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
            if patience_counter > patience:
                break

    dis.load_state_dict(best_model_state)
    
    return dis, train_loss, test_loss, test_accuracy, i