# Generation and Download of datasets
# Str based output of datasets
# Hyperparams enter here aswell.

import torch
import tensorkrowch as tk
import sklearn.datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from typing import Union, Sequence, Callable, Dict
from torch.utils.data import TensorDataset, DataLoader

# TODO: Wrap this in one function, call it something like "raw_data_generation"
# TODO: Include more datasets, see e.g. https://github.com/AWehenkel/UMNN/blob/master/lib/toy_data.py

# this i hate, make it different later, less cross dependent
batch_size = 32
num_batches = 25  # tune this for more samples. 
num_samples = batch_size * num_batches  # total number of samples per class and dataset, rn 1

# Parameters for split into train, val, test
test_size = 0.2
valid_size = 0.25

# Total real samples
N = int(np.ceil(num_samples / ((1-test_size) * (1-valid_size)))) #for initial training of discriminator using only training split
B = 2 * N #two classes
print(N, 32*25)

# Using sklearn for the first two datasets
X_moons, t_moons = sklearn.datasets.make_moons(n_samples=B, noise=0.1, random_state=42)
X_circ, t_circ = sklearn.datasets.make_circles(n_samples=B, noise=0.1, random_state=42, factor=0.4)


# Handcrafting the 2 spiral set
theta = np.sqrt(np.random.rand(N))*2*np.pi 

r_a = 2*theta + np.pi
data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
x_a = data_a + np.random.randn(N, 2)

r_b = -2*theta - np.pi
data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
x_b = data_b + np.random.randn(N, 2)

res_a = np.append(x_a, np.zeros((N, 1)), axis=1)
res_b = np.append(x_b, np.ones((N, 1)), axis=1)

res = np.append(res_a, res_b, axis=0)
np.random.shuffle(res)
X_spiral, t_spiral = res[:, :-1], res[:, -1]


# Preprocessing to [0,1]^2
# TODO: Add documentation
# TODO: Think of ditching dictionary structure
def preprocess_pipeline(X, t, test_size=test_size, valid_size=valid_size, random_state=42):
    # First split: separate test set
    X_temp, X_test, t_temp, t_test = train_test_split(
        X, t, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation set from remaining data
    # valid_size is relative to the size of X_temp
    X_train, X_valid, t_train, t_valid = train_test_split(
        X_temp, t_temp, test_size=valid_size, random_state=random_state
    )
    
    # Initialize and fit the MinMaxScaler on training data only
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform validation and test sets using the scaler fitted on training data
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    # Clip all scaled data to [0,1]
    X_train = np.clip(X_train_scaled, 0, 1)
    X_valid = np.clip(X_valid_scaled, 0, 1)
    X_test = np.clip(X_test_scaled, 0, 1)

    # Convert to PyTorch tensors for subsequent training
    X_train = torch.FloatTensor(X_train)
    X_valid = torch.FloatTensor(X_valid)
    X_test = torch.FloatTensor(X_test)
    t_train = torch.LongTensor(t_train)
    t_valid = torch.LongTensor(t_valid)
    t_test = torch.LongTensor(t_test)
    
    return (
        {
            'X_train': X_train,
            't_train': t_train,
            'X_valid': X_valid,
            't_valid': t_valid,
            'X_test': X_test,
            't_test': t_test,
        },
        scaler
        )

batch_size = 32
phys_dim = 6

# TODO: Here is an input embedding. Think of initialising it here or in `mps_utils.py`
# TODO: Redo documentation
def mps_cat_loader( X: torch.Tensor, 
                    t: torch.Tensor, 
                    embedding: Callable[[torch.Tensor, int], torch.Tensor], 
                    batch_size: int, 
                    phys_dim: int,
                    split: str) -> DataLoader:
    """
    Create DataLoaders for multiple datasets and splits

    Parameters:
        datasets: dict of dict containing X and t tensors for each split
        batch_size: int, size of batches
        embedding: optional function to transform the input data,
                   important for MPS.
                   Embedding needs to have domain D with [0,1]^2 subset of D
        phys_dim: int, dimension of embedding space

    Returns:
        loader: dataloader for categorisation using mps
    """
    X = embedding(X, phys_dim)
    dataset = TensorDataset(X, t)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split in ['train', 'valid']),
        drop_last=(split == 'train')
    )
    return loader

# TODO: Add documentation
def real_loader(X: torch.Tensor,
                batch_size: int,
                split: str):
    dataset = TensorDataset(X)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split in ['train', 'valid']),
        drop_last=(split == 'train')
    )
    return loader

# Data loader for the discrimination between real and synthesied examples

# TODO: Add documentation
def discrimination_dataloader(samples_train:torch.Tensor, 
                              samples_test: torch.Tensor, 
                              dataset_train: torch.Tensor,
                              dataset_test: torch.Tensor,
                              batch_size: int):        
        
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