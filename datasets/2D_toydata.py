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

#-----------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#------Hyperparameter config here---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------

# TODO: (Added as an issue) Add Hyperparameter config here.
# TODO: (Added as an issue) Load or initialise embedding here. Needed for preprocessing and dataloader


batch_size = 32
phys_dim = 6
num_batches = 25  # tune this for more samples. 
num_samples = batch_size * num_batches  # total number of samples per class and dataset, rn 1

# Parameters for split into train, val, test
test_size = 0.2
valid_size = 0.25

# Total real samples
N = int(np.ceil(num_samples / ((1-test_size) * (1-valid_size)))) #for initial training of discriminator using only training split
print(N, 32*25)


#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#----Preprocessing-----------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------


# Preprocessing to [0,1]^2
# TODO: Think of ditching dictionary structure
# TODO: (Added as an issue) Make preprocessing embedding dependent
def preprocess_pipeline(X: np.ndarray, 
                        t: np.ndarray,
                        test_size: float,
                        valid_size: float, 
                        random_state: int):
    
    """
    Preprocess 2D data wit MinMaxScaler to fit into [0,1]^2. 
    / (or [-1, 1]^2 for other embeddings).
    Also split data into train, validation, and test.

    Parameters
    ----------
    X: array, shape: (batch_size, n_features)
        The array of raw features
    t: array, shape: (batch_size,)
        Array of class labels
    test_size: float
        Proportion of test samples relative to total number of raw samples
    valid_size: float
        (1-test_size) * valid_size is the proportion of valid samples
    random_state: int
        Seed

    Returns
    -------
    dict
        dictionary of preprocessed and split data. 
        Keys: {"X_split", "t_split"| split in {"train", "valid", "test"}}
    """
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

#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#-----Raw data generation----------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------


# TODO: (Added as an issue) Include more 2D datasets, see e.g. https://github.com/AWehenkel/UMNN/blob/master/lib/toy_data.py
def raw_data_gen(title: str, size: int, noise: float, random_state: int,
                 factor: float = 0.4) -> tuple[np.ndarray, np.ndarray]: # maybe remove random state here
    """Unprocessed generation of 2D toydata. Comes with labels.

    Parameters
    ----------
    title: str
        The title of the dataset to select which dataset to generate.
    size: int
        The number of samples generated per class.
    noise: float
        Quantifies the strength of perturbations around the true data manifold.
    random_state: int
        Seed
    factor: float
        Quantifies the ratio of radii in the 2 circles dataset. 

    Returns
    -------
    tuple[np.ndarray, np.ndarray], shape: [(num_cls*size, n_feat), (num_cls,)]
    """
    if title == "2 moons":
        X, t = sklearn.datasets.make_moons(n_samples=2*size, noise=noise, random_state=random_state)
        return X, t
    
    elif title == "2 circles":
        X,t = sklearn.datasets.make_circles(n_samples=2*size, noise=noise, random_state=random_state, 
                                             factor=factor)
        return X, t
    
    elif title == "2 spirals":
        theta = np.sqrt(np.random.rand(size))*2*np.pi 

        r_1 = 2*theta + np.pi
        data_1 = np.array([np.cos(theta)*r_1, np.sin(theta)*r_1]).T
        x_1 = data_1 + noise * np.random.randn(size, 2)

        r_2 = -2*theta - np.pi
        data_2 = np.array([np.cos(theta)*r_2, np.sin(theta)*r_2]).T
        x_2 = data_2 + noise * np.random.randn(size, 2)

        res_1 = np.append(x_1, np.zeros((size, 1)), axis=1)
        res_2 = np.append(x_2, np.ones((size, 1)), axis=1)

        res = np.append(res_1, res_2, axis=0)
        np.random.shuffle(res)
        X, t = res[:, :-1], res[:, -1]
        return X, t
    
    else:
        print(f"title string = {title} does not match. Setting title='2 moons'.")
        return raw_data_gen("2 moons", size, noise, random_state)

#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------DataLoader------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------


def mps_cat_loader( X: torch.Tensor, 
                    t: torch.Tensor, 
                    embedding: Callable[[torch.Tensor, int], torch.Tensor], 
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
            The number of examples of features
        embedding: tk.embeddings.embedding
            Embedding for features X.
        phys_dim: int, 
            Dimension of embedding space

    Returns
    -------
        DataLoader
            Dataloader for supervised classification.
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

def real_loader(X: torch.Tensor,
                batch_size: int,
                split: str) -> DataLoader:
    """
    Loader for real, but unlabelled data. Used in ad. train.
    
    Parameters
    ----------
    X: tensor
        whole batch of preprocessed, non-embedded data features
    batch_size: int
    split: str in {'train', 'valid'}
        adtrain needs only train for training and valid for evaluation

    Returns
    -------
    DataLoader
    """
    dataset = TensorDataset(X)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=(split in ['train', 'valid']),
                        drop_last=(split == 'train')
                        )
    return loader

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