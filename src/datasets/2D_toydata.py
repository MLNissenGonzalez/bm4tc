# Generation and Download of datasets
# Str based output of datasets
# Hyperparams enter here aswell.

import torch
import sklearn.datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Sequence

#-----------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#------Hyperparameter config here---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------

# TODO: (Added as an issue) Add Hyperparameter config here. No, src stays free of configuration code. 
# TODO: (Added as an issue) Load or initialise embedding here. Needed for preprocessing



#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#----Preprocessing-----------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------

# Preprocessing to [0,1]^2
# TODO: Think of ditching dictionary structure

def _embedding_to_range(embedding: str):
    """
    Given embedding identifier, return the associated domain of that embedding. 
    """
    if isinstance(embedding, str):
        if embedding == "fourier":
            return (0., 1.)
        elif embedding == "legendre":
            return (-1., 1.)
        else:
            raise ValueError(f"{embedding} not recognised")

    else:
        raise TypeError("Expected embedding name as string.")
    

def preprocess_pipeline(X: np.ndarray, 
                        t: np.ndarray,
                        split: Sequence[float],
                        random_state: int,
                        embedding: str):
    
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
    split: list of floats
        [train, valid, test], e.g. [0.8, 0.1, 0.1]
    random_state: int
        Seed

    Returns
    -------
    dict
        dictionary of preprocessed and split data. 
        Keys: {"X_split", "t_split"| split in {"train", "valid", "test"}}
    """
    # First split: separate test set
    X_remain, X_test, t_remain, t_test = train_test_split(
        X, t, test_size=split[-1], random_state=random_state
    )
    
    # Second split: separate validation set from remaining data
    # ratio_new = ratio_old / ratio_remaining
    X_train, X_valid, t_train, t_valid = train_test_split(
        X_remain, t_remain, test_size=split[1]/(1-split[-1]), random_state=random_state
    )
    
    # Initialize and fit the MinMaxScaler on training data only
    scaler = MinMaxScaler(_embedding_to_range(embedding=embedding), clip=True) # returns range^input_dim. Would need a custom scaler for feature by feature image range.
    X_train = scaler.fit_transform(X_train)
    
    # Transform validation and test sets using the scaler fitted on training data
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

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