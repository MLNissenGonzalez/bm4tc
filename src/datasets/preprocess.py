import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Sequence, Tuple, Dict
from src.mps.utils import _embedding_to_range

import logging
logger = logging.getLogger(__name__)

#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#----Preprocessing-----------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------

    
def preprocess_pipeline(X_raw: np.ndarray, 
                        t_raw: np.ndarray,
                        split: Sequence[float],
                        random_state: int,
                        embedding: str
                        )-> Tuple[Dict[str, torch.Tensor], 
                                  Dict[str, torch.Tensor], 
                                  MinMaxScaler]:
    
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
    tuple X, t, scaler
        X   dict
            split gives keys
        t   dict
        scaler  MinMaxScaler
            fitted to training data
    """
    # Initalize dictionaries
    X = {}
    t = {}

    # First split: separate test set
    X_remain, X["test"], t_remain, t["test"] = train_test_split(
        X_raw, t_raw, test_size=split[-1], random_state=random_state
    )
    
    # Second split: separate validation set from remaining data
    # ratio_new = ratio_old / ratio_remaining
    X["train"], X["valid"], t["train"], t["valid"] = train_test_split(
        X_remain, t_remain, test_size=split[1]/(1-split[-1]), random_state=random_state
    )
    
    # Initialize and fit the MinMaxScaler to training data only
    scaler = MinMaxScaler(_embedding_to_range(embedding=embedding), clip=True) # returns range^input_dim. Would need a custom scaler for feature by feature image range.
    X["train"] = scaler.fit_transform(X["train"])
    
    # Transform validation and test sets using the scaler fitted on training data
    X["valid"]= scaler.transform(X["valid"])
    X["test"] = scaler.transform(X["test"])

    # Convert to PyTorch tensors for subsequent training
    for split in ["train", "valid", "test"]:
        X[split]= torch.FloatTensor(X[split])
        t[split] = torch.LongTensor(t[split])

    logger.info("Data preprocessed.")
    return (X, t, scaler)