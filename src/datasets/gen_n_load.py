# Generating toy data and loading other datasets
# Not part of the later experiments, I think

# TODO: Add documentation

import sklearn.datasets
import numpy as np
from dataclasses import dataclass
from schemas import DatasetConfig, DataGenDowConfig
import os
import numpy.typing as npt

import logging
logger = logging.getLogger(__name__)
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#----Helper functions-----------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------

@dataclass
class LabelledDataset:
    name: str
    X: np.ndarray
    t: np.ndarray
    size: int
    num_feat: int
    num_cls: int

# -----------------------------
# Dataset directories
# -----------------------------
_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".datasets")
)

# -----------------------------
# Supported dataset types
# -----------------------------
_TWO_DIM_DATA = ["moons", "circles", "spirals"]
_SK_DATA = []   # placeholders for sklearn datasets
_NIST_DATA = [] # placeholders for MNIST, FashionMNIST, etc.

_CANONICAL_FOLDERS = _TWO_DIM_DATA + _SK_DATA + _NIST_DATA

# -----------------------------
# Parse dataset name
# -----------------------------

def _parse_dataset_name(name: str):
    """
    Returns (canonical_folder, variant_name)
    e.g. "2moons_6k" -> ("moons", "2moons_6k")
    """
    key = name.replace(" ", "").lower()
    for folder in _CANONICAL_FOLDERS:
        if folder in key:
            return folder, key
    raise ValueError(f"Dataset {name} not recognised")

#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#----Generation of toy data-----------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------

# TODO: (Added as an issue) Include more 2D datasets, see e.g. https://github.com/AWehenkel/UMNN/blob/master/lib/toy_data.py

def _two_dim_generator(cfg: DataGenDowConfig) -> tuple[np.ndarray, np.ndarray]: # maybe remove random state here
    """Unprocessed generation of 2D toydata. Comes with labels.

    Parameters
    ----------
    cfg: DataGenDowConfig
        Configuration object containing all hyperparameters needed to generate 2D toydata.
        
        Fields:
        - name: str
            name of the dataset to be generated
        - size: int
            The number of samples to be generated per class
        - noise: float
            Strength of noise ontop of true data manifold.
        - seed: int | List[ints]
            Seed(s) used for data generation. 
        - circ_factor: Optional[float]
            Factor of radii of two circles.

    Returns
    -------
    tuple[np.ndarray, np.ndarray], shape: [(num_cls*size, n_feat), (num_cls,)]
    """
    logger.info("New 2D data generated")
    canonical, _ = _parse_dataset_name(cfg.name)

    if canonical == "moons":
        X, t = sklearn.datasets.make_moons(n_samples=2*cfg.size, noise=cfg.noise, random_state=cfg.seed)
        return X, t
    
    elif canonical == "circles":
        X,t = sklearn.datasets.make_circles(n_samples=2*cfg.size, noise=cfg.noise, random_state=cfg.seed, 
                                             factor=cfg.circ_factor)
        return X, t
    
    # TODO: Set seeds
    elif canonical == "spirals":
        theta = np.sqrt(np.random.rand(cfg.size))*2*np.pi 

        r_1 = 2*theta + np.pi
        data_1 = np.array([np.cos(theta)*r_1, np.sin(theta)*r_1]).T
        x_1 = data_1 + cfg.noise * np.random.randn(cfg.size, 2)

        r_2 = -2*theta - np.pi
        data_2 = np.array([np.cos(theta)*r_2, np.sin(theta)*r_2]).T
        x_2 = data_2 + cfg.noise * np.random.randn(cfg.size, 2)

        res_1 = np.append(x_1, np.zeros((cfg.size, 1)), axis=1)
        res_2 = np.append(x_2, np.ones((cfg.size, 1)), axis=1)

        res = np.append(res_1, res_2, axis=0)
        np.random.shuffle(res)
        X, t = res[:, :-1], res[:, -1]
        return X, t
    

#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#----Download of datasets---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#----Loading of datasets----------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------

# TODO: Add documentation

def _generate_or_download(cfg: DataGenDowConfig, path:str) -> None:
    os.makedirs(path, exist_ok=True)
    name = cfg.name.replace(" ", "").lower()
    canonical, variant = _parse_dataset_name(name)

    if canonical in _TWO_DIM_DATA:
        X, t = _two_dim_generator(cfg)
    else:
        # Extend this block to handle _SK_DATA or _NIST_DATA
        raise ValueError(f"Dataset {name} not supported.")
    
    np.savez(os.path.join(path, f"{variant}.npz"), X=X, y=t)

def load_dataset(cfg: DatasetConfig) -> LabelledDataset:
    canonical, variant = _parse_dataset_name(cfg.gen_dow_kwargs.name)    
    dataset_dir = os.path.join(_DATA_DIR, canonical)
    dataset_file = os.path.join(dataset_dir, f"{variant}.npz")

    if not os.path.exists(dataset_file):
        _generate_or_download(cfg=cfg.gen_dow_kwargs, path=dataset_dir)
    data = np.load(dataset_file)

    X: npt.NDArray[np.float32] = data["X"]  # shape: (size, n_feat)
    t: npt.NDArray[np.int_] = data["y"]     # shape: (size,)
    logger.debug(f"{X.shape=}, {t.shape=}")
    return LabelledDataset(
        name=variant,
        X=X,
        t=t,
        size=X.shape[0],
        num_feat=X.shape[1],
        num_cls=len(np.unique(t))
    )