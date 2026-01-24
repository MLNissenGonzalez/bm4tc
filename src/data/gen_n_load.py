# Generating toy data and loading other datasets
# Not part of the later experiments, I think

# TODO: Add documentation (ADDED AS ISSUE)

import sklearn.datasets
import numpy as np
from dataclasses import dataclass
from src.utils.schemas import DatasetConfig, DataGenDowConfig
import os
import numpy.typing as npt

import logging
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ----Helper functions-----------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------


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
_NIST_DATA = []  # placeholders for MNIST, FashionMNIST, etc.
_TS_DATA = [] # placeholder for time-series data

_CANONICAL_FOLDERS = _TWO_DIM_DATA + _SK_DATA + _NIST_DATA + _TS_DATA

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

# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ----2D toy data-----------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# TODO: (Added as an issue, lowest priority) Include more 2D datasets, see e.g. https://github.com/AWehenkel/UMNN/blob/master/lib/toy_data.py


# maybe remove random state here
def _two_dim_generator(cfg: DataGenDowConfig) -> tuple[np.ndarray, np.ndarray]:
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
        X, t = sklearn.datasets.make_moons(
            n_samples=2*cfg.size, noise=cfg.noise, random_state=cfg.seed)
        return X, t

    elif canonical == "circles":
        X, t = sklearn.datasets.make_circles(n_samples=2*cfg.size, noise=cfg.noise, random_state=cfg.seed,
                                             factor=cfg.circ_factor)
        return X, t

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


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ----MNIST and other small image dataset---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# TODO: Add code to download and prepreprocess MNIST dataset (ADDED AS ISSUE, highpriority)


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ----TIME Series dataset dataset---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# TODO: Add code to download and prepreprocess Timeseries dataset 

# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ----Loading of datasets----------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# TODO: Add documentation (ADDED AS ISSUE)

def _generate_or_download(cfg: DataGenDowConfig, path: str) -> None:
    """
    Generate or download a dataset according to configuration and save it locally.

    This function currently supports 2D toy datasets generated via 
    `sklearn.datasets.make_*` utilities (e.g., `make_moons`, `make_circles`, 
    `make_blobs`). It ensures the target directory exists, generates the 
    dataset, and stores it as a compressed `.npz` file.

    Parameters
    ----------
    cfg : DataGenDowConfig
        Dataset generation or download configuration. Must contain:
        - `name`: str — the dataset name (e.g., "moons", "blobs").
          Names are normalized (spaces removed, lowercased) before parsing.
    path : str
        Target directory where the dataset file will be saved.
        Created automatically if it does not exist.

    Saves
    -----
    `{variant}.npz` in the given path, containing:
        - `X` : np.ndarray of shape (n_samples, n_features), dtype float32
            Feature matrix of the generated dataset.
        - `y` : np.ndarray of shape (n_samples,), dtype int
            Integer labels corresponding to each sample.

    Raises
    ------
    ValueError
        If the dataset name is not supported by `_TWO_DIM_DATA`.

    Notes
    -----
    Future extensions may include `_SK_DATA` (scikit-learn datasets) or 
    `_NIST_DATA` (MNIST variants).
    """

    os.makedirs(path, exist_ok=True)
    name = cfg.name.replace(" ", "").lower()
    canonical, variant = _parse_dataset_name(name)

    if canonical in _TWO_DIM_DATA:
        X, t = _two_dim_generator(cfg)
    else:
        # TODO: Extend this block to handle _SK_DATA or _NIST_DATA
        raise ValueError(f"Dataset {name} not supported.")

    np.savez(os.path.join(path, f"{variant}.npz"), X=X, y=t)


def load_dataset(cfg: DatasetConfig) -> LabelledDataset:
    """
    Load a labeled dataset from disk, generating it on demand if necessary.

    This function ensures that the dataset specified in the configuration exists
    in the local data directory. If not present, it triggers dataset generation
    via `_generate_or_download`. The loaded data are wrapped into a
    `LabelledDataset` dataclass for convenient access to features, labels, and
    dataset metadata.

    Parameters
    ----------
    cfg : DatasetConfig
        Dataset configuration object. Must include:
        - `gen_dow_kwargs.name`: str — dataset name (e.g., "moons", "circles").
        - `overwrite`: bool — if True, regenerate even if file exists.
        The dataset file is expected under `_DATA_DIR/<canonical>/<variant>.npz`.

    Returns
    -------
    LabelledDataset
        Dataclass containing:
        - `name` : str — dataset variant name.
        - `X` : np.ndarray of shape (n_samples, n_features), dtype float32
        - `t` : np.ndarray of shape (n_samples,), dtype int
        - `size` : int — number of samples (`X.shape[0]`)
        - `num_feat` : int — number of features per sample (`X.shape[1]`)
        - `num_cls` : int — number of distinct label classes.

    Side Effects
    ------------
    - Creates directories under `_DATA_DIR` if missing.
    - May trigger dataset generation and disk writes.
    - Logs dataset shapes at debug level.

    Raises
    ------
    ValueError
        If dataset generation fails or the dataset type is unsupported.
    """

    canonical, variant = _parse_dataset_name(cfg.gen_dow_kwargs.name)
    dataset_dir = os.path.join(_DATA_DIR, canonical)
    dataset_file = os.path.join(dataset_dir, f"{variant}.npz")

    # Check overwrite flag (default False for backward compatibility)
    overwrite = getattr(cfg, 'overwrite', False)

    if overwrite or not os.path.exists(dataset_file):
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
