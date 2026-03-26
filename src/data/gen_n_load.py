# Generating toy data and loading other datasets
# Not part of the later experiments, I think

# TODO: Add documentation (ADDED AS ISSUE)

import sklearn.datasets
import numpy as np
from dataclasses import dataclass
from src.utils.schemas import DatasetConfig, DataGenDowConfig
import os
import numpy.typing as npt
from pathlib import Path
from typing import Optional

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
    ucr_train_size: Optional[int] = None  # non-None only for UCR datasets


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
_NIST_DATA = ["mnist"]
_TS_DATA = ["ecg200", "italypowerdemand", "chlorineconcentration",
            "syntheticcontrol", "cricketx", "crickety", "cricketz"]

_UCR_TS_DATASET_DIRS = {          # canonical → folder name inside ucr_ts/
    "ecg200": "ECG200",
    "italypowerdemand": "ItalyPowerDemand",
    "chlorineconcentration": "ChlorineConcentration",
    "syntheticcontrol": "SyntheticControl",
    "cricketx": "CricketX",
    "crickety": "CricketY",
    "cricketz": "CricketZ",
}

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
        rng = np.random.RandomState(cfg.seed)
        theta = np.sqrt(rng.rand(cfg.size))*2*np.pi

        r_1 = 2*theta + np.pi
        data_1 = np.array([np.cos(theta)*r_1, np.sin(theta)*r_1]).T
        x_1 = data_1 + cfg.noise * rng.randn(cfg.size, 2)

        r_2 = -2*theta - np.pi
        data_2 = np.array([np.cos(theta)*r_2, np.sin(theta)*r_2]).T
        x_2 = data_2 + cfg.noise * rng.randn(cfg.size, 2)

        res_1 = np.append(x_1, np.zeros((cfg.size, 1)), axis=1)
        res_2 = np.append(x_2, np.ones((cfg.size, 1)), axis=1)

        res = np.append(res_1, res_2, axis=0)
        rng.shuffle(res)
        X, t = res[:, :-1], res[:, -1]
        return X, t


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ----MNIST and other small image dataset---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

def _nist_generator(cfg: DataGenDowConfig) -> tuple[np.ndarray, np.ndarray]:
    """Download and preprocess MNIST dataset.

    Combines train (60k) + test (10k) splits into one pool and optionally
    subsamples a balanced set of ``cfg.size`` samples per class.

    Parameters
    ----------
    cfg : DataGenDowConfig
        Must contain:
        - ``size``: int or None — samples per class; None means use all.
        - ``seed``: int — RNG seed for subsampling.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(X_np, y_np)`` where ``X_np`` has shape (N, 784) in float32 ∈ [0, 1]
        and ``y_np`` has shape (N,) as int.
    """
    import torchvision.datasets as tv_datasets

    # yann.lecun.com returns HTTP 200 with garbage content → MD5 fails → RuntimeError
    # (not URLError), so torchvision's mirror fallback is never triggered. Force the
    # working S3 mirror directly.
    tv_datasets.MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/"]

    # Raw files go to _DATA_DIR/MNIST/raw/ (torchvision convention)
    train_ds = tv_datasets.MNIST(root=_DATA_DIR, train=True, download=True)
    test_ds = tv_datasets.MNIST(root=_DATA_DIR, train=False, download=True)

    # Access tensors directly — avoids iterating the full dataset
    data = np.concatenate(
        [train_ds.data.numpy(), test_ds.data.numpy()], axis=0
    )  # (70000, 28, 28), uint8
    targets = np.concatenate(
        [train_ds.targets.numpy(), test_ds.targets.numpy()], axis=0
    )  # (70000,), int64

    # Flatten spatial dims and normalise to float32 in [0, 1]
    data = data.reshape(-1, 784).astype(np.float32) / 255.0  # (70000, 784)
    targets = targets.astype(int)

    if cfg.size is not None:
        rng = np.random.RandomState(cfg.seed)
        indices = []
        for c in range(10):
            class_idx = np.where(targets == c)[0]
            chosen = rng.choice(class_idx, size=cfg.size, replace=False)
            indices.append(chosen)
        indices = np.concatenate(indices)
        rng.shuffle(indices)
        data = data[indices]
        targets = targets[indices]

    logger.info(f"MNIST loaded: {data.shape[0]} samples, {data.shape[1]} features")
    return data, targets


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ----TIME Series dataset dataset---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# UCR 2018 archive: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip
# Password: "someone"


def _download_and_extract_ucr(url: str, password: str, folder_name: str, dest_dir: Path) -> None:
    """Download the full UCR 2018 zip and extract only the requested dataset folder."""
    import urllib.request
    import zipfile
    import tempfile
    import shutil

    logger.info(f"Downloading UCR archive from {url} (this may take a while)...")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        urllib.request.urlretrieve(url, tmp_path)
        logger.info(f"Download complete. Extracting '{folder_name}'...")
        prefix = f"UCRArchive_2018/{folder_name}/"
        with zipfile.ZipFile(tmp_path, "r") as zf:
            members = [m for m in zf.namelist() if m.startswith(prefix)]
            if not members:
                raise FileNotFoundError(
                    f"Could not find '{prefix}' inside the archive."
                )
            for member in members:
                zf.extract(member, path=tmp_path + "_extracted", pwd=password.encode())
        extracted_src = Path(tmp_path + "_extracted") / "UCRArchive_2018" / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for item in extracted_src.iterdir():
            shutil.move(str(item), str(dest_dir / item.name))
        shutil.rmtree(tmp_path + "_extracted", ignore_errors=True)
        logger.info(f"Extracted '{folder_name}' to {dest_dir}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _ucr_ts_loader(cfg: DataGenDowConfig):
    """Load a UCR time-series dataset from TSV files.

    Returns (X, t, ucr_train_size) where ucr_train_size is the number of
    original training samples (before merging with test).
    """
    canonical, _ = _parse_dataset_name(cfg.name)
    folder = _UCR_TS_DATASET_DIRS[canonical]
    ts_dir = Path(_DATA_DIR) / "ucr_ts" / folder

    if not ts_dir.exists():
        if cfg.dow_link:
            url = cfg.dow_link[0]
            password = getattr(cfg, "dow_password", None) or "someone"
            _download_and_extract_ucr(url, password, folder, ts_dir)
        else:
            raise FileNotFoundError(
                f"UCR dataset directory not found: {ts_dir}. "
                "Set dow_link in the dataset config to enable auto-download."
            )

    prefix = folder  # e.g. "ECG200"
    train_raw = np.loadtxt(ts_dir / f"{prefix}_TRAIN.tsv")   # (N_train, 1+T)
    test_raw  = np.loadtxt(ts_dir / f"{prefix}_TEST.tsv")    # (N_test,  1+T)

    ucr_train_size = len(train_raw)

    raw = np.vstack([train_raw, test_raw])
    labels_raw = raw[:, 0]
    X = raw[:, 1:].astype(np.float32)

    # Remap labels to 0-indexed integers
    unique = sorted(np.unique(labels_raw))
    label_map = {v: i for i, v in enumerate(unique)}
    t = np.array([label_map[v] for v in labels_raw], dtype=int)

    logger.info(
        f"UCR '{folder}' loaded: {len(X)} samples, {X.shape[1]} time steps, "
        f"UCR train size = {ucr_train_size}"
    )
    return X, t, ucr_train_size

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
        np.savez(os.path.join(path, f"{variant}.npz"), X=X, y=t)
    elif canonical in _NIST_DATA:
        X, t = _nist_generator(cfg)
        np.savez(os.path.join(path, f"{variant}.npz"), X=X, y=t)
    elif canonical in _TS_DATA:
        X, t, ucr_train_size = _ucr_ts_loader(cfg)
        np.savez(os.path.join(path, f"{variant}.npz"),
                 X=X, y=t, ucr_train_size=np.array(ucr_train_size))
    else:
        raise ValueError(f"Dataset {name} not supported.")


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
        logger.info(f"Generated dataset '{variant}' and saved to {dataset_file}")
    else:
        logger.info(f"Loaded dataset from {dataset_file}")
    data = np.load(dataset_file)

    X: npt.NDArray[np.float32] = data["X"]  # shape: (size, n_feat)
    t: npt.NDArray[np.int_] = data["y"]     # shape: (size,)
    ucr_train_size = int(data["ucr_train_size"]) if "ucr_train_size" in data else None
    logger.debug(f"{X.shape=}, {t.shape=}")
    return LabelledDataset(
        name=variant,
        X=X,
        t=t,
        size=X.shape[0],
        num_feat=X.shape[1],
        num_cls=len(np.unique(t)),
        ucr_train_size=ucr_train_size,
    )
