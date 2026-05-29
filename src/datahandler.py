import sklearn.datasets
import numpy as np
from dataclasses import dataclass, field
import os
import numpy.typing as npt
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging

import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from math import ceil

from src.model import ConditionalBornMachine


@dataclass
class DataGenDowConfig:
    name: str = "circles"
    size: Optional[int] = None
    seed: Optional[int] = None
    noise: Optional[float] = None
    circ_factor: Optional[float] = None
    dow_link: Optional[List[str]] = None
    dow_password: Optional[str] = None
    resize: Optional[int] = None  # target side length for MNIST resize (None = 28×28, i.e. 784 features)


@dataclass
class DatasetConfig:
    name: str = "circles"
    gen_dow_kwargs: DataGenDowConfig = field(default_factory=DataGenDowConfig)
    split: Tuple[float, float, float] = field(default_factory=lambda: (0.7, 0.15, 0.15))
    split_seed: int = 42
    overwrite: bool = False
    use_ucr_split: bool = False
    scaler: str = "minmax"

logger = logging.getLogger(__name__)


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
    os.path.join(os.path.dirname(__file__), "..", ".datasets")
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
    if cfg.resize is not None:
        import torch.nn.functional as F
        n = cfg.resize
        data_t = torch.from_numpy(data).float().unsqueeze(1) / 255.0  # (N, 1, 28, 28)
        data_t = F.interpolate(data_t, size=(n, n), mode='bilinear', align_corners=False)
        data = data_t.squeeze(1).numpy().reshape(-1, n * n).astype(np.float32)  # (N, n²)
    else:
        data = data.reshape(-1, 784).astype(np.float32) / 255.0  # (N, 784)
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


def _npz_stem(cfg: DataGenDowConfig) -> str:
    """Return the npz filename stem for a dataset config (without .npz extension)."""
    _, variant = _parse_dataset_name(cfg.name)
    if cfg.resize is not None:
        return f"{variant}_r{cfg.resize}"
    return variant


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
        np.savez(os.path.join(path, f"{_npz_stem(cfg)}.npz"), X=X, y=t)
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
    stem = _npz_stem(cfg.gen_dow_kwargs)
    dataset_file = os.path.join(dataset_dir, f"{stem}.npz")

    overwrite = cfg.overwrite

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
        name=stem,
        X=X,
        t=t,
        size=X.shape[0],
        num_feat=X.shape[1],
        num_cls=len(np.unique(t)),
        ucr_train_size=ucr_train_size,
    )


class LinearScaler:
    """
    Global linear scaler: fits a single linear transform to map the
    global min/max of the training data to ``feature_range``.

    Unlike MinMaxScaler, which normalizes each feature independently,
    LinearScaler applies one shared transform across all features.
    This preserves relative feature magnitudes and avoids amplifying
    near-constant background pixels (e.g. MNIST border pixels).

    Reduces to a passthrough when the data already lies in
    ``feature_range`` (e.g. MNIST ∈ [0,1] with SimpEmbedding or
    FourierEmbedding, both targeting [0,1]).
    """

    def __init__(self, feature_range=(0., 1.), clip=False):
        self.feature_range = feature_range
        self.clip = clip
        self._scale: float = 1.0
        self._offset: float = 0.0

    def fit(self, X):
        x_min = float(X.min())
        x_max = float(X.max())
        t_min, t_max = self.feature_range
        if x_max == x_min:
            self._scale = 1.0
            self._offset = t_min
        else:
            self._scale = (t_max - t_min) / (x_max - x_min)
            self._offset = t_min - x_min * self._scale
        return self

    def transform(self, X):
        result = X * self._scale + self._offset
        if self.clip:
            import numpy as np
            result = np.clip(result, self.feature_range[0], self.feature_range[1])
        return result

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


_SCALER_MAP = {
    "minmax": MinMaxScaler,
    "linear": LinearScaler,
}


class DataHandler:
    """
    Central data management for loading, preprocessing, and providing data loaders.

    Handles dataset loading, train/valid/test splitting, feature scaling,
    and provides DataLoaders for both classification and GAN-style training.

    Attributes:
        data: Dict of split tensors {"train", "valid", "test"} after preprocessing.
        labels: Dict of label tensors {"train", "valid", "test"}.
        data_dim: Number of input features.
        num_cls: Number of classes.
        classification: DataLoaders for classification training.
        discrimination: DataLoaders for GAN-style training (samples per class).
    """

    def __init__(self, cfg: DatasetConfig):
        """
        Initialize DataHandler with dataset configuration.

        Args:
            cfg: Dataset configuration with gen_dow_kwargs, split ratios, etc.
        """
        self.cfg = cfg
        self.data = None
        self.labels = None
        self.means: List[torch.Tensor] = None
        self.covs: List[torch.Tensor] = None
        self.classification: Dict[str, DataLoader] = None
        self.discrimination: Dict[str, DataLoader] = None
        self.data_dim: int = None
        self.num_cls: int = None
        self.total_size: int = None
        self.num_spc: List[int] = None
        self.ucr_train_size: Optional[int] = None

    def load(self):
        """Load raw dataset from disk (or generate if needed)."""
        lbld_data = load_dataset(self.cfg)
        self.data, self.labels = lbld_data.X, lbld_data.t
        self.data_dim, self.num_cls, self.total_size = lbld_data.num_feat, lbld_data.num_cls, lbld_data.size
        self.ucr_train_size = lbld_data.ucr_train_size  # non-None only for UCR datasets


    def _compute_mean_and_covariance(self, data: torch.FloatTensor):
        mean = data.mean(dim=0)
        cov = torch.cov(data.T)
        return mean, cov

    def _get_scaler(self, scaler_name: str):
        key = scaler_name.lower().replace("-", "").replace(" ", "")
        scaler_cls = _SCALER_MAP.get(key)
        if scaler_cls is None:
            raise ValueError(f"Unknown scaler '{scaler_name}'")
        self.scaler = scaler_cls(feature_range=self.input_range, clip=True)


    def split_and_rescale(
            self,
            bornmachine: ConditionalBornMachine,
            scaler_name: str = None   # None → read from self.cfg.scaler
    ):
        """
        Split data into train/valid/test and rescale to embedding range.

        Args:
            bornmachine: ConditionalBornMachine (used to determine input range from embedding).
            scaler_name: Scaler type ("minmax" or "linear"). If None, reads from self.cfg.scaler.
        """
        # Check that data is already loaded to handler, otherwise, load it:
        if self.data is None:
            self.load(self.cfg)
        # Final input range depends on the embedding chosen for the Born-Machine.
        self.input_range = bornmachine.input_range
        # Initalize dictionaries
        data = {}
        labels = {}

        if getattr(self.cfg, 'use_ucr_split', False) and self.ucr_train_size is not None:
            # Honour original UCR train/test boundary; carve valid from test half
            n_train = self.ucr_train_size
            train_data   = self.data[:n_train]
            train_labels = self.labels[:n_train]
            rest_data    = self.data[n_train:]
            rest_labels  = self.labels[n_train:]
            n_half = len(rest_data) // 2
            data["train"]  = train_data
            data["valid"]  = rest_data[:n_half]
            data["test"]   = rest_data[n_half:]
            labels["train"] = train_labels
            labels["valid"] = rest_labels[:n_half]
            labels["test"]  = rest_labels[n_half:]
        else:
            split_ratios = self.cfg.split
            # First split: separate test set
            (remaining_data, data["test"],
             remaining_labels, labels["test"]) = train_test_split(
                                                self.data, self.labels,
                                                test_size=split_ratios[-1],
                                                random_state=self.cfg.split_seed
                                            )
            # Second split: separate validation set from remaining data
            # ratio_new = ratio_old / ratio_remaining
            (data["train"], data["valid"],
             labels["train"], labels["valid"]) = train_test_split(
                                                remaining_data, remaining_labels,
                                                test_size=split_ratios[1]/(1-split_ratios[-1]),
                                                random_state=self.cfg.split_seed
                                            )
        # Fit scaler to training data to avoid data leakage.
        if scaler_name is None:
            scaler_name = getattr(self.cfg, 'scaler', 'minmax')
        self._get_scaler(scaler_name)
        data["train"] = self.scaler.fit_transform(data["train"])
        # Transform validation and test sets using the scaler fitted on training data
        data["valid"] = self.scaler.transform(data["valid"])
        data["test"] = self.scaler.transform(data["test"])
        # Convert to PyTorch tensors for subsequent training
        self.data, self.labels = {}, {}
        for split in ["train", "valid", "test"]:
            self.data[split] = torch.FloatTensor(data[split])
            self.labels[split] = torch.LongTensor(labels[split])

        # Always build classified_data; means/covs only for low-dim data
        all_data = torch.cat([d for d in self.data.values()], dim=0)
        all_labels = torch.cat([l for l in self.labels.values()])
        self.classified_data = []
        if self.data_dim < 1e2:
            self.means, self.covs = [], []
        for c in range(self.num_cls):
            class_data = all_data[all_labels == c]
            self.classified_data.append(class_data)
            if self.data_dim < 1e2:
                mean, cov = self._compute_mean_and_covariance(class_data)
                self.means.append(mean), self.covs.append(cov)
        min_size = min(cd.shape[0] for cd in self.classified_data)
        self.classified_data = torch.stack(
            [cd[:min_size] for cd in self.classified_data], dim=1
        )  # (min_size, num_cls, data_dim)
        del all_data
        del all_labels

    def get_classification_loaders(self, batch_size: int = 64):
        """Create DataLoaders for classification training (data, labels) pairs.

        Args:
            batch_size: Number of samples per batch. Should match the trainer's
                        configured batch_size (from ClassificationConfig, etc.).
        """
        if not isinstance(self.data, dict):
            raise AttributeError(f"Call split_and_rescale first.")

        self.classification = {}
        for split, split_data in self.data.items():
            split_labels = self.labels[split]
            lbd_data = TensorDataset(split_data, split_labels)
            effective_bs = min(batch_size, len(split_data))
            if effective_bs < batch_size:
                logger.warning(
                    f"batch_size ({batch_size}) exceeds {split} split size "
                    f"({len(split_data)}); clamping to {effective_bs}."
                )
            self.classification[split] = DataLoader(lbd_data,
                                             batch_size=effective_bs,
                                             drop_last=(split=="train"),
                                             shuffle=(split=="train"))

    def get_discrimination_loaders(self, batch_size: int = 16):
        """
        Create DataLoaders for GAN-style training.

        Provides batches of shape (batch_size, num_classes, data_dim) for
        comparing real samples against synthetic samples from the generator.

        Args:
            batch_size: Number of samples per class per batch.
        """
        num_spc = self.classified_data.shape[0]

        # Convert ratios to cumulative boundaries
        ratios = self.cfg.split
        assert abs(sum(ratios) - 1.0) < 1e-6, "split ratios must sum to 1"

        boundaries = [
            int(round(ratios[0] * num_spc)),
            int(round((ratios[0] + ratios[1]) * num_spc)),
            num_spc
        ] # i.e. the indices indicating where to split the data

        splits = {
            "train": (0, boundaries[0]),
            "valid": (boundaries[0], boundaries[1]),
            "test":  (boundaries[1], boundaries[2]),
        }

        self.discrimination = {}

        for split, (a, b) in splits.items():
            split_tensor = self.classified_data[a:b]  # shape: (num_spc, num_classes, data_dim)

            self.discrimination[split] = DataLoader(
                split_tensor,
                batch_size=batch_size,
                drop_last=(split == "train"),
                shuffle=(split == "train"),
            )

        del self.classified_data
        self.num_spc = [
            int(round(ratios[0] * num_spc)),
            int(round(ratios[1] * num_spc)),
            int(round(ratios[2] * num_spc))
        ]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

    # --- load_dataset smoke test ---
    ds_cfg = DatasetConfig(
        name="spirals",
        gen_dow_kwargs=DataGenDowConfig(name="spirals", size=32, seed=42, noise=0.1),
        overwrite=True,
    )
    ds = load_dataset(ds_cfg)
    assert ds.X.shape == (64, 2), f"Unexpected shape: {ds.X.shape}"
    assert ds.num_cls == 2, f"Unexpected num_cls: {ds.num_cls}"
    assert ds.X.dtype in (np.float32, np.float64), f"Unexpected dtype: {ds.X.dtype}"
    print(f"  spirals  shape={ds.X.shape}  classes={ds.num_cls}  dtype={ds.X.dtype}")
    print("load_dataset smoke test passed.")

    # --- DataHandler smoke test ---
    from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig
    device = torch.device("cpu")
    bm = ConditionalBornMachine(
        cfg=CBMConfig(embedding="legendre", init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3)),
        data_dim=2, num_classes=2, device=device,
    )
    dh = DataHandler(ds_cfg)
    dh.load()
    dh.split_and_rescale(bm)
    dh.get_classification_loaders(batch_size=4)
    lo, hi = bm.input_range
    for split, loader in dh.classification.items():
        x_batch, y_batch = next(iter(loader))
        assert x_batch.shape[1] == 2, f"{split}: unexpected feature dim"
        assert x_batch.min() >= lo - 1e-5 and x_batch.max() <= hi + 1e-5, \
            f"{split}: data out of range [{lo}, {hi}]"
        print(f"  {split:6s}  batch={tuple(x_batch.shape)}  range=[{x_batch.min():.3f}, {x_batch.max():.3f}]")
    print("DataHandler smoke test passed.")
