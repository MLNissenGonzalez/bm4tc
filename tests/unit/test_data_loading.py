import multiprocessing
import os

import pytest
import torch


def _make_spirals_handler():
    from bm4tc.pipeline.data import DatasetConfig, DataGenDowConfig
    from bm4tc.pipeline.data import DataHandler
    from bm4tc.core.model import ConditionalBornMachine, CBMConfig, MPSInitConfig

    bm = ConditionalBornMachine(
        cfg=CBMConfig(embedding="legendre", init_kwargs=MPSInitConfig(in_dim=2, bond_dim=2, std=1e-3)),
        data_dim=2, num_classes=2, device=torch.device("cpu"),
    )
    ds_cfg = DatasetConfig(
        name="spirals",
        gen_dow_kwargs=DataGenDowConfig(name="spirals", size=32, seed=42, noise=0.1),
        overwrite=True,
    )
    dh = DataHandler(ds_cfg)
    return bm, dh


def test_spirals_small():
    import numpy as np
    from bm4tc.pipeline.data import DatasetConfig, DataGenDowConfig, load_dataset

    ds_cfg = DatasetConfig(
        name="spirals",
        gen_dow_kwargs=DataGenDowConfig(name="spirals", size=32, seed=42, noise=0.1),
        overwrite=True,
    )
    ds = load_dataset(ds_cfg)
    assert ds.X.shape == (64, 2), f"Expected (64, 2), got {ds.X.shape}"
    assert ds.num_cls == 2, f"Expected 2 classes, got {ds.num_cls}"
    assert ds.X.dtype in (np.float32, np.float64)


def test_generation_settings_name_the_cached_file(tmp_path, monkeypatch):
    """A cached dataset is reused only for the same settings: a spirals file of
    another size must not be loaded in place of the configured one."""
    from bm4tc.pipeline.data import DatasetConfig, DataGenDowConfig, load_dataset

    monkeypatch.setenv("BM4TC_DATA_ROOT", str(tmp_path))
    sizes = {}
    for size in (32, 48):
        cfg = DatasetConfig(name="spirals", gen_dow_kwargs=DataGenDowConfig(
            name="spirals", size=size, seed=42, noise=0.1))
        sizes[size] = load_dataset(cfg).X.shape[0]
    assert sizes == {32: 64, 48: 96}
    assert sorted(p.name for p in (tmp_path / ".datasets" / "spirals").glob("*.npz")) == [
        "spirals_n32_s42_noise0.1.npz", "spirals_n48_s42_noise0.1.npz"]


def _load_spirals(root: str) -> tuple:
    os.environ["BM4TC_DATA_ROOT"] = root
    from bm4tc.pipeline.data import DatasetConfig, DataGenDowConfig, load_dataset
    cfg = DatasetConfig(name="spirals", gen_dow_kwargs=DataGenDowConfig(
        name="spirals", size=20000, seed=42, noise=0.1))
    return load_dataset(cfg).X.shape


def test_parallel_first_use_creates_the_dataset_once(tmp_path):
    """Units starting together on an empty cache all load a whole file (the first
    creates it under a lock; the write is atomic), and no temporary file is left."""
    with multiprocessing.get_context("fork").Pool(6) as pool:
        shapes = pool.map(_load_spirals, [str(tmp_path)] * 6)
    assert shapes == [(40000, 2)] * 6
    assert [p.name for p in (tmp_path / ".datasets" / "spirals").glob("*.tmp")] == []


def test_handler_spirals_small():
    lo, hi = -1.0, 1.0  # legendre range
    bm, dh = _make_spirals_handler()
    dh.load()
    dh.split_and_rescale(bm.input_range)
    dh.get_classification_loaders(batch_size=4)

    assert dh.classification is not None
    for split in ("train", "valid", "test"):
        assert split in dh.classification
        x, y = next(iter(dh.classification[split]))
        assert x.shape[1] == 2
        assert x.min() >= lo - 1e-5 and x.max() <= hi + 1e-5, \
            f"{split}: data out of range [{x.min():.3f}, {x.max():.3f}]"
        assert y.dtype == torch.long


@pytest.mark.requires_download
def test_mnist_loading():
    import numpy as np
    from bm4tc.pipeline.data import DatasetConfig, DataGenDowConfig, load_dataset

    ds_cfg = DatasetConfig(
        name="mnist",
        gen_dow_kwargs=DataGenDowConfig(name="mnist"),
        overwrite=False,
    )
    ds = load_dataset(ds_cfg)
    assert ds.X.ndim == 2
    assert ds.X.shape[1] == 784, f"Expected 784 features, got {ds.X.shape[1]}"
    assert ds.num_cls == 10, f"Expected 10 classes, got {ds.num_cls}"
    assert ds.X.dtype in (np.float32, np.float64)


@pytest.mark.requires_download
def test_mnist_resize():
    import numpy as np
    from bm4tc.pipeline.data import DatasetConfig, DataGenDowConfig, load_dataset

    resize = 12
    ds_cfg = DatasetConfig(
        name="mnist",
        gen_dow_kwargs=DataGenDowConfig(name="mnist", resize=resize),
        overwrite=False,
    )
    ds = load_dataset(ds_cfg)
    assert ds.X.ndim == 2
    assert ds.X.shape[1] == resize ** 2, f"Expected {resize**2} features, got {ds.X.shape[1]}"
    assert ds.num_cls == 10, f"Expected 10 classes, got {ds.num_cls}"
    assert ds.X.dtype in (np.float32, np.float64)
    assert ds.X.min() >= 0.0 and ds.X.max() <= 1.0, "Pixel values out of [0, 1]"
    assert ds.name == "mnist_r12", f"Unexpected stem: {ds.name}"


@pytest.mark.requires_download
def test_italy_power_demand_loading():
    import numpy as np
    from bm4tc.pipeline.data import DatasetConfig, DataGenDowConfig, load_dataset

    ds_cfg = DatasetConfig(
        name="italypowerdemand",
        gen_dow_kwargs=DataGenDowConfig(name="italypowerdemand"),
        use_ucr_split=True,
        overwrite=False,
    )
    ds = load_dataset(ds_cfg)
    assert ds.X.ndim == 2
    assert ds.num_cls == 2, f"Expected 2 classes, got {ds.num_cls}"
    assert ds.ucr_train_size is not None, "ucr_train_size should be set for UCR datasets"
    assert ds.X.dtype in (np.float32, np.float64)


def test_robust_sigmoid_is_one_global_transform_robust_to_outliers():
    """MPSTime's scaled robust sigmoid (docs/timeseries.md): one transform for every
    feature, monotone, the train extremes on the range ends, and a spike does not
    squeeze the other values (a min/max scaler leaves them a sliver of the range)."""
    import numpy as np
    from bm4tc.pipeline.data import LinearScaler, RobustSigmoidScaler

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 20))
    X[0, 3] = 40.0                                             # one spike
    scaler = RobustSigmoidScaler(feature_range=(-1., 1.), clip=True)
    Y = scaler.fit_transform(X)
    assert Y.min() == -1. and Y.max() == 1.
    order = np.argsort(X, axis=None)
    assert np.all(np.diff(Y.flatten()[order]) >= 0)            # monotone, one transform
    spread = lambda Z: np.median(Z[1:].max(1) - Z[1:].min(1))  # the series without the spike
    linear = LinearScaler(feature_range=(-1., 1.)).fit_transform(X)
    assert spread(Y) > 0.6 * 2 and spread(linear) < 0.3 * 2
    assert np.all(scaler.transform(X * 10) <= 1.)              # unseen extremes are clipped


def test_spirals_cell_masses_match_the_generator():
    """The law written in _spirals_cell_masses is the law _two_dim_generator samples
    (clipped into the outer cells), cell by cell."""
    import numpy as np
    from bm4tc.pipeline.data import DataGenDowConfig, _spirals_cell_masses, _two_dim_generator

    cfg = DataGenDowConfig(name="spirals", size=500_000, seed=3, noise=0.5)
    X, t = _two_dim_generator(cfg)
    edges = np.linspace(-12, 12, 17)        # the outer cells hold the tails
    masses = _spirals_cell_masses(cfg, edges, scale=np.ones(2), shift=np.zeros(2))

    assert masses.shape == (2, 16, 16)
    np.testing.assert_allclose(masses.sum(axis=(1, 2)), 1.0, atol=1e-9)
    clipped = np.clip(X, edges[0], edges[-1])
    for c in (0, 1):
        counts, _, _ = np.histogram2d(*clipped[t == c].T, bins=[edges, edges])
        np.testing.assert_allclose(masses[c], counts / counts.sum(), atol=2e-3)


def test_class_masses_only_for_a_known_law():
    import numpy as np
    from bm4tc.pipeline.data import DataHandler, DatasetConfig

    assert DataHandler(DatasetConfig(name="moons")).class_masses(np.linspace(-1, 1, 5)) is None
