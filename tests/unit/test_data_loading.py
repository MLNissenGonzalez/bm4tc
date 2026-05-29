import pytest
import torch


def _make_spirals_handler():
    from src.datahandler import DatasetConfig, DataGenDowConfig
    from src.datahandler import DataHandler
    from src.model import ConditionalBornMachine, CBMConfig, MPSInitConfig

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
    from src.datahandler import DatasetConfig, DataGenDowConfig, load_dataset

    ds_cfg = DatasetConfig(
        name="spirals",
        gen_dow_kwargs=DataGenDowConfig(name="spirals", size=32, seed=42, noise=0.1),
        overwrite=True,
    )
    ds = load_dataset(ds_cfg)
    assert ds.X.shape == (64, 2), f"Expected (64, 2), got {ds.X.shape}"
    assert ds.num_cls == 2, f"Expected 2 classes, got {ds.num_cls}"
    assert ds.X.dtype in (np.float32, np.float64)


def test_handler_spirals_small():
    lo, hi = -1.0, 1.0  # legendre range
    bm, dh = _make_spirals_handler()
    dh.load()
    dh.split_and_rescale(bm)
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
    from src.datahandler import DatasetConfig, DataGenDowConfig, load_dataset

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
    from src.datahandler import DatasetConfig, DataGenDowConfig, load_dataset

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
    from src.datahandler import DatasetConfig, DataGenDowConfig, load_dataset

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
