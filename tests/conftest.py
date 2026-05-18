import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests requiring model forward passes; skip with -m 'not slow'")
    config.addinivalue_line("markers", "requires_download: marks tests that download external data; skip unless --run-downloads is passed")


def pytest_addoption(parser):
    parser.addoption("--run-downloads", action="store_true", default=False,
                     help="Run tests that download external datasets (MNIST, UCR, etc.)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-downloads"):
        skip_dl = pytest.mark.skip(reason="needs --run-downloads to run")
        for item in items:
            if item.get_closest_marker("requires_download"):
                item.add_marker(skip_dl)
from torch.utils.data import DataLoader, TensorDataset

DATA_DIM = 4
NUM_CLASSES = 2


@pytest.fixture(scope="session")
def born_machine():
    from omegaconf import OmegaConf
    from src.models.born import BornMachine
    cfg = OmegaConf.create({
        "embedding": "fourier",
        "init_kwargs": {
            "in_dim": 4,
            "bond_dim": 2,
            "out_position": 2,
            "boundary": "obc",
            "init_method": "randn",
            "dtype": "complex64",
            "n_features": None,
            "out_dim": None,
            "std": 1e-3,
        },
        "model_path": None,
    })
    bm = BornMachine(cfg=cfg, data_dim=DATA_DIM, num_classes=NUM_CLASSES, device="cpu")
    bm.cache_log_Z()
    return bm


@pytest.fixture
def x_batch():
    return torch.rand(8, DATA_DIM)


@pytest.fixture
def y_batch():
    return torch.randint(0, NUM_CLASSES, (8,))


@pytest.fixture
def clean_loader():
    ds = TensorDataset(
        torch.rand(32, DATA_DIM),
        torch.randint(0, NUM_CLASSES, (32,)),
    )
    return DataLoader(ds, batch_size=8, shuffle=False)
