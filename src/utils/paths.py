import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Return the base directory for all data I/O (datasets, outputs, analysis).

    Reads BM4TC_DATA_ROOT from the environment.  When unset, falls back to the
    repository root, preserving the existing repo-relative layout.

    On a cluster where code and data live on separate filesystems, set:
        export BM4TC_DATA_ROOT=/path/to/data/root
    """
    env = os.environ.get("BM4TC_DATA_ROOT")
    return Path(env) if env else _PROJECT_ROOT
