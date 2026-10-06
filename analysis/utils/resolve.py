"""Infer a sweep's training regime and embedding from its output path.

Output paths have the format::

    outputs/{dataset}/{nat|at}/{embedding}/{arch}/{kind}_{ddmm}/

Phase 4 replaces this with run manifests (D17).
"""

import re
from typing import Optional

from src.utils.embeddings import _EMBEDDING_RANGE_SIZE, embedding_range_size  # noqa: F401 (re-export)

_REGIMES = ("nat", "at")
_KNOWN_EMBEDDINGS = set(_EMBEDDING_RANGE_SIZE.keys())


def resolve_regime_from_path(sweep_dir: str) -> Optional[str]:
    """``"nat"`` or ``"at"`` from the first matching directory name, else None.

    Examples:
        >>> resolve_regime_from_path("outputs/circles/nat/fourier/d4r3/seed_sweep_a0_0102")
        'nat'
        >>> resolve_regime_from_path("outputs/circles/at/legendre/d10r6/seed_sweep_2804")
        'at'
    """
    for token in str(sweep_dir).replace("\\", "/").split("/"):
        if token.lower() in _REGIMES:
            return token.lower()
    return None


def resolve_embedding_from_path(sweep_dir: str) -> Optional[str]:
    """The embedding named in the path (directory or ``_``-separated token), else None."""
    for token in re.split(r"[/_]", str(sweep_dir).replace("\\", "/")):
        if token.lower() in _KNOWN_EMBEDDINGS:
            return token.lower()
    return None
