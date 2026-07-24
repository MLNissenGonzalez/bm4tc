"""Device selection shared by JEM training and evaluation entry points."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> torch.device:
    """Match the MPS policy: prefer CUDA when available, otherwise use CPU."""
    name = str(requested).lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    elif name.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is unavailable; falling back to CPU.")
        name = "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        logger.info("Using %s (%s)", device, torch.cuda.get_device_name(device))
    else:
        logger.info("Using %s", device)
    return device
