# utils that are practical for experiments
import torch
import numpy as np
import random
import os
import logging


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------Random Seed -----------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

def set_seed(seed: int):
    """
    Set random seeds across Python, NumPy, and PyTorch for reproducible
    *training* randomness (model init, DataLoader shuffling, PGD, sampling).

    Must be called **after** data loading and **before** model creation.
    Data pipeline seeds (``gen_dow_kwargs.seed``, ``dataset.split_seed``)
    are handled independently by their respective functions.

    Parameters
    ----------
    seed : int
        Integer value used to seed all random number generators.
        Corresponds to ``tracking.seed`` in the Hydra config.

    Notes
    -----
    - Seeds Python's `random` module, NumPy, and PyTorch (CPU and GPU).
    - For PyTorch, also sets `torch.backends.cudnn.deterministic=True` to
      enforce deterministic algorithms in cuDNN.
    - Disables `torch.backends.cudnn.benchmark` to avoid non-deterministic
      optimizations.
    - Sets the `PYTHONHASHSEED` environment variable for hash-based operations.
    - May reduce performance due to disabling some GPU optimizations.

    Examples
    --------
    >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Ensure deterministic behavior in cuDNN (can slow things down!)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set PYTHONHASHSEED environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)




def sample_quality_control(synths: torch.FloatTensor,
                           upper: float, lower: float):
    """
    Inspect generated samples for out-of-bound values.

    Parameters
    ----------
    synths : torch.FloatTensor
        Tensor of generated samples, shape (num_samples, num_classes, num_features).
    upper : float
        Upper bound for acceptable sample values.
    lower : float
        Lower bound for acceptable sample values.

    Notes
    -----
    - Logs the number of "bad" positions where values exceed bounds.
    - Reports the first 200 offending indices and values.
    - Logs per-class and per-feature maximum absolute value and mean.
    - Useful for debugging numerical instabilities in generative models.
    """
    bad_idx = (
        (synths.abs() > upper) | (synths < lower)
    ).nonzero(as_tuple=False)  # (sample_idx, class_idx, feat_idx)
    logger.info(f"bad positions count = {bad_idx.shape[0]}")
    if bad_idx.shape[0] > 0:
        # show first few offending indices and their values
        for i in range(min(200, bad_idx.shape[0])):
            s, c, f = bad_idx[i].tolist()
            val = synths[s, c, f].item()
            logger.info(f"BAD value at sample={s}, class={c}, feat={f}: {val}")
        # show global per-dim & per-class stats
        logger.info("per-class-per-dim max abs:")
        for c in range(synths.shape[1]):
            for f in range(synths.shape[2]):
                m = synths[:, c, f].abs().max().item()
                logger.info(
                    f" class={c}, feat={f}, max_abs={m:.4g}, mean={synths[:,c,f].mean().item():.4g}")
    else:
        logger.info("No values above threshold found.")
