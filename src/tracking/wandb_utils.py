import src.utils.schemas as schemas
import wandb
import hydra
from datetime import datetime
from pathlib import Path
from typing import *
import logging
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

def init_wandb(cfg: schemas.Config) -> wandb.Run:
    """
    Initialize a Weights & Biases (wandb) run with configuration and runtime context.

    This function prepares a structured and reproducible WandB logging environment
    by converting the Hydra/OmegaConf configuration into a flat dict, determining
    runtime information (e.g. run directory, mode, job index), and generating a
    descriptive group and run name for both single and multi-run jobs.

    Naming Convention
    -----------------
    - **Group**: ``{experiment}_{regime}_{archinfo}_{dataset}_{date}``
      e.g. ``hpo_adv_d30D18fourier_moons_4k_1502`` (multirun)
      or   ``default_cls_d30D18fourier_moons_4k_1502_1430`` (single run)
    - **Run name**: ``{job_num}`` (0-indexed), e.g. ``0``, ``1``, ``2``
    - **regime**: concatenation of trainer codes (cls, gen, adv, gan)
      for each non-null trainer section.
    - **archinfo**: ``d{in_dim}D{bond_dim}{embedding}``

    Parameters
    ----------
    cfg : schemas.Config
        Full experiment configuration object (Hydra/OmegaConf structured config).
        Must include the following nested sections:
        - ``cfg.experiment`` (purpose-only label, e.g. "hpo", "best", "seed_sweep")
        - ``cfg.dataset.name``
        - ``cfg.born.init_kwargs`` (contains ``bond_dim``, ``in_dim``)
        - ``cfg.born.embedding``
        - ``cfg.tracking`` (contains ``project``, ``entity``, ``mode``)
        - ``cfg.trainer.*`` (classification, generative, adversarial, ganstyle)

    Returns
    -------
    wandb.Run
        The active wandb run instance.
    """

    # Convert only loggable types
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)

    # Job Info
    runtime_cfg = hydra.core.hydra_config.HydraConfig.get()
    run_dir = Path(runtime_cfg.runtime.output_dir)
    job_num = int(runtime_cfg.job.get("num", 0))

    # Regime
    regime_parts = []
    for key, code in [("trainer.classification", "cls"), ("trainer.generative", "gen"),
                       ("trainer.adversarial", "adv"), ("trainer.ganstyle", "gan")]:
        if OmegaConf.select(cfg, key) is not None:
            regime_parts.append(code)
    regime = "".join(regime_parts) or "none"

    # Architecture
    archinfo = f"d{cfg.born.init_kwargs.in_dim}D{cfg.born.init_kwargs.bond_dim}{cfg.born.embedding}"

    # Date
    mode = runtime_cfg.mode.value
    now = datetime.now()
    date_str = now.strftime("%d%m_%H%M") if mode == 1 else now.strftime("%d%m")

    group_key = f"{cfg.experiment}_{regime}_{archinfo}_{cfg.dataset.name}_{date_str}"
    run_name = str(job_num)

    # Initializing the wandb object
    run = wandb.init(
        project=cfg.tracking.project,
        entity=cfg.tracking.entity,
        dir=str(run_dir),
        config=wandb_cfg,
        group=group_key,
        name=run_name,
        mode=cfg.tracking.mode,
        reinit="finish_previous"
    )
    return run

def record(results: Dict[str, Any], stage: str, set: str, step: Optional[int] = None):
    """
    Upload evaluation results to W&B. Handles both numeric and visualization metrics.
    """
    upload_dict = {}

    for metric_name, result in results.items():
        if result is None:
            continue  # Skip explicitly empty entries (like 'rob' placeholder)

        if metric_name == "viz":
            # Handle both Figure and Axes inputs
            fig = result.figure if hasattr(result, "figure") else result
            upload_dict[f"samples/{stage}"] = wandb.Image(fig)
            plt.close(fig)

        elif isinstance(result, (float, int)):
            upload_dict[f"{stage}/{set}/{metric_name}"] = float(result)

        else:
            logger.warning(f"Skipping metric '{metric_name}' — unsupported type: {type(result)}")

    if upload_dict:
        wandb.log(upload_dict, step=step)


def log_dataset_viz(datahandler):
    """Log a scatter plot of the full (rescaled) dataset to W&B under 'dataset/all'."""
    if datahandler.data_dim != 2:
        logger.debug(f"Skipping dataset viz for data_dim={datahandler.data_dim} (only 2D supported)")
        return

    from src.tracking.visualisation import visualise_samples
    import torch

    all_data = torch.cat([datahandler.data[s] for s in ("train", "valid", "test")], dim=0)
    all_labels = torch.cat([datahandler.labels[s] for s in ("train", "valid", "test")])
    ax = visualise_samples(all_data, all_labels)
    fig = ax.figure
    wandb.log({"dataset/all": wandb.Image(fig)})
    plt.close(fig)


from src.models import BornClassifier, BornGenerator
import numpy as np

def log_grads(bm_view: BornClassifier | BornGenerator, step: int, watch_freq: int, stage: str):
    """
    Logs gradient statistics of all MPS parameters to Weights & Biases (wandb).

    The function periodically records absolute gradient histograms for each trainable
    tensor in the MPS. This is useful for monitoring training stability, detecting
    vanishing or exploding gradients, and diagnosing optimizer behavior.

    Logging occurs only every `watch_freq` steps.

    Parameters
    ----------
    mps : tk.models.MPS
        The MPS model whose gradients are logged.
    step : int
        Current global training step.
    watch_freq : int
        Frequency (in steps) at which gradient statistics are recorded.
    stage : str
        Label for the current training phase (e.g., "pre", "gan").
        Used as a prefix in the wandb metric names.

    Notes
    -----
    - Gradients are detached and moved to CPU before histogram computation.
    - Parameters without gradients at the current step are marked with
      `{stage}_mps_abs_grad/{name}/has_grad = 0`.
    - To reduce logging overhead, scalar summaries (mean/std/max) are currently disabled
      but can be re-enabled by uncommenting the respective lines.
    """
    if step % watch_freq != 0:
        return

    log_grads = {}
    for name, tensor in bm_view.named_parameters():
        if tensor.grad is not None:
            g = tensor.grad.detach().abs().cpu().numpy()
            log_grads[f"{stage}_mps_abs_grad/{name}/hist"] = wandb.Histogram(
                np_histogram=np.histogram(a=g, bins=64)
            )
            # log_grads[f"{stage}_mps_abs_grad/{name}/mean"] = g.mean()
            # log_grads[f"{stage}_mps_abs_grad/{name}/std"] = g.std()
            # log_grads[f"{stage}_mps_abs_grad/{name}/max"] = g.max()
            # log_grads[f"{stage}_mps_abs_grad/{name}/has_grad"] = 1
        else:
            log_grads[f"{stage}_mps_abs_grad/{name}/has_grad"] = 0

    wandb.log(log_grads)
