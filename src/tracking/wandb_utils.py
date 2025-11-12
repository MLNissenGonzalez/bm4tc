import src.utils.schemas as schemas
import wandb
import omegaconf
import hydra
import datetime
from pathlib import Path
from typing import *
import logging
logger = logging.getLogger(__name__)

def init_wandb(cfg: schemas.Config) -> wandb.Run:
    """
    Initialize a Weights & Biases (wandb) run with configuration and runtime context.

    This function prepares a structured and reproducible WandB logging environment
    by converting the Hydra/OmegaConf configuration into a flat dict, determining
    runtime information (e.g. run directory, mode, job index), and generating a
    descriptive group and run name for both single and multi-run jobs.

    Behavior
    --------
    - Supports both single and multirun Hydra modes.
    - Constructs descriptive `group` and `name` fields that encode dataset,
      experiment name, date, and key model hyperparameters.
    - Ensures that runs are properly reinitialized when multiple jobs are spawned
      in the same process (via `reinit="finish_previous"`).

    Parameters
    ----------
    cfg : schemas.Config
        Full experiment configuration object (Hydra/OmegaConf structured config).
        Must include the following nested sections:
        - `cfg.experiment` (experiment identifier)
        - `cfg.dataset.name`
        - `cfg.model.mps.init_kwargs` (contains `bond_dim`, `in_dim`)
        - `cfg.pretrain.mps.max_epoch`
        - `cfg.gantrain.max_epoch`
        - `cfg.wandb` (contains `project`, `entity`, `mode`)

    Returns
    -------
    wandb.Run
        The active wandb run instance.

    Raises
    ------
    TypeError
        If any Hydra sweeper parameter group has an unexpected type.
    """

    # Convert only loggable types
    wandb_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)

    # Job Info
    runtime_cfg = hydra.core.hydra_config.HydraConfig.get()
    run_dir = Path(runtime_cfg.runtime.output_dir)
    job_num = int(runtime_cfg.job.get("num", 0)) + 1

    # Job Mode
    mode = runtime_cfg.mode.value
    total_num = 1
    if mode == 1:  # single run
        now = datetime.now().strftime("%d%b%y_%I%p%M")
    else:  # multirun
        now = datetime.now().strftime("%d%b%y")
        params = runtime_cfg.sweeper.params
        for group in params.values():
            # handle both list and string forms
            if isinstance(group, (list, tuple, omegaconf.ListConfig)):
                options = group
            elif isinstance(group, str):
                options = group.split(",")
            else:
                raise TypeError(
                    f"Unexpected sweeper param type: {type(group)} ({group})")
            total_num *= len(options)

    # Group (folder) and run name
    group_key = f"{cfg.experiment}_{cfg.dataset.name}_{now}"
    run_name = f"job{job_num}/{total_num}_D{cfg.model.mps.init_kwargs.bond_dim}-d{cfg.model.mps.init_kwargs.in_dim}-pre{cfg.pretrain.mps.max_epoch}-gan{cfg.gantrain.max_epoch}"

    # Initializing the wandb object
    run = wandb.init(
        project=cfg.track.project,
        entity=cfg.track.entity,
        dir=str(run_dir),
        config=wandb_cfg,
        group=group_key,
        name=run_name,
        mode=cfg.track.mode,
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

        elif isinstance(result, (float, int)):
            upload_dict[f"{stage}_mps/{set}/{metric_name}"] = float(result)

        else:
            logger.warning(f"Skipping metric '{metric_name}' — unsupported type: {type(result)}")

    if upload_dict:
        wandb.log(upload_dict, step=step)


from models import BornClassifier, BornGenerator
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

# TODO: Think about moving this completly to the models.
import torch
# General saving function
def save_model(model: torch.nn.Module, run_name: str, model_type: str):
    """
    Save model inside the Hydra run's output directory:
    ${hydra:run.dir}/models/{model_type}_{run_name}.pt
    """
    assert model_type in ["pre_mps", "pre_dis", "gan_mps", "gan_dis"], \
        f"Invalid model_type {model_type}"

    # Hydra's current run dir
    run_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Models subfolder inside it
    folder = run_dir / "models"
    folder.mkdir(parents=True, exist_ok=True)

    # Save path
    filename = f"{model_type}_{run_name}.pt"
    save_path = folder / filename

    # Save state dict
    torch.save(model.state_dict(), save_path)

    return str(save_path)