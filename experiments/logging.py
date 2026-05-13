import json
from pathlib import Path
from typing import Callable


def make_logger(output_dir: Path, wandb_run=None) -> Callable[[int, dict], None]:
    """
    Returns an on_epoch_end callback that writes epoch metrics to log.json
    and optionally forwards them to a W&B run.
    """
    log_path = output_dir / "log.json"
    records = []

    def log(epoch: int, metrics: dict) -> None:
        records.append({"epoch": epoch, **metrics})
        log_path.write_text(json.dumps(records, indent=2))
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, **metrics})

    return log
