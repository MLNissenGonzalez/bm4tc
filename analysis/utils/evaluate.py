"""
Post-hoc per-model evaluation for sweep analysis.

Loads each saved model from a sweep directory and recomputes metrics
using ``MetricFactory.create()`` directly, giving the user control over
metric selection, evasion/sampling configs, and evaluation splits.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from .mia_utils import load_run_config, find_model_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Configuration for post-hoc evaluation.

    Attributes:
        compute_acc: Evaluate clean classification accuracy.
        compute_rob: Evaluate adversarial robustness.
        compute_mia: Run membership inference attack evaluation.
        compute_cls_loss: Evaluate classification loss (NLL).
        compute_gen_loss: Evaluate generative loss (joint NLL).
        compute_fid: Evaluate FID-like score.
        splits: Data splits to evaluate on (e.g. ["valid", "test"]).
        evasion_override: Dict of evasion config fields to override,
            or None to use each run's own config.
        sampling_override: Dict of sampling config fields to override,
            or None to use each run's own config.
        mia_features: Dict of MIA feature toggles (passed to MIAFeatureConfig).
        device: Torch device string.
    """
    compute_acc: bool = True
    compute_rob: bool = True
    compute_mia: bool = True
    compute_cls_loss: bool = False
    compute_gen_loss: bool = False
    compute_fid: bool = False
    splits: List[str] = field(default_factory=lambda: ["test"])
    evasion_override: Optional[Dict[str, Any]] = None
    sampling_override: Optional[Dict[str, Any]] = None
    mia_features: Optional[Dict[str, bool]] = None
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_config_overrides(
    cfg,
    evasion_override: Optional[Dict[str, Any]],
    sampling_override: Optional[Dict[str, Any]],
):
    """Mutate *cfg* in-place, overriding tracking.evasion / tracking.sampling."""
    if evasion_override:
        for key, value in evasion_override.items():
            OmegaConf.update(cfg, f"tracking.evasion.{key}", value, force_add=True)
    if sampling_override:
        for key, value in sampling_override.items():
            OmegaConf.update(cfg, f"tracking.sampling.{key}", value, force_add=True)


def resolve_stop_criterion(
    stop_crit: str,
    df: pd.DataFrame,
    eval_split: str = "test",
) -> Tuple[Optional[str], bool, str]:
    """Map a stop-criterion name to the corresponding eval column.

    Args:
        stop_crit: Short name like "acc", "loss", "rob".
        df: Eval DataFrame (to check available columns).
        eval_split: Split used during evaluation (e.g. "test").

    Returns:
        (column_name, minimize, label) or (None, True, "") if unresolvable.
    """
    prefix = f"eval/{eval_split}"
    if stop_crit == "acc":
        col = f"{prefix}/acc"
        return (col, False, "Accuracy (stop crit)") if col in df.columns else (None, True, "")
    elif stop_crit == "loss" or stop_crit == "clsloss":
        col = f"{prefix}/clsloss"
        return (col, True, "Cls Loss (stop crit)") if col in df.columns else (None, True, "")
    elif stop_crit == "rob":
        rob_cols = [c for c in df.columns if c.startswith(f"{prefix}/rob/")]
        if rob_cols:
            df["_avg_rob_acc"] = df[rob_cols].mean(axis=1)
            return ("_avg_rob_acc", False, "Avg Robust Accuracy (stop crit)")
        return (None, True, "")
    elif stop_crit == "genloss":
        col = f"{prefix}/genloss"
        return (col, True, "Gen Loss (stop crit)") if col in df.columns else (None, True, "")
    return (None, True, "")


# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------

def evaluate_run(
    run_dir: Path,
    eval_cfg: EvalConfig,
) -> Dict[str, float]:
    """Load a single run's model and compute metrics post-hoc.

    Args:
        run_dir: Path to the Hydra output directory for one run.
        eval_cfg: Evaluation configuration.

    Returns:
        Flat dict with keys like ``eval/test/acc``, ``eval/valid/rob/0.1``,
        ``eval/mia_accuracy``, ``eval/mia_auc_roc``.
    """
    from src.models import BornMachine
    from src.data.handler import DataHandler
    from src.tracking.evaluator import MetricFactory

    run_dir = Path(run_dir)
    device = torch.device(eval_cfg.device)
    results: Dict[str, float] = {}

    # 1. Load config
    cfg = load_run_config(run_dir)

    # 2. Apply evasion/sampling overrides
    _apply_config_overrides(cfg, eval_cfg.evasion_override, eval_cfg.sampling_override)

    # 3. Load model
    checkpoint_path = find_model_checkpoint(run_dir)
    bm = BornMachine.load(str(checkpoint_path))
    bm.to(device)

    # 4. Sync generator to match classifier tensors (needed for gen loss & FID)
    if eval_cfg.compute_gen_loss or eval_cfg.compute_fid:
        bm.sync_tensors(after="classification")

    # 5. Ensure dataset is regenerated with correct seed
    OmegaConf.update(cfg, "dataset.overwrite", True, force_add=True)

    # 6. Load data
    datahandler = DataHandler(cfg.dataset)
    datahandler.load()
    datahandler.split_and_rescale(bm)
    datahandler.get_classification_loaders()

    # 7. Evaluate each metric on each split
    metric_names = []
    if eval_cfg.compute_acc:
        metric_names.append("acc")
    if eval_cfg.compute_cls_loss:
        metric_names.append("clsloss")
    if eval_cfg.compute_rob:
        metric_names.append("rob")
    if eval_cfg.compute_gen_loss:
        metric_names.append("genloss")
    if eval_cfg.compute_fid:
        metric_names.append("fid")

    for split in eval_cfg.splits:
        context: Dict[str, Any] = {}
        for metric_name in metric_names:
            try:
                metric = MetricFactory.create(
                    metric_name=metric_name,
                    freq=1,
                    cfg=cfg,
                    datahandler=datahandler,
                    device=device,
                )
                result = metric.evaluate(bm, split, context)

                if metric_name == "rob":
                    # RobustnessMetric returns a list of accuracies
                    strengths = cfg.tracking.evasion.strengths
                    for i, strength in enumerate(strengths):
                        results[f"eval/{split}/rob/{strength}"] = float(result[i])
                else:
                    results[f"eval/{split}/{metric_name}"] = float(result)
            except Exception as e:
                logger.warning(f"Metric '{metric_name}' failed on split '{split}': {e}")
                if metric_name == "rob":
                    strengths = getattr(cfg.tracking.evasion, "strengths", [])
                    for strength in strengths:
                        results[f"eval/{split}/rob/{strength}"] = np.nan
                else:
                    results[f"eval/{split}/{metric_name}"] = np.nan

    # 8. MIA
    if eval_cfg.compute_mia:
        try:
            from .mia import MIAEvaluation, MIAFeatureConfig

            feature_kwargs = eval_cfg.mia_features or {}
            feature_config = MIAFeatureConfig(**feature_kwargs)
            mia_eval = MIAEvaluation(feature_config=feature_config)

            mia_results = mia_eval.evaluate(
                bm,
                datahandler.classification["train"],
                datahandler.classification["test"],
                device,
            )
            results["eval/mia_accuracy"] = mia_results.attack_accuracy
            results["eval/mia_auc_roc"] = mia_results.auc_roc
        except Exception as e:
            logger.warning(f"MIA evaluation failed: {e}")
            results["eval/mia_accuracy"] = np.nan
            results["eval/mia_auc_roc"] = np.nan

    # 9. Cleanup
    del bm
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Sweep-level evaluation
# ---------------------------------------------------------------------------

def evaluate_sweep(
    sweep_dir: str,
    eval_cfg: EvalConfig,
    config_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Evaluate all runs in a sweep directory.

    Args:
        sweep_dir: Path to the sweep directory (contains numbered sub-dirs).
        eval_cfg: Evaluation configuration.
        config_keys: Hydra config keys to extract into the DataFrame
            (e.g. ``["dataset.name", "tracking.seed"]``).

    Returns:
        DataFrame with one row per run, containing config values and
        eval metric columns.
    """
    sweep_path = Path(sweep_dir)
    run_dirs = sorted(
        [d for d in sweep_path.iterdir() if d.is_dir() and (d / ".hydra" / "config.yaml").exists()],
        key=lambda d: d.name,
    )

    if not run_dirs:
        print(f"No valid run directories found in {sweep_path}")
        return pd.DataFrame()

    print(f"Found {len(run_dirs)} runs in {sweep_path}")

    rows = []
    for i, run_dir in enumerate(run_dirs):
        print(f"  [{i + 1}/{len(run_dirs)}] Evaluating {run_dir.name}...")

        row: Dict[str, Any] = {
            "run_name": run_dir.name,
            "run_path": str(run_dir),
        }

        # Extract config values
        if config_keys:
            try:
                cfg = load_run_config(run_dir)
                for key in config_keys:
                    try:
                        val = OmegaConf.select(cfg, key)
                        col_name = "config/" + key.split(".")[-1]
                        row[col_name] = val
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Could not load config for {run_dir.name}: {e}")

        # Evaluate
        try:
            metrics = evaluate_run(run_dir, eval_cfg)
            row.update(metrics)
        except Exception as e:
            import traceback
            print(f"    FAILED: {e}")
            traceback.print_exc()
            logger.warning(f"Run {run_dir.name} failed: {e}")

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nEvaluation complete. {len(df)} runs processed.")
    return df
