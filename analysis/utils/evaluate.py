"""
Post-hoc per-model evaluation for sweep analysis.

Loads each saved model from a sweep directory and recomputes metrics
using src/ functions directly. All results are for the test split only.

Result dict key conventions (flat, no split prefix):
    acc                         clean classification accuracy
    dis_loss                    discriminative NLL loss
    gen_loss                    generative (joint) NLL loss
    rob/<abs_eps>               robust accuracy at absolute epsilon
    mia_accuracy, mia_auc_roc   membership inference attack results
    uq_*                        uncertainty quantification results

Attack strength conventions:
    evasion_override["strengths"] — absolute epsilon values (caller converts fractions
        to absolute by multiplying by embedding range size before passing here).
    Model's own evasion config strengths — fraction-based; converted internally via
        bm.input_range.
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
        compute_dis_loss: Evaluate discriminative NLL loss.
        compute_gen_loss: Evaluate generative (joint) NLL loss.
        compute_rob: Evaluate adversarial robustness. Attempted for all models;
            skipped with a warning if no evasion config is present.
        compute_mia: Run membership inference attack evaluation.
        compute_uq: Uncertainty quantification (detection + purification).
        evasion_override: Dict of evasion config fields to override, or None to
            use each run's own config. Strengths here are ABSOLUTE (pre-multiplied
            by range size). Example: {"method": "PGD", "num_steps": 40,
            "strengths": [0.05, 0.10, 0.15]}.
        mia_features: Feature toggle dict for MIAFeatureConfig.
        mia_adversarial_strength: Absolute epsilon for adversarial MIA.
        mia_adversarial_num_steps: PGD steps for adversarial MIA.
        mia_adversarial_step_size: PGD step size. None = auto.
        mia_adversarial_norm: Lp norm for adversarial MIA.
        uq_config: Dict of kwargs for UQConfig.
        joint_uq_config: Dict of kwargs for a second UQ pass with JOINT_PGD.
        device: Torch device string.
    """
    compute_acc: bool = True
    compute_dis_loss: bool = False
    compute_gen_loss: bool = False
    compute_rob: bool = True
    compute_mia: bool = True
    compute_uq: bool = False
    evasion_override: Optional[Dict[str, Any]] = None
    mia_features: Optional[Dict[str, bool]] = None
    mia_adversarial_strength: Optional[float] = None
    mia_adversarial_num_steps: int = 20
    mia_adversarial_step_size: Optional[float] = None
    mia_adversarial_norm: Any = "inf"
    uq_config: Optional[Dict[str, Any]] = None
    joint_uq_config: Optional[Dict[str, Any]] = None
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_rob_params(
    cbm,
    cfg,
    evasion_override: Optional[Dict[str, Any]],
) -> Optional[Tuple[Any, List[float]]]:
    """Build (attack_object, abs_strengths) for robustness evaluation.

    Returns None if no evasion config is available and no override is given.
    Strengths in evasion_override are absolute; strengths from the model config
    are fractions converted via bm.input_range.
    """
    from src.utils.evasion import EvasionConfig, build_attack

    if evasion_override is not None:
        strengths_abs = [float(s) for s in evasion_override.get("strengths", [])]
        ec = EvasionConfig(
            method=evasion_override.get("method", "PGD"),
            norm=evasion_override.get("norm", "inf"),
            num_steps=evasion_override.get("num_steps", 10),
            step_size=evasion_override.get("step_size", None),
            random_start=evasion_override.get("random_start", True),
            strengths=strengths_abs,
        )
    else:
        try:
            raw = OmegaConf.to_container(cfg.trainer.adversarial.evasion, resolve=True)
            ec = EvasionConfig(**raw)
            range_size = float(cbm.input_range[1] - cbm.input_range[0])
            strengths_abs = [s * range_size for s in ec.strengths]
        except Exception as e:
            logger.warning(f"No evasion config found; skipping robustness eval. ({e})")
            return None

    if not strengths_abs:
        return None

    return build_attack(ec), strengths_abs


# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------

def evaluate_run(
    run_dir: Path,
    eval_cfg: EvalConfig,
) -> Dict[str, Any]:
    """Load a single run's model and compute metrics post-hoc on the test set.

    Args:
        run_dir: Path to the Hydra output directory for one run.
        eval_cfg: Evaluation configuration.

    Returns:
        Flat dict with keys like ``acc``, ``dis_loss``, ``rob/0.1``,
        ``mia_accuracy``, ``uq_clean_accuracy``, etc.
    """
    from src.models import ConditionalBornMachine
    from src.data.handler import DataHandler
    from src.trainer.utils import eval_metrics, eval_rob

    run_dir = Path(run_dir)
    device = torch.device(eval_cfg.device)
    results: Dict[str, Any] = {}

    # 1. Load config
    cfg = load_run_config(run_dir)

    # 2. Load model
    checkpoint_path = find_model_checkpoint(run_dir)
    cbm = ConditionalBornMachine.load(str(checkpoint_path))
    cbm.to(device)

    # 3. Load data
    OmegaConf.update(cfg, "dataset.overwrite", True, force_add=True)
    datahandler = DataHandler(cfg.dataset)
    datahandler.load()
    datahandler.split_and_rescale(cbm)
    datahandler.get_classification_loaders()
    loader = datahandler.classification["test"]

    # 4. Core metrics (single forward pass)
    if eval_cfg.compute_acc or eval_cfg.compute_dis_loss or eval_cfg.compute_gen_loss:
        try:
            dis_loss, acc, gen_loss = eval_metrics(cbm, loader, device)
            if eval_cfg.compute_acc:
                results["acc"] = acc
            if eval_cfg.compute_dis_loss:
                results["dis_loss"] = dis_loss
            if eval_cfg.compute_gen_loss:
                results["gen_loss"] = gen_loss
        except Exception as e:
            logger.warning(f"eval_metrics failed: {e}")
            if eval_cfg.compute_acc:
                results["acc"] = np.nan
            if eval_cfg.compute_dis_loss:
                results["dis_loss"] = np.nan
            if eval_cfg.compute_gen_loss:
                results["gen_loss"] = np.nan

    # 5. Robustness
    if eval_cfg.compute_rob:
        rob_params = _get_rob_params(cbm, cfg, eval_cfg.evasion_override)
        if rob_params is not None:
            attack, strengths_abs = rob_params
            for abs_eps in strengths_abs:
                try:
                    rob_acc = eval_rob(cbm, loader, attack, abs_eps, device)
                    results[f"rob/{abs_eps}"] = rob_acc
                except Exception as e:
                    logger.warning(f"eval_rob failed at eps={abs_eps}: {e}")
                    results[f"rob/{abs_eps}"] = np.nan

    # 6. MIA
    if eval_cfg.compute_mia:
        try:
            from .mia import MIAEvaluation, MIAFeatureConfig

            feature_config = MIAFeatureConfig(**(eval_cfg.mia_features or {}))
            mia_eval = MIAEvaluation(
                feature_config=feature_config,
                adversarial_strength=eval_cfg.mia_adversarial_strength,
                adversarial_num_steps=eval_cfg.mia_adversarial_num_steps,
                adversarial_step_size=eval_cfg.mia_adversarial_step_size,
                adversarial_norm=eval_cfg.mia_adversarial_norm,
            )
            mia_results = mia_eval.evaluate(
                cbm,
                datahandler.classification["train"],
                datahandler.classification["test"],
                device,
            )
            results["mia_accuracy"] = mia_results.attack_accuracy
            results["mia_auc_roc"] = mia_results.auc_roc

            if "correct_prob" in mia_results.feature_names:
                cp_idx = mia_results.feature_names.index("correct_prob")
                results["mia_train_correct_probs"] = mia_results.train_features[:, cp_idx].tolist()
                results["mia_test_correct_probs"] = mia_results.test_features[:, cp_idx].tolist()

            if mia_results.adversarial_worst_case_threshold is not None:
                for feat_name, metrics in mia_results.adversarial_worst_case_threshold.items():
                    results[f"adv_mia_wc/{feat_name}"] = metrics["accuracy"]
                results["adv_mia_wc_best"] = max(
                    m["accuracy"] for m in mia_results.adversarial_worst_case_threshold.values()
                )
                if mia_results.worst_case_threshold:
                    for feat_name, metrics in mia_results.worst_case_threshold.items():
                        results[f"mia_wc/{feat_name}"] = metrics["accuracy"]
                    results["mia_wc_best"] = max(
                        m["accuracy"] for m in mia_results.worst_case_threshold.values()
                    )
        except Exception as e:
            logger.warning(f"MIA evaluation failed: {e}")
            results["mia_accuracy"] = np.nan
            results["mia_auc_roc"] = np.nan

    # 7. UQ
    uq_results = None
    if eval_cfg.compute_uq:
        try:
            from .uq import UQEvaluation, UQConfig

            uq_eval = UQEvaluation(config=UQConfig(**(eval_cfg.uq_config or {})))
            uq_results = uq_eval.evaluate(cbm, datahandler.classification["test"], device)

            results["uq_clean_accuracy"] = uq_results.clean_accuracy
            results["uq_clean_log_px_mean"] = float(uq_results.clean_log_px.mean())

            for eps, acc in uq_results.adv_accuracies.items():
                results[f"uq_adv_acc/{eps}"] = acc
            for (pct, eps), rate in uq_results.detection_rates.items():
                results[f"uq_detection/{pct}pct/{eps}"] = rate
            for (pct, eps), rate in uq_results.err_rate_detected.items():
                results[f"uq_det_err_detected/{pct}pct/{eps}"] = rate
            for (pct, eps), rate in uq_results.err_rate_passed.items():
                results[f"uq_det_err_passed/{pct}pct/{eps}"] = rate
            for (eps, radius), m in uq_results.purification_results.items():
                results[f"uq_purify_acc/{eps}/{radius}"] = m.accuracy_after_purify
                results[f"uq_purify_recovery/{eps}/{radius}"] = m.recovery_rate
            for (eps, n_sweeps), m in uq_results.gibbs_purification_results.items():
                results[f"gibbs_purify_acc/{eps}/{n_sweeps}"] = m.accuracy_after_purify
                results[f"gibbs_purify_recovery/{eps}/{n_sweeps}"] = m.recovery_rate
            for radius, m in uq_results.clean_purification_results.items():
                results[f"uq_clean_purify_acc/{radius}"] = m.accuracy_after_purify
            for n_sweeps, m in uq_results.clean_gibbs_purification_results.items():
                results[f"gibbs_clean_purify_acc/{n_sweeps}"] = m.accuracy_after_purify
        except Exception as e:
            logger.warning(f"UQ evaluation failed: {e}")
            results["uq_clean_accuracy"] = np.nan

    # 7b. Joint-attack UQ: second pass with JOINT_PGD
    if eval_cfg.compute_uq and eval_cfg.joint_uq_config is not None:
        try:
            from .uq import UQEvaluation, UQConfig

            joint_uq_eval = UQEvaluation(config=UQConfig(**eval_cfg.joint_uq_config))
            joint_uq_results = joint_uq_eval.evaluate(
                cbm, datahandler.classification["test"], device
            )
            for eps, acc in joint_uq_results.adv_accuracies.items():
                results[f"uq_joint_adv_acc/{eps}"] = acc
            for (pct, eps), rate in joint_uq_results.detection_rates.items():
                results[f"uq_joint_detection/{pct}pct/{eps}"] = rate
            for (pct, eps), rate in joint_uq_results.err_rate_detected.items():
                results[f"uq_joint_det_err_detected/{pct}pct/{eps}"] = rate
            for (pct, eps), rate in joint_uq_results.err_rate_passed.items():
                results[f"uq_joint_det_err_passed/{pct}pct/{eps}"] = rate
            for (eps, radius), m in joint_uq_results.purification_results.items():
                results[f"uq_joint_purify_acc/{eps}/{radius}"] = m.accuracy_after_purify
                results[f"uq_joint_purify_recovery/{eps}/{radius}"] = m.recovery_rate
            for (eps, n_sweeps), m in joint_uq_results.gibbs_purification_results.items():
                results[f"gibbs_joint_purify_acc/{eps}/{n_sweeps}"] = m.accuracy_after_purify
                results[f"gibbs_joint_purify_recovery/{eps}/{n_sweeps}"] = m.recovery_rate
        except Exception as e:
            logger.warning(f"Joint-attack UQ evaluation failed: {e}")

    # 8. Cleanup
    del cbm
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
        config_keys: Hydra config keys to extract into the DataFrame.

    Returns:
        DataFrame with one row per run. Columns: ``run_name``, ``run_path``,
        ``config/<key>`` for each config_key, then flat metric columns.
    """
    sweep_path = Path(sweep_dir)
    run_dirs = sorted(
        [d for d in sweep_path.iterdir()
         if d.is_dir() and (d / ".hydra" / "config.yaml").exists()],
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

        if config_keys:
            try:
                cfg = load_run_config(run_dir)
                for key in config_keys:
                    try:
                        val = OmegaConf.select(cfg, key)
                        row["config/" + key] = val
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Could not load config for {run_dir.name}: {e}")

        try:
            metrics = evaluate_run(run_dir, eval_cfg)
            row.update(metrics)
        except Exception as e:
            print(f"    FAILED: {e}")
            logger.warning(f"Run {run_dir.name} failed: {e}")

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nEvaluation complete. {len(df)} runs processed.")
    return df
