"""
Single-model post-hoc evaluation.

Loads a saved model from a Hydra run directory and recomputes metrics on the
test split. All results are test-set only.

Result dict key conventions (flat):
    acc                         clean classification accuracy
    dis_loss                    discriminative NLL loss
    gen_loss                    generative (joint) NLL loss
    rob/<eps_rel>               robust accuracy at relative epsilon
    mia_accuracy, mia_auc_roc   membership inference attack
    uq_*                        uncertainty quantification

Budget convention (see "Budget vocabulary" in CLAUDE.md):
    Every budget entering this module is RELATIVE — a fraction of the input domain
    width. That holds for AnalysisConfig.evasion_override["eps_rel"], for the model's
    own evasion config, and for uq_config. Absolute values are derived per model via
    rel_to_abs(eps_rel, range_size_of(cbm)) at the point of use, and metric keys are
    written in relative units.

CLI usage:
    python analysis/run.py <run_dir> [--no-acc] [--no-dis-loss] [--no-gen-loss]
                                      [--no-rob] [--no-mia] [--no-uq]
                                      [--device DEVICE]
"""

import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if "__file__" in dir():
    project_root = Path(__file__).parent.parent
else:
    project_root = Path.cwd().parent
    if not (project_root / "src").exists():
        project_root = Path.cwd()

sys.path.insert(0, str(project_root))

import numpy as np
import torch
from omegaconf import OmegaConf

from analysis.utils.mia_utils import load_run_config, find_model_checkpoint
from src.utils.embeddings import fmt_budget, range_size_of, rel_to_abs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    """Configuration for post-hoc single-run evaluation.

    Attributes:
        compute_acc: Evaluate clean classification accuracy.
        compute_dis_loss: Evaluate discriminative NLL loss.
        compute_gen_loss: Evaluate generative (joint) NLL loss.
        compute_rob: Evaluate adversarial robustness.
        compute_mia: Run membership inference attack evaluation.
        compute_uq: Uncertainty quantification (detection + purification).
        evasion_override: Dict of evasion config fields to override, or None to
            use each run's own config. Budgets are RELATIVE fractions of the input
            domain. Example: {"method": "PGD", "num_steps": 40,
            "eps_rel": [0.05, 0.10, 0.15]}.
        mia_features: Feature toggle dict for MIAFeatureConfig.
        mia_adv_eps_rel: Relative epsilon for adversarial MIA.
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
    mia_adv_eps_rel: Optional[float] = None
    mia_adversarial_num_steps: int = 20
    mia_adversarial_step_size: Optional[float] = None
    mia_adversarial_norm: Any = "inf"
    uq_config: Optional[Dict[str, Any]] = None
    joint_uq_config: Optional[Dict[str, Any]] = None
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_ROB_EPS_REL = [0.05, 0.1]


def _get_rob_params(
    cbm,
    cfg,
    evasion_override: Optional[Dict[str, Any]],
) -> Optional[Tuple[Any, List[float]]]:
    """Build (attack_object, eps_rel_list) for robustness evaluation.

    Budgets are relative everywhere — in the override, in the model's own evasion
    config, and in the fallback. The caller converts to absolute once, per model.
    """
    from src.utils.evasion import EvasionConfig, build_attack

    if evasion_override is not None:
        eps_rel = [float(s) for s in evasion_override.get("eps_rel", [])]
        ec = EvasionConfig(
            method=evasion_override.get("method", "PGD"),
            norm=evasion_override.get("norm", "inf"),
            num_steps=evasion_override.get("num_steps", 10),
            step_size=evasion_override.get("step_size", None),
            random_start=evasion_override.get("random_start", True),
            eps_rel=eps_rel,
        )
    else:
        try:
            raw = OmegaConf.to_container(cfg.trainer.adversarial.evasion, resolve=True)
            ec = EvasionConfig(**raw)
            eps_rel = list(ec.eps_rel)
        except Exception:
            ec = EvasionConfig(method="PGD")
            eps_rel = list(_DEFAULT_ROB_EPS_REL)

    if not eps_rel:
        return None

    return build_attack(ec), eps_rel


# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------

def analyze_run(
    run_dir: Path,
    cfg: AnalysisConfig,
) -> Dict[str, Any]:
    """Load a single run's model and compute metrics post-hoc on the test set.

    Args:
        run_dir: Path to the Hydra output directory for one run.
        cfg: Evaluation configuration.

    Returns:
        Flat dict with keys like ``acc``, ``dis_loss``, ``rob/0.1``,
        ``mia_accuracy``, ``uq_clean_accuracy``, etc.
    """
    from src.model import ConditionalBornMachine
    from src.datahandler import DataHandler
    from src.utils.train import eval_metrics, eval_rob

    run_dir = Path(run_dir)
    device = torch.device(cfg.device)
    results: Dict[str, Any] = {}

    # 1. Load config
    run_cfg = load_run_config(run_dir)

    # 2. Load model
    checkpoint_path = find_model_checkpoint(run_dir)
    cbm = ConditionalBornMachine.load(str(checkpoint_path), accumulate=True)
    cbm.to(device)

    # 3. Load data
    OmegaConf.update(run_cfg, "dataset.overwrite", True, force_add=True)
    datahandler = DataHandler(run_cfg.dataset)
    datahandler.load()
    datahandler.split_and_rescale(cbm)
    datahandler.get_classification_loaders()
    loader = datahandler.classification["test"]

    # 4. Core metrics
    if cfg.compute_acc or cfg.compute_dis_loss or cfg.compute_gen_loss:
        try:
            dis_loss, acc, gen_loss = eval_metrics(cbm, loader, device, progress=True)
            if cfg.compute_acc:
                results["acc"] = acc
            if cfg.compute_dis_loss:
                results["dis_loss"] = dis_loss
            if cfg.compute_gen_loss:
                results["gen_loss"] = gen_loss
        except Exception as e:
            logger.warning(f"eval_metrics failed: {e}")
            if cfg.compute_acc:
                results["acc"] = np.nan
            if cfg.compute_dis_loss:
                results["dis_loss"] = np.nan
            if cfg.compute_gen_loss:
                results["gen_loss"] = np.nan

    # 5. Robustness
    if cfg.compute_rob:
        rob_params = _get_rob_params(cbm, run_cfg, cfg.evasion_override)
        if rob_params is not None:
            attack, eps_rel_list = rob_params
            range_size = range_size_of(cbm)
            for eps_rel in eps_rel_list:
                eps_abs = rel_to_abs(eps_rel, range_size)
                key = f"rob/{fmt_budget(eps_rel)}"
                try:
                    results[key] = eval_rob(cbm, loader, attack, eps_abs, device, progress=True)
                except Exception as e:
                    logger.warning(f"eval_rob failed at eps_rel={eps_rel}: {e}")
                    results[key] = np.nan

    # 6. MIA
    if cfg.compute_mia:
        try:
            from src.analysis.mia import MIAEvaluation, MIAFeatureConfig

            feature_config = MIAFeatureConfig(**(cfg.mia_features or {}))
            # MIA's attack takes an absolute epsilon; convert the authored fraction here.
            mia_eps_abs = (
                None if cfg.mia_adv_eps_rel is None
                else rel_to_abs(cfg.mia_adv_eps_rel, range_size_of(cbm))
            )
            mia_eval = MIAEvaluation(
                feature_config=feature_config,
                adv_eps_abs=mia_eps_abs,
                adversarial_num_steps=cfg.mia_adversarial_num_steps,
                adversarial_step_size=cfg.mia_adversarial_step_size,
                adversarial_norm=cfg.mia_adversarial_norm,
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
    if cfg.compute_uq:
        try:
            from src.analysis.uq import UQEvaluation, UQConfig

            uq_eval = UQEvaluation(config=UQConfig(**(cfg.uq_config or {})))
            uq_results = uq_eval.evaluate(cbm, datahandler.classification["test"], device)

            results["uq_clean_accuracy"] = uq_results.clean_accuracy
            results["uq_clean_log_px_mean"] = float(uq_results.clean_log_px.mean())

            for eps_rel, acc in uq_results.adv_accuracies.items():
                results[f"uq_adv_acc/{fmt_budget(eps_rel)}"] = acc
            for (pct, eps_rel), rate in uq_results.detection_rates.items():
                results[f"uq_detection/{pct}pct/{fmt_budget(eps_rel)}"] = rate
            for (pct, eps_rel), rate in uq_results.err_rate_detected.items():
                results[f"uq_det_err_detected/{pct}pct/{fmt_budget(eps_rel)}"] = rate
            for (pct, eps_rel), rate in uq_results.err_rate_passed.items():
                results[f"uq_det_err_passed/{pct}pct/{fmt_budget(eps_rel)}"] = rate
            for (eps_rel, delta_rel), m in uq_results.purification_results.items():
                results[f"uq_purify_acc/{fmt_budget(eps_rel)}/{fmt_budget(delta_rel)}"] = m.accuracy_after_purify
                results[f"uq_purify_recovery/{fmt_budget(eps_rel)}/{fmt_budget(delta_rel)}"] = m.recovery_rate
            for (eps_rel, n_sweeps), m in uq_results.gibbs_purification_results.items():
                results[f"gibbs_purify_acc/{fmt_budget(eps_rel)}/{n_sweeps}"] = m.accuracy_after_purify
                results[f"gibbs_purify_recovery/{fmt_budget(eps_rel)}/{n_sweeps}"] = m.recovery_rate
            for delta_rel, m in uq_results.clean_purification_results.items():
                results[f"uq_clean_purify_acc/{fmt_budget(delta_rel)}"] = m.accuracy_after_purify
            for n_sweeps, m in uq_results.clean_gibbs_purification_results.items():
                results[f"gibbs_clean_purify_acc/{n_sweeps}"] = m.accuracy_after_purify
        except Exception as e:
            logger.warning(f"UQ evaluation failed: {e}")
            results["uq_clean_accuracy"] = np.nan

    # 7b. Joint-attack UQ
    if cfg.compute_uq and cfg.joint_uq_config is not None:
        try:
            from src.analysis.uq import UQEvaluation, UQConfig

            joint_uq_eval = UQEvaluation(config=UQConfig(**cfg.joint_uq_config))
            joint_uq_results = joint_uq_eval.evaluate(
                cbm, datahandler.classification["test"], device
            )
            for eps_rel, acc in joint_uq_results.adv_accuracies.items():
                results[f"uq_joint_adv_acc/{fmt_budget(eps_rel)}"] = acc
            for (pct, eps_rel), rate in joint_uq_results.detection_rates.items():
                results[f"uq_joint_detection/{pct}pct/{fmt_budget(eps_rel)}"] = rate
            for (pct, eps_rel), rate in joint_uq_results.err_rate_detected.items():
                results[f"uq_joint_det_err_detected/{pct}pct/{fmt_budget(eps_rel)}"] = rate
            for (pct, eps_rel), rate in joint_uq_results.err_rate_passed.items():
                results[f"uq_joint_det_err_passed/{pct}pct/{fmt_budget(eps_rel)}"] = rate
            for (eps_rel, delta_rel), m in joint_uq_results.purification_results.items():
                results[f"uq_joint_purify_acc/{fmt_budget(eps_rel)}/{fmt_budget(delta_rel)}"] = m.accuracy_after_purify
                results[f"uq_joint_purify_recovery/{fmt_budget(eps_rel)}/{fmt_budget(delta_rel)}"] = m.recovery_rate
            for (eps_rel, n_sweeps), m in joint_uq_results.gibbs_purification_results.items():
                results[f"gibbs_joint_purify_acc/{fmt_budget(eps_rel)}/{n_sweeps}"] = m.accuracy_after_purify
                results[f"gibbs_joint_purify_recovery/{fmt_budget(eps_rel)}/{n_sweeps}"] = m.recovery_rate
        except Exception as e:
            logger.warning(f"Joint-attack UQ evaluation failed: {e}")

    # 8. Cleanup
    del cbm
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Post-hoc single-run evaluation. Loads a saved model and "
                    "recomputes metrics on the test set.",
    )
    parser.add_argument("run_dir", help="Path to Hydra run directory (contains .hydra/config.yaml).")
    parser.add_argument("--no-acc", action="store_true", help="Skip clean accuracy.")
    parser.add_argument("--no-dis-loss", action="store_true", help="Skip discriminative NLL loss.")
    parser.add_argument("--no-gen-loss", action="store_true", help="Skip generative NLL loss.")
    parser.add_argument("--no-rob", action="store_true", help="Skip robustness evaluation.")
    parser.add_argument("--no-mia", action="store_true", help="Skip MIA evaluation.")
    parser.add_argument("--no-uq", action="store_true", help="Skip UQ evaluation.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device (default: cuda if available, else cpu).")
    args = parser.parse_args()

    cfg = AnalysisConfig(
        compute_acc=not args.no_acc,
        compute_dis_loss=not args.no_dis_loss,
        compute_gen_loss=not args.no_gen_loss,
        compute_rob=not args.no_rob,
        compute_mia=not args.no_mia,
        compute_uq=not args.no_uq,
        device=args.device,
    )

    print(f"\nEvaluating: {args.run_dir}")
    print(f"Device: {args.device}\n")

    results = analyze_run(Path(args.run_dir), cfg)

    print("=" * 50)
    print("Results")
    print("=" * 50)
    for key, val in sorted(results.items()):
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        elif isinstance(val, list):
            print(f"  {key}: [list of {len(val)} values]")
        else:
            print(f"  {key}: {val}")
