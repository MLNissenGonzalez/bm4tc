#!/usr/bin/env python3
"""
Find the best-metric run in a seed_sweep output dir and patch model_path
in a target YAML config.

Auto-detects the selection metric from the sweep's .hydra/config.yaml
(trainer type + stop_crit). Reads wandb-summary.json from each numbered
run subdir to select the best run.

Usage:
    python tools/patch_checkpoint.py SWEEP_DIR [--config CONFIG] [--dry-run]

Example:
    python tools/patch_checkpoint.py \\
        outputs/spirals/nat/legendre/d10r6/seed_sweep_a0_0601 \\
        --config configs/experiments/spirals/at/legendre/d10r6/seed_sweep.yaml \\
        --dry-run
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.utils.wandb_fetcher import (
    _get_nested_value,
    _load_hydra_config,
    _load_wandb_summary,
)

DEFAULT_CONFIG = "configs/experiments/spirals/at/legendre/d10r6/seed_sweep.yaml"


def _detect_trainer_and_stop_crit(cfg: Dict) -> Tuple[str, str, List]:
    if _get_nested_value(cfg, "trainer.adversarial") is not None:
        trainer = "at"
        stop_crit = _get_nested_value(cfg, "trainer.adversarial.stop_crit") or "rob"
    else:
        trainer = "nat"
        stop_crit = _get_nested_value(cfg, "trainer.nll.stop_crit") or "dis_loss"
    strengths = _get_nested_value(cfg, "tracking.evasion.strengths") or []
    return trainer, stop_crit, strengths


def _metric_for_stop_crit(trainer: str, stop_crit: str, strengths: List) -> Tuple[str, bool]:
    if stop_crit == "dis_loss":
        return "dis/valid/dis_loss", True
    if stop_crit == "gen_loss":
        return "gen/valid/gen_loss", True
    if stop_crit == "acc":
        prefix = "dis" if trainer == "nat" else "adv"
        return f"{prefix}/valid/acc", False
    if stop_crit == "rob":
        strength = strengths[0] if strengths else 0.1
        return f"adv/valid/rob/{strength}", False
    prefix = "dis" if trainer == "nat" else "adv"
    return f"{prefix}/valid/acc", False


def find_best_run(
    sweep_dir: Path,
    metric_key: Optional[str] = None,
    minimize: Optional[bool] = None,
) -> Tuple[int, Path]:
    """Return (best_run_index, model_path) for the best run in sweep_dir."""
    run_dirs = sorted(
        [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not run_dirs:
        raise ValueError(f"No numbered subdirectories found in {sweep_dir}")

    if metric_key is None:
        for rd in run_dirs:
            cfg = _load_hydra_config(rd)
            if cfg is not None:
                trainer, stop_crit, strengths = _detect_trainer_and_stop_crit(cfg)
                metric_key, minimize = _metric_for_stop_crit(trainer, stop_crit, strengths)
                print(f"Auto-detected: trainer={trainer!r}, stop_crit={stop_crit!r}")
                print(f"  metric: {metric_key} ({'minimize' if minimize else 'maximize'})")
                break
        else:
            print("WARNING: No .hydra/config.yaml found; defaulting to dis/valid/dis_loss (minimize).")
            metric_key = "dis/valid/dis_loss"
            minimize = True

    if minimize is None:
        minimize = False

    scores: Dict[int, float] = {}
    for rd in run_dirs:
        summary = _load_wandb_summary(rd)
        if summary is None:
            continue
        val = summary.get(metric_key)
        if val is not None:
            scores[int(rd.name)] = float(val)

    if not scores:
        print(f"WARNING: No runs have '{metric_key}' in wandb-summary.json. Falling back to run 0.")
        best_idx = 0
    else:
        best_idx = min(scores, key=scores.__getitem__) if minimize else max(scores, key=scores.__getitem__)
        print(f"Best run: {best_idx}  ({metric_key}={scores[best_idx]:.6g})")
        print(f"All scores: { {k: f'{v:.4g}' for k, v in sorted(scores.items())} }")

    return best_idx, sweep_dir / str(best_idx) / "models" / "model"


def patch_config(config_path: Path, model_path: str, dry_run: bool = False) -> bool:
    """Replace the model_path: line in config_path. Returns True if changed."""
    content = config_path.read_text()
    pattern = re.compile(r'^(model_path:\s*).*$', re.MULTILINE)
    m = pattern.search(content)
    if m is None:
        print(f"WARNING: No 'model_path:' key found in {config_path}. Nothing patched.")
        return False

    new_content = pattern.sub(rf'\g<1>{model_path}', content)
    print(f"\n  - {m.group(0)}")
    print(f"  + model_path: {model_path}")

    if dry_run:
        print("\n(dry run — file not written)")
    else:
        config_path.write_text(new_content)
        print(f"\nPatched: {config_path.relative_to(PROJECT_ROOT)}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch model_path in an AT config with the best checkpoint from a seed_sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("sweep_dir", help="Seed_sweep output dir (contains numbered subdirs)")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"YAML config to patch (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--metric", default=None,
                        help="Override metric key (e.g. 'dis/valid/dis_loss')")
    parser.add_argument("--minimize", action="store_true",
                        help="Minimize the metric (only used with --metric)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_absolute():
        sweep_dir = PROJECT_ROOT / sweep_dir
    if not sweep_dir.exists():
        print(f"ERROR: Sweep dir not found: {sweep_dir}")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    minimize_override = args.minimize if args.metric else None
    best_idx, model_path_abs = find_best_run(sweep_dir, metric_key=args.metric, minimize=minimize_override)

    # Store as path relative to project root (matching existing config style)
    try:
        model_path = str(model_path_abs.relative_to(PROJECT_ROOT))
    except ValueError:
        model_path = str(model_path_abs)

    patch_config(config_path, model_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
