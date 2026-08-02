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
        outputs/spirals/nat/legendre/d10r6c64/seed_sweep/a0_0108 \\
        --config configs/experiments/spirals/at/legendre/d10r6c64/seed_sweep/a0.yaml \\
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


def default_configs_for(sweep_dir: Path) -> List[str]:
    """AT configs that should warm-start from this NAT sweep, derived from its path.

    A static default cannot stay correct now that the AT arm is per-arch: an
    alpha=0 sweep at d4r3c64 must not patch the d10r6c64 AT configs. Given
    ``outputs/{dataset}/nat/{emb}/{arch}/[stage/]{name}_{DDMM}`` this returns the
    AT seed sweeps at the *same* dataset/embedding/arch, plus that embedding's
    shared AT hpo configs. Returns [] when nothing matches, so the caller can say
    so rather than patch the wrong file.
    """
    parts = sweep_dir.resolve().parts
    if "outputs" not in parts:
        return []
    i = len(parts) - 1 - parts[::-1].index("outputs")
    try:
        dataset, _regime, emb, arch = parts[i + 1: i + 5]
    except ValueError:
        return []

    at_root = PROJECT_ROOT / "configs" / "experiments" / dataset / "at" / emb
    found = sorted((at_root / arch / "seed_sweep").glob("*.yaml"))
    found += sorted((at_root / "hpo").glob("*.yaml"))
    found += sorted((at_root / arch / "hpo").glob("*.yaml"))
    return [str(p.relative_to(PROJECT_ROOT)) for p in found]


def _detect_trainer_and_stop_crit(cfg: Dict) -> Tuple[str, str, List]:
    if _get_nested_value(cfg, "trainer.adversarial") is not None:
        trainer = "at"
        stop_crit = _get_nested_value(cfg, "trainer.adversarial.stop_crit") or "rob"
    else:
        trainer = "nat"
        stop_crit = _get_nested_value(cfg, "trainer.nll.stop_crit") or "dis_loss"
    eps_rel = (
        _get_nested_value(cfg, "tracking.evasion.eps_rel")
        or _get_nested_value(cfg, "tracking.evasion.strengths")  # pre-rename runs
        or []
    )
    return trainer, stop_crit, eps_rel


def _metric_for_stop_crit(trainer: str, stop_crit: str, eps_rel: List) -> Tuple[str, bool]:
    if stop_crit == "dis_loss":
        return "dis_loss/valid", True
    if stop_crit == "gen_loss":
        return "gen_loss/valid", True
    if stop_crit == "acc":
        return "acc/valid", False
    if stop_crit == "rob":
        # AT logs "rob/valid/{eps_rel}". Fall back to the bare key for runs that
        # predate the rename; _lookup_metric resolves either form.
        if eps_rel:
            return f"rob/valid/{float(eps_rel[0]):g}", False
        return "rob/valid", False
    return "acc/valid", False


def _lookup_metric(summary: Dict, metric_key: str):
    """Read ``metric_key`` from a run summary, tolerating the rob/valid rename.

    An exact hit wins. Otherwise, if the key is a ``rob/valid`` variant, accept the
    single other variant present so pre- and post-rename runs stay comparable.
    """
    if metric_key in summary:
        return summary[metric_key]
    if not metric_key.startswith("rob/valid"):
        return None
    candidates = [k for k in summary if k == "rob/valid" or k.startswith("rob/valid/")]
    return summary[candidates[0]] if len(candidates) == 1 else None


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
                trainer, stop_crit, eps_rel = _detect_trainer_and_stop_crit(cfg)
                metric_key, minimize = _metric_for_stop_crit(trainer, stop_crit, eps_rel)
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
        val = _lookup_metric(summary, metric_key)
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
        description="Patch model_path in AT configs with the best checkpoint from a seed_sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("sweep_dir", help="Seed_sweep output dir (contains numbered subdirs)")
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        default=None,
        metavar="CONFIG",
        help=(
            "YAML config to patch (repeatable). Defaults to the AT configs at the "
            "same dataset/embedding/arch as SWEEP_DIR."
        ),
    )
    parser.add_argument("--metric", default=None,
                        help="Override metric key (e.g. 'dis_loss/valid')")
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

    config_paths_raw = (
        args.configs if args.configs is not None else default_configs_for(sweep_dir)
    )
    if not config_paths_raw:
        print(f"ERROR: No AT configs found for {sweep_dir}. Pass --config explicitly.")
        sys.exit(1)
    config_paths = []
    for raw in config_paths_raw:
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            print(f"ERROR: Config not found: {p}")
            sys.exit(1)
        config_paths.append(p)

    minimize_override = args.minimize if args.metric else None
    best_idx, model_path_abs = find_best_run(sweep_dir, metric_key=args.metric, minimize=minimize_override)

    # Store as path relative to project root (matching existing config style)
    try:
        model_path = str(model_path_abs.relative_to(PROJECT_ROOT))
    except ValueError:
        model_path = str(model_path_abs)

    for config_path in config_paths:
        patch_config(config_path, model_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
