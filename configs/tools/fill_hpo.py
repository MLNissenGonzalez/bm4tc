#!/usr/bin/env python3
"""
Fill seed_sweep YAML configs from HPO best run.

Queries W&B (with local Hydra output fallback) for the best HPO run per combo
and patches the seed_sweep YAML in-place, replacing `???  # FILL FROM HPO`
placeholders with the actual hyperparameter values.

Usage:
    python configs/tools/fill_hpo.py <dataset> [options]

Examples:
    python configs/tools/fill_hpo.py circles_hard
    python configs/tools/fill_hpo.py moons_hard --training-type adv --dry-run
"""

import argparse
import difflib
import re
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path so we can import from analysis/ and src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.utils.wandb_fetcher import (
    _get_nested_value,
    _load_hydra_config,
    _load_wandb_summary,
    WANDB_AVAILABLE,
)

# =============================================================================
# Constants
# =============================================================================

TRAINING_DIRS = {
    "cls": "classification",
    "gen": "generative",
    "adv": "adversarial",
}

TRAINER_KEYS = {
    "cls": "classification",
    "gen": "generative",
    "adv": "adversarial",
}

# Stage prefix used in W&B metric keys
STAGE_PREFIXES = {
    "cls": "pre",
    "gen": "gen",
    "adv": "adv",
}

# arch string → (in_dim, bond_dim)
ARCH_DIMS = {
    "d4D3": (4, 3),
    "d6D4": (6, 4),
    "d10D6": (10, 6),
    "d30D18": (30, 18),
}

DEFAULT_ENTITY = "martin-nissen-gonzalez-heidelberg-university"
DEFAULT_PROJECT = "gan_train"

# Regex to match `key: ???  # optional comment` lines
FILL_PATTERN = re.compile(
    r'^(\s*)(lr|weight_decay|clean_weight):\s*\?\?\?(\s*(?:#[^\n]*)?)$',
    re.MULTILINE,
)

# Regex for overwrite mode — matches already-filled numeric/float values too
OVERWRITE_PATTERN = re.compile(
    r'^(\s*)(lr|weight_decay|clean_weight):\s*\S+(\s*(?:#[^\n]*)?)$',
    re.MULTILINE,
)


# =============================================================================
# Metric resolution from HPO config
# =============================================================================

def get_metric_info(hpo_cfg: Dict, training_type: str) -> Tuple[str, bool]:
    """Return (wandb_metric_key, minimize) from the HPO config's stop_crit."""
    trainer_key = TRAINER_KEYS[training_type]
    prefix = STAGE_PREFIXES[training_type]
    stop_crit = _get_nested_value(hpo_cfg, f"trainer.{trainer_key}.stop_crit")

    if stop_crit == "clsloss":
        return "pre/valid/clsloss", True
    elif stop_crit == "genloss":
        return "gen/valid/genloss", True
    elif stop_crit == "acc":
        return f"{prefix}/valid/acc", False
    elif stop_crit == "rob":
        strengths = _get_nested_value(hpo_cfg, "tracking.evasion.strengths")
        strength = strengths[0] if isinstance(strengths, list) else strengths
        return f"adv/valid/rob/{strength}", False
    else:
        # Fallback: use accuracy metric for the stage
        return f"{prefix}/valid/acc", False


# =============================================================================
# Parameter extraction (shared by W&B and local paths)
# =============================================================================

def _extract_params(config: Dict, training_type: str) -> Dict[str, Any]:
    """Extract lr, weight_decay (and clean_weight for adv) from a config dict."""
    trainer_key = TRAINER_KEYS[training_type]
    lr = _get_nested_value(config, f"trainer.{trainer_key}.optimizer.kwargs.lr")
    wd = _get_nested_value(config, f"trainer.{trainer_key}.optimizer.kwargs.weight_decay")

    params: Dict[str, Any] = {}
    if lr is not None:
        params["lr"] = lr
    if wd is not None:
        params["weight_decay"] = wd

    if training_type == "adv":
        cw = _get_nested_value(config, "trainer.adversarial.clean_weight")
        if cw is not None:
            params["clean_weight"] = cw

    return params


# =============================================================================
# W&B query
# =============================================================================

def query_wandb(
    dataset: str,
    training_type: str,
    embedding: str,
    arch: str,
    metric_key: str,
    minimize: bool,
    entity: str,
    project: str,
) -> Optional[Dict[str, Any]]:
    """Query W&B for the best finished HPO run matching the combo."""
    if not WANDB_AVAILABLE:
        return None

    import wandb

    in_dim, bond_dim = ARCH_DIMS[arch]
    archinfo = f"d{in_dim}D{bond_dim}{embedding}"

    # Group naming: hpo_{training_type}_{archinfo}_{dataset}_{date}
    group_pattern = f"^hpo_{training_type}_{archinfo}_{dataset}_"

    try:
        api = wandb.Api()
        runs = list(api.runs(
            path=f"{entity}/{project}",
            filters={
                "state": "finished",
                "group": {"$regex": group_pattern},
            },
        ))
    except Exception as e:
        print(f"  [wandb] Error querying runs: {e}")
        return None

    if not runs:
        print(f"  [wandb] No finished runs found for group regex: {group_pattern}")
        return None

    # Sort client-side by metric
    sentinel = float("inf") if minimize else float("-inf")

    def get_metric(run: Any) -> float:
        val = run.summary.get(metric_key)
        return val if val is not None else sentinel

    best_run = min(runs, key=get_metric) if minimize else max(runs, key=get_metric)
    metric_val = best_run.summary.get(metric_key)
    print(f"  [wandb] Best run: {best_run.name} (group={best_run.group}), "
          f"{metric_key}={metric_val:.6g}" if metric_val is not None
          else f"  [wandb] Best run: {best_run.name} (group={best_run.group}), "
               f"{metric_key}=N/A")

    params = _extract_params(best_run.config, training_type)
    if not params:
        print(f"  [wandb] Could not extract params from run config.")
        return None
    return params


# =============================================================================
# Local Hydra fallback
# =============================================================================

def query_local(
    dataset: str,
    training_type: str,
    embedding: str,
    arch: str,
    metric_key: str,
    minimize: bool,
    outputs_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Walk outputs dir to find matching HPO trials; return params from best."""
    in_dim, bond_dim = ARCH_DIMS[arch]
    trainer_key = TRAINER_KEYS[training_type]

    best_metric: Optional[float] = None
    best_params: Optional[Dict[str, Any]] = None

    for config_path in outputs_dir.rglob(".hydra/config.yaml"):
        run_dir = config_path.parent.parent  # .../NNN/

        cfg = _load_hydra_config(run_dir)
        if cfg is None:
            continue

        # Filter: must be an HPO run for this exact combo
        if _get_nested_value(cfg, "experiment") != "hpo":
            continue
        if _get_nested_value(cfg, "dataset.name") != dataset:
            continue
        if _get_nested_value(cfg, "born.embedding") != embedding:
            continue
        if _get_nested_value(cfg, "born.init_kwargs.bond_dim") != bond_dim:
            continue
        if _get_nested_value(cfg, f"trainer.{trainer_key}") is None:
            continue

        summary = _load_wandb_summary(run_dir)
        if summary is None:
            continue

        metric_val = summary.get(metric_key)
        if metric_val is None:
            continue

        is_better = (
            best_metric is None
            or (minimize and metric_val < best_metric)
            or (not minimize and metric_val > best_metric)
        )
        if is_better:
            best_metric = metric_val
            best_params = _extract_params(cfg, training_type)

    if best_params is not None:
        print(f"  [local] Best trial: {metric_key}={best_metric:.6g}")
    else:
        print(f"  [local] No matching trials found in {outputs_dir}")

    return best_params


# =============================================================================
# YAML patching
# =============================================================================

def _format_value(val: Any) -> str:
    """Format a value for YAML output (preserves float precision)."""
    if isinstance(val, float):
        # repr gives the shortest exact round-trip representation
        return repr(val)
    return str(val)


def patch_yaml(content: str, params: Dict[str, Any], overwrite: bool = False) -> Tuple[str, int]:
    """
    Replace `key: ???  # comment` lines with actual values.
    In overwrite mode, also replaces already-filled values for the same keys.
    Only replaces keys present in params. Returns (new_content, num_replaced).
    """
    pattern = OVERWRITE_PATTERN if overwrite else FILL_PATTERN
    count = 0

    def replacer(m: re.Match) -> str:
        nonlocal count
        indent = m.group(1)
        key = m.group(2)
        comment = m.group(3)

        if key not in params:
            return m.group(0)  # Leave unchanged

        count += 1
        return f"{indent}{key}: {_format_value(params[key])}{comment}"

    new_content = pattern.sub(replacer, content)
    return new_content, count


# =============================================================================
# Per-combo processing
# =============================================================================

def process_combo(
    dataset: str,
    training_type: str,
    embedding: str,
    arch: str,
    source: str,
    entity: str,
    project: str,
    outputs_dir: Path,
    configs_root: Path,
    dry_run: bool,
    overwrite: bool = False,
) -> bool:
    """Process one combo. Returns True if successfully patched (or dry-run patched)."""
    training_dir = TRAINING_DIRS[training_type]
    base = configs_root / "experiments" / training_dir / embedding / arch

    hpo_cfg_path = base / "hpo" / f"{dataset}.yaml"
    seed_cfg_path = base / "seed_sweeps" / f"{dataset}.yaml"

    # Skip silently if either config file is absent
    if not hpo_cfg_path.exists() or not seed_cfg_path.exists():
        return False

    print(f"\n[{training_type}/{embedding}/{arch}/{dataset}]")

    # Load HPO config to determine the metric
    with open(hpo_cfg_path, "r") as f:
        hpo_cfg = yaml.safe_load(f)

    metric_key, minimize = get_metric_info(hpo_cfg, training_type)
    print(f"  Metric: {metric_key} ({'minimize' if minimize else 'maximize'})")

    # Query sources in priority order
    params: Optional[Dict[str, Any]] = None

    if source in ("wandb", "both"):
        params = query_wandb(
            dataset, training_type, embedding, arch,
            metric_key, minimize, entity, project,
        )

    if params is None and source in ("local", "both"):
        params = query_local(
            dataset, training_type, embedding, arch,
            metric_key, minimize, outputs_dir,
        )

    if not params:
        print("  WARNING: Could not find HPO results — skipping.")
        return False

    print(f"  Params: {params}")

    # Read and patch the seed_sweep YAML
    with open(seed_cfg_path, "r") as f:
        content = f.read()

    new_content, count = patch_yaml(content, params, overwrite=overwrite)

    if count == 0:
        if overwrite:
            print("  WARNING: No matching keys found (lr/weight_decay/clean_weight) — wrong file?")
        else:
            print("  WARNING: No ??? placeholders found (already filled or wrong file).")
        return False

    rel_path = seed_cfg_path.relative_to(PROJECT_ROOT)
    print(f"  Patching {count} placeholder(s) in {rel_path}")

    if dry_run:
        diff_lines = list(difflib.unified_diff(
            content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=str(rel_path),
            tofile=str(rel_path) + " (patched)",
        ))
        if diff_lines:
            print("  Diff:")
            for line in diff_lines:
                print("   ", line, end="")
            print()
    else:
        with open(seed_cfg_path, "w") as f:
            f.write(new_content)

    return True


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill seed_sweep configs from HPO best run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset", help="Dataset name (e.g., circles_hard)")
    parser.add_argument(
        "--training-type",
        nargs="+",
        choices=["cls", "gen", "adv"],
        default=["cls", "gen", "adv"],
        metavar="TYPE",
        help="Training type(s) to process (default: all three)",
    )
    parser.add_argument(
        "--embedding",
        nargs="+",
        choices=["fourier", "legendre", "hermite"],
        default=["fourier", "legendre", "hermite"],
        help="Embedding(s) to process (default: all three)",
    )
    parser.add_argument(
        "--arch",
        nargs="+",
        choices=["d4D3", "d6D4", "d10D6", "d30D18"],
        default=["d4D3", "d6D4", "d10D6", "d30D18"],
        help="Architecture(s) to process (default: both)",
    )
    parser.add_argument(
        "--source",
        choices=["wandb", "local", "both"],
        default="both",
        help="Where to look for HPO results (default: both, wandb first)",
    )
    parser.add_argument(
        "--entity",
        default=DEFAULT_ENTITY,
        help="W&B entity (default: %(default)s)",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="W&B project (default: %(default)s)",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Local Hydra outputs root (default: outputs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print diff without writing files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already-filled values (default: skip files with no ??? placeholders)",
    )

    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.is_absolute():
        outputs_dir = PROJECT_ROOT / outputs_dir

    configs_root = PROJECT_ROOT / "configs"

    total = 0
    patched = 0

    for training_type in args.training_type:
        for embedding in args.embedding:
            for arch in args.arch:
                total += 1
                success = process_combo(
                    dataset=args.dataset,
                    training_type=training_type,
                    embedding=embedding,
                    arch=arch,
                    source=args.source,
                    entity=args.entity,
                    project=args.project,
                    outputs_dir=outputs_dir,
                    configs_root=configs_root,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                )
                if success:
                    patched += 1

    suffix = " (dry run)" if args.dry_run else ""
    print(f"\nDone. Patched {patched}/{total} combos{suffix}.")


if __name__ == "__main__":
    main()
