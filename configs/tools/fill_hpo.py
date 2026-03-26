#!/usr/bin/env python3
"""
Fill seed_sweep YAML configs from HPO best run.

Queries W&B (with local Hydra output fallback) for the best HPO run per combo
and patches the seed_sweep YAML in-place, replacing `???  # FILL FROM HPO`
placeholders with the actual hyperparameter values.

Usage:
    python configs/tools/fill_hpo.py --list
    python configs/tools/fill_hpo.py --dry-run
    python configs/tools/fill_hpo.py --filter-type adv --dry-run
    python configs/tools/fill_hpo.py --filter-dataset circles --filter-embedding legendre
    python configs/tools/fill_hpo.py --force
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

VALID_TYPES = ["adversarial", "classification", "generative"]

TYPE_SHORT = {
    "cls": "classification",
    "adv": "adversarial",
    "gen": "generative",
}

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

def parse_arch(arch: str):
    """'d3D10' → (3, 10, '')  |  'd3D10c64' → (3, 10, 'c64')"""
    m = re.match(r"d(\d+)D(\d+)(c\d+)?$", arch)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3) or ""
    return None, None, ""

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

    in_dim, bond_dim, dtype_suffix = parse_arch(arch)
    archinfo = f"d{in_dim}D{bond_dim}{dtype_suffix}{embedding}"

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
    in_dim, bond_dim, dtype_suffix = parse_arch(arch)
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
        _expected_dtype = {"c64": "complex64", "c128": "complex128"}.get(dtype_suffix)
        if _get_nested_value(cfg, "born.init_kwargs.dtype") != _expected_dtype:
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
# Discovery
# =============================================================================

def discover_combos(configs_root: Path) -> List[Dict]:
    """Discover all combos that have both hpo/ and seed_sweeps/ yamls."""
    combos = []
    for typ in VALID_TYPES:
        type_dir = configs_root / "experiments" / typ
        if not type_dir.is_dir():
            continue
        type_short = next(k for k, v in TRAINING_DIRS.items() if v == typ)
        for embedding_dir in sorted(type_dir.iterdir()):
            if not embedding_dir.is_dir():
                continue
            embedding = embedding_dir.name
            for arch_dir in sorted(embedding_dir.iterdir()):
                if not arch_dir.is_dir():
                    continue
                arch = arch_dir.name
                hpo_dir = arch_dir / "hpo"
                seed_dir = arch_dir / "seed_sweeps"
                if not hpo_dir.is_dir() or not seed_dir.is_dir():
                    continue
                for hpo_path in sorted(hpo_dir.glob("*.yaml")):
                    dataset = hpo_path.stem
                    seed_path = seed_dir / f"{dataset}.yaml"
                    if not seed_path.exists():
                        continue
                    combos.append({
                        "type":       typ,
                        "type_short": type_short,
                        "embedding":  embedding,
                        "arch":       arch,
                        "dataset":    dataset,
                        "hpo_path":   hpo_path,
                        "seed_path":  seed_path,
                    })
    return combos


def is_filled(seed_path: Path) -> bool:
    """Return True if the seed_sweep YAML has no ??? placeholders."""
    return not bool(FILL_PATTERN.search(seed_path.read_text()))


# =============================================================================
# Per-combo processing
# =============================================================================

def process_combo(
    combo: Dict,
    source: str,
    entity: str,
    project: str,
    outputs_dir: Path,
    dry_run: bool,
    force: bool = False,
) -> bool:
    """Process one combo. Returns True if successfully patched (or dry-run patched)."""
    training_type = combo["type_short"]
    embedding     = combo["embedding"]
    arch          = combo["arch"]
    dataset       = combo["dataset"]
    hpo_cfg_path  = combo["hpo_path"]
    seed_cfg_path = combo["seed_path"]

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

    new_content, count = patch_yaml(content, params, overwrite=force)

    if count == 0:
        if force:
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
    parser.add_argument("--filter-dataset", metavar="DS",
                        help="Substring match on dataset name (e.g. circles).")
    parser.add_argument("--filter-type", metavar="TYPE",
                        help="cls | adv | gen (or full name).")
    parser.add_argument("--filter-embedding", metavar="EMB",
                        help="fourier | legendre | hermite | chebychev1 | chebychev2")
    parser.add_argument("--filter-arch", metavar="ARCH",
                        help="e.g. d4D3, d10D6")
    parser.add_argument("--source", choices=["wandb", "local", "both"], default="both",
                        help="Where to look for HPO results (default: both, wandb first)")
    parser.add_argument("--entity", default=DEFAULT_ENTITY,
                        help="W&B entity (default: %(default)s)")
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help="W&B project (default: %(default)s)")
    parser.add_argument("--outputs-dir", default="outputs",
                        help="Local Hydra outputs root (default: outputs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print diff without writing files")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite already-filled values")
    parser.add_argument("--list", action="store_true",
                        help="Print status of all discovered combos and exit.")

    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.is_absolute():
        outputs_dir = PROJECT_ROOT / outputs_dir

    configs_root = PROJECT_ROOT / "configs"

    combos = discover_combos(configs_root)
    if not combos:
        print("No HPO combos found under configs/experiments/.")
        return

    if args.list:
        for c in combos:
            status = "OK " if is_filled(c["seed_path"]) else "   "
            print(f"[{status}] {c['type_short']}/{c['embedding']}/{c['arch']}/{c['dataset']}")
        return

    # Apply filters
    if args.filter_type:
        typ = TYPE_SHORT.get(args.filter_type, args.filter_type)
        combos = [c for c in combos if c["type"] == typ]
    if args.filter_embedding:
        combos = [c for c in combos if c["embedding"] == args.filter_embedding]
    if args.filter_arch:
        combos = [c for c in combos if c["arch"] == args.filter_arch]
    if args.filter_dataset:
        combos = [c for c in combos if args.filter_dataset in c["dataset"]]

    todo    = [c for c in combos if args.force or not is_filled(c["seed_path"])]
    skipped = len(combos) - len(todo)

    if skipped:
        print(f"Skipping {skipped} already-filled combo(s).")
    if not todo:
        print("Nothing to do. Use --force to re-fill already-filled combos.")
        return

    total   = len(todo)
    patched = 0
    for combo in todo:
        try:
            success = process_combo(
                combo=combo,
                source=args.source,
                entity=args.entity,
                project=args.project,
                outputs_dir=outputs_dir,
                dry_run=args.dry_run,
                force=args.force,
            )
            if success:
                patched += 1
        except Exception as e:
            label = f"{combo['type_short']}/{combo['embedding']}/{combo['arch']}/{combo['dataset']}"
            print(f"  WARNING: Exception processing {label}: {e}")

    suffix = " (dry run)" if args.dry_run else ""
    print(f"\nDone. Patched {patched}/{total} combos{suffix}.")


if __name__ == "__main__":
    main()
