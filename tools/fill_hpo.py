#!/usr/bin/env python3
"""
Fill seed_sweep YAML configs from HPO best run.

Queries W&B (with local Hydra output fallback) for the best HPO run per combo
and patches the corresponding seed_sweep YAML in-place, replacing
`???  # FILL FROM HPO` placeholders with actual hyperparameter values.

Config layout (new structure):
  configs/experiments/{dataset}/{nat|at}/{embedding}/{arch}/{kind}.yaml

HPO kinds end in "hpo" (e.g. hpo_a0, hpo_a1, hpo_a05, hpo).
Matching seed_sweep kinds replace "hpo" with "seed_sweep" in the stem
(e.g. hpo_a0 → seed_sweep_a0).

Usage:
    python tools/fill_hpo.py --list
    python tools/fill_hpo.py --dry-run
    python tools/fill_hpo.py --trainer at --dry-run
    python tools/fill_hpo.py --dataset circles --embedding legendre
    python tools/fill_hpo.py --force
"""

import argparse
import difflib
import re
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.paths import data_root as _data_root

from analysis.utils.wandb_fetcher import (
    _get_nested_value,
    _load_hydra_config,
    _load_wandb_summary,
    WANDB_AVAILABLE,
)

# =============================================================================
# Constants
# =============================================================================

VALID_TRAINERS = ["nat", "at"]

DEFAULT_ENTITY = "martin-nissen-gonzalez-heidelberg-university"
DEFAULT_PROJECT = "bm4tc"

# Regex to match `key: ???  # optional comment` lines
FILL_PATTERN = re.compile(
    r'^(\s*)(lr|weight_decay|clean_weight|soft_strength):\s*\?\?\?(\s*(?:#[^\n]*)?)$',
    re.MULTILINE,
)

# Regex for overwrite mode — also matches already-filled numeric values
OVERWRITE_PATTERN = re.compile(
    r'^(\s*)(lr|weight_decay|clean_weight|soft_strength):\s*\S+(\s*(?:#[^\n]*)?)$',
    re.MULTILINE,
)

# Maps trainer type to the Hydra config key for trainer params
TRAINER_CONFIG_KEY = {
    "nat": "nll",
    "at":  "adversarial",
}


# =============================================================================
# Kind pairing: HPO → seed_sweep
# =============================================================================

def _seed_kind_for_hpo(hpo_kind: str) -> str:
    """Return the matching seed_sweep kind name for an HPO kind name.

    Examples:
        hpo_a0  → seed_sweep_a0
        hpo_a1  → seed_sweep_a1
        hpo_a05 → seed_sweep_a05
        hpo     → seed_sweep
        hpo_linf → seed_sweep_linf
    """
    if hpo_kind.startswith("hpo"):
        suffix = hpo_kind[len("hpo"):]  # e.g. "", "_a0", "_a1", "_linf"
        return "seed_sweep" + suffix
    return hpo_kind  # fallback


def _seed_kind_for_hpo_staged(hpo_kind: str) -> str:
    """Map a staged (bare) HPO kind to its matching seed_sweep kind.

    In the staged layout the stage lives in the folder, so kinds are bare:
        a0            → a0
        pretrained_a1 → a1   (warm-start HPO informs the alpha=1 seed sweep)
        pretrained_a05 → a05
    """
    if hpo_kind.startswith("pretrained_"):
        return hpo_kind[len("pretrained_"):]
    return hpo_kind


# =============================================================================
# Metric resolution from HPO config
# =============================================================================

def get_metric_info(hpo_cfg: Dict, trainer: str) -> Tuple[str, bool]:
    """Return (wandb_metric_key, minimize) from the HPO config's stop_crit."""
    trainer_key = TRAINER_CONFIG_KEY[trainer]
    stop_crit = _get_nested_value(hpo_cfg, f"trainer.{trainer_key}.stop_crit")

    if stop_crit == "dis_loss":
        return "dis_loss/valid", True
    elif stop_crit == "gen_loss":
        return "gen_loss/valid", True
    elif stop_crit == "mixed_loss":
        return "mixed_loss/valid", True
    elif stop_crit == "acc":
        return "acc/valid", False
    elif stop_crit == "rob":
        # AT logs "rob/valid/{eps_rel}"; fall back to the bare key for runs that
        # predate the rename. Lookups go through _lookup_metric, which accepts either.
        eps_rel = (
            _get_nested_value(hpo_cfg, f"trainer.{trainer_key}.evasion.eps_rel")
            or _get_nested_value(hpo_cfg, f"trainer.{trainer_key}.evasion.strengths")
            or []
        )
        if eps_rel:
            return f"rob/valid/{float(eps_rel[0]):g}", False
        return "rob/valid", False
    else:
        return "acc/valid", False


# =============================================================================
# Parameter extraction
# =============================================================================

def _extract_params(config: Dict, trainer: str) -> Dict[str, Any]:
    """Extract lr, weight_decay (and clean_weight for at) from a config dict."""
    trainer_key = TRAINER_CONFIG_KEY[trainer]
    lr = _get_nested_value(config, f"trainer.{trainer_key}.optimizer.kwargs.lr")
    wd = _get_nested_value(config, f"trainer.{trainer_key}.optimizer.kwargs.weight_decay")
    soft = _get_nested_value(config, f"trainer.{trainer_key}.norm_control.soft_strength")

    params: Dict[str, Any] = {}
    if lr is not None:
        params["lr"] = lr
    if wd is not None:
        params["weight_decay"] = wd
    if soft is not None:
        params["soft_strength"] = soft

    if trainer == "at":
        cw = _get_nested_value(config, "trainer.adversarial.clean_weight")
        if cw is not None:
            params["clean_weight"] = cw

    return params


# =============================================================================
# W&B query
# =============================================================================

def _lookup_metric(summary: Dict, metric_key: str) -> Optional[float]:
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


def _best_metric_from_history(run: Any, metric_key: str, minimize: bool) -> Optional[float]:
    """Best (min if ``minimize`` else max) value of ``metric_key`` over the run's
    full training history, or None if the metric was never logged.

    The run summary holds only the *last* logged value. With early stopping the
    run is selected on its best epoch, not its final one, so we scan the full
    step history for the true best rather than trusting ``run.summary``.
    """
    best: Optional[float] = None
    for row in run.scan_history(keys=[metric_key]):
        val = row.get(metric_key)
        if not isinstance(val, (int, float)):
            continue
        if best is None or (val < best if minimize else val > best):
            best = val
    return best


def query_wandb(
    dataset: str,
    trainer: str,
    embedding: str,
    arch: str,
    hpo_kind: str,
    metric_key: str,
    minimize: bool,
    entity: str,
    project: str,
    hpo_stage: str = "",
) -> Optional[Dict[str, Any]]:
    """Query W&B for the best finished HPO run matching the combo.

    W&B group format: {dataset}/{trainer}/{embedding}/{arch}/[{stage}/]{hpo_kind}{sep}{date}
    where {sep} is "/" (older runs) or "_" (newer runs, e.g. hpo_a0_2506).
    We match by prefix ending at that separator so, e.g., hpo_a0 does not also
    match hpo_a05.
    """
    if not WANDB_AVAILABLE:
        return None

    import wandb

    stage_seg = f"{hpo_stage}/" if hpo_stage else ""
    group_base = f"{dataset}/{trainer}/{embedding}/{arch}/{stage_seg}{hpo_kind}"
    group_prefix = group_base + "/"
    group_pattern = f"^{re.escape(group_base)}[/_]"

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
        print(f"  [wandb] No finished runs found for group prefix: {group_prefix}")
        return None

    # Rank by the best value reached during training (not the last-logged
    # summary value), falling back to summary only if history lacks the metric.
    scored: List[Tuple[float, Any]] = []
    for run in runs:
        metric_val = _best_metric_from_history(run, metric_key, minimize)
        if metric_val is None:
            metric_val = _lookup_metric(run.summary, metric_key)
        if isinstance(metric_val, (int, float)):
            scored.append((metric_val, run))

    if not scored:
        print(f"  [wandb] No runs logged metric {metric_key} for: {group_prefix}")
        return None

    best_metric, best_run = (min if minimize else max)(scored, key=lambda x: x[0])
    print(
        f"  [wandb] Best run: {best_run.name} (group={best_run.group}), "
        f"best {metric_key}={best_metric:.6g}"
    )

    # Slim run objects from api.runs() frequently carry an empty .config; fetch
    # the full run so hyperparameters are read reliably.
    try:
        config = api.run(f"{entity}/{project}/{best_run.id}").config
    except Exception as e:
        print(f"  [wandb] Error fetching full run config ({e}); using slim config.")
        config = best_run.config

    params = _extract_params(config, trainer)
    if not params:
        print("  [wandb] Could not extract params from run config.")
        return None
    return params


# =============================================================================
# Local Hydra fallback
# =============================================================================

def query_local(
    dataset: str,
    trainer: str,
    embedding: str,
    arch: str,
    hpo_kind: str,
    metric_key: str,
    minimize: bool,
    outputs_dir: Path,
    hpo_stage: str = "",
) -> Optional[Dict[str, Any]]:
    """Walk outputs dir to find matching HPO trials; return params from best.

    NOTE: this fallback ranks trials by the *last-epoch* value in each run's
    local ``wandb-summary.json`` — training history is not available on disk, so
    (unlike ``query_wandb``) it cannot select on the best-over-training value.
    Prefer ``--source wandb`` when the runs are on W&B.
    """
    trainer_key = TRAINER_CONFIG_KEY[trainer]

    best_metric: Optional[float] = None
    best_params: Optional[Dict[str, Any]] = None

    # Search under outputs/{dataset}/{trainer}/{embedding}/{arch}/[{stage}/]{hpo_kind}_*/
    search_root = outputs_dir / dataset / trainer / embedding / arch
    if hpo_stage:
        search_root = search_root / hpo_stage
    if not search_root.exists():
        print(f"  [local] No output dir found: {search_root}")
        return None

    for run_dir_candidate in search_root.glob(f"{hpo_kind}_*"):
        for config_path in run_dir_candidate.rglob(".hydra/config.yaml"):
            run_dir = config_path.parent.parent

            cfg = _load_hydra_config(run_dir)
            if cfg is None:
                continue

            # Basic filters
            if _get_nested_value(cfg, "dataset.name") != dataset:
                continue
            if _get_nested_value(cfg, "born.embedding") != embedding:
                continue
            if _get_nested_value(cfg, f"trainer.{trainer_key}") is None:
                continue

            summary = _load_wandb_summary(run_dir)
            if summary is None:
                continue

            metric_val = _lookup_metric(summary, metric_key)
            if metric_val is None:
                continue

            is_better = (
                best_metric is None
                or (minimize and metric_val < best_metric)
                or (not minimize and metric_val > best_metric)
            )
            if is_better:
                best_metric = metric_val
                best_params = _extract_params(cfg, trainer)

    if best_params is not None:
        print(f"  [local] Best trial: {metric_key}={best_metric:.6g}")
    else:
        print(f"  [local] No matching trials found under {search_root}/{hpo_kind}_*")

    return best_params


# =============================================================================
# YAML patching
# =============================================================================

def _format_value(val: Any) -> str:
    if isinstance(val, float):
        return repr(val)
    return str(val)


def patch_yaml(content: str, params: Dict[str, Any], overwrite: bool = False) -> Tuple[str, int]:
    """Replace placeholder lines with actual values. Returns (new_content, num_replaced)."""
    pattern = OVERWRITE_PATTERN if overwrite else FILL_PATTERN
    count = 0

    def replacer(m: re.Match) -> str:
        nonlocal count
        indent = m.group(1)
        key = m.group(2)
        comment = m.group(3)
        if key not in params:
            return m.group(0)
        count += 1
        return f"{indent}{key}: {_format_value(params[key])}{comment}"

    return pattern.sub(replacer, content), count


# =============================================================================
# Discovery
# =============================================================================

def discover_combos(configs_root: Path) -> List[Dict]:
    """Discover all (hpo_kind, seed_kind) pairs under the new config structure."""
    combos = []

    experiments_root = configs_root / "experiments"
    for dataset_dir in sorted(experiments_root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name == "tests":
            continue
        dataset = dataset_dir.name

        for trainer_dir in sorted(dataset_dir.iterdir()):
            if not trainer_dir.is_dir() or trainer_dir.name not in VALID_TRAINERS:
                continue
            trainer = trainer_dir.name

            for embedding_dir in sorted(trainer_dir.iterdir()):
                if not embedding_dir.is_dir():
                    continue
                embedding = embedding_dir.name

                for arch_dir in sorted(embedding_dir.iterdir()):
                    if not arch_dir.is_dir():
                        continue
                    arch = arch_dir.name

                    hpo_stage_dir = arch_dir / "hpo"
                    if hpo_stage_dir.is_dir():
                        # Staged layout: hpo/{kind}.yaml ↔ seed_sweep/{kind}.yaml
                        seed_stage_dir = arch_dir / "seed_sweep"
                        for hpo_path in sorted(hpo_stage_dir.glob("*.yaml")):
                            hpo_kind = hpo_path.stem
                            seed_kind = _seed_kind_for_hpo_staged(hpo_kind)
                            seed_path = seed_stage_dir / f"{seed_kind}.yaml"
                            if not seed_path.exists():
                                continue
                            combos.append({
                                "dataset":    dataset,
                                "trainer":    trainer,
                                "embedding":  embedding,
                                "arch":       arch,
                                "hpo_stage":  "hpo",
                                "hpo_kind":   hpo_kind,
                                "seed_kind":  seed_kind,
                                "hpo_path":   hpo_path,
                                "seed_path":  seed_path,
                            })
                        continue

                    # Legacy flat layout: hpo_*.yaml ↔ seed_sweep_*.yaml
                    for hpo_path in sorted(arch_dir.glob("hpo*.yaml")):
                        hpo_kind = hpo_path.stem
                        seed_kind = _seed_kind_for_hpo(hpo_kind)
                        seed_path = arch_dir / f"{seed_kind}.yaml"

                        if not seed_path.exists():
                            continue

                        combos.append({
                            "dataset":   dataset,
                            "trainer":   trainer,
                            "embedding": embedding,
                            "arch":      arch,
                            "hpo_stage": "",
                            "hpo_kind":  hpo_kind,
                            "seed_kind": seed_kind,
                            "hpo_path":  hpo_path,
                            "seed_path": seed_path,
                        })

    return combos


def is_filled(seed_path: Path) -> bool:
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
    """Process one combo. Returns True if successfully patched."""
    dataset   = combo["dataset"]
    trainer   = combo["trainer"]
    embedding = combo["embedding"]
    arch      = combo["arch"]
    hpo_kind  = combo["hpo_kind"]
    hpo_stage = combo.get("hpo_stage", "")
    seed_path = combo["seed_path"]

    label = f"{dataset}/{trainer}/{embedding}/{arch}: {hpo_kind} → {combo['seed_kind']}"
    print(f"\n[{label}]")

    with open(combo["hpo_path"]) as f:
        hpo_cfg = yaml.safe_load(f)

    metric_key, minimize = get_metric_info(hpo_cfg, trainer)
    print(f"  Metric: {metric_key} ({'minimize' if minimize else 'maximize'})")

    params: Optional[Dict[str, Any]] = None

    if source in ("wandb", "both"):
        params = query_wandb(
            dataset, trainer, embedding, arch, hpo_kind,
            metric_key, minimize, entity, project, hpo_stage,
        )

    if params is None and source in ("local", "both"):
        params = query_local(
            dataset, trainer, embedding, arch, hpo_kind,
            metric_key, minimize, outputs_dir, hpo_stage,
        )

    if not params:
        print("  WARNING: Could not find HPO results — skipping.")
        return False

    print(f"  Params: {params}")

    with open(seed_path) as f:
        content = f.read()

    new_content, count = patch_yaml(content, params, overwrite=force)

    if count == 0:
        if force:
            print("  WARNING: No matching keys found — wrong file?")
        else:
            print("  WARNING: No ??? placeholders found (already filled or wrong file).")
        return False

    rel_path = seed_path.relative_to(PROJECT_ROOT)
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
        with open(seed_path, "w") as f:
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
    parser.add_argument("--dataset", metavar="DS",
                        help="Substring match on dataset name.")
    parser.add_argument("--trainer", metavar="TRAINER",
                        help="nat | at")
    parser.add_argument("--embedding", metavar="EMB",
                        help="fourier | legendre | hermite | …")
    parser.add_argument("--arch", metavar="ARCH",
                        help="e.g. d4r3, d10r6")
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
        outputs_dir = _data_root() / outputs_dir

    configs_root = PROJECT_ROOT / "configs"
    combos = discover_combos(configs_root)

    if not combos:
        print("No HPO combos found under configs/experiments/.")
        return

    if args.list:
        for c in combos:
            status = "OK " if is_filled(c["seed_path"]) else "   "
            print(f"[{status}] {c['dataset']}/{c['trainer']}/{c['embedding']}/{c['arch']}: "
                  f"{c['hpo_kind']} → {c['seed_kind']}")
        return

    if args.trainer:
        combos = [c for c in combos if c["trainer"] == args.trainer]
    if args.embedding:
        combos = [c for c in combos if c["embedding"] == args.embedding]
    if args.arch:
        combos = [c for c in combos if c["arch"] == args.arch]
    if args.dataset:
        combos = [c for c in combos if args.dataset in c["dataset"]]

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
            if process_combo(
                combo=combo,
                source=args.source,
                entity=args.entity,
                project=args.project,
                outputs_dir=outputs_dir,
                dry_run=args.dry_run,
                force=args.force,
            ):
                patched += 1
        except Exception as e:
            label = f"{combo['dataset']}/{combo['trainer']}/{combo['embedding']}/{combo['arch']}"
            print(f"  WARNING: Exception processing {label}: {e}")

    suffix = " (dry run)" if args.dry_run else ""
    print(f"\nDone. Patched {patched}/{total} combos{suffix}.")


if __name__ == "__main__":
    main()
