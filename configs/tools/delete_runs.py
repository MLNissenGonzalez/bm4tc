#!/usr/bin/env python3
"""
Delete sweep outputs: local dirs, W&B runs/artifacts, and analysis dirs.

Discovers sweep roots under outputs/{kind}/{regime}/{embedding}/{arch}/{dataset}_{date}/
and deletes the matching local directories, W&B runs + artifacts, and mirrored
analysis/outputs/ directories.

Usage:
    python configs/tools/delete_runs.py [filters] [options]

Filter flags (all accept one or more values; OR-within / AND-across):
    --kind         hpo | seed_sweep | test
    --regime       cls | gen | adv | gan
    --embedding    fourier | legendre | hermite | chebychev1 | chebychev2
    --arch         d4D3 | d6D4 | d10D6 | d30D18
    --dataset      substring match on dataset name (e.g. "circles", "moons_4k")
    --date         substring match on date string (e.g. "2202", "23")
    --state        W&B run state filter: finished | failed | crashed | running

Options:
    --list         Show all discovered sweep roots with [will-delete]/[skip] status
    --dry-run      Show full deletion report without deleting
    --local-only   Skip W&B; only remove local dirs + analysis dirs
    --wandb-only   Skip local/analysis dirs; only remove W&B runs and artifacts
    --entity       W&B entity (default: martin-nissen-gonzalez-heidelberg-university)
    --project      W&B project (default: gan_train)

Examples:
    python configs/tools/delete_runs.py --list
    python configs/tools/delete_runs.py --kind hpo --regime gen --dry-run
    python configs/tools/delete_runs.py --dataset circles --date 2102 --dry-run
    python configs/tools/delete_runs.py --kind test --dry-run
    python configs/tools/delete_runs.py --kind test
    python configs/tools/delete_runs.py --kind hpo --wandb-only --dry-run
"""

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Bootstrap: add project root to sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.utils.wandb_fetcher import (
    _load_hydra_config,
    _get_nested_value,
    WANDB_AVAILABLE,
)

# =============================================================================
# Constants
# =============================================================================

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "outputs"
DEFAULT_ENTITY = "martin-nissen-gonzalez-heidelberg-university"
DEFAULT_PROJECT = "gan_train"

# Matches: <dataset>_<DDMM>  or  <dataset>_<DDMM>_<HHMM>
DATE_RE = re.compile(r"^(.+?)_(\d{4}(?:_\d{4})?)$")

# Matches wandb run dir: run-<TIMESTAMP>-<RUNID>
WANDB_RUN_DIR_RE = re.compile(r"^run-\d+T\d+-([a-z0-9]+)$")


# =============================================================================
# Path parsing
# =============================================================================


def parse_sweep_path(path: Path) -> Optional[Dict]:
    """Parse outputs/{kind}/{regime}/{embedding}/{arch}/{dataset}_{date}."""
    try:
        rel = path.relative_to(OUTPUTS_DIR)
    except ValueError:
        return None

    parts = rel.parts
    if len(parts) != 5:
        return None

    kind, regime, embedding, arch, leaf = parts

    m = DATE_RE.match(leaf)
    if not m:
        return None

    dataset, date = m.group(1), m.group(2)

    return {
        "kind": kind,
        "regime": regime,
        "embedding": embedding,
        "arch": arch,
        "dataset": dataset,
        "date": date,
        "path": path,
    }


def is_sweep_root(path: Path) -> bool:
    """True if dir has numbered subdirs (multirun) or direct .hydra/ (single run)."""
    if not path.is_dir():
        return False
    if (path / ".hydra").is_dir():
        return True
    return any(c.is_dir() and c.name.isdigit() for c in path.iterdir())


# =============================================================================
# Discovery
# =============================================================================


def find_sweep_roots(outputs_dir: Path) -> List[Dict]:
    """Walk outputs/ at depth 5; return parsed dicts for each sweep root."""
    results = []
    if not outputs_dir.is_dir():
        return results

    for kind_dir in outputs_dir.iterdir():
        if not kind_dir.is_dir():
            continue
        for regime_dir in kind_dir.iterdir():
            if not regime_dir.is_dir():
                continue
            for embedding_dir in regime_dir.iterdir():
                if not embedding_dir.is_dir():
                    continue
                for arch_dir in embedding_dir.iterdir():
                    if not arch_dir.is_dir():
                        continue
                    for sweep_dir in arch_dir.iterdir():
                        if not sweep_dir.is_dir():
                            continue
                        if not is_sweep_root(sweep_dir):
                            continue
                        info = parse_sweep_path(sweep_dir)
                        if info is not None:
                            results.append(info)

    results.sort(key=lambda x: (x["kind"], x["regime"], x["embedding"], x["arch"], x["dataset"], x["date"]))
    return results


def apply_filters(targets: List[Dict], args) -> List[Dict]:
    """Filter by kind, regime, embedding, arch, dataset substring, date substring."""
    filtered = []
    for info in targets:
        if args.kind and info["kind"] not in args.kind:
            continue
        if args.regime and info["regime"] not in args.regime:
            continue
        if args.embedding and info["embedding"] not in args.embedding:
            continue
        if args.arch and info["arch"] not in args.arch:
            continue
        if args.dataset and not any(s in info["dataset"] for s in args.dataset):
            continue
        if args.date and not any(s in info["date"] for s in args.date):
            continue
        filtered.append(info)
    return filtered


# =============================================================================
# W&B run resolution
# =============================================================================


def get_tracking_mode(sweep_root: Path) -> str:
    """Read tracking.mode from first .hydra/config.yaml found. Returns 'disabled'/'online'/etc."""
    # Single run: .hydra/ directly in sweep root
    cfg = _load_hydra_config(sweep_root)
    if cfg is not None:
        return _get_nested_value(cfg, "tracking.mode", "online")

    # Multirun: .hydra/ inside numbered subdirs
    for child in sorted(sweep_root.iterdir()):
        if child.is_dir() and child.name.isdigit():
            cfg = _load_hydra_config(child)
            if cfg is not None:
                return _get_nested_value(cfg, "tracking.mode", "online")

    return "online"  # assume enabled if config unreadable


def extract_wandb_run_ids(sweep_root: Path) -> List[str]:
    """Find wandb/run-TIMESTAMP-RUNID/ dirs; return run IDs (last '-' segment)."""
    run_ids: List[str] = []

    def _scan(d: Path) -> None:
        wandb_dir = d / "wandb"
        if wandb_dir.is_dir():
            for entry in wandb_dir.iterdir():
                if entry.is_dir():
                    m = WANDB_RUN_DIR_RE.match(entry.name)
                    if m:
                        run_ids.append(m.group(1))

    _scan(sweep_root)
    for child in sweep_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            _scan(child)

    return run_ids


def reconstruct_group_pattern(info: Dict) -> str:
    """Build W&B group regex from path components."""
    archinfo = f"{info['arch']}{info['embedding']}"
    return f"^{info['kind']}_{info['regime']}_{archinfo}_{info['dataset']}_{info['date']}$"


def resolve_wandb_runs(
    info: Dict,
    entity: str,
    project: str,
    state_filter: Optional[List[str]],
) -> List[Any]:
    """Primary: fetch by extracted run IDs. Fallback: query by group pattern."""
    if not WANDB_AVAILABLE:
        return []

    import wandb

    api = wandb.Api()
    api_path = f"{entity}/{project}"

    # Primary: fetch by run ID
    run_ids = extract_wandb_run_ids(info["path"])
    runs: List[Any] = []

    for run_id in run_ids:
        try:
            run = api.run(f"{api_path}/{run_id}")
            if state_filter is None or run.state in state_filter:
                runs.append(run)
        except Exception:
            pass  # run may not exist in W&B (e.g. offline-only)

    if not runs:
        # Fallback: group regex query
        group_pattern = reconstruct_group_pattern(info)
        filters: Dict = {"group": {"$regex": group_pattern}}
        if state_filter:
            filters["state"] = {"$in": state_filter}
        try:
            runs = list(api.runs(path=api_path, filters=filters))
        except Exception as e:
            print(f"  [wandb] WARNING: group query failed: {e}")

    return runs


# =============================================================================
# Analysis dir resolution
# =============================================================================


def find_analysis_dir(info: Dict, analysis_root: Path) -> Optional[Path]:
    """Check analysis/outputs/{regime}/{embedding}/{arch}/{dataset}_{date}/."""
    candidate = (
        analysis_root
        / info["regime"]
        / info["embedding"]
        / info["arch"]
        / f"{info['dataset']}_{info['date']}"
    )
    return candidate if candidate.is_dir() else None


# =============================================================================
# Size / count helpers
# =============================================================================


def dir_size_mb(path: Path) -> float:
    """Total size of directory in MB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def count_job_dirs(sweep_root: Path) -> int:
    """Count numbered job subdirs; 1 for single-run."""
    if (sweep_root / ".hydra").is_dir():
        return 1
    return sum(1 for c in sweep_root.iterdir() if c.is_dir() and c.name.isdigit())


# =============================================================================
# Per-sweep info gathering
# =============================================================================


def gather_sweep_info(
    info: Dict,
    entity: str,
    project: str,
    state_filter: Optional[List[str]],
    local_only: bool,
    wandb_only: bool,
) -> Dict:
    """Collect local size, W&B runs, artifact count, and analysis dir."""
    result = dict(info)

    if not wandb_only:
        result["job_dirs"] = count_job_dirs(info["path"])
        result["size_mb"] = dir_size_mb(info["path"])
        result["analysis_dir"] = find_analysis_dir(info, ANALYSIS_DIR)
    else:
        result["job_dirs"] = 0
        result["size_mb"] = 0.0
        result["analysis_dir"] = None

    tracking_mode = get_tracking_mode(info["path"])
    result["tracking_mode"] = tracking_mode

    if not local_only and tracking_mode != "disabled" and WANDB_AVAILABLE:
        result["wandb_runs"] = resolve_wandb_runs(info, entity, project, state_filter)
    else:
        result["wandb_runs"] = []

    return result


# =============================================================================
# Summary report
# =============================================================================


def print_report(gathered: List[Dict], wandb_only: bool, local_only: bool) -> None:
    """Print summary of what will be deleted."""
    total_size = 0.0
    total_runs = 0
    total_artifacts = 0
    total_analysis = 0

    print(f"\nFound {len(gathered)} sweep root(s) to delete:\n")

    for i, g in enumerate(gathered, 1):
        rel = g["path"].relative_to(PROJECT_ROOT)
        print(f"[{i}] {rel}/")

        if not wandb_only:
            print(f"    Local:    {g['job_dirs']} job dir(s), {g['size_mb']:.1f} MB")
            total_size += g["size_mb"]

        if not local_only:
            if g["tracking_mode"] == "disabled":
                print(f"    W&B:      tracking disabled — skipping")
            else:
                runs = g["wandb_runs"]
                n_runs = len(runs)
                if n_runs > 0:
                    states = ", ".join(sorted({r.state for r in runs}))
                    n_artifacts = sum(sum(1 for _ in r.logged_artifacts()) for r in runs)
                    total_runs += n_runs
                    total_artifacts += n_artifacts
                    print(f"    W&B:      {n_runs} run(s) ({states}) | artifacts: {n_artifacts}")
                else:
                    print(f"    W&B:      no runs found in W&B")

        if not wandb_only:
            ad = g["analysis_dir"]
            if ad is not None:
                print(f"    Analysis: {ad.relative_to(PROJECT_ROOT)}/")
                total_analysis += 1
            else:
                print(f"    Analysis: (none found)")

        print()

    parts = []
    if not wandb_only:
        parts.append(f"{len(gathered)} local dir(s) (~{total_size:.0f} MB)")
    if not local_only:
        parts.append(f"{total_runs} W&B run(s)")
        parts.append(f"{total_artifacts} artifact(s)")
    if not wandb_only:
        parts.append(f"{total_analysis} analysis dir(s)")
    print("Total: " + ", ".join(parts))


# =============================================================================
# Deletion
# =============================================================================


def delete_sweep(g: Dict, local_only: bool, wandb_only: bool, dry_run: bool) -> None:
    """Delete one sweep: W&B artifacts → W&B runs → local dir → analysis dir."""
    prefix = "(dry) " if dry_run else ""

    # 1. W&B artifacts
    if not local_only and g["wandb_runs"]:
        for run in g["wandb_runs"]:
            try:
                for artifact in run.logged_artifacts():
                    if not dry_run:
                        artifact.delete(delete_aliases=True)
                    print(f"  [wandb] {prefix}Deleted artifact: {artifact.name}")
            except Exception as e:
                print(f"  [wandb] WARNING: artifact deletion failed for run {run.id}: {e}")

    # 2. W&B runs
    if not local_only and g["wandb_runs"]:
        for run in g["wandb_runs"]:
            try:
                if not dry_run:
                    run.delete()
                print(f"  [wandb] {prefix}Deleted run: {run.id}")
            except Exception as e:
                print(f"  [wandb] WARNING: run deletion failed for {run.id}: {e}")

    # 3. Local sweep dir
    if not wandb_only:
        rel = g["path"].relative_to(PROJECT_ROOT)
        if not dry_run:
            shutil.rmtree(g["path"])
        print(f"  [local] {prefix}Removed: {rel}/")

    # 4. Analysis dir
    if not wandb_only and g["analysis_dir"] is not None:
        ad_rel = g["analysis_dir"].relative_to(PROJECT_ROOT)
        if not dry_run:
            shutil.rmtree(g["analysis_dir"])
        print(f"  [analysis] {prefix}Removed: {ad_rel}/")


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete sweep outputs: local dirs, W&B runs/artifacts, and analysis dirs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Filter flags
    parser.add_argument("--kind", nargs="+", metavar="KIND",
                        help="hpo | seed_sweep | test")
    parser.add_argument("--regime", nargs="+", metavar="REGIME",
                        help="cls | gen | adv | gan")
    parser.add_argument("--embedding", nargs="+", metavar="EMB",
                        help="fourier | legendre | hermite | chebychev1 | chebychev2")
    parser.add_argument("--arch", nargs="+", metavar="ARCH",
                        help="d4D3 | d6D4 | d10D6 | d30D18")
    parser.add_argument("--dataset", nargs="+", metavar="STR",
                        help="Substring match on dataset name")
    parser.add_argument("--date", nargs="+", metavar="STR",
                        help="Substring match on date string")
    parser.add_argument("--state", nargs="+", metavar="STATE",
                        help="W&B run state filter: finished | failed | crashed | running")

    # Options
    parser.add_argument("--list", action="store_true",
                        help="Show all discovered sweep roots with filter status; no deletion")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show full report without deleting (no confirmation prompt)")
    parser.add_argument("--local-only", action="store_true",
                        help="Skip W&B deletion; only remove local dirs + analysis dirs")
    parser.add_argument("--wandb-only", action="store_true",
                        help="Skip local/analysis dirs; only remove W&B runs and artifacts")
    parser.add_argument("--entity", default=DEFAULT_ENTITY,
                        help="W&B entity (default: %(default)s)")
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help="W&B project (default: %(default)s)")

    args = parser.parse_args()

    if args.local_only and args.wandb_only:
        parser.error("--local-only and --wandb-only are mutually exclusive")

    # ----- List mode -----
    if args.list:
        all_roots = find_sweep_roots(OUTPUTS_DIR)
        matched_ids = {id(x) for x in apply_filters(all_roots, args)}
        print(f"Discovered {len(all_roots)} sweep root(s):\n")
        for info in all_roots:
            rel = info["path"].relative_to(PROJECT_ROOT)
            tag = "[will-delete]" if id(info) in matched_ids else "[skip]"
            print(f"  {tag:13s} {rel}/")
        print()
        return

    # ----- Discover + filter -----
    all_roots = find_sweep_roots(OUTPUTS_DIR)
    targets = apply_filters(all_roots, args)

    if not targets:
        print("No sweep roots matched the given filters.")
        return

    # ----- Gather per-sweep info -----
    print(f"Gathering info for {len(targets)} sweep root(s)...")
    gathered = []
    for info in targets:
        g = gather_sweep_info(
            info, args.entity, args.project, args.state,
            args.local_only, args.wandb_only,
        )
        gathered.append(g)

    # ----- Print report -----
    print_report(gathered, args.wandb_only, args.local_only)

    if args.dry_run:
        print("\n(Dry run — nothing deleted.)")
        return

    # ----- Interactive confirmation -----
    print()
    try:
        response = input("Delete all of the above? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)

    if response != "y":
        print("Aborted.")
        sys.exit(0)

    # ----- Execute deletion -----
    print()
    for g in gathered:
        rel = g["path"].relative_to(PROJECT_ROOT)
        print(f"Deleting {rel}/")
        delete_sweep(g, args.local_only, args.wandb_only, dry_run=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
