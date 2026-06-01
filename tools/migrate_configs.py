#!/usr/bin/env python3
"""
Migrate configs/experiments/ to the new unified structure.

Old layout:
  configs/experiments/nll/{dis|gen|mixed}/{emb}/{arch}/{kind}/{dataset}.yaml
  configs/experiments/adversarial/{emb}/{arch}/{kind}/{dataset}.yaml

New layout:
  configs/experiments/{dataset}/{nat|at}/{emb}/{arch}/{kind}.yaml

Kind name mapping (alpha suffix encodes which alpha value the config fixes):

  nll/dis/  → trainer=nat, kind suffix _a0  (alpha=0, discriminative)
  nll/gen/  → trainer=nat, kind suffix _a1  (alpha=1, generative)
  nll/mixed/ → trainer=nat, per-kind rename (see _MIXED_KIND_MAP):
      hpo_mixed              → hpo_a05
      seed_sweep_mixed_alpha05 → seed_sweep_a05
      seed_sweep_mixed       → seed_sweep_a01
      alpha_curve_mixed      → alpha_curve   (alpha is the sweep axis)
      grid_sweep_mixed       → grid_sweep    (alpha is the sweep axis)
      cls_reg                → cls_reg_a1
  adversarial/ → trainer=at, kind unchanged

Also strips per-file hydra.sweep.dir overrides from migrated configs (the global
template in configs/config.yaml now controls output paths).

Usage:
    python tools/migrate_configs.py --dry-run    # print plan
    python tools/migrate_configs.py --execute    # git mv + strip overrides
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_ROOT = PROJECT_ROOT / "configs" / "experiments"

# Kind names that live under nll/dis/ or nll/gen/ get a per-sub-dir suffix.
_DIS_SUFFIX = "_a0"
_GEN_SUFFIX = "_a1"

# Explicit kind renames for nll/mixed/.
# Keys are the old folder name, values are the new yaml stem (no .yaml).
_MIXED_KIND_MAP: Dict[str, str] = {
    "hpo_mixed":               "hpo_a05",
    "seed_sweep_mixed_alpha05": "seed_sweep_a05",
    "seed_sweep_mixed":        "seed_sweep_a01",
    "alpha_curve_mixed":       "alpha_curve",
    "grid_sweep_mixed":        "grid_sweep",
    "cls_reg":                 "cls_reg_a1",
}

# Regex that matches a hydra: → run:/sweep: dir override block we want to strip.
# We strip any "  run:\n    dir: ..." or "  sweep:\n    dir: ...\n    subdir: ..."
# inside a top-level `hydra:` key.  We only remove those two sub-keys, not the
# entire hydra block (sweeper params etc. stay).
_HYDRA_RUN_DIR_RE = re.compile(
    r"^  run:\n    dir:.*\n",
    re.MULTILINE,
)
_HYDRA_SWEEP_DIR_RE = re.compile(
    r"^  sweep:\n    dir:.*\n    subdir:.*\n",
    re.MULTILINE,
)


def _map_kind(kind: str, sub: str) -> str:
    """Map old kind folder name → new kind stem given the nll sub-dir."""
    if sub == "dis":
        return kind + _DIS_SUFFIX
    if sub == "gen":
        return kind + _GEN_SUFFIX
    if sub == "mixed":
        if kind in _MIXED_KIND_MAP:
            return _MIXED_KIND_MAP[kind]
        # Fallback: keep as-is (should not happen with known configs)
        return kind
    return kind  # adversarial: unchanged


def _strip_hydra_dir_overrides(content: str) -> Tuple[str, bool]:
    """Remove per-file hydra run/sweep dir overrides. Returns (new_content, changed)."""
    new = _HYDRA_RUN_DIR_RE.sub("", content)
    new = _HYDRA_SWEEP_DIR_RE.sub("", new)
    changed = new != content
    # If the hydra: key is now empty (only contains whitespace lines), remove it.
    new = re.sub(r"^hydra:\s*\n(?=\S|\Z)", "", new, flags=re.MULTILINE)
    return new, changed


def _compute_moves() -> List[Tuple[Path, Path, str]]:
    """
    Returns list of (src_path, dst_path, sub) for all files to migrate.
    sub is the nll sub-dir ("dis", "gen", "mixed") or "at" for adversarial.
    """
    moves: List[Tuple[Path, Path, str]] = []

    for p in sorted(CONFIGS_ROOT.rglob("*.yaml")):
        rel = p.relative_to(CONFIGS_ROOT)
        parts = rel.parts

        if parts[0] == "tests":
            continue

        if parts[0] == "nll":
            if len(parts) != 6:
                continue
            _, sub, emb, arch, kind_folder, dataset_yaml = parts
            if sub not in ("dis", "gen", "mixed"):
                continue
            trainer = "nat"
            dataset = Path(dataset_yaml).stem
            new_kind = _map_kind(kind_folder, sub)
        elif parts[0] == "adversarial":
            if len(parts) != 5:
                continue
            _, emb, arch, kind_folder, dataset_yaml = parts
            trainer = "at"
            dataset = Path(dataset_yaml).stem
            sub = "at"
            new_kind = kind_folder  # unchanged
        else:
            continue

        dst = CONFIGS_ROOT / dataset / trainer / emb / arch / f"{new_kind}.yaml"
        moves.append((p, dst, sub))

    return moves


def _print_dry_run(moves: List[Tuple[Path, Path, str]]) -> None:
    print(f"Migration plan: {len(moves)} files to move\n")
    # Check for any remaining collisions (should be none with per-sub suffixes)
    dst_counts: Dict[str, int] = {}
    for _, dst, _ in moves:
        k = str(dst)
        dst_counts[k] = dst_counts.get(k, 0) + 1
    collisions = {k: v for k, v in dst_counts.items() if v > 1}
    if collisions:
        print("WARNING: unexpected collisions:")
        for k, n in collisions.items():
            print(f"  {n}× → {k}")
        print()

    print("Moves:")
    for src, dst, _ in sorted(moves, key=lambda x: str(x[1])):
        src_rel = src.relative_to(CONFIGS_ROOT)
        dst_rel = dst.relative_to(CONFIGS_ROOT)
        print(f"  {src_rel}")
        print(f"    → {dst_rel}")


def _execute_moves(moves: List[Tuple[Path, Path, str]]) -> None:
    # Create all required destination parent dirs
    dst_parents = sorted({dst.parent for _, dst, _ in moves})
    for parent in dst_parents:
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    # git mv each file
    failed = 0
    stripped = 0
    for src, dst, sub in sorted(moves, key=lambda x: str(x[1])):
        src_rel = src.relative_to(PROJECT_ROOT)
        dst_rel = dst.relative_to(PROJECT_ROOT)

        # Strip hydra dir overrides from mixed configs before moving
        if sub == "mixed":
            content = src.read_text()
            new_content, changed = _strip_hydra_dir_overrides(content)
            if changed:
                src.write_text(new_content)
                stripped += 1

        cmd = ["git", "mv", str(src_rel), str(dst_rel)]
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"  ERROR: git mv failed for {src_rel}", file=sys.stderr)
            failed += 1

    if stripped:
        print(f"Stripped hydra.sweep.dir overrides from {stripped} file(s) before moving.")

    if failed:
        print(f"\n{failed} git mv command(s) failed.", file=sys.stderr)
        sys.exit(1)

    # Remove now-empty old directories
    old_roots = [CONFIGS_ROOT / "nll", CONFIGS_ROOT / "adversarial"]
    for old_root in old_roots:
        if old_root.exists():
            cmd = ["git", "rm", "-r", "--ignore-unmatch",
                   str(old_root.relative_to(PROJECT_ROOT))]
            subprocess.run(cmd, cwd=PROJECT_ROOT)

    print(f"\nDone. Moved {len(moves)} files.")
    print("Review with: git status && git diff --stat HEAD")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate experiment configs to new dataset-first structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the move plan without making any changes.")
    parser.add_argument("--execute", action="store_true",
                        help="Run git mv to move the files.")
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        print("Specify --dry-run to preview or --execute to apply.")
        parser.print_help()
        sys.exit(0)

    moves = _compute_moves()

    if args.dry_run:
        _print_dry_run(moves)
    elif args.execute:
        _execute_moves(moves)


if __name__ == "__main__":
    main()
