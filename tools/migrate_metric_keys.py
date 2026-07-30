#!/usr/bin/env python3
"""
One-off migration: absolute -> relative budget keys in analysis CSVs.

Analysis output used to key epsilon/radius columns by their ABSOLUTE model-domain
value (`rob/0.2`, `uq_purify_acc/0.2/0.2`). They are now keyed RELATIVE — as
fractions of the input domain width — so the same column means the same thing on
every embedding. See "Budget vocabulary" in CLAUDE.md.

This script rewrites the headers of already-written CSVs in place:

    rob/0.2                 ->  rob/0.1          (legendre, range_size 2.0)
    uq_purify_acc/0.2/0.2   ->  uq_purify_acc/0.1/0.1
    gibbs_purify_acc/0.2/3  ->  gibbs_purify_acc/0.1/3     (k is a sweep count)

and appends the two provenance columns new runs already write: `range_size` and
`eps_unit="rel"`.

It also renames retired-vocabulary columns whose *value* semantics never changed:

    gibbs_step_radius  ->  gibbs_step_delta_rel

That pass runs on every file, including ones already carrying `eps_unit` — the budget
migration and the vocabulary rename shipped separately, so a file can need one and not
the other.

The `range_size` is derived from the embedding in each file's path. Files whose
embedding cannot be resolved are REPORTED AND SKIPPED, never guessed.

`baselines/jem/` is excluded: JEM has no embedding, its `input_range` is a plain
config field, and its budgets are absolute by design.

Usage:
    python tools/migrate_metric_keys.py                # dry run (default)
    python tools/migrate_metric_keys.py --apply
    python tools/migrate_metric_keys.py --apply --root analysis/outputs/circles
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from analysis.utils.resolve import (
    resolve_embedding_from_path,
    embedding_range_size,
)
from src.utils.embeddings import fmt_budget

# Filenames to migrate. `evaluation_summary.csv` is excluded: those are JEM-only
# and keep metric names in the index column, not the header.
TARGET_NAMES = ("evaluation_data.csv", "gibbs_data.csv", "summary.csv")

EXCLUDE_PARTS = ("baselines",)

# Column families and how many trailing segments are budgets.
#   (prefix, n_budget_segments) — segments beyond that are counts/percentiles.
# `gibbs_*/{eps}/{k}`: only the FIRST segment is a budget; k is a sweep count.
# `uq_detection/{q}pct/{eps}`: the `{q}pct` segment is a percentile, not a budget.
_NUM = r"[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"

# (regex, list of group indices that hold budgets)
_PATTERNS: List[Tuple[re.Pattern, List[int]]] = [
    # two-budget families: {eps}/{delta}
    (re.compile(rf"^((?:eval/(?:test|valid)/)?(?:uq|uq_joint)_purify_(?:acc|recovery))/({_NUM})/({_NUM})$"), [2, 3]),
    # {eps}/{k} — k is a sweep count, left alone
    (re.compile(rf"^((?:eval/(?:test|valid)/)?gibbs(?:_joint)?_purify_(?:acc|recovery|log_px_mean))/({_NUM})/(\d+)$"), [2]),
    # {q}pct/{eps} — q is a percentile, left alone
    (re.compile(rf"^((?:eval/(?:test|valid)/)?uq(?:_joint)?_(?:detection|det_err_detected|det_err_passed))/(\d+pct)/({_NUM})$"), [3]),
    # single-budget families
    (re.compile(rf"^((?:eval/(?:test|valid)/)?rob)/({_NUM})$"), [2]),
    (re.compile(rf"^((?:eval/(?:test|valid)/)?(?:uq|uq_joint)_adv_acc)/({_NUM})$"), [2]),
    (re.compile(rf"^((?:eval/(?:test|valid)/)?gibbs(?:_joint)?_adv_acc)/({_NUM})$"), [2]),
    (re.compile(rf"^((?:eval/(?:test|valid)/)?gibbs(?:_joint)?_adv_log_px_mean)/({_NUM})$"), [2]),
    (re.compile(rf"^((?:eval/(?:test|valid)/)?(?:uq|uq_joint)_clean_purify_acc)/({_NUM})$"), [2]),
]

# Plain column renames: retired vocabulary, identical value semantics. `step_radius`
# was ALREADY a fraction of the input range when it was written (c769ecb renamed the
# knob `step_radius: 0.1` -> `step_delta_rel: 0.1` without touching the number), so
# this is a header rewrite only -- never scale these values.
VOCAB_RENAMES = {"gibbs_step_radius": "gibbs_step_delta_rel"}

# `gibbs_clean_purify_acc/{k}` and `gibbs_clean_log_px_mean/{k}` are keyed by sweep
# count only — no budget to convert. Matched here so they are explicitly left alone
# rather than silently falling through the single-budget patterns above.
_SWEEP_ONLY = re.compile(
    r"^(?:eval/(?:test|valid)/)?gibbs(?:_joint)?_clean_(?:purify_acc|log_px_mean)/\d+$"
)


def convert_column(col: str, range_size: float) -> Optional[str]:
    """Return the relative-keyed name for ``col``, or None if it is not a budget column."""
    if _SWEEP_ONLY.match(col):
        return None
    for pattern, budget_groups in _PATTERNS:
        m = pattern.match(col)
        if not m:
            continue
        parts = list(m.groups())
        for gi in budget_groups:
            abs_val = float(m.group(gi))
            parts[gi - 1] = fmt_budget(abs_val / range_size)
        return "/".join(parts)
    return None


def find_targets(root: Path) -> List[Path]:
    out = []
    for name in TARGET_NAMES:
        for p in root.rglob(name):
            if any(part in EXCLUDE_PARTS for part in p.parts):
                continue
            out.append(p)
    return sorted(out)


def migrate_file(path: Path, apply: bool) -> Dict[str, object]:
    """Migrate one CSV. Returns a status dict; does not raise on skippable files."""
    rel = path.relative_to(PROJECT_ROOT) if path.is_absolute() else path

    df = pd.read_csv(path)

    vocab = {c: VOCAB_RENAMES[c] for c in df.columns if c in VOCAB_RENAMES}
    clash = [v for v in vocab.values() if v in df.columns]
    if clash:
        return {"path": rel, "status": f"collision:{clash[0]}", "renames": vocab}

    if "eps_unit" in df.columns:
        # Budget keys are already relative, but the vocabulary rename may still be due.
        if vocab:
            if apply:
                df.rename(columns=vocab).to_csv(path, index=False)
            return {"path": rel, "status": "vocab-renamed", "renames": vocab}
        return {"path": rel, "status": "already-relative", "renames": {}}

    embedding = resolve_embedding_from_path(str(path))
    if embedding is None:
        return {"path": rel, "status": "unresolved-embedding", "renames": {}}
    range_size = embedding_range_size(embedding)

    renames: Dict[str, str] = {}
    for col in df.columns:
        new = convert_column(col, range_size)
        if new is not None and new != col:
            renames[col] = new

    collisions = [v for v in renames.values() if v in df.columns and v not in renames]
    if collisions:
        return {"path": rel, "status": f"collision:{collisions[0]}", "renames": renames}

    renames.update(vocab)

    if apply:
        df = df.rename(columns=renames)
        df["range_size"] = range_size
        df["eps_unit"] = "rel"
        df.to_csv(path, index=False)

    return {
        "path": rel,
        "status": "migrated" if renames else "no-budget-columns",
        "renames": renames,
        "embedding": embedding,
        "range_size": range_size,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="analysis/outputs",
                    help="Directory to scan (default: analysis/outputs).")
    ap.add_argument("--apply", action="store_true",
                    help="Write the changes. Without this, the script only reports.")
    ap.add_argument("--verbose", action="store_true",
                    help="Show every column rename, not just per-file counts.")
    args = ap.parse_args()

    root = (PROJECT_ROOT / args.root) if not Path(args.root).is_absolute() else Path(args.root)
    if not root.exists():
        raise SystemExit(f"No such directory: {root}")

    targets = find_targets(root)
    print(f"{'APPLY' if args.apply else 'DRY RUN'} — scanning {root}")
    print(f"Found {len(targets)} candidate CSV files "
          f"(excluding {'/'.join(EXCLUDE_PARTS)})\n")

    counts: Dict[str, int] = {}
    unresolved, collisions = [], []
    total_renames = 0

    for path in targets:
        res = migrate_file(path, args.apply)
        status = str(res["status"])
        counts[status] = counts.get(status, 0) + 1
        total_renames += len(res["renames"])

        if status == "unresolved-embedding":
            unresolved.append(res["path"])
        elif status.startswith("collision:"):
            collisions.append((res["path"], status))
        elif status == "migrated":
            print(f"  {res['path']}")
            print(f"      embedding={res['embedding']} range_size={res['range_size']}  "
                  f"{len(res['renames'])} columns")
            if args.verbose:
                for old, new in res["renames"].items():
                    print(f"        {old}  ->  {new}")
        elif status == "vocab-renamed":
            print(f"  {res['path']}")
            print(f"      vocabulary only, {len(res['renames'])} columns: "
                  + ", ".join(f"{o} -> {n}" for o, n in res["renames"].items()))

    print("\n" + "=" * 64)
    for status, n in sorted(counts.items()):
        print(f"  {status:24s} {n:4d} files")
    print(f"  {'total column renames':24s} {total_renames:4d}")

    if unresolved:
        print(f"\nSKIPPED — embedding not resolvable from path ({len(unresolved)}):")
        for p in unresolved:
            print(f"    {p}")
    if collisions:
        print(f"\nSKIPPED — rename would collide with an existing column ({len(collisions)}):")
        for p, why in collisions:
            print(f"    {p}  [{why}]")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
