#!/usr/bin/env python3
"""
One-off migration: replace the dead alpha column in analysis CSVs with the live keys.

`analysis/sweep.py` used to extract `trainer.generative.criterion.kwargs.alpha` into
`config/trainer.generative.criterion.kwargs.alpha`. That path has not existed since the
trainer refactor — alpha lives at `trainer.nll.alpha` (NAT) / `trainer.adversarial.alpha`
(AT) — so the column is all-NaN in every CSV written before 2026-07-31. `sweep.py` now
extracts both live keys, which fixes future CSVs; this script repairs the existing ones.

The run directories live on mathqi, so the configs cannot be re-read locally. W&B is the
source of truth: each CSV's `run_path` column names the sweep, whose tail after `outputs/`
is exactly the W&B group, and each row's `run_name` is the W&B run name within it.

Every row is cross-checked before anything is written, and any failure skips the WHOLE
file (never a partial rewrite):

  * all rows agree on one sweep / W&B group;
  * every CSV row has a matching W&B run. A relaunch leaves several runs sharing a name;
    they are told apart by the seed the CSV also records, and only a genuine (name, seed)
    collision is fatal;
  * exactly one of `trainer.nll.alpha` / `trainer.adversarial.alpha` is set per run —
    except for AT runs predating 27ac8c1 (2026-06-24), whose `adversarial` node has no
    `alpha` field at all because that trainer hardcoded `mixed_nll(..., alpha=0.0)`;
    those are alpha=0 by construction and are filled as such;
  * the W&B seed equals the CSV's own `config/tracking.seed` where present;
  * the dead column is entirely NaN, so nothing is overwritten.

Both live columns are written, the inactive one empty — the same shape `sweep.py` now
produces, so migrated and freshly written CSVs share one schema.

The two spirals `alpha_curve` CSVs were migrated first, by the single-purpose form of this
script; those additionally cross-checked alpha against the Hydra product ordering of
`configs/experiments/spirals/nat/legendre/d10r6/alpha_curve.yaml` (10 alphas x 5 seeds,
seed varying fastest). They carry the live column already and are skipped here.

`baselines/jem/` is excluded: JEM has its own `alpha` column and no hydra trainer node.

Usage:
    python tools/backfill_alpha_column.py                  # dry run over every CSV
    python tools/backfill_alpha_column.py --apply
    python tools/backfill_alpha_column.py --root analysis/outputs/mnist_full_r12 --apply
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

ENTITY = "martin-nissen-gonzalez-heidelberg-university"
PROJECT = "bm4tc"

DEAD_COL = "config/trainer.generative.criterion.kwargs.alpha"
NLL_COL = "config/trainer.nll.alpha"
ADV_COL = "config/trainer.adversarial.alpha"
SEED_COL = "config/tracking.seed"

EXCLUDE_PARTS = ("baselines",)

# 27ac8c1 "feat(at): adversarial alpha>0 support". Adversarial runs created before this
# had no alpha field and a hardcoded alpha=0 objective.
AT_ALPHA_COMMIT_DATE = "2026-06-24"


class Unresolved(Exception):
    """This file cannot be migrated safely; report it and leave it alone."""


def find_targets(root: Path) -> List[Path]:
    """Every `evaluation_data.csv` under `root` still carrying the dead column."""
    out = []
    for path in sorted(root.rglob("evaluation_data.csv")):
        if any(part in EXCLUDE_PARTS for part in path.parts):
            continue
        header = pd.read_csv(path, nrows=0).columns
        if DEAD_COL in header:
            out.append(path)
    return out


def group_of(df: pd.DataFrame, path: Path) -> str:
    """The W&B group name implied by the CSV's run paths."""
    if "run_path" not in df.columns:
        raise Unresolved("no run_path column")
    groups = set()
    for rp in df["run_path"].dropna():
        rp = str(rp).replace("\\", "/")
        if "/outputs/" not in rp:
            raise Unresolved(f"run_path {rp!r} has no /outputs/ segment")
        # .../outputs/<group>/<run_name>
        groups.add(rp.split("/outputs/", 1)[1].rsplit("/", 1)[0])
    if len(groups) != 1:
        raise Unresolved(f"rows span several sweeps: {sorted(groups)}")
    return groups.pop()


def fetch_group(api, group: str, cache: Dict[str, dict]) -> Dict[str, Dict[int, Tuple[str, float]]]:
    """{run_name: {seed: (column, alpha)}} for one W&B group.

    `api.runs()` returns runs with an empty `.config` (lazy load), so each run's config
    has to be fetched individually via `api.run()`.
    """
    if group in cache:
        return {name: {int(s): tuple(v) for s, v in seeds.items()}
                for name, seeds in cache[group].items()}

    listed = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group}, per_page=200))
    if not listed:
        # Older sweeps log a group with an extra _HHMM launch-time suffix that the output
        # directory name drops (dir `alpha_curve_0506` <-> group `alpha_curve_0506_1501`).
        pattern = f"^{re.escape(group)}_[0-9]{{4}}$"
        listed = list(api.runs(f"{ENTITY}/{PROJECT}",
                               filters={"group": {"$regex": pattern}}, per_page=200))
        found = sorted({r.group for r in listed})
        if len(found) > 1:
            raise Unresolved(f"group {group!r} matches several W&B groups: {found}")
        if not listed:
            raise Unresolved(f"no W&B runs in group {group!r} (nor {group}_HHMM)")
        print(f"    (matched W&B group {found[0]!r})")

    out: Dict[str, Dict[int, Tuple[str, float]]] = {}
    for r in listed:
        cfg = api.run(f"{ENTITY}/{PROJECT}/{r.id}").config
        trainer = cfg.get("trainer") or {}
        nll_alpha = (trainer.get("nll") or {}).get("alpha")
        adv_alpha = (trainer.get("adversarial") or {}).get("alpha")

        # AT sweeps older than 27ac8c1 (2026-06-24, "feat(at): alpha>0 support") have an
        # `adversarial` node with no `alpha` field, because that trainer called
        # `mixed_nll(..., alpha=0.0)` unconditionally. Their alpha is 0 by construction,
        # not by inference. The date guard keeps a *future* config that loses its alpha
        # field from silently becoming 0.
        if (nll_alpha is None and adv_alpha is None
                and "adversarial" in trainer and "alpha" not in trainer["adversarial"]
                and r.created_at < AT_ALPHA_COMMIT_DATE):
            adv_alpha = 0.0

        if (nll_alpha is None) == (adv_alpha is None):
            raise Unresolved(
                f"run {r.name}: expected exactly one of nll/adversarial alpha, got "
                f"nll={nll_alpha!r} adversarial={adv_alpha!r}"
            )
        column = NLL_COL if nll_alpha is not None else ADV_COL
        seed = int((cfg.get("tracking") or {}).get("seed", -1))
        entry = (column, float(nll_alpha if nll_alpha is not None else adv_alpha))
        # A relaunch leaves several runs sharing a name; they are told apart by seed,
        # which the CSV also records. Only a genuine (name, seed) collision is fatal.
        by_seed = out.setdefault(r.name, {})
        if seed in by_seed and by_seed[seed] != entry:
            raise Unresolved(f"W&B runs named {r.name!r} at seed {seed} disagree: "
                             f"{by_seed[seed]} vs {entry}")
        by_seed[seed] = entry

    cache[group] = {name: {str(s): list(v) for s, v in seeds.items()}
                    for name, seeds in out.items()}
    return out


def build_columns(df: pd.DataFrame, wb: Dict[str, Dict[int, Tuple[str, float]]]
                  ) -> Tuple[pd.Series, pd.Series, str]:
    """Validate every row and return (nll_column, adv_column, regime_label)."""
    if DEAD_COL not in df.columns:
        raise Unresolved("dead column already gone")
    if df[DEAD_COL].notna().any():
        raise Unresolved("dead column is NOT all-NaN; refusing to overwrite data")

    nll_vals: List[Optional[float]] = []
    adv_vals: List[Optional[float]] = []
    columns_seen = set()

    for _, row in df.iterrows():
        name = str(row["run_name"])
        if name not in wb:
            raise Unresolved(f"run {name!r} has no W&B counterpart in the group")
        by_seed = wb[name]
        csv_seed = row.get(SEED_COL)

        if len(by_seed) == 1:
            (seed, (column, alpha)), = by_seed.items()
            if pd.notna(csv_seed) and int(csv_seed) != seed:
                raise Unresolved(f"run {name}: CSV seed {int(csv_seed)} != W&B seed {seed}")
        else:
            # Relaunched run name: the CSV's own seed says which attempt this row is.
            if pd.isna(csv_seed):
                raise Unresolved(f"run {name}: {len(by_seed)} W&B runs and no CSV seed "
                                 f"to disambiguate (seeds {sorted(by_seed)})")
            if int(csv_seed) not in by_seed:
                raise Unresolved(f"run {name}: CSV seed {int(csv_seed)} matches none of "
                                 f"the W&B seeds {sorted(by_seed)}")
            column, alpha = by_seed[int(csv_seed)]
        columns_seen.add(column)

        nll_vals.append(alpha if column == NLL_COL else None)
        adv_vals.append(alpha if column == ADV_COL else None)

    if len(columns_seen) != 1:
        raise Unresolved(f"sweep mixes NAT and AT runs: {sorted(columns_seen)}")

    regime = "nat" if columns_seen == {NLL_COL} else "at"
    return (pd.Series(nll_vals, index=df.index, dtype="float64"),
            pd.Series(adv_vals, index=df.index, dtype="float64"), regime)


def migrate(path: Path, api, cache: Dict[str, dict], apply: bool) -> Optional[str]:
    """Migrate one CSV. Returns a one-line report, or None when skipped."""
    rel = path.relative_to(PROJECT_ROOT)
    df = pd.read_csv(path)
    group = group_of(df, path)
    wb = fetch_group(api, group, cache)
    nll_col, adv_col, regime = build_columns(df, wb)

    alphas = (nll_col if regime == "nat" else adv_col).dropna().unique()
    report = (f"{rel}\n    group={group}  regime={regime}  rows={len(df)}  "
              f"alpha={sorted(float(a) for a in alphas)}")

    if not apply:
        return report + "  [dry run]"

    pos = df.columns.get_loc(DEAD_COL)
    out = df.drop(columns=[DEAD_COL])
    out.insert(pos, ADV_COL, adv_col)
    out.insert(pos, NLL_COL, nll_col)
    out.to_csv(path, index=False)

    back = pd.read_csv(path)
    pd.testing.assert_frame_equal(
        back.drop(columns=[NLL_COL, ADV_COL]), df.drop(columns=[DEAD_COL]), check_exact=False
    )
    active = back[NLL_COL] if regime == "nat" else back[ADV_COL]
    inactive = back[ADV_COL] if regime == "nat" else back[NLL_COL]
    assert active.notna().all() and inactive.isna().all()
    return report + "  [written]"


def main() -> None:
    import wandb

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--apply", action="store_true", help="write the CSVs (default: dry run)")
    ap.add_argument("--root", default="analysis/outputs", help="subtree to scan")
    ap.add_argument("--cache", default=None,
                    help="JSON file of fetched W&B configs, reused across invocations")
    args = ap.parse_args()

    root = (PROJECT_ROOT / args.root).resolve()
    targets = find_targets(root)
    print(f"{len(targets)} CSV(s) under {root.relative_to(PROJECT_ROOT)} carry {DEAD_COL}\n")
    if not targets:
        return

    cache_path = Path(args.cache) if args.cache else None
    cache: Dict[str, dict] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"loaded {len(cache)} cached group(s) from {cache_path}\n")

    api = wandb.Api()
    done, skipped = [], []
    try:
        for i, path in enumerate(targets, 1):
            try:
                report = migrate(path, api, cache, args.apply)
                print(f"[{i}/{len(targets)}] {report}")
                done.append(path)
            except Unresolved as e:
                print(f"[{i}/{len(targets)}] {path.relative_to(PROJECT_ROOT)}\n    SKIP: {e}")
                skipped.append((path, str(e)))
    finally:
        if cache_path:
            cache_path.write_text(json.dumps(cache))

    print(f"\n{len(done)} migrated, {len(skipped)} skipped")
    for path, why in skipped:
        print(f"  SKIP {path.relative_to(PROJECT_ROOT)}: {why}")
    if not args.apply:
        print("\nRe-run with --apply to write.")


if __name__ == "__main__":
    main()
