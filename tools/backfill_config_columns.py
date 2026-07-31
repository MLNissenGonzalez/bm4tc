#!/usr/bin/env python3
"""
Backfill `config/*` columns into analysis CSVs from W&B.

`analysis/sweep.py` writes one `config/<dotted hydra key>` column per entry in its
`CONFIG_KEYS`. When that list gains a key — or, historically, when a key in it turned out to
be a dead path — CSVs written earlier lack the column and every reader has to special-case
them. This script fills them in retroactively, so one schema covers the repo.

The run directories live on mathqi, so the configs cannot be re-read locally. W&B is the
source of truth: each CSV's `run_path` column names the sweep, whose tail after `outputs/`
is the W&B group, and each row's `run_name` is the run name within it.

Two passes, both driven off the same resolved rows:

  ALPHA (a replacement).  `config/trainer.generative.criterion.kwargs.alpha` has not existed
  since the trainer refactor and is all-NaN wherever it appears. It is replaced in place by
  `config/trainer.nll.alpha` + `config/trainer.adversarial.alpha`, the inactive one empty —
  the shape `sweep.py` now produces. All 65 affected CSVs were migrated on 2026-07-31; the
  pass stays because it is the only record of the mapping and makes the script idempotent.

  ENSURE (an addition).  For each `--keys` entry, append `config/<key>` if it is missing.
  Default: `descriptor`, `model_path` — the warm/cold discriminator and its evidence.

Every row is cross-checked before anything is written, and any failure skips the WHOLE file
(never a partial rewrite):

  * all rows agree on one sweep / W&B group;
  * every CSV row has a matching W&B run. A relaunch leaves several runs sharing a name;
    they are told apart by the seed the CSV also records, and only a genuine (name, seed)
    collision is fatal;
  * exactly one of `trainer.nll.alpha` / `trainer.adversarial.alpha` is set per run —
    except for AT runs predating 27ac8c1 (2026-06-24), whose `adversarial` node has no
    `alpha` field at all because that trainer hardcoded `mixed_nll(..., alpha=0.0)`;
    those are alpha=0 by construction and are filled as such;
  * the W&B seed equals the CSV's own `config/tracking.seed` where present;
  * a dead alpha column, if present, is entirely NaN — nothing is overwritten.

Two naming quirks this has to absorb: older sweeps log a group carrying an extra `_HHMM`
launch-time suffix that the output directory name drops (dir `alpha_curve_0506` <-> group
`alpha_curve_0506_1501`), and the archived pre-refactor layout under
`analysis/outputs/seed_sweep/{cls,gen,adv,comb,cls_reg}/…/d10D6/` has no W&B counterpart at
all — those are reported and skipped, never guessed.

`baselines/jem/` is excluded: JEM has its own columns and no hydra trainer node.

Usage:
    python tools/backfill_config_columns.py                       # dry run over every CSV
    python tools/backfill_config_columns.py --apply
    python tools/backfill_config_columns.py --keys descriptor --apply
    python tools/backfill_config_columns.py --root analysis/outputs/mnist_full_r12 --apply
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

ENTITY = "martin-nissen-gonzalez-heidelberg-university"
PROJECT = "bm4tc"

DEAD_COL = "config/trainer.generative.criterion.kwargs.alpha"
NLL_COL = "config/trainer.nll.alpha"
ADV_COL = "config/trainer.adversarial.alpha"
SEED_COL = "config/tracking.seed"

DEFAULT_KEYS = ["descriptor", "model_path"]

EXCLUDE_PARTS = ("baselines",)

# 27ac8c1 "feat(at): adversarial alpha>0 support". Adversarial runs created before this
# had no alpha field and a hardcoded alpha=0 objective.
AT_ALPHA_COMMIT_DATE = "2026-06-24"


class Unresolved(Exception):
    """This file cannot be migrated safely; report it and leave it alone."""


def find_targets(root: Path, keys: List[str]) -> List[Path]:
    """Every `evaluation_data.csv` under `root` that either carries the dead alpha column
    or is missing one of the requested `config/<key>` columns."""
    wanted = {f"config/{k}" for k in keys}
    out = []
    for path in sorted(root.rglob("evaluation_data.csv")):
        if any(part in EXCLUDE_PARTS for part in path.parts):
            continue
        header = set(pd.read_csv(path, nrows=0).columns)
        if DEAD_COL in header or wanted - header:
            out.append(path)
    return out


def group_of(df: pd.DataFrame) -> str:
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


def _select(cfg: dict, dotted: str) -> Any:
    """`OmegaConf.select` over the plain dict W&B hands back."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def fetch_group(api, group: str, keys: List[str], cache: Dict[str, dict],
                ddmm: Optional[str] = None) -> Dict[str, Dict[int, dict]]:
    """{run_name: {seed: {"alpha_col", "alpha", "values"}}} for one W&B group.

    `api.runs()` returns runs with an empty `.config` (lazy load), so each run's config
    has to be fetched individually via `api.run()`.
    """
    cache_key = f"{group}|{ddmm or ''}"
    for key in (cache_key, group):
        if key in cache:
            return {name: {int(s): v for s, v in seeds.items()}
                    for name, seeds in cache[key].items()}

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
        if not listed and ddmm is not None:
            # Last resort: two launches of one sweep can share a single W&B group named
            # after only one of the dates (the spirals alpha_curve pair both live in
            # `alpha_curve_0506_1501`). Look at sibling groups of the same sweep family
            # and keep the runs created on this directory's date. `resolve_rows` still
            # has to find every CSV row among them, so a wrong family cannot slip through.
            family = re.sub(r"_[0-9]{4}$", "", group)
            sibling = f"^{re.escape(family)}_[0-9]{{4}}(_[0-9]{{4}})?$"
            candidates = list(api.runs(f"{ENTITY}/{PROJECT}",
                                       filters={"group": {"$regex": sibling}}, per_page=200))
            listed = [r for r in candidates if _ddmm_of(r.created_at) == ddmm]
            if listed:
                print(f"    (no group {group!r}; took {len(listed)}/{len(candidates)} runs "
                      f"dated {ddmm} from {sorted({r.group for r in listed})})")
        if not listed:
            raise Unresolved(f"no W&B runs in group {group!r} (nor {group}_HHMM)")
        if found:
            print(f"    (matched W&B group {found[0]!r})")

    fetched = [(r, api.run(f"{ENTITY}/{PROJECT}/{r.id}").config) for r in listed]
    try:
        out = _entries(fetched, keys)
    except Unresolved:
        # Two sweeps can share one W&B group (the spirals alpha_curve pair both log to
        # `alpha_curve_0506_1501`), so run names collide. The directory's DDMM token says
        # which launch this CSV is; keep only the runs created that day.
        if ddmm is None:
            raise
        subset = [(r, c) for r, c in fetched if _ddmm_of(r.created_at) == ddmm]
        if not subset:
            raise
        out = _entries(subset, keys)
        print(f"    (disambiguated {len(subset)}/{len(fetched)} runs by dir date {ddmm})")

    cache[cache_key] = {name: {str(s): v for s, v in seeds.items()}
                        for name, seeds in out.items()}
    return out


def _ddmm_of(created_at: str) -> str:
    """W&B `created_at` (YYYY-MM-DDTHH:MM:SSZ) as the DDMM token used in directory names."""
    return created_at[8:10] + created_at[5:7]


def _entries(fetched: List[Tuple[Any, dict]], keys: List[str]) -> Dict[str, Dict[int, dict]]:
    """{run_name: {seed: entry}} from already-fetched (run, config) pairs."""
    out: Dict[str, Dict[int, dict]] = {}
    for r, cfg in fetched:
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
        entry = {
            "alpha_col": NLL_COL if nll_alpha is not None else ADV_COL,
            "alpha": float(nll_alpha if nll_alpha is not None else adv_alpha),
            "values": {k: _select(cfg, k) for k in keys},
        }
        seed = int((cfg.get("tracking") or {}).get("seed", -1))
        # A relaunch leaves several runs sharing a name; they are told apart by seed,
        # which the CSV also records. Only a genuine (name, seed) collision is fatal.
        by_seed = out.setdefault(r.name, {})
        if seed in by_seed and by_seed[seed] != entry:
            raise Unresolved(f"W&B runs named {r.name!r} at seed {seed} disagree")
        by_seed[seed] = entry
    return out


def resolve_rows(df: pd.DataFrame, wb: Dict[str, Dict[int, dict]]) -> List[dict]:
    """The W&B entry for each CSV row, matched on run name and seed."""
    entries = []
    for _, row in df.iterrows():
        name = str(row["run_name"])
        if name not in wb:
            raise Unresolved(f"run {name!r} has no W&B counterpart in the group")
        by_seed = wb[name]
        csv_seed = row.get(SEED_COL)

        if len(by_seed) == 1:
            (seed, entry), = by_seed.items()
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
            entry = by_seed[int(csv_seed)]
        entries.append(entry)
    return entries


def alpha_columns(df: pd.DataFrame, entries: List[dict]) -> Tuple[pd.Series, pd.Series, str]:
    """(nll_column, adv_column, regime) for the ALPHA replacement pass."""
    if df[DEAD_COL].notna().any():
        raise Unresolved("dead alpha column is NOT all-NaN; refusing to overwrite data")

    columns_seen = {e["alpha_col"] for e in entries}
    if len(columns_seen) != 1:
        raise Unresolved(f"sweep mixes NAT and AT runs: {sorted(columns_seen)}")
    col = columns_seen.pop()

    vals = [e["alpha"] for e in entries]
    empty = [None] * len(entries)
    if col == NLL_COL:
        return (pd.Series(vals, index=df.index, dtype="float64"),
                pd.Series(empty, index=df.index, dtype="float64"), "nat")
    return (pd.Series(empty, index=df.index, dtype="float64"),
            pd.Series(vals, index=df.index, dtype="float64"), "at")


def migrate(path: Path, api, keys: List[str], cache: Dict[str, dict],
            apply: bool) -> Optional[str]:
    """Migrate one CSV. Returns a one-line report, or None when already up to date."""
    rel = path.relative_to(PROJECT_ROOT)
    df = pd.read_csv(path)
    missing = [k for k in keys if f"config/{k}" not in df.columns]
    do_alpha = DEAD_COL in df.columns
    if not missing and not do_alpha:
        return None

    group = group_of(df)
    # Directory names end in the launch date; it disambiguates a shared W&B group.
    ddmm = path.parent.name.rsplit("_", 1)[-1]
    wb = fetch_group(api, group, keys, cache, ddmm if ddmm.isdigit() else None)
    entries = resolve_rows(df, wb)

    notes = []
    out = df.copy()
    if do_alpha:
        nll_col, adv_col, regime = alpha_columns(df, entries)
        pos = out.columns.get_loc(DEAD_COL)
        out = out.drop(columns=[DEAD_COL])
        out.insert(pos, ADV_COL, adv_col)
        out.insert(pos, NLL_COL, nll_col)
        alphas = sorted({e["alpha"] for e in entries})
        notes.append(f"alpha={alphas} ({regime})")
    for key in missing:
        col = pd.Series([e["values"].get(key) for e in entries], index=df.index)
        out[f"config/{key}"] = col
        seen = sorted({str(v) for v in col.dropna().unique()})
        notes.append(f"{key}={seen if seen else '<all empty>'}")

    report = f"{rel}\n    group={group}  rows={len(df)}  " + "  ".join(notes)
    if not apply:
        return report + "  [dry run]"

    out.to_csv(path, index=False)

    # Nothing but the intended columns may have moved.
    back = pd.read_csv(path)
    added = [f"config/{k}" for k in missing] + ([NLL_COL, ADV_COL] if do_alpha else [])
    pd.testing.assert_frame_equal(
        back.drop(columns=added),
        df.drop(columns=[DEAD_COL]) if do_alpha else df,
        check_exact=False,
    )
    return report + "  [written]"


def main() -> None:
    import wandb

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--apply", action="store_true", help="write the CSVs (default: dry run)")
    ap.add_argument("--root", default="analysis/outputs", help="subtree to scan")
    ap.add_argument("--keys", default=",".join(DEFAULT_KEYS),
                    help="comma-separated hydra keys to ensure a config/<key> column for")
    ap.add_argument("--cache", default=None,
                    help="JSON file of fetched W&B configs, reused across invocations")
    args = ap.parse_args()

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    root = (PROJECT_ROOT / args.root).resolve()
    targets = find_targets(root, keys)
    print(f"{len(targets)} CSV(s) under {root.relative_to(PROJECT_ROOT)} need "
          f"{keys} (or carry the dead alpha column)\n")
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
                report = migrate(path, api, keys, cache, args.apply)
                if report is None:
                    continue
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
