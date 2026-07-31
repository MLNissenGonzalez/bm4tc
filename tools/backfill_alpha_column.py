#!/usr/bin/env python3
"""
One-off migration: repair the dead alpha column in the spirals `alpha_curve` CSVs.

`analysis/sweep.py` used to extract `trainer.generative.criterion.kwargs.alpha` into
`config/trainer.generative.criterion.kwargs.alpha`. That path has not existed since the
trainer refactor — alpha lives at `trainer.nll.alpha` (NAT) / `trainer.adversarial.alpha`
(AT) — so the column is all-NaN in every CSV written since. `2dtoy.ipynb` groups the
alpha curve by that column, which silently drops all 50 intermediate runs and leaves a
straight line between the two endpoint sweeps.

`sweep.py` now extracts the live keys, which fixes *future* CSVs. The run directories for
these two sweeps live on mathqi, so the existing CSVs cannot be re-derived locally; this
script backfills them from W&B instead.

The mapping is cross-checked two independent ways before anything is written:

  1. W&B: each run's own `trainer.nll.alpha` and `tracking.seed`, fetched per run
     (`api.runs()` returns runs with an EMPTY `.config` — lazy load — so the config must
     come from `api.run(<path>)`).
  2. The sweep structure implied by
     `configs/experiments/spirals/nat/legendre/d10r6/alpha_curve.yaml`:
     `alpha: choice(...10 values...) x tracking.seed: range(1, 6)`. Hydra's basic sweeper
     takes the cartesian product with the LAST param varying fastest, so job `n` has
     `seed = n % 5 + 1` and a single alpha shared by the block `n // 5`.

and against the CSV's own `config/tracking.seed`. Any mismatch aborts the whole migration.

The two sweeps share one W&B group and are told apart by creation date:
2026-06-02 -> alpha_curve_0206, 2026-06-05 -> alpha_curve_0506.

**They do NOT share an alpha grid.** Only `alpha_curve_0506` matches the config as it stands
today (`0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1`); `alpha_curve_0206` is an older,
log-spaced grid (`0, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 0.8, 1`). So the grid itself is
taken from W&B per sweep; what is checked against the config is the *structure* (10 alphas,
5 seeds each, one alpha per block of 5 consecutive job indices, seeds 1..5 within a block)
and, for 0506 only, that the recovered grid equals the config's.

The new column replaces the dead one in place (same position); every other column is left
byte-identical and that is asserted after the rewrite.

Usage:
    python tools/backfill_alpha_column.py                # dry run (default)
    python tools/backfill_alpha_column.py --apply
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

ENTITY = "martin-nissen-gonzalez-heidelberg-university"
PROJECT = "bm4tc"
EXPERIMENT = "alpha_curve"

DEAD_COL = "config/trainer.generative.criterion.kwargs.alpha"
LIVE_COL = "config/trainer.nll.alpha"
SEED_COL = "config/tracking.seed"

# hydra.sweeper.params in configs/experiments/spirals/nat/legendre/d10r6/alpha_curve.yaml
CONFIG_ALPHAS = [0.0, 1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 7e-1, 9e-1, 1.0]
N_ALPHAS = 10
N_SEEDS = 5

# W&B creation date -> (analysis output directory, grid must equal CONFIG_ALPHAS)
TARGETS: Dict[str, Tuple[str, bool]] = {
    # older, log-spaced grid: 0, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 0.8, 1
    "2026-06-02": ("analysis/outputs/spirals/nat/legendre/d10r6/alpha_curve_0206", False),
    "2026-06-05": ("analysis/outputs/spirals/nat/legendre/d10r6/alpha_curve_0506", True),
}


def check_structure(wb: Dict[str, Tuple[float, int]], label: str,
                    match_config: bool) -> None:
    """Assert the W&B mapping has the shape Hydra's product sweep must produce."""
    jobs = sorted(int(n) for n in wb)
    if jobs != list(range(N_ALPHAS * N_SEEDS)):
        raise SystemExit(f"{label}: job indices are not 0..{N_ALPHAS * N_SEEDS - 1}: {jobs}")

    grid = []
    for block in range(N_ALPHAS):
        entries = [wb[str(block * N_SEEDS + k)] for k in range(N_SEEDS)]
        alphas = {a for a, _ in entries}
        if len(alphas) != 1:
            raise SystemExit(f"{label}: block {block} spans several alphas {alphas}")
        if [s for _, s in entries] != list(range(1, N_SEEDS + 1)):
            raise SystemExit(
                f"{label}: block {block} seeds are {[s for _, s in entries]}, expected 1..5")
        grid.append(alphas.pop())

    if len(set(grid)) != N_ALPHAS:
        raise SystemExit(f"{label}: alpha grid has duplicates: {grid}")
    if grid != sorted(grid):
        raise SystemExit(f"{label}: alpha grid is not ascending in job order: {grid}")
    if match_config and any(abs(g - c) > 1e-12 for g, c in zip(grid, CONFIG_ALPHAS)):
        raise SystemExit(f"{label}: grid {grid} != config grid {CONFIG_ALPHAS}")
    print(f"  {label}: structure OK — grid {grid}")


def fetch_wandb() -> Dict[str, Dict[str, Tuple[float, int]]]:
    """{date: {run_name: (alpha, seed)}} for every `alpha_curve` run in the project."""
    import wandb

    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}",
                         filters={"config.experiment": EXPERIMENT},
                         per_page=200))
    print(f"W&B: {len(runs)} runs with experiment={EXPERIMENT!r}")

    out: Dict[str, Dict[str, Tuple[float, int]]] = {}
    for i, r in enumerate(runs, 1):
        date = r.created_at[:10]
        if date not in TARGETS:
            continue
        # .config is empty on listed runs; refetch the run to get it.
        cfg = api.run(f"{ENTITY}/{PROJECT}/{r.id}").config
        alpha = cfg["trainer"]["nll"]["alpha"]
        seed = cfg["tracking"]["seed"]
        out.setdefault(date, {})[r.name] = (float(alpha), int(seed))
        if i % 20 == 0:
            print(f"  fetched {i}/{len(runs)} configs")
    return out


def check_and_build(csv_path: Path, wb: Dict[str, Tuple[float, int]]) -> pd.Series:
    """Validate every row against W&B and the config-derived mapping; return the α column."""
    df = pd.read_csv(csv_path)

    if LIVE_COL in df.columns:
        raise SystemExit(f"{csv_path}: already carries {LIVE_COL}; refusing to touch it")
    if DEAD_COL not in df.columns:
        raise SystemExit(f"{csv_path}: no {DEAD_COL} column to replace")
    if df[DEAD_COL].notna().any():
        raise SystemExit(f"{csv_path}: {DEAD_COL} is NOT all-NaN; refusing to overwrite data")

    alphas = []
    for _, row in df.iterrows():
        name = str(row["run_name"])

        if name not in wb:
            raise SystemExit(f"{csv_path}: run {name!r} not found in W&B group")
        w_alpha, w_seed = wb[name]

        if w_seed != int(name) % N_SEEDS + 1:
            raise SystemExit(
                f"{csv_path} run {name}: W&B seed {w_seed} != product-order seed "
                f"{int(name) % N_SEEDS + 1}")
        if int(row[SEED_COL]) != w_seed:
            raise SystemExit(
                f"{csv_path} run {name}: CSV seed {row[SEED_COL]} != W&B seed {w_seed}")
        alphas.append(w_alpha)

    return pd.Series(alphas, index=df.index, name=LIVE_COL)


def migrate(csv_path: Path, wb: Dict[str, Tuple[float, int]], apply: bool) -> None:
    print(f"\n--- {csv_path.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(csv_path)
    alpha = check_and_build(csv_path, wb)

    counts = alpha.value_counts().sort_index()
    print(f"  {len(df)} rows validated against W&B + sweep config")
    print(f"  alpha grid: {[float(a) for a in counts.index.tolist()]}")
    print(f"  runs per alpha: {sorted(set(counts.tolist()))}")
    print(f"  {DEAD_COL}  ->  {LIVE_COL}")

    if not apply:
        print("  DRY RUN — nothing written")
        return

    pos = df.columns.get_loc(DEAD_COL)
    out = df.drop(columns=[DEAD_COL])
    out.insert(pos, LIVE_COL, alpha)
    out.to_csv(csv_path, index=False)

    # Re-read and assert nothing but the alpha column moved.
    back = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(
        back.drop(columns=[LIVE_COL]), df.drop(columns=[DEAD_COL]), check_exact=False
    )
    assert back[LIVE_COL].notna().all() and len(back[LIVE_COL].unique()) == N_ALPHAS
    print(f"  WROTE — other columns verified unchanged")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--apply", action="store_true", help="write the CSVs (default: dry run)")
    args = ap.parse_args()

    wb = fetch_wandb()
    for date, (rel, match_config) in TARGETS.items():
        runs = wb.get(date, {})
        if len(runs) != N_ALPHAS * N_SEEDS:
            raise SystemExit(
                f"{date}: expected {N_ALPHAS * N_SEEDS} W&B runs, found {len(runs)}")
        check_structure(runs, date, match_config)
        migrate(PROJECT_ROOT / rel / "evaluation_data.csv", runs, args.apply)

    if not args.apply:
        print("\nRe-run with --apply to write.")


if __name__ == "__main__":
    main()
