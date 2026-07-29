# %% [markdown]
# # Gibbs Purification Sweep Analysis
#
# Standalone counterpart to `analysis/sweep.py`, split out because Gibbs purification
# is orders of magnitude more expensive than every other post-hoc metric and is worth
# running (and re-running) on its own schedule.
#
# **What it does:** for each run in a seed sweep, generate PGD adversarial examples at
# the same attack strengths `sweep.py` uses, then purify them with class-marginalized
# Gibbs sampling for 1, 3 and 6 sweeps and measure accuracy recovery.
#
# **Attack-radius agnostic.** The Gibbs step radius is a *per-sweep* L∞ move
# (default 0.05 of the feature domain), not a global perturbation ball: the window
# re-centres at the start of every sweep, so after k sweeps a coordinate can have
# travelled up to k × step_radius × (hi − lo). Purification strength is set by the
# number of sweeps alone, so the defense never has to be told the attacker's budget.
# Each result column therefore reports a k, not a radius.
#
# **Outputs** (mirrors `sweep.py`'s path convention, distinct filenames so both can
# coexist in the same directory):
#   analysis/outputs/{rel}/gibbs_data.csv      one row per run
#   analysis/outputs/{rel}/gibbs_summary.txt   mean ± std table
#
# CLI usage:
#     python analysis/gibbs.py <sweep_dir> [--n-eval N] [--sweeps 1,3,6] ...

# %%
import sys
import argparse
import logging
import time
from pathlib import Path

if "__file__" in dir():
    project_root = Path(__file__).parent.parent
else:
    project_root = Path.cwd().parent
    if not (project_root / "src").exists():
        project_root = Path.cwd()

sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch

from src.utils.paths import data_root as _data_root
from analysis.utils.resolve import (
    resolve_embedding_from_path as _resolve_embedding,
    resolve_regime_from_path as _resolve_regime_from_path,
    embedding_range_size as _embedding_range_size,
)

logger = logging.getLogger(__name__)

# %%
# =============================================================================
# CONFIGURATION - EDIT THIS SECTION
# =============================================================================

SWEEP_DIR = "outputs/moons/nat/legendre/d10r6/seed_sweep_a1_2707"

# --- COST CONTROL: how many test samples Gibbs is evaluated on ---------------
# Gibbs is essentially all of this script's cost and scales *linearly* in this
# number, so it is the primary knob. The same fixed subsample is reused for the
# clean row and every epsilon, and is seeded so it is identical across model
# seeds and alphas — every comparison in the output table is therefore paired.
# Sampling error on a reported mean accuracy is about 0.5/sqrt(N_EVAL):
#   N_EVAL=100 -> ~±5%,  N_EVAL=1000 -> ~±1.6%,  N_EVAL=4000 -> ~±0.8%.
# None (or --n-eval all) = use the full test split.
N_EVAL = 1000
N_EVAL_SEED = 0
# -----------------------------------------------------------------------------

# --- ATTACK CONFIG (kept in step with analysis/sweep.py) ---------------------
# Fractions of the embedding range size; converted to absolute below.
_STRENGTH_FRACTIONS = [0.05, 0.1, 0.15]
ATTACK = {
    "method": "PGD",
    "norm": "inf",
    "num_steps": 40,
}

# --- GIBBS CONFIG ------------------------------------------------------------
GIBBS = {
    # Purification strength knob. Cost is linear in max(n_sweeps).
    "n_sweeps": [1, 3, 6],
    # Candidate values per feature. With a step_radius these all land *inside*
    # the window, so this is the resolution of the window, not of the domain.
    "num_bins": 96,
    # Candidate-expansion peak ∝ gibbs_batch_size × num_bins. On a 24 GB GPU, 24
    # is safe for resized MNIST (r12) at num_bins=96; drop to ~8 for full-res.
    "gibbs_batch_size": 24,
    # Per-sweep L∞ step as a fraction of the feature domain. Deliberately small:
    # strength comes from n_sweeps, not from this.
    "step_radius": 0.1,
}

# Chunk size for the non-Gibbs forwards (class probs, log p(x), attack generation).
EVAL_BATCH_SIZE = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# --- CLI overrides -----------------------------------------------------------
_cli = argparse.ArgumentParser(
    add_help=True,
    description="Gibbs-purification analysis for a seed sweep (see module docstring).",
)
_cli.add_argument("sweep_dir", nargs="?", default=None)
_cli.add_argument(
    "--n-eval", default=None,
    help="Test samples to evaluate Gibbs on — THE cost knob (cost is linear in it). "
         f"'all' or 0 = full test split. Default: {N_EVAL}.",
)
_cli.add_argument("--n-eval-seed", type=int, default=None,
                  help="Seed for the evaluation subsample. Keep fixed across runs "
                       "you intend to compare.")
_cli.add_argument("--sweeps", default=None,
                  help="Comma-separated Gibbs sweep counts, e.g. '1,3,6'.")
_cli.add_argument("--step-radius", type=float, default=None,
                  help="Per-sweep L-inf step as a fraction of the feature domain.")
_cli.add_argument("--num-bins", type=int, default=None,
                  help="Candidate values evaluated per feature.")
_cli.add_argument("--batch-size", type=int, default=None,
                  help="Gibbs batch size (memory: batch-size x num-bins per forward).")
_cli.add_argument("--strengths", default=None,
                  help="Comma-separated attack strengths as FRACTIONS of the "
                       "embedding range, e.g. '0.05,0.1,0.15'.")
_cli.add_argument("--device", default=None)
_cli.add_argument("--limit-runs", type=int, default=None,
                  help="Only evaluate the first N runs in the sweep directory.")
_cli_args, _ = _cli.parse_known_args()

if _cli_args.sweep_dir is not None:
    SWEEP_DIR = _cli_args.sweep_dir
if _cli_args.n_eval is not None:
    _raw = str(_cli_args.n_eval).strip().lower()
    N_EVAL = None if _raw in ("all", "0", "none", "") else int(_raw)
if _cli_args.n_eval_seed is not None:
    N_EVAL_SEED = _cli_args.n_eval_seed
if _cli_args.sweeps is not None:
    GIBBS["n_sweeps"] = [int(s) for s in _cli_args.sweeps.split(",") if s.strip()]
if _cli_args.step_radius is not None:
    GIBBS["step_radius"] = _cli_args.step_radius
if _cli_args.num_bins is not None:
    GIBBS["num_bins"] = _cli_args.num_bins
if _cli_args.batch_size is not None:
    GIBBS["gibbs_batch_size"] = _cli_args.batch_size
if _cli_args.strengths is not None:
    _STRENGTH_FRACTIONS = [float(s) for s in _cli_args.strengths.split(",") if s.strip()]
if _cli_args.device is not None:
    DEVICE = _cli_args.device

# %%
# --- Resolve embedding / regime from the sweep path --------------------------
REGIME = _resolve_regime_from_path(SWEEP_DIR)
if REGIME is None:
    print(f"WARNING: Could not auto-detect training regime from '{SWEEP_DIR}'.")
    REGIME = "unknown"

_EMBEDDING = _resolve_embedding(SWEEP_DIR)
if _EMBEDDING is None:
    print(f"WARNING: Could not detect embedding from '{SWEEP_DIR}'. "
          "Assuming Fourier (range size 1.0).")
_RANGE_SIZE = _embedding_range_size(_EMBEDDING)

# Attack strengths are ABSOLUTE from here on (fraction x range size), matching
# the convention in analysis/run.py and analysis/sweep.py.
ATTACK["strengths"] = [s * _RANGE_SIZE for s in _STRENGTH_FRACTIONS]

SWEEP_POINTS = sorted({int(s) for s in GIBBS["n_sweeps"]})

print("=" * 68)
print(f"Gibbs purification analysis: {SWEEP_DIR}")
print("=" * 68)
print(f"Regime: {REGIME}  |  Embedding: {_EMBEDDING or 'unknown'} "
      f"(range size {_RANGE_SIZE})  |  Device: {DEVICE}")
print(f"Attack: {ATTACK['method']} L{ATTACK['norm']} steps={ATTACK['num_steps']}  "
      f"strengths={[round(s, 6) for s in ATTACK['strengths']]} "
      f"(fractions {_STRENGTH_FRACTIONS})")
print(f"Gibbs:  sweeps={SWEEP_POINTS}  step_radius={GIBBS['step_radius']} "
      f"num_bins={GIBBS['num_bins']}  batch={GIBBS['gibbs_batch_size']}")
print(f"N_EVAL: {N_EVAL if N_EVAL is not None else 'full test split'} "
      f"(seed {N_EVAL_SEED})")

# %% [markdown]
# ## Per-run evaluation

# %%
from analysis.utils.mia_utils import load_run_config, find_model_checkpoint
from src.analysis.uq import _batched_forward, _recover_after_failure


def _subsample_indices(n: int, n_eval, seed: int):
    """Fixed random subsample of `range(n)`, or None to use everything.

    Seeded independently of any model/data seed so the SAME test samples are drawn
    for every run in the sweep and for every attack strength within a run — which
    is what makes the clean/adv/purified columns paired rather than independent.
    """
    if n_eval is None or n_eval >= n:
        return None
    rng = np.random.default_rng(seed)
    return torch.from_numpy(np.sort(rng.permutation(n)[:n_eval]))


def _accuracy(born, x, labels, device):
    preds = _batched_forward(born.class_probabilities, x, EVAL_BATCH_SIZE, device).argmax(dim=1)
    return preds == labels


def _mean_log_px(born, x, device):
    return float(_batched_forward(born.marginal_log_probability, x, EVAL_BATCH_SIZE, device).mean())


def _generate_adversarial(attack, born, x, labels, eps, device):
    """Batched PGD generation over an already-subsampled tensor."""
    out = []
    for i in range(0, len(x), EVAL_BATCH_SIZE):
        xb = x[i:i + EVAL_BATCH_SIZE].to(device)
        yb = labels[i:i + EVAL_BATCH_SIZE].to(device)
        out.append(attack.generate(born, xb, yb, eps, device).detach().cpu())
    return torch.cat(out)


def gibbs_analyze_run(run_dir: Path) -> dict:
    """Load one run's model and measure Gibbs purification on clean + adversarial data.

    Returns a flat dict of metrics. Column names reuse the conventions already
    emitted by analysis/run.py (`gibbs_purify_acc/{eps}/{k}` etc.) so downstream
    readers need no special-casing; separation from sweep.py is by filename.
    """
    from src.model import ConditionalBornMachine
    from src.datahandler import DataHandler
    from src.utils.evasion import RobustnessEvaluation
    from src.utils.train import CriterionConfig
    from src.analysis.purification import GibbsPurification
    from omegaconf import OmegaConf

    device = torch.device(DEVICE)
    results: dict = {}

    # 1. Model — mirrors analysis/run.py: accumulate=True forces the overflow-safe
    #    amplitude path regardless of what the checkpoint's config says.
    run_cfg = load_run_config(run_dir)
    cbm = ConditionalBornMachine.load(str(find_model_checkpoint(run_dir)), accumulate=True)
    cbm.to(device)
    cbm.eval()

    # 2. Data
    OmegaConf.update(run_cfg, "dataset.overwrite", True, force_add=True)
    datahandler = DataHandler(run_cfg.dataset)
    datahandler.load()
    datahandler.split_and_rescale(cbm)
    datahandler.get_classification_loaders()
    loader = datahandler.classification["test"]

    x_test = torch.cat([b for b, _ in loader])
    y_test = torch.cat([lb for _, lb in loader])

    # 3. Subsample ONCE, before the attack loop, so PGD generation is charged on
    #    N_EVAL samples too rather than on the whole split.
    idx = _subsample_indices(len(x_test), N_EVAL, N_EVAL_SEED)
    if idx is not None:
        x_test, y_test = x_test[idx], y_test[idx]

    results["gibbs_n_samples"] = len(x_test)
    results["gibbs_n_eval_seed"] = N_EVAL_SEED
    results["gibbs_step_radius"] = GIBBS["step_radius"]
    results["gibbs_num_bins"] = GIBBS["num_bins"]

    cbm.cache_log_Z()

    purifier = GibbsPurification(
        num_bins=GIBBS["num_bins"],
        gibbs_batch_size=GIBBS["gibbs_batch_size"],
        step_radius=GIBBS["step_radius"],
    )

    # 4. Clean reference: accuracy with no attack, before and after purification.
    #    Establishes the cost of purifying something that did not need it.
    try:
        clean_correct = _accuracy(cbm, x_test, y_test, device)
        results["gibbs_clean_acc"] = clean_correct.float().mean().item()
        results["gibbs_clean_log_px_mean"] = _mean_log_px(cbm, x_test, device)

        for k, (x_pur, log_px_after) in purifier.purify_snapshots(
            cbm, x_test, SWEEP_POINTS, device
        ).items():
            results[f"gibbs_clean_purify_acc/{k}"] = (
                _accuracy(cbm, x_pur, y_test, device).float().mean().item()
            )
            results[f"gibbs_clean_log_px_mean/{k}"] = float(log_px_after.mean())
    except Exception as e:
        logger.warning(f"Clean Gibbs failed: {e}; skipping")
        _recover_after_failure(cbm)

    # 5. Per attack strength: attack, score undefended, purify, score again.
    attack = RobustnessEvaluation(
        method=ATTACK["method"],
        norm=ATTACK["norm"],
        criterion=CriterionConfig(name="nll", kwargs=None),
        strengths=ATTACK["strengths"],
        num_steps=ATTACK["num_steps"],
        random_start=True,
    )

    for eps in ATTACK["strengths"]:
        try:
            x_adv = _generate_adversarial(attack, cbm, x_test, y_test, eps, device)
            adv_correct = _accuracy(cbm, x_adv, y_test, device)
            misclassified = ~adv_correct
            results[f"gibbs_adv_acc/{eps}"] = adv_correct.float().mean().item()
            results[f"gibbs_adv_log_px_mean/{eps}"] = _mean_log_px(cbm, x_adv, device)

            snapshots = purifier.purify_snapshots(cbm, x_adv, SWEEP_POINTS, device)
        except Exception as e:
            logger.warning(f"Gibbs failed (eps={eps}): {e}; skipping")
            _recover_after_failure(cbm)
            continue

        for k, (x_pur, log_px_after) in snapshots.items():
            try:
                correct_after = _accuracy(cbm, x_pur, y_test, device)
                n_misclf = int(misclassified.sum())
                results[f"gibbs_purify_acc/{eps}/{k}"] = correct_after.float().mean().item()
                results[f"gibbs_purify_recovery/{eps}/{k}"] = (
                    float((misclassified & correct_after).sum()) / n_misclf
                    if n_misclf > 0 else 1.0
                )
                results[f"gibbs_purify_log_px_mean/{eps}/{k}"] = float(log_px_after.mean())
            except Exception as e:
                logger.warning(f"Gibbs scoring failed (eps={eps}, k={k}): {e}; skipping")
                _recover_after_failure(cbm)

    return results


# %%
sweep_path = _data_root() / SWEEP_DIR
run_dirs = sorted(
    [d for d in sweep_path.iterdir()
     if d.is_dir() and (d / ".hydra" / "config.yaml").exists()],
    key=lambda d: d.name,
) if sweep_path.exists() else []

if _cli_args.limit_runs is not None:
    run_dirs = run_dirs[:_cli_args.limit_runs]

if not run_dirs:
    print(f"No valid run directories found in {sweep_path}")

# %%
# --- Projected cost, printed before any work starts --------------------------
if run_dirs:
    _n_cond = 1 + len(ATTACK["strengths"])         # clean + one per epsilon
    _n_eff = N_EVAL if N_EVAL is not None else "N"
    print()
    print(f"Found {len(run_dirs)} runs.")
    if N_EVAL is not None:
        _per_cond = N_EVAL * GIBBS["num_bins"] * max(SWEEP_POINTS)
        print(f"Projected cost: {_per_cond:,} x data_dim candidate contractions per "
              f"condition, {_n_cond} conditions per run, {len(run_dirs)} runs.")
        print(f"  (= N_EVAL {N_EVAL} x num_bins {GIBBS['num_bins']} x "
              f"max_sweeps {max(SWEEP_POINTS)}; linear in N_EVAL — halve it to halve "
              f"the runtime.)")
    else:
        print(f"Projected cost: full test split x {GIBBS['num_bins']} bins x "
              f"{max(SWEEP_POINTS)} sweeps x data_dim, {_n_cond} conditions per run.")
    print()

# %%
rows = []
_t_start = time.time()
for i, run_dir in enumerate(run_dirs):
    print(f"  [{i + 1}/{len(run_dirs)}] {run_dir.name} ...", flush=True)
    row = {"run_name": run_dir.name, "run_path": str(run_dir)}
    _t0 = time.time()
    try:
        row.update(gibbs_analyze_run(run_dir))
    except Exception as e:
        print(f"    FAILED: {e}")
        logger.warning(f"Run {run_dir.name} failed: {e}")
    row["gibbs_runtime_s"] = time.time() - _t0
    print(f"    done in {row['gibbs_runtime_s']:.1f}s")
    rows.append(row)

df = pd.DataFrame(rows)
if not df.empty:
    print(f"\nEvaluation complete: {len(df)} runs in {time.time() - _t_start:.1f}s total.")

# %% [markdown]
# ## Export

# %%
# Mirror the sweep path under analysis/outputs/ (same convention as sweep.py).
_sp = Path(SWEEP_DIR)
if _sp.is_absolute():
    try:
        _sp = _sp.relative_to(_data_root())
    except ValueError:
        pass
try:
    _rel = _sp.relative_to("outputs")
except ValueError:
    _rel = _sp
output_dir = _data_root() / "analysis" / "outputs" / _rel
sweep_name = str(_rel)

if not df.empty:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

# %%
if not df.empty:
    def _smean(col):
        return df[col].dropna().mean() if col in df.columns and df[col].notna().any() else float("nan")

    def _sstd(col):
        return df[col].dropna().std() if col in df.columns and df[col].notna().any() else float("nan")

    def _sstderr(col):
        if col not in df.columns:
            return float("nan")
        v = df[col].dropna()
        return v.std() / (len(v) ** 0.5) if len(v) > 1 else float("nan")

    def _fmt(v, w=10):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—".rjust(w)
        return f"{v:.4f}".rjust(w)

    _eps_list = ATTACK["strengths"]
    _cols = ["eps=0"] + [f"{e:.4g}" for e in _eps_list]

    # Rows: undefended, then one per sweep count.
    _table = [("No defense", [_smean("gibbs_clean_acc")]
                             + [_smean(f"gibbs_adv_acc/{e}") for e in _eps_list])]
    for k in SWEEP_POINTS:
        _table.append((
            f"Gibbs (k={k})",
            [_smean(f"gibbs_clean_purify_acc/{k}")]
            + [_smean(f"gibbs_purify_acc/{e}/{k}") for e in _eps_list],
        ))

    _label_w = max(len(r[0]) for r in _table) + 2
    _col_w = 10

    summary_path = output_dir / "gibbs_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 68 + "\n")
        f.write(f"Gibbs Purification: {sweep_name}\n")
        f.write("=" * 68 + "\n\n")
        f.write(f"Regime: {REGIME}  |  Runs: {len(df)}  |  Device: {DEVICE}\n")
        f.write(f"Embedding: {_EMBEDDING or 'unknown'}  (range size {_RANGE_SIZE})\n")
        f.write(f"Attack: {ATTACK['method']} L{ATTACK['norm']} "
                f"steps={ATTACK['num_steps']}  fractions={_STRENGTH_FRACTIONS}\n")
        _n_used = df["gibbs_n_samples"].dropna()
        f.write(f"N_EVAL: {int(_n_used.iloc[0]) if len(_n_used) else 'n/a'} test samples "
                f"(seed {N_EVAL_SEED})"
                f"{'' if N_EVAL is not None else '  [full test split]'}\n")
        f.write(f"Gibbs: sweeps={SWEEP_POINTS}  step_radius={GIBBS['step_radius']} "
                f"(per-sweep L-inf step, NOT a global budget: after k sweeps the\n")
        f.write(f"       envelope is k x {GIBBS['step_radius']} x range)  "
                f"num_bins={GIBBS['num_bins']}\n\n")

        f.write("-" * 68 + "\n")
        f.write("Accuracy vs Perturbation Strength (mean across runs)\n")
        f.write("-" * 68 + "\n\n")
        f.write(" " * _label_w + "".join(c.rjust(_col_w) for c in _cols) + "\n")
        for _lbl, _vals in _table:
            f.write(f"{_lbl:<{_label_w}}" + "".join(_fmt(v) for v in _vals) + "\n")
        f.write("\n")

        f.write("-" * 68 + "\n")
        f.write("Across-run std / stderr (same layout)\n")
        f.write("-" * 68 + "\n\n")
        for _stat, _fn in (("std", _sstd), ("stderr", _sstderr)):
            f.write(f"{_stat}:\n")
            f.write(" " * _label_w + "".join(c.rjust(_col_w) for c in _cols) + "\n")
            _rows_stat = [("No defense", [_fn("gibbs_clean_acc")]
                                         + [_fn(f"gibbs_adv_acc/{e}") for e in _eps_list])]
            for k in SWEEP_POINTS:
                _rows_stat.append((
                    f"Gibbs (k={k})",
                    [_fn(f"gibbs_clean_purify_acc/{k}")]
                    + [_fn(f"gibbs_purify_acc/{e}/{k}") for e in _eps_list],
                ))
            for _lbl, _vals in _rows_stat:
                f.write(f"{_lbl:<{_label_w}}" + "".join(_fmt(v) for v in _vals) + "\n")
            f.write("\n")
        f.write("Note: the above is spread ACROSS RUNS. Within a single run, the\n")
        f.write(f"binomial sampling error on each accuracy is about "
                f"{0.5 / max(int(_n_used.iloc[0]) if len(_n_used) else 1, 1) ** 0.5:.4f} "
                f"at this N_EVAL.\n\n")

        f.write("-" * 68 + "\n")
        f.write("Recovery rate (fraction of misclassified adv examples fixed)\n")
        f.write("-" * 68 + "\n\n")
        f.write(" " * _label_w + "".join(f"{e:.4g}".rjust(_col_w) for e in _eps_list) + "\n")
        for k in SWEEP_POINTS:
            f.write(f"{f'Gibbs (k={k})':<{_label_w}}"
                    + "".join(_fmt(_smean(f"gibbs_purify_recovery/{e}/{k}")) for e in _eps_list)
                    + "\n")
        f.write("\n")

        if "gibbs_runtime_s" in df.columns:
            _rt = df["gibbs_runtime_s"].dropna()
            if len(_rt):
                f.write(f"Runtime: {_rt.mean():.1f}s per run (total {_rt.sum():.1f}s)\n")
        f.write("=" * 68 + "\n")

    print(f"\nExported summary to: {summary_path}")
    with open(summary_path) as f:
        print(f.read())

# %%
if not df.empty:
    _export_cols = [c for c in df.columns if not c.startswith("_")]
    df[_export_cols].to_csv(output_dir / "gibbs_data.csv", index=False)
    print(f"Saved gibbs_data.csv ({len(df)} runs, {len(_export_cols)} columns)")

# %%
print("\n" + "=" * 68)
print("Gibbs Purification Analysis Complete")
print("=" * 68)

# %%
