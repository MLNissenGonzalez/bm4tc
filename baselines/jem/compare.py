"""Combine JEM and MPS sweep CSVs without changing either analysis pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _spec(value: str) -> tuple[float, Path]:
    alpha, path = value.split("=", 1)
    return float(alpha), Path(path)


def _load(entries: list[str], family: str) -> pd.DataFrame:
    frames = []
    for entry in entries:
        alpha, path = _spec(entry)
        frame = pd.read_csv(path)
        frame["alpha"] = alpha
        frame["family"] = family
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _check_budget_units(frames: list[pd.DataFrame]) -> None:
    """Refuse to merge CSVs that disagree on the epsilon-column convention.

    Budget columns (``rob/{eps}``, ``uq_purify_acc/{eps}/{delta}``) are joined by
    name. MPS analysis writes them RELATIVE and tags the file with ``eps_unit``;
    the JEM pipeline still writes them absolute and has no such column. Merging
    the two silently yields disjoint, half-NaN columns — e.g. legendre ``rob/0.1``
    (relative) and JEM ``rob/0.2`` (absolute) are the *same* budget under two
    names. Fail loudly instead.
    """
    units = {
        (f["eps_unit"].iloc[0] if "eps_unit" in f.columns and len(f) else "abs")
        for f in frames
    }
    if len(units) > 1:
        raise ValueError(
            "Refusing to combine CSVs with mixed epsilon conventions: found "
            f"{sorted(units)}. Relative-keyed files carry eps_unit='rel' (and a "
            "range_size column); files without it are absolute-keyed. Re-run the "
            "absolute-keyed analysis, or migrate it with tools/migrate_metric_keys.py."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine existing JEM/MPS evaluation_data.csv files."
    )
    parser.add_argument("--jem", action="append", default=[], metavar="ALPHA=CSV")
    parser.add_argument("--mps", action="append", default=[], metavar="ALPHA=CSV")
    parser.add_argument("--at", action="append", default=[], metavar="NAME=CSV")
    parser.add_argument("--output", default="analysis/outputs/baselines/jem/comparison")
    args = parser.parse_args()

    frames = [_load(args.jem, "JEM"), _load(args.mps, "MPS")]
    for entry in args.at:
        name, path = entry.split("=", 1)
        frame = pd.read_csv(path)
        frame["alpha"] = np.nan
        frame["family"] = name
        frames.append(frame)
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError("Provide at least one --jem, --mps or --at CSV.")

    _check_budget_units(frames)

    df = pd.concat(frames, ignore_index=True, sort=False)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "combined_runs.csv", index=False)

    id_cols = {"alpha", "family"}
    metric_cols = [
        c for c in df.select_dtypes(include="number").columns if c not in id_cols
    ]
    summary = (
        df.groupby(["family", "alpha"], dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
    )
    summary.to_csv(out / "comparison_summary.csv")

    for metric in ("acc", "rob/0.3", "uq_purify_acc/0.3/0.2"):
        if metric not in df:
            continue
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for family, group in df.dropna(subset=["alpha"]).groupby("family"):
            stats = group.groupby("alpha")[metric].agg(["mean", "std"]).sort_index()
            ax.errorbar(
                stats.index,
                stats["mean"],
                yerr=stats["std"].fillna(0),
                marker="o",
                capsize=3,
                label=family,
            )
        ax.set_xscale("symlog", linthresh=0.005)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05) if "acc" in metric or "rob" in metric else None
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        safe = metric.replace("/", "_")
        fig.savefig(out / f"{safe}.png", dpi=150)
        plt.close(fig)
    print(f"Saved comparison to {out}")


if __name__ == "__main__":
    main()
