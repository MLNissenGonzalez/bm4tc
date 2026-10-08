"""Reading study results for figures: a study's ``results.csv``, narrowed to one
model (one grid cell) or to a curve over one axis, as mean and std over seeds.

A *model* in a paper item is a study plus a filter on its identity columns::

    {study: mnist_nat, where: {beta: 0.5, arch: d3r40}, label: "MPS $\\beta=0.5$"}

and must select exactly one grid cell; the seeds of that cell are what the mean
and std are over. A *metric* is a key of :mod:`bm4tc.pipeline.metrics` in which
``{eps}`` stands for the item's budget (``rob/test/{eps}``), optionally minus
another key (``purify/test/{eps}/d0.1`` minus ``rob/test/{eps}``: the gain).
"""
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from bm4tc.core.embeddings import fmt_budget
from bm4tc.pipeline.runs import outputs_root, parse_arch

CELL = ("study", "embedding", "arch", "beta", "eps")   # identity minus the seed


@dataclass
class Model:
    study: str
    label: str = ""
    where: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, spec: Mapping, where: Optional[Mapping] = None) -> "Model":
        """A model spec, with the item's ``where`` under its own."""
        return cls(spec["study"], spec.get("label", spec["study"]),
                   {**(where or {}), **spec.get("where", {})})


@dataclass
class Metric:
    key: str
    label: str = ""
    minus: Optional[str] = None
    best: Optional[str] = None          # tables: bold the max | min

    @classmethod
    def parse(cls, spec) -> "Metric":
        if isinstance(spec, str):
            return cls(spec, spec)
        return cls(spec["key"], spec.get("label", spec["key"]), spec.get("minus"),
                   spec.get("best"))


def fill(template: str, **values) -> str:
    """``rob/test/{eps}`` with eps=0.1 -> ``rob/test/0.1`` (budgets as in the keys)."""
    return template.format(**{k: fmt_budget(v) if isinstance(v, float) else v
                              for k, v in values.items()})


@lru_cache(maxsize=None)
def _read(path: Path, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def load(study: str) -> pd.DataFrame:
    """The study's results.csv (one row per analysed run)."""
    path = outputs_root() / study / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"{study}: no results at {path}; run "
                                f"`python -m bm4tc run {study}` (or `analyse`) first")
    return _read(path, path.stat().st_mtime).copy()


def _matches(column: pd.Series, value) -> pd.Series:
    if value is None:
        return column.isna()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return np.isclose(pd.to_numeric(column, errors="coerce"), float(value))
    return column.astype(str) == str(value)


def rows(model: Model) -> pd.DataFrame:
    """The study's rows that pass the model's filter."""
    df = load(model.study)
    for name, value in model.where.items():
        if name not in df.columns:
            raise KeyError(f"{model.study}: no identity column {name!r} to filter on")
        df = df[_matches(df[name], value)]
    if df.empty:
        raise LookupError(f"{model.study}: no runs with {model.where}")
    return df


def one_cell(df: pd.DataFrame, what: str) -> pd.DataFrame:
    cells = df[list(CELL)].astype(str).drop_duplicates()
    if len(cells) > 1:
        raise LookupError(f"{what} selects {len(cells)} grid cells, not one; narrow its "
                          f"`where`:\n{cells.to_string(index=False)}")
    return df


def values(df: pd.DataFrame, metric: Metric, **subs) -> np.ndarray:
    """The metric per row (run)."""
    def column(template):
        name = fill(template, **subs)
        if name not in df.columns:
            raise KeyError(f"no column {name!r} in results (has e.g. "
                           f"{sorted(c for c in df.columns if '/' in c)[:8]})")
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    out = column(metric.key)
    return out - column(metric.minus) if metric.minus else out


def stats(v: np.ndarray) -> Tuple[float, float, int]:
    """Mean, std (ddof 1; 0 for one value) and count over the finite values."""
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)


def cell_stats(model: Model, metric: Metric, **subs) -> Tuple[float, float, int]:
    """Mean, std, n over the seeds of the one cell the model selects."""
    df = one_cell(rows(model), f"{model.study} {model.where}")
    return stats(values(df, metric, **subs))


# ── Curves ──────────────────────────────────────────────────────────────────

AXES = {"beta": "beta", "bond_dim": "arch", "arch": "arch"}   # x from an identity column


def _x_value(axis: str, raw) -> float:
    return float(parse_arch(raw)[1]) if axis == "bond_dim" else raw


def curve(model: Model, metric: Metric, x: str, **subs) -> pd.DataFrame:
    """Columns x, mean, std, n: the metric over ``x``. x is an identity column
    (beta, bond_dim) grouped over cells, or a placeholder ``{x}`` in the metric key
    (``purify_gibbs/test/{eps}/k{x}``, ``rob/test/{x}``) read off one cell's columns."""
    df = rows(model)
    out = []
    if x in AXES:
        for raw, group in df.groupby(AXES[x], dropna=False):
            one_cell(group, f"{model.study} {model.where} at {x}={raw}")
            out.append((_x_value(x, raw), *stats(values(group, metric, **subs))))
    else:
        one_cell(df, f"{model.study} {model.where}")
        pattern = re.escape(fill(metric.key, **subs, x="\0")).replace("\\\0", "\0")
        pattern = re.compile(pattern.replace("\0", r"([0-9.e+-]+)") + "$")
        xs = sorted({float(m.group(1)) for c in df.columns if (m := pattern.match(c))})
        if not xs:
            raise KeyError(f"{model.study}: no column matches {metric.key!r} over {x}")
        for v in xs:
            out.append((v, *stats(values(df, metric, **subs, x=fmt_budget(v)))))
    return pd.DataFrame(out, columns=["x", "mean", "std", "n"]).sort_values("x")


def model_list(specs: List[Mapping], where: Optional[Mapping] = None) -> List[Model]:
    return [Model.parse(s, where) for s in specs]
