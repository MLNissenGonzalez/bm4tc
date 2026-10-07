"""Figures from study results (no checkpoints): mean over seeds, std as band or
error bar. Each kind takes the item's spec and the output path without suffix,
and returns the files it wrote.

    curve      metric vs one axis: alpha, bond_dim, or {x} in the key (eps, sweeps)
    bars       metrics (groups) x models (bars), at one budget
    coverage   accuracy on passed examples vs fraction passed, over the detection
               percentiles q (one curve per model)
"""
from pathlib import Path
from typing import Any, Dict, List, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bm4tc.pipeline.figures.results import (Metric, Model, cell_stats, curve as curve_of,  # noqa: E402
                                            fill, model_list, one_cell, rows, stats, values)

STYLE = {"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 8,
         "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42}
XLABELS = {"alpha": r"$\alpha$", "bond_dim": "bond dimension $r$", "eps": r"$\varepsilon$",
           "sweeps": "sweeps $k$", "q": "detection percentile $q$"}


def save(fig, out: Path) -> List[Path]:
    path = out.parent / f"{out.name}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return [path]


def _alpha_axis(ax, xs):
    nonzero = sorted(x for x in xs if x > 0)
    if nonzero:
        ax.set_xscale("symlog", linthresh=nonzero[0], linscale=0.5)
    ax.set_xticks(sorted(set(xs)))
    ax.set_xticklabels([f"{x:g}" for x in sorted(set(xs))])
    ax.minorticks_off()


def curve(spec: Mapping[str, Any], out: Path) -> List[Path]:
    """``x``, ``series: [{study, where, key, label, minus, axis: left|right}]``,
    optional ``eps``, ``where``, ``ylabel``, ``ylabel_right``, ``ylim``, ``title``."""
    x = spec["x"]
    subs = {"eps": float(spec["eps"])} if "eps" in spec else {}
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=spec.get("size", (4.5, 3.2)))
        right = None
        xs = []
        for i, s in enumerate(spec["series"]):
            model, metric = Model.parse(s, spec.get("where")), Metric.parse(s)
            c = curve_of(model, metric, x, **subs)
            if s.get("axis") == "right":
                right = right or ax.twinx()
            target = right if s.get("axis") == "right" else ax
            color = s.get("color", f"C{i}")
            target.plot(c.x, c["mean"], marker="o", ms=3, lw=1.5, color=color,
                        ls=s.get("style", "-"), label=s.get("label", metric.key))
            target.fill_between(c.x, c["mean"] - c["std"], c["mean"] + c["std"],
                                color=color, alpha=0.15, lw=0)
            xs += list(c.x)
        if x == "alpha":
            _alpha_axis(ax, xs)
        ax.set_xlabel(spec.get("xlabel", XLABELS.get(x, x)))
        ax.set_ylabel(spec.get("ylabel", ""))
        if "ylim" in spec:
            ax.set_ylim(*spec["ylim"])
        if right is not None:
            right.set_ylabel(spec.get("ylabel_right", ""))
            right.grid(False)
        handles = ax.get_legend_handles_labels()
        if right is not None:
            more = right.get_legend_handles_labels()
            handles = (handles[0] + more[0], handles[1] + more[1])
        ax.legend(*handles)
        if "title" in spec:
            ax.set_title(fill(spec["title"], **subs))
        return save(fig, out)


def grid(spec: Mapping[str, Any]):
    """Models x metrics of (mean, std, n), for bars and tables."""
    subs = {"eps": float(spec["eps"])} if "eps" in spec else {}
    models = model_list(spec["models"], spec.get("where"))
    metrics = [Metric.parse(m) for m in spec["metrics"]]
    cells = {(i, j): cell_stats(m, k, **subs)
             for i, m in enumerate(models) for j, k in enumerate(metrics)}
    return models, metrics, cells


def bars(spec: Mapping[str, Any], out: Path) -> List[Path]:
    """``models``, ``metrics`` (the groups), optional ``eps``, ``where``, ``ylabel``,
    ``title``."""
    models, metrics, cells = grid(spec)
    width = 0.8 / len(models)
    x = np.arange(len(metrics))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=spec.get("size", (1.2 + 1.1 * len(metrics), 3.2)))
        for i, m in enumerate(models):
            mean = [cells[i, j][0] for j in range(len(metrics))]
            std = [cells[i, j][1] for j in range(len(metrics))]
            ax.bar(x + (i - (len(models) - 1) / 2) * width, mean, width, yerr=std,
                   capsize=2, label=m.label, color=f"C{i}", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([k.label for k in metrics])
        ax.set_ylabel(spec.get("ylabel", "Accuracy"))
        ax.set_ylim(*spec.get("ylim", (0, 1.05)))
        ax.grid(axis="x", visible=False)
        ax.legend(ncol=spec.get("legend_cols", 2))
        if "title" in spec:
            ax.set_title(fill(spec["title"], eps=float(spec.get("eps", 0))))
        return save(fig, out)


def coverage(spec: Mapping[str, Any], out: Path) -> List[Path]:
    """``models``, ``eps``, optional ``joint`` (the joint attack's columns).
    x = fraction of attacked examples passed by the detector, y = accuracy on
    them; one point per detection percentile q, labelled."""
    eps = float(spec["eps"])
    suffix = "_joint" if spec.get("joint") else ""
    passed = Metric(f"detect{suffix}/test/{{eps}}/q{{x}}")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=spec.get("size", (4, 3.2)))
        for i, model in enumerate(model_list(spec["models"], spec.get("where"))):
            df = one_cell(rows(model), f"{model.study} {model.where}")
            qs = curve_of(model, passed, "q", eps=eps).x
            pts = []
            for q in qs:
                det = stats(values(df, Metric(f"detect{suffix}/test/{{eps}}/q{q:g}"), eps=eps))[0]
                err = stats(values(df, Metric(f"err_passed{suffix}/test/{{eps}}/q{q:g}"),
                                   eps=eps))[0]
                pts.append((1 - det, 1 - err, q))
            pts = np.array(pts)
            ax.plot(pts[:, 0], pts[:, 1], marker="o", ms=3, color=f"C{i}", label=model.label)
            for px, py, q in pts:
                ax.annotate(f"{q:g}", (px, py), fontsize=6, xytext=(2, 2),
                            textcoords="offset points")
        ax.set_xlabel("fraction passed")
        ax.set_ylabel("accuracy on passed")
        ax.legend()
        return save(fig, out)


KINDS: Dict[str, Any] = {"curve": curve, "bars": bars, "coverage": coverage}
