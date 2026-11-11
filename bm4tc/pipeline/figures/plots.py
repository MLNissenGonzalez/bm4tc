"""Figures from study results (no checkpoints): mean over seeds, std as band or
error bar. Each kind takes the item's spec and the output path without suffix,
and returns the files it wrote.

    curve      metric vs one axis: beta, bond_dim, or {x} in the key (eps, sweeps);
               with ``panels``, the same series in one panel per dataset
    bars      metrics (groups) x models (bars), at one budget
    coverage   accuracy on passed examples vs fraction passed, over the detection
               percentiles q (one curve per model)
"""
from pathlib import Path
from typing import Any, Dict, List, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from bm4tc.core.embeddings import fmt_budget  # noqa: E402

from bm4tc.pipeline.figures.results import (Metric, Model, cell_stats, curve as curve_of,  # noqa: E402
                                            fill, model_list, one_cell, rows, stats, values)

STYLE = {"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 8,
         "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42}
XLABELS = {"beta": r"$\beta$", "bond_dim": "bond dimension $r$",
           "phys_dim": "physical dimension $d$", "eps": r"$\varepsilon$",
           "sweeps": "sweeps $k$", "q": "detection percentile $q$"}


def save(fig, out: Path) -> List[Path]:
    path = out.parent / f"{out.name}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return [path]


def _beta_positions(betas) -> Dict[float, float]:
    """Where each β sits on the x axis: at its log weight ratio ln(β/(1-β)) (D86),
    so a ladder of equal ratio steps is evenly spaced; 0 and 1 one ladder step
    (ln 10) beyond the outermost interior values."""
    inner = {b: float(np.log(b / (1 - b))) for b in set(betas) if 0 < b < 1}
    lo, hi = (min(inner.values()), max(inner.values())) if inner else (0.0, 0.0)
    step = float(np.log(10))
    return {**inner, 0.0: lo - step, 1.0: hi + step}


def _beta_axis(ax, positions: Dict[float, float], betas):
    ticks = sorted(set(betas))
    ax.set_xticks([positions[b] for b in ticks])
    ax.set_xticklabels([f"{b:g}" for b in ticks])
    ax.minorticks_off()


def curve(spec: Mapping[str, Any], out: Path) -> List[Path]:
    """``x``, ``series: [{study, where, key, label, minus, axis: left|right}]``,
    optional ``eps``, ``where``, ``ylabel``, ``ylabel_right``, ``ylim``, ``title``.

    Small multiples: ``panels: [{title, where, eps: [e1, e2], shade, ...}]`` draws the
    series in every panel; one panel gives a single figure in the same style. Any
    other panel field fills its placeholder in a series' study and key: ``data: ecg200``
    makes ``"{data}_nat"`` ``ecg200_nat``, ``delta: 0.01`` makes ``d{delta}`` ``d0.01``.
    A series whose key holds ``{eps}`` is drawn once per radius of the panel, marker
    o for the first and * for the second; ``eps_labels`` names the two in the legend,
    the panel title gives their values. A series with ``thin: true`` is a thin pale
    line without markers (e.g. clean accuracy). Every line has its std band over
    seeds. ``shade: {from, to, label}`` greys the x range between those two values.
    One legend below the panels; ``ncols`` (default 3). The panels share their y
    axis unless ``sharey: false`` (each scaled to its own data; then no ``ylim``).
    The item's ``eps`` and ``axis: right`` do not combine with panels."""
    if "panels" in spec:
        return _panels(spec, out)
    x = spec["x"]
    subs = {"eps": float(spec["eps"])} if "eps" in spec else {}
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=spec.get("size", (4.5, 3.2)))
        right = None
        series = [(s, curve_of(Model.parse(s, spec.get("where")), Metric.parse(s), x, **subs))
                  for s in spec["series"]]
        xs = [v for _, c in series for v in c.x]
        positions = _beta_positions(xs) if x == "beta" else None
        for i, (s, c) in enumerate(series):
            metric = Metric.parse(s)
            if positions is not None:
                c = c.assign(x=[positions[v] for v in c.x])
            if s.get("axis") == "right":
                right = right or ax.twinx()
            target = right if s.get("axis") == "right" else ax
            color = s.get("color", f"C{i}")
            target.plot(c.x, c["mean"], marker="o", ms=3, lw=1.5, color=color,
                        ls=s.get("style", "-"), label=s.get("label", metric.key))
            target.fill_between(c.x, c["mean"] - c["std"], c["mean"] + c["std"],
                                color=color, alpha=0.15, lw=0)
        if positions is not None:
            _beta_axis(ax, positions, xs)
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


MARKERS = (("o", 3.5, "●"), ("*", 7, "★"))   # a panel's radii: marker, size, glyph
PANEL_FIELDS = {"title", "where", "eps", "shade"}       # the others fill placeholders


class _Keep(dict):
    def __missing__(self, name):
        return "{" + name + "}"


def _partial(template, subs: Mapping[str, Any]):
    """Fill the placeholders named in ``subs`` (budgets as in the keys), keep the
    others (``{eps}``, ``{x}``) for later; a list is filled item by item."""
    if isinstance(template, list):
        return [_partial(t, subs) for t in template]
    return template.format_map(_Keep({k: fmt_budget(v) if isinstance(v, float) else v
                                      for k, v in subs.items()}))


def _panels(spec: Mapping[str, Any], out: Path) -> List[Path]:
    """``curve`` with ``panels`` (see its docstring)."""
    if "eps" in spec:
        raise ValueError("curve: `eps` and `panels` do not combine; each panel names "
                         "its own radii")
    if any(s.get("axis") == "right" for s in spec["series"]):
        raise ValueError("curve: `axis: right` and `panels` do not combine")
    sharey = spec.get("sharey", True)
    if not sharey and "ylim" in spec:
        raise ValueError("curve: `sharey: false` and `ylim` do not combine; each panel "
                         "is scaled to its own data")
    x, panels = spec["x"], spec["panels"]

    def lines(panel):   # (series index, series, radius index or None, curve)
        radii = list(panel.get("eps", []))
        if len(radii) > len(MARKERS):
            raise ValueError(f"curve panel {panel['title']!r}: at most {len(MARKERS)} radii")
        where = {**spec.get("where", {}), **panel.get("where", {})}
        subs = {k: v for k, v in panel.items() if k not in PANEL_FIELDS}
        for k, s in enumerate(spec["series"]):
            s = {**s, **{f: _partial(s[f], subs) for f in ("study", "key", "minus") if f in s}}
            model, metric = Model.parse(s, where), Metric.parse(s)
            if "{eps}" not in metric.key:
                yield k, s, None, curve_of(model, metric, x)
                continue
            if not radii:
                raise ValueError(f"curve panel {panel['title']!r}: series {metric.key!r} "
                                 f"needs the panel's `eps`")
            for i, e in enumerate(radii):
                yield k, s, i, curve_of(model, metric, x, eps=float(e))

    drawn = [list(lines(p)) for p in panels]
    xs = [v for d in drawn for *_, c in d for v in c.x]
    positions = _beta_positions(xs) if x == "beta" else {v: v for v in xs}
    ticks = sorted(set(positions.values()))
    ncols = min(spec.get("ncols", 3), len(panels))
    nrows = -(-len(panels) // ncols)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, sharey=sharey, squeeze=False,
                                 figsize=spec.get("size", (3.0 * ncols + 0.6, 2.4 * nrows)))
        for ax, panel, d in zip(axes.flat, panels, drawn):
            for k, s, i, c in d:
                at = [positions[v] for v in c.x]
                style = dict(color=s.get("color", f"C{k}"), ls=s.get("style", "-"))
                if s.get("thin"):
                    ax.plot(at, c["mean"], lw=0.8, alpha=0.5, **style)
                else:
                    marker, size, _ = MARKERS[i or 0]
                    ax.plot(at, c["mean"], lw=1.3, marker=marker, ms=size, **style)
                ax.fill_between(at, c["mean"] - c["std"], c["mean"] + c["std"],
                                color=style["color"], alpha=0.07 if s.get("thin") else 0.12,
                                lw=0)
            if "shade" in panel:
                _shade(ax, panel["shade"], positions, ticks)
            eps = panel.get("eps", [])   # one radius needs no marker key
            radii = "   ".join((f"{MARKERS[i][2]} " if len(eps) > 1 else "")
                               + f"$\\varepsilon={float(e):g}$" for i, e in enumerate(eps))
            ax.set_title(f"{panel['title']}   {radii}".strip(), fontsize=8.5)
            if x == "beta":
                _beta_axis(ax, positions, xs)
            elif x in ("bond_dim", "phys_dim"):
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"{t:g}" for t in ticks])
            ax.tick_params(labelsize=7)
            if "ylim" in spec:
                ax.set_ylim(*spec["ylim"])
        for ax in axes.flat[len(panels):]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel(spec.get("ylabel", ""))
        for ax in axes[-1]:
            ax.set_xlabel(spec.get("xlabel", XLABELS.get(x, x)))
        fig.tight_layout()
        handles, cols = _panel_legend(spec, panels), spec.get("legend_cols", 6)
        rows = -(-len(handles) // cols)   # matplotlib fills columns first: read across rows
        handles = [handles[r * cols + c] for c in range(cols) for r in range(rows)
                   if r * cols + c < len(handles)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                   ncol=cols, frameon=False)
        return save(fig, out)


def _shade(ax, shade: Mapping[str, Any], positions: Dict[float, float], ticks: List[float]):
    """Grey the x range from ``shade['from']`` to ``shade['to']``, out to halfway to the
    neighbouring ticks; nothing if the range is not on the axis (e.g. an AT grid)."""
    if float(shade["from"]) not in positions or float(shade["to"]) not in positions:
        return
    lo, hi = positions[float(shade["from"])], positions[float(shade["to"])]
    j, k = ticks.index(lo), ticks.index(hi)
    gap = (ticks[-1] - ticks[0]) / max(len(ticks) - 1, 1) / 2
    lo = (ticks[j - 1] + lo) / 2 if j > 0 else lo - gap
    hi = (ticks[k + 1] + hi) / 2 if k + 1 < len(ticks) else hi + gap
    ax.axvspan(lo, hi, color="0.92", zorder=0, lw=0)
    if shade.get("label"):
        ax.text((lo + hi) / 2, 0.97, shade["label"], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7)


def _panel_legend(spec: Mapping[str, Any], panels) -> List[Line2D]:
    """One entry per labelled series (its colour and line), then one per radius
    (its marker)."""
    handles = [Line2D([], [], color=s.get("color", f"C{k}"), ls=s.get("style", "-"),
                      lw=0.8 if s.get("thin") else 1.3, alpha=0.5 if s.get("thin") else 1,
                      label=s["label"])
               for k, s in enumerate(spec["series"]) if "label" in s]
    n = max(len(p.get("eps", [])) for p in panels)
    default = [f"radius {i + 1}" for i in range(n)] if n > 1 else []   # one radius: no key
    names = spec.get("eps_labels", default)[:n]
    handles += [Line2D([], [], color="k", ls="", marker=MARKERS[i][0], ms=MARKERS[i][1],
                       label=name) for i, name in enumerate(names)]
    return handles


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
