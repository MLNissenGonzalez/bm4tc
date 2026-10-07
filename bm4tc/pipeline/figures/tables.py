"""Tables from study results: models (rows) x metrics (columns), mean +- std over
seeds, as a booktabs ``tabular`` for ``\\input`` (needs ``\\usepackage{booktabs}``).

One kind covers the paper's tables: the headline (clean / robust / purified /
gain = purified minus robust), detection (q columns), the joint attack (joint
minus PGD) and the summary over radii (one table per budget). A metric's ``best:
max | min`` sets the best mean in bold.
"""
import math
from pathlib import Path
from typing import Any, List, Mapping

from bm4tc.pipeline.figures.plots import grid


def number(mean: float, std: float, digits: int, signed: bool, bold: bool) -> str:
    if mean != mean:
        return "---"
    sign = "+" if signed else ""
    body = f"{mean:{sign}.{digits}f}"
    if std == std and std > 0:
        body += rf" \pm {std:.{digits}f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def table(spec: Mapping[str, Any], out: Path) -> List[Path]:
    """``models``, ``metrics``, optional ``eps``, ``where``, ``digits`` (3),
    ``std`` (true: show it), ``header`` (the first column's title)."""
    models, metrics, cells = grid(spec)
    digits, show_std = spec.get("digits", 3), spec.get("std", True)
    best = {}
    for j, metric in enumerate(metrics):
        means = [cells[i, j][0] for i in range(len(models))]
        finite = [m for m in means if not math.isnan(m)]
        if metric.best in ("max", "min") and finite:
            best[j] = max(finite) if metric.best == "max" else min(finite)
    lines = [rf"\begin{{tabular}}{{l{'c' * len(metrics)}}}", r"\toprule",
             " & ".join([spec.get("header", "Model")] + [m.label for m in metrics]) + r" \\",
             r"\midrule"]
    for i, model in enumerate(models):
        row = [model.label]
        for j, metric in enumerate(metrics):
            mean, std, _ = cells[i, j]
            row.append(number(mean, std if show_std else float("nan"), digits,
                              signed=metric.minus is not None,
                              bold=j in best and mean == best[j]))
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = out.parent / f"{out.name}.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return [path]
