"""Paper figures and tables from studies (D28, D75): ``python -m bm4tc figures <paper>``.

A paper manifest, ``configs/papers/<paper>.yaml``, names each figure or table and
says how to draw it::

    out: figures/journal          # output directory, under the repository root
    items:
      mnist_alpha_acc:            # -> figures/journal/mnist_alpha_acc.pdf
        kind: curve
        x: alpha
        eps: 0.1
        where: {arch: d3r40}
        series:
          - {study: mnist_nat, key: acc/test, label: Clean}
          - {study: mnist_nat, key: "rob/test/{eps}", label: Robust}

Kinds (each documented in its module):

    curve, bars, coverage      plots.py        study results only
    table                      tables.py       study results only -> .tex
    density, samples, transfer checkpoints.py  one seed's checkpoint per model

A list ``eps: [0.1, 0.2]`` makes one output per budget (``{name}_eps0.1``). Results
come from each study's ``results.csv`` (``analyse`` writes it); metric keys are those
of :mod:`bm4tc.pipeline.metrics`. An item that fails (e.g. a study not run yet) is
reported and the others are still drawn.
"""
import logging
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from omegaconf import OmegaConf

from bm4tc.core.embeddings import fmt_budget
from bm4tc.pipeline.figures import checkpoints, plots, tables
from bm4tc.pipeline.runs import CONFIGS, REPO

logger = logging.getLogger(__name__)

KINDS = {**plots.KINDS, "table": tables.table, **checkpoints.KINDS}
PAPERS = CONFIGS / "papers"


def manifest(paper: str) -> dict:
    path = PAPERS / f"{paper}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No paper {paper!r} at {path}")
    m = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    unknown = {name: item.get("kind") for name, item in m["items"].items()
               if item.get("kind") not in KINDS}
    if unknown:
        raise ValueError(f"{paper}: unknown kinds {unknown}; known: {sorted(KINDS)}")
    return m


def draw(name: str, item: dict, out_dir: Path) -> List[Path]:
    """One item: one output per budget if ``eps`` is a list."""
    kind = KINDS[item["kind"]]
    if isinstance(item.get("eps"), list):
        return [p for eps in item["eps"]
                for p in kind({**item, "eps": eps}, out_dir / f"{name}_eps{fmt_budget(eps)}")]
    return kind(item, out_dir / name)


def make(paper: str, only: Optional[Iterable[str]] = None,
         out_dir: Optional[Path] = None) -> Dict[str, Optional[List[Path]]]:
    """Draw the paper's items (or only those named) into ``out_dir`` (default: the
    manifest's ``out``); None for a failed item."""
    m = manifest(paper)
    only = set(only or m["items"])
    unknown = only - set(m["items"])
    if unknown:
        raise KeyError(f"{paper}: no items {sorted(unknown)}")
    out_dir = Path(out_dir or REPO / m.get("out", f"figures/{paper}"))
    done = {}
    for name, item in m["items"].items():
        if name not in only:
            continue
        try:
            done[name] = draw(name, item, out_dir)
            logger.info(f"{name}: " + ", ".join(str(p) for p in done[name]))
        except Exception as e:   # one missing study must not stop the others
            done[name] = None
            logger.error(f"{name} ({item['kind']}) failed: {e}")
            logger.debug(traceback.format_exc())
    return done
