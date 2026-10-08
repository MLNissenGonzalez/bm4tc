"""bm4tc.pipeline.figures on a synthetic results.csv: selection, mean/std over
seeds, curves, tables, and the manifest runner."""
import numpy as np
import pandas as pd
import pytest
import yaml

from bm4tc.pipeline import figures
from bm4tc.pipeline.figures import results, tables
from bm4tc.pipeline.figures.results import Metric, Model

SEEDS = (1, 2, 3)


def _row(arch, beta, seed):
    r = {"study": "toy", "dataset": "spirals", "model": "mps", "regime": "nat",
         "embedding": "legendre", "arch": arch, "beta": beta, "eps": np.nan, "seed": seed}
    base = 0.9 - beta / 2 + (0.05 if arch == "d3r40" else 0) + 0.01 * seed
    r["acc/test"] = base
    for eps in (0.1, 0.2):
        r[f"rob/test/{eps:g}"] = base - eps
        r[f"purify/test/{eps:g}/d0.1"] = base - eps / 2
        for q in (5, 10):
            r[f"detect/test/{eps:g}/q{q}"] = q / 100 + eps
            r[f"err_passed/test/{eps:g}/q{q}"] = 0.2 - q / 100
        for k in (1, 3):
            r[f"purify_gibbs/test/{eps:g}/k{k}"] = base - eps + k / 100
    return r


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.setenv("BM4TC_DATA_ROOT", str(tmp_path))
    rows = [_row(a, al, s) for a in ("d3r20", "d3r40") for al in (0.0, 0.01, 1.0) for s in SEEDS]
    (tmp_path / "outputs" / "toy").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(tmp_path / "outputs" / "toy" / "results.csv", index=False)
    return tmp_path


def test_cell_stats_is_over_the_seeds(toy):
    mean, std, n = results.cell_stats(Model("toy", where={"arch": "d3r20", "beta": 0.01}),
                                      Metric("rob/test/{eps}"), eps=0.1)
    assert n == 3
    assert mean == pytest.approx(0.9 - 0.005 + 0.02 - 0.1)
    assert std == pytest.approx(0.01)


def test_minus_is_per_run(toy):
    mean, std, _ = results.cell_stats(
        Model("toy", where={"arch": "d3r20", "beta": 0}),
        Metric("purify/test/{eps}/d0.1", minus="rob/test/{eps}"), eps=0.2)
    assert (mean, std) == pytest.approx((0.1, 0.0))


def test_several_cells_are_refused(toy):
    with pytest.raises(LookupError, match="2 grid cells"):
        results.cell_stats(Model("toy", where={"beta": 0}), Metric("acc/test"))


def test_missing_study_and_column(toy):
    with pytest.raises(FileNotFoundError, match="run `python -m bm4tc run nope`"):
        results.load("nope")
    with pytest.raises(KeyError, match="no column"):
        results.cell_stats(Model("toy", where={"arch": "d3r20", "beta": 0}), Metric("x/test"))


def test_curve_over_an_identity_column(toy):
    c = results.curve(Model("toy", where={"arch": "d3r40"}), Metric("acc/test"), "beta")
    assert list(c.x) == [0, 0.01, 1]
    assert c["mean"].tolist() == pytest.approx([0.97, 0.965, 0.47])
    c = results.curve(Model("toy", where={"beta": 0}), Metric("acc/test"), "bond_dim")
    assert list(c.x) == [20, 40]


def test_curve_over_a_key_placeholder(toy):
    model = Model("toy", where={"arch": "d3r20", "beta": 0})
    c = results.curve(model, Metric("purify_gibbs/test/{eps}/k{x}"), "sweeps", eps=0.1)
    assert list(c.x) == [1, 3]
    c = results.curve(model, Metric("rob/test/{x}"), "eps")
    assert list(c.x) == [0.1, 0.2]
    assert c["mean"].tolist() == pytest.approx([0.82, 0.72])


def test_table(toy, tmp_path):
    spec = {"eps": 0.1, "where": {"arch": "d3r40"},
            "models": [{"study": "toy", "where": {"beta": a}, "label": f"a{a}"} for a in (0, 1)],
            "metrics": [{"key": "acc/test", "label": "Clean", "best": "max"},
                        {"key": "purify/test/{eps}/d0.1", "minus": "rob/test/{eps}",
                         "label": "Gain"}]}
    (path,) = tables.table(spec, tmp_path / "t")
    tex = path.read_text().splitlines()
    assert tex[2] == r"Model & Clean & Gain \\"
    assert tex[4] == r"a0 & $\mathbf{0.970 \pm 0.010}$ & $+0.050$ \\"
    assert tex[5] == r"a1 & $0.470 \pm 0.010$ & $+0.050$ \\"


def test_manifest_draws_every_item_and_reports_failures(toy, tmp_path, monkeypatch):
    model = {"study": "toy", "where": {"arch": "d3r20", "beta": 0}, "label": "M"}
    paper = {"items": {
        "beta": {"kind": "curve", "x": "beta", "eps": 0.1, "where": {"arch": "d3r20"},
                  "series": [{"study": "toy", "key": "acc/test"},
                             {"study": "toy", "key": "rob/test/{eps}", "axis": "right"}]},
        "bars": {"kind": "bars", "eps": [0.1, 0.2], "models": [model],
                 "metrics": ["acc/test", "rob/test/{eps}"]},
        "coverage": {"kind": "coverage", "eps": 0.1, "models": [model]},
        "missing": {"kind": "table", "models": [{"study": "nope"}], "metrics": ["acc/test"]},
    }}
    monkeypatch.setattr(figures, "PAPERS", tmp_path)
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(paper))
    done = figures.make("p", out_dir=tmp_path / "fig")
    assert done["missing"] is None
    assert sorted(p.name for p in done["bars"]) == ["bars_eps0.1.pdf", "bars_eps0.2.pdf"]
    assert all(p.exists() for n, paths in done.items() if paths for p in paths)

    paper["items"]["bad"] = {"kind": "pie"}
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(paper))
    with pytest.raises(ValueError, match="unknown kinds"):
        figures.make("p", out_dir=tmp_path / "fig")
