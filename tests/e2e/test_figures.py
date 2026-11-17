"""`figures` on the seam studies: every kind of configs/papers/tests/seam.yaml is
drawn from real results and checkpoints (MPS and JEM), through the CLI."""
import numpy as np
import pytest

from tests.e2e.conftest import python

STUDIES = ("tests/seam_nat", "tests/seam_at", "tests/seam_jem_nat")


@pytest.fixture(scope="module")
def root(seam):
    for study in STUDIES:   # seam_at is warm from seam_nat
        out = seam(study)
    return out


@pytest.mark.slow
def test_every_item_is_drawn(root):
    stdout = python(["-m", "bm4tc", "figures", "tests/seam"], root)
    assert "0 failed" in stdout
    names = {p.name for p in (root / "figures").iterdir()}
    assert names == {"purify_vs_eps.pdf", "jem_beta.pdf", "jem_sweeps.pdf",
                     "defenses_eps0.1.pdf", "defenses_eps0.15.pdf", "coverage.pdf",
                     "headline.tex", "density.pdf", "samples.pdf"}
    tex = (root / "figures" / "headline.tex").read_text()
    assert tex.count(r"\\") == 4 and "NAT &" in tex and "JEM &" in tex


@pytest.mark.slow
def test_transfer_examples(root, monkeypatch):
    from bm4tc.pipeline.figures.checkpoints import transfer_examples

    monkeypatch.setenv("BM4TC_DATA_ROOT", str(root))
    r = transfer_examples({
        "eps": 0.15, "delta": 0.1, "attack_steps": 5, "purify_steps": 5, "max_attack": 100,
        "models": [{"study": "tests/seam_nat"}, {"study": "tests/seam_at", "source": True},
                   {"study": "tests/seam_jem_nat", "where": {"beta": 0.75}}]})
    keep, labels = r["keep"], r["labels"]
    assert r["source"] == 1 and len(keep) > 0
    assert float(r["clean"].min()) >= 0 and float(r["clean"].max()) <= 1
    assert np.abs(r["adv"] - r["clean"]).max() <= 0.15 + 1e-6   # relative budget, in [0, 1]
    s = r["source"]   # every kept example fools the attacked model, and only it is filtered on
    assert (r["clean_pred"][s][keep] == labels[keep]).all()
    assert (r["adv_pred"][s][keep] != labels[keep]).all()
    fooled = (r["clean_pred"][s] == labels) & (r["adv_pred"][s] != labels)
    assert np.array_equal(keep, np.flatnonzero(fooled))
    for k in range(3):
        assert r["purified"][k].shape == (len(keep), 2)


@pytest.mark.slow
def test_class_means_match_the_joint_on_a_grid(root, monkeypatch):
    """The sweep's per-class feature means equal brute-force integration of
    p(x, c) over the 2-D grid, with the same trapezoidal weights."""
    import torch

    from bm4tc.pipeline.figures.checkpoints import class_means, load
    from bm4tc.pipeline.figures.results import model_list

    monkeypatch.setenv("BM4TC_DATA_ROOT", str(root))
    net, _, _ = load(model_list([{"study": "tests/seam_nat"}], None)[0])
    bins = 60
    means = class_means(net, bins).double()

    axis = torch.linspace(*net.input_range, bins, dtype=torch.float64)
    w = torch.trapezoid(torch.eye(bins, dtype=torch.float64), x=axis, dim=0)
    g1, g2 = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack([g1.ravel(), g2.ravel()], dim=1)
    with torch.no_grad():
        log_joint = net.log_joint(grid.float().to(next(net.parameters()).device)).double().cpu()
    p = (log_joint - log_joint.max()).exp() * torch.outer(w, w).ravel()[:, None]   # (N, classes)
    brute = (grid.T @ p / p.sum(dim=0)).T                                              # (classes, 2)
    assert means.shape == brute.shape == (net.out_dim, 2)
    assert torch.allclose(means, brute, atol=1e-4)
