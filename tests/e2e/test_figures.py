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
    assert names == {"purify_vs_eps.pdf", "jem_alpha.pdf", "jem_sweeps.pdf",
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
                   {"study": "tests/seam_jem_nat", "where": {"alpha": 0.5}}]})
    keep, labels = r["keep"], r["labels"]
    assert r["source"] == 1 and len(keep) > 0
    assert float(r["clean"].min()) >= 0 and float(r["clean"].max()) <= 1
    assert np.abs(r["adv"] - r["clean"]).max() <= 0.15 + 1e-6   # relative budget, in [0, 1]
    for k in range(3):   # every kept example fools every model
        assert (r["clean_pred"][k][keep] == labels[keep]).all()
        assert (r["adv_pred"][k][keep] != labels[keep]).all()
        assert r["purified"][k].shape == (len(keep), 2)
