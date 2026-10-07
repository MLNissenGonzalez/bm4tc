"""Every study composes under the schema (D25), job by job, and unknown keys fail."""
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError, MissingMandatoryValue

from bm4tc.pipeline.runs import CONFIGS, Job, Study
from bm4tc.core.train import evasion_config

STUDIES = sorted(p.relative_to(CONFIGS / "studies").with_suffix("").as_posix()
                 for p in (CONFIGS / "studies").rglob("*.yaml"))
# Studies waiting for a decision; they must fail loudly until it is made.
PENDING = {"mnist_nat": "mnist_capacity picks the rank (D44)",
           "mnist_at": "mnist_capacity picks the rank (D44)"}


@pytest.mark.parametrize("name", STUDIES)
def test_study_composes_under_schema(name):
    study = Study(name)
    if name in PENDING:
        with pytest.raises(MissingMandatoryValue):
            study.cells()
        return
    for cell in study.cells():  # one seed per cell: seeds only set tracking.seed
        cfg = Job(study, cell, study.seeds()[0]).compose(hparams={})  # no `select` yet
        OmegaConf.to_container(cfg, resolve=True)
        if cfg.trainer.evasion is not None:  # untyped in the schema (D53)
            evasion_config(cfg.trainer.evasion)
        assert (cfg.trainer.evasion is not None) == (study.regime == "at")
        # Every HPO key lands on an existing node (open dicts such as
        # optimizer.kwargs take new keys).
        for key in (study.cfg.hpo.space if study.cfg.hpo else {}):
            parent, _, leaf = key.rpartition(".")
            node = OmegaConf.select(cfg, parent)
            assert node is not None and (leaf in node or parent.endswith("kwargs")), key


def test_unknown_key_is_rejected():
    study = Study("tests/seam_nat")
    study.cfg.config["trainer.stop_crit"] = "acc"
    with pytest.raises(ConfigAttributeError):
        study.jobs()[0].compose()


def test_unknown_study_key_is_rejected():
    from bm4tc.pipeline import runs
    with pytest.raises(Exception, match="nonsense"):
        OmegaConf.merge(OmegaConf.structured(runs.StudyConfig), {"nonsense": 1})


def test_unknown_evasion_key_is_rejected():
    with pytest.raises(Exception, match="eps"):
        evasion_config({"method": "PGD", "eps": 0.1})
