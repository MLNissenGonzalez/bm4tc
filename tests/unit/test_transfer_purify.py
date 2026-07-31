"""Unit tests for the pure helpers of analysis.visualize.mnist_transfer_purify.

These target the prediction-array logic (transfer filtering, figure-row selection), the
model-path resolution, and the statistics report formatter; no model, no GPU, so they run
fast and unmarked.
"""

import numpy as np

import pytest

from analysis.visualize.mnist_transfer_purify import (
    _format_stats,
    _parse_models,
    _resolve_run_dir,
    _select_rows,
    _transfer_mask,
)
from analysis.utils import load_run_config


def test_transfer_mask_requires_all_models_fooled():
    labels = np.array([0, 1, 2, 3, 4])
    clean = [
        np.array([0, 1, 2, 9, 4]),  # idx 3: model A misclassifies clean -> exclude
        np.array([0, 1, 2, 3, 9]),  # idx 4: model B misclassifies clean -> exclude
        np.array([0, 1, 2, 3, 4]),
    ]
    adv = [
        np.array([7, 1, 8, 7, 7]),  # idx 1: model A still correct on adv -> exclude
        np.array([7, 7, 8, 7, 7]),
        np.array([7, 7, 8, 7, 7]),
    ]
    mask = _transfer_mask(clean, adv, labels)
    # Only idx 0 and 2 satisfy: all clean-correct AND all adv-wrong.
    assert mask.tolist() == [True, False, True, False, False]


def test_transfer_mask_single_model():
    labels = np.array([0, 1, 2])
    mask = _transfer_mask([np.array([0, 1, 9])], [np.array([5, 1, 5])], labels)
    assert mask.tolist() == [True, False, False]


def test_transfer_mask_all_excluded_when_clean_wrong():
    labels = np.array([5, 6])
    zeros = np.array([0, 0])
    mask = _transfer_mask([zeros, zeros], [zeros, zeros], labels)
    assert mask.tolist() == [False, False]


def test_transfer_mask_empty_model_list_keeps_everything():
    # Degenerate but well-defined: no model constrains the mask.
    assert _transfer_mask([], [], np.array([1, 2, 3])).tolist() == [True, True, True]


def test_select_rows_picks_first_occurrence_per_class():
    labels = np.array([3, 0, 5, 3, 0, 9])
    rows = _select_rows(labels, [0, 3, 5, 9])
    assert rows == {0: 1, 3: 0, 5: 2, 9: 5}


def test_select_rows_preserves_requested_order_and_reports_missing():
    labels = np.array([5, 5, 0])
    rows = _select_rows(labels, [9, 0, 3])
    assert list(rows.keys()) == [9, 0, 3]
    assert rows[9] is None and rows[3] is None
    assert rows[0] == 2


def test_parse_models_label_may_contain_equals():
    # mathtext labels carry '=' -- the key is before the first '=', the path after the last.
    specs = [r"a001=purified $\alpha=0.01$=outputs/x/0/models/model",
             "at=outputs/y/seed_sweep"]
    assert _parse_models(specs) == [
        ("a001", r"purified $\alpha=0.01$", "outputs/x/0/models/model"),
        ("at", "at", "outputs/y/seed_sweep"),
    ]


def test_parse_models_defaults_when_empty():
    from analysis.visualize.mnist_transfer_purify import MODELS

    assert _parse_models(None) is MODELS
    assert _parse_models([]) is MODELS


@pytest.mark.parametrize("spec", ["nokey", "=onlypath", "keyonly="])
def test_parse_models_rejects_malformed(spec):
    with pytest.raises(ValueError):
        _parse_models([spec])


def _make_run(tmp_path, name="0", ckpt="model"):
    """Build a minimal run dir: <name>/.hydra/config.yaml + <name>/models/<ckpt>."""
    run = tmp_path / name
    (run / ".hydra").mkdir(parents=True)
    (run / ".hydra" / "config.yaml").write_text("{}\n")
    (run / "models").mkdir()
    (run / "models" / ckpt).write_text("checkpoint")
    return run


def test_resolve_run_dir_from_checkpoint_file(tmp_path):
    run = _make_run(tmp_path)
    run_dir, ckpt = _resolve_run_dir(str(run / "models" / "model"))
    assert run_dir == run
    assert ckpt == run / "models" / "model"


def test_resolve_run_dir_from_run_dir(tmp_path):
    run = _make_run(tmp_path)
    run_dir, ckpt = _resolve_run_dir(str(run))
    assert run_dir == run
    assert ckpt is None  # left to find_model_checkpoint


def test_resolve_run_dir_from_sweep_root_without_csv_falls_back_to_zero(tmp_path):
    sweep = tmp_path / "seed_sweep_a001_1206"
    _make_run(sweep, name="0")
    _make_run(sweep, name="1")
    run_dir, ckpt = _resolve_run_dir(str(sweep))
    assert run_dir == sweep / "0"
    assert ckpt is None


def test_resolve_run_dir_from_sweep_root_uses_best_run_csv(tmp_path):
    sweep = tmp_path / "outputs" / "ds" / "nat" / "seed_sweep_a001_1206"
    _make_run(sweep, name="0")
    _make_run(sweep, name="1")
    csv = tmp_path / "analysis" / "outputs" / "ds" / "nat" / "seed_sweep_a001_1206"
    csv.mkdir(parents=True)
    # Mirror the real schema: run_path (str) keeps the selected row object-dtype, so
    # str(run_name) stays "1" rather than being float-coerced to "1.0".
    (csv / "evaluation_data.csv").write_text(
        "run_name,run_path,acc\n0,/outputs/ds/nat/sweep/0,0.80\n1,/outputs/ds/nat/sweep/1,0.95\n")
    run_dir, ckpt = _resolve_run_dir(str(sweep))
    assert run_dir == sweep / "1"  # highest acc, not the first run
    assert ckpt is None


def test_load_run_config_reconstructs_a_final_mps_seed_sweep_without_hydra(tmp_path):
    run = (
        tmp_path
        / "outputs/mnist_full_r12/nat/legendre/d3r20c64/seed_sweep_a0_1206/3"
    )
    run.mkdir(parents=True)

    cfg = load_run_config(run)

    assert cfg.dataset.name == "mnist_full_r12"
    assert cfg.born.embedding == "legendre"
    assert cfg.born.init_kwargs.in_dim == 3
    assert cfg.trainer.nll.alpha == 0.0
    assert cfg.tracking.seed == 4


def _stats_fixture(**overrides):
    stats = {
        "model_keys": ["a001", "at"],
        "model_labels": {"a001": "purified a=0.01", "at": "AT-model purified"},
        "paths": {"a001": "outputs/x/nat/a001", "at": "outputs/x/at/seed_sweep"},
        "run_dirs": {"a001": "outputs/x/nat/a001/0", "at": "outputs/x/at/seed_sweep/2"},
        "attack_source": "at",
        # Relative budgets with their legendre (width 2.0) absolute counterparts.
        "eps_rel": 0.3, "abs_eps": 0.6, "attack_num_steps": 40,
        "delta_rel": 0.3, "abs_delta": 0.6, "purify_num_steps": 20,
        "lo": -1.0, "hi": 1.0, "seed": 0, "device": "cpu", "eval_batch_size": 128,
        "n_attacked": 500,
        "n_eligible": 40,
        "eligible_per_class": {0: 20, 3: 20},
        "clean_acc": {"a001": 0.9, "at": 0.8},
        "adv_acc": {"a001": 0.1, "at": 0.05},
        "transfer_rate": {"a001": 0.75, "at": None},
        "stats_cap": 20,
        "n_purified": 20,
        "purify_acc": {"a001": 0.5, "at": 0.25},
        "purify_per_class": {"a001": {0: 0.6, 3: None}, "at": {0: 0.3, 3: None}},
        "purified_per_class_n": {0: 20, 3: 0},
        "row_index": {0: 0, 3: None},
    }
    stats.update(overrides)
    return stats


def test_format_stats_reports_rates_cap_and_missing_rows():
    text = _format_stats(_stats_fixture())
    assert "test inputs attacked      500" in text
    assert "0.750" in text            # transfer rate
    assert "n/a" in text              # attack source has no transfer rate against itself
    assert "capped at --stats-cap 20" in text
    assert "0.500" in text and "0.250" in text  # overall purification success
    assert "WARNING: classes [3]" in text


def test_format_stats_no_cap_note_when_under_cap():
    text = _format_stats(_stats_fixture(n_purified=10, n_eligible=10, stats_cap=20))
    assert "capped at" not in text
    assert "evaluated on 10 of 10 eligible examples" in text


def test_format_stats_no_warning_when_all_rows_present():
    text = _format_stats(_stats_fixture(row_index={0: 0, 3: 7}))
    assert "WARNING" not in text
    assert "eligible-set index 7" in text
