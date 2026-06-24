"""Unit tests for W&B group-key derivation.

The group must be derived from the Hydra output dir (one source of truth shared
by all jobs in a sweep), not from ``datetime.now()`` which gave each job its own
minute and scattered a sweep across many single-run groups.
"""

from pathlib import Path

from experiments.tracking import _derive_group_key


def test_multirun_strips_job_subdir():
    run_dir = Path("/proj/outputs/cricketx/nat/legendre/d3r60c64/hpo_a0_2406/0")
    assert _derive_group_key(run_dir, is_multirun=True) == \
        "cricketx/nat/legendre/d3r60c64/hpo_a0_2406"


def test_singlerun_keeps_leaf():
    run_dir = Path("/proj/outputs/cricketx/nat/legendre/d3r60c64/hpo_a0_2406_1530")
    assert _derive_group_key(run_dir, is_multirun=False) == \
        "cricketx/nat/legendre/d3r60c64/hpo_a0_2406_1530"


def test_stage_segment_preserved():
    run_dir = Path("/proj/outputs/mnist_full_r12/at/legendre/d3r20c64/hpo/at_a05_2406/3")
    assert _derive_group_key(run_dir, is_multirun=True) == \
        "mnist_full_r12/at/legendre/d3r20c64/hpo/at_a05_2406"


def test_all_sweep_jobs_share_one_group():
    base = "/proj/outputs/cricketx/at/legendre/d3r100c64/at_a1_2406"
    keys = {_derive_group_key(Path(f"{base}/{j}"), is_multirun=True) for j in range(4)}
    assert keys == {"cricketx/at/legendre/d3r100c64/at_a1_2406"}


def test_data_root_prefix_ignored():
    """A BM4TC_DATA_ROOT prefix before 'outputs' must not leak into the group."""
    run_dir = Path("/scratch/data/outputs/ecg200/nat/legendre/d3r60c64/hpo_a0_2406/2")
    assert _derive_group_key(run_dir, is_multirun=True) == \
        "ecg200/nat/legendre/d3r60c64/hpo_a0_2406"
