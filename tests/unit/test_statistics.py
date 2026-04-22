import pytest
import numpy as np
import pandas as pd
from analysis.utils.statistics import (
    clean_column_name,
    compute_statistics,
    get_best_run,
    compute_pareto_frontier,
    compute_metric_correlations,
)


# ---- clean_column_name ----

def test_clean_column_name_config_prefix():
    assert clean_column_name("config/lr") == "Lr"


def test_clean_column_name_eval_prefix():
    result = clean_column_name("eval/test/acc")
    assert result == "Test / Acc"


def test_clean_column_name_underscores():
    result = clean_column_name("mia_accuracy")
    assert result == "Mia Accuracy"


# ---- compute_statistics ----

def test_compute_statistics_basic():
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0]})
    stats = compute_statistics(df, "val")
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["best"] == pytest.approx(4.0)
    assert stats["n"] == 4
    assert stats["std"] > 0


def test_compute_statistics_single_row():
    df = pd.DataFrame({"val": [5.0]})
    stats = compute_statistics(df, "val")
    assert stats["n"] == 1
    assert np.isnan(stats["std"])


def test_compute_statistics_all_nan():
    df = pd.DataFrame({"val": [np.nan, np.nan]})
    stats = compute_statistics(df, "val")
    assert stats["n"] == 0
    assert np.isnan(stats["mean"])


def test_compute_statistics_missing_col():
    df = pd.DataFrame({"other": [1.0, 2.0]})
    stats = compute_statistics(df, "val")
    assert stats["n"] == 0
    assert np.isnan(stats["mean"])


def test_compute_statistics_effective_n():
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0]})
    stats = compute_statistics(df, "val", effective_n=2)
    expected_stderr = df["val"].std() / np.sqrt(2)
    assert stats["stderr"] == pytest.approx(expected_stderr, rel=1e-4)


# ---- get_best_run ----

def test_get_best_run_maximize():
    df = pd.DataFrame({"metric": [0.5, 0.9, 0.7], "name": ["a", "b", "c"]})
    best = get_best_run(df, "metric", minimize=False)
    assert best["name"] == "b"


def test_get_best_run_minimize():
    df = pd.DataFrame({"metric": [0.5, 0.1, 0.7], "name": ["a", "b", "c"]})
    best = get_best_run(df, "metric", minimize=True)
    assert best["name"] == "b"


def test_get_best_run_missing_col_returns_none():
    df = pd.DataFrame({"other": [1.0, 2.0]})
    assert get_best_run(df, "missing", minimize=False) is None


def test_get_best_run_all_nan_returns_none():
    df = pd.DataFrame({"metric": [np.nan, np.nan]})
    assert get_best_run(df, "metric", minimize=False) is None


# ---- compute_pareto_frontier ----

def test_compute_pareto_frontier_known():
    x = np.array([0.5, 0.8, 0.3, 0.7])
    y = np.array([0.8, 0.3, 0.9, 0.6])
    # Points: (0.5, 0.8), (0.8, 0.3), (0.3, 0.9), (0.7, 0.6)
    # Pareto (maximize both): (0.5,0.8) is dominated by none of the others?
    # (0.8,0.3) is dominated by (0.5,0.8)? No, 0.8 > 0.5 so not dominated.
    # Actually let me think: point i is dominated if there's j s.t. x[j]>=x[i] and y[j]>=y[i] with at least one strict.
    # (0.3, 0.9): dominated by (0.5, 0.8)? 0.5>0.3, but 0.8<0.9. Not dominated.
    # (0.7, 0.6): dominated by (0.5,0.8)? 0.5<0.7 no. Dominated by (0.8,0.3)? 0.3<0.6 no.
    # So all 4 might be on frontier. Let me check more carefully.
    # (0.5, 0.8): is there j with x[j]>=0.5 and y[j]>=0.8 and at least one strict?
    #   (0.8, 0.3): 0.3 < 0.8, no. (0.3, 0.9): 0.3 < 0.5, no. (0.7, 0.6): 0.6 < 0.8, no.
    #   Not dominated.
    # (0.8, 0.3): is there j with x[j]>=0.8 and y[j]>=0.3 and one strict?
    #   None have x>=0.8 except itself. Not dominated.
    # (0.3, 0.9): is there j with x[j]>=0.3 and y[j]>=0.9 and one strict?
    #   None have y>=0.9 except itself. Not dominated.
    # (0.7, 0.6): is there j with x[j]>=0.7 and y[j]>=0.6 and one strict?
    #   (0.8,0.3): 0.3 < 0.6, no. (0.5,0.8): 0.5 < 0.7, no. Not dominated.
    # All Pareto — let's use a simpler case instead.
    x2 = np.array([0.3, 0.5, 0.8])
    y2 = np.array([0.9, 0.7, 0.3])
    mask = compute_pareto_frontier(x2, y2, maximize_x=True, maximize_y=True)
    # All three: (0.3,0.9) not dominated (highest y), (0.8,0.3) not dominated (highest x)
    # (0.5,0.7): dominated by neither? (0.8,0.3) has higher x but lower y. Not dominated.
    assert mask.sum() == 3


def test_compute_pareto_frontier_all_pareto():
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])
    mask = compute_pareto_frontier(x, y)
    assert mask.all()


# ---- compute_metric_correlations ----

def test_compute_metric_correlations_identical_cols():
    vals = np.linspace(0, 1, 20)
    df = pd.DataFrame({"a": vals, "b": vals})
    corr = compute_metric_correlations(df, ["a", "b"])
    assert not corr.empty
    assert corr.iloc[0, 1] == pytest.approx(1.0, abs=1e-5)
