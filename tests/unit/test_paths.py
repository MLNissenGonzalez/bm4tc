import os
from pathlib import Path

import pytest

from src.utils.paths import data_root, _PROJECT_ROOT


def test_default_is_project_root(monkeypatch):
    monkeypatch.delenv("BM4TC_DATA_ROOT", raising=False)
    assert data_root() == _PROJECT_ROOT


def test_env_var_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("BM4TC_DATA_ROOT", str(tmp_path))
    assert data_root() == tmp_path


def test_env_var_returns_path_type(monkeypatch, tmp_path):
    monkeypatch.setenv("BM4TC_DATA_ROOT", str(tmp_path))
    assert isinstance(data_root(), Path)
