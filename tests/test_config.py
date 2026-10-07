"""
tests/test_config.py - the MLflow tracking URI is absolute and per machine (issue #20).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sim.config import REPO_ROOT, SimConfig, check_tracking_uri, default_tracking_uri


class TestTrackingUri:

    def test_default_is_the_repo_root_store_as_an_absolute_path(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        assert default_tracking_uri() == f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
        assert Path(REPO_ROOT / "pyproject.toml").is_file()

    def test_environment_variable_sets_the_store_per_machine(self, monkeypatch, tmp_path):
        uri = f"sqlite:///{tmp_path / 'server.db'}"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
        assert SimConfig().mlflow_tracking_uri == uri

    @pytest.mark.parametrize("uri", ["sqlite:///mlflow.db", "mlruns", "file:mlruns"])
    def test_relative_local_stores_are_refused(self, uri):
        with pytest.raises(ValueError, match="relative path"):
            check_tracking_uri(uri)
        with pytest.raises(ValueError, match="relative path"):
            SimConfig(mlflow_tracking_uri=uri)

    @pytest.mark.parametrize(
        "uri", ["sqlite:////abs/mlflow.db", "/abs/mlruns", "file:///abs/mlruns", "http://tracking:5000"],
    )
    def test_absolute_and_remote_stores_pass(self, uri):
        check_tracking_uri(uri)
