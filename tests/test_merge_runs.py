"""
tests/test_merge_runs.py - merging harness runs across MLflow stores (issue #20).

Two runs of the same backend on the same config, each in its own sqlite store
as if on two machines, must merge into one table. The merge must refuse runs
whose `scored_pairs.csv` differ, and every harness run must carry its split
fingerprint. Synthetic fixtures only; MLflow writes to a tmp dir.
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from experiments.backends import BiasOnlyBackend
from experiments.compare_backends import run_backend
from scripts.merge_runs import (
    RunMismatchError,
    load_from_artifacts,
    load_from_mlflow,
    main,
    merge_runs,
)
from sim.population import build_user_assignments

CAP = 3


@pytest.fixture(scope="module")
def assignments(tiny_config, env):
    return build_user_assignments(tiny_config, env, np.random.default_rng(tiny_config.random_seed))


def _harness_run(tiny_config, env, assignments, store: Path, cap: int = CAP) -> tuple[str, str]:
    """One bias_only harness run in its own store. Returns (tracking_uri, run_id)."""
    store.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{store / 'mlflow.db'}"
    run_id, _ = run_backend(
        tiny_config, env, assignments, BiasOnlyBackend(),
        max_items_per_user=cap, item_selection="first",
        tracking_uri=uri, run_name="bias_only",
    )
    return uri, run_id


@pytest.fixture(scope="module")
def two_stores(tiny_config, env, assignments, tmp_path_factory):
    """The same arm, run twice into two separate stores."""
    return [
        _harness_run(tiny_config, env, assignments, tmp_path_factory.mktemp(f"machine{i}"))
        for i in (1, 2)
    ]


class TestSplitFingerprint:

    def test_every_harness_run_logs_split_cache_key(self, tiny_config, two_stores):
        for uri, run_id in two_stores:
            params = mlflow.MlflowClient(tracking_uri=uri).get_run(run_id).data.params
            assert params["split_cache_key"] == tiny_config.split_cache_key()

    def test_harness_refuses_a_relative_tracking_uri(self, tiny_config, env, assignments):
        with pytest.raises(ValueError, match="relative path"):
            run_backend(
                tiny_config, env, assignments, BiasOnlyBackend(),
                max_items_per_user=CAP, item_selection="first",
                tracking_uri="sqlite:///mlflow.db", run_name="bias_only",
            )


class TestMerge:

    def test_two_stores_merge_into_one_table(self, two_stores):
        arms = [load_from_mlflow(uri, run_id) for uri, run_id in two_stores]
        summary, per_pair = merge_runs(arms)

        assert len(summary) == 2
        assert summary["backend"].tolist() == ["bias_only", "bias_only"]
        # Same arm, same pairs: identical metrics, recomputed from the parquet.
        assert summary.loc[0, "error/mae"] == pytest.approx(summary.loc[1, "error/mae"])
        assert summary.loc[0, "meta/pair_count"] == len(per_pair) > 0
        labels = [arm.label for arm in arms]
        np.testing.assert_array_equal(
            per_pair[f"predicted_rating[{labels[0]}]"], per_pair[f"predicted_rating[{labels[1]}]"]
        )
        assert not per_pair.duplicated(["userId", "movieId"]).any()

    def test_artifacts_directory_merges_with_a_store_run(self, two_stores, tmp_path):
        """The cross-machine route: one run downloaded and copied, one local."""
        (uri_a, run_a), (uri_b, run_b) = two_stores
        copied = tmp_path / "copied-from-server"
        mlflow.MlflowClient(tracking_uri=uri_b).download_artifacts(run_b, "", str(copied))
        summary, _ = merge_runs([load_from_mlflow(uri_a, run_a), load_from_artifacts(copied)])
        assert summary["source"].tolist() == [run_a[:8], "copied-from-server"]

    def test_refuses_runs_that_scored_different_pairs(self, tiny_config, env, assignments, two_stores, tmp_path):
        uri_a, run_a = two_stores[0]
        uri_c, run_c = _harness_run(tiny_config, env, assignments, tmp_path / "machine3", cap=CAP - 1)
        with pytest.raises(RunMismatchError, match="different pairs"):
            merge_runs([load_from_mlflow(uri_a, run_a), load_from_mlflow(uri_c, run_c)])

    def test_refuses_equal_counts_that_are_not_equal_pairs(self, two_stores, tmp_path):
        """Same pair count, one pair swapped: still refused."""
        (uri_a, run_a), (uri_b, run_b) = two_stores
        tampered = tmp_path / "tampered"
        mlflow.MlflowClient(tracking_uri=uri_b).download_artifacts(run_b, "", str(tampered))
        for name in ("scored_pairs.csv", "per_item_predictions.parquet"):
            path = tampered / name
            frame = pd.read_csv(path) if name.endswith(".csv") else pd.read_parquet(path)
            frame.loc[0, "movieId"] = frame["movieId"].max() + 1
            frame.to_csv(path, index=False) if name.endswith(".csv") else frame.to_parquet(path, index=False)

        arms = [load_from_mlflow(uri_a, run_a), load_from_artifacts(tampered)]
        assert len(arms[0].scored_pairs) == len(arms[1].scored_pairs)
        with pytest.raises(RunMismatchError, match="1 only in"):
            merge_runs(arms)

    def test_refuses_different_split_fingerprints(self, two_stores):
        arms = [load_from_mlflow(uri, run_id) for uri, run_id in two_stores]
        arms[1].split_cache_key = "000000000000"
        with pytest.raises(RunMismatchError, match="different splits"):
            merge_runs(arms)

    def test_refuses_different_actual_ratings(self, two_stores):
        arms = [load_from_mlflow(uri, run_id) for uri, run_id in two_stores]
        arms[1].predictions.loc[0, "actual_rating"] += 1.0
        with pytest.raises(RunMismatchError, match="actual ratings"):
            merge_runs(arms)

    def test_cli_exits_non_zero_on_mismatch_and_writes_on_success(
        self, tiny_config, env, assignments, two_stores, tmp_path
    ):
        (uri_a, run_a), (uri_b, run_b) = two_stores
        out = tmp_path / "out"
        main(["--run", uri_a, run_a, "--run", uri_b, run_b, "--out", str(out)])
        assert len(pd.read_csv(out / "summary.csv")) == 2
        assert len(pd.read_parquet(out / "per_pair.parquet")) > 0

        uri_c, run_c = _harness_run(tiny_config, env, assignments, tmp_path / "machine3", cap=CAP - 1)
        with pytest.raises(SystemExit, match="refusing to merge"):
            main(["--run", uri_a, run_a, "--run", uri_c, run_c])
