"""
tests/test_compare_backends.py - the backend comparison harness (issue #14).

The harness is only useful if every arm scores the same (user, item) pairs,
so that is the property pinned hardest: two different backends through the
harness must produce the same pairs and byte-identical `scored_pairs.csv`.
Synthetic fixtures only; MLflow writes to a tmp dir.
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from experiments.backends import AssociativeBackend, BiasOnlyBackend
from experiments.bias_only_null import score_bias_only
from experiments.compare_backends import (
    compute_metrics,
    run_backend,
    score_backend,
    unavailable_reason,
)
from sim.population import build_user_assignments
from sim.residual_factors import ResidualFactors

CAP = 3


@pytest.fixture(scope="module")
def assignments(tiny_config, env):
    return build_user_assignments(tiny_config, env, np.random.default_rng(tiny_config.random_seed))


@pytest.fixture(scope="module")
def associative(env):
    return AssociativeBackend.from_env(env)


@pytest.fixture(scope="module")
def runs(tiny_config, env, assignments, associative, tmp_path_factory):
    uri = f"sqlite:///{tmp_path_factory.mktemp('mlruns') / 'mlflow.db'}"
    out = {}
    for backend in (BiasOnlyBackend(), associative):
        run_id, metrics = run_backend(
            tiny_config, env, assignments, backend,
            max_items_per_user=CAP, item_selection="first",
            tracking_uri=uri, run_name=backend.name,
        )
        dst = tmp_path_factory.mktemp(backend.name)
        mlflow.set_tracking_uri(uri)
        mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=str(dst))
        out[backend.name] = (dst, metrics)
    return out


def _score(tiny_config, env, assignments, backend, cap=CAP, selection="first"):
    return score_backend(
        env, assignments, backend, cap, selection,
        seed=tiny_config.random_seed, split=tiny_config.recommender_eval_split,
    )


class TestSamePairs:

    def test_two_backends_score_exactly_the_same_pairs(self, tiny_config, env, assignments, associative):
        for selection in ("first", "random"):
            a = _score(tiny_config, env, assignments, BiasOnlyBackend(), selection=selection)
            b = _score(tiny_config, env, assignments, associative, selection=selection)
            assert len(a) > 0
            pd.testing.assert_frame_equal(a[["userId", "movieId"]], b[["userId", "movieId"]])
            np.testing.assert_array_equal(a["actual_rating"], b["actual_rating"])

    def test_pairs_match_the_bias_only_null(self, tiny_config, env, assignments):
        frame = _score(tiny_config, env, assignments, BiasOnlyBackend())
        null = score_bias_only(
            env, assignments, CAP, seed=tiny_config.random_seed,
            split=tiny_config.recommender_eval_split,
        )
        assert list(zip(frame["userId"], frame["movieId"])) == list(zip(null["userId"], null["movieId"]))
        np.testing.assert_array_equal(frame["predicted_rating"], null["pred_null"])

    def test_scored_pairs_csv_is_byte_identical(self, runs):
        a = (runs["bias_only"][0] / "scored_pairs.csv").read_bytes()
        b = (runs["associative"][0] / "scored_pairs.csv").read_bytes()
        assert len(a) > len("userId,movieId\n")
        assert a == b


class TestArtifacts:

    def test_per_item_predictions_parquet(self, runs):
        dst, metrics = runs["associative"]
        frame = pd.read_parquet(Path(dst) / "per_item_predictions.parquet")
        assert list(frame.columns) == [
            "userId", "movieId", "backend", "predicted_residual",
            "actual_residual", "predicted_rating", "actual_rating",
        ]
        assert (frame["backend"] == "associative").all()
        np.testing.assert_allclose(
            frame["predicted_rating"] - frame["predicted_residual"],
            frame["actual_rating"] - frame["actual_residual"],
        )
        assert len(frame) == metrics["meta/pair_count"]

    def test_metrics_logged(self, runs):
        _, metrics = runs["bias_only"]
        for key in ("error/mae", "error/rmse", "error/mae_clipped", "error/mae_user_se",
                    "meta/user_count", "meta/item_count", "meta/nan_count"):
            assert np.isfinite(metrics[key]), key
        assert metrics["meta/nan_count"] == 0


class TestMetrics:

    def test_nan_is_counted_and_excluded(self):
        frame = pd.DataFrame({
            "userId": [1, 1, 2, 2],
            "predicted_rating": [3.0, np.nan, 6.0, 2.0],
            "actual_rating": [4.0, 1.0, 5.0, 2.0],
        })
        m = compute_metrics(frame)
        assert m["meta/nan_count"] == 1
        assert m["meta/item_count"] == 3
        assert m["meta/pair_count"] == 4
        assert m["meta/user_count"] == 2
        assert m["error/mae"] == pytest.approx(2 / 3)
        # Clipping 6.0 to 5.0 removes that error entirely.
        assert m["error/mae_clipped"] == pytest.approx(1 / 3)
        # User 1 mean error 1, user 2 mean error 0.
        assert m["error/mae_clipped_user_se"] == pytest.approx(np.std([1.0, 0.0], ddof=1) / np.sqrt(2))

    def test_uncovered_items_are_nan_not_zero(self, tiny_config, env, assignments):
        # Factors for one user and item that appear nowhere in the fixture.
        unrelated = ResidualFactors(
            user_ids=np.array([10_000]), item_ids=np.array([10_000]),
            user_factors=np.ones((1, 2)), item_factors=np.ones((1, 2)), singular_values=None,
        )
        frame = _score(tiny_config, env, assignments, AssociativeBackend(unrelated))
        assert len(frame) > 0
        assert frame["predicted_rating"].isna().all()
        assert frame["actual_rating"].notna().all()


class TestCli:

    @pytest.mark.parametrize("name", ["residual_profile", "item_item"])
    def test_stub_backends_are_refused(self, name):
        assert unavailable_reason(name)

    @pytest.mark.parametrize("name", ["bias_only", "associative", "llm", "sasrec"])
    def test_buildable_backends_are_allowed(self, name):
        assert unavailable_reason(name) is None
