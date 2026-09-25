"""
tests/test_bias_only_null.py - the bias-only null reference (issue #26).

The null is only meaningful if it scores exactly the pairs the published arms
scored and predicts nothing but the bias baseline, so those are the two
properties pinned here, plus the clustered-interval arithmetic the report
quotes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.bias_only_null import clustered_mae, paired_difference, score_bias_only
from experiments.llm_vs_associative import score_associative
from sim.population import build_user_assignments
from sim.user_agent import SimulatedUser


@pytest.fixture(scope="module")
def eval_context(tiny_config, env):
    rng = np.random.default_rng(tiny_config.random_seed)
    assignments = build_user_assignments(tiny_config, env, rng)
    users, _ = SimulatedUser.build_population(tiny_config, env, rng, assignments=assignments)
    return assignments, users


class TestScoreBiasOnly:

    def test_scores_the_same_pairs_as_the_associative_arm(self, tiny_config, env, eval_context):
        """A null on a different item set recreates the epic #1 confound."""
        assignments, users = eval_context
        for selection in ("first", "random"):
            frame = score_bias_only(
                env, assignments, 2, item_selection=selection,
                seed=tiny_config.random_seed, split=tiny_config.recommender_eval_split,
            )
            _, actual, pairs = score_associative(
                env, assignments, users, max_items_per_user=2,
                item_selection=selection, seed=tiny_config.random_seed,
            )
            assert list(zip(frame["userId"], frame["movieId"])) == pairs
            np.testing.assert_allclose(frame["rating"], actual)
            assert len(frame) > 0

    def test_predicts_the_bias_baseline_unclamped(self, tiny_config, env, eval_context):
        assignments, _ = eval_context
        frame = score_bias_only(
            env, assignments, None, split=tiny_config.recommender_eval_split,
        )
        expected = [env.get_rating_bias(u, m) for u, m in zip(frame["userId"], frame["movieId"])]
        np.testing.assert_allclose(frame["pred_null"], expected)

    def test_debiased_residual_is_zero(self, tiny_config, env, eval_context):
        """The defining property: the null carries no interaction signal."""
        assignments, _ = eval_context
        frame = score_bias_only(env, assignments, 3, split=tiny_config.recommender_eval_split)
        residuals = [
            env.debias_rating(u, m, p)
            for u, m, p in zip(frame["userId"], frame["movieId"], frame["pred_null"])
        ]
        np.testing.assert_allclose(residuals, 0.0, atol=1e-12)


class TestClusteredIntervals:

    @pytest.fixture
    def frame(self):
        # User 1 errors 0 and 2, user 2 errors 1 and 1, user 3 errors 3 and 3.
        return pd.DataFrame({
            "userId": [1, 1, 2, 2, 3, 3],
            "rating": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            "a":      [3.0, 5.0, 4.0, 2.0, 0.0, 6.0],
            "b":      [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        })

    def test_se_is_over_per_user_means(self, frame):
        stats = clustered_mae(frame, "a")
        per_user = np.array([1.0, 1.0, 3.0])
        assert stats["mae"] == pytest.approx(10 / 6)
        assert stats["user_mae"] == pytest.approx(per_user.mean())
        assert stats["se"] == pytest.approx(per_user.std(ddof=1) / np.sqrt(3))
        assert stats["n_users"] == 3
        assert stats["n_items"] == 6

    def test_user_and_item_means_differ_when_users_are_unequal(self, frame):
        stats = clustered_mae(frame.iloc[:5], "a")
        assert stats["mae"] == pytest.approx(7 / 5)
        assert stats["user_mae"] == pytest.approx((1.0 + 1.0 + 3.0) / 3)

    def test_paired_difference_sign_and_value(self, frame):
        stats = paired_difference(frame, "a", "b")
        # b has zero error everywhere, so the difference is a's per-user MAE.
        assert stats["diff"] == pytest.approx(5 / 3)
        assert stats["t"] > 0
        assert stats["n_users"] == 3
