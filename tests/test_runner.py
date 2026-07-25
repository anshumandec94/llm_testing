from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from sim.runner import SimulationRunner


class TestRecommenderOnlyProfile:
    def test_recommender_only_logs_popularity_diagnostics(
        self, tiny_config, env, tmp_path, monkeypatch
    ):
        cfg = replace(
            tiny_config,
            experiment_profile="recommender_only",
            mlflow_tracking_uri=str(tmp_path / "mlruns"),
            experiment_name="test-recommender-only",
            force_rebuild_embeddings=False,
        )
        monkeypatch.chdir(tmp_path)

        summary = SimulationRunner(cfg).run()

        assert len(summary) == 1
        expected_summary_cols = {
            "round",
            "user_count",
            "hit_rate",
            f"ndcg_at_{cfg.rec_list_size}",
            "fraction_users_with_holdout_hit",
            "mean_user_recommended_popularity_mean",
            "mean_user_recommended_popularity_std",
            "mean_user_heldout_popularity_mean",
            "mean_user_heldout_popularity_std",
            "mean_user_comparison_popularity_mean",
            "mean_user_comparison_popularity_std",
            "mean_user_heldout_debiased_rating_mean",
            "mean_user_heldout_debiased_rating_std",
            "mean_user_popularity_mean_delta",
            "std_user_popularity_mean_delta",
            "mean_user_heldout_recommender_score_mean",
            "mean_user_heldout_recommender_score_std",
            "mean_user_heldout_internal_score_mean",
            "mean_user_heldout_internal_score_std",
            "mean_user_heldout_recommender_residual_pearson",
            "mean_user_heldout_recommender_residual_spearman",
            "mean_user_heldout_internal_residual_pearson",
            "mean_user_heldout_internal_residual_spearman",
            "mean_user_heldout_score_mean_gap",
            "std_user_heldout_score_mean_gap",
            "fraction_users_recommender_score_mean_gt_internal",
            "fraction_users_recommender_residual_pearson_gt_internal",
            "global_heldout_recommender_residual_pearson",
            "global_heldout_recommender_residual_spearman",
            "global_heldout_internal_residual_pearson",
            "global_heldout_internal_residual_spearman",
            "fraction_users_recommended_mean_gt_comparison_mean",
        }
        assert expected_summary_cols.issubset(summary.columns)

        row = summary.iloc[0]
        assert row["round"] == 1
        assert row["user_count"] > 0
        assert 0.0 <= row["fraction_users_with_holdout_hit"] <= 1.0
        assert 0.0 <= row["fraction_users_recommended_mean_gt_comparison_mean"] <= 1.0

        user_diag_path = tmp_path / "mlartifacts" / "recommender_only_user_diagnostics.parquet"
        recs_path = tmp_path / "mlartifacts" / "recommender_only_recommendations.parquet"
        heldout_scores_path = tmp_path / "mlartifacts" / "recommender_only_heldout_scores.parquet"
        summary_path = tmp_path / "mlartifacts" / "summary.csv"

        assert user_diag_path.exists()
        assert recs_path.exists()
        assert heldout_scores_path.exists()
        assert summary_path.exists()

        user_df = pd.read_parquet(user_diag_path)
        required_user_cols = {
            "userId",
            "recommended_item_count",
            "heldout_item_count",
            "comparison_item_count",
            "hit_rate",
            f"ndcg_at_{cfg.rec_list_size}",
            "recommended_popularity_mean",
            "recommended_popularity_std",
            "heldout_popularity_mean",
            "heldout_popularity_std",
            "comparison_popularity_mean",
            "comparison_popularity_std",
            "popularity_mean_delta",
            "heldout_debiased_rating_mean",
            "heldout_debiased_rating_std",
            "heldout_recommender_scored_count",
            "heldout_internal_scored_count",
            "heldout_recommender_score_mean",
            "heldout_recommender_score_std",
            "heldout_internal_score_mean",
            "heldout_internal_score_std",
            "heldout_recommender_residual_pearson",
            "heldout_recommender_residual_spearman",
            "heldout_internal_residual_pearson",
            "heldout_internal_residual_spearman",
            "heldout_score_mean_gap",
        }
        assert required_user_cols.issubset(user_df.columns)
        assert len(user_df) == int(row["user_count"])
        assert np.allclose(
            user_df["heldout_popularity_mean"], user_df["comparison_popularity_mean"]
        )
        assert np.allclose(
            user_df["popularity_mean_delta"],
            user_df["recommended_popularity_mean"] - user_df["comparison_popularity_mean"],
        )
        assert np.allclose(
            user_df["heldout_score_mean_gap"],
            user_df["heldout_recommender_score_mean"] - user_df["heldout_internal_score_mean"],
        )
        assert np.isclose(
            row["mean_user_popularity_mean_delta"],
            user_df["popularity_mean_delta"].mean(),
        )
        assert np.isclose(
            row["mean_user_heldout_score_mean_gap"],
            user_df["heldout_score_mean_gap"].mean(),
        )
        assert np.isclose(
            row["mean_user_heldout_debiased_rating_mean"],
            user_df["heldout_debiased_rating_mean"].mean(),
        )

        heldout_scores_df = pd.read_parquet(heldout_scores_path)
        required_heldout_score_cols = {
            "userId",
            "movieId",
            "rating",
            "rating_bias",
            "debiased_rating",
            "recommender_score",
            "internal_score",
        }
        assert required_heldout_score_cols.issubset(heldout_scores_df.columns)
        assert np.allclose(
            heldout_scores_df["debiased_rating"],
            heldout_scores_df["rating"] - heldout_scores_df["rating_bias"],
        )

    def test_recommender_only_supports_traditional_agents(
        self, tiny_config, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        for agent_type in ("residual_profile", "item_item"):
            cfg = replace(
                tiny_config,
                agent_type=agent_type,
                experiment_profile="recommender_only",
                mlflow_tracking_uri=str(tmp_path / f"mlruns-{agent_type}"),
                experiment_name=f"test-{agent_type}",
                force_rebuild_embeddings=False,
            )
            summary = SimulationRunner(cfg).run()
            assert len(summary) == 1
            assert summary.iloc[0]["user_count"] > 0

    def test_recommender_only_supports_replicated_agent_comparisons(
        self, tiny_config, tmp_path, monkeypatch, env
    ):
        cfg = replace(
            tiny_config,
            experiment_profile="recommender_only",
            agent_types=["associative", "item_item"],
            agent_assignment_mode="one_per_agent_type",
            mlflow_tracking_uri=str(tmp_path / "mlruns-replicated"),
            experiment_name="test-replicated-comparison",
            force_rebuild_embeddings=False,
        )
        monkeypatch.chdir(tmp_path)

        summary = SimulationRunner(cfg).run()

        assert len(summary) == 1
        assert summary.iloc[0]["user_count"] == len(env.eval_users) * 2

        user_df = pd.read_parquet(
            tmp_path / "mlartifacts" / "recommender_only_user_diagnostics.parquet"
        )
        assert {"simulation_user_id", "agent_type"}.issubset(user_df.columns)
        assert len(user_df) == len(env.eval_users) * 2
        assert set(user_df["agent_type"]) == {"associative", "item_item"}
        per_user_agent_counts = (
            user_df.groupby("userId")["agent_type"].nunique().to_dict()
        )
        assert set(per_user_agent_counts.values()) == {2}


class TestExplicitOnlyFeedback:
    def test_user_session_filters_to_explicit_ratings_for_learning(
        self, tiny_config, monkeypatch
    ):
        runner = SimulationRunner(tiny_config)
        rng = np.random.default_rng(tiny_config.random_seed)
        ctx = runner._setup_components(rng)
        uid = ctx.env.eval_users[0]
        ua = ctx.users[uid]
        held_ids = ctx.held_out_sets[uid]
        rated_movie = int(ctx.env.movie_meta["movieId"].iloc[0])

        captured: dict[str, Any] = {}

        def fake_update_user(user_id, interactions):
            captured["recommender"] = (user_id, interactions)

        def fake_ua_update(self, rnd, interactions, acted_item_factors, cfg):
            captured["agent"] = {
                "round": rnd,
                "interactions": interactions,
                "factor_ids": sorted(acted_item_factors),
            }

        def fake_request_loop(user_id, rnd, _ua, _held_ids, _ctx):
            return (
                [],
                [
                    (101, "watch", 4.5),
                    (102, "add_to_list", 3.0),
                    (rated_movie, "rate", 4.0),
                ],
            )

        monkeypatch.setattr(ctx.recommender, "update_user", fake_update_user)
        monkeypatch.setattr(ua, "update", fake_ua_update.__get__(ua, type(ua)))
        monkeypatch.setattr(ua, "will_attend", lambda _rng: True)
        monkeypatch.setattr(runner, "_run_request_loop", fake_request_loop)

        runner._run_user_session(uid, 1, ua, held_ids, ctx)

        assert captured["recommender"] == (uid, [(rated_movie, 4.0)])
        agent_call = captured["agent"]
        assert agent_call["round"] == 1
        assert agent_call["factor_ids"] == [rated_movie]
        assert len(agent_call["interactions"]) == 1
        movie_id, action, residual = agent_call["interactions"][0]
        assert movie_id == rated_movie
        assert action == "rate"
        assert residual == ctx.env.debias_rating(uid, rated_movie, 4.0)

    def test_explicit_only_feedback_debiases_against_base_user_in_replicated_mode(
        self, tiny_config, monkeypatch
    ):
        cfg = replace(
            tiny_config,
            agent_types=["associative", "item_item"],
            agent_assignment_mode="one_per_agent_type",
        )
        runner = SimulationRunner(cfg)
        rng = np.random.default_rng(cfg.random_seed)
        ctx = runner._setup_components(rng)
        ua = next(user for user in ctx.users.values() if user.uid != user.base_user_id)
        uid = ua.uid
        held_ids = ctx.held_out_sets[uid]
        rated_movie = int(ctx.env.movie_meta["movieId"].iloc[0])

        captured: dict[str, Any] = {}

        def fake_update_user(user_id, interactions):
            captured["recommender"] = (user_id, interactions)

        def fake_ua_update(self, rnd, interactions, acted_item_factors, cfg):
            captured["agent"] = interactions

        def fake_request_loop(user_id, rnd, _ua, _held_ids, _ctx):
            return ([], [(rated_movie, "rate", 4.0)])

        monkeypatch.setattr(ctx.recommender, "update_user", fake_update_user)
        monkeypatch.setattr(ua, "update", fake_ua_update.__get__(ua, type(ua)))
        monkeypatch.setattr(ua, "will_attend", lambda _rng: True)
        monkeypatch.setattr(runner, "_run_request_loop", fake_request_loop)

        runner._run_user_session(uid, 1, ua, held_ids, ctx)

        assert captured["recommender"] == (uid, [(rated_movie, 4.0)])
        assert captured["agent"] == [
            (rated_movie, "rate", ctx.env.debias_rating(ua.base_user_id, rated_movie, 4.0))
        ]
