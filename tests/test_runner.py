from __future__ import annotations

from dataclasses import replace

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
            "mean_user_popularity_mean_delta",
            "std_user_popularity_mean_delta",
            "mean_user_heldout_recommender_score_mean",
            "mean_user_heldout_recommender_score_std",
            "mean_user_heldout_internal_score_mean",
            "mean_user_heldout_internal_score_std",
            "mean_user_heldout_score_mean_gap",
            "std_user_heldout_score_mean_gap",
            "fraction_users_recommender_score_mean_gt_internal",
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
            "heldout_recommender_scored_count",
            "heldout_internal_scored_count",
            "heldout_recommender_score_mean",
            "heldout_recommender_score_std",
            "heldout_internal_score_mean",
            "heldout_internal_score_std",
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
