"""
tests/test_environment.py — tests for the Environment component.

Covers: data loading, hold-out split correctness, and embedding persistence
via ChromaDB for both associative (MF) and semantic (sentence-transformer)
collections.
"""
from __future__ import annotations

import numpy as np
import pytest

class TestDataLoading:
    def test_train_ratings_not_empty(self, env):
        assert len(env.train_ratings) > 0

    def test_held_out_not_empty(self, env):
        assert len(env.held_out) > 0

    def test_movie_meta_has_required_columns(self, env):
        for col in ("movieId", "title", "genres", "overview"):
            assert col in env.movie_meta.columns, f"Missing column: {col}"

    def test_train_and_held_out_are_disjoint(self, env):
        """No (userId, movieId) pair should appear in both splits."""
        train_pairs = set(zip(env.train_ratings["userId"], env.train_ratings["movieId"]))
        validation_pairs = set(zip(env.validation["userId"], env.validation["movieId"]))
        held_pairs = set(zip(env.held_out["userId"], env.held_out["movieId"]))
        assert len(train_pairs & validation_pairs) == 0
        assert len(train_pairs & held_pairs) == 0
        assert len(validation_pairs & held_pairs) == 0

    def test_held_out_only_for_eval_users(self, env):
        held_users = set(env.held_out["userId"].unique())
        eval_users = set(env.eval_users)
        assert held_users.issubset(eval_users)

    def test_validation_only_for_eval_users(self, env):
        validation_users = set(env.validation["userId"].unique())
        eval_users = set(env.eval_users)
        assert validation_users.issubset(eval_users)


class TestHoldOutSplit:
    def test_eval_users_populated(self, env):
        assert len(env.eval_users) > 0

    def test_eval_users_have_min_ratings_in_full_data(self, env, tiny_config):
        user_counts = env.all_ratings.groupby("userId").size()
        for uid in env.eval_users:
            assert user_counts.get(uid, 0) >= tiny_config.min_ratings

    def test_held_out_fraction_is_roughly_correct(self, env, tiny_config):
        """Each eval user should have ~holdout_frac of their ratings held out."""
        for uid in env.eval_users:
            user_total = len(env.all_ratings[env.all_ratings["userId"] == uid])
            user_held = len(env.held_out[env.held_out["userId"] == uid])
            expected = max(1, int(user_total * tiny_config.holdout_frac))
            assert user_held == expected, (
                f"User {uid}: expected {expected} held-out ratings, got {user_held}"
            )

    def test_validation_split_exists_for_eval_users(self, env, tiny_config):
        for uid in env.eval_users:
            user_total = len(env.all_ratings[env.all_ratings["userId"] == uid])
            user_validation = len(env.validation[env.validation["userId"] == uid])
            if tiny_config.validation_frac <= 0:
                assert user_validation == 0
                continue
            expected = max(1, int(user_total * tiny_config.validation_frac))
            held_out_count = max(1, int(user_total * tiny_config.holdout_frac))
            max_available = max(0, user_total - held_out_count - 1)
            assert user_validation == min(expected, max_available)


class TestAssociativeEmbeddings:
    def test_collection_exists(self, env):
        col = env.chroma_client.get_collection(env.assoc_collection_name)
        assert col is not None

    def test_collection_has_items(self, env):
        col = env.chroma_client.get_collection(env.assoc_collection_name)
        assert col.count() > 0

    def test_get_item_factors_returns_dict(self, env):
        movie_ids = env.movie_meta["movieId"].head(5).tolist()
        factors = env.get_item_factors(movie_ids)
        assert isinstance(factors, dict)
        # At least some should be present (movies may not all be in training)
        assert len(factors) > 0

    def test_item_factor_has_correct_dimensionality(self, env, tiny_config):
        movie_ids = env.movie_meta["movieId"].head(10).tolist()
        factors = env.get_item_factors(movie_ids)
        for mid, vec in factors.items():
            assert vec.shape == (tiny_config.mf_features,), (
                f"Item {mid} factor has shape {vec.shape}, expected ({tiny_config.mf_features},)"
            )

    def test_user_factors_persisted_on_disk(self, env, tiny_config):
        assert env.user_factor_cache_path.exists(), "user_factors cache not written to disk"

    def test_get_user_factor_returns_vector_for_known_user(self, env, tiny_config):
        # At least some eval users should be in the training set
        found = False
        for uid in env.eval_users:
            vec = env.get_user_factor(uid)
            if vec is not None:
                assert vec.shape == (tiny_config.mf_features,)
                found = True
                break
        assert found, "No eval user had an associative user-factor vector"


class TestSemanticEmbeddings:
    def test_collection_exists(self, env):
        col = env.chroma_client.get_collection(env.semantic_collection_name)
        assert col is not None

    def test_collection_has_items(self, env):
        col = env.chroma_client.get_collection(env.semantic_collection_name)
        assert col.count() > 0

    def test_semantic_vectors_have_nonzero_dimensionality(self, env):
        movie_ids = env.movie_meta["movieId"].head(3).tolist()
        vecs = env.get_semantic_vectors(movie_ids)
        assert len(vecs) == 3
        for mid, vec in vecs.items():
            assert vec.ndim == 1
            assert vec.shape[0] > 0

    def test_semantic_vectors_are_different_per_movie(self, env):
        """Two movies with different descriptions should have different vectors."""
        movie_ids = env.movie_meta["movieId"].head(5).tolist()
        vecs = env.get_semantic_vectors(movie_ids)
        vec_list = list(vecs.values())
        if len(vec_list) >= 2:
            similarity = np.dot(vec_list[0], vec_list[1]) / (
                np.linalg.norm(vec_list[0]) * np.linalg.norm(vec_list[1]) + 1e-9
            )
            # They should not be identical (cosine < 1.0)
            assert similarity < 1.0 - 1e-6


class TestUserPrefEmbeddings:
    def test_collection_exists(self, env):
        col = env.chroma_client.get_collection(env.user_pref_collection_name)
        assert col is not None

    def test_collection_has_items(self, env):
        col = env.chroma_client.get_collection(env.user_pref_collection_name)
        assert col.count() > 0

    def test_get_user_pref_item_factors_returns_dict(self, env):
        movie_ids = env.movie_meta["movieId"].head(5).tolist()
        factors = env.get_user_pref_item_factors(movie_ids)
        assert isinstance(factors, dict)
        assert len(factors) > 0

    def test_item_pref_factor_correct_dimensionality(self, env, tiny_config):
        movie_ids = env.movie_meta["movieId"].head(10).tolist()
        factors = env.get_user_pref_item_factors(movie_ids)
        for mid, vec in factors.items():
            assert vec.shape == (tiny_config.user_pref_features,), (
                f"Item {mid} user-pref factor has shape {vec.shape}, "
                f"expected ({tiny_config.user_pref_features},)"
            )

    def test_user_pref_factors_persisted_on_disk(self, env, tiny_config):
        assert env.user_pref_factor_cache_path.exists(), (
            "user_pref_factors cache not written to disk"
        )

    def test_get_user_pref_factor_for_known_user(self, env, tiny_config):
        found = False
        for uid in env.eval_users:
            vec = env.get_user_pref_factor(uid)
            if vec is not None:
                assert vec.shape == (tiny_config.user_pref_features,)
                found = True
                break
        assert found, "No eval user had a user-pref factor vector"

    def test_get_user_pref_factor_cold_start_fallback(self, env, tiny_config):
        """Unknown users should receive a plausible fallback vector (centroid + noise)."""
        unknown_uid = -999_999
        vec = env.get_user_pref_factor(unknown_uid)
        assert vec is not None
        assert vec.shape == (tiny_config.user_pref_features,)

    def test_item_pref_factors_are_unit_norm(self, env):
        """Stored item factors should be L2-normalised."""
        movie_ids = env.movie_meta["movieId"].head(10).tolist()
        factors = env.get_user_pref_item_factors(movie_ids)
        for mid, vec in factors.items():
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 1e-5, (
                f"Item {mid} user-pref factor has norm {norm:.4f}, expected ~1.0"
            )
