"""
tests/test_recommender.py — tests for the Recommender component.

Covers: recommendation generation, training-item exclusion, seen-item
tracking (mark_sent / advance_round), and the re-request mechanism.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lenskit.data import ItemList

from sim.recommender import Recommender


class TestRecommendBasic:
    def test_recommend_returns_item_list(self, recommender, env):
        uid = env.eval_users[0]
        recs = recommender.recommend(uid, n=5)
        assert isinstance(recs, ItemList)

    def test_recommend_returns_at_most_n_items(self, recommender, env):
        uid = env.eval_users[0]
        recs = recommender.recommend(uid, n=5)
        assert len(recs) <= 5

    def test_recommend_returns_nonzero_items(self, recommender, env):
        uid = env.eval_users[0]
        recs = recommender.recommend(uid, n=10)
        assert len(recs) > 0, "Recommender returned 0 items for an eval user"

    def test_recommendation_ids_are_integers(self, recommender, env):
        uid = env.eval_users[0]
        recs = recommender.recommend(uid, n=5)
        for iid in recs.ids():
            assert isinstance(int(iid), int)


class TestSeenItemExclusion:
    def test_training_items_not_in_recommendations(self, recommender, env):
        """Items the user rated in training should not appear in recs."""
        uid = env.eval_users[0]
        trained_items = set(
            env.train_ratings[env.train_ratings["userId"] == uid]["movieId"].tolist()
        )
        if not trained_items:
            pytest.skip("User has no training ratings")

        recs = recommender.recommend(uid, n=20)
        rec_ids = {int(iid) for iid in recs.ids()}
        overlap = rec_ids & trained_items
        assert len(overlap) == 0, (
            f"Training items leaked into recs: {overlap}"
        )

    def test_mark_sent_prevents_repeat_in_rerequests(self, recommender, env):
        """
        After mark_sent(), the sent items should not appear in the next
        recommend() call within the same round.
        """
        uid = env.eval_users[0]
        batch1 = recommender.recommend(uid, n=5)
        recommender.mark_sent(uid, batch1)

        batch2 = recommender.recommend(uid, n=5)
        recommender.mark_sent(uid, batch2)

        batch1_ids = {int(iid) for iid in batch1.ids()}
        batch2_ids = {int(iid) for iid in batch2.ids()}
        overlap = batch1_ids & batch2_ids
        assert len(overlap) == 0, (
            f"Re-request repeated {len(overlap)} items from the first batch: {overlap}"
        )

        # Clean up: advance round so this test doesn't pollute others
        recommender.advance_round()

    def test_advance_round_clears_round_seen(self, recommender, env):
        """After advance_round(), round_seen should be empty."""
        uid = env.eval_users[0]
        batch = recommender.recommend(uid, n=5)
        recommender.mark_sent(uid, batch)

        recommender.advance_round()

        # _round_seen should now be empty (or not contain this user)
        assert uid not in recommender._round_seen or len(recommender._round_seen[uid]) == 0

    def test_advance_round_moves_seen_into_all_seen(self, recommender, env):
        """Items from a round should be in _all_seen after advance_round()."""
        uid = env.eval_users[0]
        batch = recommender.recommend(uid, n=5)
        sent_ids = {int(iid) for iid in batch.ids()}
        recommender.mark_sent(uid, batch)
        recommender.advance_round()

        # All sent items should be in _all_seen now
        assert sent_ids.issubset(recommender._all_seen[uid]), (
            "Some sent items were not promoted to _all_seen after advance_round()"
        )


class TestFeedbackAndRetrain:
    def test_update_user_adds_to_feedback(self, recommender, env):
        uid = env.eval_users[0]
        # Build a known accepted ItemList from movies not in training (to avoid
        # all_seen conflicts) — test update_user directly, not via top_n.
        all_movies = set(env.movie_meta["movieId"].tolist())
        user_training = set(
            env.train_ratings[env.train_ratings["userId"] == uid]["movieId"].tolist()
        )
        # Items never seen by this user and not yet in all_seen
        candidates = list(all_movies - user_training - recommender._all_seen.get(uid, set()))
        if len(candidates) < 2:
            pytest.skip("Not enough unseen movies to test feedback")

        interactions = [(mid, 4.5) for mid in candidates[:2]]
        before = sum(len(v) for v in recommender._feedback.values())
        recommender.update_user(uid, interactions)
        after = sum(len(v) for v in recommender._feedback.values())
        assert after == before + 2

    def test_update_user_adds_items_to_all_seen(self, recommender, env):
        uid = env.eval_users[0]
        batch = recommender.recommend(uid, n=3)
        recommender.mark_sent(uid, batch)
        accepted = batch.top_n(2)
        accepted_ids = {int(iid) for iid in accepted.ids()}
        recommender.update_user(uid, accepted)

        assert accepted_ids.issubset(recommender._all_seen[uid])
        recommender.advance_round()


class TestExpandedTrainRatings:
    """
    `_build_expanded_train_ratings` returns the Environment's frame directly
    when no replication is needed, so it must never be mutated in place.
    """

    def test_does_not_mutate_environment_train_ratings(self, tiny_config, env):
        """Replication must not write back into the shared training frame."""
        before = env.train_ratings.copy(deep=True)
        base_uid = int(env.train_ratings["userId"].iloc[0])
        Recommender(tiny_config, env, user_base_map={-12345: base_uid})
        pd.testing.assert_frame_equal(env.train_ratings, before)

    def test_no_replication_does_not_duplicate_rows(self, tiny_config, env):
        """An empty base map must yield exactly the training ratings."""
        rec = Recommender(tiny_config, env, user_base_map={})
        assert len(rec._expanded_train_ratings) == len(env.train_ratings)

    def test_replication_clones_base_user_rows(self, tiny_config, env):
        """A replicated user gets a full copy of its base user's ratings."""
        base_uid = int(env.train_ratings["userId"].iloc[0])
        sim_uid = -12345
        rec = Recommender(tiny_config, env, user_base_map={sim_uid: base_uid})
        expanded = rec._expanded_train_ratings

        n_base = int((env.train_ratings["userId"] == base_uid).sum())
        n_sim = int((expanded["userId"] == sim_uid).sum())
        assert n_sim == n_base, (
            f"Replicated user {sim_uid} has {n_sim} rows, expected {n_base} "
            f"cloned from base user {base_uid}."
        )
        assert len(expanded) == len(env.train_ratings) + n_base
