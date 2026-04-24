"""
tests/test_agents.py — tests for the AssociativeAgent.

Covers: scoring output shape, score range, and re-ranking order.
"""
from __future__ import annotations

import numpy as np
import pytest

from lenskit.data import ItemList
from sim.agents.associative import AssociativeAgent


@pytest.fixture(scope="module")
def assoc_agent(env):
    return AssociativeAgent(env)


class TestAssociativeAgent:
    def test_evaluate_returns_item_list(self, assoc_agent, recommender, env, population):
        uid = env.eval_users[0]
        candidates = recommender.recommend(uid, n=10)
        if len(candidates) == 0:
            pytest.skip("No candidates returned for this user")
        persona = population[uid]
        candidate_ids = [int(iid) for iid in candidates.ids()]
        item_factors = env.get_user_pref_item_factors(candidate_ids)
        result = assoc_agent.evaluate(candidates, persona, item_factors)
        assert isinstance(result, ItemList)
        recommender.advance_round()

    def test_evaluate_preserves_item_count(self, assoc_agent, recommender, env, population):
        uid = env.eval_users[0]
        candidates = recommender.recommend(uid, n=10)
        if len(candidates) == 0:
            pytest.skip("No candidates returned for this user")
        persona = population[uid]
        candidate_ids = [int(iid) for iid in candidates.ids()]
        item_factors = env.get_user_pref_item_factors(candidate_ids)
        result = assoc_agent.evaluate(candidates, persona, item_factors)
        assert len(result) == len(candidates)
        recommender.advance_round()

    def test_evaluate_returns_scores(self, assoc_agent, recommender, env, population):
        uid = env.eval_users[0]
        candidates = recommender.recommend(uid, n=10)
        if len(candidates) == 0:
            pytest.skip("No candidates returned for this user")
        persona = population[uid]
        candidate_ids = [int(iid) for iid in candidates.ids()]
        item_factors = env.get_user_pref_item_factors(candidate_ids)
        result = assoc_agent.evaluate(candidates, persona, item_factors)
        scores = result.scores()
        assert scores is not None, "Agent should attach scores to the ItemList"
        assert len(scores) == len(result)
        recommender.advance_round()

    def test_scores_are_in_dot_product_range(self, assoc_agent, recommender, env, population):
        """L2-normalised dot products must be in [-1.0, 1.0]."""
        uid = env.eval_users[0]
        candidates = recommender.recommend(uid, n=10)
        if len(candidates) == 0:
            pytest.skip("No candidates returned for this user")
        persona = population[uid]
        candidate_ids = [int(iid) for iid in candidates.ids()]
        item_factors = env.get_user_pref_item_factors(candidate_ids)
        result = assoc_agent.evaluate(candidates, persona, item_factors)
        scores = result.scores()
        assert scores is not None
        assert np.all(scores >= -1.0 - 1e-6)
        assert np.all(scores <= 1.0 + 1e-6)
        recommender.advance_round()

    def test_update_is_noop(self, assoc_agent, env):
        """AssociativeAgent.update() should not raise (it is a no-op)."""
        uid = env.eval_users[0]
        assoc_agent.update(uid, [])  # empty interactions list

    def test_unknown_user_gets_zero_scores_for_missing_items(self, assoc_agent, env, population):
        """Items absent from item_factors dict receive score 0.0."""
        uid = env.eval_users[0]
        persona = population[uid]
        movie_ids = env.movie_meta["movieId"].head(5).tolist()
        candidates = ItemList(item_ids=np.array(movie_ids, dtype=np.int64))
        # Pass empty item_factors so every item is missing → all zeros
        result = assoc_agent.evaluate(candidates, persona, item_factors={})
        scores = result.scores()
        assert scores is not None
        assert np.all(scores == 0.0)

