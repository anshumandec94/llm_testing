"""
tests/test_agents.py — tests for the AssociativeAgent.

Covers: scoring output shape, score range, and re-ranking order.
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest

from lenskit.data import ItemList
from sim.agents import AGENT_REGISTRY, build_agent
from sim.agents.associative import AssociativeAgent
from sim.agents.item_item import ItemItemNeighborhoodAgent
from sim.agents.residual_profile import ResidualProfileAgent
from sim.config import SimConfig


@pytest.fixture(scope="module")
def assoc_agent(env):
    return AssociativeAgent(env)


@pytest.fixture
def history_env():
    class DummyEnv:
        def __init__(self):
            self.config = type("Cfg", (), {"user_pref_features": 2})()
            self.eval_users = [1]
            self.train_ratings = pd.DataFrame(
                [
                    {"userId": 1, "movieId": 1, "rating": 5.0},
                    {"userId": 1, "movieId": 2, "rating": 1.0},
                ]
            )
            self._item_vectors = {
                1: np.array([1.0, 0.0], dtype=np.float64),
                2: np.array([-1.0, 0.0], dtype=np.float64),
                3: np.array([1.0, 0.0], dtype=np.float64),
                4: np.array([-1.0, 0.0], dtype=np.float64),
            }

        def get_user_pref_item_factors(self, movie_ids):
            return {
                int(mid): self._item_vectors[int(mid)]
                for mid in movie_ids
                if int(mid) in self._item_vectors
            }

        def debias_rating(self, user_id, movie_id, rating):
            _ = user_id, movie_id
            return float(rating - 3.0)

    return DummyEnv()


@pytest.fixture
def simple_persona(sample_persona):
    persona = copy.deepcopy(sample_persona)
    persona.user_id = 1
    return persona


class TestAssociativeAgent:
    def test_registry_exposes_associative_baseline_alias(self, env):
        agent = build_agent(SimConfig(agent_type="associative_baseline"), env)
        assert isinstance(agent, AssociativeAgent)
        assert "associative_baseline" in AGENT_REGISTRY

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


class TestTraditionalAgents:
    def test_registry_builds_traditional_agents(self, history_env):
        residual = build_agent(SimConfig(agent_type="residual_profile"), history_env)
        item_item = build_agent(SimConfig(agent_type="item_item"), history_env)
        assert isinstance(residual, ResidualProfileAgent)
        assert isinstance(item_item, ItemItemNeighborhoodAgent)
        assert "residual_profile" in AGENT_REGISTRY
        assert "item_item" in AGENT_REGISTRY

    def test_residual_profile_prefers_liked_and_penalizes_disliked(
        self, history_env, simple_persona
    ):
        agent = ResidualProfileAgent(history_env)
        candidates = ItemList(item_ids=np.array([3, 4], dtype=np.int64))
        item_factors = history_env.get_user_pref_item_factors([3, 4])
        result = agent.evaluate(candidates, simple_persona, item_factors)
        scores = result.scores()
        assert scores is not None
        assert scores[0] > 0
        assert scores[1] < 0

    def test_item_item_prefers_neighbors_of_liked_items(
        self, history_env, simple_persona
    ):
        agent = ItemItemNeighborhoodAgent(history_env)
        candidates = ItemList(item_ids=np.array([3, 4], dtype=np.int64))
        item_factors = history_env.get_user_pref_item_factors([3, 4])
        result = agent.evaluate(candidates, simple_persona, item_factors)
        scores = result.scores()
        assert scores is not None
        assert scores[0] > scores[1]

    def test_online_negative_update_reduces_residual_profile_score(
        self, history_env, simple_persona
    ):
        agent = ResidualProfileAgent(history_env)
        candidates = ItemList(item_ids=np.array([3], dtype=np.int64))
        item_factors = history_env.get_user_pref_item_factors([3])
        before = agent.evaluate(candidates, simple_persona, item_factors).scores()
        assert before is not None
        agent.update(1, [(3, "rate", -1.0)])
        after = agent.evaluate(candidates, simple_persona, item_factors).scores()
        assert after is not None
        assert after[0] < before[0]
