"""
tests/test_persona.py — tests for AgentPersona, build_persona, build_population.

Covers: persona construction, act() sampling, update_preference().
"""
from __future__ import annotations

import numpy as np
import pytest

from sim.archetypes import ARCHETYPE_REGISTRY
from sim.persona import AgentPersona, build_persona, build_population


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


class TestBuildPersona:
    def test_returns_agent_persona(self, env, tiny_config, rng):
        archetype_cfg = ARCHETYPE_REGISTRY["casual"]
        persona = build_persona(env.eval_users[0], archetype_cfg, env, rng)
        assert isinstance(persona, AgentPersona)

    def test_user_id_stored(self, env, tiny_config, rng):
        uid = env.eval_users[0]
        archetype_cfg = ARCHETYPE_REGISTRY["casual"]
        persona = build_persona(uid, archetype_cfg, env, rng)
        assert persona.user_id == uid

    def test_pref_vector_correct_dim(self, env, tiny_config, rng):
        archetype_cfg = ARCHETYPE_REGISTRY["casual"]
        persona = build_persona(env.eval_users[0], archetype_cfg, env, rng)
        assert persona.pref_vector.shape == (tiny_config.user_pref_features,)

    def test_pref_vector_unit_norm(self, env, tiny_config, rng):
        archetype_cfg = ARCHETYPE_REGISTRY["casual"]
        persona = build_persona(env.eval_users[0], archetype_cfg, env, rng)
        norm = np.linalg.norm(persona.pref_vector)
        assert abs(norm - 1.0) < 1e-5, f"pref_vector norm is {norm:.4f}, expected 1.0"

    def test_budget_starts_at_one(self, env, rng):
        archetype_cfg = ARCHETYPE_REGISTRY["casual"]
        persona = build_persona(env.eval_users[0], archetype_cfg, env, rng)
        assert persona.budget == pytest.approx(1.0)

    def test_tau_is_positive(self, env, rng):
        archetype_cfg = ARCHETYPE_REGISTRY["casual"]
        persona = build_persona(env.eval_users[0], archetype_cfg, env, rng)
        assert persona.tau > 0


class TestBuildPopulation:
    def test_population_covers_all_eval_users(self, env, tiny_config):
        rng = np.random.default_rng(0)
        pop = build_population(tiny_config, env, rng)
        for uid in env.eval_users:
            assert uid in pop, f"eval user {uid} missing from population"

    def test_population_all_agent_personas(self, env, tiny_config):
        rng = np.random.default_rng(0)
        pop = build_population(tiny_config, env, rng)
        for uid, persona in pop.items():
            assert isinstance(persona, AgentPersona)


class TestAct:
    def _make_candidates(self, env, n=10):
        movie_ids = env.movie_meta["movieId"].head(n).tolist()
        return movie_ids

    def test_act_returns_list(self, sample_persona, env, tiny_config):
        rng = np.random.default_rng(7)
        movie_ids = self._make_candidates(env)
        item_factors = env.get_user_pref_item_factors(movie_ids)
        scores = np.ones(len(movie_ids), dtype=np.float32) * 0.5
        result = sample_persona.act(movie_ids, scores, item_factors, tiny_config, rng)
        assert isinstance(result, list)

    def test_act_tuples_have_correct_structure(self, sample_persona, env, tiny_config):
        rng = np.random.default_rng(7)
        movie_ids = self._make_candidates(env)
        item_factors = env.get_user_pref_item_factors(movie_ids)
        scores = np.ones(len(movie_ids), dtype=np.float32) * 0.5
        result = sample_persona.act(movie_ids, scores, item_factors, tiny_config, rng)
        for item in result:
            assert len(item) == 3
            movie_id, action, signal = item
            assert isinstance(movie_id, int)
            assert isinstance(action, str)
            assert isinstance(signal, float)

    def test_act_actions_are_valid(self, sample_persona, env, tiny_config):
        rng = np.random.default_rng(7)
        movie_ids = self._make_candidates(env)
        item_factors = env.get_user_pref_item_factors(movie_ids)
        scores = np.ones(len(movie_ids), dtype=np.float32) * 0.5
        result = sample_persona.act(movie_ids, scores, item_factors, tiny_config, rng)
        valid_actions = {"watch", "rate", "add_to_list"}
        for _, action, _ in result:
            assert action in valid_actions, f"Unexpected action: {action!r}"

    def test_act_signals_in_range(self, sample_persona, env, tiny_config):
        rng = np.random.default_rng(7)
        movie_ids = self._make_candidates(env)
        item_factors = env.get_user_pref_item_factors(movie_ids)
        scores = np.ones(len(movie_ids), dtype=np.float32) * 0.5
        result = sample_persona.act(movie_ids, scores, item_factors, tiny_config, rng)
        for _, _, signal in result:
            assert 0.0 <= signal <= 5.0 + 1e-6, f"Signal {signal} out of [0, 5]"

    def test_zero_budget_returns_empty(self, env, tiny_config, rng):
        archetype_cfg = ARCHETYPE_REGISTRY["casual"]
        persona = build_persona(env.eval_users[0], archetype_cfg, env, rng)
        persona.budget = 0.0
        movie_ids = self._make_candidates(env)
        item_factors = env.get_user_pref_item_factors(movie_ids)
        scores = np.ones(len(movie_ids), dtype=np.float32) * 0.5
        result = persona.act(movie_ids, scores, item_factors, tiny_config, rng)
        assert result == []

    def test_score_floor_filters_all(self, env, tiny_config, rng):
        archetype_cfg = ARCHETYPE_REGISTRY["casual"]
        persona = build_persona(env.eval_users[0], archetype_cfg, env, rng)
        persona.score_floor = 1.0  # nothing passes; scores are ≤ 1.0 but never == 1.0
        movie_ids = self._make_candidates(env)
        item_factors = env.get_user_pref_item_factors(movie_ids)
        scores = np.zeros(len(movie_ids), dtype=np.float32)  # all below 1.0
        result = persona.act(movie_ids, scores, item_factors, tiny_config, rng)
        assert result == []


class TestUpdatePreference:
    def test_update_changes_pref_vector(self, env, tiny_config, rng):
        archetype_cfg = ARCHETYPE_REGISTRY["casual"]
        persona = build_persona(env.eval_users[0], archetype_cfg, env, rng)
        original = persona.pref_vector.copy()
        movie_ids = env.movie_meta["movieId"].head(5).tolist()
        item_factors = env.get_user_pref_item_factors(movie_ids)
        if not item_factors:
            pytest.skip("No item factors available for pref-update test")
        interactions = [(mid, "watch", 4.5) for mid in list(item_factors.keys())[:2]]
        persona.update_preference(interactions, item_factors)
        assert not np.allclose(persona.pref_vector, original), (
            "pref_vector should have changed after update"
        )

    def test_update_preserves_unit_norm(self, env, tiny_config, rng):
        archetype_cfg = ARCHETYPE_REGISTRY["casual"]
        persona = build_persona(env.eval_users[0], archetype_cfg, env, rng)
        movie_ids = env.movie_meta["movieId"].head(10).tolist()
        item_factors = env.get_user_pref_item_factors(movie_ids)
        if not item_factors:
            pytest.skip("No item factors available for pref-update test")
        interactions = [(mid, "watch", 4.5) for mid in list(item_factors.keys())[:3]]
        persona.update_preference(interactions, item_factors)
        norm = np.linalg.norm(persona.pref_vector)
        assert abs(norm - 1.0) < 1e-5, f"pref_vector norm after update is {norm:.4f}"

    def test_empty_interactions_no_change(self, env, rng):
        archetype_cfg = ARCHETYPE_REGISTRY["casual"]
        persona = build_persona(env.eval_users[0], archetype_cfg, env, rng)
        original = persona.pref_vector.copy()
        persona.update_preference([], {})
        assert np.allclose(persona.pref_vector, original)
