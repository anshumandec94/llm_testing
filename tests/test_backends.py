"""
tests/test_backends.py - the PreferenceBackend adapters (issue #13).

Every backend must return a debiased-rating residual aligned with its input,
nan for what it cannot score, and must reproduce the reference scoring it
wraps pair for pair. Synthetic fixtures only.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from lenskit.data import ItemList

from experiments.backends import (
    BACKEND_REGISTRY,
    AssociativeBackend,
    BiasOnlyBackend,
    ItemItemBackend,
    LLMBackend,
    PreferenceBackend,
    ResidualProfileBackend,
    SASRecBackend,
)
from experiments.bias_only_null import score_bias_only
from experiments.llm_vs_associative import score_associative_residual
from sim.agents.base import AbstractAgent
from sim.population import build_user_assignments
from sim.residual_factors import fit_env_residual_als, fit_env_residual_factors

UNSEEN_ITEM = 10_000
UNSEEN_USER = 10_000


@pytest.fixture(scope="module")
def assignments(tiny_config, env):
    rng = np.random.default_rng(tiny_config.random_seed)
    return build_user_assignments(tiny_config, env, rng)


@pytest.fixture(scope="module")
def null_frame(tiny_config, env, assignments):
    return score_bias_only(env, assignments, None, split=tiny_config.recommender_eval_split)


def _by_user(frame):
    return frame.groupby("userId", sort=False)["movieId"].apply(lambda s: [int(m) for m in s])


def _bias(env, frame):
    return np.array([env.get_rating_bias(u, m) for u, m in zip(frame["userId"], frame["movieId"])])


def test_registry_names_all_six_arms():
    assert set(BACKEND_REGISTRY) == {
        "bias_only", "associative", "residual_profile", "item_item", "llm", "sasrec",
    }


class TestBiasOnly:

    def test_is_a_backend_returning_exact_zeros(self):
        backend = BiasOnlyBackend()
        assert isinstance(backend, PreferenceBackend)
        out = backend.predict(1, [1, 2, UNSEEN_ITEM])
        assert out.shape == (3,)
        assert np.array_equal(out, np.zeros(3))

    def test_reconstructs_the_bias_only_null_pair_for_pair(self, env, null_frame):
        backend = BiasOnlyBackend()
        residual = np.concatenate([
            backend.predict(int(u), mids) for u, mids in _by_user(null_frame).items()
        ])
        np.testing.assert_array_equal(_bias(env, null_frame) + residual, null_frame["pred_null"].to_numpy())


class TestAssociative:

    @pytest.fixture(scope="class", params=["als", "svd"])
    def factors(self, request, env):
        fit = fit_env_residual_als if request.param == "als" else fit_env_residual_factors
        return fit(env)

    def test_equals_the_committed_residual_scoring(self, tiny_config, env, assignments, factors):
        """Acceptance: the PR #30 arm is clip(bias + this), on the same pairs."""
        predicted, _, pairs = score_associative_residual(env, assignments, factors)
        backend = AssociativeBackend(factors)
        assert isinstance(backend, PreferenceBackend)
        residual = np.concatenate([
            backend.predict(u, [m for uu, m in pairs if uu == u])
            for u in dict.fromkeys(u for u, _ in pairs)
        ])
        assert not np.isnan(residual).any()
        bias = np.array([env.get_rating_bias(u, m) for u, m in pairs])
        np.testing.assert_allclose(np.clip(bias + residual, 1.0, 5.0), predicted, rtol=0, atol=1e-12)
        # Where the clip does not bite, the backend is the arm minus the bias.
        inside = (bias + residual >= 1.0) & (bias + residual <= 5.0)
        assert inside.any()
        np.testing.assert_allclose(residual[inside], (np.array(predicted) - bias)[inside], atol=1e-12)

    def test_cold_user_and_item_are_nan(self, factors):
        backend = AssociativeBackend(factors)
        user, item = int(factors.user_ids[0]), int(factors.item_ids[0])
        out = backend.predict(user, [item, UNSEEN_ITEM])
        assert np.isfinite(out[0]) and np.isnan(out[1])
        assert np.isnan(backend.predict(UNSEEN_USER, [item])).all()

    def test_from_env_uses_the_chosen_als_fit(self, env):
        backend = AssociativeBackend.from_env(env)
        assert backend.factors.singular_values is None  # ALS, not the SVD
        reference = fit_env_residual_als(env)
        np.testing.assert_array_equal(backend.factors.user_factors, reference.user_factors)


class _FixedRatingAgent(AbstractAgent):
    """Stands in for LLMAgent: a known [1, 5] rating per item, nan if listed."""

    def __init__(self, ratings: dict[int, float]) -> None:
        self.ratings = ratings
        self.seen_users: list[int] = []

    def evaluate(self, candidates, persona, item_factors):
        self.seen_users.append(persona.user_id)
        ids = [int(i) for i in candidates.ids()]
        return ItemList(candidates, scores=np.array([self.ratings[i] for i in ids], dtype=np.float32))

    def update(self, user_id, interactions):
        pass


class TestLLM:

    def test_subtracts_the_bias_from_the_rating(self, env):
        user = int(env.eval_users[0])
        agent = _FixedRatingAgent({1: 4.5, 2: 1.0, 3: 3.0})
        backend = LLMBackend(env, agent)
        assert isinstance(backend, PreferenceBackend)
        out = backend.predict(user, [1, 2, 3])
        expected = np.array([4.5, 1.0, 3.0]) - [env.get_rating_bias(user, m) for m in (1, 2, 3)]
        np.testing.assert_allclose(out, expected, atol=1e-6)
        assert agent.seen_users == [user]

    def test_unknown_item_and_non_finite_score_are_nan(self, env):
        agent = _FixedRatingAgent({1: 4.0, 2: float("nan")})
        backend = LLMBackend(env, agent)
        out = backend.predict(int(env.eval_users[0]), [1, UNSEEN_ITEM, 2])
        assert np.isfinite(out[0]) and np.isnan(out[1]) and np.isnan(out[2])
        assert np.isnan(backend.predict(1, [UNSEEN_ITEM])).all()

    def test_cold_user_is_still_scored(self, env):
        backend = LLMBackend(env, _FixedRatingAgent({1: 4.0}))
        out = backend.predict(UNSEEN_USER, [1])
        np.testing.assert_allclose(out, [4.0 - env.get_rating_bias(UNSEEN_USER, 1)], atol=1e-6)

    def test_wraps_the_real_agent_through_evaluate(self, env):
        """A real LLMAgent, with only model loading and generation mocked."""
        from sim.agents.llm import LLMAgent

        with patch("mlx_lm.load", return_value=(object(), None)):
            agent = LLMAgent(env, model_id="mock-model", use_few_shot=False)
        user = int(env.eval_users[0])
        with patch.object(LLMAgent, "_build_prompt", return_value="p"), \
                patch.object(LLMAgent, "_call_llm", side_effect=[2.5, 4.0]):
            out = LLMBackend(env, agent).predict(user, [5, 6])
        expected = np.array([2.5, 4.0]) - [env.get_rating_bias(user, m) for m in (5, 6)]
        np.testing.assert_allclose(out, expected, atol=1e-6)


@pytest.mark.parametrize("cls", [ResidualProfileBackend, ItemItemBackend, SASRecBackend])
def test_unfilled_arms_refuse_to_construct(cls, env):
    with pytest.raises(NotImplementedError):
        cls(env)
