"""
tests/test_residual_factors.py - the rating-unit associative baseline (issue #27).

The residual SVD is only a fix if it is fitted on training data alone, is
reconstructed in rating units without double-counting the singular values,
scores exactly the pairs the null scores, and leaves the persona preference
space the simulation runs on untouched. Those four properties are pinned here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.bias_only_null import score_bias_only
from experiments.llm_vs_associative import score_associative_residual
from sim.population import build_user_assignments
from sim.residual_factors import (
    fit_env_residual_als,
    fit_env_residual_factors,
    fit_residual_als,
    fit_residual_factors,
    training_residuals,
)
from sim.user_agent import SimulatedUser


@pytest.fixture(scope="module")
def assignments(tiny_config, env):
    rng = np.random.default_rng(tiny_config.random_seed)
    return build_user_assignments(tiny_config, env, rng)


@pytest.fixture(scope="module")
def factors(env):
    return fit_env_residual_factors(env)


def _pairs(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return set(zip(frame["userId"].astype(int), frame["movieId"].astype(int)))


class TestReconstructionScale:

    def test_recovers_an_exact_low_rank_matrix_in_its_own_units(self):
        """U from fit_transform already carries Sigma, so U @ V must be the data.

        A doubly-scaled reconstruction would come back multiplied by the
        singular values, and a normalised one would be a cosine; both fail.
        """
        rng = np.random.default_rng(0)
        users, items = rng.normal(size=(12, 2)) * 1.5, rng.normal(size=(9, 2))
        truth = users @ items.T
        uu, ii = np.meshgrid(np.arange(12), np.arange(9), indexing="ij")
        frame = pd.DataFrame({
            "userId": uu.ravel() + 1,
            "movieId": ii.ravel() + 100,
            "residual": truth.ravel(),
        })
        fitted = fit_residual_factors(frame, n_components=2, random_state=0)
        recon = np.array([[fitted.residual(u + 1, i + 100) for i in range(9)] for u in range(12)])
        np.testing.assert_allclose(recon, truth, atol=1e-8)
        # The item side is unit-norm right singular vectors, not rescaled.
        np.testing.assert_allclose(np.linalg.norm(fitted.item_factors, axis=0), 1.0)

    def test_unseen_user_or_item_contributes_nothing(self, factors, env):
        some_item = int(factors.item_ids[0])
        some_user = int(factors.user_ids[0])
        assert factors.residual(-1, some_item) == 0.0
        assert factors.residual(some_user, -1) == 0.0

    def test_dimension_defaults_to_the_persona_capacity(self, tiny_config, factors):
        assert factors.n_components == tiny_config.user_pref_features


    def test_vectorised_lookup_matches_the_scalar_one(self, factors, env):
        users = np.array(list(env.train_ratings["userId"][:50]) + [-1])
        items = np.array(list(env.train_ratings["movieId"][:50]) + [int(factors.item_ids[0])])
        expected = [factors.residual(u, m) for u, m in zip(users, items)]
        np.testing.assert_allclose(factors.residuals(users, items), expected)
        assert not factors.covered(users, items)[-1]


class TestComponentGuards:

    @pytest.fixture
    def small(self):
        rng = np.random.default_rng(1)
        uu, ii = np.meshgrid(np.arange(6), np.arange(5), indexing="ij")
        return pd.DataFrame({
            "userId": uu.ravel(), "movieId": ii.ravel(), "residual": rng.normal(size=30),
        })

    @pytest.mark.parametrize("fit", [fit_residual_factors, fit_residual_als])
    def test_zero_components_is_a_clear_error(self, small, fit):
        with pytest.raises(ValueError, match="at least 1"):
            fit(small, n_components=0, random_state=0)

    def test_svd_refuses_to_fit_fewer_dimensions_than_asked(self, small):
        """ARPACK caps k at min(shape) - 1; asking for more must not silently shrink."""
        with pytest.raises(ValueError, match="too large"):
            fit_residual_factors(small, n_components=5, random_state=0)
        assert fit_residual_factors(small, n_components=4, random_state=0).n_components == 4

    def test_svd_is_exact_and_seed_independent(self, small):
        a = fit_residual_factors(small, n_components=3, random_state=0)
        b = fit_residual_factors(small, n_components=3, random_state=123)
        recon_a = a.user_factors @ a.item_factors.T
        np.testing.assert_allclose(recon_a, b.user_factors @ b.item_factors.T, atol=1e-10)
        dense = small.pivot(index="userId", columns="movieId", values="residual").to_numpy()
        assert a.singular_values is not None
        np.testing.assert_allclose(np.sort(a.singular_values)[::-1], np.linalg.svd(dense)[1][:3])


class TestResidualAls:

    def test_matches_a_row_by_row_reference(self):
        """The vectorised half-steps must solve the same normal equations as a loop."""
        rng = np.random.default_rng(3)
        n_users, n_items, k, lam = 15, 12, 3, 0.7
        mask = rng.random((n_users, n_items)) < 0.4
        mask[:, 0] = True  # every user observed at least once
        users, items = np.nonzero(mask)
        values = rng.normal(size=len(users))
        frame = pd.DataFrame({"userId": users, "movieId": items, "residual": values})
        fitted = fit_residual_als(frame, k, random_state=5, regularization=lam, iterations=4)

        init = np.random.default_rng(5)
        p = init.normal(scale=0.1, size=(n_users, k))
        q = init.normal(scale=0.1, size=(n_items, k))
        dense = np.zeros((n_users, n_items))
        dense[users, items] = values
        for _ in range(4):
            for u in range(n_users):
                cols = np.nonzero(mask[u])[0]
                x = q[cols]
                p[u] = np.linalg.solve(x.T @ x + lam * np.eye(k), x.T @ dense[u, cols])
            for i in range(n_items):
                rows = np.nonzero(mask[:, i])[0]
                x = p[rows]
                q[i] = np.linalg.solve(x.T @ x + lam * np.eye(k), x.T @ dense[rows, i])
        np.testing.assert_allclose(fitted.user_factors, p, atol=1e-10)
        np.testing.assert_allclose(fitted.item_factors, q, atol=1e-10)

    def test_recovers_a_fully_observed_low_rank_matrix(self):
        rng = np.random.default_rng(0)
        truth = rng.normal(size=(10, 2)) @ rng.normal(size=(2, 8))
        uu, ii = np.meshgrid(np.arange(10), np.arange(8), indexing="ij")
        frame = pd.DataFrame({"userId": uu.ravel(), "movieId": ii.ravel(), "residual": truth.ravel()})
        fitted = fit_residual_als(frame, 2, random_state=0, regularization=1e-9, iterations=200)
        np.testing.assert_allclose(fitted.user_factors @ fitted.item_factors.T, truth, atol=1e-5)

    def test_env_als_is_fitted_on_training_cells_only(self, env, tiny_config):
        fitted = fit_env_residual_als(env)
        train = env.train_ratings
        np.testing.assert_array_equal(fitted.user_ids, np.sort(train["userId"].unique()))
        np.testing.assert_array_equal(fitted.item_ids, np.sort(train["movieId"].unique()))
        assert fitted.n_components == tiny_config.user_pref_features
        assert fitted.singular_values is None


class TestFittedOnTrainingOnly:

    def test_residuals_cover_exactly_the_training_ratings(self, env):
        frame = training_residuals(env)
        assert len(frame) == len(env.train_ratings)
        assert _pairs(frame) == _pairs(env.train_ratings)
        assert not _pairs(frame) & _pairs(env.held_out)
        assert not _pairs(frame) & _pairs(env.validation)

    def test_residuals_are_the_environment_debiased_ratings(self, env):
        frame = training_residuals(env)
        expected = [
            env.debias_rating(u, m, r)
            for u, m, r in zip(env.train_ratings["userId"], env.train_ratings["movieId"], env.train_ratings["rating"])
        ]
        np.testing.assert_allclose(frame["residual"], expected, atol=1e-12)

    def test_eval_users_are_factorised_from_training_rows_only(self, env, factors):
        """Held-out ratings must not reach an eval user's factor row."""
        held = env.held_out.copy()
        frame = training_residuals(env)
        refit = fit_residual_factors(frame, factors.n_components, env.config.random_seed)
        np.testing.assert_allclose(refit.user_factors, factors.user_factors)
        # Control: had held-out residuals entered the matrix, the fit would move.
        leaky = pd.concat([
            frame,
            held.assign(residual=held["rating"] - 3.0)[["userId", "movieId", "residual"]],
        ], ignore_index=True)
        assert not np.allclose(
            fit_residual_factors(leaky, factors.n_components, env.config.random_seed).user_factors[:3],
            factors.user_factors[:3],
        )
        for uid in env.eval_users:
            assert factors.has_user(uid)
            assert (env.train_ratings["userId"] == uid).sum() == (frame["userId"] == uid).sum()


class TestPersonaSpaceUntouched:

    def test_fitting_does_not_change_persona_pref_vectors(self, tiny_config, env, assignments):
        def pref_vectors():
            rng = np.random.default_rng(tiny_config.random_seed)
            users, _ = SimulatedUser.build_population(tiny_config, env, rng, assignments=assignments)
            return {uid: u.persona.pref_vector.copy() for uid, u in users.items()}

        before_vectors = pref_vectors()
        before_factors = {u: v.copy() for u, v in env._user_pref_factors.items()}
        before_collection = env.user_pref_collection_name
        fit_env_residual_factors(env, n_components=3)
        fit_env_residual_als(env, n_components=3)
        after_vectors = pref_vectors()

        assert env.user_pref_collection_name == before_collection
        assert env._user_pref_factors.keys() == before_factors.keys()
        for uid, vec in before_factors.items():
            np.testing.assert_array_equal(env._user_pref_factors[uid], vec)
        assert before_vectors.keys() == after_vectors.keys()
        for uid, vec in before_vectors.items():
            np.testing.assert_array_equal(after_vectors[uid], vec)
            assert np.linalg.norm(vec) == pytest.approx(1.0)


class TestScoreAssociativeResidual:

    @pytest.mark.parametrize("selection", ["first", "random"])
    def test_scores_the_same_pairs_as_the_null(self, tiny_config, env, assignments, factors, selection):
        null = score_bias_only(
            env, assignments, 2, item_selection=selection,
            seed=tiny_config.random_seed, split=tiny_config.recommender_eval_split,
        )
        _, actual, pairs = score_associative_residual(
            env, assignments, factors, max_items_per_user=2,
            item_selection=selection, seed=tiny_config.random_seed,
        )
        assert pairs == list(zip(null["userId"], null["movieId"]))
        np.testing.assert_allclose(actual, null["rating"])
        assert len(pairs) > 0

    def test_predicts_clipped_bias_plus_residual(self, env, assignments, factors):
        predicted, _, pairs = score_associative_residual(env, assignments, factors)
        expected = [
            float(np.clip(env.get_rating_bias(u, m) + factors.residual(u, m), 1.0, 5.0))
            for u, m in pairs
        ]
        np.testing.assert_allclose(predicted, expected)
        assert all(1.0 <= p <= 5.0 for p in predicted)
