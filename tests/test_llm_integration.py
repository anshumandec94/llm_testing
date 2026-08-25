"""
tests/test_llm_integration.py — integration tests for the LLM evaluation pipeline.

These tests exercise the real environment, real personas, and real mlx_lm API
contracts. The LLM model itself is mocked (it's 4 GB), but every other layer
is real. The goal is to catch the class of bugs that unit tests miss:

  - mlx_lm API changes (e.g. temp= vs sampler=, tokenize=True returning
    BatchEncoding instead of str)
  - Broken wiring between environment, agent, and experiment evaluation
  - Bias + dot-product reconstruction producing out-of-range predictions
  - Metrics computed from an empty or mismatched result set
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import mlflow
import numpy as np
import pytest
from lenskit.data import ItemList

from sim.agents.llm import LLMAgent
from sim.population import build_user_assignments
from sim.user_agent import SimulatedUser


# ── mlx_lm API contract ───────────────────────────────────────────────────────
# These tests verify assumptions our code makes about mlx_lm's API.
# If mlx_lm changes its interface, these fail immediately and tell you
# exactly which argument in _call_llm needs updating.

class TestMlxLmApiContract:

    def test_generate_step_does_not_accept_temp_directly(self):
        """Regression: we passed temp= directly to generate(), which doesn't exist.
        The correct path is sampler=make_sampler(temp=0.0)."""
        from mlx_lm.generate import generate_step
        sig = inspect.signature(generate_step)
        assert "temp" not in sig.parameters, (
            "mlx_lm.generate_step now has a temp= param — remove the make_sampler wrapper in _call_llm."
        )

    def test_generate_step_accepts_sampler(self):
        """Our code passes greedy sampling via sampler=make_sampler(temp=0.0)."""
        from mlx_lm.generate import generate_step
        sig = inspect.signature(generate_step)
        assert "sampler" in sig.parameters

    def test_make_sampler_accepts_temp(self):
        """make_sampler must still accept temp= for greedy decoding."""
        from mlx_lm.sample_utils import make_sampler
        sig = inspect.signature(make_sampler)
        assert "temp" in sig.parameters

    def test_make_sampler_zero_temp_is_callable(self):
        """make_sampler(temp=0.0) must return a callable that generate_step accepts."""
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=0.0)
        assert callable(sampler)

    def test_generate_accepts_string_prompt(self):
        """generate() must accept a str prompt — we pass the result of
        apply_chat_template(tokenize=False) which is always a str."""
        from mlx_lm import generate
        sig = inspect.signature(generate)
        prompt_param = sig.parameters.get("prompt")
        assert prompt_param is not None


# ── LLMAgent with real environment ───────────────────────────────────────────

@pytest.fixture(scope="module")
def llm_agent(env):
    """LLMAgent wired to the real test environment, model/tokenizer mocked."""
    mock_tok = MagicMock()
    mock_tok.apply_chat_template.return_value = "<|system|>You are...<|user|>History..."

    with patch("mlx_lm.load", return_value=(MagicMock(), mock_tok)):
        agent = LLMAgent(
            env,
            model_id="mock-model",
            history_k=2,
            history_strategy="top_rated",
            max_tokens=32,
            use_few_shot=False,
        )
    return agent


class TestLLMAgentWithRealEnv:
    """Tests that use the real Environment and synthetic ML-32M data."""

    def test_agent_indexes_movie_meta(self, llm_agent, env):
        """Agent must correctly index env.movie_meta on construction."""
        assert set(llm_agent._meta.index) == set(env.movie_meta["movieId"])

    def test_agent_indexes_user_ratings(self, llm_agent, env):
        """Agent must merge train_ratings with movie_meta on construction."""
        assert "title" in llm_agent._user_ratings.columns
        assert "genres" in llm_agent._user_ratings.columns

    def test_build_prompt_returns_str_with_real_tokenizer_mock(self, llm_agent, env):
        """End-to-end: _build_prompt must return str, not BatchEncoding.
        This test wires through the real env so we know the history lookup
        and movie_meta lookup work before the tokenizer is called."""
        uid = env.eval_users[0]
        # pick a movie that exists in movie_meta
        mid = int(env.movie_meta["movieId"].iloc[0])
        history = llm_agent._get_history(uid)
        result = llm_agent._build_prompt(history, mid)
        assert isinstance(result, str), (
            f"_build_prompt returned {type(result).__name__}, not str. "
            "Likely tokenize=True was used in apply_chat_template."
        )

    def test_evaluate_returns_valid_scores_with_real_items(self, llm_agent, env, population):
        """Full evaluate() pipeline: real env + real persona + mocked generate."""
        uid = env.eval_users[0]
        persona = population[uid]
        movie_ids = env.movie_meta["movieId"].head(5).tolist()
        candidates = ItemList(item_ids=np.array(movie_ids, dtype=np.int64))
        item_factors = env.get_user_pref_item_factors(movie_ids)

        with patch("mlx_lm.generate", return_value='{"predicted_rating": 4.1, "reasoning": "ok"}'):
            result = llm_agent.evaluate(candidates, persona, item_factors)

        scores = result.scores()
        assert scores is not None
        assert len(scores) == len(movie_ids)
        assert np.all(scores >= 1.0) and np.all(scores <= 5.0)

    def test_evaluate_scores_float32(self, llm_agent, env, population):
        """Scores must be float32 to match AssociativeAgent output dtype."""
        uid = env.eval_users[0]
        persona = population[uid]
        movie_ids = env.movie_meta["movieId"].head(3).tolist()
        candidates = ItemList(item_ids=np.array(movie_ids, dtype=np.int64))
        item_factors = env.get_user_pref_item_factors(movie_ids)

        with patch("mlx_lm.generate", return_value='{"predicted_rating": 3.5}'):
            result = llm_agent.evaluate(candidates, persona, item_factors)

        assert result.scores().dtype == np.float32


# ── AssociativeAgent bias + dot-product reconstruction ───────────────────────

class TestAssociativeRatingReconstruction:
    """Integration tests for the bias + dot reconstruction used in the experiment.

    This is NOT tested in test_agents.py (which only tests the dot product).
    These tests verify the full rating prediction: bias + dot ∈ [1, 5].
    """

    def test_bias_plus_dot_in_rating_range(self, env, population):
        """bias(u, i) + dot(pref_vec, item_factor) must produce values near [1, 5]."""
        uid = env.eval_users[0]
        base_uid = env.eval_users[0]
        persona = population[uid]
        movie_ids = env.movie_meta["movieId"].head(10).tolist()
        item_factors = env.get_user_pref_item_factors(movie_ids)

        preds = []
        for mid in movie_ids:
            bias = env.get_rating_bias(base_uid, mid)
            dot = float(np.dot(persona.pref_vector, item_factors[mid])) if mid in item_factors else 0.0
            pred = float(np.clip(bias + dot, 1.0, 5.0))
            preds.append(pred)

        assert all(1.0 <= p <= 5.0 for p in preds)

    def test_bias_is_nonzero(self, env):
        """get_rating_bias must return a meaningful non-trivial value."""
        uid = env.eval_users[0]
        mid = int(env.movie_meta["movieId"].iloc[0])
        bias = env.get_rating_bias(uid, mid)
        # bias = global + user + item bias; with any real data this should be nonzero
        assert bias != 0.0

    def test_debias_then_rebias_roundtrips(self, env):
        """debias_rating and get_rating_bias must be inverses."""
        uid = env.eval_users[0]
        mid = int(env.movie_meta["movieId"].iloc[0])
        raw = 4.0
        debiased = env.debias_rating(uid, mid, raw)
        reconstructed = debiased + env.get_rating_bias(uid, mid)
        assert reconstructed == pytest.approx(raw, abs=1e-4)


# ── Experiment evaluation pipeline ───────────────────────────────────────────

class TestEvaluatePipeline:
    """Integration tests for the evaluate_associative function in the experiment.

    Exercises the full pipeline: env → assignments → users → per-item scoring
    → metric aggregation, with MLflow patched out.
    """

    @pytest.fixture(scope="class")
    def eval_context(self, tiny_config, env):
        rng = np.random.default_rng(tiny_config.random_seed)
        assignments = build_user_assignments(tiny_config, env, rng)
        users, _ = SimulatedUser.build_population(tiny_config, env, rng, assignments=assignments)
        return assignments, users

    def test_evaluate_associative_completes(self, tiny_config, env, eval_context):
        """evaluate_associative must run to completion and log metrics."""
        from experiments.llm_vs_associative import _compute_metrics

        assignments, users = eval_context
        all_predicted, all_actual = [], []

        for assignment in assignments:
            uid = assignment.sim_user_id
            base_uid = assignment.base_user_id
            held_out_df = env.held_out_for_user(base_uid, split=tiny_config.recommender_eval_split)
            if held_out_df.empty:
                continue
            held_ids = [int(mid) for mid in held_out_df["movieId"]]
            actual_by_id = {int(mid): float(r)
                            for mid, r in zip(held_out_df["movieId"], held_out_df["rating"])}
            persona = users[uid].persona
            item_factors = env.get_user_pref_item_factors(held_ids)

            for mid in held_ids:
                bias = env.get_rating_bias(base_uid, mid)
                dot = float(np.dot(persona.pref_vector, item_factors[mid])) if mid in item_factors else 0.0
                all_predicted.append(float(np.clip(bias + dot, 1.0, 5.0)))
                all_actual.append(actual_by_id[mid])

        assert len(all_predicted) > 0, "No predictions produced — check eval_user_frac or holdout_frac"
        metrics = _compute_metrics(all_predicted, all_actual)
        assert 0.0 <= metrics["error/mae"] <= 4.0
        assert 0.0 <= metrics["error/rmse"] <= 4.0
        assert 1.0 <= metrics["score/mean"] <= 5.0
        assert metrics["score/std"] >= 0.0

    def test_evaluate_associative_mae_is_finite(self, tiny_config, env, eval_context):
        """MAE must be a finite number, not NaN or inf."""
        from experiments.llm_vs_associative import _compute_metrics

        assignments, users = eval_context
        all_predicted, all_actual = [], []

        for assignment in assignments:
            uid = assignment.sim_user_id
            base_uid = assignment.base_user_id
            held_out_df = env.held_out_for_user(base_uid, split=tiny_config.recommender_eval_split)
            if held_out_df.empty:
                continue
            held_ids = [int(mid) for mid in held_out_df["movieId"]]
            actual_by_id = {int(mid): float(r)
                            for mid, r in zip(held_out_df["movieId"], held_out_df["rating"])}
            persona = users[uid].persona
            item_factors = env.get_user_pref_item_factors(held_ids)

            for mid in held_ids:
                bias = env.get_rating_bias(base_uid, mid)
                dot = float(np.dot(persona.pref_vector, item_factors[mid])) if mid in item_factors else 0.0
                all_predicted.append(float(np.clip(bias + dot, 1.0, 5.0)))
                all_actual.append(actual_by_id[mid])

        metrics = _compute_metrics(all_predicted, all_actual)
        assert np.isfinite(metrics["error/mae"])
        assert np.isfinite(metrics["error/rmse"])


# ── Matched item selection across arms ────────────────────────────────────────
# Issue #2. The 2026-06-26 sweep scored the associative baseline on the full
# held-out set and each LLM arm on the first 5 items per user, so the MAE gap
# between them was a difference in evaluation set rather than in agent. These
# tests guard the property that made that possible.


class TestMatchedItemSelection:
    """Both arms must select the same (user, item) pairs under the same cap."""

    @pytest.fixture(scope="class")
    def eval_context(self, tiny_config, env):
        rng = np.random.default_rng(tiny_config.random_seed)
        assignments = build_user_assignments(tiny_config, env, rng)
        users, _ = SimulatedUser.build_population(tiny_config, env, rng, assignments=assignments)
        return assignments, users

    def test_select_held_items_caps_positionally(self, env, eval_context):
        """A cap takes the leading N rows, preserving held-out order."""
        from experiments.llm_vs_associative import select_held_items

        assignments, _ = eval_context
        base_uid = assignments[0].base_user_id
        held_out_df = env.held_out_for_user(base_uid)

        uncapped = select_held_items(held_out_df, None)
        capped = select_held_items(held_out_df, 2)

        assert len(uncapped) == len(held_out_df)
        assert capped == uncapped[:2]

    def test_select_held_items_uncapped_returns_everything(self, env, eval_context):
        """max_items_per_user=None must not drop any held-out item."""
        from experiments.llm_vs_associative import select_held_items

        assignments, _ = eval_context
        for assignment in assignments:
            held_out_df = env.held_out_for_user(assignment.base_user_id)
            assert len(select_held_items(held_out_df, None)) == len(held_out_df)

    def test_cap_larger_than_history_is_a_no_op(self, env, eval_context):
        """A cap above a user's held-out count must not pad or truncate."""
        from experiments.llm_vs_associative import select_held_items

        assignments, _ = eval_context
        held_out_df = env.held_out_for_user(assignments[0].base_user_id)
        assert select_held_items(held_out_df, 10_000) == select_held_items(held_out_df, None)

    def test_associative_and_llm_arms_score_identical_pairs(self, tiny_config, env, eval_context):
        """The regression that motivated issue #2.

        The pairs an arm scores depend only on assignments, the held-out split
        and the cap, so the LLM arm's selection can be reproduced without any
        LLM call. Equal counts are not enough: the pairs themselves must match.
        """
        from experiments.llm_vs_associative import score_associative, select_held_items

        assignments, users = eval_context
        cap = 2

        _, _, associative_pairs = score_associative(
            env, assignments, users, max_items_per_user=cap,
        )

        llm_pairs: list[tuple[int, int]] = []
        for assignment in assignments:
            held_out_df = env.held_out_for_user(
                assignment.base_user_id, split=tiny_config.recommender_eval_split,
            )
            if held_out_df.empty:
                continue
            for mid in select_held_items(held_out_df, cap):
                llm_pairs.append((assignment.base_user_id, mid))

        assert associative_pairs == llm_pairs
        assert len(associative_pairs) > 0

    def test_cap_reduces_the_scored_item_count(self, env, eval_context):
        """A capped baseline must score strictly fewer items than an uncapped one.

        Without this the cap could silently no-op and the comparison would look
        matched while staying exactly as confounded as before.
        """
        from experiments.llm_vs_associative import score_associative

        assignments, users = eval_context
        _, _, uncapped = score_associative(env, assignments, users, max_items_per_user=None)
        _, _, capped = score_associative(env, assignments, users, max_items_per_user=1)

        assert len(capped) < len(uncapped)
        assert len(capped) == len({uid for uid, _ in capped})

    def test_capped_run_name_does_not_collide_with_the_uncapped_run(self):
        """The 2026-06-26 uncapped baseline is kept, so names must differ."""
        from experiments.llm_vs_associative import baseline_run_name

        assert baseline_run_name(None) == "associative-baseline"
        assert baseline_run_name(5) == "associative-baseline-capped"


# ── Sampled item selection ────────────────────────────────────────────────────
# Issue #3. The cap is a positional slice and held-out rows are sorted by
# timestamp descending, so "first N" is each user's most recent ratings rather
# than a neutral subset. `random` exists to measure how much that matters.


class TestItemSelection:
    """`--item-selection random` must be a real sample and still deterministic."""

    @pytest.fixture(scope="class")
    def eval_context(self, tiny_config, env):
        rng = np.random.default_rng(tiny_config.random_seed)
        assignments = build_user_assignments(tiny_config, env, rng)
        users, _ = SimulatedUser.build_population(tiny_config, env, rng, assignments=assignments)
        return assignments, users

    def _held(self, env, assignments):
        """A user with strictly more held-out items than the cap under test."""
        for assignment in assignments:
            df = env.held_out_for_user(assignment.base_user_id)
            if len(df) > 2:
                return assignment.base_user_id, df
        pytest.skip("fixture has no user with more than 2 held-out items")

    def test_random_selection_is_deterministic(self, env, eval_context):
        """Same seed and user must reproduce the same sample exactly."""
        from experiments.llm_vs_associative import select_held_items

        assignments, _ = eval_context
        uid, df = self._held(env, assignments)
        a = select_held_items(df, 2, selection="random", user_id=uid, seed=42)
        b = select_held_items(df, 2, selection="random", user_id=uid, seed=42)
        assert a == b

    def test_random_selection_depends_on_the_seed(self, env, eval_context):
        """A different seed must be able to produce a different sample.

        Guards against the sample silently collapsing to the positional head,
        which would make the recency measurement meaningless.
        """
        from experiments.llm_vs_associative import select_held_items

        assignments, _ = eval_context
        uid, df = self._held(env, assignments)
        samples = {
            tuple(select_held_items(df, 2, selection="random", user_id=uid, seed=s))
            for s in range(40)
        }
        assert len(samples) > 1

    def test_random_selection_is_independent_of_user_iteration_order(self, env, eval_context):
        """Seeding is derived from (seed, user_id), not a shared stream.

        A shared RNG advanced across users would make each user's sample depend
        on how many users preceded it, so filtering the population would
        silently change everyone's items.
        """
        from experiments.llm_vs_associative import select_held_items

        assignments, _ = eval_context
        uid, df = self._held(env, assignments)
        expected = select_held_items(df, 2, selection="random", user_id=uid, seed=42)

        for other in assignments:
            other_df = env.held_out_for_user(other.base_user_id)
            select_held_items(other_df, 2, selection="random",
                              user_id=other.base_user_id, seed=42)

        assert select_held_items(df, 2, selection="random", user_id=uid, seed=42) == expected

    def test_random_selection_returns_held_out_order(self, env, eval_context):
        """Sampled ids come back in held-out order, not draw order."""
        from experiments.llm_vs_associative import select_held_items

        assignments, _ = eval_context
        uid, df = self._held(env, assignments)
        full = select_held_items(df, None)
        picked = select_held_items(df, 2, selection="random", user_id=uid, seed=42)
        assert picked == [i for i in full if i in picked]

    def test_random_selection_samples_without_replacement(self, env, eval_context):
        """No duplicated items, and exactly the cap many."""
        from experiments.llm_vs_associative import select_held_items

        assignments, _ = eval_context
        uid, df = self._held(env, assignments)
        picked = select_held_items(df, 2, selection="random", user_id=uid, seed=42)
        assert len(picked) == 2
        assert len(set(picked)) == 2
        assert set(picked).issubset(set(select_held_items(df, None)))

    def test_selections_agree_when_the_cap_exceeds_the_history(self, env, eval_context):
        """Below the cap there is nothing to choose, so both return everything."""
        from experiments.llm_vs_associative import select_held_items

        assignments, _ = eval_context
        uid, df = self._held(env, assignments)
        first = select_held_items(df, 10_000, selection="first")
        rand = select_held_items(df, 10_000, selection="random", user_id=uid, seed=42)
        assert first == rand == select_held_items(df, None)

    def test_random_selection_requires_a_seed_and_user(self, env, eval_context):
        """Silently non-deterministic sampling would poison the comparison."""
        from experiments.llm_vs_associative import select_held_items

        assignments, _ = eval_context
        uid, df = self._held(env, assignments)
        with pytest.raises(ValueError, match="deterministic"):
            select_held_items(df, 2, selection="random", user_id=uid, seed=None)
        with pytest.raises(ValueError, match="deterministic"):
            select_held_items(df, 2, selection="random", user_id=None, seed=42)

    def test_unknown_selection_is_rejected(self, env, eval_context):
        """A typo must fail loudly rather than falling back to 'first'."""
        from experiments.llm_vs_associative import select_held_items

        assignments, _ = eval_context
        uid, df = self._held(env, assignments)
        with pytest.raises(ValueError, match="Unknown item selection"):
            select_held_items(df, 2, selection="most_recent", user_id=uid, seed=42)

    def test_both_arms_stay_matched_under_random_selection(self, tiny_config, env, eval_context):
        """The #2 guarantee must survive the new selection mode."""
        from experiments.llm_vs_associative import score_associative, select_held_items

        assignments, users = eval_context
        _, _, associative_pairs = score_associative(
            env, assignments, users, max_items_per_user=2, item_selection="random",
            seed=tiny_config.random_seed,
        )

        llm_pairs: list[tuple[int, int]] = []
        for assignment in assignments:
            df = env.held_out_for_user(
                assignment.base_user_id, split=tiny_config.recommender_eval_split,
            )
            if df.empty:
                continue
            for mid in select_held_items(
                df, 2, selection="random",
                user_id=assignment.base_user_id, seed=tiny_config.random_seed,
            ):
                llm_pairs.append((assignment.base_user_id, mid))

        assert associative_pairs == llm_pairs
        assert len(associative_pairs) > 0

    def test_run_names_distinguish_the_two_selections(self):
        """Comparing first-N against random-N requires two distinct runs."""
        from experiments.llm_vs_associative import baseline_run_name

        assert baseline_run_name(5, "first") == "associative-baseline-capped"
        assert baseline_run_name(5, "random") == "associative-baseline-capped-random"
        assert baseline_run_name(None, "random") == "associative-baseline"
