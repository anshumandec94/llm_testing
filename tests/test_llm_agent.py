"""
tests/test_llm_agent.py — unit and integration tests for LLMAgent.

Covers the three classes of bugs that have occurred:
  1. BatchEncoding bug: apply_chat_template(tokenize=True) returned a
     BatchEncoding object instead of a string, crashing mlx_lm.generate.
  2. temp kwarg bug: mlx_lm.generate does not accept temp= directly;
     greedy sampling must be passed via sampler=make_sampler(temp=0.0).
  3. Parse fallback: LLM output that doesn't match the JSON schema must
     return 3.0 (neutral) rather than raising.

No real LLM is loaded. The model and tokenizer are mocked so these tests
run in milliseconds without GPU or network access.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
from lenskit.data import ItemList

from sim.agents.llm import LLMAgent


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_env():
    """Minimal environment stub sufficient for LLMAgent."""
    n_movies = 10
    movie_meta = pd.DataFrame({
        "movieId": range(1, n_movies + 1),
        "title":   [f"Movie {i} (200{i})" for i in range(1, n_movies + 1)],
        "genres":  ["Action|Comedy", "Drama", "Thriller", "Animation", "Sci-Fi",
                    "Romance", "Horror", "Documentary", "Musical", "Western"],
        "overview": [f"Overview for movie {i}." if i % 2 == 0 else ""
                     for i in range(1, n_movies + 1)],
    })
    train_ratings = pd.DataFrame({
        "userId":    [1, 1, 1, 1, 1, 2, 2],
        "movieId":   [1, 2, 3, 4, 5, 1, 3],
        "rating":    [5.0, 1.0, 4.0, 2.0, 3.0, 4.5, 2.0],
        "timestamp": [100, 200, 300, 400, 500, 100, 200],
    })

    env = MagicMock()
    env.movie_meta = movie_meta
    env.train_ratings = train_ratings
    return env


@pytest.fixture(scope="module")
def mock_tokenizer():
    """Tokenizer that returns a plain string from apply_chat_template."""
    tok = MagicMock()
    tok.apply_chat_template.return_value = "<mock_prompt_string>"
    return tok


@pytest.fixture(scope="module")
def agent(mock_env, mock_tokenizer):
    """LLMAgent with mocked model/tokenizer — no real LLM loaded."""
    with patch("mlx_lm.load", return_value=(MagicMock(), mock_tokenizer)):
        return LLMAgent(
            mock_env,
            model_id="mock-model",
            history_k=2,
            history_strategy="top_rated",
            max_tokens=64,
            overview_max_chars=200,
            use_few_shot=False,   # keeps prompts short in tests
        )


@pytest.fixture
def persona():
    p = MagicMock()
    p.user_id = 1
    p.pref_vector = np.array([0.5, 0.5, 0.0, 0.0])
    return p


# ── _parse_rating ─────────────────────────────────────────────────────────────

class TestParseRating:
    def test_parses_valid_json(self):
        assert LLMAgent._parse_rating('{"predicted_rating": 4.2, "reasoning": "x"}') == pytest.approx(4.2)

    def test_parses_integer_value(self):
        assert LLMAgent._parse_rating('{"predicted_rating": 5, "reasoning": "x"}') == pytest.approx(5.0)

    def test_clamps_above_5(self):
        assert LLMAgent._parse_rating('{"predicted_rating": 6.0}') == pytest.approx(5.0)

    def test_clamps_below_1(self):
        assert LLMAgent._parse_rating('{"predicted_rating": 0.0}') == pytest.approx(1.0)

    def test_fallback_on_empty_string(self):
        assert LLMAgent._parse_rating("") == pytest.approx(3.0)

    def test_fallback_on_garbage(self):
        assert LLMAgent._parse_rating("sorry I cannot rate that") == pytest.approx(3.0)

    def test_fallback_on_out_of_range_bare_float(self):
        # Bare float outside [1,5] should return 3.0, not a clamped value
        assert LLMAgent._parse_rating("The answer is 99.0") == pytest.approx(3.0)

    def test_secondary_bare_float_in_range(self):
        # If no JSON key found but a bare [1-5] float is present, use it
        result = LLMAgent._parse_rating("I think 4.5 is appropriate")
        assert result == pytest.approx(4.5)


# ── _build_prompt ─────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_returns_str_not_batch_encoding(self, agent):
        """Regression: apply_chat_template(tokenize=True) returned BatchEncoding."""
        history = [{"title": "Movie 1", "genres": "Action", "overview": "Good film.", "rating": 5.0}]
        result = agent._build_prompt(history, candidate_id=3)
        assert isinstance(result, str), (
            f"_build_prompt must return str, got {type(result).__name__}. "
            "This was the BatchEncoding bug — ensure tokenize=False in apply_chat_template."
        )

    def test_tokenizer_called_with_tokenize_false(self, agent, mock_tokenizer):
        """Regression: apply_chat_template must be called with tokenize=False."""
        history = [{"title": "Movie 1", "genres": "Action", "overview": "", "rating": 4.0}]
        mock_tokenizer.apply_chat_template.reset_mock()
        agent._build_prompt(history, candidate_id=2)
        _, kwargs = mock_tokenizer.apply_chat_template.call_args
        assert kwargs.get("tokenize") is False, (
            "apply_chat_template must use tokenize=False. "
            "tokenize=True returns a BatchEncoding that crashes mlx_lm.generate."
        )

    def test_returns_nonempty_string(self, agent):
        history = [{"title": "Movie 1", "genres": "Action", "overview": "Test.", "rating": 5.0}]
        result = agent._build_prompt(history, candidate_id=3)
        assert len(result) > 0

    def test_unknown_movie_id_falls_back_gracefully(self, agent):
        """Movies not in env.movie_meta should not raise."""
        result = agent._build_prompt([], candidate_id=99999)
        assert isinstance(result, str)

    def test_empty_overview_omitted_from_format(self, agent):
        """Movies with no overview should produce a prompt without empty quotes."""
        row = {"title": "Movie 1", "genres": "Action", "overview": "", "rating": 4.0}
        formatted = agent._format_movie(row)
        assert '""' not in formatted

    def test_overview_truncated_to_max_chars(self, agent):
        long_overview = "A" * 500
        row = {"title": "Film", "genres": "Drama", "overview": long_overview, "rating": 3.0}
        formatted = agent._format_movie(row)
        assert long_overview not in formatted
        assert "A" * 200 in formatted


# ── _call_llm ─────────────────────────────────────────────────────────────────

class TestCallLlm:
    def test_passes_string_to_generate(self, agent):
        """Regression: mlx_lm.generate must receive a str, not a token list or BatchEncoding."""
        with patch("mlx_lm.generate", return_value='{"predicted_rating": 4.0}') as mock_gen:
            agent._call_llm("<mock_prompt>")
        args, _ = mock_gen.call_args
        prompt_arg = args[2] if len(args) > 2 else mock_gen.call_args.kwargs.get("prompt")
        assert isinstance(prompt_arg, str), (
            f"mlx_lm.generate must receive a str prompt, got {type(prompt_arg).__name__}."
        )

    def test_uses_sampler_not_temp_kwarg(self, agent):
        """Regression: mlx_lm.generate does not accept temp= directly.
        Greedy sampling requires sampler=make_sampler(temp=0.0)."""
        with patch("mlx_lm.generate", return_value='{"predicted_rating": 3.5}') as mock_gen:
            with patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()) as mock_sampler:
                agent._call_llm("<mock_prompt>")
        # Verify temp= was NOT passed directly to generate
        _, kwargs = mock_gen.call_args
        assert "temp" not in kwargs, (
            "temp= must not be passed directly to mlx_lm.generate — use sampler=make_sampler(temp=0.0)."
        )
        # Verify make_sampler was called
        mock_sampler.assert_called_once()

    def test_returns_neutral_on_generate_failure(self, agent):
        with patch("mlx_lm.generate", side_effect=RuntimeError("GPU exploded")):
            result = agent._call_llm("<mock_prompt>")
        assert result == pytest.approx(3.0)

    def test_returns_neutral_on_bad_output(self, agent):
        with patch("mlx_lm.generate", return_value="I don't know"):
            result = agent._call_llm("<mock_prompt>")
        assert result == pytest.approx(3.0)

    def test_parses_valid_generate_output(self, agent):
        with patch("mlx_lm.generate", return_value='{"predicted_rating": 4.7, "reasoning": "great"}'):
            result = agent._call_llm("<mock_prompt>")
        assert result == pytest.approx(4.7)


# ── _get_history ──────────────────────────────────────────────────────────────

class TestGetHistory:
    def test_top_rated_returns_highest_rated(self, agent):
        history = agent._get_history(user_id=1)
        ratings = [h["rating"] for h in history]
        assert ratings == sorted(ratings, reverse=True)

    def test_top_rated_respects_k(self, agent):
        history = agent._get_history(user_id=1)
        assert len(history) <= agent.history_k

    def test_polarized_includes_both_extremes(self, mock_env, mock_tokenizer):
        with patch("mlx_lm.load", return_value=(MagicMock(), mock_tokenizer)):
            pol_agent = LLMAgent(mock_env, model_id="mock", history_k=2,
                                 history_strategy="polarized", use_few_shot=False)
        history = pol_agent._get_history(user_id=1)
        ratings = [h["rating"] for h in history]
        # Should include both a high and a low rated movie
        assert max(ratings) >= 4.0
        assert min(ratings) <= 2.0

    def test_recent_returns_most_recent(self, mock_env, mock_tokenizer):
        with patch("mlx_lm.load", return_value=(MagicMock(), mock_tokenizer)):
            rec_agent = LLMAgent(mock_env, model_id="mock", history_k=2,
                                 history_strategy="recent", use_few_shot=False)
        history = rec_agent._get_history(user_id=1)
        # User 1's most recent items by timestamp are movieId 5 (ts=500) and 4 (ts=400)
        movie_ids = {int(h["title"].split()[1]) for h in history}
        assert 5 in movie_ids

    def test_unknown_user_returns_empty(self, agent):
        history = agent._get_history(user_id=9999)
        assert history == []

    def test_invalid_strategy_falls_back_without_raising(self, mock_env, mock_tokenizer):
        with patch("mlx_lm.load", return_value=(MagicMock(), mock_tokenizer)):
            bad_agent = LLMAgent(mock_env, model_id="mock", history_k=2,
                                 history_strategy="nonexistent", use_few_shot=False)
        history = bad_agent._get_history(user_id=1)
        assert isinstance(history, list)


# ── evaluate() ───────────────────────────────────────────────────────────────

class TestEvaluate:
    @pytest.fixture
    def item_factors(self):
        return {i: np.random.default_rng(i).standard_normal(4).astype(np.float32)
                for i in range(1, 6)}

    def test_returns_item_list(self, agent, persona, item_factors):
        candidates = ItemList(item_ids=np.array([1, 2, 3], dtype=np.int64))
        with patch("mlx_lm.generate", return_value='{"predicted_rating": 3.8}'):
            result = agent.evaluate(candidates, persona, item_factors)
        assert isinstance(result, ItemList)

    def test_scores_in_rating_range(self, agent, persona, item_factors):
        candidates = ItemList(item_ids=np.array([1, 2, 3], dtype=np.int64))
        with patch("mlx_lm.generate", return_value='{"predicted_rating": 4.1}'):
            result = agent.evaluate(candidates, persona, item_factors)
        scores = result.scores()
        assert scores is not None
        assert np.all(scores >= 1.0)
        assert np.all(scores <= 5.0)

    def test_preserves_item_count(self, agent, persona, item_factors):
        ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        candidates = ItemList(item_ids=ids)
        with patch("mlx_lm.generate", return_value='{"predicted_rating": 3.0}'):
            result = agent.evaluate(candidates, persona, item_factors)
        assert len(result) == 5

    def test_falls_back_to_neutral_on_llm_failure(self, agent, persona, item_factors):
        """Each failed LLM call returns 3.0; evaluate() must not raise."""
        candidates = ItemList(item_ids=np.array([1, 2], dtype=np.int64))
        with patch("mlx_lm.generate", side_effect=RuntimeError("crash")):
            result = agent.evaluate(candidates, persona, item_factors)
        scores = result.scores()
        assert scores is not None
        assert np.all(scores == pytest.approx(3.0))

    def test_update_is_noop(self, agent):
        agent.update(user_id=1, interactions=[(1, "rate", 4.0)])
