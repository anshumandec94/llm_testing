"""
sim.agents.llm — LLM-based scoring agent via mlx-lm.

Scores each candidate by prompting a local LLM to predict a rating in [1, 5].
The prompt gives the model a user's rating history (title, genres, overview,
and the rating they gave) and asks it to predict the rating for a new movie.
No persona or archetype information is provided — the LLM infers user
preferences solely from their past ratings, matching the information boundary
of the AssociativeAgent (which uses only the preference vector derived from
training-set ratings).

Returns raw predicted ratings in [1, 5]. The AssociativeAgent comparison
baseline is produced in the experiment file via bias + dot-product, also in
[1, 5], so MAE/RMSE are directly comparable.

History strategies
------------------
"top_rated"  — top-k items by explicit rating
"recent"     — top-k items by timestamp
"both"       — top-k by rating + top-k by timestamp (union, deduplicated)
"polarized"  — top-k highest-rated + top-k lowest-rated
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from lenskit.data import ItemList

from sim.agents.base import AbstractAgent
from sim.environment import Environment
from sim.persona import AgentPersona

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a movie rating predictor. You will be given a user's rating history and \
a candidate movie. Based only on what the user has liked or disliked before, \
predict the rating they would give on a scale from 1.0 to 5.0.
Respond ONLY with a JSON object in this exact format:
{"predicted_rating": <float between 1.0 and 5.0>, "reasoning": "<one sentence>"}
No other text."""

_FEW_SHOT_EXAMPLES = [
    {
        "user": (
            'History:\n'
            '- "Toy Story" (1995) | Animation, Comedy | '
            '"A cowboy doll is threatened by the arrival of a new spaceman toy..." | Rated: 4.0\n\n'
            'Predict rating for:\n'
            '"Shrek" (2001) | Animation, Comedy | '
            '"A green ogre named Shrek finds his swamp invaded by exiled fairytale characters..."'
        ),
        "assistant": '{"predicted_rating": 3.8, "reasoning": "Similar animated comedy era and humor but different emotional register."}',
    },
    {
        "user": (
            'History:\n'
            '- "Gravity" (2013) | Sci-Fi, Thriller | '
            '"Two astronauts work together to survive after a disaster in orbit..." | Rated: 3.5\n\n'
            'Predict rating for:\n'
            '"The Martian" (2015) | Sci-Fi, Drama | '
            '"An astronaut stranded on Mars must survive alone while NASA works to rescue him..."'
        ),
        "assistant": '{"predicted_rating": 4.1, "reasoning": "Same space-survival genre; warmer optimistic tone better suits the user\'s moderate rating of tense sci-fi."}',
    },
]


class LLMAgent(AbstractAgent):
    """
    Scores candidates by prompting a local LLM via mlx-lm.

    The model is loaded once at init and shared across all evaluate() calls.
    Returns predicted ratings in [1.0, 5.0] directly as item scores.

    Parameters
    ----------
    env:
        Initialised Environment (provides movie_meta and train_ratings).
    model_id:
        HuggingFace model ID to load via mlx-lm.
    history_k:
        Number of past items to include per history slot.
    history_strategy:
        One of "top_rated", "recent", "both", "polarized".
    max_tokens:
        Maximum new tokens per LLM generation call.
    overview_max_chars:
        Character truncation limit for movie overview text.
    use_few_shot:
        Whether to prepend fixed few-shot examples to anchor output format.
    """

    def __init__(
        self,
        env: Environment,
        model_id: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
        history_k: int = 2,
        history_strategy: str = "top_rated",
        max_tokens: int = 64,
        overview_max_chars: int = 300,
        use_few_shot: bool = True,
    ) -> None:
        from mlx_lm import load  # heavy import — deferred until agent is actually used

        self.env = env
        self.history_k = history_k
        self.history_strategy = history_strategy
        self.max_tokens = max_tokens
        self.overview_max_chars = overview_max_chars
        self.use_few_shot = use_few_shot

        self._meta = env.movie_meta.set_index("movieId")
        # Index by userId for O(log n) per-user lookups over the full training set.
        self._user_ratings = (
            env.train_ratings
            .merge(
                env.movie_meta[["movieId", "title", "genres", "overview"]],
                on="movieId",
                how="left",
            )
            .set_index("userId")
            .sort_index()
        )

        logger.info("Loading LLM: %s", model_id)
        self.model, self.tokenizer = load(model_id)
        logger.info("LLM loaded.")

    # ── Public interface ────────────────────────────────────────────────────

    def evaluate(
        self,
        candidates: ItemList,
        persona: AgentPersona,
        item_factors: dict[int, np.ndarray],
    ) -> ItemList:
        candidate_ids = [int(iid) for iid in candidates.ids()]
        history = self._get_history(persona.user_id)

        scores: list[float] = []
        for movie_id in candidate_ids:
            prompt_tokens = self._build_prompt(history, movie_id)
            rating = self._call_llm(prompt_tokens)
            scores.append(rating)

        return ItemList(candidates, scores=np.array(scores, dtype=np.float32))

    def update(
        self,
        user_id: int,
        interactions: list[tuple[int, str, float]],
    ) -> None:
        # No weight updates. History is read from env.train_ratings.
        # Future extension: maintain a per-user in-session buffer here.
        pass

    # ── Private helpers ─────────────────────────────────────────────────────

    def _get_history(self, user_id: int) -> list[dict]:
        """Return past-rated items for the given user per the history strategy."""
        try:
            user_df = self._user_ratings.loc[[user_id]].reset_index()
        except KeyError:
            return []

        k = self.history_k

        if self.history_strategy == "top_rated":
            subset = user_df.nlargest(k, "rating")
        elif self.history_strategy == "recent":
            subset = user_df.nlargest(k, "timestamp")
        elif self.history_strategy == "both":
            top = user_df.nlargest(max(1, k // 2), "rating")
            recent = user_df.nlargest(max(1, k - len(top)), "timestamp")
            subset = pd.concat([top, recent]).drop_duplicates("movieId").head(k * 2)
        elif self.history_strategy == "polarized":
            # Top-k highest rated + top-k lowest rated. Gives the model signal
            # about both ends of the user's preference spectrum.
            top = user_df.nlargest(k, "rating")
            bottom = user_df.nsmallest(k, "rating")
            subset = pd.concat([top, bottom]).drop_duplicates("movieId")
        else:
            logger.warning(
                "Unknown history_strategy %r; falling back to top_rated.",
                self.history_strategy,
            )
            subset = user_df.nlargest(k, "rating")

        return subset[["title", "genres", "overview", "rating"]].fillna("").to_dict("records")

    def _format_movie(self, row: dict) -> str:
        title = str(row.get("title", "Unknown")).strip()
        genres = str(row.get("genres", "")).replace("|", ", ").strip()
        overview = str(row.get("overview", "")).strip()[: self.overview_max_chars]
        if overview:
            return f'"{title}" | {genres} | "{overview}"'
        return f'"{title}" | {genres}'

    def _build_prompt(self, history: list[dict], candidate_id: int) -> str:
        """Build a tokenized prompt for a single candidate item."""
        try:
            candidate_row = self._meta.loc[candidate_id].to_dict()
        except KeyError:
            candidate_row = {"title": f"Movie {candidate_id}", "genres": "", "overview": ""}

        if history:
            history_lines = "\n".join(
                f'- {self._format_movie(h)} | Rated: {float(h["rating"]):.1f}/5.0'
                for h in history
            )
        else:
            history_lines = "(no rating history available)"

        user_content = (
            f"History:\n{history_lines}\n\n"
            f"Predict rating for:\n{self._format_movie(candidate_row)}"
        )

        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        if self.use_few_shot:
            for ex in _FEW_SHOT_EXAMPLES:
                messages.append({"role": "user", "content": ex["user"]})
                messages.append({"role": "assistant", "content": ex["assistant"]})

        messages.append({"role": "user", "content": user_content})

        # tokenize=False returns a plain string; mlx_lm.generate handles
        # tokenization internally and expects a string, not token ids.
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _call_llm(self, prompt: str) -> float:
        """Call the LLM and return a predicted rating in [1.0, 5.0]."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        try:
            text = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                sampler=make_sampler(temp=0.0),  # greedy — deterministic
                verbose=False,
            )
            return self._parse_rating(text)
        except Exception as exc:
            logger.warning("LLM call failed (%s); returning neutral score 3.0.", exc)
            return 3.0

    @staticmethod
    def _parse_rating(text: str) -> float:
        """Extract predicted_rating from JSON output; fallback to 3.0 (neutral)."""
        match = re.search(r'"predicted_rating"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
        if match:
            return max(1.0, min(5.0, float(match.group(1))))
        # Secondary: any bare float in [1, 5] range in the response.
        numbers = re.findall(r'\b([1-5](?:\.[0-9]+)?)\b', text)
        if numbers:
            return max(1.0, min(5.0, float(numbers[-1])))
        return 3.0
