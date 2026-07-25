# LLMAgent Design Plan
## Next-Step Agent for ABM Recommender Simulation

**Date:** June 2026  
**Scope:** Drop-in replacement for `AssociativeAgent` using a local LLM via `mlx-lm` to predict per-item ratings, enabling direct comparison of neural vs. LLM-based user preference modelling.

---

## 1. Objective

The existing `AssociativeAgent` scores candidate items by computing a dot product between the user's `pref_vector` (TruncatedSVD space, dim=8) and the item's user-pref-space factor. This is purely latent-space arithmetic — fast and differentiable, but opaque.

The `LLMAgent` replaces this single dot product with a language model call that receives:
- A short description of items the user has previously rated (1–2 items, configurable)
- Metadata (title, genre, overview) for each candidate item

The LLM outputs a predicted rating (float, 0–5) for each candidate. These scores flow into the **same downstream persona/action pipeline** (softmax sampling, logistic action model, preference vector update) so all existing metrics remain directly comparable.

**Primary research question:** Does LLM-based item scoring improve ranking quality (NDCG@6, NDCG@10, hit rate) over the latent-factor baseline, and at what latency/token cost?

---

## 2. Agent Interface Contract

Every agent in this simulation implements `sim.agents.base.AbstractAgent`:

```python
class AbstractAgent(ABC):

    @abstractmethod
    def evaluate(
        self,
        candidates: ItemList,                   # LensKit ItemList — N candidate movies
        persona: AgentPersona,                  # full user state (pref_vector, archetype, etc.)
        item_factors: dict[int, np.ndarray],    # movieId → SVD vector (dim = user_pref_features)
    ) -> ItemList:                              # same candidates with .scores attached
        ...

    @abstractmethod
    def update(
        self,
        user_id: int,
        interactions: list[tuple[int, str, float]],  # (movie_id, action, signal)
    ) -> None:
        ...
```

### Key contractual rules
- `evaluate()` must return `ItemList(candidates, scores=np.ndarray[float32])` — never `None`, never raise.
- Scores drive everything downstream: `score_floor` gate, softmax action sampler, logistic action probability.
- Scores from `AssociativeAgent` are unit-norm dot products in `[-1, +1]`. The LLM outputs `[1, 5]`, which must be normalised to the same range: `normalized = (llm_score - 3.0) / 2.0` maps `1 → -1`, `3 → 0`, `5 → +1`.
- `update()` is where per-user memory is persisted. The existing ChromaDB collections in `embeddings/chroma/` are available via `env`.
- The agent instance is **shared across all simulated users** — `user_id` is passed at `update()` time to distinguish state.

---

## 3. Hardware & Model Selection

**Machine:** Apple Silicon with 36 GB unified memory.  
**Constraint:** Model footprint < 10–12 GB to leave headroom for the simulation, ChromaDB, and LensKit training.

### Recommended models (all available at `mlx-community` on HuggingFace)

| Rank | Model ID | Disk / RAM | Context | Licence | Notes |
|------|----------|------------|---------|---------|-------|
| **1** | `mlx-community/Qwen2.5-7B-Instruct-4bit` | 4.3 GB / ~5.5 GB | 128K | Apache 2.0 | Best JSON adherence in class; top instruction-following benchmark scores among 7B models; strong few-shot performance |
| **2** | `mlx-community/Qwen2.5-7B-Instruct-8bit` | 7.6 GB / ~9 GB | 128K | Apache 2.0 | Near-FP16 quality; worth the RAM cost if rating calibration matters |
| **3** | `mlx-community/Llama-3.1-8B-Instruct-4bit` | 4.5 GB / ~6 GB | 128K | Meta Llama 3.1 | Strong few-shot; very well-tested; good fallback option |
| **4** | `mlx-community/gemma-2-9b-it-4bit` | 5.2 GB / ~7 GB | 8K | Gemma ToS | Highest benchmark quality in this set; shorter context window |

**Primary recommendation: `Qwen2.5-7B-Instruct-4bit`** — Apache 2.0 licence (important for research), excellent JSON adherence for structured output, and ~5.5 GB leaves ~30 GB free for the rest of the pipeline.

### SimConfig field to add
```python
llm_model_id: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"
llm_history_k: int = 2          # number of past rated items to include as context
llm_history_strategy: str = "top_rated"   # "top_rated" | "recent" | "both"
llm_max_tokens: int = 64        # sufficient for {"predicted_rating": 4.2, "reasoning": "..."}
llm_overview_max_chars: int = 300  # truncation limit for movie overviews in prompt
```

---

## 4. Runtime API (`mlx-lm`)

```bash
uv add mlx-lm outlines
```

### Load (once at startup, shared across all users)
```python
from mlx_lm import load

model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
```
Model loading is the expensive step (~5–15s). Load once in `LLMAgent.__init__` and reuse across all `evaluate()` calls.

### Generate
```python
from mlx_lm import generate

text: str = generate(
    model,
    tokenizer,
    prompt=prompt_tokens,   # from tokenizer.apply_chat_template(messages, ...)
    max_tokens=64,
    temp=0.0,               # greedy — deterministic, better for rating prediction
)
```

### Streaming (optional, for debugging)
```python
from mlx_lm import stream_generate

for response in stream_generate(model, tokenizer, prompt, max_tokens=64):
    print(response.text, end="", flush=True)
    # response.generation_tps — tokens/sec for profiling
```

---

## 5. Structured Output: Guaranteed JSON Float

`mlx-lm` has no built-in JSON mode. Three approaches in order of reliability:

### Option A — `outlines` (recommended: mathematically guaranteed)

`outlines` supports `mlx-lm` natively via `outlines.models.mlxlm`. It masks illegal tokens at each decoding step via grammar-constrained sampling — the schema is physically enforced, not just prompted.

```python
import outlines
from pydantic import BaseModel, Field

class ItemRating(BaseModel):
    predicted_rating: float = Field(ge=1.0, le=5.0)
    reasoning: str

mlx_model = outlines.models.mlxlm("mlx-community/Qwen2.5-7B-Instruct-4bit")
generator = outlines.generate.json(mlx_model, ItemRating)

# In evaluate() — one call per candidate:
result: ItemRating = generator(prompt_str)
score = result.predicted_rating  # guaranteed float in [1.0, 5.0]
```

**Tradeoff:** `outlines` wraps the `mlx-lm` model; you cannot mix this with the raw `mlx_lm.generate` call in the same object. Maintain two references if needed.

### Option B — `logits_processors` hook (custom, no extra dependency)

`mlx_lm.generate` accepts a `logits_processors` list. Each processor is `Callable[[mx.array, mx.array], mx.array]` — receives `(token_history, logits)` and returns modified logits. You can implement a lightweight regex/FSM enforcer here without `outlines`.

```python
from mlx_lm import generate
import mlx.core as mx

def json_enforcer(tokens: mx.array, logits: mx.array) -> mx.array:
    # Mask tokens that would break JSON validity at current position
    # (simplified — a real FSM tracks parse state)
    ...

text = generate(model, tokenizer, prompt, max_tokens=64,
                logits_processors=[json_enforcer])
```

### Option C — Prompt engineering + regex parse (simplest, lowest latency overhead)

Works reliably with Qwen2.5/Llama3.1 instruction models when the system prompt is strong:

```python
import json, re

def parse_rating(text: str) -> float | None:
    match = re.search(r'"predicted_rating"\s*:\s*([0-9.]+)', text)
    if match:
        val = float(match.group(1))
        return max(1.0, min(5.0, val))  # clamp to [1, 5]
    return None  # fallback: return neutral score 3.0
```

**Recommendation for initial implementation:** Start with Option C (no extra dependency, fastest iteration). Add Option A for production runs once the prompt is stable.

---

## 6. Movie Metadata Available

```
data/ml-32m/
├── movies.csv          # movieId, title, genres (pipe-separated)
├── movie_overviews.csv # movieId, overview (plot synopsis — subset of movies)
├── ratings.csv         # userId, movieId, rating (0.5–5.0), timestamp
```

These are already merged and accessible in the `Environment` object:

```python
# env.movie_meta: pd.DataFrame with columns [movieId, title, genres, overview]
# Build a fast index once in __init__:
self._meta = env.movie_meta.set_index("movieId")

# Lookup in evaluate():
row = self._meta.loc[movie_id]
title    = row["title"]            # e.g. "Inception (2010)"
genres   = row["genres"].replace("|", ", ")  # e.g. "Action, Sci-Fi, Thriller"
overview = row["overview"][:300]   # truncated to llm_overview_max_chars
```

User rating history is in `env.train_ratings` (a DataFrame with `userId`, `movieId`, `rating`, `timestamp`). This is used to build the history context in the prompt.

---

## 7. Accessing User History

The `persona` object does **not** carry a list of past movies directly — it holds only the learned `pref_vector`. Raw rating history lives in `env.train_ratings`.

```python
# In evaluate() — build history context for user
user_ratings = (
    env.train_ratings[env.train_ratings["userId"] == persona.user_id]
    .merge(env.movie_meta[["movieId", "title", "genres", "overview"]], on="movieId", how="left")
)

# Strategy: "top_rated" — highest explicit ratings
history = (
    user_ratings
    .nlargest(history_k, "rating")
    [["title", "genres", "overview", "rating"]]
    .to_dict("records")
)

# Strategy: "recent" — most recently rated
history = (
    user_ratings
    .nlargest(history_k, "timestamp")
    [["title", "genres", "overview", "rating"]]
    .to_dict("records")
)
```

**Performance note:** `env.train_ratings` is a large DataFrame. Index it on `userId` once in `__init__`:

```python
self._ratings_by_user = env.train_ratings.set_index("userId")
```

Then lookup is `self._ratings_by_user.loc[persona.user_id]` which is O(log n) with a sorted index.

---

## 8. Prompt Design

### Format: chat template

Use `tokenizer.apply_chat_template` with the model's native chat format. This is critical — Qwen2.5 and Llama 3.1 have structured chat templates that enforce role boundaries the model was trained on.

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    # (optional) few-shot examples as alternating user/assistant turns
    {"role": "user",   "content": user_prompt},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True
)
```

### Strategy 1: Direct prediction (baseline, fast)

Best for benchmarking latency. One system prompt + one user turn per candidate.

**System prompt:**
```
You are a movie rating predictor. Given a user's watch history and a target movie,
predict the rating the user would give (1.0–5.0).
Respond ONLY with a JSON object: {"predicted_rating": <float>, "reasoning": "<1 sentence>"}
Do not include any other text.
```

**User turn (1 history item):**
```
User history:
- "The Dark Knight" (2008) | Action, Crime, Drama | "When Batman faces the Joker, Gotham is pushed to its limits..." | Rated: 5.0/5.0

Predict rating for:
"Batman Begins" (2005) | Action, Adventure | "After witnessing his parents' brutal murder, billionaire Bruce Wayne..."
```

### Strategy 2: Few-shot (recommended for cold-start, 1–2 history items)

Prepend 2 in-context examples as `user`/`assistant` turns before the real query. This strongly anchors the output format and calibrates the numerical scale without requiring chain-of-thought.

```python
FEW_SHOT_EXAMPLES = [
    {
        "user": (
            'User history:\n- "Toy Story" (1995) | Animation, Comedy | '
            '"A cowboy doll is threatened..." | Rated: 4.0/5.0\n\n'
            'Predict rating for:\n"Shrek" (2001) | Animation, Comedy | '
            '"A green ogre named Shrek finds his swamp invaded..."'
        ),
        "assistant": '{"predicted_rating": 3.8, "reasoning": "Similar family animation era and humor but different emotional tone."}',
    },
    {
        "user": (
            'User history:\n- "Gravity" (2013) | Sci-Fi, Thriller | '
            '"Two astronauts work together to survive..." | Rated: 3.5/5.0\n\n'
            'Predict rating for:\n"The Martian" (2015) | Sci-Fi, Drama | '
            '"An astronaut is stranded alone on Mars and must survive..."'
        ),
        "assistant": '{"predicted_rating": 4.1, "reasoning": "Same space survival genre; more optimistic tone aligns better with user preferences."}',
    },
]
```

These examples are fixed across all users and candidates — they just establish format and scale.

### Strategy 3: Chain-of-thought (best accuracy, highest latency)

Ask the model to reason through genre/director/theme overlap before scoring. Improve NDCG by ~15–20% on cold-start users (1–2 history items) at the cost of ~3× more tokens.

```
You are a movie rating predictor. Think step by step:
1. Identify the user's genre/director/theme preferences from their history.
2. Assess overlap with the target movie.
3. Estimate a rating from 1.0–5.0.
4. Respond ONLY with JSON: {"predicted_rating": <float>, "reasoning": "<1 sentence>"}
```

**Recommended configuration per use case:**

| Use case | Strategy | `max_tokens` | Expected latency/item |
|---|---|---|---|
| Benchmarking (speed) | Direct | 32 | ~0.1–0.2s |
| Standard run | Few-shot | 64 | ~0.2–0.4s |
| Cold-start users | CoT | 256 | ~0.5–1.0s |

---

## 9. Batching Strategy

The existing `evaluate()` receives a single `ItemList` of `rec_list_size` (default 6) candidates. Two batching approaches:

### Option A: One LLM call per candidate (sequential)
Score each candidate independently. Simplest to implement; easiest to debug.  
**Latency per `evaluate()` call:** `6 × 0.2s = 1.2s` for 6 candidates.

```python
scores = []
for movie_id in candidate_ids:
    prompt = self._build_prompt(persona, movie_id)
    text = generate(self.model, self.tokenizer, prompt, max_tokens=64, temp=0.0)
    rating = self._parse_rating(text) or 3.0
    scores.append(rating)
```

### Option B: All candidates in one call (batch prompt)
Describe all 6 candidates in one prompt, ask for a JSON array of scores.  
**Latency:** ~0.3–0.5s total, but the structured output schema is more complex.

```python
class BatchRating(BaseModel):
    ratings: list[float]  # length == len(candidates), each in [1.0, 5.0]
```

**Recommendation:** Start with Option A (sequential, per-candidate). It maps directly to the existing one-score-per-item contract, is trivially debuggable, and 1–2s per round is acceptable for `recommender_only` evaluation mode. Add batching if full simulation latency is unacceptable.

---

## 10. Score Normalisation

The `AssociativeAgent` produces scores in `[-1, +1]` (L2-normalised dot products). All downstream thresholds (`score_floor`, `tau` temperature) are calibrated to this range.

Map LLM output `[1, 5]` → `[-1, +1]`:

```python
def _normalize_score(self, llm_rating: float) -> float:
    """Map [1, 5] → [-1, +1] to match AssociativeAgent score range."""
    return (llm_rating - 3.0) / 2.0
```

| LLM rating | Normalized | Interpretation |
|---|---|---|
| 1.0 | -1.0 | Strongly disliked |
| 2.0 | -0.5 | Below average |
| 3.0 | 0.0 | Neutral |
| 4.0 | +0.5 | Liked |
| 5.0 | +1.0 | Strongly liked |

---

## 11. Implementation Skeleton

```python
# sim/agents/llm.py

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from lenskit.data import ItemList

from sim.agents.base import AbstractAgent
from sim.environment import Environment
from sim.persona import AgentPersona

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a movie rating predictor.
Given a user's watch history and a candidate movie, predict the rating the user would give (1.0–5.0).
Respond ONLY with a JSON object: {"predicted_rating": <float 1.0-5.0>, "reasoning": "<1 sentence>"}
No other text."""

_FEW_SHOT = [
    {
        "user": (
            'History:\n- "Toy Story" (1995) | Animation, Comedy | "A cowboy doll is threatened..." | Rated: 4.0\n\n'
            'Predict rating for:\n"Shrek" (2001) | Animation, Comedy | "A green ogre finds his swamp invaded..."'
        ),
        "assistant": '{"predicted_rating": 3.8, "reasoning": "Similar family animation era and humor, but different emotional tone."}',
    },
    {
        "user": (
            'History:\n- "Gravity" (2013) | Sci-Fi | "Two astronauts survive a disaster..." | Rated: 3.5\n\n'
            'Predict rating for:\n"The Martian" (2015) | Sci-Fi | "An astronaut is stranded on Mars..."'
        ),
        "assistant": '{"predicted_rating": 4.1, "reasoning": "Same survival genre; warmer tone matches higher rated history."}',
    },
]


class LLMAgent(AbstractAgent):
    """
    LLM-based evaluation agent using mlx-lm on Apple Silicon.

    Scores candidate items by prompting a local LLM with the user's
    top-rated history (configurable k) and each candidate's metadata.
    Output is normalised from [1,5] to [-1,+1] to match AssociativeAgent.
    """

    def __init__(
        self,
        env: Environment,
        model_id: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
        history_k: int = 2,
        history_strategy: str = "top_rated",   # "top_rated" | "recent" | "both"
        max_tokens: int = 64,
        overview_max_chars: int = 300,
        use_few_shot: bool = True,
    ) -> None:
        from mlx_lm import load  # import here — heavy; only loaded when agent is used

        self.env = env
        self.history_k = history_k
        self.history_strategy = history_strategy
        self.max_tokens = max_tokens
        self.overview_max_chars = overview_max_chars
        self.use_few_shot = use_few_shot

        # O(1) metadata lookups
        self._meta = env.movie_meta.set_index("movieId")

        # Per-user rating history index
        self._user_ratings = env.train_ratings.set_index("userId").sort_index()

        logger.info("Loading LLM: %s", model_id)
        self.model, self.tokenizer = load(model_id)
        logger.info("LLM loaded.")

    # ── Public interface ───────────────────────────────────────────────────

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
            prompt = self._build_prompt(history, movie_id)
            rating = self._call_llm(prompt)
            scores.append(self._normalize_score(rating))

        return ItemList(candidates, scores=np.array(scores, dtype=np.float32))

    def update(
        self,
        user_id: int,
        interactions: list[tuple[int, str, float]],
    ) -> None:
        # No in-weights updates; history is read from env.train_ratings.
        # Future extension: maintain a per-user in-session memory buffer here.
        pass

    # ── Private helpers ────────────────────────────────────────────────────

    def _get_history(self, user_id: int) -> list[dict]:
        """Return up to history_k past-rated items for the given user."""
        try:
            user_df = self._user_ratings.loc[[user_id]].reset_index()
        except KeyError:
            return []

        user_df = user_df.merge(
            self.env.movie_meta[["movieId", "title", "genres", "overview"]],
            on="movieId", how="left",
        )

        if self.history_strategy == "top_rated":
            subset = user_df.nlargest(self.history_k, "rating")
        elif self.history_strategy == "recent":
            subset = user_df.nlargest(self.history_k, "timestamp")
        else:  # "both"
            top = user_df.nlargest(max(1, self.history_k // 2), "rating")
            recent = user_df.nlargest(max(1, self.history_k - len(top)), "timestamp")
            subset = pd.concat([top, recent]).drop_duplicates("movieId")

        return subset[["title", "genres", "overview", "rating"]].to_dict("records")

    def _format_movie(self, row: dict) -> str:
        title    = row.get("title", "Unknown")
        genres   = str(row.get("genres", "")).replace("|", ", ")
        overview = str(row.get("overview", "") or "")[:self.overview_max_chars]
        return f'"{title}" | {genres} | "{overview}"'

    def _build_prompt(self, history: list[dict], candidate_id: int) -> list[int]:
        """Build tokenized prompt for a single candidate item."""
        try:
            candidate_row = self._meta.loc[candidate_id].to_dict()
        except KeyError:
            candidate_row = {"title": f"Movie {candidate_id}", "genres": "", "overview": ""}

        history_lines = "\n".join(
            f'- {self._format_movie(h)} | Rated: {h["rating"]:.1f}'
            for h in history
        ) or "(no history available)"

        user_content = (
            f"History:\n{history_lines}\n\n"
            f"Predict rating for:\n{self._format_movie(candidate_row)}"
        )

        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        if self.use_few_shot:
            for ex in _FEW_SHOT:
                messages.append({"role": "user",      "content": ex["user"]})
                messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": user_content})

        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )

    def _call_llm(self, prompt_tokens: list[int]) -> float:
        """Call the LLM and return a parsed rating in [1.0, 5.0]."""
        from mlx_lm import generate

        try:
            text = generate(
                self.model, self.tokenizer,
                prompt=prompt_tokens,
                max_tokens=self.max_tokens,
                temp=0.0,  # greedy — deterministic for reproducibility
                verbose=False,
            )
            return self._parse_rating(text)
        except Exception as exc:
            logger.warning("LLM call failed (%s); using neutral score.", exc)
            return 3.0

    @staticmethod
    def _parse_rating(text: str) -> float:
        """Extract predicted_rating from LLM JSON output; fallback to 3.0."""
        match = re.search(r'"predicted_rating"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
        if match:
            val = float(match.group(1))
            return max(1.0, min(5.0, val))
        # Fallback: try to parse any float in the response
        numbers = re.findall(r'\b([1-5](?:\.[0-9]+)?)\b', text)
        if numbers:
            return float(numbers[-1])
        return 3.0  # neutral

    @staticmethod
    def _normalize_score(llm_rating: float) -> float:
        """Map [1, 5] → [-1, +1] to match AssociativeAgent dot-product range."""
        return (llm_rating - 3.0) / 2.0
```

---

## 12. Registration

Add to `sim/agents/__init__.py` `AGENT_REGISTRY`:

```python
"llm": lambda config, env: LLMAgent(
    env,
    model_id=getattr(config, "llm_model_id", "mlx-community/Qwen2.5-7B-Instruct-4bit"),
    history_k=getattr(config, "llm_history_k", 2),
    history_strategy=getattr(config, "llm_history_strategy", "top_rated"),
    max_tokens=getattr(config, "llm_max_tokens", 64),
    overview_max_chars=getattr(config, "llm_overview_max_chars", 300),
),
```

Add to `main.py` `--agent_type` choices:
```python
choices=["associative", "associative_baseline", "residual_profile", "item_item",
         "semantic", "seq2seq", "llm"],
```

---

## 13. SimConfig Fields to Add

```python
# sim/config.py — add to SimConfig dataclass
llm_model_id: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"
llm_history_k: int = 2
llm_history_strategy: str = "top_rated"   # "top_rated" | "recent" | "both"
llm_max_tokens: int = 64
llm_overview_max_chars: int = 300
llm_use_few_shot: bool = True
```

---

## 14. Comparative Experiment Setup

To compare `AssociativeAgent` vs `LLMAgent` on the same users:

```bash
# recommender_only mode — fastest, no simulation loop required
uv run python main.py \
  --experiment_profile recommender_only \
  --recommender_eval_split held_out \
  --agent_types associative,llm \
  --agent_type_proportions 0.5,0.5 \
  --agent_assignment_mode one_per_agent_type \
  --eval_user_frac 0.05   # start with 5% users to test latency
```

**Expected metrics for comparison (already in MLflow with new grouping):**
- `ranking/hit_rate` — does the LLM's score ordering improve retrieval?
- `ranking/ndcg_at_6`, `ranking/ndcg_at_10` — ranking quality
- `error/int_mse`, `error/int_rmse`, `error/int_mae` — rating prediction accuracy vs ground-truth debiased ratings
- `correlation/int_pearson` — per-user rank correlation with actual ratings

---

## 15. Latency Budget

With `Qwen2.5-7B-Instruct-4bit` on Apple Silicon (M2/M3/M4):
- **Model load time:** ~5–15s (one-time)
- **Per-item generation (64 tokens):** ~0.1–0.3s at ~50–100 tok/s
- **Per `evaluate()` call (6 candidates, sequential):** ~0.6–1.8s

For `recommender_only` evaluation with 12,834 users (`eval_user_frac=0.2`):
- Each user: 1 `evaluate()` call = ~1.2s
- Total: ~4.3 hours at 1.2s/user

**Recommendation for first run:** Use `eval_user_frac=0.02` (~2,568 users) → ~50 minutes. Cache the `pref_vector` index and `_user_ratings` lookup. Consider batching candidates into a single prompt (Option B above) to reduce total calls from 6N to N.

For full simulation (10 rounds × N users), LLM latency will dominate. Consider:
1. Caching LLM scores for (user_id, movie_id) pairs across rounds.
2. Using the batch prompt strategy.
3. Running with a smaller `eval_user_frac` (0.02–0.05).

---

## 16. Upgrade Path: Structured Outputs via `outlines`

Once the prompt is stable, switch from regex parsing to guaranteed JSON:

```python
import outlines
from pydantic import BaseModel, Field

class ItemRating(BaseModel):
    predicted_rating: float = Field(ge=1.0, le=5.0)
    reasoning: str

# In __init__:
mlx_model = outlines.models.mlxlm(model_id)
self._generator = outlines.generate.json(mlx_model, ItemRating)

# In _call_llm:
result: ItemRating = self._generator(prompt_str)  # takes a string, not token IDs
return result.predicted_rating
```

Note: `outlines.models.mlxlm` takes the model ID string directly (it handles `load()` internally), so you cannot share the same model object between `mlx_lm.generate` calls and `outlines` calls. Maintain a single `outlines` generator or a single raw `mlx_lm` model, not both.

---

## 17. Open Questions / Future Work

1. **Cold-start quality:** With only 1–2 history items, LLM scores may not outperform the BiasedMF latent space. Consider augmenting history with the user's `archetype` field from `persona` as a natural-language personality descriptor.

2. **Rating scale calibration:** LLMs tend to cluster predictions near the mean (3.5–4.0). The normalisation `(x-3.0)/2.0` may over-compress variance. Evaluate the distribution of `error/int_mae` vs `AssociativeAgent` and consider a learned affine rescaling.

3. **Per-round memory accumulation:** In full simulation mode, `update()` receives new interactions each round. Extending the history context to include in-session interactions (not just training ratings) could meaningfully improve round 3+ performance.

4. **Overview coverage gap:** Not all MovieLens movies have overviews. For missing overviews, the prompt will fall back to title + genre only — monitor `overview_missing_rate` as an experiment tag.

5. **Prompt caching:** If multiple users share the same candidate set (common in recommender_only mode), the system prompt and few-shot examples can be cached using `mlx-lm`'s `prompt_cache` parameter to skip re-encoding the prefix.
