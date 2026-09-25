"""
experiments.backends - one interface for every arm of the backend comparison.

A `PreferenceBackend` predicts, for one user and a list of items, the
DEBIASED-rating residual of each item, so that

    env.get_rating_bias(user, item) + residual

reconstructs a rating. Pinning every arm to that unit is what makes their
MAEs comparable. The harness deliberately does not route through
`AbstractAgent.evaluate()`: that method serves the simulation loop, and its
units differ by agent (a cosine for associative, a [1, 5] rating for the
LLM). Each adapter below does its own conversion instead, and `sim/` is left
untouched.

`np.nan` marks an item a backend genuinely cannot score (no factor row, no
metadata). It is never folded into an average here; the harness counts and
logs it.

All state comes from the training split alone (`env.train_ratings` and the
bias model fitted on it), exactly as `experiments/bias_only_null.py` and the
issue #27 residual arms use. Validation and held-out ratings never enter.

Clipping to [1, 5] is a reporting choice and belongs to the harness, so every
backend returns the unclipped residual.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import numpy as np
from lenskit.data import ItemList

from sim.environment import Environment
from sim.residual_factors import ResidualFactors

if TYPE_CHECKING:
    from sim.agents.base import AbstractAgent
    from sim.persona import AgentPersona


@runtime_checkable
class PreferenceBackend(Protocol):
    """Predicts debiased-rating residuals aligned with `item_ids`."""

    name: str

    def predict(self, user_id: int, item_ids: list[int]) -> np.ndarray: ...


class BiasOnlyBackend:
    """The null: a residual of exactly zero, i.e. no user-item interaction.

    `bias + 0` is `pred_null` in `experiments/bias_only_null.py`, pair for
    pair (unclamped; the clamped null is the harness's to compute). The bias
    model scores every pair, falling back to global + known terms, so nothing
    is ever nan.
    """

    name = "bias_only"

    def predict(self, user_id: int, item_ids: list[int]) -> np.ndarray:
        return np.zeros(len(item_ids), dtype=np.float64)


class AssociativeBackend:
    """Rating-unit residual factors: `U[user] @ V[item]` (issues #27, #13).

    The factors are injected, so the backend does not care which fit made
    them. The chosen fitter is the observed-cells ALS (`fit_residual_als`,
    untuned, 8 dims at the published capacity); the zero-imputed SVD is a
    shrunk estimator (PR #30). Use `from_env` for that choice.

    A user or item with no factor row is nan rather than the zero that
    `ResidualFactors.residual` falls back to, so the harness sees it.
    """

    name = "associative"

    def __init__(self, factors: ResidualFactors) -> None:
        self.factors = factors

    @classmethod
    def from_env(cls, env: Environment) -> AssociativeBackend:
        """ALS on the environment's training residuals, the chosen fitter."""
        from sim.residual_factors import fit_env_residual_als

        return cls(fit_env_residual_als(env))

    def predict(self, user_id: int, item_ids: list[int]) -> np.ndarray:
        users = np.full(len(item_ids), int(user_id))
        items = np.asarray(item_ids, dtype=np.int64)
        covered = self.factors.covered(users, items)
        return np.where(covered, self.factors.residuals(users, items), np.nan)


@dataclass
class _UserOnly:
    """Stand-in persona. The wrapped agents read `persona.user_id` only.

    A real `AgentPersona` would sample archetype traits and, for a cold user,
    draw from `env.rng`, so the prediction would depend on population state.
    If an agent ever reads more, this fails loudly with an AttributeError.
    """

    user_id: int


class LLMBackend:
    """`LLMAgent.evaluate` minus `env.get_rating_bias(u, i)`.

    The agent returns a rating in [1, 5]; subtracting the bias is the whole
    unit reconciliation. An item missing from `env.movie_meta` is nan, since
    the prompt would carry only a placeholder id. A cold user is still
    scored: the prompt says there is no history and the model's content-only
    guess is a legitimate prediction.

    Unparseable output cannot be nan through `evaluate`: `LLMAgent` maps it,
    and any call failure, to a neutral 3.0 that is indistinguishable from a
    real 3.0. Any non-finite score the agent does return is passed through as
    nan, so a stricter agent needs no change here.
    """

    name = "llm"

    def __init__(self, env: Environment, agent: AbstractAgent) -> None:
        self.env = env
        self.agent = agent
        self._known_items = set(int(m) for m in env.movie_meta["movieId"])

    def predict(self, user_id: int, item_ids: list[int]) -> np.ndarray:
        out = np.full(len(item_ids), np.nan)
        known = [i for i, mid in enumerate(item_ids) if int(mid) in self._known_items]
        if not known:
            return out
        ids = [int(item_ids[i]) for i in known]
        scored = self.agent.evaluate(
            ItemList(item_ids=np.array(ids, dtype=np.int64)),
            cast("AgentPersona", _UserOnly(int(user_id))),
            self.env.get_user_pref_item_factors(ids),
        )
        scores = scored.scores()
        if scores is None:
            return out
        ratings = np.asarray(scores, dtype=np.float64)
        bias = np.array([self.env.get_rating_bias(int(user_id), mid) for mid in ids])
        out[known] = np.where(np.isfinite(ratings), ratings - bias, np.nan)
        return out


class ResidualProfileBackend:
    """Not a rating-unit arm yet (see `_NOT_RATING_UNITS`)."""

    name = "residual_profile"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(_NOT_RATING_UNITS)

    def predict(self, user_id: int, item_ids: list[int]) -> np.ndarray:
        raise NotImplementedError(_NOT_RATING_UNITS)


class ItemItemBackend:
    """Not a rating-unit arm yet (see `_NOT_RATING_UNITS`)."""

    name = "item_item"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(_NOT_RATING_UNITS)

    def predict(self, user_id: int, item_ids: list[int]) -> np.ndarray:
        raise NotImplementedError(_NOT_RATING_UNITS)


# Both agents score sum_j r_j cos(v_i, v_j) / sum_j |r_j|, a dimensionless
# value in [-1, 1] rather than stars, and any rescaling to stars would be a
# fitted calibrator or a different model.
_NOT_RATING_UNITS = (
    "the agent's score is a |residual|-weighted mean cosine in [-1, 1], not a "
    "debiased rating residual, and has no honest conversion to stars"
)


class SASRecBackend:
    """Registration point for SASRec's rating head; issue #19 fills it."""

    name = "sasrec"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("SASRecBackend is filled by issue #19.")

    def predict(self, user_id: int, item_ids: list[int]) -> np.ndarray:
        raise NotImplementedError("SASRecBackend is filled by issue #19.")


# Every arm the harness can name, keyed by `name`. Construction arguments
# differ per backend, so the harness builds each one itself.
BACKEND_REGISTRY: dict[str, type] = {
    cls.name: cls
    for cls in (
        BiasOnlyBackend,
        AssociativeBackend,
        ResidualProfileBackend,
        ItemItemBackend,
        LLMBackend,
        SASRecBackend,
    )
}
