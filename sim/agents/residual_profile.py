from __future__ import annotations

import numpy as np
from lenskit.data import ItemList

from sim.agents._history import (
    build_residual_history,
    ensure_item_vectors,
    update_residual_history,
)
from sim.agents.base import AbstractAgent
from sim.persona import AgentPersona


class ResidualProfileAgent(AbstractAgent):
    """Score items from a signed profile built from debiased rating history."""

    def __init__(self, env) -> None:
        self.env = env
        self._history, self._item_vectors = build_residual_history(
            env, user_ids=getattr(env, "eval_users", None)
        )
        self._profiles: dict[int, np.ndarray] = {}
        for user_id in self._history:
            self._profiles[user_id] = self._rebuild_profile(user_id)

    def _rebuild_profile(self, user_id: int) -> np.ndarray:
        dims = self.env.config.user_pref_features
        profile = np.zeros(dims, dtype=np.float64)
        total_weight = 0.0
        user_history = self._history.get(user_id, {})
        ensure_item_vectors(self.env, self._item_vectors, list(user_history))
        for movie_id, residual in user_history.items():
            item_vec = self._item_vectors.get(movie_id)
            if item_vec is None:
                continue
            profile += residual * item_vec
            total_weight += abs(residual)
        if total_weight > 0:
            profile /= total_weight
        return profile

    def evaluate(
        self,
        candidates: ItemList,
        persona: AgentPersona,
        item_factors: dict[int, np.ndarray],
    ) -> ItemList:
        profile = self._profiles.get(persona.user_id)
        if profile is None:
            profile = np.zeros(self.env.config.user_pref_features, dtype=np.float64)
        candidate_ids = [int(iid) for iid in candidates.ids()]
        scores = np.array(
            [
                float(np.dot(profile, item_factors[mid]))
                if mid in item_factors
                else 0.0
                for mid in candidate_ids
            ],
            dtype=np.float32,
        )
        return ItemList(candidates, scores=scores)

    def update(
        self,
        user_id: int,
        interactions: list[tuple[int, str, float]],
    ) -> None:
        user_history = update_residual_history(self._history, user_id, interactions)
        ensure_item_vectors(self.env, self._item_vectors, list(user_history))
        self._profiles[user_id] = self._rebuild_profile(user_id)
