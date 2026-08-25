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


class ItemItemNeighborhoodAgent(AbstractAgent):
    """Score candidates by similarity to signed liked/disliked history."""

    def __init__(self, env) -> None:
        self.env = env
        self._history, self._item_vectors = build_residual_history(
            env, user_ids=getattr(env, "eval_users", None)
        )

    def evaluate(
        self,
        candidates: ItemList,
        persona: AgentPersona,
        item_factors: dict[int, np.ndarray],
    ) -> ItemList:
        user_history = self._history.get(persona.user_id, {})
        if not user_history:
            return ItemList(
                candidates,
                scores=np.zeros(len(candidates), dtype=np.float32),
            )

        ensure_item_vectors(self.env, self._item_vectors, list(user_history))
        candidate_ids = [int(iid) for iid in candidates.ids()]
        scores: list[float] = []
        for movie_id in candidate_ids:
            candidate_vec = item_factors.get(movie_id)
            if candidate_vec is None:
                scores.append(0.0)
                continue
            numerator = 0.0
            denominator = 0.0
            for hist_movie_id, residual in user_history.items():
                hist_vec = self._item_vectors.get(hist_movie_id)
                if hist_vec is None:
                    continue
                numerator += residual * float(np.dot(candidate_vec, hist_vec))
                denominator += abs(residual)
            scores.append(numerator / denominator if denominator > 0 else 0.0)
        return ItemList(candidates, scores=np.array(scores, dtype=np.float32))

    def update(
        self,
        user_id: int,
        interactions: list[tuple[int, str, float]],
    ) -> None:
        user_history = update_residual_history(self._history, user_id, interactions)
        ensure_item_vectors(self.env, self._item_vectors, list(user_history))
