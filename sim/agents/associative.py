"""
sim.agents.associative — Agent 1: Associative (preference-vector) Agent.

Scores each candidate movie by cosine similarity between:
  - the user's preference vector (persona.pref_vector) in the small
    independent MF space (TruncatedSVD, 5-10 dims), and
  - each candidate's item factor vector in the same space.

Both vectors are L2-normalised, so cosine similarity reduces to a dot
product.  This is intentionally separate from the LensKit recommender's
BiasedMF factors: the user's internal model runs in a lower-dimensional
space and evolves through their own interactions rather than from platform-
level signals.

update() is a no-op for this agent because all preference-vector updates
are delegated to persona.update_preference() in the runner.
"""
from __future__ import annotations

import logging

import numpy as np
from lenskit.data import ItemList

from sim.agents.base import AbstractAgent
from sim.environment import Environment
from sim.persona import AgentPersona

logger = logging.getLogger(__name__)


class AssociativeAgent(AbstractAgent):
    """
    Scores candidates by dot product in the user-preference MF space.

    Parameters
    ----------
    env:
        Initialised Environment.
    """

    def __init__(self, env: Environment) -> None:
        self.env = env

    def evaluate(
        self,
        candidates: ItemList,
        persona: AgentPersona,
        item_factors: dict[int, np.ndarray],
    ) -> ItemList:
        """
        Re-rank candidates by dot product with persona.pref_vector.

        Items without a factor receive a score of 0.0.
        """
        user_vec = persona.pref_vector
        candidate_ids = [int(iid) for iid in candidates.ids()]

        scores = np.array(
            [
                float(np.dot(user_vec, item_factors[mid]))
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
        """No-op. Preference updates are handled by persona.update_preference()."""
        pass
