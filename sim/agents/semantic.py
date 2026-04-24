"""
sim.agents.semantic — Agent 2: Semantic Embedding Agent (stub).

This agent will score candidates by computing the cosine similarity between:
  - a running centroid of the semantic embeddings of the user's historically
    highly-rated movies (>= 4.0 stars), and
  - each candidate's sentence-transformer content embedding.

The centroid is updated each round as the agent accepts new items.

Status: STUB — evaluate() raises NotImplementedError.
"""
from __future__ import annotations

import numpy as np
from lenskit.data import ItemList

from sim.agents.base import AbstractAgent
from sim.environment import Environment
from sim.persona import AgentPersona


class SemanticAgent(AbstractAgent):
    """
    Centroid-based semantic embedding agent (stub).

    Will score candidates by cosine similarity between
    - a running centroid of semantic embeddings of the user's high-rated items
    - each candidate's sentence-transformer content embedding.

    The blending with persona.pref_vector from the small MF space is
    intentionally deferred to implementation phase.
    """

    def __init__(self, env: Environment) -> None:
        self.env = env

    def evaluate(
        self,
        candidates: ItemList,
        persona: AgentPersona,
        item_factors: dict[int, np.ndarray],
    ) -> ItemList:
        raise NotImplementedError("SemanticAgent is not yet implemented.")

    def update(
        self,
        user_id: int,
        interactions: list[tuple[int, str, float]],
    ) -> None:
        raise NotImplementedError("SemanticAgent is not yet implemented.")
