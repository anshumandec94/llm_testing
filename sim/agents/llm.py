"""
sim.agents.llm — Agent 4: LLM + Memory Retrieval Agent (stub).

This agent will use a local LLM (via mlx-lm on Apple Silicon, or a remote
API) to reason about candidate items. Relevant memories (past liked/disliked
movies with their descriptions) are retrieved from ChromaDB and injected into
the prompt as context.

Status: STUB — evaluate() raises NotImplementedError.
"""
from __future__ import annotations

import numpy as np
from lenskit.data import ItemList

from sim.agents.base import AbstractAgent
from sim.environment import Environment
from sim.persona import AgentPersona


class LLMAgent(AbstractAgent):
    """
    LLM-based evaluation agent with ChromaDB memory retrieval (stub).

    Will use a local LLM (mlx-lm on Apple Silicon) to reason about candidate
    items using a context window of the user's interaction history retrieved
    from ChromaDB.
    """

    def __init__(self, env: Environment, model_id: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit") -> None:
        self.env = env
        self.model_id = model_id

    def evaluate(
        self,
        candidates: ItemList,
        persona: AgentPersona,
        item_factors: dict[int, np.ndarray],
    ) -> ItemList:
        raise NotImplementedError("LLMAgent is not yet implemented.")

    def update(
        self,
        user_id: int,
        interactions: list[tuple[int, str, float]],
    ) -> None:
        raise NotImplementedError("LLMAgent is not yet implemented.")
