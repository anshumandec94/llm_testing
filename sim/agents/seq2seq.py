"""
sim.agents.seq2seq — Agent 3: Sequence-to-Sequence Agent (stub).

This agent will use a trained SASRec (Self-Attentive Sequential Recommendation)
model to score candidates based on the user's rating sequence.

Reference implementation: https://github.com/kang205/SASRec

Status: STUB — evaluate() raises NotImplementedError.
"""
from __future__ import annotations

import numpy as np
from lenskit.data import ItemList

from sim.agents.base import AbstractAgent
from sim.environment import Environment
from sim.persona import AgentPersona


class Seq2SeqAgent(AbstractAgent):
    """
    Sequence-to-sequence recommendation agent — SASRec (stub).

    Will score candidates based on the user's interaction sequence using a
    trained SASRec model.
    """

    def __init__(self, env: Environment, model_path: str | None = None) -> None:
        self.env = env
        self.model_path = model_path

    def evaluate(
        self,
        candidates: ItemList,
        persona: AgentPersona,
        item_factors: dict[int, np.ndarray],
    ) -> ItemList:
        raise NotImplementedError("Seq2SeqAgent is not yet implemented.")

    def update(
        self,
        user_id: int,
        interactions: list[tuple[int, str, float]],
    ) -> None:
        raise NotImplementedError("Seq2SeqAgent is not yet implemented.")
