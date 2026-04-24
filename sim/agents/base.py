"""
sim.agents.base — AbstractAgent protocol.

Agents are responsible for scoring candidates. All sampling, action
selection, attention, and attendance logic lives on AgentPersona.

The evaluate() method receives the persona so that agent subclasses
which blend preference-space scoring with other signals (e.g. semantic
similarity in SemanticAgent, LLM reasoning in LLMAgent) have access to
the user state they need.

The update() method provides a hook for agents that maintain their own
separate model state beyond the persona's preference vector. For most
agents this will be a no-op — persona.update_preference() handles the
user-side update in the runner.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from lenskit.data import ItemList

from sim.persona import AgentPersona


class AbstractAgent(ABC):
    """
    Shared interface for all simulation agents.
    """

    @abstractmethod
    def evaluate(
        self,
        candidates: ItemList,
        persona: AgentPersona,
        item_factors: dict[int, np.ndarray],
    ) -> ItemList:
        """
        Score and re-rank the candidate item list.

        Parameters
        ----------
        candidates:
            ItemList produced by the recommender.
        persona:
            The AgentPersona for the active user. Provides ``pref_vector``
            and archetype-level config that may inform scoring.
        item_factors:
            Dict mapping movieId → item vector in the user-pref space
            (from ``env.get_user_pref_item_factors()``).

        Returns
        -------
        ItemList
            The same items with a ``score`` field attached so the runner can
            pass them to ``persona.act()``.
        """

    @abstractmethod
    def update(
        self,
        user_id: int,
        interactions: list[tuple[int, str, float]],
    ) -> None:
        """
        Update any agent-level (not persona-level) state.

        For agents without their own model state this is a no-op.
        ``persona.update_preference()`` in the runner handles the user-side
        preference vector update separately.

        Parameters
        ----------
        user_id:
            The user whose interactions are being recorded.
        interactions:
            List of ``(movie_id, action, signal_strength)`` tuples for
            acted-on items this round (action != "ignore").
        """
