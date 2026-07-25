"""
sim.user_agent — SimulatedUser: unified per-user interface for the simulation.

Wraps an AgentPersona (behavioral/preference model) and a shared scoring
Agent together, presenting a single coherent interface that the runner uses
to represent one user in the simulation:

    attended = ua.will_attend(rng)
    recs_rows, interactions = ua.step(candidates, item_factors, held_ids, rnd, req, cfg, rng)
    ua.update(rnd, interactions, acted_item_factors, cfg)

Separation of concerns
----------------------
The runner is responsible for:
  - Calling the recommender to generate and mark candidates.
  - Fetching item factors from the environment.
  - Updating the recommender with feedback signals.

SimulatedUser is responsible for:
  - Deciding whether to attend (attendance model).
  - Scoring candidates and sampling actions (agent + persona).
  - Depleting and restoring the attention budget.
  - Updating internal preference and agent memory state.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from lenskit.data import ItemList
from sim.agents import build_agent
from sim.population import UserAssignment, build_user_assignments

if TYPE_CHECKING:
    from sim.config import SimConfig
    from sim.environment import Environment
    from sim.persona import AgentPersona

logger = logging.getLogger(__name__)


class SimulatedUser:
    """Unified per-user interface wrapping a persona and a shared scoring agent.

    The persona and agent are both aspects of "the simulated user": the agent
    is the scoring/ranking model, the persona is the behavioral/preference
    model. This class orchestrates them as a single entity so the runner only
    needs to know about what a user *does*, not how they do it.

    Parameters
    ----------
    uid:
        User ID this instance represents.
    persona:
        The user's AgentPersona (behavioral traits + mutable session state).
    agent:
        Shared scoring agent (AssociativeAgent, SemanticAgent, etc.).
        The agent instance is shared across all users; uid identifies this
        user to it during update calls.
    """

    def __init__(
        self,
        uid: int,
        base_user_id: int,
        agent_type: str,
        persona: AgentPersona,
        agent,
    ) -> None:
        self.uid = uid
        self.base_user_id = base_user_id
        self.agent_type = agent_type
        self.persona = persona
        self._agent = agent

    @classmethod
    def build_population(
        cls,
        config: SimConfig,
        env: Environment,
        rng,
        assignments: list[UserAssignment] | None = None,
    ) -> tuple[dict[int, SimulatedUser], list[UserAssignment]]:
        """Build the full population of SimulatedUsers for a simulation run.

        Creates one shared scoring agent per configured agent type, then wraps
        each assigned persona in a SimulatedUser.

        Parameters
        ----------
        config:
            Simulation configuration (determines agent_type and archetype mix).
        env:
            Initialised Environment.
        rng:
            NumPy random generator used for persona sampling.

        Returns
        -------
        A pair of (population, assignments), where population maps simulated
        user ID → SimulatedUser.
        """
        from dataclasses import replace

        from sim.persona import build_population_for_assignments

        if assignments is None:
            assignments = build_user_assignments(config, env, rng)
        agent_types = sorted({assignment.agent_type for assignment in assignments})
        agents = {
            agent_type: build_agent(replace(config, agent_type=agent_type), env)
            for agent_type in agent_types
        }

        logger.info("=== Building Agent Population ===")
        population = build_population_for_assignments(config, env, assignments, rng)

        return (
            {
                assignment.sim_user_id: cls(
                    assignment.sim_user_id,
                    assignment.base_user_id,
                    assignment.agent_type,
                    population[assignment.sim_user_id],
                    agents[assignment.agent_type],
                )
                for assignment in assignments
            },
            assignments,
        )

    @property
    def agent(self):
        return self._agent

    @property
    def budget(self) -> float:
        return self.persona.budget

    # ──────────────────────────────────────────────────────────────────────
    # Attendance
    # ──────────────────────────────────────────────────────────────────────

    def will_attend(self, rng) -> bool:
        """Sample attendance for this round.

        Increments ``rounds_since_last_visit`` on absence so the persona's
        attendance model sees the correct gap on the next call.

        Returns True if the user attends, False otherwise.
        """
        attended = self.persona.attendance.will_attend(
            baseline_logit=self.persona.baseline_logit,
            recent_signal_ewma=self.persona.recent_signal_ewma,
            rounds_since_last_visit=self.persona.rounds_since_last_visit,
            rng=rng,
        )
        if not attended:
            self.persona.rounds_since_last_visit += 1
        return attended

    # ──────────────────────────────────────────────────────────────────────
    # Perceive and act (one recommendation request)
    # ──────────────────────────────────────────────────────────────────────

    def step(
        self,
        candidates: ItemList,
        item_factors: dict[int, np.ndarray],
        held_ids: set[int],
        rnd: int,
        req: int,
        cfg: SimConfig,
        rng,
    ) -> tuple[list[dict], list[tuple[int, str, float]]]:
        """Evaluate candidates and sample actions for one recommendation request.

        The agent scores candidates, the persona samples actions, and the
        attention budget is depleted. Rows are annotated with interaction type
        and signal inline so the parquet output is immediately interpretable.

        Parameters
        ----------
        candidates:
            ItemList of candidate items from the recommender.
        item_factors:
            Item factor vectors in the user-preference MF space, keyed by id.
        held_ids:
            The user's held-out item set, used for ``is_held_out`` annotation.
        rnd:
            Current round number (for row annotation).
        req:
            Zero-based request index within the round (for row annotation).
        cfg:
            Simulation configuration.
        rng:
            NumPy random generator.

        Returns
        -------
        recs_rows:
            One dict per candidate with keys: round, request, userId, movieId,
            rank, is_held_out, interaction, signal.
        new_interactions:
            List of (movieId, action, signal) for acted-on items only.
        """
        p = self.persona

        ranked = self._agent.evaluate(candidates, p, item_factors)
        ranked_ids = [int(iid) for iid in ranked.ids()]
        scores_arr = ranked.scores()
        if scores_arr is None:
            scores_arr = np.zeros(len(ranked_ids), dtype=np.float32)

        new_interactions = p.act(
            ranked_ids=ranked_ids,
            scores=scores_arr,
            item_factors=item_factors,
            config=cfg,
            rng=rng,
        )
        p.budget = p.attention.deplete(len(candidates), p.budget)

        acted_lookup = {mid: (act, sig) for mid, act, sig in new_interactions}
        recs_rows = []
        for rank, iid in enumerate(ranked_ids):
            row = {
                "round": rnd,
                "request": req + 1,
                "userId": self.base_user_id,
                "simulation_user_id": self.uid,
                "agent_type": self.agent_type,
                "movieId": iid,
                "rank": rank + 1,
                "is_held_out": iid in held_ids,
                "interaction": None,
                "signal": None,
            }
            if iid in acted_lookup:
                row["interaction"], row["signal"] = acted_lookup[iid]
            recs_rows.append(row)

        return recs_rows, new_interactions

    # ──────────────────────────────────────────────────────────────────────
    # State update (end of round)
    # ──────────────────────────────────────────────────────────────────────

    def update(
        self,
        rnd: int,
        interactions: list[tuple[int, str, float]],
        acted_item_factors: dict[int, np.ndarray],
        cfg: SimConfig,
    ) -> None:
        """Update internal state after a round's interactions.

        Covers: preference vector, agent memory, satisfaction EWMA, attendance
        counter reset, and attention budget recovery.

        The recommender is NOT updated here — that remains the runner's
        responsibility since the recommender is a platform-level component.

        Parameters
        ----------
        rnd:
            Current round number.
        interactions:
            List of explicit-rating feedback tuples for learning updates.
            The signal value is the debiased rating residual, not the raw
            action signal.
        acted_item_factors:
            Item factor vectors for the interacted-with items. Pre-fetched by
            the caller (runner has the env reference).
        cfg:
            Simulation configuration.
        """
        p = self.persona

        if interactions:
            p.update_preference(interactions, acted_item_factors)

        self._agent.update(self.uid, interactions)

        mean_sig = (
            float(np.mean([sig for _, _, sig in interactions]))
            if interactions
            else 0.0
        )
        p.recent_signal_ewma = p.attendance.update_ewma(
            p.recent_signal_ewma, mean_sig, cfg.sat_ewma_alpha
        )
        p.rounds_since_last_visit = 0
        p.last_attended_round = rnd
        p.budget = p.attention.restore(p.budget, mean_sig)
