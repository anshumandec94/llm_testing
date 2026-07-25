"""Population assignment helpers for agent-composition experiments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UserAssignment:
    """One simulated user instance and the base user history it represents."""

    sim_user_id: int
    base_user_id: int
    agent_type: str


def resolve_agent_types(config) -> list[str]:
    """Return the configured agent types, falling back to the legacy field."""
    if config.agent_types:
        return list(config.agent_types)
    return [config.agent_type]


def resolve_agent_type_proportions(config, agent_types: list[str]) -> list[float]:
    """Return normalized proportions aligned with ``agent_types``."""
    if len(agent_types) == 1:
        return [1.0]

    if config.agent_type_proportions:
        if len(config.agent_type_proportions) != len(agent_types):
            raise ValueError(
                "agent_type_proportions must match agent_types length."
            )
        proportions = np.array(config.agent_type_proportions, dtype=float)
    else:
        proportions = np.full(len(agent_types), 1.0 / len(agent_types), dtype=float)

    if np.any(proportions < 0):
        raise ValueError("agent_type_proportions cannot contain negative values.")
    total = float(proportions.sum())
    if total <= 0:
        raise ValueError("agent_type_proportions must sum to a positive value.")
    return (proportions / total).tolist()


def build_user_assignments(config, env, rng: np.random.Generator) -> list[UserAssignment]:
    """Build simulated user assignments for the configured experiment mode."""
    agent_types = resolve_agent_types(config)
    proportions = resolve_agent_type_proportions(config, agent_types)

    if (
        len(agent_types) > 1
        and config.agent_assignment_mode == "one_per_agent_type"
    ):
        assignments: list[UserAssignment] = []
        for base_user_id in env.eval_users:
            for index, agent_type in enumerate(agent_types):
                sim_user_id = -((int(base_user_id) * 1000) + index + 1)
                assignments.append(
                    UserAssignment(
                        sim_user_id=sim_user_id,
                        base_user_id=int(base_user_id),
                        agent_type=agent_type,
                    )
                )
        return assignments

    sampled_agent_types = rng.choice(agent_types, size=len(env.eval_users), p=proportions)
    return [
        UserAssignment(
            sim_user_id=int(base_user_id),
            base_user_id=int(base_user_id),
            agent_type=str(agent_type),
        )
        for base_user_id, agent_type in zip(env.eval_users, sampled_agent_types)
    ]


def assignment_metadata(config, assignments: list[UserAssignment]) -> dict[str, str]:
    """Return composition metadata suitable for MLflow tags."""
    counts: dict[str, int] = {}
    for assignment in assignments:
        counts[assignment.agent_type] = counts.get(assignment.agent_type, 0) + 1

    return {
        "agent_assignment_mode": config.agent_assignment_mode,
        "agent_types": ",".join(resolve_agent_types(config)),
        "agent_type_proportions": ",".join(
            str(value) for value in resolve_agent_type_proportions(config, resolve_agent_types(config))
        ),
        "realized_agent_counts": ",".join(
            f"{agent_type}:{counts[agent_type]}"
            for agent_type in sorted(counts)
        ),
    }
