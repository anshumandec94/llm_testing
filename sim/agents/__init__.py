"""
sim.agents — agent implementations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from sim.agents.base import AbstractAgent

if TYPE_CHECKING:
    from sim.config import SimConfig
    from sim.environment import Environment

AgentFactory = Callable[["SimConfig", "Environment"], AbstractAgent]


def _build_associative(config: SimConfig, env: Environment) -> AbstractAgent:
    from sim.agents.associative import AssociativeAgent

    return AssociativeAgent(env)


def _build_semantic(config: SimConfig, env: Environment) -> AbstractAgent:
    from sim.agents.semantic import SemanticAgent

    return SemanticAgent(env)


def _build_residual_profile(config: SimConfig, env: Environment) -> AbstractAgent:
    from sim.agents.residual_profile import ResidualProfileAgent

    return ResidualProfileAgent(env)


def _build_item_item(config: SimConfig, env: Environment) -> AbstractAgent:
    from sim.agents.item_item import ItemItemNeighborhoodAgent

    return ItemItemNeighborhoodAgent(env)


def _build_seq2seq(config: SimConfig, env: Environment) -> AbstractAgent:
    from sim.agents.seq2seq import Seq2SeqAgent

    return Seq2SeqAgent(env)


def _build_llm(config: SimConfig, env: Environment) -> AbstractAgent:
    from sim.agents.llm import LLMAgent

    return LLMAgent(
        env,
        model_id=config.llm_model_id,
        history_k=config.llm_history_k,
        history_strategy=config.llm_history_strategy,
        max_tokens=config.llm_max_tokens,
        overview_max_chars=config.llm_overview_max_chars,
        use_few_shot=config.llm_use_few_shot,
    )


AGENT_REGISTRY: dict[str, AgentFactory] = {
    "associative": _build_associative,
    "associative_baseline": _build_associative,
    "residual_profile": _build_residual_profile,
    "item_item": _build_item_item,
    "semantic": _build_semantic,
    "seq2seq": _build_seq2seq,
    "llm": _build_llm,
}


def build_agent(config: SimConfig, env: Environment) -> AbstractAgent:
    """Instantiate the configured scoring agent via the registry."""
    try:
        factory = AGENT_REGISTRY[config.agent_type]
    except KeyError as exc:
        raise ValueError(f"Unknown agent_type: {config.agent_type!r}") from exc
    return factory(config, env)


__all__ = ["AbstractAgent", "AGENT_REGISTRY", "build_agent"]
