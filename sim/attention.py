"""
sim.attention — Attention budget strategy implementations.

An AttentionStrategy is a stateless object that computes budget transitions.
The budget value itself lives on the AgentPersona so strategies are
serialisable independently of agent state.

Usage pattern (inside runner.py):
    k_eff  = persona.attention.effective_k(len(candidates), persona.budget)
    persona.budget = persona.attention.deplete(len(candidates), persona.budget)
    ...
    # at end of round:
    persona.budget = persona.attention.restore(persona.budget, mean_signal)

All methods clamp the returned budget to [0.0, 1.0].
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class AttentionStrategy(ABC):
    """
    Stateless protocol for attention budget mechanics.

    Parameters are set at construction time and remain fixed. All methods
    are pure functions of the inputs — they do not hold mutable state.
    """

    @abstractmethod
    def effective_k(self, list_size: int, current_budget: float) -> int:
        """
        Number of items the agent will actively evaluate from this batch.

        Parameters
        ----------
        list_size:
            Number of candidates in the current recommendation batch.
        current_budget:
            Agent's attention budget before depletion, in [0, 1].

        Returns
        -------
        int
            Always in [1, list_size] if budget > 0, else 0.
        """

    @abstractmethod
    def deplete(self, list_size: int, current_budget: float) -> float:
        """
        Return the new budget after processing ``list_size`` items.

        Result is clamped to [0.0, 1.0].
        """

    @abstractmethod
    def restore(self, end_budget: float, satisfaction_signal: float) -> float:
        """
        Return the budget at the start of the *next* round.

        Parameters
        ----------
        end_budget:
            Budget remaining at the end of the current round.
        satisfaction_signal:
            Mean signal strength of acted-on items this round, in [0, 5].
            Used by satisfaction-driven recovery models.

        Result is clamped to [0.0, 1.0].
        """


# ──────────────────────────────────────────────────────────────────────────────
# Built-in implementations
# ──────────────────────────────────────────────────────────────────────────────


class LinearDecayAttention(AttentionStrategy):
    """
    Budget depletes linearly with the number of items processed:
        new_budget = budget - decay_rate * list_size

    Parameters
    ----------
    decay_rate:
        Budget cost per item (default 0.15 means ~6 full batches of 1 item
        before budget exhaustion; scaled by list_size in practice).
    recovery:
        One of ``"full"`` | ``"partial"`` | ``"satisfaction"``.
    recovery_rate:
        Fraction of budget recovered per round (used by ``"partial"`` and
        ``"satisfaction"`` when below threshold).
    sat_threshold:
        Satisfaction signal threshold above which ``"satisfaction"`` recovery
        grants a full restore. Interpreted on [0, 5] scale.
    """

    def __init__(
        self,
        decay_rate: float = 0.15,
        recovery: str = "full",
        recovery_rate: float = 0.5,
        sat_threshold: float = 3.5,
    ) -> None:
        self.decay_rate = decay_rate
        self.recovery = recovery
        self.recovery_rate = recovery_rate
        self.sat_threshold = sat_threshold

    def effective_k(self, list_size: int, current_budget: float) -> int:
        if current_budget <= 0:
            return 0
        return max(1, round(current_budget * list_size))

    def deplete(self, list_size: int, current_budget: float) -> float:
        new_budget = current_budget - self.decay_rate * list_size
        return float(max(0.0, min(1.0, new_budget)))

    def restore(self, end_budget: float, satisfaction_signal: float) -> float:
        if self.recovery == "full":
            return 1.0
        if self.recovery == "satisfaction":
            if satisfaction_signal >= self.sat_threshold:
                return 1.0
            return float(min(1.0, end_budget + self.recovery_rate))
        # "partial"
        return float(min(1.0, end_budget + self.recovery_rate))


class ExponentialDecayAttention(AttentionStrategy):
    """
    Budget depletes exponentially, modelling diminishing marginal engagement:
        new_budget = budget * exp(-decay_rate * list_size)

    Slower to exhaust than linear for small lists, but never fully resets to 0.

    Parameters
    ----------
    decay_rate:
        Exponential rate constant. Higher → faster depletion per item.
    recovery:
        One of ``"full"`` | ``"partial"`` | ``"satisfaction"``.
    recovery_rate, sat_threshold:
        Same semantics as ``LinearDecayAttention``.
    """

    def __init__(
        self,
        decay_rate: float = 0.05,
        recovery: str = "full",
        recovery_rate: float = 0.5,
        sat_threshold: float = 3.5,
    ) -> None:
        import math
        self._math = math
        self.decay_rate = decay_rate
        self.recovery = recovery
        self.recovery_rate = recovery_rate
        self.sat_threshold = sat_threshold

    def effective_k(self, list_size: int, current_budget: float) -> int:
        if current_budget <= 0:
            return 0
        return max(1, round(current_budget * list_size))

    def deplete(self, list_size: int, current_budget: float) -> float:
        new_budget = current_budget * self._math.exp(-self.decay_rate * list_size)
        return float(max(0.0, min(1.0, new_budget)))

    def restore(self, end_budget: float, satisfaction_signal: float) -> float:
        if self.recovery == "full":
            return 1.0
        if self.recovery == "satisfaction":
            if satisfaction_signal >= self.sat_threshold:
                return 1.0
            return float(min(1.0, end_budget + self.recovery_rate))
        return float(min(1.0, end_budget + self.recovery_rate))


class PerRequestAttention(AttentionStrategy):
    """
    Budget depletes by a flat amount per request, regardless of list size.
    Models the idea that the cognitive cost is in *deciding to request*, not
    in reading each item in the list.
        new_budget = budget - decay_rate

    Parameters
    ----------
    decay_rate:
        Budget cost per request (default 0.3 → ~3 requests before exhaustion).
    recovery, recovery_rate, sat_threshold:
        Same semantics as ``LinearDecayAttention``.
    """

    def __init__(
        self,
        decay_rate: float = 0.3,
        recovery: str = "full",
        recovery_rate: float = 0.5,
        sat_threshold: float = 3.5,
    ) -> None:
        self.decay_rate = decay_rate
        self.recovery = recovery
        self.recovery_rate = recovery_rate
        self.sat_threshold = sat_threshold

    def effective_k(self, list_size: int, current_budget: float) -> int:
        if current_budget <= 0:
            return 0
        return max(1, round(current_budget * list_size))

    def deplete(self, list_size: int, current_budget: float) -> float:
        new_budget = current_budget - self.decay_rate
        return float(max(0.0, min(1.0, new_budget)))

    def restore(self, end_budget: float, satisfaction_signal: float) -> float:
        if self.recovery == "full":
            return 1.0
        if self.recovery == "satisfaction":
            if satisfaction_signal >= self.sat_threshold:
                return 1.0
            return float(min(1.0, end_budget + self.recovery_rate))
        return float(min(1.0, end_budget + self.recovery_rate))


# ──────────────────────────────────────────────────────────────────────────────
# Registry: maps string names to constructors (used by build_persona)
# ──────────────────────────────────────────────────────────────────────────────

ATTENTION_REGISTRY: dict[str, type[AttentionStrategy]] = {
    "LinearDecay": LinearDecayAttention,
    "ExponentialDecay": ExponentialDecayAttention,
    "PerRequest": PerRequestAttention,
}
