"""
sim.attendance — Attendance strategy implementations.

An AttendanceStrategy decides, each round, whether a user visits the
recommender system. Like AttentionStrategy, it is stateless — all mutable
state lives on the AgentPersona.

Usage pattern (inside runner.py):
    attended = persona.attendance.will_attend(
        baseline_logit=persona.baseline_logit,
        recent_signal_ewma=persona.recent_signal_ewma,
        rounds_since_last_visit=persona.rounds_since_last_visit,
        rng=rng,
    )
    # After the round:
    mean_sig = ...
    persona.recent_signal_ewma = persona.attendance.update_ewma(
        persona.recent_signal_ewma, mean_sig, alpha
    )
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AttendanceStrategy(ABC):
    """
    Stateless protocol for attendance probability mechanics.
    """

    @abstractmethod
    def will_attend(
        self,
        baseline_logit: float,
        recent_signal_ewma: float,
        rounds_since_last_visit: int,
        rng: np.random.Generator,
    ) -> bool:
        """
        Return True if the user will visit the recommender this round.

        Parameters
        ----------
        baseline_logit:
            Per-user intercept sampled from archetype prior at persona init.
            Encodes individual engagement propensity.
        recent_signal_ewma:
            Exponentially-weighted moving average of the user's recent
            satisfaction signals (mean interaction signal strength).
        rounds_since_last_visit:
            Number of consecutive rounds this user has been absent. 0 if
            they attended the previous round.
        rng:
            Random generator (from the runner) for reproducible draws.
        """

    @abstractmethod
    def update_ewma(
        self,
        current_ewma: float,
        new_signal: float,
        alpha: float,
    ) -> float:
        """
        Update and return the satisfaction EWMA.

        Parameters
        ----------
        current_ewma:
            Previous EWMA value.
        new_signal:
            Mean signal strength this round (0 if user was absent).
        alpha:
            Smoothing factor in (0, 1). Higher → more weight on recent signal.
        """


# ──────────────────────────────────────────────────────────────────────────────
# Built-in implementations
# ──────────────────────────────────────────────────────────────────────────────


class LogisticAttendance(AttendanceStrategy):
    """
    Attendance probability computed via a logistic function of four components:

        logit = baseline + β_sat * ewma + recency_component + Gumbel(0, noise_scale)
        P(attend) = σ(logit)

    where ``recency_component`` is a bonus for visiting last round or a
    growing penalty for long absences.

    This is the most behaviourally expressive model and the recommended
    default for studying emergent dynamics.

    Parameters
    ----------
    beta_sat:
        Coefficient on the satisfaction EWMA. Higher → more sensitive to
        content quality.
    recency_bonus:
        Added to logit when the user attended the previous round.
    recency_penalty:
        Subtracted per-absent-round (compounding) for users who have been
        away for more than one round.
    noise_scale:
        Scale of the per-round Gumbel noise. Ensures non-deterministic
        attendance even for users with extreme baseline logits.
    """

    def __init__(
        self,
        beta_sat: float = 0.3,
        recency_bonus: float = 0.3,
        recency_penalty: float = 0.1,
        noise_scale: float = 0.2,
    ) -> None:
        self.beta_sat = beta_sat
        self.recency_bonus = recency_bonus
        self.recency_penalty = recency_penalty
        self.noise_scale = noise_scale

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def will_attend(
        self,
        baseline_logit: float,
        recent_signal_ewma: float,
        rounds_since_last_visit: int,
        rng: np.random.Generator,
    ) -> bool:
        recency = (
            self.recency_bonus
            if rounds_since_last_visit == 0
            else -self.recency_penalty * rounds_since_last_visit
        )
        # Gumbel noise: -log(-log(u)) for u ~ Uniform(0,1)
        u = float(rng.uniform())
        u = max(1e-10, min(1 - 1e-10, u))
        gumbel = -np.log(-np.log(u)) * self.noise_scale

        logit = (
            baseline_logit
            + self.beta_sat * recent_signal_ewma
            + recency
            + gumbel
        )
        prob = self._sigmoid(logit)
        return bool(rng.random() < prob)

    def update_ewma(self, current_ewma: float, new_signal: float, alpha: float) -> float:
        return alpha * new_signal + (1 - alpha) * current_ewma


class ThresholdAttendance(AttendanceStrategy):
    """
    Simple rule-based attendance: attends if EWMA satisfaction exceeds a
    threshold; otherwise attends with a fixed fallback probability.

    Useful as a crisp behavioural model for ablation studies.

    Parameters
    ----------
    threshold:
        Satisfaction EWMA level (on [0, 5] scale) above which the user
        always attends.
    fallback_prob:
        Attendance probability when EWMA is below threshold.
    """

    def __init__(self, threshold: float = 3.0, fallback_prob: float = 0.3) -> None:
        self.threshold = threshold
        self.fallback_prob = fallback_prob

    def will_attend(
        self,
        baseline_logit: float,
        recent_signal_ewma: float,
        rounds_since_last_visit: int,
        rng: np.random.Generator,
    ) -> bool:
        if recent_signal_ewma >= self.threshold:
            return True
        return bool(rng.random() < self.fallback_prob)

    def update_ewma(self, current_ewma: float, new_signal: float, alpha: float) -> float:
        return alpha * new_signal + (1 - alpha) * current_ewma


class AlwaysAttend(AttendanceStrategy):
    """
    User always attends every round. Useful as a baseline / ablation to
    isolate the effect of attendance variation from other dynamics.
    """

    def will_attend(
        self,
        baseline_logit: float,
        recent_signal_ewma: float,
        rounds_since_last_visit: int,
        rng: np.random.Generator,
    ) -> bool:
        return True

    def update_ewma(self, current_ewma: float, new_signal: float, alpha: float) -> float:
        return alpha * new_signal + (1 - alpha) * current_ewma


class BernoulliAttendance(AttendanceStrategy):
    """
    Independent Bernoulli draw each round with fixed ``attend_prob``.

    Attendance is stateless — no memory of history, no sensitivity to
    satisfaction. Useful for studying the effect of attendance rate alone.

    Parameters
    ----------
    attend_prob:
        Fixed probability of attending any given round, in [0, 1].
    """

    def __init__(self, attend_prob: float = 0.7) -> None:
        self.attend_prob = attend_prob

    def will_attend(
        self,
        baseline_logit: float,
        recent_signal_ewma: float,
        rounds_since_last_visit: int,
        rng: np.random.Generator,
    ) -> bool:
        return bool(rng.random() < self.attend_prob)

    def update_ewma(self, current_ewma: float, new_signal: float, alpha: float) -> float:
        return alpha * new_signal + (1 - alpha) * current_ewma


# ──────────────────────────────────────────────────────────────────────────────
# Registry: maps string names to constructors (used by build_persona)
# ──────────────────────────────────────────────────────────────────────────────

ATTENDANCE_REGISTRY: dict[str, type[AttendanceStrategy]] = {
    "LogisticAttendance": LogisticAttendance,
    "ThresholdAttendance": ThresholdAttendance,
    "AlwaysAttend": AlwaysAttend,
    "BernoulliAttendance": BernoulliAttendance,
}
