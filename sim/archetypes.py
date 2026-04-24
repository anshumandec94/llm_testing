"""
sim.archetypes — Default archetype configurations.

Each archetype defines the prior distributions from which per-user traits are
sampled at persona initialisation. These are the "factory settings" for
different behavioural types; they can be overridden in ``SimConfig``.

Three default archetypes are provided:

``casual``
    Moderate engagement, broad taste. Low softmax temperature (explores a
    bit), mostly adds-to-list, moderate attention.

``binger``
    High engagement, focused taste. Low softmax temperature (strongly prefers
    top-scored items), frequently watches. Slow attention decay (sustains
    long sessions).

``critic``
    Selective, high standards. Higher score floor (ignores low-quality items),
    rates everything, aggressive attention decay (short focused sessions).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArchetypeConfig:
    """
    Prior distributions for a behavioural archetype.

    Per-user traits are sampled once from these priors at persona
    initialisation and then frozen for the lifetime of the simulation.

    Notation: *_mean / *_std define a Normal(mean, std) prior.
    Values are clipped to reasonable ranges after sampling.
    """

    name: str

    # ── Softmax temperature ────────────────────────────────────────────────
    # τ → 0: near-deterministic top-k selection
    # τ → ∞: near-uniform random sampling
    tau_mean: float = 1.0
    tau_std: float = 0.2

    # ── Score floor ────────────────────────────────────────────────────────
    # Items with preference score below this are excluded from sampling.
    score_floor_mean: float = -0.5
    score_floor_std: float = 0.1

    # ── Action logit weights ───────────────────────────────────────────────
    # P(action | score) = sigmoid(intercept + weight * score)
    # Priority order within act(): watch → rate → add_to_list → ignore.
    action_intercepts: dict = field(default_factory=lambda: {
        "watch": -1.5,
        "rate": -0.5,
        "add_to_list": 0.5,
    })
    action_weights: dict = field(default_factory=lambda: {
        "watch": 2.0,
        "rate": 1.5,
        "add_to_list": 1.0,
    })

    # ── Online preference learning rate ────────────────────────────────────
    lr_mean: float = 0.05
    lr_std: float = 0.01

    # ── Attendance baseline logit ──────────────────────────────────────────
    # Higher = more likely to show up each round.
    baseline_logit_mean: float = 1.0
    baseline_logit_std: float = 0.5

    # ── Attention strategy ─────────────────────────────────────────────────
    attention_strategy: str = "LinearDecay"
    attention_kwargs: dict = field(default_factory=lambda: {
        "decay_rate": 0.15,
        "recovery": "full",
    })

    # ── Attendance strategy ────────────────────────────────────────────────
    attendance_strategy: str = "LogisticAttendance"
    attendance_kwargs: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Default archetype instances
# ──────────────────────────────────────────────────────────────────────────────

CASUAL_ARCHETYPE = ArchetypeConfig(
    name="casual",
    tau_mean=1.0,
    tau_std=0.2,
    score_floor_mean=-0.5,
    score_floor_std=0.1,
    action_intercepts={"watch": -2.0, "rate": -1.0, "add_to_list": 0.5},
    action_weights={"watch": 1.5, "rate": 1.0, "add_to_list": 1.2},
    lr_mean=0.05,
    lr_std=0.01,
    baseline_logit_mean=0.5,
    baseline_logit_std=0.5,
    attention_strategy="LinearDecay",
    attention_kwargs={"decay_rate": 0.12, "recovery": "full"},
    attendance_strategy="LogisticAttendance",
    attendance_kwargs={"beta_sat": 0.3, "recency_bonus": 0.2},
)

BINGER_ARCHETYPE = ArchetypeConfig(
    name="binger",
    tau_mean=0.5,
    tau_std=0.1,
    score_floor_mean=0.0,
    score_floor_std=0.1,
    action_intercepts={"watch": -0.5, "rate": -1.5, "add_to_list": -1.0},
    action_weights={"watch": 2.5, "rate": 1.0, "add_to_list": 0.8},
    lr_mean=0.08,
    lr_std=0.02,
    baseline_logit_mean=1.5,
    baseline_logit_std=0.4,
    attention_strategy="ExponentialDecay",
    attention_kwargs={"decay_rate": 0.03, "recovery": "full"},
    attendance_strategy="LogisticAttendance",
    attendance_kwargs={"beta_sat": 0.5, "recency_bonus": 0.4},
)

CRITIC_ARCHETYPE = ArchetypeConfig(
    name="critic",
    tau_mean=0.8,
    tau_std=0.15,
    score_floor_mean=0.2,
    score_floor_std=0.1,
    action_intercepts={"watch": -2.5, "rate": 0.5, "add_to_list": -0.5},
    action_weights={"watch": 1.5, "rate": 2.0, "add_to_list": 0.5},
    lr_mean=0.03,
    lr_std=0.01,
    baseline_logit_mean=0.8,
    baseline_logit_std=0.6,
    attention_strategy="PerRequest",
    attention_kwargs={"decay_rate": 0.35, "recovery": "partial", "recovery_rate": 0.6},
    attendance_strategy="ThresholdAttendance",
    attendance_kwargs={"threshold": 3.5, "fallback_prob": 0.2},
)

# Convenient lookup by name
ARCHETYPE_REGISTRY: dict[str, ArchetypeConfig] = {
    "casual": CASUAL_ARCHETYPE,
    "binger": BINGER_ARCHETYPE,
    "critic": CRITIC_ARCHETYPE,
}
