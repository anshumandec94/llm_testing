"""
sim.persona — AgentPersona dataclass and population factory.

Each eval user is represented by exactly one AgentPersona instance (1:1
mapping). The persona holds:
  - fixed behavioural traits sampled from archetype priors at init
  - the user's evolving preference vector in the small-MF space
  - mutable session state (budget, EWMA, attendance counters)
  - concrete AttentionStrategy and AttendanceStrategy objects

Population construction
-----------------------
``build_population()`` is the main entry point. It assigns archetypes to users
according to ``SimConfig.archetype_mix``, then calls ``build_persona()`` for
each user.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from sim.archetypes import ARCHETYPE_REGISTRY, ArchetypeConfig
from sim.attention import ATTENTION_REGISTRY, AttentionStrategy
from sim.attendance import ATTENDANCE_REGISTRY, AttendanceStrategy

if TYPE_CHECKING:
    from sim.config import SimConfig
    from sim.environment import Environment

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# AgentPersona
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AgentPersona:
    """
    Per-user state and behavioural configuration.

    Fixed traits
    ------------
    Sampled once from archetype priors at ``build_persona()`` and frozen.
    These encode *who this user is* — their individual quirks within the
    archetype they were assigned.

    Evolving state
    --------------
    ``pref_vector`` drifts via gradient steps in ``update_preference()``.
    ``budget``, ``recent_signal_ewma``, ``rounds_since_last_visit``, and
    ``last_attended_round`` are updated by the runner each round.
    """

    # ── Identity ───────────────────────────────────────────────────────────
    user_id: int
    archetype: str

    # ── Fixed behavioural traits ───────────────────────────────────────────
    tau: float
    score_floor: float
    action_intercepts: dict   # {"watch": float, "rate": float, "add_to_list": float}
    action_weights: dict      # same keys
    lr: float                 # preference vector learning rate
    baseline_logit: float     # attendance propensity intercept

    # ── Injected strategy objects ──────────────────────────────────────────
    attention: AttentionStrategy
    attendance: AttendanceStrategy

    # ── Evolving preference state ──────────────────────────────────────────
    pref_vector: np.ndarray   # unit-norm, shape (user_pref_features,)

    # ── Mutable session state ──────────────────────────────────────────────
    budget: float = 1.0
    recent_signal_ewma: float = 0.0
    rounds_since_last_visit: int = 0
    last_attended_round: int = 0

    # ──────────────────────────────────────────────────────────────────────
    # Action model
    # ──────────────────────────────────────────────────────────────────────

    def act(
        self,
        ranked_ids: list[int],
        scores: np.ndarray,
        item_factors: dict[int, np.ndarray],
        config,  # SimConfig — avoid circular import with string type hint
        rng: np.random.Generator,
    ) -> list[tuple[int, str, float]]:
        """
        Sample items from the scored candidate list and select an action per
        sampled item.

        Parameters
        ----------
        ranked_ids:
            Ordered movie IDs from ``agent.evaluate()``.
        scores:
            Preference scores aligned with ``ranked_ids``.
        item_factors:
            Dict mapping movieId → user-pref-space item vector.
        config:
            SimConfig (for watch_signal, add_to_list_signal, beta params).
        rng:
            Runner-level random generator.

        Returns
        -------
        list of (movie_id, action, signal_strength)
            Only items where action is not "ignore" are included.
        """
        if self.budget <= 0 or len(ranked_ids) == 0:
            return []

        # ── 1. Filter by score floor ───────────────────────────────────────
        eligible_mask = scores >= self.score_floor
        eligible_ids = [mid for mid, ok in zip(ranked_ids, eligible_mask) if ok]
        eligible_scores = scores[eligible_mask]

        if len(eligible_ids) == 0:
            return []

        # ── 2. Softmax sampling ────────────────────────────────────────────
        k_eff = self.attention.effective_k(len(eligible_ids), self.budget)
        k_eff = min(k_eff, len(eligible_ids))
        if k_eff == 0:
            return []

        logits = eligible_scores / max(self.tau, 1e-6)
        logits -= logits.max()  # numerical stability
        probs = np.exp(logits)
        probs /= probs.sum()

        chosen_indices = rng.choice(
            len(eligible_ids), size=k_eff, replace=False, p=probs
        )

        # ── 3. Action selection per chosen item ────────────────────────────
        interactions: list[tuple[int, str, float]] = []

        for idx in chosen_indices:
            movie_id = eligible_ids[idx]
            score = float(eligible_scores[idx])

            action, signal = self._select_action(score, config, rng)
            if action != "ignore":
                interactions.append((movie_id, action, signal))

        return interactions

    def _select_action(
        self, score: float, config, rng: np.random.Generator
    ) -> tuple[str, float]:
        """
        Draw an action from the logistic action model conditioned on score.
        Priority: watch → rate → add_to_list → ignore.
        """
        for action in ("watch", "rate", "add_to_list"):
            b = self.action_intercepts.get(action, 0.0)
            w = self.action_weights.get(action, 1.0)
            prob = _sigmoid(b + w * score)
            if rng.random() < prob:
                signal = self._signal_for(action, score, config, rng)
                return action, signal

        return "ignore", 0.0

    def _signal_for(
        self, action: str, score: float, config, rng: np.random.Generator
    ) -> float:
        """Map action + score to a recommendation signal in [0, 5]."""
        if action == "watch":
            return float(config.watch_signal)
        if action == "add_to_list":
            return float(config.add_to_list_signal)
        # "rate": Beta-sampled signal in [1, 5]
        alpha = max(0.1, score * config.beta_alpha_max)
        beta = max(0.1, (1.0 - score) * config.beta_beta_max)
        raw = float(rng.beta(alpha, beta))  # in [0, 1]
        return 1.0 + 4.0 * raw             # rescale to [1, 5]

    # ──────────────────────────────────────────────────────────────────────
    # Preference update
    # ──────────────────────────────────────────────────────────────────────

    def update_preference(
        self,
        interactions: list[tuple[int, str, float]],
        item_factors: dict[int, np.ndarray],
    ) -> None:
        """
        Update preference vector via a gradient step toward each acted-on
        item's embedding, weighted by signal strength. Renormalises to unit
        length after each update.

        Parameters
        ----------
        interactions:
            List of ``(movie_id, action, signal_strength)`` tuples. Only
            acted-on items (action != "ignore") should be passed here.
        item_factors:
            Dict mapping movieId → item vector in user-pref space.
        """
        for movie_id, _action, signal in interactions:
            if movie_id not in item_factors:
                continue
            item_vec = item_factors[movie_id]
            # Normalise signal to [0, 1] for the update step
            norm_signal = signal / 5.0
            self.pref_vector = self.pref_vector + self.lr * norm_signal * item_vec

        norm = np.linalg.norm(self.pref_vector)
        if norm > 0:
            self.pref_vector /= norm


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(x)))


# ──────────────────────────────────────────────────────────────────────────────
# Factory functions
# ──────────────────────────────────────────────────────────────────────────────


def build_persona(
    user_id: int,
    archetype_cfg: ArchetypeConfig,
    env: "Environment",
    rng: np.random.Generator,
) -> AgentPersona:
    """
    Construct an AgentPersona for one user by sampling traits from the
    archetype's prior distributions.

    The preference vector is initialised from the user's SVD-derived factor
    in the small independent MF. If the user has no factor (cold-start), the
    centroid-with-noise fallback from Environment is used.
    """
    # Sample fixed traits
    tau = float(np.clip(rng.normal(archetype_cfg.tau_mean, archetype_cfg.tau_std), 0.1, 5.0))
    score_floor = float(rng.normal(archetype_cfg.score_floor_mean, archetype_cfg.score_floor_std))
    lr = float(np.clip(rng.normal(archetype_cfg.lr_mean, archetype_cfg.lr_std), 1e-4, 1.0))
    baseline_logit = float(rng.normal(archetype_cfg.baseline_logit_mean, archetype_cfg.baseline_logit_std))

    # Copy dicts (avoid shared mutable defaults across personas)
    action_intercepts = dict(archetype_cfg.action_intercepts)
    action_weights = dict(archetype_cfg.action_weights)

    # Preference vector: from small MF or cold-start fallback
    pref_vector = env.get_user_pref_factor(user_id)
    if pref_vector is None:
        dims = env.config.user_pref_features
        pref_vector = rng.normal(0, 1, size=dims).astype(np.float32)
    pref_vector = pref_vector.copy().astype(np.float64)
    norm = np.linalg.norm(pref_vector)
    if norm > 0:
        pref_vector /= norm

    # Instantiate strategy objects
    attention_cls = ATTENTION_REGISTRY.get(archetype_cfg.attention_strategy)
    if attention_cls is None:
        raise ValueError(
            f"Unknown attention strategy: {archetype_cfg.attention_strategy!r}. "
            f"Available: {list(ATTENTION_REGISTRY)}"
        )
    attention = attention_cls(**archetype_cfg.attention_kwargs)

    attendance_cls = ATTENDANCE_REGISTRY.get(archetype_cfg.attendance_strategy)
    if attendance_cls is None:
        raise ValueError(
            f"Unknown attendance strategy: {archetype_cfg.attendance_strategy!r}. "
            f"Available: {list(ATTENDANCE_REGISTRY)}"
        )
    attendance = attendance_cls(**archetype_cfg.attendance_kwargs)

    return AgentPersona(
        user_id=user_id,
        archetype=archetype_cfg.name,
        tau=tau,
        score_floor=score_floor,
        action_intercepts=action_intercepts,
        action_weights=action_weights,
        lr=lr,
        baseline_logit=baseline_logit,
        attention=attention,
        attendance=attendance,
        pref_vector=pref_vector,
    )


def build_population(
    config: "SimConfig",
    env: "Environment",
    rng: np.random.Generator,
) -> dict[int, AgentPersona]:
    """
    Build the full population of AgentPersona instances for all eval users.

    Archetypes are assigned by sampling from ``config.archetype_mix``
    proportions (normalised). Any archetype name not present in
    ``ARCHETYPE_REGISTRY`` is looked up in ``config.archetype_configs`` if
    that attribute exists, allowing custom archetypes.

    Parameters
    ----------
    config:
        Experiment configuration (provides ``archetype_mix``, ``eval_users``
        via ``env``, and ``random_seed`` for reproducibility).
    env:
        Initialised Environment.
    rng:
        Seeded random generator.

    Returns
    -------
    dict mapping user_id → AgentPersona
    """
    mix = config.archetype_mix
    names = list(mix.keys())
    proportions = np.array([mix[n] for n in names], dtype=float)
    proportions /= proportions.sum()

    assigned_archetypes = rng.choice(names, size=len(env.eval_users), p=proportions)

    population: dict[int, AgentPersona] = {}

    archetype_counts: dict[str, int] = {}
    for uid, arch_name in zip(env.eval_users, assigned_archetypes):
        # Resolve archetype config
        if arch_name in ARCHETYPE_REGISTRY:
            arch_cfg = ARCHETYPE_REGISTRY[arch_name]
        elif hasattr(config, "archetype_configs") and arch_name in config.archetype_configs:
            arch_cfg = config.archetype_configs[arch_name]
        else:
            raise ValueError(
                f"Archetype {arch_name!r} not found in ARCHETYPE_REGISTRY or "
                f"config.archetype_configs."
            )

        population[uid] = build_persona(uid, arch_cfg, env, rng)
        archetype_counts[arch_name] = archetype_counts.get(arch_name, 0) + 1

    logger.info(
        "Built population of %d personas: %s",
        len(population),
        archetype_counts,
    )
    return population
