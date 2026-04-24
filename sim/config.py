"""
sim.config — central configuration dataclass for the simulation.

All hyperparameters live here. Pass a SimConfig instance to each component
so that experiments are fully reproducible and self-documenting in MLflow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SimConfig:
    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir: Path = field(default_factory=lambda: Path("data/ml-32m"))
    embeddings_dir: Path = field(default_factory=lambda: Path("embeddings/chroma"))

    # ── MLflow ─────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "mlruns"
    experiment_name: str = "abm-recsys"

    # ── Data split ─────────────────────────────────────────────────────────
    eval_user_frac: float = 0.05
    holdout_frac: float = 0.2
    min_ratings: int = 50

    # ── Simulation loop ────────────────────────────────────────────────────
    num_rounds: int = 10
    rec_list_size: int = 20
    # Minimum number of acted-on (non-ignore) interactions per round.
    accept_k: int = 5
    max_requests_per_round: int = 3

    # ── Recommender (LensKit BiasedMF / ALS) ───────────────────────────────
    mf_features: int = 64

    # ── User preference model (small independent MF via TruncatedSVD) ──────
    # Dimensionality of the user's internal preference representation.
    # Intentionally smaller than mf_features to keep user-side model coarse.
    user_pref_features: int = 8

    # ── Semantic embeddings (sentence-transformers) ────────────────────────
    semantic_model: str = "all-MiniLM-L6-v2"

    # ── Reproducibility ─────────────────────────────────────────────────────
    random_seed: int = 42

    # ── Embedding cache ─────────────────────────────────────────────────────
    force_rebuild_embeddings: bool = False

    # ── Agent ───────────────────────────────────────────────────────────────
    # One of: "associative", "semantic", "seq2seq", "llm"
    agent_type: str = "associative"

    # ── Action model ───────────────────────────────────────────────────────
    # Implicit signal strength for "watch" and "add_to_list" actions.
    # "rate" actions produce an explicit Beta-sampled value in [1, 5].
    watch_signal: float = 4.5
    add_to_list_signal: float = 3.0
    # Beta distribution shape parameters for "rate" signal sampling.
    beta_alpha_max: float = 8.0
    beta_beta_max: float = 8.0

    # ── Attention defaults (can be overridden per archetype) ────────────────
    default_attention_strategy: str = "LinearDecay"
    default_attention_decay_rate: float = 0.15
    default_attention_recovery: str = "full"
    default_attention_recovery_rate: float = 0.5

    # ── Attendance defaults (can be overridden per archetype) ───────────────
    default_attendance_strategy: str = "LogisticAttendance"
    # EWMA smoothing factor for satisfaction signal (used to compute attendance).
    sat_ewma_alpha: float = 0.4
    attend_recency_bonus: float = 0.3
    attend_recency_penalty: float = 0.1
    attend_noise_scale: float = 0.2

    # ── Population mix ─────────────────────────────────────────────────────
    # Dict of archetype_name → proportion. Values are normalised at runtime.
    # e.g. {"casual": 0.6, "binger": 0.4}
    archetype_mix: dict = field(default_factory=lambda: {"casual": 1.0})

    def as_dict(self) -> dict:
        """Return a flat dict of all parameters (for MLflow logging)."""
        return {
            "data_dir": str(self.data_dir),
            "embeddings_dir": str(self.embeddings_dir),
            "eval_user_frac": self.eval_user_frac,
            "holdout_frac": self.holdout_frac,
            "min_ratings": self.min_ratings,
            "num_rounds": self.num_rounds,
            "rec_list_size": self.rec_list_size,
            "accept_k": self.accept_k,
            "max_requests_per_round": self.max_requests_per_round,
            "mf_features": self.mf_features,
            "user_pref_features": self.user_pref_features,
            "semantic_model": self.semantic_model,
            "random_seed": self.random_seed,
            "agent_type": self.agent_type,
            "watch_signal": self.watch_signal,
            "add_to_list_signal": self.add_to_list_signal,
            "archetype_mix": str(self.archetype_mix),
        }
