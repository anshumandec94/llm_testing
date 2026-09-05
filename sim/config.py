"""
sim.config — central configuration dataclass for the simulation.

All hyperparameters live here. Pass a SimConfig instance to each component
so that experiments are fully reproducible and self-documenting in MLflow.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class SimConfig:
    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir: Path = field(default_factory=lambda: Path("data/ml-32m"))
    embeddings_dir: Path = field(default_factory=lambda: Path("embeddings/chroma"))

    # ── MLflow ─────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "abm-recsys"

    # ── Data split ─────────────────────────────────────────────────────────
    eval_user_frac: float = 0.2
    validation_frac: float = 0.1
    holdout_frac: float = 0.2
    min_ratings: int = 50

    # ── Simulation loop ────────────────────────────────────────────────────
    # "full" runs the existing simulation loop; "recommender_only" evaluates
    # the raw LensKit factorization model on the first recommendation batch.
    experiment_profile: str = "full"
    recommender_eval_split: str = "held_out"
    num_rounds: int = 10
    rec_list_size: int = 6
    # Minimum number of acted-on (non-ignore) interactions per round.
    accept_k: int = 5
    max_requests_per_round: int = 3
    # k values for NDCG evaluation in recommender_only mode.
    # Larger k gives more interpretable ranking quality signals against a large catalog.
    ndcg_eval_ks: list[int] = field(default_factory=lambda: [10, 20, 50])

    # ── Recommender (LensKit BiasedMF / ALS) ───────────────────────────────
    mf_features: int = 64
    mf_epochs: int = 10
    mf_regularization: float = 0.1
    mf_damping: float = 5.0

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
    # One of: "associative", "associative_baseline", "residual_profile",
    # "item_item", "semantic", "seq2seq", "llm"
    agent_type: str = "associative"
    agent_types: list[str] = field(default_factory=list)
    agent_type_proportions: list[float] = field(default_factory=list)
    agent_assignment_mode: str = "one_to_one"

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
    archetype_mix: dict = field(default_factory=lambda: {"casual": .7, "binger": .2, "critic": .1 })

    # ── LLM agent ──────────────────────────────────────────────────────────
    # mlx-lm model ID from HuggingFace (mlx-community namespace).
    llm_model_id: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    # Number of past rated items to include as history context.
    llm_history_k: int = 2
    # "top_rated"  — top-k by explicit rating
    # "recent"     — top-k by timestamp
    # "both"       — top-k by rating + top-k by timestamp (deduplicated)
    # "polarized"  — top-k highest-rated + top-k lowest-rated
    llm_history_strategy: str = "top_rated"
    # Max new tokens for LLM generation per item.
    llm_max_tokens: int = 64
    # Truncation limit for movie overviews in the prompt.
    llm_overview_max_chars: int = 300
    # Whether to prepend fixed few-shot examples to anchor output format.
    llm_use_few_shot: bool = True

    # ── SASRec ─────────────────────────────────────────────────────────────
    # Architecture defaults are the reference MovieLens settings, shared by
    # kang205/SASRec and pmixer/SASRec.pytorch. See sim/agents/sasrec_model.py
    # for the full parity notes and the divergences between the two.
    sasrec_hidden_units: int = 50
    sasrec_num_blocks: int = 2
    sasrec_num_heads: int = 1
    sasrec_dropout_rate: float = 0.2
    # Sequence length. 200 is the reference ML-1M setting; ML-32M histories run
    # longer, so this is worth revisiting before the real runs.
    sasrec_maxlen: int = 200
    # False reproduces pmixer's default post-norm block, which is the port
    # target. True switches to pre-norm, which is closer to the published
    # TensorFlow original but not identical to it.
    sasrec_norm_first: bool = False
    # pmixer lets padded timesteps act as attention keys; kang205 masks them
    # out. False is the pmixer behaviour, True the kang205 behaviour, exposed
    # so the difference can be measured rather than argued about.
    sasrec_mask_padded_keys: bool = False
    # Weight on the rating head in L = L_bce + w * L_mse. 0.0 is the ablation
    # that measures what the rating head cost the ranking objective.
    sasrec_rating_loss_weight: float = 1.0
    # Whether input positions carry their debiased residual alongside the item
    # id. Behind a flag so the rating-context contribution can be ablated.
    sasrec_inject_rating: bool = True
    # Hidden width of the rating head MLP.
    sasrec_rating_head_hidden: int = 64

    def as_dict(self) -> dict:
        """Return a flat dict of all parameters (for MLflow logging)."""
        return {
            "data_dir": str(self.data_dir),
            "embeddings_dir": str(self.embeddings_dir),
            "eval_user_frac": self.eval_user_frac,
            "validation_frac": self.validation_frac,
            "holdout_frac": self.holdout_frac,
            "min_ratings": self.min_ratings,
            "experiment_profile": self.experiment_profile,
            "recommender_eval_split": self.recommender_eval_split,
            "num_rounds": self.num_rounds,
            "rec_list_size": self.rec_list_size,
            "accept_k": self.accept_k,
            "max_requests_per_round": self.max_requests_per_round,
            "mf_features": self.mf_features,
            "mf_epochs": self.mf_epochs,
            "mf_regularization": self.mf_regularization,
            "mf_damping": self.mf_damping,
            "user_pref_features": self.user_pref_features,
            "semantic_model": self.semantic_model,
            "random_seed": self.random_seed,
            "agent_type": self.agent_type,
            "agent_types": str(self.agent_types),
            "agent_type_proportions": str(self.agent_type_proportions),
            "agent_assignment_mode": self.agent_assignment_mode,
            "watch_signal": self.watch_signal,
            "add_to_list_signal": self.add_to_list_signal,
            "archetype_mix": str(self.archetype_mix),
            "llm_model_id": self.llm_model_id,
            "llm_history_k": self.llm_history_k,
            "llm_history_strategy": self.llm_history_strategy,
            "llm_max_tokens": self.llm_max_tokens,
            "llm_overview_max_chars": self.llm_overview_max_chars,
            "llm_use_few_shot": self.llm_use_few_shot,
            "sasrec_hidden_units": self.sasrec_hidden_units,
            "sasrec_num_blocks": self.sasrec_num_blocks,
            "sasrec_num_heads": self.sasrec_num_heads,
            "sasrec_dropout_rate": self.sasrec_dropout_rate,
            "sasrec_maxlen": self.sasrec_maxlen,
            "sasrec_norm_first": self.sasrec_norm_first,
            "sasrec_mask_padded_keys": self.sasrec_mask_padded_keys,
            "sasrec_rating_loss_weight": self.sasrec_rating_loss_weight,
            "sasrec_inject_rating": self.sasrec_inject_rating,
            "sasrec_rating_head_hidden": self.sasrec_rating_head_hidden,
        }

    def to_json_dict(self) -> dict:
        """Return a JSON-serializable representation of the config."""
        payload: dict[str, object] = {}
        for cfg_field in fields(self):
            value = getattr(self, cfg_field.name)
            if isinstance(value, Path):
                payload[cfg_field.name] = str(value)
            else:
                payload[cfg_field.name] = value
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "SimConfig":
        """Construct a config from a plain dict."""
        valid_fields = {cfg_field.name for cfg_field in fields(cls)}
        unknown = sorted(set(payload) - valid_fields)
        if unknown:
            raise ValueError(f"Unknown SimConfig fields: {unknown}")

        kwargs = dict(payload)
        for path_field in ("data_dir", "embeddings_dir"):
            if path_field in kwargs:
                kwargs[path_field] = Path(kwargs[path_field])
        return cls(**kwargs)

    @classmethod
    def from_json_file(cls, path: Path) -> "SimConfig":
        """Load a config from a JSON file."""
        return cls.from_dict(json.loads(path.read_text()))

    def platform_mf_kwargs(self) -> dict[str, int | float]:
        """Return LensKit BiasedMF configuration in one shared place."""
        return {
            "features": self.mf_features,
            "epochs": self.mf_epochs,
            "regularization": self.mf_regularization,
            "damping": self.mf_damping,
        }

    def split_cache_key(self) -> str:
        """Return a stable cache key for data-split-defining parameters."""
        return self._cache_key(
            {
                "data_dir": str(self.data_dir),
                "eval_user_frac": self.eval_user_frac,
                "validation_frac": self.validation_frac,
                "holdout_frac": self.holdout_frac,
                "min_ratings": self.min_ratings,
                "random_seed": self.random_seed,
            }
        )

    def platform_factor_cache_key(self) -> str:
        """Return a cache key for platform MF artifacts."""
        return self._cache_key(
            {
                "split": self.split_cache_key(),
                "mf_features": self.mf_features,
                "mf_epochs": self.mf_epochs,
                "mf_regularization": self.mf_regularization,
                "mf_damping": self.mf_damping,
            }
        )

    def user_pref_cache_key(self) -> str:
        """Return a cache key for the user-preference factor space."""
        return self._cache_key(
            {
                "split": self.split_cache_key(),
                "user_pref_features": self.user_pref_features,
            }
        )

    def semantic_cache_key(self) -> str:
        """Return a cache key for semantic embeddings."""
        return self._cache_key(
            {
                "data_dir": str(self.data_dir),
                "semantic_model": self.semantic_model,
            }
        )

    @staticmethod
    def _cache_key(payload: dict) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
