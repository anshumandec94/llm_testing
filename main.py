"""
Entry point for running simulation experiments.

Usage examples
--------------
# Run with defaults from SimConfig:
    uv run python main.py

# Run the raw factorization-only diagnostic on held-out sets:
    uv run python main.py --experiment_profile recommender_only

# Force-rebuild embeddings (e.g. after changing mf_features or semantic_model):
    uv run python main.py --force_rebuild_embeddings

# Override any SimConfig field via keyword arguments:
    uv run python main.py --agent_type associative --num_rounds 5 --eval_user_frac 0.2

# Run HPO automatically if no cached best config exists, then use the results:
    uv run python main.py --hpo_config configs/hpo.json

# Launch the MLflow UI to inspect results:
    mlflow ui --backend-store-uri mlruns
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sim.config import SimConfig
from sim.runner import SimulationRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(part) for part in _parse_csv_list(raw)]


def parse_args(argv: list[str] | None = None) -> SimConfig:
    """Parse CLI arguments and return a populated SimConfig."""
    defaults = SimConfig()
    parser = argparse.ArgumentParser(
        description="Run an ABM recommender-system simulation experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python main.py\n"
            "  uv run python main.py --experiment_profile recommender_only\n"
            "  uv run python main.py --num_rounds 5 --eval_user_frac 0.2\n"
            "  uv run python main.py --force_rebuild_embeddings"
        ),
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    parser.add_argument("--data_dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--embeddings_dir", type=Path, default=defaults.embeddings_dir)

    # ── MLflow ─────────────────────────────────────────────────────────────
    parser.add_argument("--mlflow_tracking_uri", default=defaults.mlflow_tracking_uri)
    parser.add_argument("--experiment_name", default=defaults.experiment_name)

    # ── Split ──────────────────────────────────────────────────────────────
    parser.add_argument("--eval_user_frac", type=float, default=defaults.eval_user_frac)
    parser.add_argument("--validation_frac", type=float, default=defaults.validation_frac)
    parser.add_argument("--holdout_frac", type=float, default=defaults.holdout_frac)
    parser.add_argument("--min_ratings", type=int, default=defaults.min_ratings)

    # ── Simulation ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--experiment_profile",
        default=defaults.experiment_profile,
        choices=["full", "recommender_only"],
        help=(
            "Experiment mode: 'full' runs the simulation loop; "
            "'recommender_only' evaluates held-out items directly and logs "
            "raw recommender/internal latent-space diagnostics."
        ),
    )
    parser.add_argument(
        "--recommender_eval_split",
        default=defaults.recommender_eval_split,
        choices=["validation", "held_out"],
        help="Split used by recommender_only diagnostics.",
    )
    parser.add_argument("--num_rounds", type=int, default=defaults.num_rounds)
    parser.add_argument("--rec_list_size", type=int, default=defaults.rec_list_size)
    parser.add_argument("--accept_k", type=int, default=defaults.accept_k)
    parser.add_argument(
        "--max_requests_per_round",
        type=int,
        default=defaults.max_requests_per_round,
        help="Max re-requests per user per round if agent rejects a batch.",
    )

    # ── Model ──────────────────────────────────────────────────────────────
    parser.add_argument("--mf_features", type=int, default=defaults.mf_features)
    parser.add_argument("--mf_epochs", type=int, default=defaults.mf_epochs)
    parser.add_argument(
        "--mf_regularization",
        type=float,
        default=defaults.mf_regularization,
    )
    parser.add_argument("--mf_damping", type=float, default=defaults.mf_damping)
    parser.add_argument("--semantic_model", default=defaults.semantic_model)

    # ── Agent ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--agent_type",
        default=defaults.agent_type,
        choices=[
            "associative",
            "associative_baseline",
            "residual_profile",
            "item_item",
            "semantic",
            "seq2seq",
            "llm",
        ],
    )
    parser.add_argument(
        "--agent_types",
        type=_parse_csv_list,
        default=defaults.agent_types,
        help="Comma-separated agent types for mixed/comparison experiments.",
    )
    parser.add_argument(
        "--agent_type_proportions",
        type=_parse_csv_floats,
        default=defaults.agent_type_proportions,
        help="Comma-separated proportions aligned with --agent_types.",
    )
    parser.add_argument(
        "--agent_assignment_mode",
        default=defaults.agent_assignment_mode,
        choices=["one_to_one", "one_per_agent_type"],
        help="Assign one agent type per base user or replicate each base user across agent types.",
    )

    # ── Misc ───────────────────────────────────────────────────────────────
    parser.add_argument("--random_seed", type=int, default=defaults.random_seed)
    parser.add_argument(
        "--force_rebuild_embeddings",
        action="store_true",
        default=defaults.force_rebuild_embeddings,
        help="Re-build ChromaDB collections even if they already exist.",
    )

    # ── HPO ────────────────────────────────────────────────────────────────
    # These flags are consumed by main(); parse_args includes them so argparse
    # doesn't reject them when both HPO and sim flags are passed together.
    parser.add_argument(
        "--hpo_config",
        type=Path,
        default=None,
        help=(
            "Path to an HPO JSON config file. If provided and no cached best "
            "config exists at --hpo_artifact_dir, HPO is run automatically "
            "and its best MF params are applied before the main experiment."
        ),
    )
    parser.add_argument(
        "--hpo_artifact_dir",
        type=Path,
        default=Path("mlartifacts"),
        help="Directory to read/write HPO artifacts (default: mlartifacts).",
    )

    args = parser.parse_args(argv)

    return SimConfig(
        data_dir=args.data_dir,
        embeddings_dir=args.embeddings_dir,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
        eval_user_frac=args.eval_user_frac,
        validation_frac=args.validation_frac,
        holdout_frac=args.holdout_frac,
        min_ratings=args.min_ratings,
        experiment_profile=args.experiment_profile,
        recommender_eval_split=args.recommender_eval_split,
        num_rounds=args.num_rounds,
        rec_list_size=args.rec_list_size,
        accept_k=args.accept_k,
        max_requests_per_round=args.max_requests_per_round,
        mf_features=args.mf_features,
        mf_epochs=args.mf_epochs,
        mf_regularization=args.mf_regularization,
        mf_damping=args.mf_damping,
        semantic_model=args.semantic_model,
        agent_type=args.agent_type,
        agent_types=args.agent_types,
        agent_type_proportions=args.agent_type_proportions,
        agent_assignment_mode=args.agent_assignment_mode,
        random_seed=args.random_seed,
        force_rebuild_embeddings=args.force_rebuild_embeddings,
    )


def _parse_hpo_flags(argv: list[str] | None) -> tuple[Path | None, Path]:
    """Extract HPO-only flags without enforcing all sim-config args."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--hpo_config", type=Path, default=None)
    p.add_argument("--hpo_artifact_dir", type=Path, default=Path("mlartifacts"))
    ns, _ = p.parse_known_args(argv)
    return ns.hpo_config, ns.hpo_artifact_dir


def _apply_hpo_if_needed(
    config: SimConfig,
    hpo_config_path: Path | None,
    artifact_dir: Path,
) -> SimConfig:
    """
    Return a config with HPO-tuned params applied.

    The cache file (``artifact_dir/best_recommender_config.json``) stores ONLY
    the params that were tuned across all HPO runs (partial dict, not a full
    SimConfig).  Multiple HPO runs merge into the same file — e.g. running
    ``hpo.json`` saves MF params, then running ``hpo_svd.json`` adds
    ``user_pref_features`` without erasing the MF results.

    Resolution order
    ----------------
    1. If the cache already exists, apply its params and return immediately
       (HPO is not re-run).
    2. If a ``hpo_config_path`` is provided and no cache exists, run HPO now,
       which writes the cache, then apply it.
    3. If neither condition holds, return the config unchanged.
    """
    from dataclasses import fields as dc_fields, replace

    # Params that HPO is allowed to tune — applied from the cache when found.
    _TUNABLE = {
        "mf_features",
        "mf_epochs",
        "mf_regularization",
        "mf_damping",
        "user_pref_features",
    }

    best_config_path = artifact_dir / "best_recommender_config.json"

    if best_config_path.exists():
        logger.info("Using cached HPO best config from %s", best_config_path)
        cached = json.loads(best_config_path.read_text())
        valid_keys = {f.name for f in dc_fields(SimConfig)}
        apply_params = {k: v for k, v in cached.items() if k in valid_keys and k in _TUNABLE}
        if apply_params:
            logger.info("Applying tuned params: %s", apply_params)
        return replace(config, **apply_params)

    if hpo_config_path is None:
        return config

    logger.info("No cached HPO result found; running HPO from %s …", hpo_config_path)
    from sim.hpo import HPOConfig, run_hpo

    run_hpo(HPOConfig.from_json_file(hpo_config_path), artifact_dir=artifact_dir)

    if best_config_path.exists():
        cached = json.loads(best_config_path.read_text())
        valid_keys = {f.name for f in dc_fields(SimConfig)}
        apply_params = {k: v for k, v in cached.items() if k in valid_keys and k in _TUNABLE}
        if apply_params:
            logger.info("Applying tuned params: %s", apply_params)
        return replace(config, **apply_params)

    return config


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    hpo_config_path, hpo_artifact_dir = _parse_hpo_flags(argv)
    config = _apply_hpo_if_needed(config, hpo_config_path, hpo_artifact_dir)
    logger.info("Starting experiment: agent=%s  rounds=%d", config.agent_type, config.num_rounds)
    runner = SimulationRunner(config)
    summary = runner.run()
    print("\n=== Simulation Summary ===")
    print(summary.to_string(index=False))
    print(f"\nMLflow UI: mlflow ui --backend-store-uri {config.mlflow_tracking_uri!s}")


if __name__ == "__main__":
    main(sys.argv[1:])
