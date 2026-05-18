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

# Launch the MLflow UI to inspect results:
    mlflow ui --backend-store-uri mlruns
"""
from __future__ import annotations

import argparse
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
    parser.add_argument("--semantic_model", default=defaults.semantic_model)

    # ── Agent ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--agent_type",
        default=defaults.agent_type,
        choices=["associative", "semantic", "seq2seq", "llm"],
    )

    # ── Misc ───────────────────────────────────────────────────────────────
    parser.add_argument("--random_seed", type=int, default=defaults.random_seed)
    parser.add_argument(
        "--force_rebuild_embeddings",
        action="store_true",
        default=defaults.force_rebuild_embeddings,
        help="Re-build ChromaDB collections even if they already exist.",
    )

    args = parser.parse_args(argv)

    return SimConfig(
        data_dir=args.data_dir,
        embeddings_dir=args.embeddings_dir,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
        eval_user_frac=args.eval_user_frac,
        holdout_frac=args.holdout_frac,
        min_ratings=args.min_ratings,
        experiment_profile=args.experiment_profile,
        num_rounds=args.num_rounds,
        rec_list_size=args.rec_list_size,
        accept_k=args.accept_k,
        max_requests_per_round=args.max_requests_per_round,
        mf_features=args.mf_features,
        semantic_model=args.semantic_model,
        agent_type=args.agent_type,
        random_seed=args.random_seed,
        force_rebuild_embeddings=args.force_rebuild_embeddings,
    )


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    logger.info("Starting experiment: agent=%s  rounds=%d", config.agent_type, config.num_rounds)
    runner = SimulationRunner(config)
    summary = runner.run()
    print("\n=== Simulation Summary ===")
    print(summary.to_string(index=False))
    print(f"\nMLflow UI: mlflow ui --backend-store-uri {config.mlflow_tracking_uri!s}")


if __name__ == "__main__":
    main(sys.argv[1:])
