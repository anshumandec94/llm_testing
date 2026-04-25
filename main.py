"""
Entry point for running simulation experiments.

Usage examples
--------------
# Run with defaults (AssociativeAgent, 10 rounds, 5% eval users):
    python main.py

# Override any SimConfig field via keyword arguments:
    python main.py --agent_type associative --num_rounds 5 --eval_user_frac 0.02

# Force-rebuild embeddings (e.g. after changing mf_features or semantic_model):
    python main.py --force_rebuild_embeddings

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
    parser = argparse.ArgumentParser(
        description="Run an ABM recommender-system simulation experiment."
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    parser.add_argument("--data_dir", default="data/ml-32m")
    parser.add_argument("--embeddings_dir", default="embeddings/chroma")

    # ── MLflow ─────────────────────────────────────────────────────────────
    parser.add_argument("--mlflow_tracking_uri", default="mlruns")
    parser.add_argument("--experiment_name", default="abm-recsys")

    # ── Split ──────────────────────────────────────────────────────────────
    parser.add_argument("--eval_user_frac", type=float, default=0.05)
    parser.add_argument("--holdout_frac", type=float, default=0.2)
    parser.add_argument("--min_ratings", type=int, default=50)

    # ── Simulation ─────────────────────────────────────────────────────────
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--rec_list_size", type=int, default=6)
    parser.add_argument("--accept_k", type=int, default=5)
    parser.add_argument(
        "--max_requests_per_round",
        type=int,
        default=3,
        help="Max re-requests per user per round if agent rejects a batch.",
    )

    # ── Model ──────────────────────────────────────────────────────────────
    parser.add_argument("--mf_features", type=int, default=64)
    parser.add_argument("--semantic_model", default="all-MiniLM-L6-v2")

    # ── Agent ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--agent_type",
        default="associative",
        choices=["associative", "semantic", "seq2seq", "llm"],
    )

    # ── Misc ───────────────────────────────────────────────────────────────
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--force_rebuild_embeddings",
        action="store_true",
        default=False,
        help="Re-build ChromaDB collections even if they already exist.",
    )

    args = parser.parse_args(argv)

    return SimConfig(
        data_dir=Path(args.data_dir),
        embeddings_dir=Path(args.embeddings_dir),
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
        eval_user_frac=args.eval_user_frac,
        holdout_frac=args.holdout_frac,
        min_ratings=args.min_ratings,
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
