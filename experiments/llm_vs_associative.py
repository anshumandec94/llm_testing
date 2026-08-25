"""
Comparison experiment: AssociativeAgent vs LLM agent variants.

Each held-out user is evaluated by BOTH agent types on their held-out items.
The metric is rating prediction error — how close is the agent's predicted
rating to the rating the user actually gave?

AssociativeAgent prediction:
    predicted = env.get_rating_bias(uid, mid) + dot(pref_vector, item_factor)
    This reconstructs a full rating from the bias baseline (global + user +
    item bias) and the residual preference signal from TruncatedSVD — the
    same decomposition the model was fitted on.

LLM prediction:
    The LLM receives k examples of movies the user has rated (with title,
    genres, overview, and the rating given) and predicts a rating in [1, 5]
    for each held-out item. It infers user preferences from content alone.

Both predictions are in [1, 5] so MAE and RMSE are directly comparable.

Usage:
    uv run python experiments/llm_vs_associative.py

View results:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
    Navigate to the "llm-agent-comparison" experiment.

Add or remove LLM variants by editing LLM_VARIANTS below.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
from lenskit.data import ItemList
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.agents.llm import LLMAgent
from sim.config import SimConfig
from sim.environment import Environment
from sim.population import build_user_assignments
from sim.user_agent import SimulatedUser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "llm-agent-comparison"
MLFLOW_URI = "sqlite:///mlflow.db"

# ── Base config shared across all runs ───────────────────────────────────────
# eval_user_frac controls who gets EVALUATED — the recommender bias model and
# embeddings are always built from all eligible users' training data.
BASE_CONFIG = SimConfig(
    experiment_name=EXPERIMENT_NAME,
    mlflow_tracking_uri=MLFLOW_URI,
    experiment_profile="recommender_only",
    recommender_eval_split="held_out",
    eval_user_frac=0.001,
    random_seed=42,
    mf_features=64,
    mf_epochs=10,
    mf_regularization=0.1,
    mf_damping=5.0,
)

# ── LLM variants to evaluate ─────────────────────────────────────────────────
# Each dict becomes kwargs for LLMAgent and a label for the MLflow run.
# Add or remove rows freely.
LLM_VARIANTS: list[dict] = [
    {"run_name": "llm-top_rated-k2",  "history_strategy": "top_rated",  "history_k": 2, "use_few_shot": True},
    {"run_name": "llm-top_rated-k5",  "history_strategy": "top_rated",  "history_k": 5, "use_few_shot": True},
    {"run_name": "llm-polarized-k2",  "history_strategy": "polarized",  "history_k": 2, "use_few_shot": True},
    {"run_name": "llm-recent-k3",     "history_strategy": "recent",     "history_k": 3, "use_few_shot": True},
    {"run_name": "llm-polarized-k3-no-fewshot", "history_strategy": "polarized", "history_k": 3, "use_few_shot": False},
]


# ── Shared setup (run once) ───────────────────────────────────────────────────

def _setup():
    """Build environment and user personas. No recommender needed."""
    cfg = BASE_CONFIG
    logger.info("Setting up environment (eval_user_frac=%.2f)...", cfg.eval_user_frac)
    env = Environment(cfg)
    rng = np.random.default_rng(cfg.random_seed)
    assignments = build_user_assignments(cfg, env, rng)
    # Build with agent_type="associative" — we only need the personas, not the
    # agent objects. LLM agents are instantiated separately per variant.
    users, _ = SimulatedUser.build_population(cfg, env, rng, assignments=assignments)
    logger.info("Setup complete: %d evaluation users.", len(assignments))
    return env, assignments, users


# ── Per-agent evaluation ──────────────────────────────────────────────────────

def _compute_metrics(predicted: list[float], actual: list[float]) -> dict:
    p = np.array(predicted, dtype=float)
    a = np.array(actual, dtype=float)
    return {
        "error/mae":   float(np.mean(np.abs(p - a))),
        "error/rmse":  float(np.sqrt(np.mean((p - a) ** 2))),
        "score/mean":  float(np.mean(p)),
        "score/std":   float(np.std(p)),
    }


def evaluate_associative(env: Environment, assignments, users: dict, run_name: str) -> None:
    """Evaluate AssociativeAgent: bias + dot-product per held-out item."""
    cfg = BASE_CONFIG
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "agent_type": "associative",
            "eval_user_frac": cfg.eval_user_frac,
            "random_seed": cfg.random_seed,
            "eval_split": cfg.recommender_eval_split,
        })

        all_predicted: list[float] = []
        all_actual: list[float] = []

        for assignment in assignments:
            uid = assignment.sim_user_id
            base_uid = assignment.base_user_id
            held_out_df = env.held_out_for_user(base_uid, split=cfg.recommender_eval_split)
            if held_out_df.empty:
                continue

            held_ids = [int(mid) for mid in held_out_df["movieId"]]
            actual_by_id = {
                int(mid): float(r)
                for mid, r in zip(held_out_df["movieId"], held_out_df["rating"])
            }
            persona = users[uid].persona
            item_factors = env.get_user_pref_item_factors(held_ids)

            for mid in held_ids:
                bias = env.get_rating_bias(base_uid, mid)
                dot = float(np.dot(persona.pref_vector, item_factors[mid])) if mid in item_factors else 0.0
                pred = float(np.clip(bias + dot, 1.0, 5.0))
                all_predicted.append(pred)
                all_actual.append(actual_by_id[mid])

        metrics = _compute_metrics(all_predicted, all_actual)
        metrics["meta/user_count"] = float(len(assignments))
        metrics["meta/item_count"] = float(len(all_predicted))
        mlflow.log_metrics(metrics)
        logger.info(
            "associative  MAE=%.4f  RMSE=%.4f  score_mean=%.4f  score_std=%.4f",
            metrics["error/mae"], metrics["error/rmse"],
            metrics["score/mean"], metrics["score/std"],
        )


def evaluate_llm(
    env: Environment,
    assignments,
    users: dict,
    run_name: str,
    history_strategy: str,
    history_k: int,
    use_few_shot: bool,
    max_items_per_user: int = 5,
) -> None:
    """Evaluate one LLM variant on the same held-out users.

    max_items_per_user caps how many held-out items are scored per user.
    Each item is one LLM call (~3-5 s on Apple Silicon). With 2566 users
    and no cap the experiment takes days; 5 items/user keeps it ~3 hours.
    """
    cfg = BASE_CONFIG
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info(
        "Loading LLM agent (%s, k=%d, few_shot=%s)...",
        history_strategy, history_k, use_few_shot,
    )
    agent = LLMAgent(
        env,
        model_id=cfg.llm_model_id,
        history_k=history_k,
        history_strategy=history_strategy,
        max_tokens=cfg.llm_max_tokens,
        overview_max_chars=cfg.llm_overview_max_chars,
        use_few_shot=use_few_shot,
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "agent_type": "llm",
            "llm_model_id": cfg.llm_model_id,
            "llm_history_strategy": history_strategy,
            "llm_history_k": history_k,
            "llm_use_few_shot": use_few_shot,
            "llm_max_items_per_user": max_items_per_user,
            "eval_user_frac": cfg.eval_user_frac,
            "random_seed": cfg.random_seed,
            "eval_split": cfg.recommender_eval_split,
        })

        all_predicted: list[float] = []
        all_actual: list[float] = []
        first_call_logged = False
        t_start = time.time()

        with tqdm(assignments, desc=run_name, unit="user") as pbar:
            for i, assignment in enumerate(pbar):
                uid = assignment.sim_user_id
                base_uid = assignment.base_user_id
                held_out_df = env.held_out_for_user(base_uid, split=cfg.recommender_eval_split)
                if held_out_df.empty:
                    continue

                # Cap items per user to keep runtime predictable.
                held_ids = [int(mid) for mid in held_out_df["movieId"]][:max_items_per_user]
                actual_by_id = {
                    int(mid): float(r)
                    for mid, r in zip(held_out_df["movieId"], held_out_df["rating"])
                }
                persona = users[uid].persona
                item_factors = env.get_user_pref_item_factors(held_ids)

                # Log the first user's history and first raw LLM output so we
                # can verify the prompt is wired correctly and the LLM is parsing.
                if not first_call_logged:
                    history = agent._get_history(persona.user_id)
                    logger.info(
                        "First user (uid=%d): %d held-out items, %d history items, "
                        "history titles: %s",
                        base_uid,
                        len(held_ids),
                        len(history),
                        [h.get("title", "?") for h in history],
                    )
                    if held_ids:
                        prompt = agent._build_prompt(history, held_ids[0])
                        from mlx_lm import generate
                        from mlx_lm.sample_utils import make_sampler
                        raw = generate(
                            agent.model, agent.tokenizer, prompt=prompt,
                            max_tokens=cfg.llm_max_tokens,
                            sampler=make_sampler(temp=0.0), verbose=False,
                        )
                        parsed = agent._parse_rating(raw)
                        logger.info(
                            "First LLM call — movie_id=%d  raw_output=%r  parsed=%.2f",
                            held_ids[0], raw[:120], parsed,
                        )
                    first_call_logged = True

                scored = agent.evaluate(
                    ItemList(item_ids=np.array(held_ids, dtype=np.int64)),
                    persona,
                    item_factors,
                )
                scores = scored.scores()
                if scores is None:
                    continue

                for mid, score in zip([int(i) for i in scored.ids()], scores):
                    all_predicted.append(float(score))
                    all_actual.append(actual_by_id[mid])

                # Update progress bar with live metrics.
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta_min = (len(assignments) - i - 1) / rate / 60 if rate > 0 else 0
                pbar.set_postfix(
                    items=len(all_predicted),
                    eta=f"{eta_min:.0f}m",
                    score=f"{np.mean(all_predicted):.2f}" if all_predicted else "—",
                )

                # Periodic summary log every 50 users.
                if (i + 1) % 50 == 0:
                    logger.info(
                        "%s  user %d/%d  items=%d  elapsed=%.0fs  eta=%.0fm",
                        run_name, i + 1, len(assignments),
                        len(all_predicted), elapsed, eta_min,
                    )

        metrics = _compute_metrics(all_predicted, all_actual)
        metrics["meta/user_count"] = float(len(assignments))
        metrics["meta/item_count"] = float(len(all_predicted))
        mlflow.log_metrics(metrics)
        logger.info(
            "%s  MAE=%.4f  RMSE=%.4f  score_mean=%.4f  score_std=%.4f",
            run_name,
            metrics["error/mae"], metrics["error/rmse"],
            metrics["score/mean"], metrics["score/std"],
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    variant_names = [v["run_name"] for v in LLM_VARIANTS]
    parser = argparse.ArgumentParser(
        description="Compare AssociativeAgent vs one LLM variant on held-out rating prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available variants:\n"
            + "\n".join(f"  {n}" for n in variant_names)
            + "\n\nExample:\n"
            "  uv run python experiments/llm_vs_associative.py --variant llm-top_rated-k2\n"
            "  uv run python experiments/llm_vs_associative.py --all"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--variant",
        choices=variant_names,
        metavar="VARIANT",
        help=f"LLM variant to run. One of: {', '.join(variant_names)}",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all LLM variants sequentially (slow).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap held-out items scored per user. "
            "Full set (~45/user × 2566 users) takes ~4-8 days per variant. "
            "Use --max-items 5 for a quick ~3h run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    variants_to_run = (
        LLM_VARIANTS if args.all
        else [v for v in LLM_VARIANTS if v["run_name"] == args.variant]
    )

    max_items = args.max_items  # None = full held-out set

    print(f"=== {EXPERIMENT_NAME} ===")
    print(f"  eval_user_frac: {BASE_CONFIG.eval_user_frac}  |  seed: {BASE_CONFIG.random_seed}")
    print(f"  max_items_per_user: {max_items or 'all (~45/user)'}")
    print(f"  MLflow: {MLFLOW_URI}  →  experiment '{EXPERIMENT_NAME}'\n")

    env, assignments, users = _setup()

    print("[1] Associative baseline")
    evaluate_associative(env, assignments, users, run_name="associative-baseline")

    print(f"\n[2] LLM variant(s): {[v['run_name'] for v in variants_to_run]}")
    for i, variant in enumerate(variants_to_run, 1):
        run_name = variant["run_name"]
        kwargs = {k: v for k, v in variant.items() if k != "run_name"}
        if max_items is not None:
            kwargs["max_items_per_user"] = max_items
        print(f"  [{i}/{len(variants_to_run)}] {run_name}")
        evaluate_llm(env, assignments, users, run_name=run_name, **kwargs)

    print(f"\nDone. View: uv run mlflow ui --backend-store-uri {MLFLOW_URI}")


if __name__ == "__main__":
    main()
