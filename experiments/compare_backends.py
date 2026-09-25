"""
The backend comparison harness: one `PreferenceBackend` arm per invocation.

Each run scores one backend (`experiments/backends.py`) over the evaluation
users of a published sweep, on the pairs `select_held_items` chooses, and
logs one MLflow run. One arm per invocation is deliberate: splits are
deterministic given `SimConfig.split_cache_key()`, so arms scored on
different machines (the mlx LLM, a CUDA SASRec) still score the same pairs
and can sit in one table. `scored_pairs.csv` is the evidence that they did.

Per run, as MLflow artifacts:
    scored_pairs.csv               the exact (userId, movieId) pairs selected,
                                   including any the backend returned nan for
    per_item_predictions.parquet   one row per pair: userId, movieId, backend,
                                   predicted_residual, actual_residual,
                                   predicted_rating, actual_rating

Metrics:
    error/mae, error/rmse                  bias + residual, unclipped
    error/mae_clipped, error/rmse_clipped  clipped to [1, 5], the unit the
                                           clamped bias-only null is in
    error/mae_user_se (and _clipped)       SE of the per-user mean absolute
                                           error, the user-clustered interval
    meta/user_count, meta/item_count       users and items behind the MAEs
    meta/nan_count                         pairs the backend could not score
    meta/pair_count                        all selected pairs, nan included

nan predictions are excluded from every error metric, never imputed, so
`meta/item_count` is `meta/pair_count - meta/nan_count`.

Usage:
    uv run python experiments/compare_backends.py --backend bias_only
    uv run python experiments/compare_backends.py --backend associative --sweep u2566
    uv run python experiments/compare_backends.py --backend sasrec --checkpoint runs/sasrec/checkpoint.pt

The SASRec arm scores a checkpoint from `scripts/train_sasrec.py`, which must
have been trained on this run's split (the sweep's `eval_user_frac` on
`BASE_CONFIG`) and `sasrec_maxlen`; anything else is refused. Its run also
logs the checkpoint path as a param and its saved config as
`sasrec_checkpoint_config.json`.
"""

import argparse
import dataclasses
import logging
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backends import (
    BACKEND_REGISTRY,
    AssociativeBackend,
    BiasOnlyBackend,
    LLMBackend,
    PreferenceBackend,
    SASRecBackend,
)
from experiments.bias_only_null import ITEM_SELECTION, MAX_ITEMS, SWEEPS, clustered_mae
from experiments.llm_vs_associative import BASE_CONFIG, MLFLOW_URI, _log_scored_pairs, select_held_items
from sim.config import SimConfig
from sim.environment import Environment
from sim.population import build_user_assignments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "backend-comparison"
# Backends the harness can construct. The rest of BACKEND_REGISTRY are stubs
# that raise NotImplementedError from their constructor.
BUILDABLE = ("bias_only", "associative", "llm", "sasrec")


def build_backend(name: str, env: Environment, checkpoint: Path | None = None) -> PreferenceBackend:
    """Construct one arm. The LLM uses the config's default LLM settings.

    `checkpoint` is required for, and only used by, `sasrec`.
    """
    if name == "sasrec":
        if checkpoint is None:
            raise ValueError("the sasrec backend needs a checkpoint (--checkpoint)")
        return SASRecBackend.from_checkpoint(env, checkpoint)
    if name == "bias_only":
        return BiasOnlyBackend()
    if name == "associative":
        return AssociativeBackend.from_env(env)
    if name == "llm":
        from sim.agents.llm import LLMAgent

        cfg = env.config
        agent = LLMAgent(
            env,
            model_id=cfg.llm_model_id,
            history_k=cfg.llm_history_k,
            history_strategy=cfg.llm_history_strategy,
            max_tokens=cfg.llm_max_tokens,
            overview_max_chars=cfg.llm_overview_max_chars,
            use_few_shot=cfg.llm_use_few_shot,
        )
        return LLMBackend(env, agent)
    return BACKEND_REGISTRY[name](env)


def unavailable_reason(name: str) -> str | None:
    """Why `name` cannot be scored, checked before the environment is built."""
    if name in BUILDABLE:
        return None
    try:
        BACKEND_REGISTRY[name]()
    except NotImplementedError as exc:
        return str(exc)
    return "it is registered but not wired into the harness"


def score_backend(
    env: Environment,
    assignments,
    backend: PreferenceBackend,
    max_items_per_user: int | None,
    item_selection: str,
    seed: int,
    split: str,
) -> pd.DataFrame:
    """One row per selected pair, in `score_bias_only` order.

    Pairs come from `select_held_items` alone, so they do not depend on the
    backend. Ratings are bias + residual, unclipped; nan passes through.
    """
    frames = []
    for assignment in assignments:
        uid = assignment.base_user_id
        held_out_df = env.held_out_for_user(uid, split=split)
        if held_out_df.empty:
            continue
        held_ids = select_held_items(
            held_out_df, max_items_per_user,
            selection=item_selection, user_id=uid, seed=seed,
        )
        actual_by_id = {
            int(mid): float(r)
            for mid, r in zip(held_out_df["movieId"], held_out_df["rating"])
        }
        residual = np.asarray(backend.predict(uid, held_ids), dtype=np.float64)
        if residual.shape != (len(held_ids),):
            raise ValueError(f"{backend.name} returned shape {residual.shape} for {len(held_ids)} items")
        bias = np.array([env.get_rating_bias(uid, mid) for mid in held_ids], dtype=np.float64)
        actual = np.array([actual_by_id[mid] for mid in held_ids], dtype=np.float64)
        frames.append(pd.DataFrame({
            "userId": np.full(len(held_ids), uid, dtype=np.int64),
            "movieId": np.asarray(held_ids, dtype=np.int64),
            "backend": backend.name,
            "predicted_residual": residual,
            "actual_residual": actual - bias,
            "predicted_rating": bias + residual,
            "actual_rating": actual,
        }))
    return pd.concat(frames, ignore_index=True)


def compute_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Error metrics over the non-nan rows, with user-clustered SEs."""
    scored = frame[frame["predicted_rating"].notna()].rename(columns={"actual_rating": "rating"})
    scored = scored.assign(predicted_clipped=scored["predicted_rating"].clip(1.0, 5.0))
    raw = clustered_mae(scored, "predicted_rating")
    clipped = clustered_mae(scored, "predicted_clipped")
    return {
        "error/mae": raw["mae"],
        "error/rmse": raw["rmse"],
        "error/mae_user_mean": raw["user_mae"],
        "error/mae_user_se": raw["se"],
        "error/mae_clipped": clipped["mae"],
        "error/rmse_clipped": clipped["rmse"],
        "error/mae_clipped_user_mean": clipped["user_mae"],
        "error/mae_clipped_user_se": clipped["se"],
        "meta/user_count": float(raw["n_users"]),
        "meta/item_count": float(raw["n_items"]),
        "meta/nan_count": float(frame["predicted_rating"].isna().sum()),
        "meta/pair_count": float(len(frame)),
    }


def run_backend(
    cfg: SimConfig,
    env: Environment,
    assignments,
    backend: PreferenceBackend,
    *,
    max_items_per_user: int | None,
    item_selection: str,
    tracking_uri: str,
    run_name: str,
) -> tuple[str, dict[str, float]]:
    """Score `backend` and log one MLflow run. Returns (run_id, metrics).

    A backend with a `describe()` (SASRec) also logs the params and the saved
    training config it returns, so the run names the model it scored.
    """
    frame = score_backend(
        env, assignments, backend, max_items_per_user, item_selection,
        seed=cfg.random_seed, split=cfg.recommender_eval_split,
    )
    metrics = compute_metrics(frame)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "backend": backend.name,
            "eval_user_frac": cfg.eval_user_frac,
            "max_items_per_user": max_items_per_user,
            "item_selection": item_selection,
            "random_seed": cfg.random_seed,
            "eval_split": cfg.recommender_eval_split,
            "split_cache_key": cfg.split_cache_key(),
        })
        mlflow.log_metrics(metrics)
        describe = getattr(backend, "describe", None)
        if callable(describe):
            params, trained_config = describe()
            mlflow.log_params(params)
            mlflow.log_dict(trained_config, f"{backend.name}_checkpoint_config.json")
        _log_scored_pairs(list(zip(frame["userId"].tolist(), frame["movieId"].tolist())))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_item_predictions.parquet"
            frame.to_parquet(path, index=False)
            mlflow.log_artifact(str(path))
    logger.info(
        "%s  MAE=%.4f (clipped %.4f, user SE %.4f) over %d items, %d users, %d nan",
        run_name, metrics["error/mae"], metrics["error/mae_clipped"],
        metrics["error/mae_clipped_user_se"], int(metrics["meta/item_count"]),
        int(metrics["meta/user_count"]), int(metrics["meta/nan_count"]),
    )
    return run.info.run_id, metrics


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score one preference backend on the evaluation pairs.")
    parser.add_argument("--backend", required=True, choices=sorted(BACKEND_REGISTRY))
    parser.add_argument(
        "--sweep", choices=sorted(SWEEPS), default="u128",
        help="Evaluation cohort, as in bias_only_null.py. Default u128.",
    )
    parser.add_argument(
        "--max-items", type=int, default=MAX_ITEMS, metavar="N",
        help=f"Held-out items per user (default {MAX_ITEMS}). 0 means all.",
    )
    parser.add_argument("--item-selection", choices=["first", "random"], default=ITEM_SELECTION)
    parser.add_argument("--mlflow-uri", default=MLFLOW_URI)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="SASRec checkpoint from scripts/train_sasrec.py. Required for --backend sasrec.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    reason = unavailable_reason(args.backend)
    if reason is not None:
        sys.exit(f"error: backend {args.backend!r} cannot be scored yet: {reason}")
    if (args.backend == "sasrec") != (args.checkpoint is not None):
        sys.exit("error: --checkpoint is required for, and only accepted with, --backend sasrec")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        sys.exit(f"error: checkpoint {args.checkpoint} does not exist")

    cfg = dataclasses.replace(BASE_CONFIG, eval_user_frac=SWEEPS[args.sweep]["eval_user_frac"])
    max_items = args.max_items or None
    env = Environment(cfg)
    assignments = build_user_assignments(cfg, env, np.random.default_rng(cfg.random_seed))
    backend = build_backend(args.backend, env, checkpoint=args.checkpoint)
    cap = f"{args.item_selection}{max_items}" if max_items else "all"
    run_id, metrics = run_backend(
        cfg, env, assignments, backend,
        max_items_per_user=max_items, item_selection=args.item_selection,
        tracking_uri=args.mlflow_uri, run_name=f"{args.backend}-{args.sweep}-{cap}",
    )
    print(f"run {run_id}: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
