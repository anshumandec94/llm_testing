"""
sim.runner — simulation loop with MLflow experiment tracking.

The SimulationRunner orchestrates:
  1. Initialisation (Environment → Recommender → Agent → Population).
  2. Simulation rounds:
       - Attendance gate (skip absent users).
       - Recommend → agent.evaluate → persona.act → update.
       - Attention budget depletion and recovery.
  3. Per-round metric computation (NDCG@k, hit rate, holdout recall,
     attendance rate, action mix, signal strength).
  4. MLflow logging of all parameters, per-round metrics, and artefacts.

Usage (from main.py or a notebook):
    config = SimConfig(agent_type="associative", num_rounds=5)
    runner = SimulationRunner(config)
    runner.run()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from lenskit.data import ItemList
from tqdm.auto import tqdm

from sim.config import SimConfig
from sim.environment import Environment
from sim.recommender import Recommender
from sim.user_agent import SimulatedUser

logger = logging.getLogger(__name__)


@dataclass
class SimContext:
    """Holds all stateful simulation components for one run.

    Passed through the runner's private methods so that each method
    receives a single context object instead of a long tuple of
    individually named components.
    """

    env: Environment
    recommender: Recommender
    users: dict[int, SimulatedUser]
    held_out_sets: dict[int, set[int]]
    # Cumulative held-out items surfaced across rounds for recall tracking
    surfaced: dict[int, set[int]] = field(default_factory=dict)
    rng: Any = None  # np.random.Generator, set in _setup_components


@dataclass
class RecommenderOnlyContext:
    """State needed for first-batch raw recommender evaluation."""

    env: Environment
    recommender: Recommender
    held_out_sets: dict[int, set[int]]
    popularity_counts: pd.Series


def _hit_rate(recs: ItemList, held_out_ids: set[int]) -> float:
    if not held_out_ids or len(recs) == 0:
        return 0.0
    rec_ids = {int(iid) for iid in recs.ids()}
    return len(rec_ids & held_out_ids) / len(held_out_ids)


def _ndcg_at_k(recs: ItemList, relevant_ids: set[int], k: int) -> float:
    ids = [int(iid) for iid in recs.ids()][:k]
    dcg = sum(
        1.0 / np.log2(rank + 2) for rank, iid in enumerate(ids) if iid in relevant_ids
    )
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant_ids), k)))
    return float(dcg / ideal) if ideal > 0 else 0.0


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.std(ddof=0))


def _item_popularity(popularity_counts: pd.Series, item_ids: list[int]) -> np.ndarray:
    if not item_ids:
        return np.array([], dtype=float)
    return popularity_counts.reindex(item_ids, fill_value=0).to_numpy(dtype=float)


def _dot_scores(
    user_vector: np.ndarray | None, item_vectors: dict[int, np.ndarray], item_ids: list[int]
) -> tuple[np.ndarray, int]:
    if user_vector is None or not item_ids:
        return np.array([], dtype=float), 0

    scores: list[float] = []
    scored_count = 0
    for mid in item_ids:
        item_vector = item_vectors.get(mid)
        if item_vector is None:
            continue
        scores.append(float(np.dot(user_vector, item_vector)))
        scored_count += 1

    return np.array(scores, dtype=float), scored_count


class SimulationRunner:
    """Runs one complete simulation experiment."""

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.experiment_name)

    # ──────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Execute the simulation and return a DataFrame of per-round metrics."""
        cfg = self.config
        t0 = time.time()

        run_name = f"{cfg.experiment_profile}-{cfg.agent_type}-seed{cfg.random_seed}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(cfg.as_dict())
            mlflow.set_tags(
                {
                    "experiment_profile": cfg.experiment_profile,
                    "agent_type": cfg.agent_type,
                    "random_seed": str(cfg.random_seed),
                    "run_kind": (
                        "analysis"
                        if cfg.experiment_profile == "recommender_only"
                        else "simulation"
                    ),
                }
            )

            if cfg.experiment_profile == "recommender_only":
                summary_df = self._run_recommender_only()
            else:
                rng = np.random.default_rng(cfg.random_seed)
                ctx = self._setup_components(rng)

                round_records: list[dict] = []
                round_iter = tqdm(
                    range(1, cfg.num_rounds + 1),
                    total=cfg.num_rounds,
                    desc="Simulation rounds",
                    unit="round",
                    leave=False,
                )
                for rnd in round_iter:
                    logger.info("--- Round %d / %d ---", rnd, cfg.num_rounds)
                    if rnd > 1:
                        ctx.recommender.retrain()
                    metrics = self._run_round(rnd, ctx)
                    mlflow.log_metrics(metrics, step=rnd)
                    round_records.append({"round": rnd, **metrics})

                summary_df = pd.DataFrame(round_records)
            self._save_summary(summary_df)
            elapsed = time.time() - t0
            mlflow.log_metric("elapsed_seconds", elapsed)
            logger.info("Simulation complete in %.1f s.", elapsed)

        return summary_df

    # ──────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────

    def _setup_components(self, rng: np.random.Generator) -> SimContext:
        """Build and return a SimContext with all stateful simulation components."""
        cfg = self.config

        logger.info("=== Initialising Environment ===")
        env = Environment(cfg)

        logger.info("=== Initialising Recommender ===")
        recommender = Recommender(cfg, env)

        users = SimulatedUser.build_population(cfg, env, rng)

        held_out_sets: dict[int, set[int]] = {
            uid: set(env.held_out_for_user(uid)["movieId"].tolist())
            for uid in env.eval_users
        }
        surfaced: dict[int, set[int]] = {uid: set() for uid in env.eval_users}

        return SimContext(
            env=env,
            recommender=recommender,
            users=users,
            held_out_sets=held_out_sets,
            surfaced=surfaced,
            rng=rng,
        )

    def _setup_recommender_only(self) -> RecommenderOnlyContext:
        """Build only the pieces needed for raw first-batch evaluation."""
        cfg = self.config

        logger.info("=== Initialising Environment ===")
        env = Environment(cfg)

        logger.info("=== Initialising Recommender ===")
        recommender = Recommender(cfg, env)

        held_out_sets: dict[int, set[int]] = {
            uid: set(env.held_out_for_user(uid)["movieId"].tolist())
            for uid in env.eval_users
        }
        # Count popularity from training data so evaluation does not use
        # held-out interactions as prior knowledge.
        popularity_counts = env.train_ratings.groupby("movieId").size()
        if not isinstance(popularity_counts, pd.Series):
            raise TypeError("Expected popularity counts to be a pandas Series.")
        popularity_counts = popularity_counts.astype(float)

        return RecommenderOnlyContext(
            env=env,
            recommender=recommender,
            held_out_sets=held_out_sets,
            popularity_counts=popularity_counts,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Round-level loop
    # ──────────────────────────────────────────────────────────────────────

    def _run_round(self, rnd: int, ctx: SimContext) -> dict:
        """Simulate one round across all eval users and return aggregate metrics."""
        cfg = self.config
        attend_flags: list[bool] = []
        hit_rates: list[float] = []
        ndcgs: list[float] = []
        holdout_recalls: list[float] = []
        signals: list[float] = []
        budgets_consumed: list[float] = []
        action_counts: dict[str, int] = {"watch": 0, "rate": 0, "add_to_list": 0}
        all_recs_rows: list[dict] = []

        user_iter = tqdm(
            ctx.users.items(),
            total=len(ctx.users),
            desc=f"Round {rnd} users",
            unit="user",
            leave=False,
        )
        for uid, ua in user_iter:
            held_ids = ctx.held_out_sets[uid]

            attended, recs_rows, interactions, budget_delta = self._run_user_session(
                uid, rnd, ua, held_ids, ctx
            )
            attend_flags.append(attended)

            if not attended:
                continue

            first_batch = ItemList(
                item_ids=np.array(
                    [r["movieId"] for r in recs_rows if r["request"] == 1],
                    dtype=np.int64,
                )
            )
            hit_rates.append(_hit_rate(first_batch, held_ids))
            ndcgs.append(_ndcg_at_k(first_batch, held_ids, cfg.rec_list_size))
            holdout_recalls.append(
                len(ctx.surfaced[uid]) / len(held_ids) if held_ids else 0.0
            )
            budgets_consumed.append(budget_delta)
            for _, act, sig in interactions:
                if act in action_counts:
                    action_counts[act] += 1
                signals.append(sig)
            all_recs_rows.extend(recs_rows)

        ctx.recommender.advance_round()

        # ── Aggregate metrics ──────────────────────────────────────────
        def _mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        total_actions = sum(action_counts.values())
        metrics = {
            "attendance_rate": sum(attend_flags) / len(ctx.env.eval_users),
            "hit_rate": _mean(hit_rates),
            f"ndcg_at_{cfg.rec_list_size}": _mean(ndcgs),
            "holdout_recall": _mean(holdout_recalls),
            "mean_signal_strength": _mean(signals),
            "mean_attention_consumed": _mean(budgets_consumed),
            "action_watch_frac": action_counts["watch"] / total_actions
            if total_actions
            else 0.0,
            "action_rate_frac": action_counts["rate"] / total_actions
            if total_actions
            else 0.0,
            "action_addlist_frac": action_counts["add_to_list"] / total_actions
            if total_actions
            else 0.0,
        }
        logger.info(
            "  attend=%.2f  hit=%.4f  ndcg@%d=%.4f  recall=%.4f  "
            "signal=%.2f  watch=%.2f  rate=%.2f  addlist=%.2f",
            metrics["attendance_rate"],
            metrics["hit_rate"],
            cfg.rec_list_size,
            metrics[f"ndcg_at_{cfg.rec_list_size}"],
            metrics["holdout_recall"],
            metrics["mean_signal_strength"],
            metrics["action_watch_frac"],
            metrics["action_rate_frac"],
            metrics["action_addlist_frac"],
        )

        if all_recs_rows:
            recs_path = Path("mlartifacts") / f"recs_round_{rnd:03d}.parquet"
            recs_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(all_recs_rows).to_parquet(recs_path, index=False)
            mlflow.log_artifact(str(recs_path), artifact_path="recs")

        return metrics

    def _run_recommender_only(self) -> pd.DataFrame:
        """Evaluate the raw recommender on the first batch only."""
        ctx = self._setup_recommender_only()
        cfg = self.config
        user_rows: list[dict[str, float | int]] = []
        rec_rows: list[dict[str, float | int]] = []

        user_iter = tqdm(
            ctx.env.eval_users,
            total=len(ctx.env.eval_users),
            desc="Evaluating users",
            unit="user",
            leave=False,
        )
        for uid in user_iter:
            held_out_df = ctx.env.held_out_for_user(uid)
            held_ids = ctx.held_out_sets[uid]
            recs = ctx.recommender.recommend(uid, n=cfg.rec_list_size)

            rec_ids = [int(iid) for iid in recs.ids()]
            rec_pop = _item_popularity(ctx.popularity_counts, rec_ids)
            cmp_ids = [int(mid) for mid in held_out_df["movieId"].tolist()]
            cmp_pop = _item_popularity(ctx.popularity_counts, cmp_ids)
            recommender_user = ctx.env.get_user_factor(uid)
            internal_user = ctx.env.get_user_pref_factor(uid)
            recommender_item_vectors = ctx.env.get_item_factors(cmp_ids)
            internal_item_vectors = ctx.env.get_user_pref_item_factors(cmp_ids)
            heldout_recommender_scores, heldout_recommender_scored_count = _dot_scores(
                recommender_user, recommender_item_vectors, cmp_ids
            )
            heldout_internal_scores, heldout_internal_scored_count = _dot_scores(
                internal_user, internal_item_vectors, cmp_ids
            )

            rec_mean, rec_std = _mean_std(rec_pop)
            cmp_mean, cmp_std = _mean_std(cmp_pop)
            hit_rate = _hit_rate(recs, held_ids)
            ndcg = _ndcg_at_k(recs, held_ids, cfg.rec_list_size)
            heldout_recommender_score_mean, heldout_recommender_score_std = _mean_std(
                heldout_recommender_scores
            )
            heldout_internal_score_mean, heldout_internal_score_std = _mean_std(
                heldout_internal_scores
            )

            user_rows.append(
                {
                    "userId": uid,
                    "recommended_item_count": len(rec_ids),
                    "heldout_item_count": len(cmp_ids),
                    "hit_rate": hit_rate,
                    f"ndcg_at_{cfg.rec_list_size}": ndcg,
                    "recommended_popularity_mean": rec_mean,
                    "recommended_popularity_std": rec_std,
                    "heldout_popularity_mean": cmp_mean,
                    "heldout_popularity_std": cmp_std,
                    "popularity_mean_delta": rec_mean - cmp_mean,
                    "heldout_recommender_scored_count": heldout_recommender_scored_count,
                    "heldout_internal_scored_count": heldout_internal_scored_count,
                    "heldout_recommender_score_mean": heldout_recommender_score_mean,
                    "heldout_recommender_score_std": heldout_recommender_score_std,
                    "heldout_internal_score_mean": heldout_internal_score_mean,
                    "heldout_internal_score_std": heldout_internal_score_std,
                    "heldout_score_mean_gap": (
                        heldout_recommender_score_mean - heldout_internal_score_mean
                    ),
                    # Backward-compatible aliases for earlier artifact names.
                    "comparison_item_count": len(cmp_ids),
                    "comparison_popularity_mean": cmp_mean,
                    "comparison_popularity_std": cmp_std,
                }
            )

            for rank, mid in enumerate(rec_ids, start=1):
                rec_rows.append(
                    {
                        "userId": uid,
                        "rank": rank,
                        "movieId": mid,
                        "popularity": float(rec_pop[rank - 1]),
                        "is_held_out_hit": int(mid in held_ids),
                    }
                )

        user_df = pd.DataFrame(user_rows)
        rec_df = pd.DataFrame(rec_rows)
        metrics = {
            "user_count": float(len(user_df)),
            "hit_rate": float(user_df["hit_rate"].mean()) if not user_df.empty else 0.0,
            f"ndcg_at_{cfg.rec_list_size}": float(user_df[f"ndcg_at_{cfg.rec_list_size}"].mean())
            if not user_df.empty
            else 0.0,
            "fraction_users_with_holdout_hit": float((user_df["hit_rate"] > 0).mean())
            if not user_df.empty
            else 0.0,
            "mean_user_recommended_popularity_mean": float(
                user_df["recommended_popularity_mean"].mean()
            )
            if not user_df.empty
            else 0.0,
            "mean_user_recommended_popularity_std": float(
                user_df["recommended_popularity_std"].mean()
            )
            if not user_df.empty
            else 0.0,
            "mean_user_heldout_popularity_mean": float(user_df["heldout_popularity_mean"].mean())
            if not user_df.empty
            else 0.0,
            "mean_user_heldout_popularity_std": float(user_df["heldout_popularity_std"].mean())
            if not user_df.empty
            else 0.0,
            "mean_user_comparison_popularity_mean": float(
                user_df["comparison_popularity_mean"].mean()
            )
            if not user_df.empty
            else 0.0,
            "mean_user_comparison_popularity_std": float(
                user_df["comparison_popularity_std"].mean()
            )
            if not user_df.empty
            else 0.0,
            "mean_user_popularity_mean_delta": float(user_df["popularity_mean_delta"].mean())
            if not user_df.empty
            else 0.0,
            "std_user_popularity_mean_delta": float(
                user_df["popularity_mean_delta"].std(ddof=0)
            )
            if not user_df.empty
            else 0.0,
            "mean_user_heldout_recommender_score_mean": float(
                user_df["heldout_recommender_score_mean"].mean()
            )
            if not user_df.empty
            else 0.0,
            "mean_user_heldout_recommender_score_std": float(
                user_df["heldout_recommender_score_std"].mean()
            )
            if not user_df.empty
            else 0.0,
            "mean_user_heldout_internal_score_mean": float(
                user_df["heldout_internal_score_mean"].mean()
            )
            if not user_df.empty
            else 0.0,
            "mean_user_heldout_internal_score_std": float(
                user_df["heldout_internal_score_std"].mean()
            )
            if not user_df.empty
            else 0.0,
            "mean_user_heldout_score_mean_gap": float(
                user_df["heldout_score_mean_gap"].mean()
            )
            if not user_df.empty
            else 0.0,
            "std_user_heldout_score_mean_gap": float(
                user_df["heldout_score_mean_gap"].std(ddof=0)
            )
            if not user_df.empty
            else 0.0,
            "fraction_users_recommender_score_mean_gt_internal": float(
                (
                    user_df["heldout_recommender_score_mean"]
                    > user_df["heldout_internal_score_mean"]
                ).mean()
            )
            if not user_df.empty
            else 0.0,
            "fraction_users_recommended_mean_gt_comparison_mean": float(
                (
                    user_df["recommended_popularity_mean"]
                    > user_df["comparison_popularity_mean"]
                ).mean()
            )
            if not user_df.empty
            else 0.0,
        }

        mlflow.log_metrics(metrics)

        artifact_dir = Path("mlartifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)

        user_diag_path = artifact_dir / "recommender_only_user_diagnostics.parquet"
        user_df.to_parquet(user_diag_path, index=False)
        mlflow.log_artifact(str(user_diag_path), artifact_path="diagnostics")

        recs_path = artifact_dir / "recommender_only_recommendations.parquet"
        rec_df.to_parquet(recs_path, index=False)
        mlflow.log_artifact(str(recs_path), artifact_path="recs")

        heldout_score_rows: list[dict[str, float | int]] = []
        for uid in ctx.env.eval_users:
            held_out_df = ctx.env.held_out_for_user(uid)
            held_ids = [int(mid) for mid in held_out_df["movieId"].tolist()]
            ratings_by_movie = {
                int(mid): float(rating)
                for mid, rating in zip(held_out_df["movieId"], held_out_df["rating"])
            }
            recommender_user = ctx.env.get_user_factor(uid)
            internal_user = ctx.env.get_user_pref_factor(uid)
            recommender_item_vectors = ctx.env.get_item_factors(held_ids)
            internal_item_vectors = ctx.env.get_user_pref_item_factors(held_ids)

            for mid in held_ids:
                recommender_item = recommender_item_vectors.get(mid)
                internal_item = internal_item_vectors.get(mid)
                heldout_score_rows.append(
                    {
                        "userId": uid,
                        "movieId": mid,
                        "rating": ratings_by_movie[mid],
                        "recommender_score": float(np.dot(recommender_user, recommender_item))
                        if recommender_user is not None and recommender_item is not None
                        else np.nan,
                        "internal_score": float(np.dot(internal_user, internal_item))
                        if internal_user is not None and internal_item is not None
                        else np.nan,
                    }
                )

        heldout_scores_path = artifact_dir / "recommender_only_heldout_scores.parquet"
        pd.DataFrame(heldout_score_rows).to_parquet(heldout_scores_path, index=False)
        mlflow.log_artifact(str(heldout_scores_path), artifact_path="diagnostics")

        logger.info(
            "  raw hit=%.4f  ndcg@%d=%.4f  rec_pop=%.2f  held_pop=%.2f  "
            "score(rec)=%.4f  score(int)=%.4f",
            metrics["hit_rate"],
            cfg.rec_list_size,
            metrics[f"ndcg_at_{cfg.rec_list_size}"],
            metrics["mean_user_recommended_popularity_mean"],
            metrics["mean_user_heldout_popularity_mean"],
            metrics["mean_user_heldout_recommender_score_mean"],
            metrics["mean_user_heldout_internal_score_mean"],
        )

        return pd.DataFrame([{"round": 1, **metrics}])

    # ──────────────────────────────────────────────────────────────────────
    # User-session-level logic
    # ──────────────────────────────────────────────────────────────────────

    def _run_user_session(self, uid, rnd, ua: SimulatedUser, held_ids, ctx: SimContext):
        """Run one user for one round.

        Returns (attended, recs_rows, interactions, budget_consumed).
        If the user didn't attend, returns (False, [], [], 0.0).
        """
        if not ua.will_attend(ctx.rng):
            return False, [], [], 0.0

        start_budget = ua.budget
        recs_rows, interactions = self._run_request_loop(uid, rnd, ua, held_ids, ctx)

        ctx.recommender.update_user(uid, [(mid, sig) for mid, _, sig in interactions])
        acted_ids = [mid for mid, _, _ in interactions]
        acted_factors = ctx.env.get_user_pref_item_factors(acted_ids) if interactions else {}
        ua.update(rnd, interactions, acted_factors, self.config)

        ctx.surfaced[uid].update({r["movieId"] for r in recs_rows} & held_ids)
        return True, recs_rows, interactions, max(0.0, start_budget - ua.budget)

    def _run_request_loop(self, uid, rnd, ua: SimulatedUser, held_ids, ctx: SimContext):
        """Issue recommendation requests until accept_k interactions or budget exhausted.

        Returns (recs_rows, interactions).
        """
        cfg = self.config
        recs_rows: list[dict] = []
        interactions: list[tuple[int, str, float]] = []

        for req in range(cfg.max_requests_per_round):
            if ua.budget <= 0:
                break

            candidates = ctx.recommender.recommend(uid, n=cfg.rec_list_size)
            if len(candidates) == 0:
                logger.debug("User %d: no candidates left on request %d.", uid, req + 1)
                break

            ctx.recommender.mark_sent(uid, candidates)
            candidate_ids = [int(iid) for iid in candidates.ids()]
            item_factors = ctx.env.get_user_pref_item_factors(candidate_ids)

            new_rows, new_ints = ua.step(
                candidates, item_factors, held_ids, rnd, req, cfg, ctx.rng
            )
            recs_rows.extend(new_rows)
            interactions.extend(new_ints)

            if len(interactions) >= cfg.accept_k:
                break

        return recs_rows, interactions

    # ──────────────────────────────────────────────────────────────────────
    # Artefact persistence
    # ──────────────────────────────────────────────────────────────────────

    def _save_summary(self, summary_df: pd.DataFrame) -> None:
        summary_path = Path("mlartifacts") / "summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        mlflow.log_artifact(str(summary_path))
