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
from sim.population import UserAssignment, assignment_metadata, build_user_assignments
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
    assignments: dict[int, UserAssignment]
    # Cumulative held-out items surfaced across rounds for recall tracking
    surfaced: dict[int, set[int]] = field(default_factory=dict)
    rng: Any = None  # np.random.Generator, set in _setup_components


@dataclass
class RecommenderOnlyContext:
    """State needed for first-batch raw recommender evaluation."""

    env: Environment
    recommender: Recommender
    users: dict[int, SimulatedUser]
    held_out_sets: dict[int, set[int]]
    popularity_counts: pd.Series
    assignments: dict[int, UserAssignment]


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


def _score_item_rows(
    user_vector: np.ndarray | None, item_vectors: dict[int, np.ndarray], item_ids: list[int]
) -> list[tuple[int, float]]:
    if user_vector is None or not item_ids:
        return []

    score_rows: list[tuple[int, float]] = []
    for mid in item_ids:
        item_vector = item_vectors.get(mid)
        if item_vector is None:
            continue
        score_rows.append((mid, float(np.dot(user_vector, item_vector))))

    return score_rows


def _score_agent_rows(
    agent: Any,
    persona: Any,
    item_vectors: dict[int, np.ndarray],
    item_ids: list[int],
) -> list[tuple[int, float]]:
    if not item_ids:
        return []

    scored = agent.evaluate(
        ItemList(item_ids=np.array(item_ids, dtype=np.int64)),
        persona,
        item_vectors,
    )
    scores = scored.scores()
    if scores is None:
        return []

    return [(int(movie_id), float(score)) for movie_id, score in zip(scored.ids(), scores)]


def _safe_corr(
    ratings: np.ndarray, scores: np.ndarray, *, method: str = "pearson"
) -> float:
    if len(ratings) != len(scores) or len(ratings) < 2:
        return float("nan")

    ratings_s = pd.Series(ratings, dtype=float)
    scores_s = pd.Series(scores, dtype=float)
    valid = ratings_s.notna() & scores_s.notna()
    if valid.sum() < 2:
        return float("nan")

    ratings_s = ratings_s[valid]
    scores_s = scores_s[valid]
    if ratings_s.nunique() < 2 or scores_s.nunique() < 2:
        return float("nan")

    corr = ratings_s.corr(scores_s, method=method)
    return float(corr) if corr is not None and not pd.isna(corr) else float("nan")


def _mean_without_nan(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.mean()) if not clean.empty else 0.0


class SimulationRunner:
    """Runs one complete simulation experiment."""

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.experiment_name)

    # ──────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────

    def run(
        self,
        *,
        manage_mlflow: bool = True,
        run_name: str | None = None,
        extra_tags: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Execute the simulation and return a DataFrame of per-round metrics."""
        cfg = self.config
        t0 = time.time()

        resolved_run_name = run_name or (
            f"{cfg.experiment_profile}-{cfg.agent_type}-"
            f"{cfg.recommender_eval_split}-seed{cfg.random_seed}"
        )
        if manage_mlflow:
            with mlflow.start_run(run_name=resolved_run_name):
                return self._run_active(t0=t0, extra_tags=extra_tags)

        return self._run_active(t0=t0, extra_tags=extra_tags)

    def _run_active(
        self,
        *,
        t0: float,
        extra_tags: dict[str, str] | None,
    ) -> pd.DataFrame:
        """Run the experiment while logging into the active MLflow run."""
        cfg = self.config
        mlflow.log_params(cfg.as_dict())
        tags = {
            "experiment_profile": cfg.experiment_profile,
            "agent_type": cfg.agent_type,
            "recommender_eval_split": cfg.recommender_eval_split,
            "random_seed": str(cfg.random_seed),
            "run_kind": (
                "analysis"
                if cfg.experiment_profile == "recommender_only"
                else "simulation"
            ),
        }
        if extra_tags:
            tags.update(extra_tags)
        mlflow.set_tags(tags)

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
        mlflow.log_metric("meta/elapsed_seconds", elapsed)
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
        assignments = build_user_assignments(cfg, env, rng)

        logger.info("=== Building Assigned Population ===")
        users, user_assignments = SimulatedUser.build_population(
            cfg, env, rng, assignments=assignments
        )

        logger.info("=== Initialising Recommender ===")
        recommender = Recommender(
            cfg,
            env,
            user_base_map={
                assignment.sim_user_id: assignment.base_user_id
                for assignment in assignments
            },
        )
        mlflow.set_tags(assignment_metadata(cfg, user_assignments))

        held_out_sets: dict[int, set[int]] = {
            assignment.sim_user_id: set(
                env.held_out_for_user(
                    assignment.base_user_id,
                    split=cfg.recommender_eval_split,
                )[
                    "movieId"
                ].tolist()
            )
            for assignment in assignments
        }
        surfaced: dict[int, set[int]] = {
            assignment.sim_user_id: set() for assignment in assignments
        }

        return SimContext(
            env=env,
            recommender=recommender,
            users=users,
            held_out_sets=held_out_sets,
            assignments={
                assignment.sim_user_id: assignment for assignment in assignments
            },
            surfaced=surfaced,
            rng=rng,
        )

    def _setup_recommender_only(self) -> RecommenderOnlyContext:
        """Build only the pieces needed for raw first-batch evaluation."""
        cfg = self.config

        logger.info("=== Initialising Environment ===")
        env = Environment(cfg)
        rng = np.random.default_rng(cfg.random_seed)
        assignments = build_user_assignments(cfg, env, rng)
        users, user_assignments = SimulatedUser.build_population(
            cfg, env, rng, assignments=assignments
        )

        logger.info("=== Initialising Recommender ===")
        recommender = Recommender(
            cfg,
            env,
            user_base_map={
                assignment.sim_user_id: assignment.base_user_id
                for assignment in assignments
            },
        )
        mlflow.set_tags(assignment_metadata(cfg, user_assignments))

        held_out_sets: dict[int, set[int]] = {
            assignment.sim_user_id: set(
                env.held_out_for_user(
                    assignment.base_user_id,
                    split=cfg.recommender_eval_split,
                )[
                    "movieId"
                ].tolist()
            )
            for assignment in assignments
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
            users=users,
            held_out_sets=held_out_sets,
            popularity_counts=popularity_counts,
            assignments={
                assignment.sim_user_id: assignment for assignment in assignments
            },
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
        ndcgs_by_k: dict[int, list[float]] = {k: [] for k in cfg.ndcg_eval_ks}
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
            for k in cfg.ndcg_eval_ks:
                ndcgs_by_k[k].append(_ndcg_at_k(first_batch, held_ids, k))
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
            "sim/attendance_rate": sum(attend_flags) / len(ctx.users),
            "ranking/hit_rate": _mean(hit_rates),
            f"ranking/ndcg_at_{cfg.rec_list_size}": _mean(ndcgs),
            **{
                f"ranking/ndcg_at_{k}": _mean(ndcgs_by_k[k])
                for k in cfg.ndcg_eval_ks
                if k != cfg.rec_list_size
            },
            "sim/holdout_recall": _mean(holdout_recalls),
            "sim/mean_signal_strength": _mean(signals),
            "sim/mean_attention_consumed": _mean(budgets_consumed),
            "sim/action_watch_frac": action_counts["watch"] / total_actions
            if total_actions
            else 0.0,
            "sim/action_rate_frac": action_counts["rate"] / total_actions
            if total_actions
            else 0.0,
            "sim/action_addlist_frac": action_counts["add_to_list"] / total_actions
            if total_actions
            else 0.0,
        }
        logger.info(
            "  attend=%.2f  hit=%.4f  ndcg@%d=%.4f  recall=%.4f  "
            "signal=%.2f  watch=%.2f  rate=%.2f  addlist=%.2f",
            metrics["sim/attendance_rate"],
            metrics["ranking/hit_rate"],
            cfg.rec_list_size,
            metrics[f"ranking/ndcg_at_{cfg.rec_list_size}"],
            metrics["sim/holdout_recall"],
            metrics["sim/mean_signal_strength"],
            metrics["sim/action_watch_frac"],
            metrics["sim/action_rate_frac"],
            metrics["sim/action_addlist_frac"],
        )

        if all_recs_rows:
            recs_path = Path("mlartifacts") / f"recs_round_{rnd:03d}.parquet"
            recs_path.parent.mkdir(parents=True, exist_ok=True)
            recs_df = pd.DataFrame(all_recs_rows)
            subgroup_df = (
                recs_df.groupby(["agent_type"], as_index=False)
                .agg(
                    shown_count=("movieId", "count"),
                    heldout_hit_count=("is_held_out", "sum"),
                )
            )
            subgroup_df["round"] = rnd
            subgroup_path = Path("mlartifacts") / f"recs_round_{rnd:03d}_subgroups.parquet"
            recs_df.to_parquet(recs_path, index=False)
            subgroup_df.to_parquet(subgroup_path, index=False)
            mlflow.log_artifact(str(recs_path), artifact_path="recs")
            mlflow.log_artifact(str(subgroup_path), artifact_path="recs")

        return metrics

    def _run_recommender_only(self) -> pd.DataFrame:
        """Evaluate the raw recommender on the first batch only."""
        ctx = self._setup_recommender_only()
        cfg = self.config
        user_rows: list[dict[str, float | int | str]] = []
        rec_rows: list[dict[str, float | int | str]] = []

        user_iter = tqdm(
            ctx.assignments.values(),
            total=len(ctx.assignments),
            desc="Evaluating users",
            unit="user",
            leave=False,
        )
        for assignment in user_iter:
            uid = assignment.sim_user_id
            base_user_id = assignment.base_user_id
            held_out_df = ctx.env.held_out_for_user(
                base_user_id, split=cfg.recommender_eval_split
            )
            held_ids = ctx.held_out_sets[uid]
            max_eval_n = max(cfg.rec_list_size, max(cfg.ndcg_eval_ks, default=0))
            recs = ctx.recommender.recommend(uid, n=max_eval_n)

            # sim_rec_ids: what the user would actually see (rec_list_size items).
            # all_rec_ids: larger fetch used only for eval NDCG@k computation.
            all_rec_ids = [int(iid) for iid in recs.ids()]
            rec_ids = all_rec_ids[: cfg.rec_list_size]
            rec_pop = _item_popularity(ctx.popularity_counts, rec_ids)
            cmp_ids = [int(mid) for mid in held_out_df["movieId"].tolist()]
            cmp_pop = _item_popularity(ctx.popularity_counts, cmp_ids)
            ratings_by_movie = {
                int(mid): float(rating)
                for mid, rating in zip(held_out_df["movieId"], held_out_df["rating"])
            }
            debiased_ratings_by_movie = {
                mid: ctx.env.debias_rating(base_user_id, mid, rating)
                for mid, rating in ratings_by_movie.items()
            }
            ua = ctx.users[uid]
            recommender_user = ctx.env.get_user_factor(base_user_id)
            persona = ua.persona
            recommender_item_vectors = ctx.env.get_item_factors(cmp_ids)
            internal_item_vectors = ctx.env.get_user_pref_item_factors(cmp_ids)
            heldout_recommender_score_rows = _score_item_rows(
                recommender_user, recommender_item_vectors, cmp_ids
            )
            heldout_internal_score_rows = _score_agent_rows(
                ua.agent, persona, internal_item_vectors, cmp_ids
            )
            heldout_recommender_scores = np.array(
                [score for _, score in heldout_recommender_score_rows], dtype=float
            )
            heldout_internal_scores = np.array(
                [score for _, score in heldout_internal_score_rows], dtype=float
            )
            heldout_recommender_scored_count = len(heldout_recommender_score_rows)
            heldout_internal_scored_count = len(heldout_internal_score_rows)

            rec_mean, rec_std = _mean_std(rec_pop)
            cmp_mean, cmp_std = _mean_std(cmp_pop)
            hit_rate = len(set(rec_ids) & held_ids) / len(held_ids) if held_ids else 0.0
            ndcg = _ndcg_at_k(recs, held_ids, cfg.rec_list_size)
            ndcg_at_eval_ks = {k: _ndcg_at_k(recs, held_ids, k) for k in cfg.ndcg_eval_ks}
            heldout_debiased_ratings = np.array(
                [debiased_ratings_by_movie[mid] for mid in cmp_ids], dtype=float
            )
            heldout_debiased_rating_mean, heldout_debiased_rating_std = _mean_std(
                heldout_debiased_ratings
            )
            heldout_recommender_score_mean, heldout_recommender_score_std = _mean_std(
                heldout_recommender_scores
            )
            heldout_internal_score_mean, heldout_internal_score_std = _mean_std(
                heldout_internal_scores
            )
            recommender_rating_pairs = [
                (debiased_ratings_by_movie[mid], score)
                for mid, score in heldout_recommender_score_rows
            ]
            internal_rating_pairs = [
                (debiased_ratings_by_movie[mid], score)
                for mid, score in heldout_internal_score_rows
            ]
            recommender_ratings = np.array(
                [rating for rating, _ in recommender_rating_pairs], dtype=float
            )
            recommender_scores = np.array(
                [score for _, score in recommender_rating_pairs], dtype=float
            )
            internal_ratings = np.array(
                [rating for rating, _ in internal_rating_pairs], dtype=float
            )
            internal_scores = np.array(
                [score for _, score in internal_rating_pairs], dtype=float
            )
            heldout_recommender_residual_pearson = _safe_corr(
                recommender_ratings, recommender_scores, method="pearson"
            )
            heldout_recommender_residual_spearman = _safe_corr(
                recommender_ratings, recommender_scores, method="spearman"
            )
            heldout_internal_residual_pearson = _safe_corr(
                internal_ratings, internal_scores, method="pearson"
            )
            heldout_internal_residual_spearman = _safe_corr(
                internal_ratings, internal_scores, method="spearman"
            )
            heldout_recommender_mse = (
                float(np.mean((recommender_ratings - recommender_scores) ** 2))
                if len(recommender_rating_pairs) > 0
                else np.nan
            )
            heldout_internal_mse = (
                float(np.mean((internal_ratings - internal_scores) ** 2))
                if len(internal_rating_pairs) > 0
                else np.nan
            )
            heldout_recommender_rmse = (
                float(np.sqrt(heldout_recommender_mse))
                if not np.isnan(heldout_recommender_mse)
                else np.nan
            )
            heldout_internal_rmse = (
                float(np.sqrt(heldout_internal_mse))
                if not np.isnan(heldout_internal_mse)
                else np.nan
            )
            heldout_recommender_mae = (
                float(np.mean(np.abs(recommender_ratings - recommender_scores)))
                if len(recommender_rating_pairs) > 0
                else np.nan
            )
            heldout_internal_mae = (
                float(np.mean(np.abs(internal_ratings - internal_scores)))
                if len(internal_rating_pairs) > 0
                else np.nan
            )

            user_rows.append(
                {
                    "evaluation_split": cfg.recommender_eval_split,
                    "userId": base_user_id,
                    "simulation_user_id": uid,
                    "agent_type": assignment.agent_type,
                    "recommended_item_count": len(rec_ids),
                    "heldout_item_count": len(cmp_ids),
                    "hit_rate": hit_rate,
                    f"ndcg_at_{cfg.rec_list_size}": ndcg,
                    **{f"ndcg_at_{k}": v for k, v in ndcg_at_eval_ks.items()},
                    "recommended_popularity_mean": rec_mean,
                    "recommended_popularity_std": rec_std,
                    "heldout_popularity_mean": cmp_mean,
                    "heldout_popularity_std": cmp_std,
                    "popularity_mean_delta": rec_mean - cmp_mean,
                    "heldout_debiased_rating_mean": heldout_debiased_rating_mean,
                    "heldout_debiased_rating_std": heldout_debiased_rating_std,
                    "heldout_recommender_scored_count": heldout_recommender_scored_count,
                    "heldout_internal_scored_count": heldout_internal_scored_count,
                    "heldout_recommender_score_mean": heldout_recommender_score_mean,
                    "heldout_recommender_score_std": heldout_recommender_score_std,
                    "heldout_internal_score_mean": heldout_internal_score_mean,
                    "heldout_internal_score_std": heldout_internal_score_std,
                    "heldout_recommender_residual_pearson": heldout_recommender_residual_pearson,
                    "heldout_recommender_residual_spearman": heldout_recommender_residual_spearman,
                    "heldout_internal_residual_pearson": heldout_internal_residual_pearson,
                    "heldout_internal_residual_spearman": heldout_internal_residual_spearman,
                    "heldout_recommender_mse": heldout_recommender_mse,
                    "heldout_recommender_rmse": heldout_recommender_rmse,
                    "heldout_recommender_mae": heldout_recommender_mae,
                    "heldout_internal_mse": heldout_internal_mse,
                    "heldout_internal_rmse": heldout_internal_rmse,
                    "heldout_internal_mae": heldout_internal_mae,
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
                        "evaluation_split": cfg.recommender_eval_split,
                        "userId": base_user_id,
                        "simulation_user_id": uid,
                        "agent_type": assignment.agent_type,
                        "rank": rank,
                        "movieId": mid,
                        "popularity": float(rec_pop[rank - 1]),
                        "is_held_out_hit": int(mid in held_ids),
                    }
                )

        user_df = pd.DataFrame(user_rows)
        rec_df = pd.DataFrame(rec_rows)
        metrics = {
            # ── meta ──────────────────────────────────────────────────────
            "meta/user_count": float(len(user_df)),
            # ── ranking ───────────────────────────────────────────────────
            "ranking/hit_rate": float(user_df["hit_rate"].mean()) if not user_df.empty else 0.0,
            f"ranking/ndcg_at_{cfg.rec_list_size}": float(user_df[f"ndcg_at_{cfg.rec_list_size}"].mean())
            if not user_df.empty
            else 0.0,
            **{
                f"ranking/ndcg_at_{k}": float(user_df[f"ndcg_at_{k}"].mean())
                if not user_df.empty
                else 0.0
                for k in cfg.ndcg_eval_ks
            },
            "ranking/frac_users_with_hit": float((user_df["hit_rate"] > 0).mean())
            if not user_df.empty
            else 0.0,
            # ── error ─────────────────────────────────────────────────────
            "error/rec_mse": _mean_without_nan(user_df["heldout_recommender_mse"])
            if not user_df.empty else 0.0,
            "error/rec_rmse": _mean_without_nan(user_df["heldout_recommender_rmse"])
            if not user_df.empty else 0.0,
            "error/rec_mae": _mean_without_nan(user_df["heldout_recommender_mae"])
            if not user_df.empty else 0.0,
            "error/int_mse": _mean_without_nan(user_df["heldout_internal_mse"])
            if not user_df.empty else 0.0,
            "error/int_rmse": _mean_without_nan(user_df["heldout_internal_rmse"])
            if not user_df.empty else 0.0,
            "error/int_mae": _mean_without_nan(user_df["heldout_internal_mae"])
            if not user_df.empty else 0.0,
            # ── popularity ────────────────────────────────────────────────
            "popularity/rec_mean": float(user_df["recommended_popularity_mean"].mean())
            if not user_df.empty else 0.0,
            "popularity/rec_std": float(user_df["recommended_popularity_std"].mean())
            if not user_df.empty else 0.0,
            "popularity/heldout_mean": float(user_df["heldout_popularity_mean"].mean())
            if not user_df.empty else 0.0,
            "popularity/heldout_std": float(user_df["heldout_popularity_std"].mean())
            if not user_df.empty else 0.0,
            "popularity/delta_mean": float(user_df["popularity_mean_delta"].mean())
            if not user_df.empty else 0.0,
            "popularity/delta_std": float(user_df["popularity_mean_delta"].std(ddof=0))
            if not user_df.empty else 0.0,
            "popularity/frac_rec_gt_heldout": float(
                (
                    user_df["recommended_popularity_mean"]
                    > user_df["comparison_popularity_mean"]
                ).mean()
            )
            if not user_df.empty else 0.0,
            # ── score ─────────────────────────────────────────────────────
            "score/rec_mean": float(user_df["heldout_recommender_score_mean"].mean())
            if not user_df.empty else 0.0,
            "score/rec_std": float(user_df["heldout_recommender_score_std"].mean())
            if not user_df.empty else 0.0,
            "score/int_mean": float(user_df["heldout_internal_score_mean"].mean())
            if not user_df.empty else 0.0,
            "score/int_std": float(user_df["heldout_internal_score_std"].mean())
            if not user_df.empty else 0.0,
            "score/gap_mean": float(user_df["heldout_score_mean_gap"].mean())
            if not user_df.empty else 0.0,
            "score/gap_std": float(user_df["heldout_score_mean_gap"].std(ddof=0))
            if not user_df.empty else 0.0,
            "score/frac_rec_gt_int": float(
                (
                    user_df["heldout_recommender_score_mean"]
                    > user_df["heldout_internal_score_mean"]
                ).mean()
            )
            if not user_df.empty else 0.0,
            # ── correlation ───────────────────────────────────────────────
            "correlation/rec_pearson": _mean_without_nan(
                user_df["heldout_recommender_residual_pearson"]
            )
            if not user_df.empty else 0.0,
            "correlation/rec_spearman": _mean_without_nan(
                user_df["heldout_recommender_residual_spearman"]
            )
            if not user_df.empty else 0.0,
            "correlation/int_pearson": _mean_without_nan(
                user_df["heldout_internal_residual_pearson"]
            )
            if not user_df.empty else 0.0,
            "correlation/int_spearman": _mean_without_nan(
                user_df["heldout_internal_residual_spearman"]
            )
            if not user_df.empty else 0.0,
            "correlation/frac_rec_pearson_gt_int": float(
                (
                    user_df["heldout_recommender_residual_pearson"]
                    > user_df["heldout_internal_residual_pearson"]
                ).mean()
            )
            if not user_df.empty else 0.0,
            # ── rating ────────────────────────────────────────────────────
            "rating/heldout_mean": float(user_df["heldout_debiased_rating_mean"].mean())
            if not user_df.empty else 0.0,
            "rating/heldout_std": float(user_df["heldout_debiased_rating_std"].mean())
            if not user_df.empty else 0.0,
        }

        artifact_dir = Path("mlartifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)

        user_diag_path = artifact_dir / "recommender_only_user_diagnostics.parquet"
        user_df.to_parquet(user_diag_path, index=False)
        mlflow.log_artifact(str(user_diag_path), artifact_path="diagnostics")

        recs_path = artifact_dir / "recommender_only_recommendations.parquet"
        rec_df.to_parquet(recs_path, index=False)
        mlflow.log_artifact(str(recs_path), artifact_path="recs")

        heldout_score_rows: list[dict[str, float | int | str]] = []
        for assignment in ctx.assignments.values():
            uid = assignment.sim_user_id
            base_user_id = assignment.base_user_id
            held_out_df = ctx.env.held_out_for_user(
                base_user_id, split=cfg.recommender_eval_split
            )
            held_ids = [int(mid) for mid in held_out_df["movieId"].tolist()]
            ratings_by_movie = {
                int(mid): float(rating)
                for mid, rating in zip(held_out_df["movieId"], held_out_df["rating"])
            }
            ua = ctx.users[uid]
            recommender_user = ctx.env.get_user_factor(base_user_id)
            persona = ua.persona
            recommender_item_vectors = ctx.env.get_item_factors(held_ids)
            internal_item_vectors = ctx.env.get_user_pref_item_factors(held_ids)
            internal_score_map = dict(
                _score_agent_rows(ua.agent, persona, internal_item_vectors, held_ids)
            )

            for mid in held_ids:
                recommender_item = recommender_item_vectors.get(mid)
                heldout_score_rows.append(
                    {
                        "evaluation_split": cfg.recommender_eval_split,
                        "userId": base_user_id,
                        "simulation_user_id": uid,
                        "agent_type": assignment.agent_type,
                        "movieId": mid,
                        "rating": ratings_by_movie[mid],
                        "rating_bias": ctx.env.get_rating_bias(base_user_id, mid),
                        "debiased_rating": ctx.env.debias_rating(
                            base_user_id, mid, ratings_by_movie[mid]
                        ),
                        "recommender_score": float(np.dot(recommender_user, recommender_item))
                        if recommender_user is not None and recommender_item is not None
                        else np.nan,
                        "internal_score": float(internal_score_map[mid])
                        if mid in internal_score_map
                        else np.nan,
                    }
                )

        heldout_scores_path = artifact_dir / "recommender_only_heldout_scores.parquet"
        heldout_scores_df = pd.DataFrame(heldout_score_rows)
        debiased = heldout_scores_df["debiased_rating"].to_numpy(dtype=float)
        rec_scores_arr = heldout_scores_df["recommender_score"].to_numpy(dtype=float)
        int_scores_arr = heldout_scores_df["internal_score"].to_numpy(dtype=float)
        valid_rec_mask = ~(np.isnan(debiased) | np.isnan(rec_scores_arr))
        valid_int_mask = ~(np.isnan(debiased) | np.isnan(int_scores_arr))
        rec_residuals_global = debiased[valid_rec_mask] - rec_scores_arr[valid_rec_mask]
        int_residuals_global = debiased[valid_int_mask] - int_scores_arr[valid_int_mask]
        metrics.update(
            {
                "correlation/global_rec_pearson": _safe_corr(
                    debiased, rec_scores_arr, method="pearson"
                )
                if not heldout_scores_df.empty
                else 0.0,
                "correlation/global_rec_spearman": _safe_corr(
                    debiased, rec_scores_arr, method="spearman"
                )
                if not heldout_scores_df.empty
                else 0.0,
                "correlation/global_int_pearson": _safe_corr(
                    debiased, int_scores_arr, method="pearson"
                )
                if not heldout_scores_df.empty
                else 0.0,
                "correlation/global_int_spearman": _safe_corr(
                    debiased, int_scores_arr, method="spearman"
                )
                if not heldout_scores_df.empty
                else 0.0,
                "error/global_rec_rmse": float(np.sqrt(np.mean(rec_residuals_global ** 2)))
                if len(rec_residuals_global) > 0
                else float("nan"),
                "error/global_rec_mae": float(np.mean(np.abs(rec_residuals_global)))
                if len(rec_residuals_global) > 0
                else float("nan"),
                "error/global_int_rmse": float(np.sqrt(np.mean(int_residuals_global ** 2)))
                if len(int_residuals_global) > 0
                else float("nan"),
                "error/global_int_mae": float(np.mean(np.abs(int_residuals_global)))
                if len(int_residuals_global) > 0
                else float("nan"),
            }
        )
        mlflow.log_metrics(metrics)
        heldout_scores_df.to_parquet(heldout_scores_path, index=False)
        mlflow.log_artifact(str(heldout_scores_path), artifact_path="diagnostics")

        logger.info(
            "  raw hit=%.4f  ndcg@%d=%.4f  rec_pop=%.2f  held_pop=%.2f  "
            "score(rec)=%.4f  score(int)=%.4f  pearson(rec)=%.4f  pearson(int)=%.4f  "
            "rmse(rec)=%.4f  mae(rec)=%.4f",
            metrics["ranking/hit_rate"],
            cfg.rec_list_size,
            metrics[f"ranking/ndcg_at_{cfg.rec_list_size}"],
            metrics["popularity/rec_mean"],
            metrics["popularity/heldout_mean"],
            metrics["score/rec_mean"],
            metrics["score/int_mean"],
            metrics["correlation/rec_pearson"],
            metrics["correlation/int_pearson"],
            metrics["error/rec_rmse"],
            metrics["error/rec_mae"],
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
        explicit_ratings = [
            (mid, act, sig) for mid, act, sig in interactions if act == "rate"
        ]
        agent_feedback = [
            (mid, act, ctx.env.debias_rating(ua.base_user_id, mid, sig))
            for mid, act, sig in explicit_ratings
        ]
        raw_rating_feedback = [(mid, sig) for mid, _, sig in explicit_ratings]

        ctx.recommender.update_user(uid, raw_rating_feedback)
        acted_ids = [mid for mid, _, _ in agent_feedback]
        acted_factors = (
            ctx.env.get_user_pref_item_factors(acted_ids) if agent_feedback else {}
        )
        ua.update(rnd, agent_feedback, acted_factors, self.config)

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
