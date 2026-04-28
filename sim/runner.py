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

        with mlflow.start_run(run_name=f"{cfg.agent_type}-seed{cfg.random_seed}"):
            mlflow.log_params(cfg.as_dict())

            rng = np.random.default_rng(cfg.random_seed)
            ctx = self._setup_components(rng)

            round_records: list[dict] = []
            for rnd in range(1, cfg.num_rounds + 1):
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

        for uid, ua in ctx.users.items():
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
