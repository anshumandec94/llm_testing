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
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from lenskit.data import ItemList

from sim.config import SimConfig
from sim.environment import Environment
from sim.persona import build_population
from sim.recommender import Recommender

logger = logging.getLogger(__name__)


def _build_agent(config: SimConfig, env: Environment):
    """Instantiate the agent specified in config.agent_type."""
    if config.agent_type == "associative":
        from sim.agents.associative import AssociativeAgent
        return AssociativeAgent(env)
    elif config.agent_type == "semantic":
        from sim.agents.semantic import SemanticAgent
        return SemanticAgent(env)
    elif config.agent_type == "seq2seq":
        from sim.agents.seq2seq import Seq2SeqAgent
        return Seq2SeqAgent(env)
    elif config.agent_type == "llm":
        from sim.agents.llm import LLMAgent
        return LLMAgent(env)
    else:
        raise ValueError(f"Unknown agent_type: {config.agent_type!r}")


def _hit_rate(recs: ItemList, held_out_ids: set[int]) -> float:
    if not held_out_ids or len(recs) == 0:
        return 0.0
    rec_ids = {int(iid) for iid in recs.ids()}
    return len(rec_ids & held_out_ids) / len(held_out_ids)


def _ndcg_at_k(recs: ItemList, relevant_ids: set[int], k: int) -> float:
    ids = [int(iid) for iid in recs.ids()][:k]
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, iid in enumerate(ids)
        if iid in relevant_ids
    )
    ideal = sum(
        1.0 / np.log2(rank + 2) for rank in range(min(len(relevant_ids), k))
    )
    return float(dcg / ideal) if ideal > 0 else 0.0


class SimulationRunner:
    """
    Runs one complete simulation experiment.

    Parameters
    ----------
    config:
        Fully specified SimConfig.
    """

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.experiment_name)

    def run(self) -> pd.DataFrame:
        """
        Execute the simulation and return a DataFrame of per-round metrics.
        """
        cfg = self.config
        t0 = time.time()

        with mlflow.start_run(run_name=f"{cfg.agent_type}-seed{cfg.random_seed}"):
            mlflow.log_params(cfg.as_dict())

            # ── Initialise components ─────────────────────────────────────
            logger.info("=== Initialising Environment ===")
            env = Environment(cfg)

            logger.info("=== Initialising Recommender ===")
            recommender = Recommender(cfg, env)

            logger.info("=== Initialising Agent (%s) ===", cfg.agent_type)
            agent = _build_agent(cfg, env)

            logger.info("=== Building Agent Population ===")
            rng = np.random.default_rng(cfg.random_seed)
            population = build_population(cfg, env, rng)

            # Per-user held-out set for quick lookup
            held_out_sets: dict[int, set[int]] = {
                uid: set(env.held_out_for_user(uid)["movieId"].tolist())
                for uid in env.eval_users
            }

            # Track held-out items surfaced across all rounds (for recall)
            surfaced: dict[int, set[int]] = {uid: set() for uid in env.eval_users}

            round_records: list[dict] = []

            # ── Simulation loop ───────────────────────────────────────────
            for rnd in range(1, cfg.num_rounds + 1):
                logger.info("--- Round %d / %d ---", rnd, cfg.num_rounds)

                if rnd > 1:
                    recommender.retrain()

                round_hit_rates: list[float] = []
                round_ndcgs: list[float] = []
                round_holdout_recalls: list[float] = []
                round_attend_flags: list[bool] = []
                round_action_counts: dict[str, int] = {
                    "watch": 0, "rate": 0, "add_to_list": 0
                }
                round_signals: list[float] = []
                round_budgets_consumed: list[float] = []
                all_recs_rows: list[dict] = []

                for uid in env.eval_users:
                    persona = population[uid]
                    held_ids = held_out_sets[uid]

                    # ── Attendance gate ───────────────────────────────────
                    attended = persona.attendance.will_attend(
                        baseline_logit=persona.baseline_logit,
                        recent_signal_ewma=persona.recent_signal_ewma,
                        rounds_since_last_visit=persona.rounds_since_last_visit,
                        rng=rng,
                    )
                    round_attend_flags.append(attended)

                    if not attended:
                        persona.rounds_since_last_visit += 1
                        continue

                    start_budget = persona.budget
                    round_interactions: list[tuple[int, str, float]] = []
                    round_recs_rows: list[dict] = []

                    # ── Inner re-request loop ─────────────────────────────
                    for req in range(cfg.max_requests_per_round):
                        if persona.budget <= 0:
                            break

                        candidates = recommender.recommend(uid, n=cfg.rec_list_size)
                        if len(candidates) == 0:
                            logger.debug(
                                "User %d: no candidates left on request %d.", uid, req + 1
                            )
                            break

                        recommender.mark_sent(uid, candidates)

                        # Fetch item factors in user-pref space
                        candidate_ids = [int(iid) for iid in candidates.ids()]
                        item_factors = env.get_user_pref_item_factors(candidate_ids)

                        # Agent scores candidates
                        ranked = agent.evaluate(candidates, persona, item_factors)

                        # Extract scores aligned with ranked item order
                        ranked_ids = [int(iid) for iid in ranked.ids()]
                        scores_arr = ranked.scores()
                        if scores_arr is None:
                            scores_arr = np.zeros(len(ranked_ids), dtype=np.float32)

                        # Persona samples and selects actions
                        new_interactions = persona.act(
                            ranked_ids=ranked_ids,
                            scores=scores_arr,
                            item_factors=item_factors,
                            config=cfg,
                            rng=rng,
                        )

                        # Deplete budget after this request
                        persona.budget = persona.attention.deplete(
                            len(candidates), persona.budget
                        )

                        round_interactions.extend(new_interactions)

                        for rank, iid in enumerate(ranked_ids):
                            round_recs_rows.append({
                                "round": rnd,
                                "request": req + 1,
                                "userId": uid,
                                "movieId": iid,
                                "rank": rank + 1,
                                "is_held_out": iid in held_ids,
                            })

                        # Stop when enough acted-on interactions collected
                        acted_count = len(round_interactions)
                        if acted_count >= cfg.accept_k:
                            break

                    # ── Post-round updates ────────────────────────────────
                    # Only non-ignore interactions (round_interactions already
                    # excludes "ignore" from persona.act)
                    rec_signals = [(mid, sig) for mid, _act, sig in round_interactions]
                    recommender.update_user(uid, rec_signals)

                    # Preference vector update
                    if round_interactions:
                        all_acted_ids = [mid for mid, _, _ in round_interactions]
                        pref_item_factors = env.get_user_pref_item_factors(all_acted_ids)
                        persona.update_preference(round_interactions, pref_item_factors)

                    agent.update(uid, round_interactions)

                    # Satisfaction and attendance state
                    mean_sig = (
                        float(np.mean([sig for _, _, sig in round_interactions]))
                        if round_interactions else 0.0
                    )
                    persona.recent_signal_ewma = persona.attendance.update_ewma(
                        persona.recent_signal_ewma, mean_sig, cfg.sat_ewma_alpha
                    )
                    persona.rounds_since_last_visit = 0
                    persona.last_attended_round = rnd

                    # Budget recovery for next round
                    persona.budget = persona.attention.restore(persona.budget, mean_sig)

                    # Accumulate population-level stats
                    for _, act, sig in round_interactions:
                        if act in round_action_counts:
                            round_action_counts[act] += 1
                        round_signals.append(sig)

                    budget_consumed = max(0.0, start_budget - persona.budget)
                    round_budgets_consumed.append(budget_consumed)

                    # Track held-out surfaces and metrics
                    seen_this_round = {r["movieId"] for r in round_recs_rows}
                    surfaced[uid].update(seen_this_round & held_ids)

                    first_batch_ids = [
                        r["movieId"] for r in round_recs_rows if r["request"] == 1
                    ]
                    first_il = ItemList(
                        item_ids=np.array(first_batch_ids, dtype=np.int64)
                    )
                    round_hit_rates.append(_hit_rate(first_il, held_ids))
                    round_ndcgs.append(_ndcg_at_k(first_il, held_ids, cfg.rec_list_size))
                    round_holdout_recalls.append(
                        len(surfaced[uid]) / len(held_ids) if held_ids else 0.0
                    )

                    all_recs_rows.extend(round_recs_rows)

                # ── End of round ──────────────────────────────────────────
                recommender.advance_round()

                # Aggregate metrics
                n_attending = sum(round_attend_flags)
                attendance_rate = float(n_attending) / len(env.eval_users)
                mean_hit = float(np.mean(round_hit_rates)) if round_hit_rates else 0.0
                mean_ndcg = float(np.mean(round_ndcgs)) if round_ndcgs else 0.0
                mean_recall = float(np.mean(round_holdout_recalls)) if round_holdout_recalls else 0.0
                mean_signal = float(np.mean(round_signals)) if round_signals else 0.0
                mean_budget_consumed = float(np.mean(round_budgets_consumed)) if round_budgets_consumed else 0.0

                total_actions = sum(round_action_counts.values())
                action_watch_frac = round_action_counts["watch"] / total_actions if total_actions else 0.0
                action_rate_frac = round_action_counts["rate"] / total_actions if total_actions else 0.0
                action_addlist_frac = round_action_counts["add_to_list"] / total_actions if total_actions else 0.0

                logger.info(
                    "  attend=%.2f  hit=%.4f  ndcg@%d=%.4f  recall=%.4f  "
                    "signal=%.2f  watch=%.2f  rate=%.2f  addlist=%.2f",
                    attendance_rate, mean_hit, cfg.rec_list_size, mean_ndcg,
                    mean_recall, mean_signal,
                    action_watch_frac, action_rate_frac, action_addlist_frac,
                )

                metrics = {
                    "attendance_rate": attendance_rate,
                    "hit_rate": mean_hit,
                    f"ndcg_at_{cfg.rec_list_size}": mean_ndcg,
                    "holdout_recall": mean_recall,
                    "mean_signal_strength": mean_signal,
                    "mean_attention_consumed": mean_budget_consumed,
                    "action_watch_frac": action_watch_frac,
                    "action_rate_frac": action_rate_frac,
                    "action_addlist_frac": action_addlist_frac,
                }
                mlflow.log_metrics(metrics, step=rnd)

                round_records.append({"round": rnd, **metrics})

                if all_recs_rows:
                    recs_df = pd.DataFrame(all_recs_rows)
                    artefact_path = (
                        Path("mlartifacts") / f"recs_round_{rnd:03d}.parquet"
                    )
                    artefact_path.parent.mkdir(parents=True, exist_ok=True)
                    recs_df.to_parquet(artefact_path, index=False)
                    mlflow.log_artifact(str(artefact_path), artifact_path="recs")

            # ── Final summary ─────────────────────────────────────────────
            summary_df = pd.DataFrame(round_records)
            summary_path = Path("mlartifacts") / "summary.csv"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(summary_path, index=False)
            mlflow.log_artifact(str(summary_path))

            elapsed = time.time() - t0
            mlflow.log_metric("elapsed_seconds", elapsed)
            logger.info("Simulation complete in %.1f s.", elapsed)

        return summary_df
