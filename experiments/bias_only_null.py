"""
The bias-only null: how much of the published MAE is a lookup table?

The null predicts `env.get_rating_bias(uid, mid)`, i.e. global + user + item
bias and a debiased residual of exactly zero. It models no user-item
interaction at all, so an arm that does not beat it is reproducing a
two-column lookup table rather than representing preference.

It is scored on exactly the pairs the published `llm-agent-comparison` arms
scored, by routing through `select_held_items` with the same cap and
selection. Computing it on any other item set would recreate the confound
epic #1 spent four sub-issues removing.

The associative baseline is re-scored alongside it on the same pairs, for two
reasons. It gives a properly paired null-vs-associative test, which the LLM
arms cannot have because their 2026-06-26 runs logged aggregates only. And it
proves the rebuilt environment is the one the published numbers came from:
if the re-scored associative MAE does not match its published value, the
null is being computed against a different bias model and the comparison is
void, so the script refuses to continue.

Issue #27 adds a third arm on the same pairs, `associative_residual`: the
bias baseline plus an un-normalised TruncatedSVD of the debiased training
residuals (`sim/residual_factors.py`), at the published arm's dimension. It
is the rating-unit version of the associative prediction, and whether it
beats the null is the question that issue asks.

A fourth, `associative_als`, is a secondary diagnostic: the same residuals
and dimension, fitted by ALS on observed cells only. The SVD treats the
unobserved 99.8% of the matrix as zeros and is shrunk by them; the ALS shows
how much of the SVD's small gain is that estimator rather than the
representation. It is one untuned configuration, not a new baseline.

Both residual arms are clipped to [1, 5], so their primary comparison is
against the clamped null. The unclamped null is reported alongside. A
dimension sweep over both is reported as a secondary table.

Intervals cluster by user, the independent sampling unit. Per-item SEs
understate them by about 25% here.

Usage:
    uv run python experiments/bias_only_null.py

Writes per-item predictions and a JSON summary to `reports/bias_only_null/`.
No MLflow run is logged: this is a one-off reference number, not an arm.
"""

import argparse
import dataclasses
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.llm_vs_associative import (
    BASE_CONFIG,
    score_associative,
    score_associative_residual,
    select_held_items,
)
from sim.environment import Environment
from sim.population import build_user_assignments
from sim.residual_factors import (
    ALS_ITERATIONS,
    ALS_REGULARIZATION,
    ResidualFactors,
    fit_residual_als,
    fit_residual_factors,
    training_residuals,
)
from sim.user_agent import SimulatedUser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "bias_only_null"

# The two published sweeps, with the MLflow values each must reproduce.
# `published_capped` is None where no capped associative run was ever logged;
# `published_uncapped` then carries the environment-identity check alone.
SWEEPS: dict[str, dict] = {
    "u128": {
        "eval_user_frac": 0.001,
        "published_capped": 0.791589,
        "published_uncapped": 0.704546,
        "llm_arms": {
            "llm-top_rated-k2": 0.913594,
            "llm-recent-k3": 0.916094,
            "llm-top_rated-k5": 0.933125,
            "llm-polarized-k3-no-fewshot": 0.939063,
            "llm-polarized-k2": 0.985156,
        },
    },
    "u2566": {
        "eval_user_frac": 0.02,
        "published_capped": None,
        "published_uncapped": 0.719526,
        "llm_arms": {"llm-top_rated-k2": 0.893383},
    },
}
MAX_ITEMS = 5
ITEM_SELECTION = "first"
# MLflow stores the published MAEs as float32-ish values rounded on logging;
# anything within this is the same environment.
REPRODUCTION_TOLERANCE = 1e-4
# Secondary: residual-SVD dimensions beyond the published arm's 8. The primary
# result stays at `user_pref_features` so capacity matches the published arm.
RESIDUAL_DIM_SWEEP = (8, 32, 64)
# ALS at k=8 only: the vectorised k=32 fit ran out of memory and swapped for
# an hour on the development machine, so it is not cheap. A secondary sweep.
ALS_DIM_SWEEP = (8,)
# Rating-unit arms, both fitted on the debiased training residuals at the
# published capacity. The SVD is the #27 deliverable; the ALS is a secondary
# diagnostic of how much the SVD's zero-imputation shrinks it.
# Each takes (training residual frame, k, seed). The frame is built once per
# sweep and shared: rebuilding the ~30M-row frame and its sparse matrix for
# every fit is what pushed the 2566-user run into swap.
RESIDUAL_ARMS = {
    "associative_residual": fit_residual_factors,
    "associative_als": fit_residual_als,
}
# Training cells sampled for the in-sample scale check.
IN_SAMPLE_CELLS = 500_000


def score_bias_only(
    env: Environment,
    assignments,
    max_items_per_user: int | None,
    item_selection: str = "first",
    seed: int | None = None,
    split: str = "held_out",
) -> pd.DataFrame:
    """Predict the bias baseline for every selected held-out item.

    Returns one row per scored pair with the unclamped prediction. Clamping
    is left to the caller because it is a reporting choice, not part of the
    model. Pairs come out in the same order `score_associative` produces.
    """
    if seed is None:
        seed = BASE_CONFIG.random_seed
    columns: dict[str, list] = {"userId": [], "movieId": [], "rating": [], "pred_null": []}
    for assignment in assignments:
        base_uid = assignment.base_user_id
        held_out_df = env.held_out_for_user(base_uid, split=split)
        if held_out_df.empty:
            continue
        held_ids = select_held_items(
            held_out_df, max_items_per_user,
            selection=item_selection, user_id=base_uid, seed=seed,
        )
        actual_by_id = {
            int(mid): float(r)
            for mid, r in zip(held_out_df["movieId"], held_out_df["rating"])
        }
        for mid in held_ids:
            columns["userId"].append(base_uid)
            columns["movieId"].append(mid)
            columns["rating"].append(actual_by_id[mid])
            columns["pred_null"].append(env.get_rating_bias(base_uid, mid))
    return pd.DataFrame(columns)


def clustered_mae(frame: pd.DataFrame, pred_col: str) -> dict:
    """MAE with a 95% interval clustered by user.

    `mae` is the item micro-average, which is what MLflow logs as
    `error/mae`. `user_mae` weights every user equally, and the SE is the SE
    of the per-user means. Under a per-user cap with every user at the cap
    the two coincide.
    """
    abs_err = (frame[pred_col] - frame["rating"]).abs()
    per_user = abs_err.groupby(frame["userId"]).mean()
    se = float(per_user.std(ddof=1) / np.sqrt(len(per_user)))
    user_mae = float(per_user.mean())
    return {
        "mae": float(abs_err.mean()),
        "user_mae": user_mae,
        "se": se,
        "ci95": [user_mae - 1.96 * se, user_mae + 1.96 * se],
        "rmse": float(np.sqrt(((frame[pred_col] - frame["rating"]) ** 2).mean())),
        "n_users": int(len(per_user)),
        "n_items": int(len(frame)),
    }


def paired_difference(frame: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Per-user paired difference in MAE, `col_a` minus `col_b`.

    Positive means `col_a` has the larger error. Users are the unit, so this
    is the test the report's recency analysis used.
    """
    err_a = (frame[col_a] - frame["rating"]).abs()
    err_b = (frame[col_b] - frame["rating"]).abs()
    diff = (err_a - err_b).groupby(frame["userId"]).mean()
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    return {
        "diff": mean,
        "se": se,
        "ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "t": mean / se if se > 0 else float("nan"),
        "n_users": int(len(diff)),
    }


def _check_reproduces(name: str, got: float, published: float | None) -> None:
    if published is None:
        return
    if abs(got - published) > REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            f"{name}: re-scored MAE {got:.6f} does not match published {published:.6f}. "
            "The rebuilt environment is not the one the published arms ran on, so "
            "the null would be compared against a different bias model. Stopping."
        )
    logger.info("%s reproduces: %.6f vs published %.6f", name, got, published)


def _selection_robustness(env, assignments, users, cfg, fitted: dict[str, ResidualFactors]) -> dict:
    """Every associative arm minus the null on the other item selections.

    The published comparison is on each user's five most recent held-out
    items. This checks the ordering of each arm against the null is not an
    artefact of that slice. The published arm's keys stay at the top level
    for continuity with #26; each rating-unit arm gets its own block.
    """
    out = {}
    for name, cap, selection in [("all", None, "first"), ("random-5", 5, "random")]:
        frame = score_bias_only(
            env, assignments, cap, item_selection=selection,
            seed=cfg.random_seed, split=cfg.recommender_eval_split,
        )
        frame["pred_null_clamped"] = frame["pred_null"].clip(1.0, 5.0)
        predicted, _, _ = score_associative(
            env, assignments, users,
            max_items_per_user=cap, item_selection=selection, seed=cfg.random_seed,
        )
        frame["pred_associative"] = predicted
        row = {
            "null_mae": clustered_mae(frame, "pred_null")["mae"],
            "null_clamped_mae": clustered_mae(frame, "pred_null_clamped")["mae"],
            "associative_mae": clustered_mae(frame, "pred_associative")["mae"],
            **paired_difference(frame, "pred_associative", "pred_null"),
        }
        for arm, factors in fitted.items():
            _add_residual_arm(env, assignments, factors, frame, cap, selection, cfg.random_seed, f"pred_{arm}")
            row[f"{arm}_mae"] = clustered_mae(frame, f"pred_{arm}")["mae"]
            row[f"{arm}_minus_null_clamped"] = paired_difference(frame, f"pred_{arm}", "pred_null_clamped")
            row[f"{arm}_minus_null"] = paired_difference(frame, f"pred_{arm}", "pred_null")
            row[f"{arm}_term"] = _residual_term_diagnostics(factors, frame)
        out[name] = row
    return out


def _add_residual_arm(
    env, assignments, factors: ResidualFactors, frame: pd.DataFrame,
    cap: int | None, selection: str, seed: int, column: str = "pred_associative_residual",
) -> None:
    """Score a rating-unit arm into `frame`, refusing if its pairs differ."""
    predicted, actual, pairs = score_associative_residual(
        env, assignments, factors,
        max_items_per_user=cap, item_selection=selection, seed=seed,
    )
    if pairs != list(zip(frame["userId"], frame["movieId"])):
        raise RuntimeError(f"null and {column} scored different pairs")
    if not np.allclose(actual, frame["rating"]):
        raise RuntimeError(f"null and {column} disagree on the actual ratings")
    frame[column] = predicted


def _residual_term_diagnostics(
    factors: ResidualFactors, frame: pd.DataFrame, train: pd.DataFrame | None = None,
) -> dict:
    """The same diagnostics as `_dot_term_diagnostics`, for a rating-unit arm.

    The term is `U[user] @ V[item]`, unclipped. A term in the right units has
    `ls_scale` near 1: below 1 means it is too large or noisy, above 1 means
    it is too timid (shrunk). `mean_term` near 0 says it adds no systematic
    offset, which is what the published arm's +0.39 stars failed.

    With `train`, the same scale and correlation are also measured in-sample
    on a fixed random sample of training cells, which separates estimator
    shrinkage (present in-sample) from a generalisation gap (held-out only).
    """
    terms = factors.residuals(frame["userId"].to_numpy(), frame["movieId"].to_numpy())
    residual = (frame["rating"] - frame["pred_null"]).to_numpy()
    covered = factors.covered(frame["userId"].to_numpy(), frame["movieId"].to_numpy())
    out = {
        "n_components": factors.n_components,
        "pairs_without_factors": int((~covered).sum()),
        "mean_term": float(terms.mean()),
        "term_std": float(terms.std()),
        "residual_std": float(residual.std()),
        "corr_term_residual": float(np.corrcoef(terms[covered], residual[covered])[0, 1]),
        "ls_scale": float(terms @ residual / (terms @ terms)),
    }
    if train is not None:
        idx = np.random.default_rng(0).choice(
            len(train), size=min(IN_SAMPLE_CELLS, len(train)), replace=False,
        )
        sample = train.iloc[idx]
        t_in = factors.residuals(sample["userId"].to_numpy(), sample["movieId"].to_numpy())
        y_in = sample["residual"].to_numpy()
        out["in_sample_cells"] = int(len(sample))
        out["in_sample_ls_scale"] = float(t_in @ y_in / (t_in @ t_in))
        out["in_sample_corr"] = float(np.corrcoef(t_in, y_in)[0, 1])
    return out


def _residual_dimension_sweep(env, assignments, cfg, train: pd.DataFrame) -> dict:
    """Secondary: both rating-unit arms at other dimensions, recent-5 and all.

    Not the primary result, which is fixed at the published arm's capacity
    so the comparison changes target and scale only. ALS is swept at k=8 only
    because its memory grows with k squared; see ALS_DIM_SWEEP.
    """
    frames = {
        name: score_bias_only(
            env, assignments, cap, item_selection="first",
            seed=cfg.random_seed, split=cfg.recommender_eval_split,
        )
        for name, cap in [("first-5", MAX_ITEMS), ("all", None)]
    }
    for frame in frames.values():
        frame["pred_null_clamped"] = frame["pred_null"].clip(1.0, 5.0)
    plan: list[tuple[str, int, Callable[[pd.DataFrame, int, int], ResidualFactors]]] = [
        ("associative_residual", k, fit_residual_factors) for k in RESIDUAL_DIM_SWEEP
    ]
    plan += [("associative_als", k, fit_residual_als) for k in ALS_DIM_SWEEP]
    out: dict[str, dict] = {}
    for arm, k, fit in plan:
        factors = fit(train, k, cfg.random_seed)
        row: dict[str, dict] = {}
        for name, frame in frames.items():
            cap = MAX_ITEMS if name == "first-5" else None
            _add_residual_arm(env, assignments, factors, frame, cap, "first", cfg.random_seed, "pred_arm")
            row[name] = {
                "mae": clustered_mae(frame, "pred_arm")["mae"],
                "minus_null_clamped": paired_difference(frame, "pred_arm", "pred_null_clamped"),
                "minus_null": paired_difference(frame, "pred_arm", "pred_null"),
                "ls_scale": _residual_term_diagnostics(factors, frame)["ls_scale"],
            }
        # SVD keys keep their #27 names (`k8`); ALS keys are prefixed.
        out[f"k{k}" if arm == "associative_residual" else f"als_k{k}"] = row
        logger.info(
            "%s k=%d: first-5 MAE %.4f (minus clamped null %+.4f), all %.4f (%+.4f)",
            arm, k, row["first-5"]["mae"], row["first-5"]["minus_null_clamped"]["diff"],
            row["all"]["mae"], row["all"]["minus_null_clamped"]["diff"],
        )
    return out


def _dot_term_diagnostics(env, assignments, users, frame: pd.DataFrame) -> dict:
    """How much held-out signal the associative dot term carries.

    The associative arm is `bias + dot`, so it can only lose to `bias` if the
    dot term is more noise than signal. `ls_scale` is the least-squares
    coefficient of the true residual on the dot term: 1.0 would mean the term
    is on the right scale, and 0 would mean it is pure noise.
    """
    pref = {a.base_user_id: users[a.sim_user_id].persona.pref_vector for a in assignments}
    factors = env.get_user_pref_item_factors([int(m) for m in frame["movieId"].unique()])
    dots = np.array([
        float(np.dot(pref[u], factors[m])) if m in factors else np.nan
        for u, m in zip(frame["userId"], frame["movieId"])
    ])
    residual = (frame["rating"] - frame["pred_null"]).to_numpy()
    ok = ~np.isnan(dots)
    return {
        "items_without_factors": int((~ok).sum()),
        "dot_std": float(dots[ok].std()),
        "residual_std": float(residual.std()),
        "corr_dot_residual": float(np.corrcoef(dots[ok], residual[ok])[0, 1]),
        "ls_scale": float(dots[ok] @ residual[ok] / (dots[ok] @ dots[ok])),
    }


def run_sweep(label: str, spec: dict) -> dict:
    cfg = dataclasses.replace(BASE_CONFIG, eval_user_frac=spec["eval_user_frac"])
    logger.info("[%s] building environment (eval_user_frac=%s)", label, cfg.eval_user_frac)
    env = Environment(cfg)
    rng = np.random.default_rng(cfg.random_seed)
    assignments = build_user_assignments(cfg, env, rng)
    users, _ = SimulatedUser.build_population(cfg, env, rng, assignments=assignments)

    # Environment identity: the uncapped associative run must come back exact.
    predicted, actual, _ = score_associative(env, assignments, users, max_items_per_user=None)
    uncapped = float(np.mean(np.abs(np.array(predicted) - np.array(actual))))
    _check_reproduces(f"{label} associative uncapped", uncapped, spec["published_uncapped"])

    frame = score_bias_only(
        env, assignments, MAX_ITEMS,
        item_selection=ITEM_SELECTION, seed=cfg.random_seed,
        split=cfg.recommender_eval_split,
    )
    assoc_pred, assoc_actual, assoc_pairs = score_associative(
        env, assignments, users,
        max_items_per_user=MAX_ITEMS, item_selection=ITEM_SELECTION, seed=cfg.random_seed,
    )
    if assoc_pairs != list(zip(frame["userId"], frame["movieId"])):
        raise RuntimeError(f"{label}: null and associative scored different pairs")
    if not np.allclose(assoc_actual, frame["rating"]):
        raise RuntimeError(f"{label}: null and associative disagree on the actual ratings")
    frame["pred_null_clamped"] = frame["pred_null"].clip(1.0, 5.0)
    frame["pred_associative"] = assoc_pred

    # Issue #27: the rating-unit arms, at the published capacity.
    train = training_residuals(env)
    k = cfg.user_pref_features
    fitted = {arm: fit(train, k, cfg.random_seed) for arm, fit in RESIDUAL_ARMS.items()}
    for arm, factors in fitted.items():
        _add_residual_arm(
            env, assignments, factors, frame, MAX_ITEMS, ITEM_SELECTION, cfg.random_seed, f"pred_{arm}",
        )
    users_col, items_col = frame["userId"].to_numpy(), frame["movieId"].to_numpy()
    frame["residual_term"] = fitted["associative_residual"].residuals(users_col, items_col)
    frame["als_term"] = fitted["associative_als"].residuals(users_col, items_col)

    null = clustered_mae(frame, "pred_null")
    null_clamped = clustered_mae(frame, "pred_null_clamped")
    assoc = clustered_mae(frame, "pred_associative")
    _check_reproduces(f"{label} associative capped", assoc["mae"], spec["published_capped"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / f"predictions_{label}.csv", index=False)

    # Unpaired, because the LLM runs logged aggregates only. Positive means
    # the arm has the larger error.
    llm_rows = {
        name: {"mae": mae, "arm_minus_null": mae - null["mae"]}
        for name, mae in spec["llm_arms"].items()
    }
    return {
        "eval_user_frac": cfg.eval_user_frac,
        "max_items_per_user": MAX_ITEMS,
        "item_selection": ITEM_SELECTION,
        "associative_uncapped_mae": uncapped,
        "null": null,
        "null_clamped": null_clamped,
        "associative_capped": assoc,
        "associative_minus_null": paired_difference(frame, "pred_associative", "pred_null"),
        "llm_arms": llm_rows,
        "associative_minus_null_clamped": paired_difference(
            frame, "pred_associative", "pred_null_clamped"
        ),
        "associative_minus_null_by_selection": _selection_robustness(
            env, assignments, users, cfg, fitted
        ),
        "dot_term": _dot_term_diagnostics(env, assignments, users, frame),
        # The rating-unit arms are clipped, so the clamped null is their
        # primary comparison; the unclamped one is kept alongside.
        "associative_residual_capped": clustered_mae(frame, "pred_associative_residual"),
        "associative_residual_minus_null_clamped": paired_difference(
            frame, "pred_associative_residual", "pred_null_clamped"
        ),
        "associative_residual_minus_null": paired_difference(
            frame, "pred_associative_residual", "pred_null"
        ),
        "associative_residual_minus_associative": paired_difference(
            frame, "pred_associative_residual", "pred_associative"
        ),
        "residual_term": _residual_term_diagnostics(fitted["associative_residual"], frame, train),
        "associative_als_secondary": {
            "config": {
                "n_components": fitted["associative_als"].n_components,
                "regularization": ALS_REGULARIZATION,
                "iterations": ALS_ITERATIONS,
                "random_state": cfg.random_seed,
                "tuned": False,
            },
            "capped": clustered_mae(frame, "pred_associative_als"),
            "minus_null_clamped": paired_difference(frame, "pred_associative_als", "pred_null_clamped"),
            "minus_null": paired_difference(frame, "pred_associative_als", "pred_null"),
            "minus_associative_residual": paired_difference(
                frame, "pred_associative_als", "pred_associative_residual"
            ),
            "term": _residual_term_diagnostics(fitted["associative_als"], frame, train),
        },
        "residual_dimension_sweep_secondary": _residual_dimension_sweep(env, assignments, cfg, train),
        "out_of_range_null_predictions": int(
            ((frame["pred_null"] < 1.0) | (frame["pred_null"] > 5.0)).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the bias-only null on the published arms' pairs.")
    parser.add_argument(
        "--sweep", choices=sorted(SWEEPS), action="append",
        help="Sweep(s) to compute. Default is both.",
    )
    args = parser.parse_args()
    labels = args.sweep or list(SWEEPS)

    # Merge into any existing summary so a single-sweep run does not drop the other.
    summary_path = OUTPUT_DIR / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update({label: run_sweep(label, SWEEPS[label]) for label in labels})
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
