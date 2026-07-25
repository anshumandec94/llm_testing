"""Config-driven recommender HPO using the existing recommender_only workflow."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from sim.config import SimConfig
from sim.runner import SimulationRunner

logger = logging.getLogger(__name__)


@dataclass
class HPOConfig:
    """Search configuration for recommender hyperparameter optimization."""

    base_config: SimConfig
    candidate_overrides: list[dict[str, Any]]
    experiment_name: str | None = None
    selection_metric: str = "error/rec_mse"
    # Set True when the selection_metric should be minimized (e.g. MSE).
    # Set False when it should be maximized (e.g. NDCG).
    minimize_selection_metric: bool = True
    validation_split: str = "validation"
    final_eval_split: str = "held_out"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HPOConfig":
        unknown = sorted(set(payload) - {
            "base_config",
            "candidate_overrides",
            "experiment_name",
            "selection_metric",
            "minimize_selection_metric",
            "validation_split",
            "final_eval_split",
        })
        if unknown:
            raise ValueError(f"Unknown HPOConfig fields: {unknown}")
        if "base_config" not in payload:
            raise ValueError("HPOConfig requires 'base_config'.")
        if "candidate_overrides" not in payload:
            raise ValueError("HPOConfig requires 'candidate_overrides'.")

        return cls(
            base_config=SimConfig.from_dict(payload["base_config"]),
            candidate_overrides=list(payload["candidate_overrides"]),
            experiment_name=payload.get("experiment_name"),
            selection_metric=payload.get("selection_metric", "error/rec_mse"),
            minimize_selection_metric=payload.get("minimize_selection_metric", True),
            validation_split=payload.get("validation_split", "validation"),
            final_eval_split=payload.get("final_eval_split", "held_out"),
        )

    @classmethod
    def from_json_file(cls, path: Path) -> "HPOConfig":
        return cls.from_dict(json.loads(path.read_text()))

    def resolved_experiment_name(self) -> str:
        return self.experiment_name or f"{self.base_config.experiment_name}-hpo"

    def resolved_selection_metric(self, config: SimConfig) -> str:
        if self.selection_metric == "ndcg_at_k":
            return f"ndcg_at_{config.rec_list_size}"
        return self.selection_metric


@dataclass
class HPOResult:
    """Artifacts and selected config from an HPO run."""

    candidate_results: pd.DataFrame
    best_config: SimConfig
    best_validation_summary: pd.DataFrame
    best_final_summary: pd.DataFrame
    search_results_path: Path
    best_config_path: Path


def _candidate_config(
    base_config: SimConfig,
    overrides: dict[str, Any],
    *,
    experiment_name: str,
    eval_split: str,
) -> SimConfig:
    return replace(
        base_config,
        **overrides,
        experiment_name=experiment_name,
        experiment_profile="recommender_only",
        recommender_eval_split=eval_split,
    )


def _selection_key(
    row: pd.Series, metric_name: str, minimize: bool
) -> tuple[float, float, float, float]:
    # Negate the primary metric when minimizing so max() always picks the best candidate.
    primary = -float(row[metric_name]) if minimize else float(row[metric_name])
    return (
        primary,
        float(row["ranking/frac_users_with_hit"]),
        float(row["correlation/rec_pearson"]),
        -abs(float(row["popularity/delta_mean"])),
    )


def run_hpo(
    hpo_config: HPOConfig,
    *,
    artifact_dir: Path | None = None,
) -> HPOResult:
    """Run validation-set HPO and confirm the best config on the final split."""
    experiment_name = hpo_config.resolved_experiment_name()
    base_config = replace(hpo_config.base_config, experiment_name=experiment_name)
    metric_name = hpo_config.resolved_selection_metric(base_config)
    artifact_dir = artifact_dir or Path("mlartifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(base_config.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    candidate_rows: list[dict[str, Any]] = []
    candidate_summaries: list[pd.DataFrame] = []

    with mlflow.start_run(run_name="hpo-search"):
        mlflow.set_tags(
            {
                "run_kind": "hpo",
                "selection_metric": metric_name,
                "validation_split": hpo_config.validation_split,
                "final_eval_split": hpo_config.final_eval_split,
            }
        )
        mlflow.log_param("candidate_count", len(hpo_config.candidate_overrides))

        for index, overrides in enumerate(hpo_config.candidate_overrides):
            candidate_cfg = _candidate_config(
                base_config,
                overrides,
                experiment_name=experiment_name,
                eval_split=hpo_config.validation_split,
            )
            candidate_run_name = f"hpo-candidate-{index:03d}"
            with mlflow.start_run(run_name=candidate_run_name, nested=True):
                summary = SimulationRunner(candidate_cfg).run(
                    manage_mlflow=False,
                    extra_tags={
                        "hpo_role": "candidate",
                        "candidate_index": str(index),
                        "candidate_overrides": json.dumps(overrides, sort_keys=True),
                    },
                )

            candidate_summaries.append(summary)
            row = summary.iloc[0].to_dict()
            row["candidate_index"] = index
            row["evaluation_split"] = hpo_config.validation_split
            row["candidate_overrides"] = json.dumps(overrides, sort_keys=True)
            row.update({f"param_{key}": value for key, value in overrides.items()})
            candidate_rows.append(row)

        results_df = pd.DataFrame(candidate_rows)
        if results_df.empty:
            raise ValueError("HPO search produced no candidate results.")

        best_index = max(
            range(len(candidate_rows)),
            key=lambda idx: _selection_key(
                results_df.iloc[idx], metric_name, hpo_config.minimize_selection_metric
            ),
        )
        best_overrides = hpo_config.candidate_overrides[best_index]
        best_validation_summary = candidate_summaries[best_index]

        best_cfg = _candidate_config(
            base_config,
            best_overrides,
            experiment_name=experiment_name,
            eval_split=hpo_config.final_eval_split,
        )
        with mlflow.start_run(run_name="hpo-best-final", nested=True):
            best_final_summary = SimulationRunner(best_cfg).run(
                manage_mlflow=False,
                extra_tags={
                    "hpo_role": "best_final",
                    "candidate_index": str(best_index),
                    "candidate_overrides": json.dumps(best_overrides, sort_keys=True),
                },
            )

        results_df["selection_metric"] = metric_name
        results_df["minimize_selection_metric"] = hpo_config.minimize_selection_metric
        results_df["selection_score"] = results_df[metric_name].astype(float)
        results_df["abs_mean_user_popularity_mean_delta"] = (
            results_df["popularity/delta_mean"].astype(float).abs()
        )
        results_df = results_df.sort_values(
            by=[
                metric_name,
                "ranking/frac_users_with_hit",
                "correlation/rec_pearson",
                "abs_mean_user_popularity_mean_delta",
            ],
            ascending=[hpo_config.minimize_selection_metric, False, False, True],
        ).reset_index(drop=True)

        results_path = artifact_dir / "hpo_candidate_results.csv"
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(str(results_path), artifact_path="hpo")

        best_config_path = artifact_dir / "best_recommender_config.json"

        # Determine which params this run actually tuned so we do a targeted merge.
        tuned_keys: set[str] = {
            k for overrides in hpo_config.candidate_overrides for k in overrides
        }

        # Preserve any previously saved params from OTHER HPO runs (e.g. SVD dims after MF run).
        existing_params: dict[str, Any] = {}
        if best_config_path.exists():
            try:
                existing_params = json.loads(best_config_path.read_text())
            except Exception:
                pass

        winner_tuned = {k: getattr(best_cfg, k) for k in tuned_keys}
        merged_params = {**existing_params, **winner_tuned}
        best_config_path.write_text(json.dumps(merged_params, indent=2, sort_keys=True))
        mlflow.log_artifact(str(best_config_path), artifact_path="hpo")

        best_validation_path = artifact_dir / "best_validation_summary.csv"
        best_validation_summary.to_csv(best_validation_path, index=False)
        mlflow.log_artifact(str(best_validation_path), artifact_path="hpo")

        best_final_path = artifact_dir / "best_final_summary.csv"
        best_final_summary.to_csv(best_final_path, index=False)
        mlflow.log_artifact(str(best_final_path), artifact_path="hpo")

        best_final_row = best_final_summary.iloc[0]
        mlflow.log_metrics(
            {
                f"best_validation_{metric_name}": float(
                    best_validation_summary.iloc[0][metric_name]
                ),
                "best_validation_frac_users_with_hit": float(
                    best_validation_summary.iloc[0]["ranking/frac_users_with_hit"]
                ),
                f"best_final_{metric_name}": float(best_final_row[metric_name]),
                "best_final_frac_users_with_hit": float(
                    best_final_row["ranking/frac_users_with_hit"]
                ),
            }
        )

    return HPOResult(
        candidate_results=results_df,
        best_config=best_cfg,
        best_validation_summary=best_validation_summary,
        best_final_summary=best_final_summary,
        search_results_path=results_path,
        best_config_path=best_config_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run recommender HPO using recommender_only validation diagnostics.",
    )
    parser.add_argument("config_file", type=Path, help="Path to an HPO JSON config file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_hpo(HPOConfig.from_json_file(args.config_file))
    print("\n=== HPO Candidate Results ===")
    print(result.candidate_results.to_string(index=False))
    print(f"\nBest config saved to: {result.best_config_path}")


if __name__ == "__main__":
    main()
