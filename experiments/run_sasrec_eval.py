"""
One command for the SASRec evaluation: train, then score against the baselines.

    uv run python experiments/run_sasrec_eval.py --sweep u2566

For one sweep this does, in order:
1. Builds the harness config for the sweep (`compare_backends.sweep_config`),
   so the checkpoint is trained on exactly the split it will be scored on.
2. Trains SASRec into `runs/sasrec_<sweep>/`. If a checkpoint is already
   there it resumes, so rerunning the same command after a dropped session
   or a killed job picks up where it stopped. A finished run is not retrained.
3. Scores `bias_only`, `associative` and `sasrec` through the harness on the
   same pairs, one MLflow run each, and prints the comparison table. The table
   is also written to `runs/sasrec_<sweep>/summary.csv`.

It always runs from the repository root, whatever the current directory,
because the split cache key hashes the relative `data_dir` literally; a run
started elsewhere would train on a split the harness then refuses.

Options:
    --epochs N                 training epochs (default: the training script's)
    --rating-loss-weight W     0 is the ranking-only ablation; gets its own run dir
    --skip-train               score an existing checkpoint only
    --baselines NAME [NAME..]  arms scored beside sasrec (default bias_only associative)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import compare_backends
from experiments.bias_only_null import ITEM_SELECTION, MAX_ITEMS
from scripts.train_sasrec import CHECKPOINT_NAME, TrainingArgs, build_training_data, resolve_config, train
from sim.environment import Environment
from sim.population import build_user_assignments

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINES = ("bias_only", "associative")
SUMMARY_COLUMNS = [
    "error/mae", "error/mae_clipped", "error/mae_clipped_user_se",
    "meta/item_count", "meta/nan_count", "meta/user_count",
]

logger = logging.getLogger(__name__)


def run_dir_name(sweep: str, rating_loss_weight: float | None) -> str:
    """`sasrec_<sweep>`, with the rating-loss weight appended when it is overridden."""
    suffix = "" if rating_loss_weight is None else f"_rlw{rating_loss_weight:g}"
    return f"sasrec_{sweep}{suffix}"


def run_eval(
    sweep: str,
    *,
    runs_dir: Path,
    training_args: TrainingArgs,
    rating_loss_weight: float | None = None,
    skip_train: bool = False,
    baselines: tuple[str, ...] = DEFAULT_BASELINES,
    mlflow_uri: str = compare_backends.MLFLOW_URI,
    max_items: int | None = MAX_ITEMS,
) -> pd.DataFrame:
    """Train (or resume) SASRec for `sweep`, score it beside `baselines`, return the table."""
    config = resolve_config(compare_backends.sweep_config(sweep), rating_loss_weight)
    run_dir = runs_dir / run_dir_name(sweep, rating_loss_weight)
    checkpoint = run_dir / CHECKPOINT_NAME

    env = Environment(config)
    if skip_train:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"--skip-train given but {checkpoint} does not exist")
    else:
        resume = checkpoint.is_file()
        logger.info("%s SASRec in %s", "Resuming" if resume else "Training", run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        train(build_training_data(env, config), config, training_args, run_dir, resume=resume)

    assignments = build_user_assignments(config, env, np.random.default_rng(config.random_seed))
    cap = f"{ITEM_SELECTION}{max_items}" if max_items else "all"
    rows = {}
    for name in (*baselines, "sasrec"):
        backend = compare_backends.build_backend(
            name, env, checkpoint=checkpoint if name == "sasrec" else None,
        )
        run_id, metrics = compare_backends.run_backend(
            config, env, assignments, backend,
            max_items_per_user=max_items, item_selection=ITEM_SELECTION,
            tracking_uri=mlflow_uri, run_name=f"{name}-{sweep}-{cap}",
        )
        rows[name] = {**{k: metrics[k] for k in SUMMARY_COLUMNS}, "mlflow_run_id": run_id}

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "backend"
    table.to_csv(run_dir / "summary.csv")
    return table


def main(argv: list[str] | None = None) -> None:
    defaults = TrainingArgs()
    parser = argparse.ArgumentParser(description="Train SASRec and score it against the baselines.")
    parser.add_argument("--sweep", choices=sorted(compare_backends.SWEEPS), required=True)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--rating-loss-weight", type=float, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--baselines", nargs="*", default=list(DEFAULT_BASELINES),
        choices=[b for b in compare_backends.BUILDABLE if b != "sasrec"],
    )
    parser.add_argument("--mlflow-uri", default=compare_backends.MLFLOW_URI)
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    ns = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runs_dir = ns.runs_dir.resolve()
    os.chdir(REPO_ROOT)
    table = run_eval(
        ns.sweep,
        runs_dir=runs_dir,
        training_args=TrainingArgs(epochs=ns.epochs),
        rating_loss_weight=ns.rating_loss_weight,
        skip_train=ns.skip_train,
        baselines=tuple(ns.baselines),
        mlflow_uri=ns.mlflow_uri,
    )
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(table.drop(columns="mlflow_run_id"))


if __name__ == "__main__":
    main()
