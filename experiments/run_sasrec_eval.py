"""
One command for the SASRec evaluation: train, then score against the baselines.

    uv run python experiments/run_sasrec_eval.py --sweep u2566

For one sweep this does, in order:
1. Builds the harness config for the sweep (`compare_backends.sweep_config`),
   so the checkpoint is trained on exactly the split it will be scored on.
2. Trains SASRec into `runs/sasrec_<sweep>/` until validation converges
   (issue #42). After every epoch the checkpoint is scored on the temporal
   validation split, which sits between train and held-out for each eval
   user, and `epoch, val_mae_clipped` is appended to `validation.csv`. The
   best epoch is kept as `best.pt`. Training stops once `--patience` epochs
   pass without improvement, or at `--epochs`, which is only a ceiling.
   Choosing the epoch on validation rather than held-out keeps the test set
   out of model selection. If a checkpoint is already there it resumes, so
   rerunning the same command after a dropped session or a killed job picks
   up where it stopped. A finished run is not retrained.
3. Scores `bias_only`, `associative` and `sasrec` (the best epoch) through the
   harness on the same pairs, one MLflow run each, and prints the comparison
   table. The table is also written to `runs/sasrec_<sweep>/summary.csv`.

It always runs from the repository root, whatever the current directory,
because the split cache key hashes the relative `data_dir` literally; a run
started elsewhere would train on a split the harness then refuses.

Options:
    --epochs N                 ceiling on training epochs (default: the training script's)
    --patience P               stop after P epochs without a validation gain
                               (default 5); 0 trains exactly --epochs and
                               scores the last epoch
    --rating-loss-weight W     0 is the ranking-only ablation; gets its own run dir
    --skip-train               score an existing checkpoint only (best.pt if present)
    --baselines NAME [NAME..]  arms scored beside sasrec (default bias_only associative)
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import shutil
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import compare_backends
from experiments.bias_only_null import ITEM_SELECTION, MAX_ITEMS
import torch

from scripts.train_sasrec import (
    CHECKPOINT_NAME,
    TrainingArgs,
    _check_resumable,
    build_training_data,
    resolve_config,
    train,
)
from sim.environment import Environment
from sim.population import build_user_assignments

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINES = ("bias_only", "associative")
SUMMARY_COLUMNS = [
    "error/mae", "error/mae_clipped", "error/mae_clipped_user_se",
    "meta/item_count", "meta/nan_count", "meta/user_count",
]

BEST_CHECKPOINT_NAME = "best.pt"
VALIDATION_LOG = "validation.csv"
DEFAULT_PATIENCE = 5

logger = logging.getLogger(__name__)


def run_dir_name(sweep: str, rating_loss_weight: float | None) -> str:
    """`sasrec_<sweep>`, with the rating-loss weight appended when it is overridden."""
    suffix = "" if rating_loss_weight is None else f"_rlw{rating_loss_weight:g}"
    return f"sasrec_{sweep}{suffix}"


def validation_mae(env: Environment, assignments, checkpoint: Path, config, max_items: int | None) -> float:
    """Clipped MAE of `checkpoint` on the validation split, scored exactly as the harness scores held-out."""
    backend = compare_backends.build_backend("sasrec", env, checkpoint=checkpoint)
    frame = compare_backends.score_backend(
        env, assignments, backend, max_items, ITEM_SELECTION, seed=config.random_seed, split="validation",
    )
    return compare_backends.compute_metrics(frame)["error/mae_clipped"]


def train_to_convergence(
    env: Environment,
    assignments,
    config,
    training_args: TrainingArgs,
    run_dir: Path,
    patience: int,
    max_items: int | None,
) -> None:
    """Train one epoch at a time, validating each, until `patience` epochs pass without a gain.

    State lives in the run dir, so an interrupted run resumes: `checkpoint.pt`
    (the last epoch, for resuming), `best.pt` (the best validation epoch) and
    `validation.csv` (one row per epoch). An epoch trained but not yet
    validated, because the run was killed in between, is validated first.
    """
    checkpoint = run_dir / CHECKPOINT_NAME
    log_path = run_dir / VALIDATION_LOG
    empty = pd.DataFrame({"epoch": pd.Series(dtype="int64"), "val_mae_clipped": pd.Series(dtype="float64")})
    history = pd.read_csv(log_path) if log_path.is_file() else empty
    epoch = int(torch.load(checkpoint, map_location="cpu", weights_only=False)["epoch"]) if checkpoint.is_file() else 0
    if epoch - len(history) not in (0, 1):
        raise ValueError(
            f"{log_path} records {len(history)} epochs but the checkpoint is at epoch {epoch}; "
            "it was trained without validation (--patience 0). Use a fresh --runs-dir."
        )
    seq_data = build_training_data(env, config)

    while True:
        if epoch > len(history):
            mae = validation_mae(env, assignments, checkpoint, config, max_items)
            if history.empty or mae < history["val_mae_clipped"].min():
                shutil.copyfile(checkpoint, run_dir / BEST_CHECKPOINT_NAME)
            history = pd.concat([history, pd.DataFrame({"epoch": [epoch], "val_mae_clipped": [mae]})], ignore_index=True)
            history.to_csv(log_path, index=False)
            run_id = torch.load(checkpoint, map_location="cpu", weights_only=False)["mlflow_run_id"]
            mlflow.MlflowClient(tracking_uri=config.mlflow_tracking_uri).log_metric(
                run_id, "val/mae_clipped", mae, step=epoch,
            )
            logger.info("epoch %d: validation MAE (clipped) %.4f", epoch, mae)
        if not history.empty:
            best_epoch = int(history.loc[history["val_mae_clipped"].idxmin(), "epoch"])
            if epoch - best_epoch >= patience:
                logger.info("converged: no validation gain in %d epochs since epoch %d", patience, best_epoch)
                return
        if epoch >= training_args.epochs:
            logger.info("stopped at the --epochs ceiling (%d) before converging", training_args.epochs)
            return
        train(seq_data, config, dataclasses.replace(training_args, epochs=epoch + 1), run_dir, resume=checkpoint.is_file())
        epoch += 1


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
    patience: int = DEFAULT_PATIENCE,
) -> pd.DataFrame:
    """Train (or resume) SASRec for `sweep`, score it beside `baselines`, return the table.

    With `patience` > 0, training stops on validation convergence and the
    best validation epoch is scored; with 0, exactly `training_args.epochs`
    are trained and the last one is scored.
    """
    config = resolve_config(compare_backends.sweep_config(sweep), rating_loss_weight)
    run_dir = runs_dir / run_dir_name(sweep, rating_loss_weight)
    checkpoint = run_dir / CHECKPOINT_NAME

    env = Environment(config)
    assignments = build_user_assignments(config, env, np.random.default_rng(config.random_seed))
    if skip_train:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"--skip-train given but {checkpoint} does not exist")
        # The same full-settings check a resume makes, so a checkpoint trained
        # under different settings is never scored as if it matched.
        _check_resumable(torch.load(checkpoint, map_location="cpu", weights_only=False), config, training_args)
    else:
        resume = checkpoint.is_file()
        logger.info("%s SASRec in %s", "Resuming" if resume else "Training", run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if patience > 0:
            train_to_convergence(env, assignments, config, training_args, run_dir, patience, max_items)
        else:
            train(build_training_data(env, config), config, training_args, run_dir, resume=resume)

    best = run_dir / BEST_CHECKPOINT_NAME
    if patience > 0 and best.is_file():
        checkpoint = best
    trained_epoch = int(torch.load(checkpoint, map_location="cpu", weights_only=False)["epoch"])
    if trained_epoch > training_args.epochs:
        logger.warning(
            "checkpoint is at epoch %d, past the requested %d; scoring the epoch-%d model",
            trained_epoch, training_args.epochs, trained_epoch,
        )

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
    table["sasrec_epoch"] = trained_epoch
    table.index.name = "backend"
    table.to_csv(run_dir / "summary.csv")
    return table


def main(argv: list[str] | None = None) -> None:
    defaults = TrainingArgs()
    parser = argparse.ArgumentParser(description="Train SASRec and score it against the baselines.")
    parser.add_argument("--sweep", choices=sorted(compare_backends.SWEEPS), required=True)
    parser.add_argument("--epochs", type=int, default=defaults.epochs, help="Ceiling on training epochs.")
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_PATIENCE,
        help=f"Stop after this many epochs without a validation gain (default {DEFAULT_PATIENCE}); 0 trains exactly --epochs.",
    )
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
        patience=ns.patience,
    )
    with pd.option_context("display.float_format", "{:.4f}".format, "display.max_columns", None, "display.width", None):
        print(table.drop(columns="mlflow_run_id"))


if __name__ == "__main__":
    main()
