"""
Merge comparison-harness runs from separate MLflow stores into one table.

Each arm of the backend comparison is one `experiments/compare_backends.py`
run, and arms may be scored on different machines (the mlx LLM on a laptop,
SASRec on a GPU box), so they land in different stores. This script reads
each run's `per_item_predictions.parquet` and `scored_pairs.csv` and puts the
arms side by side without re-running anything.

It refuses, rather than warns, unless every arm scored exactly the same
(userId, movieId) pairs against the same actual ratings. Equal
`meta/item_count` is not equal pairs. Every run must also carry the same
split fingerprint (`split_cache_key`): pairs and ratings come from the
held-out set, so arms trained on different splits can still match on both.

A run is given either by its store and id (`--run URI RUN_ID`), for a store
this machine can read, or by a directory holding its downloaded artifacts
(`--artifacts DIR`), for a run copied from another machine. A copied
`mlflow.db` does not help: it records artifact paths on the machine that
wrote it. Download on that machine instead:

    uv run mlflow artifacts download --run-id <id> --dst-path runs/sasrec-u128

Usage:
    uv run python scripts/merge_runs.py \\
        --run sqlite:///$PWD/mlflow.db 0123abcd... \\
        --artifacts runs/sasrec-u128 \\
        --out reports/backend_comparison

Writes, with `--out`:
    summary.csv           one row per arm: label, backend, source, the
                          harness metrics recomputed from the predictions,
                          and `common/` metrics over only the pairs every arm
                          scored, the like-for-like ranking when arms differ
                          in which pairs they returned nan for
    per_pair.parquet      one row per pair: userId, movieId, actual_rating,
                          and one predicted_rating column per arm
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.compare_backends import SPLIT_FILE, compute_metrics  # noqa: E402
from sim.config import check_tracking_uri  # noqa: E402

PREDICTIONS_FILE = "per_item_predictions.parquet"
PAIRS_FILE = "scored_pairs.csv"
PAIR_COLUMNS = ["userId", "movieId"]


class RunMismatchError(ValueError):
    """Two arms cannot sit in one table: different pairs, ratings or split."""


@dataclass
class ArmRun:
    """One harness run's artifacts, and where they came from."""

    label: str
    backend: str
    source: str
    predictions: pd.DataFrame
    scored_pairs: pd.DataFrame
    split_cache_key: str


def load_from_artifacts(directory: Path, split_cache_key: str | None = None, source: str | None = None) -> ArmRun:
    """Read one run from a directory of its downloaded artifacts.

    The split fingerprint comes from `split.json` unless `split_cache_key` is
    given (from the run's params). Without either the run is refused: it
    could have been scored on any split.
    """
    predictions_path = directory / PREDICTIONS_FILE
    pairs_path = directory / PAIRS_FILE
    split_path = directory / SPLIT_FILE
    if split_cache_key is None:
        if not split_path.is_file():
            raise FileNotFoundError(
                f"{split_path} is missing, so the split this run was scored on is unknown. "
                "Runs from before split.json was logged can be merged with --run instead."
            )
        split_cache_key = str(json.loads(split_path.read_text())["split_cache_key"])
    for path in (predictions_path, pairs_path):
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; is {directory} a compare_backends run's artifacts?")
    predictions = pd.read_parquet(predictions_path)
    backends = predictions["backend"].unique()
    if len(backends) != 1:
        raise ValueError(f"{predictions_path} mixes backends {sorted(backends)}; expected one arm per run")
    source = source or directory.name
    backend = str(backends[0])
    return ArmRun(
        label=f"{backend}:{source}",
        backend=backend,
        source=source,
        predictions=predictions,
        scored_pairs=pd.read_csv(pairs_path),
        split_cache_key=split_cache_key,
    )


def load_from_mlflow(tracking_uri: str, run_id: str) -> ArmRun:
    """Download one run's artifacts from `tracking_uri` and read them."""
    check_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    params = client.get_run(run_id).data.params
    if "split_cache_key" not in params:
        raise ValueError(f"run {run_id} in {tracking_uri} has no split_cache_key param; is it a harness run?")
    with tempfile.TemporaryDirectory() as tmp:
        for name in (PREDICTIONS_FILE, PAIRS_FILE):
            client.download_artifacts(run_id, name, tmp)
        return load_from_artifacts(Path(tmp), split_cache_key=params["split_cache_key"], source=run_id[:8])


def _sorted_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[PAIR_COLUMNS].astype("int64").sort_values(PAIR_COLUMNS).reset_index(drop=True)


def verify_comparable(arms: list[ArmRun]) -> None:
    """Raise `RunMismatchError` unless every arm scored the same pairs on the same split.

    Checks, in order: labels are distinct; each run's predictions cover
    exactly its own `scored_pairs.csv`; every run records the same split
    fingerprint; every arm's `scored_pairs.csv` holds the
    same pairs; and every arm saw the same actual rating for each pair.
    """
    if len(arms) < 2:
        raise ValueError(f"need at least two runs to merge, got {len(arms)}")
    labels = [arm.label for arm in arms]
    if len(set(labels)) != len(labels):
        raise ValueError(f"arm labels collide: {labels}")

    for arm in arms:
        if not _sorted_pairs(arm.predictions).equals(_sorted_pairs(arm.scored_pairs)):
            raise RunMismatchError(f"{arm.label}: {PREDICTIONS_FILE} does not cover exactly its own {PAIRS_FILE}")

    keys = {arm.label: arm.split_cache_key for arm in arms}
    if len(set(keys.values())) > 1:
        raise RunMismatchError(f"runs were scored on different splits (split_cache_key): {keys}")

    reference = arms[0]
    reference_pairs = _sorted_pairs(reference.scored_pairs)
    for arm in arms[1:]:
        pairs = _sorted_pairs(arm.scored_pairs)
        if pairs.equals(reference_pairs):
            continue
        both = reference_pairs.merge(pairs, how="outer", indicator=True)
        only_reference = int((both["_merge"] == "left_only").sum())
        only_arm = int((both["_merge"] == "right_only").sum())
        raise RunMismatchError(
            f"{arm.label} and {reference.label} scored different pairs: "
            f"{only_reference} only in {reference.label}, {only_arm} only in {arm.label} "
            f"({len(reference_pairs)} vs {len(pairs)} pairs)"
        )

    reference_actual = reference.predictions.set_index(PAIR_COLUMNS)["actual_rating"].sort_index()
    for arm in arms[1:]:
        actual = arm.predictions.set_index(PAIR_COLUMNS)["actual_rating"].sort_index()
        differing = int((actual != reference_actual).sum())
        if differing:
            raise RunMismatchError(
                f"{arm.label} and {reference.label} record different actual ratings for {differing} pairs"
            )


def merge_runs(arms: list[ArmRun]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify the arms are comparable, then return (summary, per_pair).

    `summary` has one row per arm, metrics recomputed from its predictions by
    the harness's own `compute_metrics`. Each arm's own metrics skip its own
    nan pairs, so arms that returned nan on different pairs are measured on
    different pairs; the `common/` columns repeat the metrics on only the
    pairs every arm scored. `per_pair` has one row per pair and a
    `predicted_rating[<label>]` column per arm.
    """
    verify_comparable(arms)
    scored_by_all = None
    for arm in arms:
        scored = pd.MultiIndex.from_frame(arm.predictions.loc[arm.predictions["predicted_rating"].notna(), PAIR_COLUMNS])
        scored_by_all = scored if scored_by_all is None else scored_by_all.intersection(scored)
    rows = []
    for arm in arms:
        in_common = pd.MultiIndex.from_frame(arm.predictions[PAIR_COLUMNS]).isin(scored_by_all)
        common = compute_metrics(arm.predictions[in_common]) if in_common.any() else {}
        rows.append({
            "label": arm.label, "backend": arm.backend, "source": arm.source,
            **compute_metrics(arm.predictions),
            **{f"common/{key.split('/', 1)[1]}": value for key, value in common.items() if key.startswith("error/")},
            "common/pair_count": float(in_common.sum()),
        })
    summary = pd.DataFrame(rows)
    per_pair = arms[0].predictions[PAIR_COLUMNS + ["actual_rating"]]
    for arm in arms:
        column = arm.predictions[PAIR_COLUMNS + ["predicted_rating"]].rename(
            columns={"predicted_rating": f"predicted_rating[{arm.label}]"}
        )
        per_pair = per_pair.merge(column, on=PAIR_COLUMNS, how="left", validate="one_to_one")
    return summary, per_pair.sort_values(PAIR_COLUMNS).reset_index(drop=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge compare_backends runs into one comparison table.")
    parser.add_argument(
        "--run", nargs=2, action="append", default=[], metavar=("TRACKING_URI", "RUN_ID"),
        help="A run in a store this machine can read. Repeatable.",
    )
    parser.add_argument(
        "--artifacts", type=Path, action="append", default=[], metavar="DIR",
        help="A directory of one run's downloaded artifacts. Repeatable.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write summary.csv and per_pair.parquet here.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    arms = [load_from_mlflow(uri, run_id) for uri, run_id in args.run]
    arms += [load_from_artifacts(directory) for directory in args.artifacts]
    try:
        summary, per_pair = merge_runs(arms)
    except RunMismatchError as exc:
        sys.exit(f"error: refusing to merge: {exc}")
    with pd.option_context("display.max_columns", None, "display.width", None):
        print(summary.set_index("label"))
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out / "summary.csv", index=False)
        per_pair.to_parquet(args.out / "per_pair.parquet", index=False)
        print(f"wrote {args.out / 'summary.csv'} and {args.out / 'per_pair.parquet'}")


if __name__ == "__main__":
    main()
