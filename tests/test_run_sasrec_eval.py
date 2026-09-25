"""
tests/test_run_sasrec_eval.py - the one-command SASRec train-and-score runner.

Synthetic fixtures only; the sweep, checkpoints and MLflow all live in tmp.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from experiments import compare_backends
from experiments.run_sasrec_eval import main, run_dir_name, run_eval
from scripts.train_sasrec import CHECKPOINT_NAME, TrainingArgs

TINY_SASREC = dict(
    sasrec_hidden_units=16,
    sasrec_num_blocks=1,
    sasrec_maxlen=8,
    sasrec_rating_head_hidden=8,
)


@pytest.fixture(scope="module")
def tiny_sweep(tiny_config, tmp_path_factory):
    """Point the harness at the fixture data as a sweep called 'tiny'."""
    tmp = tmp_path_factory.mktemp("runner")
    # Its own embeddings_dir: these Environments rebuild their collections,
    # which would otherwise delete the session env's.
    base = dataclasses.replace(
        tiny_config, **TINY_SASREC, embeddings_dir=tmp / "chroma",
        mlflow_tracking_uri=f"sqlite:///{tmp / 'train.db'}",
    )
    mp = pytest.MonkeyPatch()
    mp.setattr(compare_backends, "BASE_CONFIG", base)
    mp.setattr(compare_backends, "SWEEPS", {"tiny": {"eval_user_frac": tiny_config.eval_user_frac}})
    yield tmp
    mp.undo()


def test_run_dir_name():
    assert run_dir_name("u128", None) == "sasrec_u128"
    assert run_dir_name("u128", 0.0) == "sasrec_u128_rlw0"


def test_skip_train_without_checkpoint_fails(tiny_sweep):
    with pytest.raises(FileNotFoundError):
        run_eval("tiny", runs_dir=tiny_sweep / "empty", training_args=TrainingArgs(epochs=1),
                 skip_train=True, baselines=(), mlflow_uri=f"sqlite:///{tiny_sweep / 'h.db'}")


def test_trains_scores_and_resumes(tiny_sweep):
    runs = tiny_sweep / "runs"
    uri = f"sqlite:///{tiny_sweep / 'harness.db'}"

    def run(epochs: int):
        return run_eval("tiny", runs_dir=runs, training_args=TrainingArgs(epochs=epochs, batch_size=16),
                        baselines=("bias_only", "associative"), mlflow_uri=uri, max_items=None)

    first = run(1)
    assert list(first.index) == ["bias_only", "associative", "sasrec"]
    assert first.loc["sasrec", "meta/nan_count"] == 0
    assert (runs / "sasrec_tiny" / "summary.csv").is_file()

    # Rerunning the same command resumes rather than refusing, and more
    # epochs continue from the saved one.
    second = run(2)
    payload = torch.load(runs / "sasrec_tiny" / CHECKPOINT_NAME, weights_only=False)
    assert payload["epoch"] == 2
    assert list(second.index) == list(first.index)
    assert second.loc["bias_only", "error/mae"] == first.loc["bias_only", "error/mae"]


def test_cli_runs_from_anywhere(tiny_sweep, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["--sweep", "tiny", "--epochs", "1", "--baselines", "bias_only",
          "--runs-dir", str(tiny_sweep / "cli"), "--mlflow-uri", f"sqlite:///{tiny_sweep / 'cli.db'}"])
    out = capsys.readouterr().out
    assert "sasrec" in out and "bias_only" in out
