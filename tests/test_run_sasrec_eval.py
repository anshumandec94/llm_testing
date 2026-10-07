"""
tests/test_run_sasrec_eval.py - the one-command SASRec train-and-score runner.

Synthetic fixtures only; the sweep, checkpoints and MLflow all live in tmp.
"""
from __future__ import annotations

import dataclasses

import pandas as pd
import pytest
import torch

from experiments import compare_backends
import experiments.run_sasrec_eval as runner
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
                        baselines=("bias_only", "associative"), mlflow_uri=uri, max_items=None, patience=0)

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
    assert (second["sasrec_epoch"] == 2).all()

    # --skip-train refuses a checkpoint trained under different settings.
    changed = dataclasses.replace(compare_backends.BASE_CONFIG, sasrec_dropout_rate=0.37)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(compare_backends, "BASE_CONFIG", changed)
        with pytest.raises(ValueError, match="different settings"):
            run_eval("tiny", runs_dir=runs, training_args=TrainingArgs(epochs=2, batch_size=16),
                     skip_train=True, baselines=(), mlflow_uri=uri, max_items=None, patience=0)


def test_cli_runs_from_anywhere(tiny_sweep, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["--sweep", "tiny", "--epochs", "1", "--patience", "0", "--baselines", "bias_only",
          "--runs-dir", str(tiny_sweep / "cli"), "--mlflow-uri", f"sqlite:///{tiny_sweep / 'cli.db'}"])
    out = capsys.readouterr().out
    assert "sasrec" in out and "bias_only" in out


class TestValidationConvergence:
    """Issue #42: the epoch is chosen on validation, and held-out scores the best one."""

    @staticmethod
    def _scripted_validation(monkeypatch, maes: list[float]) -> list[int]:
        """Replace validation scoring with a fixed MAE per epoch; returns the epochs it was asked for."""
        asked: list[int] = []

        def fake(env, assignments, checkpoint, config, max_items):
            epoch = int(torch.load(checkpoint, weights_only=False)["epoch"])
            asked.append(epoch)
            return maes[epoch - 1]

        monkeypatch.setattr(runner, "validation_mae", fake)
        return asked

    def _run(self, runs, uri, epochs=10, patience=2):
        return run_eval("tiny", runs_dir=runs, training_args=TrainingArgs(epochs=epochs, batch_size=16),
                        baselines=("bias_only",), mlflow_uri=uri, max_items=None, patience=patience)

    def test_stops_patience_epochs_after_the_best_and_scores_the_best(self, tiny_sweep, monkeypatch):
        runs, uri = tiny_sweep / "converge", f"sqlite:///{tiny_sweep / 'converge.db'}"
        asked = self._scripted_validation(monkeypatch, [0.9, 0.7, 0.8, 0.75, 0.6, 0.6])
        table = self._run(runs, uri)

        assert asked == [1, 2, 3, 4]
        run_dir = runs / "sasrec_tiny"
        assert torch.load(run_dir / CHECKPOINT_NAME, weights_only=False)["epoch"] == 4
        assert torch.load(run_dir / "best.pt", weights_only=False)["epoch"] == 2
        assert (table["sasrec_epoch"] == 2).all()
        assert pd.read_csv(run_dir / "validation.csv")["epoch"].tolist() == [1, 2, 3, 4]

    def test_resumes_with_its_validation_history(self, tiny_sweep, monkeypatch):
        runs, uri = tiny_sweep / "resume", f"sqlite:///{tiny_sweep / 'resume.db'}"
        maes = [0.9, 0.8, 0.7, 0.75, 0.76]
        asked = self._scripted_validation(monkeypatch, maes)
        self._run(runs, uri, epochs=2)  # killed at the ceiling, still improving
        assert asked == [1, 2]

        table = self._run(runs, uri, epochs=10)  # same command, higher ceiling
        assert asked == [1, 2, 3, 4, 5]
        assert (table["sasrec_epoch"] == 3).all()

        # A converged run reruns without training or validating again.
        self._run(runs, uri, epochs=10)
        assert asked == [1, 2, 3, 4, 5]

    def test_an_unvalidated_last_epoch_is_validated_on_resume(self, tiny_sweep, monkeypatch):
        runs, uri = tiny_sweep / "gap", f"sqlite:///{tiny_sweep / 'gap.db'}"
        asked = self._scripted_validation(monkeypatch, [0.9, 0.8, 0.85, 0.86])
        self._run(runs, uri, epochs=2)
        log = runs / "sasrec_tiny" / "validation.csv"
        pd.read_csv(log).iloc[:1].to_csv(log, index=False)  # killed between training and validating epoch 2

        self._run(runs, uri, epochs=10)
        assert asked == [1, 2, 2, 3, 4]
        assert pd.read_csv(log)["epoch"].tolist() == [1, 2, 3, 4]

    def test_real_validation_scoring_runs_on_the_validation_split(self, tiny_sweep):
        runs, uri = tiny_sweep / "real", f"sqlite:///{tiny_sweep / 'real.db'}"
        self._run(runs, uri, epochs=2, patience=5)
        history = pd.read_csv(runs / "sasrec_tiny" / "validation.csv")
        assert history["epoch"].tolist() == [1, 2]
        assert history["val_mae_clipped"].between(0, 4).all()
