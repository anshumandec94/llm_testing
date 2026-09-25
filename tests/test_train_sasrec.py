"""
Tests for scripts/train_sasrec.py (issue #18).

The target-leakage regression test comes first because it is the one that
matters most: rating context may enter the model at input positions only, and
a script that feeds the target's residual in would produce an excellent
training curve on a worthless model.

Smoke scale only, on the synthetic fixtures in conftest.py. MLflow writes to a
temp directory. Nothing here touches data/ml-32m/.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from scripts.train_sasrec import (
    CHECKPOINT_NAME,
    EXPERIMENT_NAME,
    TrainingArgs,
    build_optimizer,
    build_training_data,
    load_checkpoint,
    resolve_config,
    sample_negatives,
    select_device,
    to_tensors,
    train,
    train_step,
)
from sim.agents.sasrec_data import PAD_INDEX, SasrecTrainingBatch
from sim.agents.sasrec_model import SASRec

CPU = torch.device("cpu")


@pytest.fixture
def config(tiny_config, tmp_path):
    base = dataclasses.replace(
        tiny_config,
        sasrec_hidden_units=16,
        sasrec_num_blocks=1,
        sasrec_maxlen=16,
        sasrec_rating_head_hidden=8,
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )
    return resolve_config(base)


@pytest.fixture
def seq_data(env, config):
    return build_training_data(env, config)


@pytest.fixture
def fixed_batch(seq_data, config):
    index = seq_data.training_window_index(config.sasrec_maxlen, config.sasrec_window_stride)
    batch = seq_data.training_batch(index[:32])
    user_items = {u: np.unique(s) for u, s in seq_data.user_sequences.items()}
    negatives = sample_negatives(batch, user_items, seq_data.vocab_size, np.random.default_rng(0))
    return batch, negatives


def _shuffle_target_residuals(batch: SasrecTrainingBatch, seed: int = 1) -> SasrecTrainingBatch:
    """Permute the target residuals among the non-padded target positions."""
    valid = batch.target_items != PAD_INDEX
    shuffled = batch.target_residuals.copy()
    shuffled[valid] = np.random.default_rng(seed).permutation(shuffled[valid])
    assert not np.array_equal(shuffled, batch.target_residuals)
    return dataclasses.replace(batch, target_residuals=shuffled)


def _one_step(config, seq_data, batch, negatives):
    torch.manual_seed(0)
    model = SASRec.from_config(config, item_num=seq_data.vocab_size - 1)
    optimizer = build_optimizer(model, TrainingArgs())
    torch.manual_seed(1)
    return train_step(model, optimizer, to_tensors(batch, negatives, CPU), seq_data.residual_std)


class TestTargetLeakage:
    """Only input residuals may reach the model; target residuals are labels."""

    def test_ranking_loss_ignores_target_residuals(self, config, seq_data, fixed_batch):
        assert config.sasrec_inject_rating, "leakage is only possible with injection on"
        batch, negatives = fixed_batch
        original = _one_step(config, seq_data, batch, negatives)
        shuffled = _one_step(config, seq_data, _shuffle_target_residuals(batch), negatives)
        assert torch.equal(original.bce, shuffled.bce)

    def test_total_loss_unchanged_without_the_rating_label(self, config, seq_data, fixed_batch):
        """With the MSE term off, target residuals must not affect the loss at all."""
        config = dataclasses.replace(config, sasrec_rating_loss_weight=0.0)
        batch, negatives = fixed_batch
        original = _one_step(config, seq_data, batch, negatives)
        shuffled = _one_step(config, seq_data, _shuffle_target_residuals(batch), negatives)
        assert torch.equal(original.total, shuffled.total)

    def test_detects_a_leaky_batch(self, config, seq_data, fixed_batch):
        """The test has teeth: feeding target residuals in as inputs changes the loss."""
        batch, negatives = fixed_batch
        leaky = dataclasses.replace(batch, input_residuals=batch.target_residuals)
        shuffled = _shuffle_target_residuals(batch)
        leaky_shuffled = dataclasses.replace(shuffled, input_residuals=shuffled.target_residuals)
        a = _one_step(config, seq_data, leaky, negatives)
        b = _one_step(config, seq_data, leaky_shuffled, negatives)
        assert not torch.equal(a.bce, b.bce)


class TestNegativeSampling:
    def test_negatives_are_unrated_real_items_at_valid_positions(self, seq_data, fixed_batch):
        batch, negatives = fixed_batch
        valid = batch.target_items != PAD_INDEX
        assert (negatives[~valid] == PAD_INDEX).all()
        assert (negatives[valid] >= 1).all()
        assert (negatives[valid] < seq_data.vocab_size).all()
        for row, uid in enumerate(batch.user_ids.tolist()):
            assert not np.isin(negatives[row][valid[row]], seq_data.user_sequences[uid]).any()


class TestDevice:
    def test_falls_back_to_mps_without_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert select_device() == torch.device("mps")

    def test_falls_back_to_cpu_without_cuda_or_mps(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert select_device() == torch.device("cpu")

    def test_prefers_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert select_device() == torch.device("cuda")


class TestConfig:
    def test_stride_is_resolved_from_config(self, tiny_config):
        cfg = dataclasses.replace(tiny_config, sasrec_maxlen=16, sasrec_window_stride=None)
        assert resolve_config(cfg).sasrec_window_stride == 16
        cfg = dataclasses.replace(cfg, sasrec_window_stride=4)
        assert resolve_config(cfg).sasrec_window_stride == 4

    def test_rating_loss_weight_override(self, tiny_config):
        assert resolve_config(tiny_config, 0.0).sasrec_rating_loss_weight == 0.0
        assert resolve_config(tiny_config).sasrec_rating_loss_weight == (
            tiny_config.sasrec_rating_loss_weight
        )

    def test_training_uses_the_config_stride(self, env, config):
        strided = dataclasses.replace(config, sasrec_window_stride=4)
        data = build_training_data(env, strided)
        assert data.window_stride == 4
        assert len(data.training_window_index()) > len(
            build_training_data(env, config).training_window_index()
        )


class TestTraining:
    ARGS = TrainingArgs(epochs=6, batch_size=16, lr=5e-3)

    def test_loss_decreases_without_nans(self, seq_data, config, tmp_path):
        _, history = train(seq_data, config, self.ARGS, tmp_path / "ckpt", device=CPU)
        totals = [h["total"] for h in history]
        assert all(np.isfinite(v) for h in history for v in h.values())
        assert totals[-1] < totals[0]

    def test_rating_loss_weight_zero_trains(self, seq_data, config, tmp_path):
        config = dataclasses.replace(config, sasrec_rating_loss_weight=0.0)
        _, history = train(seq_data, config, self.ARGS, tmp_path / "ckpt", device=CPU)
        assert all(h["total"] == pytest.approx(h["bce"]) for h in history)
        assert history[-1]["bce"] < history[0]["bce"]

    def test_checkpoint_reloads_to_identical_weights(self, seq_data, config, tmp_path):
        args = dataclasses.replace(self.ARGS, epochs=2)
        model, _ = train(seq_data, config, args, tmp_path / "ckpt", device=CPU)
        reloaded, payload = load_checkpoint(tmp_path / "ckpt" / CHECKPOINT_NAME)
        original = model.state_dict()
        for key, value in reloaded.state_dict().items():
            assert torch.equal(value, original[key]), key
        assert payload["epoch"] == 2
        assert payload["index_to_item"] == seq_data.index_to_item
        assert payload["residual_std"] == seq_data.residual_std
        assert payload["config"]["sasrec_window_stride"] == config.sasrec_maxlen

    def test_resume_matches_an_uninterrupted_run(self, seq_data, config, tmp_path):
        args = dataclasses.replace(self.ARGS, epochs=3)
        full, _ = train(seq_data, config, args, tmp_path / "full", device=CPU)

        train(seq_data, config, dataclasses.replace(args, epochs=1), tmp_path / "cut", device=CPU)
        resumed, history = train(seq_data, config, args, tmp_path / "cut", device=CPU, resume=True)

        assert len(history) == 2
        for key, value in resumed.state_dict().items():
            assert torch.equal(value, full.state_dict()[key]), key

    def test_refuses_to_overwrite_or_resume_a_different_config(self, seq_data, config, tmp_path):
        args = dataclasses.replace(self.ARGS, epochs=1)
        train(seq_data, config, args, tmp_path / "ckpt", device=CPU)
        with pytest.raises(FileExistsError):
            train(seq_data, config, args, tmp_path / "ckpt", device=CPU)
        other = dataclasses.replace(config, sasrec_rating_loss_weight=0.5)
        with pytest.raises(ValueError, match="sasrec_rating_loss_weight"):
            train(seq_data, other, args, tmp_path / "ckpt", device=CPU, resume=True)

    def test_logs_to_its_own_experiment(self, seq_data, config, tmp_path):
        from mlflow.tracking import MlflowClient

        args = dataclasses.replace(self.ARGS, epochs=1)
        train(seq_data, config, args, tmp_path / "ckpt", device=CPU)
        client = MlflowClient(tracking_uri=config.mlflow_tracking_uri)
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        assert experiment is not None
        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) == 1
        run = runs[0]
        assert int(run.data.params["data_n_windows"]) == len(seq_data.training_window_index())
        assert "train/loss_total" in run.data.metrics
