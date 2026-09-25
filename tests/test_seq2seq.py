"""
tests/test_seq2seq.py - SASRec wired into the agents and the harness (issue #19).

The round trip comes last and is the deliverable: a checkpoint trained by
``scripts/train_sasrec.py`` on the synthetic fixture is scored through
``experiments/compare_backends.py --backend sasrec`` and yields an MLflow run
whose ``scored_pairs.csv`` is byte-identical to the bias-only arm's.

Synthetic fixtures only; MLflow writes to tmp dirs. Nothing touches data/ml-32m/.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from lenskit.data import ItemList

import experiments.compare_backends as compare_backends
from experiments.backends import SASRecBackend
from main import parse_args
from scripts.train_sasrec import CHECKPOINT_NAME, TrainingArgs, build_training_data, resolve_config, train
from sim.agents import build_agent
from sim.agents.sasrec_data import SasrecSequenceData
from sim.agents.seq2seq import Seq2SeqAgent, check_checkpoint_matches
from sim.environment import Environment
from sim.persona import AgentPersona

TINY_SASREC = dict(
    sasrec_hidden_units=16,
    sasrec_num_blocks=1,
    sasrec_maxlen=8,
    sasrec_rating_head_hidden=8,
)


@pytest.fixture(scope="module")
def trained(tiny_config, env, tmp_path_factory):
    """A two-epoch checkpoint on the session fixture's split."""
    tmp = tmp_path_factory.mktemp("sasrec")
    config = resolve_config(dataclasses.replace(
        tiny_config, **TINY_SASREC, mlflow_tracking_uri=f"sqlite:///{tmp / 'mlflow.db'}",
    ))
    train(build_training_data(env, config), config, TrainingArgs(epochs=2, batch_size=16),
          tmp, device=torch.device("cpu"))
    return tmp / CHECKPOINT_NAME, config


@pytest.fixture(scope="module")
def agent_env(tiny_config, env):
    """The session env, under a config whose sasrec_maxlen matches the checkpoint's."""
    env.config = dataclasses.replace(tiny_config, **TINY_SASREC)
    yield env
    env.config = tiny_config


@pytest.fixture
def agent(trained, agent_env):
    return Seq2SeqAgent(agent_env, trained[0])


def _long_user(agent: Seq2SeqAgent) -> int:
    maxlen = agent.trained_config.sasrec_maxlen
    return next(u for u, s in agent.seq_data.user_sequences.items() if len(s) > maxlen + 2)


class TestScoringPath:

    def test_uses_padded_sequence_never_training_windows(self, agent, monkeypatch):
        def forbidden(*args, **kwargs):
            raise AssertionError("scoring must not build training windows")

        calls = []
        real = SasrecSequenceData.padded_sequence

        def spy(self, *args, **kwargs):
            calls.append(args)
            return real(self, *args, **kwargs)

        monkeypatch.setattr(SasrecSequenceData, "training_batch", forbidden)
        monkeypatch.setattr(SasrecSequenceData, "training_window_index", forbidden)
        monkeypatch.setattr(SasrecSequenceData, "padded_sequence", spy)
        backend = SASRecBackend(agent)
        uid = _long_user(agent)
        out = backend.predict(uid, [1, 2, 3])
        assert np.isfinite(out).all()
        assert calls

    def test_residual_is_the_unscaled_rating_head_on_the_last_maxlen(self, agent):
        uid = _long_user(agent)
        maxlen = agent.trained_config.sasrec_maxlen
        seq = agent.seq_data.user_sequences[uid]
        res = agent.seq_data.user_residuals[uid]
        movie_ids = [5, 6, 7]
        idx = torch.tensor([[agent.seq_data.item_to_index[m] for m in movie_ids]])
        _, expected = agent.model.score_items(
            torch.from_numpy(seq[-maxlen:].astype(np.int64)).unsqueeze(0), idx,
            input_residuals=torch.from_numpy(res[-maxlen:]).unsqueeze(0),
            residual_std=agent.residual_std,
        )
        np.testing.assert_allclose(agent.predict_residuals(uid, movie_ids), expected[0].numpy(), rtol=1e-6)

    def test_history_older_than_maxlen_is_ignored(self, agent):
        uid = _long_user(agent)
        before = agent.predict_residuals(uid, [1, 2, 3])
        agent.seq_data.user_sequences[uid][0] = agent.seq_data.user_sequences[uid][-1]
        np.testing.assert_array_equal(agent.predict_residuals(uid, [1, 2, 3]), before)

    def test_cold_user_and_unknown_item_are_nan(self, agent):
        assert np.isnan(agent.predict_residuals(10_000_000, [1, 2])).all()
        uid = _long_user(agent)
        out = agent.predict_residuals(uid, [1, 10_000_000, 2])
        assert np.isfinite(out[[0, 2]]).all() and np.isnan(out[1])

    def test_update_appends_without_retraining(self, agent):
        uid = _long_user(agent)
        weights = {k: v.clone() for k, v in agent.model.state_dict().items()}
        agent.update(uid, [(9, "rate", 1.25)])
        items, residuals = agent.seq_data.padded_sequence(uid)
        assert items[-1] == agent.seq_data.item_to_index[9]
        assert residuals[-1] == pytest.approx(1.25)
        for k, v in agent.model.state_dict().items():
            assert torch.equal(v, weights[k])

    def test_evaluate_scores_are_residuals_with_nan_as_zero(self, agent):
        uid = _long_user(agent)
        scored = agent.evaluate(
            ItemList(item_ids=np.array([1, 10_000_000])),
            cast(AgentPersona, SimpleNamespace(user_id=uid)), {},
        )
        scores = scored.scores()
        assert scores is not None
        assert scores[0] == pytest.approx(agent.predict_residuals(uid, [1])[0], rel=1e-6)
        assert scores[1] == 0.0


class TestRefusals:

    def test_split_mismatch_is_refused(self, trained):
        _, config = trained
        with pytest.raises(ValueError, match="split_cache_key"):
            check_checkpoint_matches(config, dataclasses.replace(config, eval_user_frac=0.5))

    def test_maxlen_mismatch_is_refused(self, trained):
        _, config = trained
        with pytest.raises(ValueError, match="sasrec_maxlen"):
            check_checkpoint_matches(config, dataclasses.replace(config, sasrec_maxlen=16))

    def test_agent_refuses_before_building_sequences(self, trained, tiny_config):
        wrong = SimpleNamespace(config=dataclasses.replace(tiny_config, **TINY_SASREC, random_seed=1))
        with pytest.raises(ValueError, match="not trained for this configuration"):
            Seq2SeqAgent(cast(Environment, wrong), trained[0])


class TestRegistration:

    def test_registry_needs_a_checkpoint(self, agent_env):
        with pytest.raises(ValueError, match="sasrec_checkpoint_path"):
            build_agent(dataclasses.replace(agent_env.config, agent_type="seq2seq"), agent_env)

    def test_registry_builds_the_agent(self, agent_env, trained):
        cfg = dataclasses.replace(
            agent_env.config, agent_type="seq2seq", sasrec_checkpoint_path=str(trained[0])
        )
        assert isinstance(build_agent(cfg, agent_env), Seq2SeqAgent)

    def test_cli_accepts_seq2seq(self):
        cfg = parse_args(["--agent_type", "seq2seq", "--sasrec_checkpoint_path", "ckpt.pt"])
        assert cfg.agent_type == "seq2seq"
        assert cfg.sasrec_checkpoint_path == "ckpt.pt"


class TestRoundTrip:
    """Train with the script's CLI, score with the harness's CLI, compare to the null."""

    @pytest.fixture(scope="class")
    def runs(self, tiny_config, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("round_trip")
        harness_uri = f"sqlite:///{tmp / 'harness.db'}"
        # Its own embeddings_dir: these Environments rebuild their collections,
        # which would otherwise delete the session env's.
        config = dataclasses.replace(
            tiny_config, **TINY_SASREC, embeddings_dir=tmp / "chroma",
            mlflow_tracking_uri=f"sqlite:///{tmp / 'train.db'}",
        )
        config_path = tmp / "config.json"
        config_path.write_text(json.dumps(config.to_json_dict()))

        from scripts.train_sasrec import main as train_main

        train_main([
            "--config", str(config_path), "--checkpoint-dir", str(tmp / "ckpt"),
            "--epochs", "2", "--batch-size", "16",
        ])
        checkpoint = tmp / "ckpt" / CHECKPOINT_NAME

        mp = pytest.MonkeyPatch()
        mp.setattr(compare_backends, "BASE_CONFIG", dataclasses.replace(config, mlflow_tracking_uri=harness_uri))
        mp.setattr(compare_backends, "SWEEPS", {"tiny": {"eval_user_frac": tiny_config.eval_user_frac}})
        try:
            common = ["--sweep", "tiny", "--max-items", "0", "--mlflow-uri", harness_uri]
            compare_backends.main(["--backend", "bias_only", *common])
            compare_backends.main(["--backend", "sasrec", "--checkpoint", str(checkpoint), *common])
        finally:
            mp.undo()

        mlflow.set_tracking_uri(harness_uri)
        out = {}
        for run in mlflow.search_runs(
            experiment_names=[compare_backends.EXPERIMENT_NAME], output_format="list"
        ):
            name = run.data.params["backend"]
            dst = tmp / name
            mlflow.artifacts.download_artifacts(run_id=run.info.run_id, dst_path=str(dst))
            out[name] = (dst, run.data)
        return out, checkpoint, config

    def test_scored_pairs_byte_identical_to_bias_only(self, runs):
        out, _, _ = runs
        a = (out["bias_only"][0] / "scored_pairs.csv").read_bytes()
        b = (out["sasrec"][0] / "scored_pairs.csv").read_bytes()
        assert len(a) > len("userId,movieId\n")
        assert a == b

    def test_sasrec_scores_every_pair(self, runs):
        out, _, _ = runs
        dst, data = out["sasrec"]
        frame = pd.read_parquet(Path(dst) / "per_item_predictions.parquet")
        assert len(frame) > 0
        assert frame["predicted_residual"].notna().all()
        assert data.metrics["meta/nan_count"] == 0
        assert np.isfinite(data.metrics["error/mae"])

    def test_run_describes_its_checkpoint(self, runs):
        out, checkpoint, config = runs
        dst, data = out["sasrec"]
        assert data.params["sasrec_checkpoint"] == str(checkpoint.resolve())
        assert data.params["ckpt_sasrec_maxlen"] == str(config.sasrec_maxlen)
        saved = json.loads((Path(dst) / "sasrec_checkpoint_config.json").read_text())
        assert saved["sasrec_maxlen"] == config.sasrec_maxlen
        assert saved["eval_user_frac"] == config.eval_user_frac
