"""
sim.agents.seq2seq - Agent 3: SASRec, scored through its rating head.

Loads a checkpoint written by ``scripts/train_sasrec.py`` (issue #18) and
predicts, for one user and a list of items, the **debiased-rating residual**
of each item: the rating head's output, which was trained by MSE against raw
``env.debias_rating`` residuals, so it is already in rating units and needs
no rescaling. ``residual_std`` from the checkpoint only normalises the rating
context injected at input positions, exactly as in training.

**Context is the inference path, never training windows.** A user's context
is ``SasrecSequenceData.padded_sequence``: their most recent ``maxlen``
training interactions (plus anything ``update`` appended this session),
left-padded. ``training_window_index`` / ``training_batch`` are training-only
and are never called here; ``tests/test_seq2seq.py`` pins that.

**A checkpoint is only scored on the split it was trained on.** The
environment's ``split_cache_key`` and ``sasrec_maxlen`` must match the
checkpoint's saved config, and the vocabulary and residual scale rebuilt from
the environment must match the ones saved, or construction raises. A model
scored on a different split would have seen some held-out ratings in training.

nan marks what the model genuinely cannot score:

* an item outside the vocabulary. The vocabulary spans every rated movie
  (#16), so this should not arise; it is logged loudly if it does.
* a user with no training interactions. SASRec has no user embedding, so an
  empty context is an all-padding sequence the model never saw in training,
  and its output would be an artefact rather than a prediction.

``evaluate`` (the simulation path) returns the residual as the score and maps
nan to 0.0, the neutral residual, as ``ItemItemNeighborhoodAgent`` does for a
user with no history.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from lenskit.data import ItemList

from sim.agents.base import AbstractAgent
from sim.agents.sasrec_data import SasrecSequenceData, build_sasrec_sequences
from sim.agents.sasrec_model import SASRec
from sim.config import SimConfig
from sim.environment import Environment
from sim.persona import AgentPersona

logger = logging.getLogger(__name__)


def load_sasrec_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[SASRec, dict]:
    """Rebuild the model from a checkpoint. Returns ``(model, payload)``, model in eval mode."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = SimConfig.from_dict(payload["config"])
    model = SASRec.from_config(config, item_num=len(payload["index_to_item"]) - 1)
    model.load_state_dict(payload["model_state"])
    model = model.to(device)
    model.eval()
    return model, payload


def check_checkpoint_matches(trained: SimConfig, current: SimConfig) -> None:
    """Refuse a checkpoint whose split or context length differs from ``current``."""
    problems = []
    if trained.split_cache_key() != current.split_cache_key():
        problems.append(
            f"split_cache_key {trained.split_cache_key()} != {current.split_cache_key()} "
            f"(trained with data_dir={trained.data_dir}, "
            f"eval_user_frac={trained.eval_user_frac}, "
            f"validation_frac={trained.validation_frac}, "
            f"holdout_frac={trained.holdout_frac}, "
            f"min_ratings={trained.min_ratings}, random_seed={trained.random_seed})"
        )
    if trained.sasrec_maxlen != current.sasrec_maxlen:
        problems.append(
            f"sasrec_maxlen {trained.sasrec_maxlen} != {current.sasrec_maxlen}"
        )
    if problems:
        raise ValueError(
            "SASRec checkpoint was not trained for this configuration; refusing "
            "to score it: " + "; ".join(problems)
        )


class Seq2SeqAgent(AbstractAgent):
    """SASRec's rating head over each user's most recent ``maxlen`` interactions."""

    def __init__(
        self,
        env: Environment,
        checkpoint_path: str | Path,
        device: torch.device | str = "cpu",
    ) -> None:
        self.env = env
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.model, payload = load_sasrec_checkpoint(self.checkpoint_path, self.device)
        self.trained_config = SimConfig.from_dict(payload["config"])
        self.epoch = int(payload["epoch"])
        self.residual_std = float(payload["residual_std"])
        check_checkpoint_matches(self.trained_config, env.config)

        maxlen = self.trained_config.sasrec_maxlen
        # This agent's own copy: `update` appends to it, so it must not be shared.
        self.seq_data: SasrecSequenceData = build_sasrec_sequences(env, maxlen=maxlen)
        if list(self.seq_data.index_to_item) != list(payload["index_to_item"]):
            raise ValueError(
                "SASRec checkpoint item vocabulary differs from the one built from "
                "this environment's ratings; refusing to score it"
            )
        if not np.isclose(self.seq_data.residual_std, self.residual_std, rtol=1e-5):
            raise ValueError(
                f"training residual_std {self.residual_std} differs from this "
                f"environment's {self.seq_data.residual_std}; the train split or "
                f"bias model is not the one the checkpoint was trained on"
            )

    def predict_residuals(self, user_id: int, movie_ids: list[int]) -> np.ndarray:
        """Debiased-rating residual per item, aligned with ``movie_ids``; nan where unscorable."""
        out = np.full(len(movie_ids), np.nan, dtype=np.float64)
        seq = self.seq_data.user_sequences.get(int(user_id))
        if seq is None or len(seq) == 0 or len(movie_ids) == 0:
            return out

        positions: list[int] = []
        indices: list[int] = []
        for pos, mid in enumerate(movie_ids):
            idx = self.seq_data.item_to_index.get(int(mid))
            if idx is None:
                logger.error(
                    "movieId %s is outside the SASRec vocabulary, which should "
                    "span every rated movie; scoring it as nan", mid,
                )
                continue
            positions.append(pos)
            indices.append(idx)
        if not indices:
            return out

        items, residuals = self.seq_data.padded_sequence(int(user_id))
        _, predicted = self.model.score_items(
            torch.from_numpy(items.astype(np.int64)).unsqueeze(0).to(self.device),
            torch.tensor([indices], dtype=torch.int64, device=self.device),
            input_residuals=torch.from_numpy(residuals).unsqueeze(0).to(self.device),
            residual_std=self.residual_std,
        )
        out[positions] = predicted[0].double().cpu().numpy()
        return out

    def evaluate(
        self,
        candidates: ItemList,
        persona: AgentPersona,
        item_factors: dict[int, np.ndarray],
    ) -> ItemList:
        movie_ids = [int(mid) for mid in candidates.ids()]
        scores = np.nan_to_num(
            self.predict_residuals(persona.user_id, movie_ids), nan=0.0
        )
        return ItemList(candidates, scores=scores.astype(np.float32))

    def update(
        self,
        user_id: int,
        interactions: list[tuple[int, str, float]],
    ) -> None:
        """Append this round's (movieId, action, debiased residual) to the user's context. No retraining."""
        new_items: list[int] = []
        new_residuals: list[float] = []
        for movie_id, _action, signal in interactions:
            idx = self.seq_data.item_to_index.get(int(movie_id))
            if idx is None:
                logger.error("movieId %s is outside the SASRec vocabulary; not appended", movie_id)
                continue
            new_items.append(idx)
            new_residuals.append(float(signal))
        if not new_items:
            return
        uid = int(user_id)
        seqs, res = self.seq_data.user_sequences, self.seq_data.user_residuals
        seqs[uid] = np.concatenate(
            [seqs.get(uid, np.zeros(0, dtype=np.int32)), np.asarray(new_items, dtype=np.int32)]
        )
        res[uid] = np.concatenate(
            [res.get(uid, np.zeros(0, dtype=np.float32)), np.asarray(new_residuals, dtype=np.float32)]
        )
