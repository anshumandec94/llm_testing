"""
Train the SASRec preference backend (issue #18).

A standalone script, not wired into ``SimulationRunner``. Everything about the
model and the data comes from one ``SimConfig`` (``--config``, a JSON file);
the flags below cover only the optimisation loop, which ``SimConfig`` does not
describe.

Training iterates **windows, not users** (issue #24): each epoch is one
shuffled pass over ``training_window_index(config.sasrec_maxlen,
config.sasrec_window_stride)``. Negatives are sampled here, one per non-padded
target position, uniformly from the items the user has not rated in training,
as pmixer's sampler does.

The objective is ``L = L_bce + rating_loss_weight * L_mse`` (see
``sim/agents/sasrec_model.py``). ``--rating-loss-weight 0`` is the ablation
that measures what the rating head costs the ranking.

**Rating context goes in at input positions only.** ``SasrecTrainingBatch``
carries ``input_residuals`` and ``target_residuals``; only the first may reach
the model. The target's residual is the label, and feeding it in leaks the
answer. ``tests/test_train_sasrec.py`` pins this.

Training hyperparameters follow pmixer/SASRec.pytorch: Adam with lr 1e-3 and
betas (0.9, 0.98), batch size 128, ``l2_emb`` 0.0 applied to the item table
only (divergence 7), 1000 epochs.

Checkpoints are written atomically every ``--checkpoint-every`` epochs (default
1, so a dropped ssh session costs at most one epoch). Each epoch's shuffle,
negatives and dropout are seeded from ``(random_seed, epoch)``, so a resumed run
reproduces an uninterrupted one exactly on the same device.

Usage:
    uv run python scripts/train_sasrec.py --config cfg.json --checkpoint-dir runs/sasrec
    # after an interruption, rerun with --resume
    uv run python scripts/train_sasrec.py --config cfg.json --checkpoint-dir runs/sasrec --resume
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.agents.sasrec_data import (  # noqa: E402
    PAD_INDEX,
    SasrecSequenceData,
    SasrecTrainingBatch,
    build_sasrec_sequences,
    resolve_window_stride,
)
from sim.agents.sasrec_model import SASRec, SasrecLosses  # noqa: E402
from sim.agents.seq2seq import load_sasrec_checkpoint  # noqa: E402
from sim.config import SimConfig  # noqa: E402
from sim.environment import Environment  # noqa: E402

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "sasrec-training"
CHECKPOINT_NAME = "checkpoint.pt"


@dataclass
class TrainingArgs:
    """Optimisation settings. Defaults are pmixer/SASRec.pytorch's."""

    epochs: int = 1000
    batch_size: int = 128
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.98
    l2_emb: float = 0.0
    checkpoint_every: int = 1


# Training args that may change between an interrupted run and its resume.
RESUMABLE_CHANGES = ("epochs", "checkpoint_every")


# ──────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────


def select_device() -> torch.device:
    """cuda if available, else mps, else cpu, so one script runs on the Mac and on CARC."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_config(config: SimConfig, rating_loss_weight: float | None = None) -> SimConfig:
    """
    The config a run actually trains with: the stride resolved to an int, so
    the checkpoint records what the model was trained on, and the rating loss
    weight overridden when the flag is given.
    """
    changes: dict[str, object] = {
        "sasrec_window_stride": resolve_window_stride(
            config.sasrec_window_stride, config.sasrec_maxlen
        )
    }
    if rating_loss_weight is not None:
        changes["sasrec_rating_loss_weight"] = float(rating_loss_weight)
    return dataclasses.replace(config, **changes)


def build_training_data(env, config: SimConfig) -> SasrecSequenceData:
    """Sequences built from the config's ``maxlen`` and stride, never hardcoded ones."""
    return build_sasrec_sequences(
        env, maxlen=config.sasrec_maxlen, window_stride=config.sasrec_window_stride
    )


# ──────────────────────────────────────────────────────────────────────────
# Batches
# ──────────────────────────────────────────────────────────────────────────


def sample_negatives(
    batch: SasrecTrainingBatch,
    user_items: dict[int, np.ndarray],
    vocab_size: int,
    rng: np.random.Generator,
    max_tries: int = 100,
) -> np.ndarray:
    """
    One negative per non-padded target position, drawn uniformly from
    ``[1, vocab_size)`` and rejected if the user rated it in training
    (pmixer's ``random_neq`` against the user's training set). Padded targets
    get ``PAD_INDEX``.
    """
    negatives = np.zeros_like(batch.target_items)
    for row, uid in enumerate(batch.user_ids.tolist()):
        valid = batch.target_items[row] != PAD_INDEX
        n = int(valid.sum())
        seen = user_items[uid]
        draws = rng.integers(1, vocab_size, size=n)
        for _ in range(max_tries):
            clash = np.isin(draws, seen, assume_unique=False)
            if not clash.any():
                break
            draws[clash] = rng.integers(1, vocab_size, size=int(clash.sum()))
        else:
            raise ValueError(
                f"could not sample a negative for user {uid}, who has rated "
                f"{len(seen)} of {vocab_size - 1} items"
            )
        negatives[row, valid] = draws
    return negatives


@dataclass
class TensorBatch:
    input_items: torch.Tensor
    input_residuals: torch.Tensor
    target_items: torch.Tensor
    target_residuals: torch.Tensor
    negative_items: torch.Tensor


def to_tensors(
    batch: SasrecTrainingBatch, negatives: np.ndarray, device: torch.device
) -> TensorBatch:
    def items(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a.astype(np.int64)).to(device)

    def floats(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a.astype(np.float32)).to(device)

    return TensorBatch(
        input_items=items(batch.input_items),
        input_residuals=floats(batch.input_residuals),
        target_items=items(batch.target_items),
        target_residuals=floats(batch.target_residuals),
        negative_items=items(negatives),
    )


# ──────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────


def build_optimizer(model: SASRec, args: TrainingArgs) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))


def train_step(
    model: SASRec,
    optimizer: torch.optim.Optimizer,
    batch: TensorBatch,
    residual_std: float,
    l2_emb: float = 0.0,
) -> SasrecLosses:
    """
    One optimisation step. Returns the pre-step losses, detached.

    Only ``batch.input_residuals`` is passed to the model. ``target_residuals``
    appears solely as the MSE label.
    """
    model.train()
    optimizer.zero_grad()
    pos_logits, neg_logits, predicted = model(
        batch.input_items,
        batch.target_items,
        batch.negative_items,
        input_residuals=batch.input_residuals,
        residual_std=residual_std,
    )
    losses = model.losses(
        pos_logits, neg_logits, predicted, batch.target_residuals, batch.target_items
    )
    total = losses.total
    if l2_emb:
        # pmixer: loss += l2_emb * torch.norm(item_emb), item table only.
        total = total + l2_emb * torch.norm(model.item_emb.weight)
    total.backward()
    optimizer.step()
    return SasrecLosses(
        bce=losses.bce.detach(), mse=losses.mse.detach(), total=total.detach()
    )


def _epoch_seed(seed: int, epoch: int) -> list[int]:
    return [int(seed), int(epoch)]


def train_epoch(
    model: SASRec,
    optimizer: torch.optim.Optimizer,
    seq_data: SasrecSequenceData,
    user_items: dict[int, np.ndarray],
    config: SimConfig,
    args: TrainingArgs,
    epoch: int,
) -> dict[str, float]:
    """One shuffled pass over every training window. Returns mean losses."""
    rng = np.random.default_rng(_epoch_seed(config.random_seed, epoch))
    torch.manual_seed(int(rng.integers(0, 2**62)))

    index = seq_data.training_window_index(config.sasrec_maxlen, config.sasrec_window_stride)
    order = rng.permutation(len(index))
    totals = {"bce": 0.0, "mse": 0.0, "total": 0.0}
    n_batches = 0
    for start in range(0, len(order), args.batch_size):
        windows = index[order[start : start + args.batch_size]]
        batch = seq_data.training_batch(windows)
        negatives = sample_negatives(batch, user_items, seq_data.vocab_size, rng)
        losses = train_step(
            model,
            optimizer,
            to_tensors(batch, negatives, model.device),
            seq_data.residual_std,
            args.l2_emb,
        )
        for key in totals:
            totals[key] += float(getattr(losses, key))
        n_batches += 1
    means = {key: value / max(n_batches, 1) for key, value in totals.items()}
    for key, value in means.items():
        if not math.isfinite(value):
            raise FloatingPointError(f"epoch {epoch}: {key} loss is {value}")
    return means


# ──────────────────────────────────────────────────────────────────────────
# Checkpoints
# ──────────────────────────────────────────────────────────────────────────


def save_checkpoint(
    path: Path,
    model: SASRec,
    optimizer: torch.optim.Optimizer,
    seq_data: SasrecSequenceData,
    config: SimConfig,
    args: TrainingArgs,
    epoch: int,
    mlflow_run_id: str | None,
) -> None:
    """Written to a temp file and renamed, so an interruption mid-save never corrupts it."""
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "index_to_item": list(seq_data.index_to_item),
        "residual_std": float(seq_data.residual_std),
        "config": config.to_json_dict(),
        "training_args": dataclasses.asdict(args),
        "mlflow_run_id": mlflow_run_id,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[SASRec, dict]:
    """Rebuild the model from a checkpoint. Returns ``(model, payload)``."""
    return load_sasrec_checkpoint(path, device)


def _check_resumable(payload: dict, config: SimConfig, args: TrainingArgs) -> None:
    saved_config = payload["config"]
    current_config = config.to_json_dict()
    config_diff = sorted(
        k for k in set(saved_config) | set(current_config)
        if saved_config.get(k) != current_config.get(k)
    )
    saved_args = payload["training_args"]
    args_diff = sorted(
        k for k, v in dataclasses.asdict(args).items()
        if k not in RESUMABLE_CHANGES and saved_args.get(k) != v
    )
    if config_diff or args_diff:
        raise ValueError(
            "checkpoint was trained with different settings; refusing to resume. "
            f"config: {config_diff}, training args: {args_diff}"
        )


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────


def train(
    seq_data: SasrecSequenceData,
    config: SimConfig,
    args: TrainingArgs,
    checkpoint_dir: Path,
    device: torch.device | None = None,
    resume: bool = False,
) -> tuple[SASRec, list[dict[str, float]]]:
    """
    Train, checkpointing to ``checkpoint_dir``, logging to MLflow.

    ``config`` should come from ``resolve_config``. Returns the model and the
    mean losses of each epoch run in this call.
    """
    if seq_data.maxlen != config.sasrec_maxlen:
        raise ValueError(
            f"sequence data built at maxlen {seq_data.maxlen}, config says {config.sasrec_maxlen}"
        )
    device = device or select_device()
    checkpoint_path = checkpoint_dir / CHECKPOINT_NAME

    torch.manual_seed(config.random_seed)
    model = SASRec.from_config(config, item_num=seq_data.vocab_size - 1).to(device)
    optimizer = build_optimizer(model, args)
    start_epoch = 1
    run_id: str | None = None

    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError(
                f"{checkpoint_path} exists; pass resume=True (--resume) to continue it"
            )
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        _check_resumable(payload, config, args)
        if payload["index_to_item"] != list(seq_data.index_to_item):
            raise ValueError("checkpoint item vocabulary differs from the current data")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"]) + 1
        run_id = payload["mlflow_run_id"]
        logger.info("Resuming from epoch %d", start_epoch)

    user_items = {uid: np.unique(seq) for uid, seq in seq_data.user_sequences.items()}
    n_windows = len(
        seq_data.training_window_index(config.sasrec_maxlen, config.sasrec_window_stride)
    )

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    history: list[dict[str, float]] = []
    with mlflow.start_run(run_id=run_id) as run:
        if run_id is None:
            mlflow.log_params(config.as_dict())
            mlflow.log_params({f"train_{k}": v for k, v in dataclasses.asdict(args).items()
                               if k not in RESUMABLE_CHANGES})
            mlflow.log_params({
                "data_n_windows": n_windows,
                "data_n_users": seq_data.num_users,
                "data_vocab_size": seq_data.vocab_size,
                "data_residual_std": seq_data.residual_std,
            })
        mlflow.set_tag("device", str(device))
        logger.info("%d windows over %d users, device %s", n_windows, seq_data.num_users, device)

        for epoch in range(start_epoch, args.epochs + 1):
            means = train_epoch(model, optimizer, seq_data, user_items, config, args, epoch)
            history.append(means)
            mlflow.log_metrics({f"train/loss_{k}": v for k, v in means.items()}, step=epoch)
            logger.info("epoch %d: %s", epoch, means)
            if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
                save_checkpoint(
                    checkpoint_path, model, optimizer, seq_data, config, args,
                    epoch, run.info.run_id,
                )
    return model, history


def main(argv: list[str] | None = None) -> None:
    defaults = TrainingArgs()
    parser = argparse.ArgumentParser(description="Train the SASRec preference backend.")
    parser.add_argument("--config", type=Path, default=None,
                        help="SimConfig JSON; SimConfig defaults if omitted.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Continue from the checkpoint in --checkpoint-dir.")
    parser.add_argument("--rating-loss-weight", type=float, default=None,
                        help="Override sasrec_rating_loss_weight; 0 is the ranking-only ablation.")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--l2-emb", type=float, default=defaults.l2_emb)
    parser.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every)
    ns = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base = SimConfig.from_json_file(ns.config) if ns.config else SimConfig()
    config = resolve_config(base, ns.rating_loss_weight)
    args = TrainingArgs(
        epochs=ns.epochs,
        batch_size=ns.batch_size,
        lr=ns.lr,
        l2_emb=ns.l2_emb,
        checkpoint_every=ns.checkpoint_every,
    )

    env = Environment(config)
    seq_data = build_training_data(env, config)
    train(seq_data, config, args, ns.checkpoint_dir, resume=ns.resume)


if __name__ == "__main__":
    main()
