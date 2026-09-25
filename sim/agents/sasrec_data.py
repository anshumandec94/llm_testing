"""
Sequence data for SASRec.

SASRec consumes one time-ordered item sequence per user. This module turns
``Environment.train_ratings`` into those sequences, plus the item vocabulary
the model's embedding table is indexed by.

Three properties are load-bearing and are asserted by ``tests/test_sasrec_data.py``
rather than assumed:

* **Sequences are built from training rows only.** ``Environment._make_holdout_split``
  already removes held-out and validation rows from ``train_ratings``, so this is
  leakage-clean by construction. It is still tested, because a change to the split
  would silently poison every SASRec number.
* **The vocabulary covers every ``movieId`` in the ratings file**, not just the
  items that appear in training sequences. That guarantees an embedding row for
  every held-out item. Dropping held-out items that lack an embedding was
  explicitly rejected: it would give the SASRec arm a different evaluation set
  and recreate the item-selection confound that epic #1 spent four sub-issues
  removing.
* **Sequences are ascending by timestamp.** Note that
  ``Environment.held_out_for_user`` returns rows *descending* by timestamp, which
  is the opposite convention. Do not reuse that ordering here.

Index 0 is reserved for padding and never names a real item. Sequences are
left-padded, so the most recent interaction is always the last position.

Training and inference see a long history differently, on purpose:

* **Inference** (``padded_sequence`` / ``padded_matrix``) uses only the user's
  most recent ``maxlen`` interactions, because that is the context the model
  actually has at prediction time.
* **Training** (``training_window_index`` / ``training_batch``) cuts the whole
  history into contiguous windows, so a user with 1,000 interactions
  contributes all of them rather than their last ``maxlen``. At ML-32M and
  ``maxlen=200`` the single-tail approach discards 39% of the training data
  (issue #24). This is safe only because the port has no user embedding: a
  user is nothing but a sequence, so splitting one history into several
  windows fragments no identity.

Alongside each item the sequence carries that interaction's debiased residual
(``rating - (global_bias + user_bias + item_bias)``, from LensKit's damped
``BiasModel``), and the whole dataset carries the train-set standard deviation of
those residuals. The model's optional rating-injection path normalises by that
standard deviation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Reserved embedding row. Never assigned to a real movieId.
PAD_INDEX = 0

# Occupies index_to_item[PAD_INDEX]. Not a movieId, and chosen negative so it
# can never collide with one.
PAD_SENTINEL_ID = -1

# pmixer/SASRec.pytorch uses maxlen=200 for MovieLens-1M.
DEFAULT_MAXLEN = 200


def resolve_window_stride(
    stride: int | np.integer | None, maxlen: int | np.integer
) -> int:
    """
    Effective training-window stride.

    ``None`` means "equal to ``maxlen``", which gives disjoint windows that
    make every interaction a target exactly once. ``0`` is rejected rather than
    treated as a second spelling of that default, so a zero that arrives by
    accident fails loudly. A stride above ``maxlen`` is rejected too: it would
    leave gaps between windows and silently discard interactions, which is the
    exact problem windowing exists to fix.

    Both arguments must be real integers. ``2.5`` would otherwise truncate and
    ``True`` would pass as ``1``, and either is far more likely a config typo
    than an intended stride.
    """
    for name, value in (("maxlen", maxlen), ("window stride", stride)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise TypeError(f"{name} must be an int, got {value!r}")
    if maxlen < 1:
        raise ValueError(f"training windows need maxlen >= 1, got {maxlen}")
    if stride is None:
        return int(maxlen)
    if not 1 <= stride <= maxlen:
        raise ValueError(
            f"window stride must be in [1, maxlen={maxlen}], got {stride}; "
            f"use None for stride = maxlen"
        )
    return int(stride)


@dataclass
class SasrecTrainingBatch:
    """
    Materialised training windows, one row per window.

    ``target_items[:, t]`` is the item that follows ``input_items[:, t]`` in
    the user's history, which is the shift ``SASRec.forward`` and
    ``SASRec.losses`` expect. Both are left-padded with ``PAD_INDEX``, and a
    padded target contributes nothing to the loss. Negatives are not sampled
    here; that is the training loop's job.
    """

    input_items: np.ndarray
    """``(n_windows, maxlen)`` int32 item indices fed to the model."""

    input_residuals: np.ndarray
    """``(n_windows, maxlen)`` float32 residuals of the input positions."""

    target_items: np.ndarray
    """``(n_windows, maxlen)`` int32 next-item targets."""

    target_residuals: np.ndarray
    """``(n_windows, maxlen)`` float32 residuals of the targets. Labels only."""

    user_ids: np.ndarray
    """``(n_windows,)`` owning userId per row. Diagnostic only; the model takes no user input."""


@dataclass(frozen=True)
class SasrecWindowIndex:
    """
    Training windows, enumerated but not materialised.

    Carries the ``maxlen`` and stride the rows were built with, so
    ``training_batch`` cannot materialise them at a different width. A batch
    narrower than its index would silently drop the oldest part of every
    window, which no shape check downstream would catch.

    Index with ``[]`` (a slice, an integer array from a shuffle, a mask) to
    take a mini-batch; the result keeps the same ``maxlen`` and stride.
    """

    rows: np.ndarray
    """``(n_windows, 2)`` int64 ``(userId, target_end)`` rows."""

    maxlen: int
    stride: int

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key) -> "SasrecWindowIndex":
        return SasrecWindowIndex(
            rows=self.rows[key].reshape(-1, 2),
            maxlen=self.maxlen,
            stride=self.stride,
        )

    @property
    def user_ids(self) -> np.ndarray:
        return self.rows[:, 0]

    @property
    def target_ends(self) -> np.ndarray:
        return self.rows[:, 1]


@dataclass
class SasrecSequenceData:
    """Time-ordered training sequences and the item vocabulary behind them."""

    item_to_index: dict[int, int]
    """movieId -> embedding row. Values start at 1; 0 is padding."""

    index_to_item: list[int]
    """Embedding row -> movieId. Position 0 holds the padding sentinel ``-1``."""

    user_sequences: dict[int, np.ndarray]
    """
    userId -> item indices, ascending by timestamp. Unpadded, untruncated.

    Each array owns its data, so a consumer may edit one user's sequence in
    place without corrupting another's.
    """

    user_residuals: dict[int, np.ndarray]
    """
    userId -> debiased residual per position, index-aligned with the matching
    ``user_sequences`` entry: ``user_residuals[u][t]`` is the residual of the
    interaction whose item is ``user_sequences[u][t]``.
    """

    residual_std: float
    """Population standard deviation of the residuals over all training interactions."""

    maxlen: int
    """Sequence length the padding helpers and training windows emit."""

    window_stride: int | None = None
    """
    Default stride for ``training_window_index``. ``None`` means ``maxlen``
    (disjoint windows); see ``resolve_window_stride``.
    """

    @property
    def vocab_size(self) -> int:
        """Number of embedding rows, including the padding row at index 0."""
        return len(self.index_to_item)

    @property
    def num_users(self) -> int:
        """
        Users, not training rows. Once windowed, one user may contribute many
        training rows; count those with ``len(training_window_index())``.
        """
        return len(self.user_sequences)

    def item_index(self, movie_id: int) -> int:
        """
        Embedding row for a movieId.

        Raises ``KeyError`` for an unknown id rather than falling back to
        ``PAD_INDEX``. Silently returning the padding row would conflate "this
        item has no embedding" with "there is no item at this position", and a
        pad token spliced mid-sequence is dropped by the model's padding mask,
        so the sequence would quietly shorten instead of failing.

        The vocabulary spans every movieId in the ratings file, so every
        held-out item resolves. An unknown id means a movie with no ratings at
        all, which is a caller bug worth surfacing. Callers that need a
        tolerant lookup should catch this and decide explicitly.
        """
        try:
            return self.item_to_index[int(movie_id)]
        except KeyError:
            raise KeyError(
                f"movieId {movie_id} is not in the SASRec vocabulary "
                f"({self.vocab_size - 1} rated items). It has no ratings, so "
                f"it has no embedding row."
            ) from None

    def padded_sequence(
        self, user_id: int, maxlen: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Left-padded item indices and residuals for one user.

        Returns two arrays of length ``maxlen``. Padded positions carry
        ``PAD_INDEX`` and a residual of ``0.0``. Sequences longer than ``maxlen``
        keep their most recent ``maxlen`` interactions.
        """
        length = self.maxlen if maxlen is None else maxlen
        if length < 0:
            raise ValueError(f"maxlen must be non-negative, got {length}")
        items = np.zeros(length, dtype=np.int32)
        residuals = np.zeros(length, dtype=np.float32)

        seq = self.user_sequences.get(int(user_id))
        # `tail == 0` is handled before the slicing below, because `seq[-0:]`
        # is the whole array rather than an empty one.
        if seq is None or len(seq) == 0 or length == 0:
            return items, residuals

        res = self.user_residuals[int(user_id)]
        tail = min(length, len(seq))
        items[length - tail :] = seq[-tail:]
        residuals[length - tail :] = res[-tail:]
        return items, residuals

    def padded_matrix(
        self, user_ids: list[int] | None = None, maxlen: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """
        ``padded_sequence`` stacked over many users.

        Returns ``(items, residuals, user_ids)`` where the arrays are
        ``(n_users, maxlen)`` and ``user_ids`` gives the row order.
        """
        ids = sorted(self.user_sequences) if user_ids is None else [int(u) for u in user_ids]
        length = self.maxlen if maxlen is None else maxlen
        items = np.zeros((len(ids), length), dtype=np.int32)
        residuals = np.zeros((len(ids), length), dtype=np.float32)
        for row, uid in enumerate(ids):
            items[row], residuals[row] = self.padded_sequence(uid, length)
        return items, residuals, ids

    def training_window_index(
        self,
        maxlen: int | None = None,
        stride: int | None = None,
        user_ids: list[int] | None = None,
    ) -> SasrecWindowIndex:
        """
        Enumerate the training windows without materialising them.

        Returns a ``SasrecWindowIndex`` of ``(userId, target_end)`` rows that
        remembers this ``maxlen`` and stride. A window's targets are the user's
        positions ``[max(1, target_end - maxlen), target_end)`` and its inputs
        are the same range shifted back by one, so every window is a
        contiguous, timestamp-ascending "predict the next item" problem. Pass
        the index (all of it, or a shuffled mini-batch of it) to
        ``training_batch``.

        Rows follow ``user_ids`` order, sorted userIds by default. As with
        ``padded_matrix``, a userId listed twice is enumerated twice, so its
        windows appear twice; deduplicate first if that is not wanted. Unknown
        userIds are skipped.

        **Anchoring.** Windows are anchored at the most recent end:
        ``target_end`` runs ``L, L - stride, L - 2 * stride, ...`` for a
        sequence of length ``L``. The newest window is therefore exactly the
        single example pmixer's sampler builds (targets are the last ``maxlen``
        interactions, inputs the ``maxlen`` before each), so the pre-windowing
        training set is a strict subset of this one. The oldest window holds
        the short remainder and is left-padded like any short user.
        Enumeration stops at the first window that reaches the start of the
        history, because every older window would be a prefix of it.

        With the default ``stride == maxlen`` the target ranges partition
        ``[1, L)``: every interaction except the user's first is a target
        exactly once. A smaller stride overlaps windows and oversamples heavy
        users. A user with fewer than two interactions has no next-item target
        and yields no window, matching pmixer's sampler.

        Training only. Inference must keep using ``padded_sequence``, which
        sees the most recent ``maxlen`` interactions and nothing else.
        """
        length = self.maxlen if maxlen is None else maxlen
        step = resolve_window_stride(
            self.window_stride if stride is None else stride, length
        )
        ids = sorted(self.user_sequences) if user_ids is None else [int(u) for u in user_ids]

        rows: list[tuple[int, int]] = []
        for uid in ids:
            seq = self.user_sequences.get(uid)
            if seq is None:
                continue
            end = len(seq)
            while end >= 2:
                rows.append((uid, end))
                if end - length <= 1:
                    break
                end -= step
        return SasrecWindowIndex(
            rows=np.array(rows, dtype=np.int64).reshape(-1, 2),
            maxlen=int(length),
            stride=step,
        )

    def training_batch(self, windows: SasrecWindowIndex) -> SasrecTrainingBatch:
        """
        Materialise a ``SasrecWindowIndex`` into padded arrays.

        Kept separate from the index so a training loop can shuffle and batch
        windows without holding every window of ML-32M in memory at once. The
        width is the index's own ``maxlen`` and cannot be overridden here, so a
        batch always holds whole windows.
        """
        if not isinstance(windows, SasrecWindowIndex):
            raise TypeError(
                "training_batch takes a SasrecWindowIndex from "
                f"training_window_index, got {type(windows).__name__}"
            )
        length = windows.maxlen
        n = len(windows)
        input_items = np.zeros((n, length), dtype=np.int32)
        input_residuals = np.zeros((n, length), dtype=np.float32)
        target_items = np.zeros((n, length), dtype=np.int32)
        target_residuals = np.zeros((n, length), dtype=np.float32)

        for row, (uid, end) in enumerate(windows.rows.tolist()):
            seq = self.user_sequences[uid]
            res = self.user_residuals[uid]
            if not 2 <= end <= len(seq):
                raise ValueError(
                    f"window end {end} is out of range for user {uid} "
                    f"with {len(seq)} interactions"
                )
            start = max(1, end - length)
            width = end - start
            target_items[row, length - width :] = seq[start:end]
            target_residuals[row, length - width :] = res[start:end]
            input_items[row, length - width :] = seq[start - 1 : end - 1]
            input_residuals[row, length - width :] = res[start - 1 : end - 1]

        return SasrecTrainingBatch(
            input_items=input_items,
            input_residuals=input_residuals,
            target_items=target_items,
            target_residuals=target_residuals,
            user_ids=windows.user_ids.copy(),
        )


def build_item_vocabulary(all_ratings: pd.DataFrame) -> tuple[dict[int, int], list[int]]:
    """
    Map every ``movieId`` in the ratings file to an embedding row.

    Ids are sorted so the mapping is deterministic across runs and machines,
    which matters because #20 merges per-item results computed on different
    hosts. Index 0 is the padding row and holds the sentinel ``-1``.
    """
    movie_ids = np.sort(all_ratings["movieId"].unique())
    index_to_item = [PAD_SENTINEL_ID] + [int(mid) for mid in movie_ids]
    item_to_index = {mid: idx for idx, mid in enumerate(index_to_item) if idx != PAD_INDEX}
    return item_to_index, index_to_item


def build_sasrec_sequences(
    env, maxlen: int = DEFAULT_MAXLEN, window_stride: int | None = None
) -> SasrecSequenceData:
    """
    Build SASRec training sequences from ``env.train_ratings``.

    All users are included; there is no ``min_ratings`` filter, because the
    benchmark scores every eval user and a filtered training set would quietly
    change which users the arm can represent.

    ``window_stride`` becomes the default for ``training_window_index``. It is
    validated here, so a bad ``SimConfig`` fails before the expensive build
    rather than at the first training batch.
    """
    resolve_window_stride(window_stride, maxlen)
    item_to_index, index_to_item = build_item_vocabulary(env.all_ratings)

    train = env.train_ratings
    # Sorting the DataFrame would hold a second copy of the whole frame alive;
    # at ML-32M that is a few GB on top of all_ratings and train_ratings, which
    # is the shape of the OOM fixed in dc08029. Pull the four columns out as
    # 1-D arrays first and sort those instead. np.lexsort applies its keys
    # last-first, so this orders by userId, then timestamp within each user,
    # and it is stable, so tied timestamps keep their order in train_ratings.
    user_ids = train["userId"].to_numpy()
    movie_ids = train["movieId"].to_numpy()
    ratings = train["rating"].to_numpy()
    timestamps = train["timestamp"].to_numpy()

    order = np.lexsort((timestamps, user_ids))
    user_ids = user_ids[order]
    movie_ids = movie_ids[order]
    ratings = ratings[order]
    del timestamps, order

    n_rows = len(user_ids)
    residuals = np.fromiter(
        (
            env.debias_rating(int(u), int(i), float(r))
            for u, i, r in zip(user_ids, movie_ids, ratings)
        ),
        dtype=np.float32,
        count=n_rows,
    )
    # Every training item is in the vocabulary because train_ratings is a
    # subset of all_ratings. item_index raises rather than padding if that
    # ever stops being true.
    indices = np.fromiter(
        (item_to_index[int(i)] for i in movie_ids),
        dtype=np.int32,
        count=n_rows,
    )

    # Sorted by userId, so each user is one contiguous block. The blocks are
    # copied rather than sliced: a view would alias the shared base arrays, so
    # editing one user's sequence in place would corrupt the source, and
    # holding any single user's sequence would pin the whole allocation.
    user_sequences: dict[int, np.ndarray] = {}
    user_residuals: dict[int, np.ndarray] = {}
    if n_rows > 0:
        boundaries = np.flatnonzero(np.diff(user_ids)) + 1
        for start, stop in zip(
            np.concatenate(([0], boundaries)),
            np.concatenate((boundaries, [n_rows])),
        ):
            uid = int(user_ids[start])
            user_sequences[uid] = indices[start:stop].copy()
            user_residuals[uid] = residuals[start:stop].copy()

    residual_std = float(np.std(residuals)) if n_rows > 0 else 0.0

    return SasrecSequenceData(
        item_to_index=item_to_index,
        index_to_item=index_to_item,
        user_sequences=user_sequences,
        user_residuals=user_residuals,
        residual_std=residual_std,
        maxlen=maxlen,
        window_stride=window_stride,
    )
