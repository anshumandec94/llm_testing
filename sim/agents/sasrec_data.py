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
    """Sequence length the padding helpers emit."""

    @property
    def vocab_size(self) -> int:
        """Number of embedding rows, including the padding row at index 0."""
        return len(self.index_to_item)

    @property
    def num_users(self) -> int:
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


def build_sasrec_sequences(env, maxlen: int = DEFAULT_MAXLEN) -> SasrecSequenceData:
    """
    Build SASRec training sequences from ``env.train_ratings``.

    All users are included; there is no ``min_ratings`` filter, because the
    benchmark scores every eval user and a filtered training set would quietly
    change which users the arm can represent.
    """
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
    )
