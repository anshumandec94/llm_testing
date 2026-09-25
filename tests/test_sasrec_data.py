"""
Tests for sim/agents/sasrec_data.py.

The four assertions the sub-issue calls out are the reason this file exists:
no held-out pair leaks into a training sequence, the vocabulary covers every
movieId in the ratings file, sequences are ascending by timestamp, and padding
sits at the left with index 0.

Uses the synthetic fixtures in conftest.py. Nothing here touches data/ml-32m/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sim.agents.sasrec_data import (
    DEFAULT_MAXLEN,
    PAD_INDEX,
    PAD_SENTINEL_ID,
    build_item_vocabulary,
    build_sasrec_sequences,
    resolve_window_stride,
)
from sim.config import SimConfig


@pytest.fixture(scope="module")
def seq_data(env):
    return build_sasrec_sequences(env, maxlen=8)


class _StubEnv:
    """
    Minimal stand-in exposing only what build_sasrec_sequences reads.

    The shared `env` fixture is dense enough that every movie appears in some
    training row, so it cannot exercise the held-out-only item case. This can.
    """

    def __init__(self, all_ratings: pd.DataFrame, train_ratings: pd.DataFrame) -> None:
        self.all_ratings = all_ratings
        self.train_ratings = train_ratings

    def debias_rating(self, user_id: int, movie_id: int, rating: float) -> float:
        return float(rating - 3.0)


@pytest.fixture
def held_out_only_env():
    """
    Movie 99 is rated once, by user 1, and that rating is held out. It appears
    in the ratings file and in no training row.
    """
    all_ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2],
            "movieId": [10, 20, 99, 10, 30],
            "rating": [4.0, 2.0, 5.0, 3.0, 1.0],
            "timestamp": [100, 200, 300, 150, 250],
        }
    )
    train_ratings = all_ratings[
        ~(
            (all_ratings["userId"] == 1)
            & (all_ratings["movieId"] == 99)
        )
    ].reset_index(drop=True)
    return _StubEnv(all_ratings, train_ratings)


class TestVocabulary:
    def test_covers_every_movie_in_the_ratings_file(self, env, seq_data):
        rated = {int(m) for m in env.all_ratings["movieId"].unique()}
        assert rated <= set(seq_data.item_to_index)

    def test_covers_items_absent_from_training_sequences(self, held_out_only_env):
        """
        The point of building the vocabulary from the ratings file rather than
        from the training rows: an item can be held out for its only rater and
        so never appear in a sequence, and it still needs an embedding row.
        """
        data = build_sasrec_sequences(held_out_only_env, maxlen=4)
        in_sequences = {int(i) for seq in data.user_sequences.values() for i in seq}

        assert data.item_index(99) not in in_sequences
        assert data.item_index(99) != PAD_INDEX
        assert data.index_to_item[data.item_index(99)] == 99
        assert set(data.item_to_index) == {10, 20, 30, 99}

    def test_index_zero_is_reserved_for_padding(self, seq_data):
        assert PAD_INDEX == 0
        assert seq_data.index_to_item[PAD_INDEX] == PAD_SENTINEL_ID
        assert PAD_SENTINEL_ID not in seq_data.item_to_index
        assert PAD_INDEX not in seq_data.item_to_index.values()

    def test_unknown_movie_raises_rather_than_returning_padding(self, seq_data):
        """
        Falling back to PAD_INDEX would splice a pad token into a sequence,
        where the model's padding mask drops it and the sequence silently
        shortens instead of failing.
        """
        with pytest.raises(KeyError, match="not in the SASRec vocabulary"):
            seq_data.item_index(10**9)

    def test_indices_are_dense_and_deterministic(self, env):
        item_to_index, index_to_item = build_item_vocabulary(env.all_ratings)
        assert sorted(item_to_index.values()) == list(range(1, len(index_to_item)))
        again, _ = build_item_vocabulary(env.all_ratings.sample(frac=1.0, random_state=1))
        assert again == item_to_index

    def test_vocab_size_counts_the_padding_row(self, env, seq_data):
        n_movies = env.all_ratings["movieId"].nunique()
        assert seq_data.vocab_size == n_movies + 1


class TestNoLeakage:
    def test_no_held_out_pair_appears_in_any_training_sequence(self, env, seq_data):
        in_sequences = {
            (uid, int(seq_data.index_to_item[int(idx)]))
            for uid, seq in seq_data.user_sequences.items()
            for idx in seq
        }
        held_out = {
            (int(r.userId), int(r.movieId))
            for r in env.held_out.itertuples()
        }
        assert not (in_sequences & held_out)

    def test_no_validation_pair_appears_in_any_training_sequence(self, env, seq_data):
        in_sequences = {
            (uid, int(seq_data.index_to_item[int(idx)]))
            for uid, seq in seq_data.user_sequences.items()
            for idx in seq
        }
        validation = {
            (int(r.userId), int(r.movieId))
            for r in env.validation.itertuples()
        }
        assert not (in_sequences & validation)

    def test_sequences_reproduce_the_training_rows_exactly(self, env, seq_data):
        total = sum(len(seq) for seq in seq_data.user_sequences.values())
        assert total == len(env.train_ratings)


class TestOrdering:
    def test_sequences_are_ascending_by_timestamp(self, env, seq_data):
        train = env.train_ratings
        for uid, seq in seq_data.user_sequences.items():
            user_rows = train[train["userId"] == uid]
            stamps = {
                int(r.movieId): int(r.timestamp) for r in user_rows.itertuples()
            }
            ordered = [stamps[int(seq_data.index_to_item[int(i)])] for i in seq]
            assert ordered == sorted(ordered)

    def test_rows_are_sorted_regardless_of_frame_order(self, held_out_only_env):
        """
        Environment builds train_ratings from timestamp-descending slices, so
        the input frame is the wrong way round. Sorting is doing real work here.
        """
        shuffled = held_out_only_env.train_ratings.sort_values(
            "timestamp", ascending=False
        )
        env_desc = _StubEnv(held_out_only_env.all_ratings, shuffled)
        data = build_sasrec_sequences(env_desc, maxlen=4)
        # User 1 trains on movies 10 (t=100) then 20 (t=200).
        assert list(data.user_sequences[1]) == [
            data.item_index(10),
            data.item_index(20),
        ]
        assert list(data.user_sequences[2]) == [
            data.item_index(10),
            data.item_index(30),
        ]

    def test_sequences_run_opposite_to_the_held_out_convention(self, env, seq_data):
        """
        held_out_for_user returns rows timestamp-descending. Assert the
        sequences are the other way round for the same user, so reusing that
        convention here fails rather than silently reversing every history.
        """
        uid = int(env.eval_users[0])
        held_stamps = [int(t) for t in env.held_out_for_user(uid)["timestamp"]]
        assert held_stamps == sorted(held_stamps, reverse=True), (
            "Environment changed convention; this test's premise is stale"
        )

        train = env.train_ratings
        user_rows = train[train["userId"] == uid]
        stamps = {int(r.movieId): int(r.timestamp) for r in user_rows.itertuples()}
        seq_stamps = [
            stamps[seq_data.index_to_item[int(i)]]
            for i in seq_data.user_sequences[uid]
        ]
        assert len(seq_stamps) > 1
        assert seq_stamps == sorted(seq_stamps)
        assert seq_stamps != sorted(seq_stamps, reverse=True)


class TestPadding:
    def test_padding_sits_at_the_left_with_index_zero(self, env, seq_data):
        short_user = min(
            seq_data.user_sequences,
            key=lambda u: len(seq_data.user_sequences[u]),
        )
        seq = seq_data.user_sequences[short_user]
        maxlen = len(seq) + 3
        items, residuals = seq_data.padded_sequence(short_user, maxlen=maxlen)

        assert len(items) == maxlen
        assert np.all(items[:3] == PAD_INDEX)
        assert np.array_equal(items[3:], seq)
        assert np.all(residuals[:3] == 0.0)
        assert np.array_equal(residuals[3:], seq_data.user_residuals[short_user])

    def test_long_sequences_keep_their_most_recent_items(self, seq_data):
        long_user = max(
            seq_data.user_sequences,
            key=lambda u: len(seq_data.user_sequences[u]),
        )
        seq = seq_data.user_sequences[long_user]
        assert len(seq) > 3
        items, residuals = seq_data.padded_sequence(long_user, maxlen=3)
        assert np.array_equal(items, seq[-3:])
        assert np.array_equal(residuals, seq_data.user_residuals[long_user][-3:])

    def test_truncation_keeps_each_item_with_its_own_residual(self, seq_data):
        """
        The alignment #17 depends on: position t of the padded tensors must
        describe one interaction. Truncating items from the tail and residuals
        from the head would pass every length and item check while pairing each
        item with a different interaction's rating.
        """
        long_user = max(
            seq_data.user_sequences,
            key=lambda u: len(seq_data.user_sequences[u]),
        )
        full_items = seq_data.user_sequences[long_user]
        full_res = seq_data.user_residuals[long_user]
        by_position = dict(zip(full_items.tolist(), full_res.tolist()))
        assert len(by_position) == len(full_items), "user repeats an item; pick another"

        for maxlen in (1, 2, len(full_items) - 1, len(full_items), len(full_items) + 4):
            items, residuals = seq_data.padded_sequence(long_user, maxlen=maxlen)
            real = items != PAD_INDEX
            assert real.any()
            for item, residual in zip(items[real].tolist(), residuals[real].tolist()):
                assert residual == pytest.approx(by_position[item])

    def test_padded_residuals_are_not_identically_zero(self, seq_data):
        """Guards the payload region, which the padding checks above skip."""
        uid = max(
            seq_data.user_sequences,
            key=lambda u: len(seq_data.user_sequences[u]),
        )
        _, residuals = seq_data.padded_sequence(uid)
        assert np.any(residuals != 0.0)

    def test_maxlen_zero_gives_empty_arrays(self, seq_data):
        uid = next(iter(seq_data.user_sequences))
        items, residuals = seq_data.padded_sequence(uid, maxlen=0)
        assert len(items) == 0
        assert len(residuals) == 0

    def test_negative_maxlen_is_rejected(self, seq_data):
        uid = next(iter(seq_data.user_sequences))
        with pytest.raises(ValueError, match="non-negative"):
            seq_data.padded_sequence(uid, maxlen=-1)

    def test_default_maxlen_comes_from_the_dataset(self, seq_data):
        items, residuals = seq_data.padded_sequence(
            next(iter(seq_data.user_sequences))
        )
        assert len(items) == seq_data.maxlen == 8
        assert len(residuals) == seq_data.maxlen

    def test_unknown_user_gives_an_all_padding_sequence(self, seq_data):
        items, residuals = seq_data.padded_sequence(-999)
        assert np.all(items == PAD_INDEX)
        assert np.all(residuals == 0.0)

    def test_padded_matrix_rows_match_padded_sequence(self, seq_data):
        uids = sorted(seq_data.user_sequences)[:5]
        items, residuals, order = seq_data.padded_matrix(uids, maxlen=6)
        assert order == uids
        assert items.shape == (5, 6)
        for row, uid in enumerate(order):
            expected_items, expected_res = seq_data.padded_sequence(uid, maxlen=6)
            assert np.array_equal(items[row], expected_items)
            assert np.array_equal(residuals[row], expected_res)


class TestResiduals:
    def test_residuals_match_env_debias_rating(self, env, seq_data):
        train = env.train_ratings
        uid = int(train["userId"].iloc[0])
        user_rows = train[train["userId"] == uid].sort_values(
            "timestamp", kind="mergesort"
        )
        expected = [
            env.debias_rating(uid, int(r.movieId), float(r.rating))
            for r in user_rows.itertuples()
        ]
        assert seq_data.user_residuals[uid] == pytest.approx(expected, abs=1e-5)

    def test_residuals_align_with_items_position_by_position(self, env, seq_data):
        """
        Not just equal lengths, which the block-slicing guarantees for free:
        position t's residual must belong to position t's item.
        """
        train = env.train_ratings
        for uid in sorted(seq_data.user_sequences)[:5]:
            user_rows = train[train["userId"] == uid]
            expected = {
                int(r.movieId): env.debias_rating(uid, int(r.movieId), float(r.rating))
                for r in user_rows.itertuples()
            }
            seq = seq_data.user_sequences[uid]
            res = seq_data.user_residuals[uid]
            assert len(seq) == len(res)
            for idx, residual in zip(seq.tolist(), res.tolist()):
                movie_id = seq_data.index_to_item[idx]
                assert residual == pytest.approx(expected[movie_id], abs=1e-5)

    def test_arrays_own_their_data(self, seq_data):
        """
        Per-user arrays are copies, not views into a shared base. #18's
        batching would otherwise be able to corrupt other users in place.
        """
        for uid in sorted(seq_data.user_sequences)[:3]:
            assert seq_data.user_sequences[uid].base is None
            assert seq_data.user_residuals[uid].base is None

    def test_residual_std_is_the_train_set_standard_deviation(self, seq_data):
        all_residuals = np.concatenate(list(seq_data.user_residuals.values()))
        assert seq_data.residual_std == pytest.approx(float(np.std(all_residuals)), rel=1e-5)
        assert seq_data.residual_std > 0.0


class TestUserCoverage:
    def test_every_training_user_gets_a_sequence(self, env, seq_data):
        assert set(seq_data.user_sequences) == {
            int(u) for u in env.train_ratings["userId"].unique()
        }

    def test_no_min_ratings_filter_is_applied(self, held_out_only_env):
        """
        A user with a single training interaction still gets a sequence. The
        `env` fixture cannot show this: min_ratings=10 there and every user
        carries far more, so any filter would be invisible.
        """
        one_row_user = pd.DataFrame(
            {"userId": [3], "movieId": [10], "rating": [5.0], "timestamp": [400]}
        )
        env = _StubEnv(
            pd.concat([held_out_only_env.all_ratings, one_row_user], ignore_index=True),
            pd.concat([held_out_only_env.train_ratings, one_row_user], ignore_index=True),
        )
        data = build_sasrec_sequences(env, maxlen=4)
        assert set(data.user_sequences) == {1, 2, 3}
        assert len(data.user_sequences[3]) == 1
        assert data.user_sequences[3][0] == data.item_index(10)

    def test_default_maxlen_matches_the_reference_implementation(self, env):
        data = build_sasrec_sequences(env)
        assert data.maxlen == DEFAULT_MAXLEN == 200


# ── Training windows (issue #24) ────────────────────────────────────────────

LONG_LEN = 23
WINDOW_MAXLEN = 8


@pytest.fixture
def long_user_env():
    """
    User 1 has 23 training interactions, far past maxlen=8, plus one held-out
    rating that is newer than all of them. User 2 has 5, user 3 has 2 and
    user 4 has 1. The `env` fixture is too uniform to pin exact window
    boundaries, so this is built by hand.

    The frame is timestamp-descending, the way Environment builds it, so the
    sort inside build_sasrec_sequences is doing real work.
    """
    rows = []
    for k in range(LONG_LEN):
        rows.append((1, 100 + k, 0.5 + 0.5 * (k % 10), 1_000 + 10 * k))
    for k in range(5):
        rows.append((2, 200 + k, 1.0 + k, 500 + k))
    rows.append((3, 100, 4.0, 50))
    rows.append((3, 101, 2.0, 60))
    rows.append((4, 102, 3.5, 70))
    columns = ("userId", "movieId", "rating", "timestamp")
    train = pd.DataFrame({name: [row[i] for row in rows] for i, name in enumerate(columns)})
    held = pd.DataFrame(
        {"userId": [1], "movieId": [999], "rating": [5.0], "timestamp": [9_999]}
    )
    all_ratings = pd.concat([train, held], ignore_index=True)
    train = train.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return _StubEnv(all_ratings, train)


@pytest.fixture
def long_data(long_user_env):
    return build_sasrec_sequences(long_user_env, maxlen=WINDOW_MAXLEN)


def _user_window_ends(index: np.ndarray, uid: int) -> list[int]:
    return [int(end) for u, end in index.tolist() if u == uid]


class TestWindowEnumeration:
    def test_a_long_user_yields_more_than_one_window(self, long_data):
        ends = _user_window_ends(long_data.training_window_index(), 1)
        assert len(ends) > 1

    def test_windows_are_anchored_at_the_most_recent_end(self, long_data):
        """
        Default stride = maxlen = 8 over 23 items: target ranges [15, 23),
        [7, 15) and the left-padded remainder [1, 7).
        """
        ends = _user_window_ends(long_data.training_window_index(), 1)
        assert ends == [23, 15, 7]

    def test_newest_window_is_the_single_pmixer_example(self, long_data):
        """
        The pre-windowing training example must survive unchanged, so the old
        training set is a strict subset of the new one: targets are the last
        maxlen interactions, inputs the maxlen interactions before each.
        """
        batch = long_data.training_batch(np.array([[1, LONG_LEN]]))
        seq = long_data.user_sequences[1]
        res = long_data.user_residuals[1]
        assert np.array_equal(batch.target_items[0], seq[-WINDOW_MAXLEN:])
        assert np.array_equal(batch.input_items[0], seq[-WINDOW_MAXLEN - 1 : -1])
        assert np.array_equal(batch.target_residuals[0], res[-WINDOW_MAXLEN:])
        assert np.array_equal(batch.input_residuals[0], res[-WINDOW_MAXLEN - 1 : -1])

    def test_oldest_remainder_is_left_padded(self, long_data):
        batch = long_data.training_batch(np.array([[1, 7]]))
        seq = long_data.user_sequences[1]
        # Targets are positions 1..6, inputs 0..5: six real slots, two pads.
        assert np.all(batch.target_items[0, :2] == PAD_INDEX)
        assert np.all(batch.input_items[0, :2] == PAD_INDEX)
        assert np.all(batch.input_residuals[0, :2] == 0.0)
        assert np.array_equal(batch.target_items[0, 2:], seq[1:7])
        assert np.array_equal(batch.input_items[0, 2:], seq[0:6])

    def test_default_stride_makes_every_later_interaction_a_target_once(self, long_data):
        index = long_data.training_window_index()
        for uid, seq in long_data.user_sequences.items():
            rows = index[index[:, 0] == uid]
            covered: list[int] = []
            for _, end in rows.tolist():
                covered.extend(range(max(1, end - WINDOW_MAXLEN), end))
            assert sorted(covered) == list(range(1, len(seq)))

    def test_smaller_stride_overlaps_and_still_covers_everything(self, long_data):
        index = long_data.training_window_index(stride=4)
        ends = _user_window_ends(index, 1)
        assert ends == [23, 19, 15, 11, 7]
        covered = set()
        for end in ends:
            covered.update(range(max(1, end - WINDOW_MAXLEN), end))
        assert covered == set(range(1, LONG_LEN))

    def test_stride_one_is_every_position(self, long_data):
        ends = _user_window_ends(long_data.training_window_index(stride=1), 1)
        assert ends == list(range(LONG_LEN, WINDOW_MAXLEN, -1))

    def test_users_too_short_for_a_target_yield_no_window(self, long_data):
        index = long_data.training_window_index()
        assert _user_window_ends(index, 4) == []
        assert _user_window_ends(index, 3) == [2]
        assert _user_window_ends(index, 2) == [5]

    def test_short_users_get_exactly_one_window(self, long_data):
        """A user within maxlen yields today's single sequence and nothing more."""
        index = long_data.training_window_index()
        for uid, seq in long_data.user_sequences.items():
            if 2 <= len(seq) <= WINDOW_MAXLEN + 1:
                assert _user_window_ends(index, uid) == [len(seq)]

    def test_stride_is_carried_by_the_dataset(self, long_user_env):
        data = build_sasrec_sequences(long_user_env, maxlen=WINDOW_MAXLEN, window_stride=4)
        assert _user_window_ends(data.training_window_index(), 1) == [23, 19, 15, 11, 7]

    def test_user_ids_filter_the_index(self, long_data):
        index = long_data.training_window_index(user_ids=[2, 1])
        assert index[:, 0].tolist() == [2, 1, 1, 1]

    def test_num_users_still_counts_users_not_windows(self, long_data):
        assert long_data.num_users == 4
        assert len(long_data.training_window_index()) == 5


class TestWindowContents:
    def test_windows_are_contiguous_and_ascending_by_timestamp(self, long_user_env, long_data):
        train = long_user_env.train_ratings
        stamps = {
            (int(r.userId), int(r.movieId)): int(r.timestamp)
            for r in train.itertuples()
        }
        index = long_data.training_window_index(stride=3)
        batch = long_data.training_batch(index)
        for row, uid in enumerate(batch.user_ids.tolist()):
            seq = long_data.user_sequences[uid].tolist()
            inputs = batch.input_items[row]
            targets = batch.target_items[row]
            real = targets != PAD_INDEX
            # Padding is identical on both sides and sits at the left.
            assert np.array_equal(real, inputs != PAD_INDEX)
            assert real[-1]
            assert not np.any(np.diff(real.astype(int)) < 0)

            real_inputs = inputs[real].tolist()
            real_targets = targets[real].tolist()
            start = seq.index(real_inputs[0])
            assert real_inputs == seq[start : start + len(real_inputs)]
            assert real_targets == seq[start + 1 : start + 1 + len(real_targets)]

            window_stamps = [
                stamps[(uid, long_data.index_to_item[i])]
                for i in [real_inputs[0], *real_targets]
            ]
            assert window_stamps == sorted(window_stamps)

    def test_each_item_keeps_its_own_residual(self, long_data):
        batch = long_data.training_batch(long_data.training_window_index(stride=5))
        for row, uid in enumerate(batch.user_ids.tolist()):
            by_item = dict(
                zip(
                    long_data.user_sequences[uid].tolist(),
                    long_data.user_residuals[uid].tolist(),
                )
            )
            for items, residuals in (
                (batch.input_items[row], batch.input_residuals[row]),
                (batch.target_items[row], batch.target_residuals[row]),
            ):
                real = items != PAD_INDEX
                for item, residual in zip(items[real].tolist(), residuals[real].tolist()):
                    assert residual == pytest.approx(by_item[item])

    def test_batch_shapes_and_dtypes(self, long_data):
        index = long_data.training_window_index()
        batch = long_data.training_batch(index)
        for arr in (batch.input_items, batch.target_items):
            assert arr.shape == (len(index), WINDOW_MAXLEN)
            assert arr.dtype == np.int32
        for arr in (batch.input_residuals, batch.target_residuals):
            assert arr.shape == (len(index), WINDOW_MAXLEN)
            assert arr.dtype == np.float32
        assert batch.user_ids.tolist() == index[:, 0].tolist()

    def test_out_of_range_window_is_rejected(self, long_data):
        with pytest.raises(ValueError, match="out of range"):
            long_data.training_batch(np.array([[1, LONG_LEN + 1]]))
        with pytest.raises(ValueError, match="out of range"):
            long_data.training_batch(np.array([[1, 1]]))

    def test_held_out_rating_never_enters_a_window(self, long_data):
        batch = long_data.training_batch(long_data.training_window_index(stride=1))
        held = long_data.item_index(999)
        assert not np.any(batch.input_items == held)
        assert not np.any(batch.target_items == held)


class TestWindowsDoNotLeak:
    """The TestNoLeakage assertions, repeated over the windowed training rows."""

    @pytest.fixture(scope="class")
    def windowed_pairs(self, seq_data):
        batch = seq_data.training_batch(seq_data.training_window_index(stride=3))
        pairs = set()
        for row, uid in enumerate(batch.user_ids.tolist()):
            for items in (batch.input_items[row], batch.target_items[row]):
                for idx in items[items != PAD_INDEX].tolist():
                    pairs.add((uid, int(seq_data.index_to_item[idx])))
        return pairs

    def test_env_fixture_has_users_longer_than_maxlen(self, seq_data):
        """Otherwise the leakage checks below would not exercise windowing."""
        index = seq_data.training_window_index()
        assert len(index) > seq_data.num_users

    def test_no_held_out_pair_appears_in_any_window(self, env, windowed_pairs):
        held_out = {(int(r.userId), int(r.movieId)) for r in env.held_out.itertuples()}
        assert not (windowed_pairs & held_out)

    def test_no_validation_pair_appears_in_any_window(self, env, windowed_pairs):
        validation = {
            (int(r.userId), int(r.movieId)) for r in env.validation.itertuples()
        }
        assert not (windowed_pairs & validation)

    def test_disjoint_windows_target_every_non_first_training_row_once(self, env, seq_data):
        batch = seq_data.training_batch(seq_data.training_window_index())
        n_targets = int(np.count_nonzero(batch.target_items != PAD_INDEX))
        assert n_targets == len(env.train_ratings) - seq_data.num_users


class TestInferenceIgnoresWindows:
    """
    Windowing is training-only. Inference must see the most recent maxlen
    interactions, whatever the stride.
    """

    def test_inference_uses_only_the_most_recent_maxlen_items(self, long_user_env):
        for stride in (None, 1, 3, WINDOW_MAXLEN):
            data = build_sasrec_sequences(
                long_user_env, maxlen=WINDOW_MAXLEN, window_stride=stride
            )
            seq = data.user_sequences[1]
            items, residuals = data.padded_sequence(1)
            assert np.array_equal(items, seq[-WINDOW_MAXLEN:])
            assert np.array_equal(residuals, data.user_residuals[1][-WINDOW_MAXLEN:])

    def test_inference_matrix_has_one_row_per_user_not_per_window(self, long_data):
        items, _, order = long_data.padded_matrix()
        assert order == [1, 2, 3, 4]
        assert items.shape == (4, WINDOW_MAXLEN)
        assert len(long_data.training_window_index()) != items.shape[0]

    def test_inference_context_ends_one_step_after_the_newest_training_input(
        self, long_data
    ):
        """
        The newest training window predicts the last training item; inference
        feeds that item in and predicts what comes after it.
        """
        newest = long_data.training_batch(np.array([[1, LONG_LEN]]))
        items, _ = long_data.padded_sequence(1)
        assert np.array_equal(items[:-1], newest.input_items[0, 1:])
        assert items[-1] == newest.target_items[0, -1]


class TestWindowStrideConfig:
    def test_none_means_maxlen(self):
        assert resolve_window_stride(None, 200) == 200

    @pytest.mark.parametrize("stride", [0, -1, 201])
    def test_out_of_range_strides_are_rejected(self, stride):
        with pytest.raises(ValueError, match="window stride"):
            resolve_window_stride(stride, 200)

    def test_bad_stride_fails_at_build_time(self, long_user_env):
        with pytest.raises(ValueError, match="window stride"):
            build_sasrec_sequences(long_user_env, maxlen=WINDOW_MAXLEN, window_stride=0)

    def test_simconfig_defaults_to_disjoint_windows(self):
        config = SimConfig()
        assert config.sasrec_window_stride is None
        assert config.as_dict()["sasrec_window_stride"] == config.sasrec_maxlen

    def test_simconfig_logs_an_explicit_stride(self):
        config = SimConfig(sasrec_maxlen=50, sasrec_window_stride=10)
        assert config.as_dict()["sasrec_window_stride"] == 10

    def test_simconfig_stride_round_trips_through_json(self):
        config = SimConfig(sasrec_window_stride=25)
        assert SimConfig.from_dict(config.to_json_dict()).sasrec_window_stride == 25
        assert SimConfig.from_dict(SimConfig().to_json_dict()).sasrec_window_stride is None
