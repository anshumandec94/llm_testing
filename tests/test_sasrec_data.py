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
)


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
