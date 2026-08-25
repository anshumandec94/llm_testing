"""
sim.recommender — LensKit 2026 recommendation pipeline wrapper.

The recommender maintains its own view of each user (trained only on
training-set ratings, then updated with simulation feedback). This mirrors
the real-world scenario where the platform's model is an imperfect approximation
of user preference, separate from the agent's internal model.

Seen-item exclusion
-------------------
Items are excluded from future recommendation requests at two scopes:

* ``_all_seen[user]``   — items rated in training OR accepted/sent in a
  previous round. Persists for the entire experiment.
* ``_round_seen[user]`` — items already sent to this user in the *current*
  round (cleared when ``advance_round()`` is called). This supports re-
  requests within a single round: if the agent rejects the first batch,
  the next batch will not repeat those items.

Re-request within a round
--------------------------
Call ``recommend()`` as many times as needed within a round. Each call
automatically excludes items already sent in that round (via ``_round_seen``)
as well as all previously seen items (via ``_all_seen``).
Call ``mark_sent(user_id, items)`` immediately after sending a batch so
those items are recorded before the next request.
Call ``advance_round()`` at the end of every round to flush ``_round_seen``
into ``_all_seen`` and prepare for the next round.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from lenskit.als import BiasedMFScorer
from lenskit.batch import recommend as lk_recommend
from lenskit.data import ItemList, from_interactions_df
from lenskit.pipeline import topn_pipeline

from sim.config import SimConfig
from sim.environment import Environment

logger = logging.getLogger(__name__)

# How many extra candidates to fetch from LensKit so that after filtering
# seen items we still have enough items to serve.
_OVERSAMPLE = 5


class Recommender:
    """
    Wraps a LensKit BiasedMF top-N pipeline.

    After initial training, feedback collected during simulation rounds is
    accumulated and the pipeline is retrained once per round (at the start of
    each round via ``retrain()``) so the recommender's representation of users
    evolves over time, independently of any agent's internal model.

    Parameters
    ----------
    config:
        Experiment configuration.
    env:
        Initialised Environment (provides training ratings and movie metadata).
        The Recommender builds and owns its own LensKit ``Dataset``.
    """

    def __init__(
        self,
        config: SimConfig,
        env: Environment,
        user_base_map: dict[int, int] | None = None,
    ) -> None:
        self.config = config
        self.env = env
        self._user_base_map = user_base_map or {}
        self._expanded_train_ratings = self._build_expanded_train_ratings()

        # Accumulated feedback: userId → list of (movieId, rating) tuples
        self._feedback: dict[int, list[tuple[int, float]]] = defaultdict(list)

        # Items the user has already seen across all completed rounds.
        # Seeded from training ratings so the recommender never re-surfaces
        # items the user rated before the simulation started.
        self._all_seen: dict[int, set[int]] = defaultdict(set)
        self._seed_seen_from_training()

        # Items sent to the user within the current round only.
        # Cleared by advance_round().
        self._round_seen: dict[int, set[int]] = defaultdict(set)

        # Build and train the initial pipeline
        logger.info(
            "Training initial LensKit BiasedMF pipeline (features=%d, epochs=%d, regularization=%.4f) …",
            config.mf_features,
            config.mf_epochs,
            config.mf_regularization,
        )
        self._build_and_train(self._expanded_train_ratings)

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_expanded_train_ratings(self) -> pd.DataFrame:
        """
        Return train ratings with replicated simulated users cloned from base users.

        The result is only ever read (grouped, concatenated, trained on), never
        mutated in place, so when no replication is required this returns the
        Environment's training ratings directly instead of duplicating them.
        That duplicate costs ~0.9 GB on ML-32M.
        """
        train = self.env.train_ratings[["userId", "movieId", "rating", "timestamp"]]
        extra_rows: list[pd.DataFrame] = []
        for sim_user_id, base_user_id in self._user_base_map.items():
            if sim_user_id == base_user_id:
                continue
            base_rows = train[train["userId"] == base_user_id]
            if base_rows.empty:
                continue
            extra_rows.append(base_rows.assign(userId=sim_user_id))

        if extra_rows:
            train = pd.concat([train, *extra_rows], ignore_index=True)
        return train

    def _seed_seen_from_training(self) -> None:
        """Pre-populate _all_seen with each user's training-set rated items."""
        for uid, group in self._expanded_train_ratings.groupby("userId"):
            self._all_seen[int(uid)].update(group["movieId"].tolist())  # ty:ignore[invalid-argument-type]

    def _build_and_train(self, ratings_df: pd.DataFrame) -> None:
        dataset = from_interactions_df(
            ratings_df,
            user_col="userId",
            item_col="movieId",
            rating_col="rating",
            timestamp_col="timestamp",
        )
        scorer = BiasedMFScorer(**self.config.platform_mf_kwargs())
        self._pipeline = topn_pipeline(scorer)
        self._pipeline.train(dataset)

    def _filter_seen(self, user_id: int, item_list: ItemList) -> ItemList:
        """
        Remove items in ``_all_seen`` or ``_round_seen`` from ``item_list``.

        Returns a new ItemList preserving scores for surviving items.
        """
        exclude = self._all_seen[user_id] | self._round_seen[user_id]
        if not exclude:
            return item_list

        ids = np.array(item_list.ids())
        scores = item_list.scores()

        mask = np.array([int(iid) not in exclude for iid in ids], dtype=bool)
        filtered_ids = ids[mask]
        filtered_scores = scores[mask] if scores is not None else None

        return ItemList(item_ids=filtered_ids, scores=filtered_scores)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def recommend(self, user_id: int, n: int | None = None) -> ItemList:
        """
        Generate top-N recommendations for a user, excluding all seen items.

        LensKit's pipeline candidate selector already excludes items rated in
        the training dataset. On top of that, this method filters out items in
        ``_all_seen`` and ``_round_seen`` (items from previous rounds and from
        earlier requests within the current round, respectively).

        An oversampling factor of ``_OVERSAMPLE × n`` candidates is requested
        from LensKit so that enough items remain after filtering.

        Callers must call ``mark_sent()`` after delivering a batch so that
        those items are excluded from subsequent re-requests this round.

        Parameters
        ----------
        user_id:
            The user to generate recommendations for.
        n:
            Number of recommendations to return. Defaults to
            ``config.rec_list_size``.

        Returns
        -------
        ItemList
            At most *n* items, ranked by the LensKit pipeline scorer, with
            previously seen items removed.
        """
        if n is None:
            n = self.config.rec_list_size

        # Request an oversized pool so filtering doesn't leave us short
        pool_size = n * _OVERSAMPLE
        recs = lk_recommend(self._pipeline, [user_id], pool_size)

        # Use .lookup() for key-based access; __getitem__ is positional only.
        item_list = recs.lookup(user_id=user_id)
        if item_list is None:
            logger.warning("No recommendations generated for user %d.", user_id)
            return ItemList(item_ids=np.array([], dtype=np.int64))

        filtered = self._filter_seen(user_id, item_list)
        return filtered.top_n(n)

    def mark_sent(self, user_id: int, items: ItemList) -> None:
        """
        Record items as sent to ``user_id`` in the current round.

        Must be called immediately after delivering a batch so that the items
        are excluded from any re-request within the same round.
        """
        self._round_seen[user_id].update(int(iid) for iid in items.ids())

    def update_user(self, user_id: int, interactions: list[tuple[int, float]]) -> None:
        """
        Record explicit user ratings as feedback for the recommender.

        Parameters
        ----------
        user_id:
            The user who interacted with items.
        interactions:
            List of ``(movie_id, raw_rating)`` tuples from explicit ``rate``
            actions only. Ratings stay on the original scale because
            ``BiasedMF`` learns the bias decomposition internally. An empty
            list is a no-op.

        Items passed here are also added to ``_all_seen`` so they will not
        be recommended again in future rounds.
        """
        for mid, signal in interactions:
            mid = int(mid)
            self._feedback[user_id].append((mid, signal))
            self._all_seen[user_id].add(mid)

    def advance_round(self) -> None:
        """
        Finalise the current round.

        Moves all items in ``_round_seen`` into ``_all_seen`` so that items
        shown but not accepted are still excluded from future rounds, then
        clears ``_round_seen`` for the next round.
        """
        for uid, seen in self._round_seen.items():
            self._all_seen[uid].update(seen)
        self._round_seen.clear()

    def retrain(self) -> None:
        """
        Retrain the pipeline on training ratings + all accumulated feedback.

        Call once at the start of each simulation round (after round 1) so
        the recommender's model reflects the feedback received so far.
        """
        if not self._feedback:
            return

        feedback_rows = [
            {"userId": uid, "movieId": mid, "rating": rating, "timestamp": 0}
            for uid, interactions in self._feedback.items()
            for mid, rating in interactions
        ]
        feedback_df = pd.DataFrame(feedback_rows)

        combined = pd.concat(
            [
                self._expanded_train_ratings,
                feedback_df,
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["userId", "movieId"], keep="last")

        logger.info(
            "Retraining recommender on %d ratings (%d feedback events) …",
            len(combined),
            len(feedback_rows),
        )
        self._build_and_train(combined)
