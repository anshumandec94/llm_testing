from __future__ import annotations

import numpy as np
import pandas as pd

def build_residual_history(
    env,
    *,
    user_ids: list[int] | None = None,
) -> tuple[dict[int, dict[int, float]], dict[int, np.ndarray]]:
    """Build signed residual history per user in the user-preference item space."""
    train = env.train_ratings
    if user_ids is not None:
        train = train[train["userId"].isin(user_ids)]

    movie_ids = sorted({int(mid) for mid in train["movieId"].tolist()})
    item_vectors = env.get_user_pref_item_factors(movie_ids)

    history: dict[int, dict[int, float]] = {}
    for row in train.itertuples(index=False):
        uid = int(row.userId)
        mid = int(row.movieId)
        if mid not in item_vectors:
            continue
        residual = env.debias_rating(uid, mid, float(row.rating))
        if abs(residual) < 1e-9:
            continue
        user_history = history.setdefault(uid, {})
        user_history[mid] = user_history.get(mid, 0.0) + residual

    return history, item_vectors


def update_residual_history(
    history: dict[int, dict[int, float]],
    user_id: int,
    interactions: list[tuple[int, str, float]],
) -> dict[int, float]:
    """Apply debiased explicit-rating residuals to a user's signed history."""
    user_history = history.setdefault(user_id, {})
    for movie_id, _action, signal in interactions:
        if abs(signal) < 1e-9:
            continue
        movie_id = int(movie_id)
        user_history[movie_id] = user_history.get(movie_id, 0.0) + float(signal)
    return user_history


def ensure_item_vectors(
    env,
    cache: dict[int, np.ndarray],
    movie_ids: list[int] | pd.Index | np.ndarray,
) -> dict[int, np.ndarray]:
    """Populate a vector cache for any unseen movie IDs."""
    missing = sorted({int(mid) for mid in movie_ids if int(mid) not in cache})
    if missing:
        cache.update(env.get_user_pref_item_factors(missing))
    return cache
