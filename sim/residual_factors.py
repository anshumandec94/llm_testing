"""
sim.residual_factors - a rating-unit latent factor model for the associative
rating baseline (issue #27).

The persona preference space (`Environment._setup_user_pref_embeddings`) is a
TruncatedSVD of RAW training ratings with both sides L2-normalised, so its
dot product is a cosine in [-1, 1], not a rating residual. That geometry is
what the simulation's personas drift in, so it is left alone. Adding that
cosine to the bias baseline to predict a rating mixes units, which is why the
published associative arm lost to a bias-only null.

This module fits a separate, un-normalised TruncatedSVD on the DEBIASED
training residuals, `rating - env.get_rating_bias(user, item)`, so that

    prediction = bias + U[user] @ components_[:, item]

is a rank-k reconstruction of the residual in stars. It is a baseline for
rating prediction only and never touches the Environment's persona space or
its ChromaDB caches.

Leakage: the matrix is built from `env.train_ratings` alone. An evaluation
user's factor row therefore comes only from their own training ratings,
exactly as their bias term does, and held-out or validation ratings never
enter the fit. Unobserved cells are implicit zeros, which on debiased
residuals means "no deviation from the bias baseline", the natural prior.

Nothing is cached on disk: the fit takes seconds and has no downstream
consumers that need it to persist, so a cache would only add a key to keep
consistent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from sim.environment import Environment


@dataclass
class ResidualFactors:
    """A fitted rank-k model of debiased residuals, in rating units.

    `user_factors` rows are `fit_transform` output, so they already carry the
    singular values. `item_factors` rows are columns of `components_`, which
    are unit-norm right singular vectors. Their dot product is the rank-k
    reconstruction of the residual; scaling either side again would double
    count the singular values.
    """

    user_ids: np.ndarray
    item_ids: np.ndarray
    user_factors: np.ndarray  # (n_users, k), U * Sigma
    item_factors: np.ndarray  # (n_items, k), V
    singular_values: np.ndarray
    _user_row: dict[int, int] = field(init=False, repr=False)
    _item_row: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._user_row = {int(u): i for i, u in enumerate(self.user_ids)}
        self._item_row = {int(m): i for i, m in enumerate(self.item_ids)}

    @property
    def n_components(self) -> int:
        return int(self.user_factors.shape[1])

    def has_user(self, user_id: int) -> bool:
        return int(user_id) in self._user_row

    def has_item(self, movie_id: int) -> bool:
        return int(movie_id) in self._item_row

    def residual(self, user_id: int, movie_id: int) -> float:
        """Predicted debiased residual for one pair, in stars.

        Zero when either side was never seen in training, which reduces the
        prediction to the bias baseline rather than inventing a signal.
        """
        u = self._user_row.get(int(user_id))
        i = self._item_row.get(int(movie_id))
        if u is None or i is None:
            return 0.0
        return float(self.user_factors[u] @ self.item_factors[i])


def training_residuals(env: Environment) -> pd.DataFrame:
    """Debiased training residuals, one row per training rating.

    Uses the Environment's own train-set bias model, so the residual is
    exactly what `env.debias_rating` would return, vectorised.
    """
    train = env.train_ratings[["userId", "movieId", "rating"]]
    user_bias = train["userId"].map(env._rating_user_biases).fillna(0.0)
    item_bias = train["movieId"].map(env._rating_item_biases).fillna(0.0)
    bias = env._rating_global_bias + user_bias + item_bias
    return pd.DataFrame({
        "userId": train["userId"].to_numpy(),
        "movieId": train["movieId"].to_numpy(),
        "residual": (train["rating"] - bias).to_numpy(dtype=np.float64),
    })


def fit_residual_factors(
    frame: pd.DataFrame,
    n_components: int,
    random_state: int,
    value_col: str = "residual",
) -> ResidualFactors:
    """Fit an un-normalised TruncatedSVD to a (userId, movieId, value) frame.

    Kept separate from `training_residuals` so the reconstruction arithmetic
    can be tested on a matrix whose true factors are known.
    """
    user_ids = np.sort(frame["userId"].unique())
    item_ids = np.sort(frame["movieId"].unique())
    rows = np.searchsorted(user_ids, frame["userId"].to_numpy())
    cols = np.searchsorted(item_ids, frame["movieId"].to_numpy())
    matrix = csr_matrix(
        (frame[value_col].to_numpy(dtype=np.float64), (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
    )
    k = min(n_components, min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=k, random_state=random_state)
    user_factors = svd.fit_transform(matrix)
    return ResidualFactors(
        user_ids=user_ids,
        item_ids=item_ids,
        user_factors=user_factors,
        item_factors=svd.components_.T.copy(),
        singular_values=svd.singular_values_.copy(),
    )


def fit_env_residual_factors(env: Environment, n_components: int | None = None) -> ResidualFactors:
    """Fit residual factors on an Environment's training split.

    `n_components` defaults to `config.user_pref_features`, the capacity of
    the published associative arm, so the two differ only in target and
    scale, not in dimension.
    """
    k = env.config.user_pref_features if n_components is None else n_components
    return fit_residual_factors(training_residuals(env), k, env.config.random_seed)
