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
enter the fit.

Unobserved cells are implicit zeros in the SVD. That makes it a SHRUNK
estimator: the matrix is about 0.2% observed, so the rank-k fit spends its
capacity reconstructing zeros and pulls every prediction toward 0. The SVD is
the issue #27 deliverable because it is the minimal change to the published
arm (same estimator family, fixed target and units). `fit_residual_als` is a
secondary diagnostic that fits the same residuals on OBSERVED cells only, so
the report can separate what the representation carries from what the
zero-imputing estimator throws away. It is untuned (one configuration) and is
not a replacement baseline; choosing that is issue #13.

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

# The one ALS configuration used by the secondary diagnostic. Untuned: chosen
# before looking at any held-out number, and not to be tuned against one.
ALS_REGULARIZATION = 5.0
ALS_ITERATIONS = 10

@dataclass
class ResidualFactors:
    """A fitted rank-k model of debiased residuals, in rating units.

    For the SVD, `user_factors` rows are `fit_transform` output, so they
    already carry the singular values, and `item_factors` rows are columns of
    `components_`, which are unit-norm right singular vectors. Their dot
    product is the rank-k reconstruction of the residual; scaling either side
    again would double count the singular values. For the ALS the two sides
    are ordinary regularised factors and `singular_values` is None.
    """

    user_ids: np.ndarray
    item_ids: np.ndarray
    user_factors: np.ndarray  # (n_users, k), U * Sigma
    item_factors: np.ndarray  # (n_items, k), V
    singular_values: np.ndarray | None
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

    def _rows(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = np.searchsorted(self.user_ids, user_ids).clip(0, len(self.user_ids) - 1)
        i = np.searchsorted(self.item_ids, movie_ids).clip(0, len(self.item_ids) - 1)
        ok = (self.user_ids[u] == user_ids) & (self.item_ids[i] == movie_ids)
        return u, i, ok

    def covered(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """Whether each pair has both a user and an item factor row."""
        return self._rows(np.asarray(user_ids), np.asarray(movie_ids))[2]

    def residuals(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """Vectorised `residual` over paired id arrays, zero where uncovered."""
        u, i, ok = self._rows(np.asarray(user_ids), np.asarray(movie_ids))
        terms = np.einsum("ij,ij->i", self.user_factors[u], self.item_factors[i])
        return np.where(ok, terms, 0.0)

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
    can be tested on a matrix whose true factors are known. ARPACK makes the
    decomposition exact rather than randomised, so the arm does not depend on
    the seed beyond the signs of the vectors, which cancel in the product.
    ARPACK needs `n_components < min(matrix.shape)`; asking for more raises
    rather than silently fitting fewer dimensions.
    """
    user_ids, item_ids, matrix = _residual_matrix(frame, value_col)
    _check_components(n_components, matrix.shape, strict=True)
    svd = TruncatedSVD(n_components=n_components, algorithm="arpack", random_state=random_state)
    user_factors = svd.fit_transform(matrix)
    return ResidualFactors(
        user_ids=user_ids,
        item_ids=item_ids,
        user_factors=user_factors,
        item_factors=svd.components_.T.copy(),
        singular_values=svd.singular_values_.copy(),
    )


def fit_residual_als(
    frame: pd.DataFrame,
    n_components: int,
    random_state: int,
    regularization: float = ALS_REGULARIZATION,
    iterations: int = ALS_ITERATIONS,
    value_col: str = "residual",
) -> ResidualFactors:
    """Secondary diagnostic: alternating least squares on OBSERVED cells only.

    Minimises the squared error over observed residuals plus
    `regularization * (|p_u|^2 + |q_i|^2)`, with each user and item row
    solved in closed form per sweep. Unlike the SVD it never sees the
    unobserved cells, so it is not shrunk toward zero by them. The defaults
    are one untuned configuration; nothing here was selected on held-out
    data.

    Each half-step is vectorised: the per-row Gram matrices are
    `pattern @ (X kron X)` and the right-hand sides `values @ X`, both
    sparse-dense products, followed by one batched solve per row chunk.
    """
    user_ids, item_ids, matrix = _residual_matrix(frame, value_col)
    _check_components(n_components, matrix.shape, strict=False)
    rng = np.random.default_rng(random_state)
    users = rng.normal(scale=0.1, size=(matrix.shape[0], n_components))
    items = rng.normal(scale=0.1, size=(matrix.shape[1], n_components))
    by_item = matrix.T.tocsr()
    for _ in range(iterations):
        users = _als_half_step(matrix, items, regularization)
        items = _als_half_step(by_item, users, regularization)
    return ResidualFactors(
        user_ids=user_ids,
        item_ids=item_ids,
        user_factors=users,
        item_factors=items,
        singular_values=None,
    )


def _als_half_step(ratings: csr_matrix, fixed: np.ndarray, regularization: float) -> np.ndarray:
    """Solve every row of `ratings` against the fixed other-side factors."""
    k = fixed.shape[1]
    pattern = ratings.copy()
    pattern.data = np.ones_like(pattern.data)
    outer = (fixed[:, :, None] * fixed[:, None, :]).reshape(len(fixed), k * k)
    ridge = regularization * np.eye(k)
    out = np.empty((ratings.shape[0], k))
    # Chunk rows so the (rows, k, k) Gram stack stays around 100 MB.
    chunk = max(1, int(1.2e7 // (k * k)))
    for start in range(0, ratings.shape[0], chunk):
        stop = min(start + chunk, ratings.shape[0])
        gram = (pattern[start:stop] @ outer).reshape(-1, k, k) + ridge
        rhs = ratings[start:stop] @ fixed
        out[start:stop] = np.linalg.solve(gram, rhs[:, :, None])[:, :, 0]
    return out


def _residual_matrix(frame: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, csr_matrix]:
    user_ids = np.sort(frame["userId"].unique())
    item_ids = np.sort(frame["movieId"].unique())
    rows = np.searchsorted(user_ids, frame["userId"].to_numpy())
    cols = np.searchsorted(item_ids, frame["movieId"].to_numpy())
    matrix = csr_matrix(
        (frame[value_col].to_numpy(dtype=np.float64), (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
    )
    return user_ids, item_ids, matrix


def _check_components(n_components: int, shape: tuple[int, int], strict: bool) -> None:
    if n_components < 1:
        raise ValueError(f"n_components must be at least 1, got {n_components}")
    limit = min(shape) - 1 if strict else min(shape)
    if n_components > limit:
        raise ValueError(
            f"n_components={n_components} is too large for a {shape[0]}x{shape[1]} "
            f"residual matrix (at most {limit}); refusing to fit fewer dimensions silently"
        )


def fit_env_residual_factors(env: Environment, n_components: int | None = None) -> ResidualFactors:
    """Fit residual factors on an Environment's training split.

    `n_components` defaults to `config.user_pref_features`, the capacity of
    the published associative arm, so the two differ only in target and
    scale, not in dimension.
    """
    k = env.config.user_pref_features if n_components is None else n_components
    return fit_residual_factors(training_residuals(env), k, env.config.random_seed)


def fit_env_residual_als(env: Environment, n_components: int | None = None) -> ResidualFactors:
    """Secondary diagnostic ALS on an Environment's training residuals.

    Same inputs, dimension default and seed as `fit_env_residual_factors`, so
    the two differ only in how unobserved cells are treated.
    """
    k = env.config.user_pref_features if n_components is None else n_components
    return fit_residual_als(training_residuals(env), k, env.config.random_seed)
