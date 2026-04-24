"""
sim.environment — data loading, hold-out splitting, and embedding management.

Responsibilities
----------------
1. Load ML-32M CSVs (ratings, movies, movie_overviews).
2. Perform a hold-out split: for a configurable fraction of users, reserve the
   most-recent N% of their ratings as ground-truth evaluation data.
3. Build a LensKit Dataset from the training (non-held-out) ratings.
4. Generate and persist two ChromaDB embedding collections:
   - ``associative_item_factors`` – item latent vectors from BiasedMF (ALS).
   - ``semantic_movie_embeddings`` – sentence-transformer encodings of
     title + genres + overview.
5. Expose helpers for other components to query embeddings.
"""
from __future__ import annotations

import logging
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
from lenskit.als import BiasedMFScorer
from lenskit.data import from_interactions_df
from lenskit.pipeline import topn_pipeline
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from sim.config import SimConfig

logger = logging.getLogger(__name__)

# ChromaDB collection names
COLLECTION_ASSOC = "associative_item_factors"
COLLECTION_SEMANTIC = "semantic_movie_embeddings"
COLLECTION_USER_PREF = "user_pref_item_factors"


class Environment:
    """
    Central data store for one simulation experiment.

    Parameters
    ----------
    config:
        Experiment configuration.
    """

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

        # ── Load raw data ──────────────────────────────────────────────────
        logger.info("Loading ML-32M data from %s …", config.data_dir)
        self._load_data()

        # ── Hold-out split ─────────────────────────────────────────────────
        logger.info("Performing hold-out split …")
        self._make_holdout_split()

        # ── Build LensKit Dataset from training ratings ─────────────────────
        logger.info("Building LensKit Dataset …")
        self.dataset = from_interactions_df(
            self.train_ratings[["userId", "movieId", "rating", "timestamp"]].copy(),
            user_col="userId",
            item_col="movieId",
            rating_col="rating",
            timestamp_col="timestamp",
        )

        # ── ChromaDB client ────────────────────────────────────────────────
        db_path = Path(config.embeddings_dir)
        db_path.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(db_path))

        # ── Build / load embedding collections ────────────────────────────
        self._setup_associative_embeddings()
        self._setup_semantic_embeddings()
        self._setup_user_pref_embeddings()

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        data_dir = Path(self.config.data_dir)

        ratings = pd.read_csv(data_dir / "ratings.csv")
        movies = pd.read_csv(data_dir / "movies.csv")

        overview_path = data_dir / "movie_overviews.csv"
        if overview_path.exists():
            overviews = pd.read_csv(overview_path)
            movies = movies.merge(overviews, on="movieId", how="left")
            movies["overview"] = movies["overview"].fillna("")
        else:
            logger.warning(
                "movie_overviews.csv not found; overview field will be empty."
            )
            movies["overview"] = ""

        self.all_ratings: pd.DataFrame = ratings
        self.movie_meta: pd.DataFrame = movies  # movieId, title, genres, overview

    def _make_holdout_split(self) -> None:
        cfg = self.config
        ratings = self.all_ratings

        # ── Select eligible users ──────────────────────────────────────────
        user_counts = ratings.groupby("userId").size()
        eligible = user_counts[user_counts >= cfg.min_ratings].index.to_numpy()

        rng = self.rng
        n_eval = max(1, int(len(eligible) * cfg.eval_user_frac))
        self.eval_users: list[int] = rng.choice(
            eligible, size=n_eval, replace=False
        ).tolist()

        logger.info(
            "Eval users: %d / %d eligible (>= %d ratings)",
            n_eval,
            len(eligible),
            cfg.min_ratings,
        )

        # ── Per-user hold-out: keep most-recent holdout_frac ratings ──────
        held_out_rows: list[pd.DataFrame] = []
        train_rows: list[pd.DataFrame] = []

        for uid in self.eval_users:
            user_df = ratings[ratings["userId"] == uid].sort_values(
                "timestamp", ascending=False
            )
            n_hold = max(1, int(len(user_df) * cfg.holdout_frac))
            held_out_rows.append(user_df.iloc[:n_hold])
            train_rows.append(user_df.iloc[n_hold:])

        # Non-eval users: all ratings go to training
        non_eval_mask = ~ratings["userId"].isin(self.eval_users)
        train_rows.append(ratings[non_eval_mask])

        self.held_out: pd.DataFrame = pd.concat(held_out_rows, ignore_index=True)
        self.train_ratings: pd.DataFrame = pd.concat(train_rows, ignore_index=True)

        logger.info(
            "Split: %d train ratings, %d held-out ratings",
            len(self.train_ratings),
            len(self.held_out),
        )

    def _collection_exists_with_data(self, name: str) -> bool:
        try:
            col = self.chroma_client.get_collection(name)
            return col.count() > 0
        except Exception:
            return False

    def _setup_associative_embeddings(self) -> None:
        """
        Train a BiasedMF model on training data and store item-factor vectors
        in ChromaDB. User vectors are stored separately for agent use.

        If the collection already exists and force_rebuild is False, loading
        is skipped and the existing collection is used.
        """
        force = self.config.force_rebuild_embeddings

        if not force and self._collection_exists_with_data(COLLECTION_ASSOC):
            logger.info(
                "ChromaDB collection '%s' already exists – skipping rebuild.",
                COLLECTION_ASSOC,
            )
            self._assoc_collection = self.chroma_client.get_collection(COLLECTION_ASSOC)
            # Load the stored user factors from metadata
            self._load_user_factors_from_chroma()
            return

        logger.info(
            "Training BiasedMF (features=%d) for associative embeddings …",
            self.config.mf_features,
        )
        scorer = BiasedMFScorer(features=self.config.mf_features)
        pipe = topn_pipeline(scorer)
        pipe.train(self.dataset)

        # Extract item and user factor matrices.
        # In LensKit 2026 the scorer carries its own item/user Vocabulary
        # attributes and the embeddings as plain numpy matrices.
        item_vectors: np.ndarray = scorer.item_embeddings  # (n_items, features)
        user_vectors: np.ndarray | None = scorer.user_embeddings  # (n_users, features)

        item_vocab = scorer.items   # Vocabulary — attribute, not a method
        user_vocab = scorer.users   # Vocabulary | None — attribute, not a method

        # Store item factors in ChromaDB
        if force:
            try:
                self.chroma_client.delete_collection(COLLECTION_ASSOC)
            except Exception:
                pass

        col = self.chroma_client.get_or_create_collection(
            COLLECTION_ASSOC,
            metadata={"description": "BiasedMF item latent factors"},
        )

        ids = [str(iid) for iid in item_vocab.ids()]
        embeddings = item_vectors.tolist()
        col.upsert(ids=ids, embeddings=embeddings)
        self._assoc_collection = col

        # Store user factors as a simple numpy dict (not in ChromaDB — too
        # many vectors to be useful for retrieval there)
        if user_vocab is not None and user_vectors is not None:
            self._user_factors: dict[int, np.ndarray] = {
                int(uid): user_vectors[i] for i, uid in enumerate(user_vocab.ids())
            }
        else:
            logger.warning("BiasedMF did not produce user embeddings; user factors unavailable.")
            self._user_factors = {}

        # Also persist user factors as a npz for reuse
        if self._user_factors:
            factors_path = Path(self.config.embeddings_dir) / "user_factors.npz"
            np.savez(
                factors_path,
                user_ids=np.array(list(self._user_factors.keys())),
                vectors=np.array(list(self._user_factors.values())),
            )
        logger.info("Associative embeddings ready (%d items).", len(ids))

    def _load_user_factors_from_chroma(self) -> None:
        """Load user factors from the persisted npz file if available."""
        factors_path = Path(self.config.embeddings_dir) / "user_factors.npz"
        if factors_path.exists():
            data = np.load(factors_path)
            self._user_factors = {
                int(uid): vec
                for uid, vec in zip(data["user_ids"], data["vectors"])
            }
            logger.info(
                "Loaded %d user factor vectors from disk.", len(self._user_factors)
            )
        else:
            logger.warning(
                "user_factors.npz not found. AssociativeAgent will not work. "
                "Re-run with force_rebuild_embeddings=True."
            )
            self._user_factors = {}

    def _setup_semantic_embeddings(self) -> None:
        """
        Encode each movie's text description with a sentence-transformer and
        store in ChromaDB. Skipped if the collection already exists.
        """
        force = self.config.force_rebuild_embeddings

        if not force and self._collection_exists_with_data(COLLECTION_SEMANTIC):
            logger.info(
                "ChromaDB collection '%s' already exists – skipping rebuild.",
                COLLECTION_SEMANTIC,
            )
            self._semantic_collection = self.chroma_client.get_collection(
                COLLECTION_SEMANTIC
            )
            return

        logger.info(
            "Encoding movie descriptions with '%s' …", self.config.semantic_model
        )
        model = SentenceTransformer(self.config.semantic_model)

        meta = self.movie_meta.copy()
        # Build a single text description per movie
        meta["text"] = (
            meta["title"].fillna("") + ". "
            + meta["genres"].str.replace("|", ", ", regex=False).fillna("") + ". "
            + meta["overview"].fillna("")
        )

        texts = meta["text"].tolist()
        movie_ids = meta["movieId"].astype(str).tolist()

        vectors = model.encode(texts, show_progress_bar=True, batch_size=256)

        if force:
            try:
                self.chroma_client.delete_collection(COLLECTION_SEMANTIC)
            except Exception:
                pass

        col = self.chroma_client.get_or_create_collection(
            COLLECTION_SEMANTIC,
            metadata={"description": "Sentence-transformer movie content embeddings"},
        )
        col.upsert(ids=movie_ids, embeddings=vectors.tolist())
        self._semantic_collection = col
        logger.info("Semantic embeddings ready (%d movies).", len(movie_ids))

    def _setup_user_pref_embeddings(self) -> None:
        """
        Train a small TruncatedSVD on the training rating matrix to produce
        low-dimensional item representations for the user preference model.

        The item side (V matrix) is stored in ChromaDB collection
        ``user_pref_item_factors``. The user side (U matrix) is stored in
        ``user_pref_factors.npz`` and used to initialise each AgentPersona's
        preference vector.

        Dimensionality is set by ``config.user_pref_features`` (default 8).
        Both item and user vectors are L2-normalised before storage.
        """
        force = self.config.force_rebuild_embeddings
        npz_path = Path(self.config.embeddings_dir) / "user_pref_factors.npz"

        if (
            not force
            and self._collection_exists_with_data(COLLECTION_USER_PREF)
            and npz_path.exists()
        ):
            logger.info(
                "ChromaDB collection '%s' already exists – skipping rebuild.",
                COLLECTION_USER_PREF,
            )
            self._user_pref_collection = self.chroma_client.get_collection(
                COLLECTION_USER_PREF
            )
            self._load_user_pref_factors_from_disk(npz_path)
            return

        logger.info(
            "Building user-preference TruncatedSVD (features=%d) …",
            self.config.user_pref_features,
        )

        # Build a sparse rating matrix: rows=users, cols=items
        # Only use training ratings.
        train = self.train_ratings[["userId", "movieId", "rating"]].copy()

        # Integer-encode users and items for the matrix
        user_ids_unique = np.sort(train["userId"].unique())
        item_ids_unique = np.sort(train["movieId"].unique())

        user_idx = {uid: i for i, uid in enumerate(user_ids_unique)}
        item_idx = {mid: i for i, mid in enumerate(item_ids_unique)}

        from scipy.sparse import csr_matrix

        rows = train["userId"].map(user_idx).to_numpy()
        cols = train["movieId"].map(item_idx).to_numpy()
        data = train["rating"].to_numpy(dtype=np.float32)

        R = csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_ids_unique), len(item_ids_unique)),
        )

        n_components = min(
            self.config.user_pref_features,
            min(R.shape) - 1,
        )

        svd = TruncatedSVD(
            n_components=n_components,
            random_state=self.config.random_seed,
        )
        # U (users × k)
        U = svd.fit_transform(R)
        # V (items × k) — right singular vectors scaled by singular values
        V = svd.components_.T  # shape: (items, k)

        # L2-normalise both sides
        U_norm = normalize(U, norm="l2")
        V_norm = normalize(V, norm="l2")

        # Store item factors in ChromaDB
        if force:
            try:
                self.chroma_client.delete_collection(COLLECTION_USER_PREF)
            except Exception:
                pass

        col = self.chroma_client.get_or_create_collection(
            COLLECTION_USER_PREF,
            metadata={"description": "TruncatedSVD item factors (user pref space)"},
        )
        ids = [str(int(mid)) for mid in item_ids_unique]
        col.upsert(ids=ids, embeddings=V_norm.tolist())
        self._user_pref_collection = col

        # Store user factors mapping uid → vector
        self._user_pref_factors: dict[int, np.ndarray] = {
            int(uid): U_norm[i] for i, uid in enumerate(user_ids_unique)
        }

        # Centroid of item factors (cold-start fallback)
        self._user_pref_centroid: np.ndarray = V_norm.mean(axis=0)

        # Persist to disk
        np.savez(
            npz_path,
            user_ids=np.array(list(self._user_pref_factors.keys())),
            vectors=np.array(list(self._user_pref_factors.values())),
            centroid=self._user_pref_centroid,
        )
        logger.info(
            "User-pref embeddings ready (%d users, %d items, %d dims).",
            len(self._user_pref_factors),
            len(ids),
            n_components,
        )

    def _load_user_pref_factors_from_disk(self, npz_path: Path) -> None:
        data = np.load(npz_path)
        self._user_pref_factors: dict[int, np.ndarray] = {
            int(uid): vec
            for uid, vec in zip(data["user_ids"], data["vectors"])
        }
        self._user_pref_centroid: np.ndarray = data["centroid"]
        logger.info(
            "Loaded %d user-pref factor vectors from disk.",
            len(self._user_pref_factors),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public query helpers
    # ──────────────────────────────────────────────────────────────────────

    def get_user_factor(self, user_id: int) -> np.ndarray | None:
        """Return the MF user-factor vector for ``user_id``, or None."""
        return self._user_factors.get(user_id)

    def get_item_factors(self, movie_ids: list[int]) -> dict[int, np.ndarray]:
        """
        Retrieve item-factor vectors for the given movie IDs from ChromaDB.

        Returns a dict mapping movieId → vector (only for IDs that exist).
        """
        str_ids = [str(mid) for mid in movie_ids]
        result = self._assoc_collection.get(ids=str_ids, include=["embeddings"])
        return {
            int(iid): np.array(vec)
            for iid, vec in zip(result["ids"], result["embeddings"])
        }

    def get_semantic_vectors(self, movie_ids: list[int]) -> dict[int, np.ndarray]:
        """Retrieve semantic embedding vectors for the given movie IDs."""
        str_ids = [str(mid) for mid in movie_ids]
        result = self._semantic_collection.get(ids=str_ids, include=["embeddings"])
        return {
            int(iid): np.array(vec)
            for iid, vec in zip(result["ids"], result["embeddings"])
        }

    def get_user_pref_item_factors(self, movie_ids: list[int]) -> dict[int, np.ndarray]:
        """
        Retrieve user-preference-space item vectors (small independent SVD)
        for the given movie IDs.

        Returns a dict mapping movieId → unit-norm vector of shape
        ``(user_pref_features,)``.
        """
        str_ids = [str(mid) for mid in movie_ids]
        result = self._user_pref_collection.get(ids=str_ids, include=["embeddings"])
        return {
            int(iid): np.array(vec)
            for iid, vec in zip(result["ids"], result["embeddings"])
        }

    def get_user_pref_factor(self, user_id: int) -> np.ndarray | None:
        """
        Return the SVD-derived preference vector for ``user_id``, or None if
        the user was not in the factorization (cold-start). Falls back to
        the centroid of all item factors with additive noise.
        """
        vec = self._user_pref_factors.get(user_id)
        if vec is not None:
            return vec
        # Cold-start fallback: centroid of all item factors + small noise
        if hasattr(self, "_user_pref_centroid"):
            noise = self.rng.normal(0, 0.01, size=self._user_pref_centroid.shape)
            fallback = self._user_pref_centroid + noise
            norm = np.linalg.norm(fallback)
            return fallback / norm if norm > 0 else fallback
        return None

    def held_out_for_user(self, user_id: int) -> pd.DataFrame:
        """Return the held-out rows for a specific user."""
        return self.held_out[self.held_out["userId"] == user_id]
