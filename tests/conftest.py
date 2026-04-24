"""
tests/conftest.py — shared fixtures for the simulation test suite.

Creates a tiny synthetic ML-32M-shaped dataset so tests run fast without
touching the real data files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sim.config import SimConfig


@pytest.fixture(scope="session")
def tiny_config(tmp_path_factory) -> SimConfig:
    """
    A SimConfig pointing at a small synthetic dataset in a temp directory.
    Uses 30 users × 60 movies with ~500 ratings.
    Embeddings are built with 8 MF features to keep the fixture fast.
    """
    base: Path = tmp_path_factory.mktemp("sim_test")
    data_dir = base / "ml-32m"
    data_dir.mkdir()
    embeddings_dir = base / "chroma"

    rng = np.random.default_rng(0)
    n_users, n_movies = 40, 80
    # Generate enough interactions so every user has >= min_ratings
    n_ratings = 3000

    user_ids = rng.integers(1, n_users + 1, size=n_ratings)
    movie_ids = rng.integers(1, n_movies + 1, size=n_ratings)
    ratings_val = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n_ratings).astype(float)
    timestamps = rng.integers(900_000_000, 1_000_000_000, size=n_ratings)

    ratings_df = (
        pd.DataFrame(
            {
                "userId": user_ids,
                "movieId": movie_ids,
                "rating": ratings_val,
                "timestamp": timestamps,
            }
        )
        .drop_duplicates(subset=["userId", "movieId"])
        .reset_index(drop=True)
    )
    ratings_df.to_csv(data_dir / "ratings.csv", index=False)

    genres_pool = [
        "Action|Comedy",
        "Drama|Romance",
        "Thriller",
        "Animation|Family",
        "Sci-Fi",
    ]
    movies_df = pd.DataFrame(
        {
            "movieId": range(1, n_movies + 1),
            "title": [f"Movie {i} ({2000 + i % 25})" for i in range(1, n_movies + 1)],
            "genres": [genres_pool[i % len(genres_pool)] for i in range(n_movies)],
        }
    )
    movies_df.to_csv(data_dir / "movies.csv", index=False)

    overviews_df = pd.DataFrame(
        {
            "movieId": range(1, n_movies + 1),
            "overview": [
                f"This is a synthetic test movie number {i}." for i in range(1, n_movies + 1)
            ],
        }
    )
    overviews_df.to_csv(data_dir / "movie_overviews.csv", index=False)

    return SimConfig(
        data_dir=data_dir,
        embeddings_dir=embeddings_dir,
        eval_user_frac=0.3,
        holdout_frac=0.2,
        min_ratings=10,
        num_rounds=2,
        rec_list_size=10,
        accept_k=3,
        max_requests_per_round=2,
        mf_features=8,
        user_pref_features=4,   # very small for test speed
        semantic_model="all-MiniLM-L6-v2",
        force_rebuild_embeddings=True,
        random_seed=0,
        archetype_mix={"casual": 1.0},
    )


@pytest.fixture(scope="session")
def env(tiny_config: SimConfig):
    """Initialised Environment built once for the whole test session."""
    from sim.environment import Environment
    return Environment(tiny_config)


@pytest.fixture(scope="session")
def recommender(tiny_config: SimConfig, env):
    """Trained Recommender built once for the whole test session."""
    from sim.recommender import Recommender
    return Recommender(tiny_config, env)


@pytest.fixture(scope="session")
def population(tiny_config: SimConfig, env):
    """Agent population built once for the whole test session."""
    from sim.persona import build_population
    rng = __import__("numpy").random.default_rng(tiny_config.random_seed)
    return build_population(tiny_config, env, rng)


@pytest.fixture(scope="session")
def sample_persona(population, env):
    """Return the persona for the first eval user."""
    uid = env.eval_users[0]
    return population[uid]
