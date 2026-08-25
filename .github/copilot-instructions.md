# Copilot Instructions

## Build, test, and lint commands

- Build the package: `uv build`
- Run the full test suite: `uv run python -m pytest -q`
- Run a single test: `uv run python -m pytest tests/test_attention.py::TestLinearDecayAttention::test_restore_full_returns_one -q`
- Run the repo's type/lint check: `uv run ty check sim tests`

## High-level architecture

- `main.py` is a thin CLI entrypoint. It converts command-line flags into a `SimConfig` and hands execution to `sim.runner.SimulationRunner`.
- `sim.environment.Environment` is the experiment bootstrapper. It loads MovieLens-style CSVs, creates the eval-user holdout split, builds the LensKit dataset, and manages three persisted embedding stores under ChromaDB:
  - associative item factors from LensKit `BiasedMF`
  - semantic movie embeddings from sentence-transformers
  - a separate low-dimensional user-preference factor space from `TruncatedSVD`
- `sim.recommender.Recommender` is the platform-side model. It trains a LensKit top-N pipeline on training ratings, accumulates simulation feedback, retrains once per round, and enforces seen-item exclusion with both `_round_seen` and `_all_seen`.
- `sim.user_agent.SimulatedUser` is the facade the runner talks to. It combines one shared scoring agent with one per-user `AgentPersona`, so the runner can treat "the user" as a single object even though scoring and behavior are separate concerns.
- `sim.persona.AgentPersona` owns the mutable user-side state: preference vector, attention budget, attendance EWMA, and attendance counters. Attention and attendance behavior are delegated to strategy objects from `sim.attention` and `sim.attendance`.
- `sim.runner.SimulationRunner` orchestrates the full loop: attendance gate -> recommendation request loop -> agent scoring -> persona action sampling -> recommender feedback update -> persona update -> per-round metrics and MLflow artifacts.
- `associative.py`, `residual_profile.py`, `item_item.py`, and `llm.py` are implemented today. `semantic.py` and `seq2seq.py` are still explicit stubs that raise `NotImplementedError`.

## Key conventions

- Keep platform state and user state separate. The recommender owns recommendation-time feedback and retraining; persona objects own preference drift, attention, and attendance. Do not merge those responsibilities when extending the simulation.
- The code intentionally uses **two latent spaces**:
  - LensKit MF factors for the recommender's platform view of users/items
  - smaller `user_pref_features` SVD factors for persona-side scoring and preference updates
  Agent `evaluate()` methods receive the persona plus user-pref-space item factors, not the recommender's internal user vectors.
- Embeddings are cached on disk in `embeddings/chroma`, and `Environment` skips rebuilding existing collections unless `force_rebuild_embeddings` is set. If you change embedding logic, dimensions, or data preparation, rebuild the caches instead of assuming code changes alone will take effect.
- Re-request behavior is stateful within a round. The runner calls `recommender.mark_sent()` immediately after each batch and `recommender.advance_round()` once per round; preserve that contract if you touch recommendation flow.
- Tests are designed to avoid the real `data/ml-32m` dataset. `tests/conftest.py` creates a temporary MovieLens-shaped dataset plus temporary embedding storage, so new tests should usually extend those fixtures instead of relying on local data files.
- The persona action model is structured, not ad hoc: score thresholding happens first, then softmax sampling over eligible items, then action selection in the fixed priority order `watch -> rate -> add_to_list -> ignore`.
- New archetypes or behavior strategies should be wired through the registries (`ARCHETYPE_REGISTRY`, `ATTENTION_REGISTRY`, `ATTENDANCE_REGISTRY`) so population construction keeps working through configuration instead of one-off branching.
