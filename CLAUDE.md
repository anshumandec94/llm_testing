# CLAUDE.md - llm_testing

PhD research codebase: LLM/agent-based simulation of recommender systems on MovieLens-32M.

---

## Commands

- Run tests: `uv run python -m pytest -q`
- Run single test: `uv run python -m pytest tests/test_foo.py::TestClass::test_name -q`
- Type/lint check: `uv run ty check sim tests`
- Build package: `uv build`
- Run simulation: `uv run python main.py [flags]`

Always use `uv` / `uv run`. Never bare `python` or `pip`.

---

## Repo Rules

- Do not modify `pyproject.toml` unless explicitly instructed.
- Prefer single config files as the source of truth for simulation runs; do not duplicate defaults in the CLI.
- Place generated analysis reports in `reports/`.
- Tests must not touch `data/ml-32m/`. Use the fixtures in `tests/conftest.py` (synthetic MovieLens-shaped data + temp embedding storage).

---

## Architecture

`main.py` is a thin CLI that builds a `SimConfig` and calls `SimulationRunner`.

**Core components:**

| Module | Role |
|---|---|
| `sim/config.py` | `SimConfig` - all hyperparameters, cache keys, serialization |
| `sim/environment.py` | Data loading, train/validation/held-out split, ChromaDB embedding stores |
| `sim/recommender.py` | Platform-side LensKit `BiasedMF` top-N; accumulates feedback, retrains per round |
| `sim/runner.py` | Orchestrates the full simulation loop; logs to MLflow |
| `sim/user_agent.py` | `SimulatedUser` facade - combines one scoring agent + one `AgentPersona` |
| `sim/persona.py` | Per-user mutable state: preference vector, attention budget, attendance EWMA |
| `sim/attention.py` | `AttentionStrategy` implementations (registry: `ATTENTION_REGISTRY`) |
| `sim/attendance.py` | `AttendanceStrategy` implementations (registry: `ATTENDANCE_REGISTRY`) |
| `sim/population.py` | Multi-agent assignment logic |
| `sim/hpo.py` | Recommender HPO harness |
| `sim/archetypes.py` | Archetype registry (casual, binger, critic) |
| `sim/agents/` | Agent implementations + registry |

**Agents (`sim/agents/`):**

| File | Status |
|---|---|
| `associative.py` | Implemented - dot-product in MF space |
| `residual_profile.py` | Implemented - weighted residual item profile |
| `item_item.py` | Implemented - item-item neighborhood scoring |
| `semantic.py` | Stub - raises `NotImplementedError` |
| `seq2seq.py` | Stub - raises `NotImplementedError` |
| `llm.py` | Stub - raises `NotImplementedError` |

New agents must be registered in `AGENT_REGISTRY` in `sim/agents/__init__.py`.

---

## Key Design Constraints

**Two latent spaces are kept separate intentionally:**
- LensKit MF factors (high-dim, default 64) - the recommender's platform view of users/items.
- TruncatedSVD factors (low-dim, default 8) - the agent/persona's preference space.
Agent `evaluate()` receives user-pref-space item factors, not the recommender's internal user vectors.
Do not collapse these.

**Platform state vs. user state:**
- `Recommender` owns recommendation-time feedback and retraining.
- `AgentPersona` owns preference drift, attention, and attendance.
Do not merge these responsibilities.

**Round contract:**
- `recommender.mark_sent()` is called immediately after each batch.
- `recommender.advance_round()` is called once per round.
Preserve this order when touching recommendation flow.

**Persona action order is fixed:** score threshold -> softmax sampling -> `watch -> rate -> add_to_list -> ignore`.

**Embedding caches** live in `embeddings/chroma/` and are keyed by config content hashes.
If you change MF dimensions, SVD features, or embedding logic, set `force_rebuild_embeddings=True` or manually clean `embeddings/chroma/`.
Stale collections under old keys are not auto-deleted.

**New archetypes / strategies** must be wired through the registries (`ARCHETYPE_REGISTRY`, `ATTENTION_REGISTRY`, `ATTENDANCE_REGISTRY`) so population construction stays config-driven.

---

## MLflow & Artifacts

- Metrics and params are logged to `mlflow.db` (local).
- `SimConfig.as_dict()` logs `agent_types` and `agent_type_proportions` as raw Python `str()` reprs - use `eval()` or re-parse when querying MLflow for these fields.
- Per-round recommendation parquets land in `mlartifacts/`.

---

## Data

- `data/ml-32m/` - MovieLens-32M CSVs (not in git, required for real runs).
- Split: `train_ratings` / `validation` / `held_out`. Splits are deterministic given `SimConfig.split_cache_key()`.
