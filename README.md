# llm_testing

`llm_testing` is a research repository for **agent-based simulation of recommender systems**. The project uses movie recommendation as a fixed test bed and asks whether different classes of simulated users and agents reveal different strengths, weaknesses, and long-horizon effects in recommendation loops.

The current dataset is **MovieLens-32M**. The codebase intentionally separates:

- the **platform recommender** and its internal user representation
- the **agent / simulated-user model** and its own preference representation
- the **simulation loop** that turns repeated recommendations into feedback, attention, attendance, and preference drift

That separation is important for the research goal: the recommender's view of the user should be allowed to differ from the user-side model that drives behavior.

## Current focus

The immediate goal is to make the simulation **interpretable and trustworthy** before expanding the set of agent implementations.

In practice, current work is focused on:

- validating that the simulation behaves as expected
- isolating recommender quality from agent-side effects
- adding diagnostics that explain weak or strong runs
- building toward reliable cross-agent comparisons

## What is implemented today

### Environment

`sim.environment.Environment` loads MovieLens-style data, creates the train / held-out split, and manages persisted embeddings in ChromaDB.

It currently builds and stores:

- **associative item factors** from LensKit matrix factorization
- **semantic movie embeddings** from sentence-transformers
- **user-preference item factors** from a smaller TruncatedSVD space used by the persona / agent side

### Recommender

`sim.recommender.Recommender` wraps a LensKit top-N pipeline and owns the platform-side recommendation state.

It handles:

- training on held-in ratings
- excluding seen items across requests and rounds
- ingesting simulation feedback
- retraining between rounds

### Agent / persona side

The runner interacts with `sim.user_agent.SimulatedUser`, which combines:

- a shared **agent implementation** for scoring candidate items
- a per-user **persona** with mutable state like preference drift, attention, and attendance

Current agent status:

- **Associative agent**: implemented
- **Semantic agent**: stub
- **seq2seq agent**: stub
- **LLM agent**: stub

### Runner and diagnostics

`sim.runner.SimulationRunner` orchestrates:

1. attendance
2. recommendation requests
3. agent scoring / ranking
4. persona actions and feedback
5. recommender updates and retraining
6. MLflow logging and artifact generation

The repo also includes a **`recommender_only`** profile that bypasses simulated-user behavior and evaluates held-out diagnostics directly. This is useful when debugging whether failures come from the recommender itself, the agent, or the closed-loop dynamics.

## Repository layout

```text
llm_testing/
├── main.py               # CLI entry point
├── sim/                  # core simulation package
├── tests/                # synthetic-fixture test suite
├── data/                 # MovieLens-style datasets
├── embeddings/           # persisted ChromaDB embeddings
├── mlruns/               # local MLflow tracking
├── reports/              # generated analysis reports and plots
└── docs/                 # planning and internal documentation
```

## Running experiments

Use `uv` / `uv run` for repo commands.

### Full simulation

```bash
uv run python main.py
```

### Recommender-only diagnostics

```bash
uv run python main.py --experiment_profile recommender_only
```

### Recommender HPO search

```bash
uv run python -m sim.hpo path/to/hpo-config.json
```

The HPO config is JSON and should include:

- `base_config`: a normal `SimConfig` payload
- `candidate_overrides`: a list of recommender-parameter override dicts

The search runs candidates on the `validation` split using the existing
`recommender_only` workflow, then reruns the selected best config on `held_out`.

### Force-rebuild embeddings

```bash
uv run python main.py --force_rebuild_embeddings
```

### View CLI help

```bash
uv run python main.py --help
```

## Development commands

```bash
uv run ty check sim tests main.py
uv run python -m pytest -q
uv build
```

## Outputs

### MLflow

Runs are tracked locally in `mlruns/`. The current setup logs params, metrics, tags, and artifacts for both full simulations and recommender-only analysis runs.

### Reports

Read-later analyses live in `reports/`.

### Docs

Planning documents and internal notes live in `docs/`.
The metric reference lives in `docs/METRIC_DICTIONARY.md`.

## Where to look next

- `sim/config.py` for experiment configuration
- `sim/environment.py` for data loading and embedding generation
- `sim/recommender.py` for the platform model
- `sim/runner.py` for evaluation modes and diagnostics
- `tests/conftest.py` for the synthetic test dataset used by the suite
