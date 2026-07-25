# Codebase Guide

> **Purpose:** A navigational and conceptual reference for the simulation. Use this document to understand what each piece of code does, why it is designed that way, and where potential inconsistencies might emerge. Keep it updated alongside significant design changes.

---

## Research Context

This repository tests whether different classes of simulated users and scoring agents reveal different strengths, weaknesses, and long-horizon effects in recommendation loops. MovieLens-32M is the fixed test bed.

The central research question is: **does the choice of agent implementation change what we can learn about a recommender, and if so, in what ways?**

The codebase intentionally keeps two representations of the user **separate**:

- The **platform recommender's** view: LensKit BiasedMF latent factors, high-dimensional (default 64), updated only from explicit ratings.
- The **agent/persona's** view: a smaller TruncatedSVD preference space (default 8 dimensions), initialized from training data and updated from explicit-only debiased residuals.

That separation is a deliberate design choice: it lets you ask whether the recommender's internal model of the user aligns with the user's own preference signal, and whether different agent implementations drive that alignment differently.

---

## Repository Layout

```
llm_testing/
├── main.py                    # CLI entry point
├── sim/                       # core simulation package
│   ├── config.py              # SimConfig — all hyperparameters
│   ├── environment.py         # data loading, splits, embeddings
│   ├── recommender.py         # platform recommender (LensKit wrapper)
│   ├── runner.py              # experiment orchestration, MLflow logging
│   ├── persona.py             # AgentPersona, archetypes, population factory
│   ├── user_agent.py          # SimulatedUser (persona + agent wrapper)
│   ├── population.py          # assignment logic for multi-agent experiments
│   ├── hpo.py                 # recommender HPO harness
│   ├── archetypes.py          # archetype registry (casual, binger, critic)
│   ├── attention.py           # AttentionStrategy implementations
│   ├── attendance.py          # AttendanceStrategy implementations
│   └── agents/
│       ├── base.py            # AbstractAgent protocol
│       ├── __init__.py        # AGENT_REGISTRY + build_agent()
│       ├── associative.py     # implemented: dot-product in MF space
│       ├── residual_profile.py # implemented: weighted residual item profile
│       ├── item_item.py       # implemented: item-item neighborhood scoring
│       ├── semantic.py        # stub (NotImplementedError)
│       ├── seq2seq.py         # stub (NotImplementedError)
│       └── llm.py             # stub (NotImplementedError)
├── tests/                     # synthetic-fixture test suite (no real ML-32M needed)
├── docs/                      # planning and reference documents
│   ├── CODEBASE_GUIDE.md      # ← this file
│   └── METRIC_DICTIONARY.md   # full metric reference
├── reports/                   # generated analysis outputs (CSVs, images, markdown)
├── data/ml-32m/               # MovieLens-32M CSVs
└── embeddings/chroma/         # persisted ChromaDB embedding store
```

---

## Component Overview

### 1. SimConfig (`sim/config.py`)

The single authoritative record of all experiment parameters. Passed to every component. Key design points:

- `as_dict()` is used to log params to MLflow.
- `to_json_dict()` / `from_dict()` / `from_json_file()` support HPO config serialization.
- **Cache keys** (`split_cache_key`, `platform_factor_cache_key`, `user_pref_cache_key`, `semantic_cache_key`) let the Environment skip expensive recomputation when only unrelated config fields change.
- **Two agent-population fields** coexist deliberately:
  - `agent_type` — the legacy single-agent field, used when `agent_types` is empty.
  - `agent_types` + `agent_type_proportions` + `agent_assignment_mode` — new multi-agent composition fields.

**Watch for:** `as_dict()` logs `agent_types` and `agent_type_proportions` as raw Python string reprs (`str([...])`). If you query MLflow for these, you'll need to `eval()` or re-parse them. Consider migrating to JSON strings for cleaner tooling.

---

### 2. Environment (`sim/environment.py`)

Owns all data I/O, splitting, and embedding generation. Constructed once per run; passed to every other component.

**Data splits:**

```
all ratings
├── train_ratings    (used to train recommender + compute biases)
├── validation       (small holdout used by HPO for candidate selection)
└── held_out         (held back for final evaluation / full-simulation assessment)
```

Splits are deterministic given `SimConfig.split_cache_key()`. The Environment skips re-splitting if a cached version with the same key exists.

**Embedding stores (ChromaDB):**

| Collection | Source | Used by |
|---|---|---|
| `associative_item_factors_{key}` | LensKit BiasedMF item factors (high-dim) | `Recommender`, `AssociativeAgent` |
| `user_pref_item_factors_{key}` | TruncatedSVD item factors (low-dim) | all agents, `AgentPersona` |
| `semantic_movie_embeddings_{key}` | sentence-transformers | `SemanticAgent` (stub) |

**Key public methods used by other components:**

| Method | Used by | Purpose |
|---|---|---|
| `held_out_for_user(uid, split=)` | Runner | Retrieve a user's held-out/validation items |
| `get_user_pref_item_factors(ids)` | Runner, agents | Low-dim item vectors for agent scoring |
| `get_item_factors(ids)` | Runner | High-dim BiasedMF item vectors for diagnostics |
| `get_user_factor(uid)` | Runner | High-dim BiasedMF user vector for diagnostics |
| `get_user_pref_factor(uid)` | `build_persona` | Initialize persona preference vector |
| `debias_rating(uid, mid, rating)` | Runner | Remove global+user+item bias from a raw rating |
| `get_rating_bias(uid, mid)` | Runner | Return the bias component alone |

**Watch for:** Embedding caches are keyed by a content hash of the relevant config params. If you change `mf_features`, `mf_epochs`, `mf_regularization`, `mf_damping`, or `user_pref_features`, the old cache collections are **not deleted** — they just sit unused under different names. Clean up `embeddings/chroma/` manually if disk space is a concern.

---

### 3. Recommender (`sim/recommender.py`)

Wraps a LensKit `BiasedMF` top-N pipeline and owns platform-side recommendation state.

**What it receives:**
- Training ratings at init (including replicated rows for simulated users in `one_per_agent_type` mode).
- Raw explicit ratings as feedback during simulation (NOT watched/add_to_list signals — see feedback design below).

**Seen-item exclusion:**
- `_all_seen`: seeded from training set at init; updated after each round via `advance_round()`.
- `_round_seen`: cleared per round; updated after each recommendation batch via `mark_sent()`.
- The caller (runner) is responsible for calling `mark_sent()` and `advance_round()` in the right order.

**Replicated users (one_per_agent_type mode):**
When a simulated user ID differs from its base user ID, the recommender creates a copy of the base user's training history under the simulated ID. This ensures replicated users are not cold-start from the recommender's perspective.

**Watch for:** Retraining happens at the end of each round via `retrain()`. The retraining combines the original expanded training data with all accumulated feedback. If feedback volume is very small relative to training data, retraining effects will be negligible. That may or may not be the intended behavior depending on your experimental design.

---

### 4. Agents (`sim/agents/`)

All agents implement the `AbstractAgent` protocol from `sim/agents/base.py`:

```python
def evaluate(self, candidates: ItemList, persona: AgentPersona, item_factors: dict[int, np.ndarray]) -> ItemList
def update(self, user_id: int, interactions: list[tuple[int, str, float]]) -> None
```

**Item vectors passed to agents** are always the **low-dimensional TruncatedSVD user-preference vectors**, not the high-dimensional BiasedMF platform vectors. This is a critical invariant: agents and personas live in one latent space; the recommender lives in a separate one.

#### AssociativeAgent (`associative.py`) ✅

- Computes a user preference vector as a weighted average of training item vectors (weighted by debiased residual ratings).
- Scores candidates with `preference_vector · item_vector`.
- **Online updates:** new explicit interactions update the preference vector as a running weighted average.
- The `associative_baseline` alias maps to the same class; the distinction is intended for future differentiation or ablation flags.

#### ResidualProfileAgent (`residual_profile.py`) ✅

- Same preference vector concept as associative, but the update is an absolute-residual-weighted average rather than a simple dot product on stored factors.
- Designed to emphasize strong preference signal (both positive and negative) more explicitly.
- Initialized from training data history via `_history.py`.

#### ItemItemNeighborhoodAgent (`item_item.py`) ✅

- Scores each candidate by computing its similarity to each item in the user's rating history.
- Score: `sum(residual × sim(candidate, hist_item)) / sum(|residual|)`.
- No persistent profile vector — it reasons directly over rated history.
- Heavier at inference time; lighter to update.

#### SemanticAgent, Seq2SeqAgent, LLMAgent — stubs

These raise `NotImplementedError` immediately. Do not configure them in experiments.

---

### 5. AgentPersona (`sim/persona.py`)

Per-user behavioral state. One persona instance per simulated user ID.

**Fixed at init (sampled from archetype priors):**

| Field | Meaning |
|---|---|
| `archetype` | Which behavioral archetype (casual, binger, critic) |
| `tau` | Softmax temperature for item sampling |
| `score_floor` | Minimum agent score for an item to be considered |
| `action_intercepts` / `action_weights` | Logistic action-selection parameters per action type |
| `lr` | Preference vector learning rate |
| `baseline_logit` | Base attendance propensity |
| `attention` / `attendance` | Strategy objects for budget and attendance modeling |

**Mutable during simulation:**

| Field | Meaning |
|---|---|
| `pref_vector` | Evolving preference representation (unit-norm, low-dim) |
| `budget` | Current attention budget (depleted per request, restored per round) |
| `recent_signal_ewma` | Exponentially weighted satisfaction signal (drives attendance) |
| `rounds_since_last_visit` | Used by attendance model |

**Action model (priority order within one request):**
1. Score candidates via the agent
2. Reject below `score_floor`
3. Softmax-sample from eligible items (temperature `tau`)
4. Per item: run logistic gate → assign action in order `watch → rate → add_to_list → ignore`

**Watch for:** The `pref_vector` starts as the TruncatedSVD user factor from training. Only explicit `rate` interactions update it. If a user never generates ratings, their preference vector never moves.

---

### 6. Population & Assignment (`sim/population.py`)

Handles the mapping from base users to simulated user instances and agent types.

**`UserAssignment`**: a frozen record of `(sim_user_id, base_user_id, agent_type)`.

**Assignment modes:**

| Mode | Behaviour | When to use |
|---|---|---|
| `one_to_one` | Each base user gets one simulated instance with a randomly drawn agent type (according to `agent_type_proportions`) | Comparing aggregate population effects across agent-type mixes |
| `one_per_agent_type` | Each base user is replicated once per configured agent type, with independent simulated IDs | Per-user comparison of how different agents respond to the same history |

In `one_to_one` mode, `agent_type_proportions` controls the mix. In `one_per_agent_type` mode, proportions are ignored and all configured agent types are used.

**Persona cloning in replicated mode:** Replicated simulated users share the same archetype and initial `pref_vector` (deep-copied from the base persona). This ensures comparisons start from the same state and diverge only due to the agent type.

---

### 7. SimulatedUser (`sim/user_agent.py`)

A thin facade that joins one `AgentPersona` with one `AbstractAgent` instance and exposes the per-step interface used by the runner.

One shared agent instance is created per agent type (not per user). All users of the same type share one agent object, which maintains per-user state internally (via `user_id` keys in its history dictionaries).

**Key attributes added alongside the original ones:**

| Attribute | Meaning |
|---|---|
| `uid` | Simulated user ID (may differ from base in replicated mode) |
| `base_user_id` | Original base user whose history initialized this instance |
| `agent_type` | The name of the scoring agent |

These are passed through to all output artifacts so that downstream analysis can group by `agent_type` or join on `userId` (base) vs `simulation_user_id`.

---

### 8. Feedback Design (Explicit-Only)

This is a deliberate design decision that affects all state update paths:

> Only **explicit `rate` interactions** produce learning updates. `watch` and `add_to_list` are behavioral outputs only and do not update the agent or recommender.

The reason: watch and add-to-list signals are artificial proxy values (`watch_signal=4.5`, `add_to_list_signal=3.0`) that were assigned by the simulation designer, not observed from real user behavior. Feeding them as learning signals would be circular.

**Debiasing split:**

| Recipient | What they receive |
|---|---|
| Agent / persona | Debiased residual: `rating - (global_bias + user_bias + item_bias)` |
| Recommender | Raw explicit rating (before debiasing) |

The rationale: BiasedMF internally learns the bias decomposition, so passing raw ratings is correct. Agents use their own preference model which operates in residual space, so they need pre-debiased residuals for update alignment.

---

### 9. SimulationRunner (`sim/runner.py`)

Orchestrates everything. Two profiles:

**`full` profile** (default):
1. For each round: attendance gate → recommendation request loop → agent scoring → persona action sampling → feedback split → recommender update → agent/persona update → metrics.
2. Logs per-round metrics and Parquet artifacts to MLflow.
3. Retrains the recommender at the end of each round.

**`recommender_only` profile:**
1. Evaluates the first recommendation batch for each user against their held-out items.
2. No simulated feedback; no retraining.
3. Produces rich diagnostic artifacts (popularity analysis, score gap analysis, residual correlation by both the recommender and the selected agent type).
4. Supports multi-agent populations — each simulated user's results carry their `agent_type` label.
5. Intended for: validating recommender quality, HPO, and cross-agent diagnostics without running the full loop.

**MLflow run management:**
- `run(manage_mlflow=True)` opens and closes its own run (default).
- `run(manage_mlflow=False)` logs into an already-active run (used by HPO to nest runs cleanly).

---

### 10. HPO (`sim/hpo.py`)

Config-driven recommender hyperparameter search using the `recommender_only` profile.

```
hpo_config.json → HPOConfig → run_hpo()
    └── foreach candidate config:
           run SimulationRunner(recommender_only, validation split)
           log as nested MLflow run
    └── select best by ndcg_at_k (tie-break: hit_fraction, residual pearson, popularity delta)
    └── rerun best config on held_out split
    └── write hpo_candidate_results.csv + best_recommender_config.json
```

Usage:
```bash
uv run python -m sim.hpo path/to/hpo-config.json
```

HPO config format:
```json
{
  "base_config": { <SimConfig fields> },
  "candidate_overrides": [
    { "mf_features": 32, "mf_epochs": 5 },
    { "mf_features": 64, "mf_epochs": 10 }
  ]
}
```

---

## Data Flow Through One Round

```
Environment
  ├── held_out_for_user(uid)       → held item IDs (ground truth)
  └── get_user_pref_item_factors() → low-dim item vectors

Recommender.recommend(uid, n)      → ItemList of candidates

SimulatedUser.step(candidates):
  ├── agent.evaluate(candidates, persona, item_factors)
  │     → ItemList with scores (agent-type-specific scoring in low-dim space)
  ├── persona.act(ranked_ids, scores)
  │     → list of (movieId, action, signal) for non-ignored items
  └── recs_rows                    → parquet output rows (with agent_type, userId, simulation_user_id)

Runner (feedback split):
  ├── explicit_ratings = [(mid, sig) for "rate" actions only]
  ├── agent_feedback   = [(mid, "rate", debias_rating(base_user_id, mid, sig))]
  └── raw_feedback     = [(mid, sig)]

Recommender.update_user(uid, raw_feedback)  → raw ratings to platform model
SimulatedUser.update(rnd, agent_feedback):
  ├── persona.update_preference(agent_feedback, acted_factors)
  ├── agent.update(uid, agent_feedback)
  └── persona EWMA, budget, attendance counters

Runner:
  ├── ctx.surfaced[uid].update(recs ∩ held_ids)
  └── metrics: hit_rate, ndcg, holdout_recall, action_mix, signal, budget
```

---

## Known Potential Inconsistencies

These are areas where the simulation code may not behave exactly as intended or may need revisiting:

1. **`watch_signal` / `add_to_list_signal` are never used for learning** (by design after the explicit-only refactor), but the config fields still exist and may mislead readers into thinking they drive learning. Consider adding a doc comment clarifying they are purely behavioral output magnitudes.

2. **`persona.update_preference` updates `pref_vector` using the residual-weighted update, but this is the same vector used to initialize agents from training data.** If a user rarely rates items in the simulation, the persona diverges from the agent's training-time initialization differently across agent types. This asymmetry may be intentional but should be monitored.

3. **`build_population` and `build_population_for_assignments` are two separate code paths** that both assign archetypes. If you change how archetypes are sampled in one, you must update the other. Consider unifying them.

4. **Recommender retraining** uses all accumulated feedback plus the original training data. Early rounds with few ratings produce near-identical models. The retrain becomes meaningful only once a non-trivial number of rating interactions have been collected.

5. **`agent_types` and `agent_type` coexist** in SimConfig. `build_user_assignments` falls back from `agent_types` to `agent_type` when the list is empty, which creates two separate code paths for the same intent. The `agent_type` field is the legacy path; `agent_types` is the intended future path. Be careful not to set both inconsistently.

6. **Persona `pref_vector` is unit-normalized at init** (from the TruncatedSVD factor). After updates via `update_preference`, the vector is no longer guaranteed to remain unit-norm (update uses a raw weighted sum, not re-normalization). Whether this matters depends on how each agent uses the vector — check agent `evaluate` implementations before assuming unit-norm.

7. **In `one_per_agent_type` mode, replicated simulated users share training history in the recommender but have independent feedback histories during simulation.** This is correct for clean cross-agent comparisons, but it means the recommender treats them as the same user at training time. If simulated feedback is substantial, the replicated users may influence each other's subsequent recommendations indirectly through retraining.

---

## Running Experiments

```bash
# Full simulation (default)
uv run python main.py

# Diagnostics only (no feedback loop)
uv run python main.py --experiment_profile recommender_only

# Mixed-agent population comparison
uv run python main.py \
    --agent_types associative,item_item,residual_profile \
    --agent_type_proportions 0.4,0.4,0.2 \
    --agent_assignment_mode one_to_one

# Per-user cross-agent comparison (replicated mode)
uv run python main.py \
    --agent_types associative,item_item \
    --agent_assignment_mode one_per_agent_type \
    --experiment_profile recommender_only

# Recommender HPO
uv run python -m sim.hpo path/to/hpo-config.json
```

---

## Development Commands

```bash
uv run ty check sim tests       # type/lint check
uv run python -m pytest -q      # full test suite
uv build                        # package build
```

Tests use synthetic fixtures from `tests/conftest.py` — no real ML-32M data required.

---

## Related Documents

| Document | Status | Purpose |
|---|---|---|
| `docs/METRIC_DICTIONARY.md` | Current | Full reference for all logged metrics and artifact columns |
| `README.md` | Outdated | Project overview (does not reflect traditional agents, HPO, or mixed populations) |
| `docs/PLANNING.md` | Historical | Original v1 architecture sketch |
| `docs/PLANNING_V2.md` | Historical | v2 behavioral model design notes |
| `docs/PLANNING_BASELINE.md` | Historical | Design notes for a passthrough baseline mode (not yet implemented) |
