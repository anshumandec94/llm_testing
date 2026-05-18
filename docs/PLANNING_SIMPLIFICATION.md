# Simulation simplification plan

## Problem and recommended approach

The repo currently mixes several distinct effects in one experiment loop: raw LensKit retrieval quality, agent-side re-ranking in a different latent space, stochastic item/action selection, attention depletion, attendance dropout, and round-by-round recommender updates. That makes low recall or NDCG hard to interpret because a bad result can come from the recommender itself, the user-side scoring space, or the simulation dynamics layered on top.

The safest way to simplify this codebase is **not** to replace the current simulation. Instead, add a small set of **explicit analysis modes** that preserve the existing architecture while stripping away one source of complexity at a time. The first mode should evaluate the recommender with no simulated-user transformation at all; later modes can reintroduce feedback and behavioral dynamics in a controlled order.

## Findings from the current codebase

- `Environment` already gives a clean temporal holdout split and owns all reusable data/embedding setup.
- `Recommender` already has the right isolation hooks for analysis: recommendation generation, seen-item tracking, feedback accumulation, and per-round retraining.
- The biggest current confound is that the implemented `AssociativeAgent` scores in the persona-side `user_pref_features` space, while the recommender ranks in LensKit MF space. When metrics are low, it is ambiguous whether retrieval is weak or the agent is overriding good candidates.
- `SimulatedUser` and `AgentPersona` are where most complexity enters: softmax sampling, action selection, preference updates, attention, and attendance.
- Tests are already organized around component seams, and `tests/conftest.py` provides a synthetic dataset, so the repo is well-positioned for an incremental simplification project.

## Recommended simplification ladder

### Phase 0: raw recommender evaluation

Add a **recommender-only evaluation path** that does not build personas or agents at all. This path should:

- use the existing holdout split from `Environment`
- call the LensKit recommender directly
- compute per-user and aggregate retrieval metrics on the first recommendation batch
- avoid `mark_sent`, action selection, feedback ingestion, and retraining

This is the diagnostic that answers: **"Can the recommender surface held-out items at all?"**

### Phase 1: passthrough static simulation

Add a **passthrough simulation mode** that still runs through the runner shape, but removes user-side transformation:

- no agent re-ranking; preserve recommender order and scores
- deterministic attendance: everyone attends
- deterministic perception/action: every exposed item is treated as `watch`, or a fixed top-k is accepted deterministically
- no preference drift
- no attention depletion
- no recommender updates or retraining

This keeps the loop shape intact while isolating exposure mechanics from behavioral noise.

### Phase 2: passthrough feedback simulation

Turn on only the recommender feedback loop while keeping all user behavior deterministic:

- same passthrough ranking
- deterministic actions/signals
- recommender feedback enabled
- retraining enabled between rounds
- still no attention, attendance, archetype heterogeneity, or preference updates

This is the diagnostic that answers: **"Does the closed loop help or hurt when the user model is no longer noisy?"**

### Phase 3: reintroduce complexity one axis at a time

Re-add the current simulation pieces in this order:

1. user preference updates
2. agent-side re-ranking
3. attention budget mechanics
4. attendance dynamics
5. archetype heterogeneity
6. alternate agent implementations

Each step should be a separate experimental mode or profile so results stay attributable.

## Design recommendations

### 1. Prefer named modes over many unrelated booleans

Avoid scattering new flags like `deterministic_actions`, `freeze_preference_vector`, `disable_rerank`, `disable_feedback`, and `disable_retrain` unless they are grouped under a higher-level preset. The cleaner design is to add a named experiment profile such as:

- `full`
- `recommender_only`
- `passthrough_static`
- `passthrough_feedback`

Then derive the lower-level behavior from that profile inside a small translation layer. This keeps the code analyzable and prevents configuration drift.

### 2. Preserve existing classes; add thin alternate paths

Do not rewrite `Environment` or `Recommender`. Most changes should be additive:

- a raw evaluation path that bypasses `SimulatedUser`
- a passthrough agent or equivalent "no rerank" path
- deterministic persona/strategy presets for simplified modes
- conditional runner behavior around feedback/retraining

### 3. Separate retrieval metrics from simulation metrics

The current runner focuses on simulation-round aggregates. For simplification work, add a second layer of metrics that makes attribution easier:

- first-batch holdout recall / hit rate / NDCG
- catalog coverage
- average rank of first held-out hit
- fraction of eval users with at least one held-out hit
- overlap between recommender rank and post-agent rank

That last metric is especially important in this repo because the recommender and persona score in different spaces.

### 4. Keep the tiny synthetic fixture as the default test bed

The first implementation target should be a fast, deterministic test matrix on the synthetic dataset in `tests/conftest.py`. Only after those modes are stable should you use the full ML-32M data for analysis runs.

## Proposed implementation plan

1. Add a small experiment-profile abstraction in config so simplified analysis modes are explicit and reproducible.
2. Add a recommender-only evaluation runner that computes raw retrieval metrics directly from `Environment` + `Recommender`.
3. Add a passthrough simulation profile that preserves runner structure but disables re-ranking, stochastic actions, attention, attendance, and preference drift.
4. Add a second passthrough profile that enables recommender feedback/retraining while keeping the user side deterministic.
5. Expand tests to cover each profile separately, starting with deterministic synthetic-data assertions.
6. Add result artifacts that compare recommender rank vs post-agent rank so future regressions are attributable.
7. Only after the simplified modes are stable, reintroduce one behavioral dimension at a time behind separate profiles.

## Test plan for the simplification work

- component tests for any new profile-selection/config logic
- unit tests for passthrough ranking behavior
- unit tests for disabled feedback vs enabled feedback
- integration test for `recommender_only` mode on the tiny fixture
- integration test for `passthrough_static` mode with deterministic metrics
- integration test for `passthrough_feedback` mode to verify retraining changes behavior without stochastic user effects

## Notes and risks

- The earlier `PLANNING_BASELINE.md` is directionally right, but it starts at a deterministic simulation baseline. The repo would benefit from going one step simpler first: a true recommender-only evaluation path.
- If simplified modes still run through the existing runner, be careful not to let hidden side effects survive, especially `mark_sent`, `update_user`, `advance_round`, and `retrain`.
- Because the implemented agent and the recommender rank in different latent spaces, any future analysis should explicitly log both orders before concluding the recommender is weak.
