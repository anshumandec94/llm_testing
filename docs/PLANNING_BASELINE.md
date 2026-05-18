# Baseline Mode — Planning Document

> **Status:** Planning only. No code changes in this document.
> Goal: introduce a fully deterministic, passthrough baseline that isolates and diagnoses raw recommender quality before re-adding behavioural complexity.

---

## 1. Problem Statement

The current simulation has many interacting components, each adding stochasticity or a learned transformation on top of what the recommender returns:

| Component | What it does to recommender output |
|---|---|
| `AssociativeAgent` | Re-ranks the LensKit candidate list using the persona's `pref_vector` (dot-product in a separate MF space) |
| `persona.act()` | Softmax-samples which items get attention, then runs a logistic gate per-item to decide action |
| `persona.update_preference()` | Gradient-steps `pref_vector` each round, changing future re-ranking |
| `LinearDecayAttention` | Depletes budget across requests; later items in a session receive less attention |
| `LogisticAttendance` | Probabilistically skips users each round, introducing noise into aggregate metrics |
| Archetype mix (casual/binger/critic) | Creates heterogeneous action behaviour, masking aggregate signal |

The result is that metrics like `hit_rate` and `ndcg@6` are a function of *all* of these at once. Before we can trust that those numbers reflect the recommender's quality, we need a baseline where every component passes through the recommender's ranking unchanged and deterministically.

---

## 2. Target Baseline Behaviour

In "baseline mode" the simulation should behave as follows:

- **Every user attends every round** (no dropout).
- **Every recommended item is acted on** (no softmax sampling, no score-floor rejection).
- **Action is always `watch`** (fixed signal = `config.watch_signal`; no logistic gate, no Beta sampling).
- **Attention budget never depletes** (all three requests within a round receive a full budget).
- **The agent does not re-rank** (LensKit's own scores are preserved; the recommender is evaluated on its own merits).
- **The preference vector is never updated** (no gradient steps; `pref_vector` stays at its initial value throughout the run).
- **All users share the same fixed archetype traits** (no per-user noise sampling).

---

## 3. Gap Analysis — What Needs to Change

### 3.1 New: `PassthroughAgent` (`sim/agents/passthrough.py`)

**Gap:** There is no agent that returns the LensKit-ranked list unchanged. `AssociativeAgent` always applies a dot-product re-rank, replacing LensKit's scores with preference-space scores. To measure the recommender in isolation, we need a no-op agent.

**Proposed behaviour:**
- `evaluate(candidates, persona, item_factors)` — returns `candidates` unchanged (no score modification, preserves whatever scores LensKit attached).
- `update(user_id, interactions)` — no-op, same as `AssociativeAgent`.

**New `agent_type` value:** `"passthrough"` — wired in `sim/user_agent.py`'s `_build_agent()`.

---

### 3.2 New: `NoDecayAttention` (`sim/attention.py`)

**Gap:** Every existing `AttentionStrategy` depletes the budget (even `PerRequestAttention`). We need a strategy where `deplete()` is a no-op, `effective_k` always equals `list_size`, and `restore()` always returns 1.0.

**Proposed class:** `NoDecayAttention(AttentionStrategy)`
- `effective_k(list_size, budget)` → `list_size` (always see every item)
- `deplete(list_size, budget)` → `budget` unchanged (always returns whatever was passed in)
- `restore(end_budget, signal)` → `1.0`

**Registry key:** `"NoDecay"` in `ATTENTION_REGISTRY`.

**Note:** `AlwaysAttend` for attendance already exists and is sufficient — no new attendance class needed.

---

### 3.3 New: `"baseline"` Archetype (`sim/archetypes.py`)

**Gap:** There is no archetype that removes sampling noise from trait initialisation. All existing archetypes sample `tau`, `score_floor`, `lr`, and `baseline_logit` from Normal priors, giving each user slightly different behaviour.

**Proposed archetype:** `BASELINE_ARCHETYPE = ArchetypeConfig(...)`

Key design choices:
- `tau_mean=1.0, tau_std=0.0` — no per-user temperature noise.
- `score_floor_mean=-999.0, score_floor_std=0.0` — effectively no score floor; every item is eligible.
- `lr_mean=0.0, lr_std=0.0` — zero learning rate (pairs with `freeze_preference_vector`, but this makes the trait itself inert).
- `baseline_logit_mean=999.0, baseline_logit_std=0.0` — deterministic attendance (pairs with `AlwaysAttend`).
- `action_intercepts={"watch": 999.0, "rate": -999.0, "add_to_list": -999.0}` — near-certain `watch`, near-zero `rate`/`add_to_list`.
- `action_weights={"watch": 0.0, "rate": 0.0, "add_to_list": 0.0}` — action probability independent of score.
- `attention_strategy="NoDecay"`, `attendance_strategy="AlwaysAttend"`.

**Registry key:** `"baseline"` in `ARCHETYPE_REGISTRY`.

---

### 3.4 New Config Flags (`sim/config.py`)

Two new boolean fields on `SimConfig`:

| Field | Default | Effect when `True` |
|---|---|---|
| `freeze_preference_vector` | `False` | `persona.update_preference()` becomes a no-op; `pref_vector` stays frozen at its initial value for the entire run |
| `deterministic_actions` | `False` | `persona.act()` skips softmax sampling and the logistic gate; accepts all items, assigns fixed `watch` action |

These are orthogonal to each other and to the agent/archetype selection, so they can be toggled individually to isolate specific effects.

**`as_dict()` must include both new keys** for MLflow parameter logging.

---

### 3.5 Modify `persona.act()` (`sim/persona.py`)

**Gap:** `act()` currently always uses softmax sampling and the logistic action gate. When `config.deterministic_actions is True`, it should skip both and return one `(movie_id, "watch", config.watch_signal)` tuple for each candidate above the score floor (which with the baseline archetype is every item).

**Proposed logic inside `act()`:**

```python
if config.deterministic_actions:
    # Accept all items, fixed watch action
    return [
        (mid, "watch", float(config.watch_signal))
        for mid in ranked_ids
    ]
```

This branch is inserted at the top of `act()`, before the softmax block.

---

### 3.6 Modify `persona.update_preference()` (`sim/persona.py`)

**Gap:** There is no way to skip the preference-vector gradient update without modifying the persona directly. The runner calls `ua.update(...)` which calls `persona.update_preference(...)` unconditionally.

**Proposed guard inside `update_preference()`:** accept an optional `freeze: bool = False` parameter.

Alternatively (and more cleanly), the runner already calls `ua.update(rnd, interactions, acted_factors, self.config)` — the `config` is passed through. The guard can be inside `SimulatedUser.update()` in `user_agent.py`:

```python
def update(self, rnd, interactions, acted_factors, config):
    self._agent.update(self.uid, interactions)
    if not config.freeze_preference_vector:
        self.persona.update_preference(interactions, acted_factors)
    # attendance EWMA + round counters updated unconditionally
```

This keeps all config-gating in `user_agent.py` and leaves `persona.update_preference()` itself unchanged.

---

## 4. TDD Plan — Tests First

Every piece above must have a failing test written *before* the implementation. The sequence below defines the order.

### Step 1 — `NoDecayAttention` tests (extend `tests/test_attention.py`)

Tests to write before implementing:
- `test_no_decay_registry_key` — `"NoDecay"` is in `ATTENTION_REGISTRY`.
- `test_no_decay_effective_k_equals_list_size` — `effective_k(10, 0.3) == 10`.
- `test_no_decay_deplete_is_noop` — `deplete(10, 0.7) == pytest.approx(0.7)`.
- `test_no_decay_restore_returns_one` — `restore(0.1, 0.0) == pytest.approx(1.0)`.

---

### Step 2 — `PassthroughAgent` tests (new file `tests/test_passthrough_agent.py`)

Tests to write before implementing:
- `test_passthrough_returns_item_list` — result is an `ItemList`.
- `test_passthrough_preserves_item_count` — `len(result) == len(candidates)`.
- `test_passthrough_preserves_ids` — result IDs identical to candidate IDs, same order.
- `test_passthrough_preserves_original_scores` — if LensKit attached scores, they are unchanged.
- `test_passthrough_update_is_noop` — `update(uid, [])` does not raise.
- `test_passthrough_wired_in_build_agent` — `_build_agent(config, env)` with `agent_type="passthrough"` returns a `PassthroughAgent` instance.

---

### Step 3 — `"baseline"` archetype tests (extend `tests/test_persona.py` or new file)

Tests to write before implementing:
- `test_baseline_archetype_in_registry` — `"baseline"` is in `ARCHETYPE_REGISTRY`.
- `test_baseline_archetype_uses_no_decay_attention` — `build_persona(...)` with baseline archetype assigns a `NoDecayAttention` instance.
- `test_baseline_archetype_uses_always_attend` — assigns an `AlwaysAttend` instance.
- `test_baseline_score_floor_accepts_all` — a persona built from baseline archetype has `score_floor` below any expected score (i.e., `< -100`).
- `test_baseline_watch_probability_near_one` — calling `_select_action(score=0.0, ...)` on a baseline persona returns `"watch"` in >99% of calls.

---

### Step 4 — `SimConfig` new fields tests (new `tests/test_config.py` or inline)

Tests to write before implementing:
- `test_config_freeze_preference_vector_defaults_false` — `SimConfig().freeze_preference_vector is False`.
- `test_config_deterministic_actions_defaults_false` — `SimConfig().deterministic_actions is False`.
- `test_config_as_dict_includes_new_keys` — both keys appear in `config.as_dict()`.

---

### Step 5 — `deterministic_actions` flag tests (extend `tests/test_persona.py`)

Tests to write before implementing:
- `test_deterministic_act_returns_all_candidates` — when `deterministic_actions=True`, `len(result) == len(ranked_ids)`.
- `test_deterministic_act_all_watch` — every returned tuple has `action == "watch"`.
- `test_deterministic_act_signal_equals_watch_signal` — every signal equals `config.watch_signal`.
- `test_deterministic_act_ignores_softmax` — result is same regardless of `tau` value.

---

### Step 6 — `freeze_preference_vector` flag tests (extend `tests/test_persona.py` or `tests/test_user_agent.py`)

Tests to write before implementing:
- `test_frozen_pref_vector_unchanged_after_update` — after calling `ua.update(...)` with `freeze_preference_vector=True`, `persona.pref_vector` is byte-identical to its pre-update value.
- `test_unfrozen_pref_vector_changes_after_update` — with `freeze_preference_vector=False` (default), `persona.pref_vector` changes after a non-empty interactions list.

---

### Step 7 — Integration test: baseline run (new `tests/test_baseline_run.py`)

This is the highest-value test: it verifies that the full simulation runs cleanly with every baseline component wired together, and that the observed metrics are what pure recommender quality would produce.

Tests to write:
- `test_baseline_run_completes` — `SimulationRunner(baseline_config).run()` returns a non-empty DataFrame without raising.
- `test_baseline_attendance_rate_is_one` — `summary_df["attendance_rate"].mean() == pytest.approx(1.0)`.
- `test_baseline_action_mix_all_watch` — `summary_df["action_watch_frac"].mean() == pytest.approx(1.0)`.
- `test_baseline_mean_attention_consumed_is_zero` — budget never depletes so `mean_attention_consumed` should be 0 (or very close, depending on how budget delta is computed).
- `test_baseline_ndcg_is_positive` — sanity check that the recommender produces above-zero NDCG on the tiny synthetic dataset.

The `baseline_config` fixture for this test file:
```python
@pytest.fixture(scope="module")
def baseline_config(tiny_config):
    from dataclasses import replace
    return replace(
        tiny_config,
        agent_type="passthrough",
        archetype_mix={"baseline": 1.0},
        freeze_preference_vector=True,
        deterministic_actions=True,
    )
```

---

## 5. Implementation Order

Respecting the TDD rule (test first, then implement), the sequence is:

1. **Write** `NoDecayAttention` tests → **Implement** `NoDecayAttention` in `sim/attention.py`
2. **Write** `PassthroughAgent` tests → **Implement** `sim/agents/passthrough.py` + wire `_build_agent()` in `sim/user_agent.py`
3. **Write** `"baseline"` archetype tests → **Implement** `BASELINE_ARCHETYPE` in `sim/archetypes.py`
4. **Write** config-field tests → **Implement** `freeze_preference_vector` + `deterministic_actions` in `sim/config.py`
5. **Write** `deterministic_actions` persona tests → **Implement** guard in `persona.act()`
6. **Write** `freeze_preference_vector` user-agent tests → **Implement** guard in `SimulatedUser.update()`
7. **Write** integration tests → **Verify** all the above wire together correctly end-to-end

---

## 6. What We Are NOT Changing

- The `Recommender` class and its LensKit pipeline — we're diagnosing it, not modifying it.
- The `Environment`, data-loading, or hold-out split logic.
- Existing archetypes (`casual`, `binger`, `critic`).
- Existing attention strategies (`LinearDecay`, `ExponentialDecay`, `PerRequest`).
- Existing attendance strategies (`LogisticAttendance`, `BernoulliAttendance`, `ThresholdAttendance`).
- `runner.py` core loop logic — baseline config is purely a configuration change; the runner needs no structural changes.
- Any existing tests.

---

## 7. Open Questions / Decisions Deferred

1. **Score preservation in `PassthroughAgent`:** LensKit's `ItemList` may or may not carry scores depending on whether the pipeline attaches them. If scores are `None`, `PassthroughAgent.evaluate()` should either leave them `None` or attach a constant (e.g., rank-based). Decision: leave `None` and document that `persona.act()` will use uniform scores in that case — verify this is handled cleanly before implementing.

2. **`deterministic_actions` with `score_floor`:** Even in deterministic mode, if `score_floor` is non-trivially large, some items may be filtered out. With the baseline archetype `score_floor = -999`, this is a non-issue in the intended usage. But if someone passes `deterministic_actions=True` without the baseline archetype, the score floor will still apply. Decide whether to enforce that `score_floor` is also bypassed in deterministic mode, or document that it is not.

3. **Budget delta metric under `NoDecayAttention`:** `_run_user_session` in `runner.py` computes `budget_consumed = max(0.0, start_budget - ua.budget)`. With `NoDecayAttention`, budget never changes, so this metric is always 0. That's correct and useful (confirms no depletion), but we should make sure it doesn't cause a divide-by-zero or unexpected log in any metric calculation.
