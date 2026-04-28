# Simulation Design — v2 Planning Document

> **Status:** Design iteration — no code changes yet.
> Updated after design Q&A on 2026-04-22.

---

## 1. Code Review

### 1.1 `sim/config.py`

`SimConfig` is a flat dataclass with all hyperparameters and maps cleanly to MLflow via `as_dict()`.

**Gaps to address in v2:**
- No fields for action-type model parameters (softmax temperature, score floor, per-action signal strengths).
- No fields for attention budget (decay model, rate, recovery model).
- No fields for attendance model (baseline prior, EWMA window, noise scale).
- No fields for small independent MF (user preference model dimensionality, learning rate).
- No fields for population-mix archetype proportions (`archetype_mix`).

---

### 1.2 `sim/environment.py`

Loads ML-32M data, performs temporal hold-out splitting, and manages two ChromaDB collections:

| Collection | Embedding source | Used by |
|---|---|---|
| `associative_item_factors` | LensKit BiasedMF item factors | Recommender / (`AssociativeAgent` legacy) |
| `semantic_movie_embeddings` | `all-MiniLM-L6-v2` sentence-transformer | `SemanticAgent` (stub) |

**v2 addition:** A third cache — `user_pref_item_factors` — will be added here. This holds item vectors from a **small independent MF** (5–10 dims, no LensKit) trained on the same rating data. These are the item representations visible to the agent's preference model. Item side is computed once and persisted alongside the existing embeddings; user side is initialized per-persona and lives inside the `AgentPersona`.

---

### 1.3 `sim/recommender.py`

Two-scope seen-item exclusion works correctly. Re-request within a round is clean.

**Gap:** `update_user` currently accepts an `ItemList` and applies a hard-coded `rating=1.0`. In v2 it must accept `list[tuple[int, float]]` so differentiated action signals (watch=4.5, rate=sampled, add_to_list=3.0) feed the underlying BiasedMF correctly.

---

### 1.4 `sim/agents/base.py` / `sim/agents/associative.py`

The current `evaluate → top_n(k)` contract is deterministic. The `update(user_id, accepted: ItemList)` signature is too coarse.

**v2 changes:**
- `evaluate` remains: scores candidates and returns `ItemList` with scores.
- `top_n` acceptance is **replaced** by a new `act()` method on the persona (not the agent class — see §3).
- `update` signature changes to accept interaction tuples: `list[tuple[int, str, float]]`.
- The `AssociativeAgent` user-vector lookup is retired for the new in-persona preference model.

---

### 1.5 `sim/runner.py`

The current inner loop calls `agent.evaluate → ranked.top_n(k) → agent.update →
recommender.update_user`. The runner does not know about attention or attendance.

**v2 gaps:** The runner must integrate attendance checking (skip absent users), attention budget depletion per request, and assemble the interaction list for both `agent.update` and `recommender.update_user`.

---

### 1.6 Tests (current state)

| File | Tests | Coverage |
|---|---|---|
| `test_environment.py` | 14 | Data loading, hold-out split, ChromaDB collections, vector dims |
| `test_recommender.py` | 10 | Basic recs, seen-item exclusion, re-request, feedback accumulation |
| `test_agents.py` | 6 | Evaluate shape/scores, update no-op, unknown user |
| **Total** | **30** | — |

**v2 gaps:** No tests for persona creation, user preference model updates, softmax sampling, action selection, attention depletion/recovery, or attendance probability.

---

## 2. v2 Architecture Overview

```
SimConfig
  └─ archetype_mix: {"casual": 0.6, "binger": 0.4}

Population init (once per run)
  ├─ Environment                     (data, hold-out, 3 embedding collections)
  ├─ Recommender (LensKit BiasedMF)  (platform-side; 64 dims)
  └─ [AgentPersona] × |eval_users|  (user-side; 5-10 dims, independent MF)

Per round:
  for uid in eval_users:
      persona.attendance.will_attend()  →  skip if absent
      for req in 1..max_requests_per_round:
          recommender.recommend()
          recommender.mark_sent()
          agent.evaluate(candidates, persona)  →  scored ItemList
          persona.act(ranked)                  →  [(mid, action, signal)]
          persona.attention.deplete()
          if enough interactions or budget=0: break
      recommender.update_user(uid, [(mid, signal)])
      agent.update(uid, interactions)
      persona.update_preference(interactions)
      persona.attendance.record()
      persona.attention.restore()
  recommender.advance_round()
  recommender.retrain()           (if rnd > 1)
```

---

## 3. `AgentPersona` Dataclass

### 3.1 Purpose

Each eval user is represented by exactly one `AgentPersona` instance (strict 1:1 mapping). The persona:

1. Stores the **user's identity** — their dataset `user_id` and their assigned archetype.
2. Carries **fixed behavioural traits** — parameters that define *how* this user engages (attention decay rate, attendance propensity, softmax temperature, action logit weights). These are sampled from archetype-level priors at init and do not change during simulation.
3. Maintains **evolving preference state** — the user's internal vector in the small independent MF space. This is the only thing that changes across rounds.
4. Holds **concrete strategy objects** — one `AttentionStrategy` and one `AttendanceStrategy` instance, injected at construction.

### 3.2 Dataclass Design

```python
@dataclass
class ArchetypeConfig:
    """Defines the prior distributions for a behavioural archetype."""
    name: str                          # e.g. "casual", "binger", "critic"

    # Softmax sampling temperature (controls diversity of item selection)
    tau_mean: float                    # mean of N(μ, σ) to sample τ per user
    tau_std: float

    # Minimum preference score for an item to be eligible for sampling
    score_floor_mean: float
    score_floor_std: float

    # Action logit weights: each action has an intercept + score weight
    # P(action | score) = sigmoid(b + w * score)
    action_intercepts: dict[str, float]   # keys: "watch", "rate", "add_to_list"
    action_weights: dict[str, float]

    # Online preference update learning rate
    lr_mean: float
    lr_std: float

    # Attention strategy class name and default kwargs
    attention_strategy: str               # e.g. "LinearDecay"
    attention_kwargs: dict

    # Attendance strategy class name and default kwargs
    attendance_strategy: str              # e.g. "LogisticAttendance"
    attendance_kwargs: dict

    # Attendance baseline logit prior (sampled per user)
    baseline_logit_mean: float
    baseline_logit_std: float
```

```python
@dataclass
class AgentPersona:
    """Per-user state and behavioural configuration."""

    # Identity
    user_id: int
    archetype: str

    # Fixed traits (sampled from archetype priors at init, then frozen)
    tau: float                            # softmax temperature
    score_floor: float                    # minimum score for sampling eligibility
    action_intercepts: dict[str, float]
    action_weights: dict[str, float]
    lr: float                             # preference vector learning rate
    baseline_logit: float                 # attendance propensity intercept

    # Evolving preference state
    pref_vector: np.ndarray               # user's vector in low-dim MF space

    # Recent engagement history (for attendance EWMA)
    recent_signal_ewma: float = 0.0
    rounds_since_last_visit: int = 0
    last_attended_round: int = 0

    # Injected strategy objects (set after construction)
    attention: AttentionStrategy = field(default=None)
    attendance: AttendanceStrategy = field(default=None)
```

### 3.3 Initialization

```python
def build_persona(
    user_id: int,
    archetype_cfg: ArchetypeConfig,
    env: Environment,
    rng: np.random.Generator,
) -> AgentPersona:
    """
    Sample traits from archetype priors, initialize preference vector from
    the user's training history, and inject strategy objects.
    """
    tau = rng.normal(archetype_cfg.tau_mean, archetype_cfg.tau_std)
    score_floor = rng.normal(...)
    lr = rng.normal(...)
    baseline_logit = rng.normal(archetype_cfg.baseline_logit_mean, ...)

    # Preference vector: user's factor from the small independent MF
    pref_vector = env.get_user_pref_factor(user_id)   # may be None → zero init

    attention = ATTENTION_REGISTRY[archetype_cfg.attention_strategy](
        **archetype_cfg.attention_kwargs
    )
    attendance = ATTENDANCE_REGISTRY[archetype_cfg.attendance_strategy](
        **archetype_cfg.attendance_kwargs
    )

    return AgentPersona(
        user_id=user_id, archetype=archetype_cfg.name,
        tau=tau, score_floor=score_floor, lr=lr,
        baseline_logit=baseline_logit, pref_vector=pref_vector,
        attention=attention, attendance=attendance,
        action_intercepts=archetype_cfg.action_intercepts,
        action_weights=archetype_cfg.action_weights,
    )
```

### 3.4 `persona.act()` — the acceptance step

`act()` replaces `runner`'s `ranked.top_n(still_needed)`. It lives on the persona because it depends entirely on persona-level state (τ, score_floor, action weights, attention budget).

```python
def act(
    self,
    ranked: ItemList,
    item_factors: dict[int, np.ndarray],   # from env.get_user_pref_item_factors()
) -> list[tuple[int, str, float]]:
    """
    1. Filter candidates below score_floor.
    2. Convert scores to softmax probabilities (temperature τ).
    3. Sample k_effective items without replacement.
    4. For each sampled item, draw an action from the logistic action model.
    5. Return (movie_id, action, signal_strength) for each acted-on item.
    """
```

`k_effective = self.attention.effective_k(list_size)` — budget-limited window (§5).

Items drawn with action `"ignore"` are still recorded as seen (via `recommender.mark_sent`), but do not appear in the returned interaction list and do not feed the recommender or update the preference vector.

### 3.5 `persona.update_preference()` — online vector update

Called by the runner after the round's interaction list is assembled:

```python
def update_preference(
    self,
    interactions: list[tuple[int, str, float]],
    item_factors: dict[int, np.ndarray],
) -> None:
    for movie_id, action, signal in interactions:
        if movie_id not in item_factors:
            continue
        item_vec = item_factors[movie_id]
        # Gradient step toward the item, scaled by signal
        self.pref_vector += self.lr * signal * item_vec
    # Renormalize to unit length
    norm = np.linalg.norm(self.pref_vector)
    if norm > 0:
        self.pref_vector /= norm
```

---

## 4. User Preference Model (Small Independent MF)

### 4.1 Rationale

The LensKit `BiasedMF` (64 dims) represents the **platform's view** of users and items. The user's internal preference model is intentionally **separate and smaller** (5–10 dims). This reflects the idea that a user's own mental model of their tastes is a coarser approximation than the platform's aggregate statistical model, and evolves through their own interactions rather than population-level signals.

### 4.2 Factorization

- **Input data:** Same training ratings as the recommender (`train_ratings`).
- **Method:** Truncated SVD / NMF via `sklearn.decomposition` — no LensKit. Fast, no additional dependencies.
- **Dimensions:** Configurable via `SimConfig.user_pref_features` (default: 8).
- **Output:**
  - Item factor matrix `M ∈ ℝ^{|items| × d}` — computed once, cached to disk (new ChromaDB collection `user_pref_item_factors` or a `.npz` file alongside `user_factors.npz`).
  - User factor matrix `U ∈ ℝ^{|users| × d}` — used only for initialization of `persona.pref_vector`; after init the per-user vector lives entirely inside the persona.

### 4.3 Scoring

When `agent.evaluate()` scores candidates:
- It retrieves `item_factors[mid]` from `env.get_user_pref_item_factors(movie_ids)`.
- It computes `score_i = cosine(persona.pref_vector, item_factors[mid])`.
- Returns these scores as the `ItemList` score field.

This replaces the current use of the LensKit item factors for agent scoring.

### 4.4 Online Update

After `persona.act()`, the runner calls `persona.update_preference(interactions, item_factors)`.

```
u_new = u_old + lr * Σ_i (signal_i * v_i)
u_new = u_new / ‖u_new‖₂
```

where `v_i` is the item vector for movie `i` and `signal_i ∈ [0, 5]` is the action signal strength.  
Item factor vectors `v_i` are **fixed** — only the user vector moves.

---

## 5. Action Model

### 5.1 Softmax Sampling

Given scored candidates (scores ∈ ℝ):

```
eligible         = {i : score_i ≥ persona.score_floor}
p_i              = exp(score_i / τ) / Σ_j exp(score_j / τ)   for i ∈ eligible
selected_indices = multinomial_sample(p, k=k_effective, replace=False)
```

- `τ = persona.tau`: lower → more deterministic (exploit); higher → more diverse (explore).
- `score_floor = persona.score_floor`: hard preference floor, items below it are never sampled.
- If `|eligible| < k_effective`, all eligible items are selected.

### 5.2 Action Selection

For each selected item `i`:

```
P(action=a | score_i) = sigmoid(b_a + w_a * score_i)   for a ∈ {watch, rate, add_to_list}
```

Actions are drawn in priority order: `watch` checked first, then `rate`, then `add_to_list`. If none trigger, the action is `ignore`.

Parameters `b_a`, `w_a` are **persona-level traits** sampled from archetype priors at init — they do not change during simulation.

### 5.3 Signal Strengths

| Action | Signal to recommender | Signal to preference update |
|---|---|---|
| `watch` | `SimConfig.watch_signal` (default 4.5) | Same |
| `rate` | Sampled: `1 + 4 * Beta(α=score*α_max, β=(1-score)*β_max)` | Same |
| `add_to_list` | `SimConfig.add_to_list_signal` (default 3.0) | Same |
| `ignore` | None (not sent to recommender) | None (not used for pref update) |

---

## 6. Attention Budget

### 6.1 Interface

```python
class AttentionStrategy(ABC):
    @abstractmethod
    def effective_k(self, list_size: int, current_budget: float) -> int:
        """Items the agent will actively evaluate from this batch."""

    @abstractmethod
    def deplete(self, list_size: int, current_budget: float) -> float:
        """Return new budget after processing list_size items."""

    @abstractmethod
    def restore(self, end_of_round_budget: float, satisfaction_signal: float) -> float:
        """Return new budget at the start of the next round."""
```

### 6.2 Built-in Implementations

| Class | `effective_k` | `deplete` | `restore` |
|---|---|---|---|
| `LinearDecayAttention` | `round(budget × list_size)` | `budget - decay_rate × list_size` | full or partial |
| `ExponentialDecayAttention` | `round(budget × list_size)` | `budget × exp(-decay_rate × list_size)` | full or partial |
| `PerRequestAttention` | `round(budget × list_size)` | `budget - decay_rate` (flat per request) | full or partial |

All three cap budget at `[0, 1]`. The recovery behaviour is controlled by a separate `recovery` parameter: `"full"` resets to 1.0; `"partial"` adds `recovery_rate`; `"satisfaction"` resets fully if `satisfaction_signal > sat_threshold`, else adds `recovery_rate`.

### 6.3 Budget State Location

Budget state (`current_budget: float`) lives on the `AgentPersona`. The strategy object is stateless — it only computes the new value given the current one. This makes both serializable independently.

```python
# In runner, per request:
k_eff = persona.attention.effective_k(len(candidates), persona.budget)
persona.budget = persona.attention.deplete(len(candidates), persona.budget)

# At end of round:
persona.budget = persona.attention.restore(persona.budget, mean_signal)
```

---

## 7. Attendance Model

### 7.1 Interface

```python
class AttendanceStrategy(ABC):
    @abstractmethod
    def will_attend(
        self,
        baseline_logit: float,
        recent_signal_ewma: float,
        rounds_since_last_visit: int,
        rng: np.random.Generator,
    ) -> bool:
        """Return True if the user visits the recommender this round."""

    @abstractmethod
    def update_ewma(
        self,
        current_ewma: float,
        new_signal: float,
        alpha: float,
    ) -> float:
        """Update and return the EWMA of recent satisfaction signals."""
```

### 7.2 Built-in Implementations

| Class | Attendance probability |
|---|---|
| `LogisticAttendance` | `sigmoid(baseline + β_sat × ewma + β_rec × clip(bonus - penalty × absent_rounds, -3, 3) + Gumbel(0, scale))` |
| `ThresholdAttendance` | Attends if `ewma > threshold`, else with probability `fallback_prob` |
| `AlwaysAttend` | Always returns `True` (useful baseline / ablation) |
| `BernoulliAttendance` | Fixed independent Bernoulli with probability `attend_prob` |

The `LogisticAttendance` is the most behaviourally expressive and is the recommended default. `AlwaysAttend` and `BernoulliAttendance` are useful for ablation studies.

### 7.3 State Update (per round, after interactions)

```python
# In runner, after all interactions for uid:
mean_sig = mean([sig for _, _, sig in interactions]) if interactions else 0.0
persona.recent_signal_ewma = persona.attendance.update_ewma(
    persona.recent_signal_ewma, mean_sig, alpha=sat_ewma_alpha
)
if attended:
    persona.rounds_since_last_visit = 0
    persona.last_attended_round = rnd
else:
    persona.rounds_since_last_visit += 1
```

---

## 8. Updated `AbstractAgent` Interface

The agent class handles **scoring only** — the preference-driven scoring of candidates. All behavioural mechanics (sampling, action, attention, attendance) live on the persona.

```python
class AbstractAgent(ABC):

    @abstractmethod
    def evaluate(
        self,
        candidates: ItemList,
        persona: AgentPersona,
        item_factors: dict[int, np.ndarray],
    ) -> ItemList:
        """
        Score candidates using persona.pref_vector and item_factors.
        Must attach a 'score' field to the returned ItemList.
        """

    @abstractmethod
    def update(
        self,
        user_id: int,
        interactions: list[tuple[int, str, float]],
    ) -> None:
        """
        Update any agent-level (not persona-level) state.
        For most agents this will be a no-op; persona.update_preference()
        handles the user-side vector update in the runner.
        """
```

Note: `agent.evaluate` no longer calls `env.get_user_factor()` directly. It receives `persona.pref_vector` (via the persona argument) and the item factors for small-MF scoring. This makes every agent implementation use the **same preference space** and the same scoring logic. Agent subclasses differ only in *what else* they feed into the score (e.g. `SemanticAgent` may blend the preference score with a semantic similarity component).

---

## 9. Population Mix

### 9.1 Config

```python
# In SimConfig:
archetype_mix: dict[str, float] = field(
    default_factory=lambda: {"casual": 1.0}
)
# Keys are archetype names, values are proportions (will be normalized).
# e.g. {"casual": 0.6, "binger": 0.4}

archetype_configs: dict[str, ArchetypeConfig] = field(
    default_factory=lambda: {
        "casual": CASUAL_ARCHETYPE_DEFAULT,
        "binger": BINGER_ARCHETYPE_DEFAULT,
        "critic": CRITIC_ARCHETYPE_DEFAULT,
    }
)
```

### 9.2 Assignment

At simulation start, eval users are assigned an archetype by sampling from `archetype_mix` proportions (with `random_seed` for reproducibility):

```python
archetypes_assigned = rng.choice(
    list(archetype_mix.keys()),
    size=len(eval_users),
    p=normalized_proportions,
)
```

Each user's assigned archetype determines which `ArchetypeConfig` is used to build their `AgentPersona`.

### 9.3 Pre-defined Archetype Defaults

| Archetype | τ | score_floor | Dominant action | Attention model |
|---|---|---|---|---|
| `casual` | 1.0 | -0.5 | add_to_list | `LinearDecay` moderate |
| `binger` | 0.5 | 0.0 | watch | `ExponentialDecay` slow |
| `critic` | 0.8 | 0.2 | rate | `PerRequest` aggressive |

These defaults are stored in `sim/archetypes.py` and can be overridden in `SimConfig`.

---

## 10. Updated `SimConfig` Fields

New fields to add (existing fields unchanged):

```python
# ── User preference model ──────────────────────────────────────────────────
user_pref_features: int = 8            # dims for the small independent MF

# ── Action model ──────────────────────────────────────────────────────────
watch_signal: float = 4.5
add_to_list_signal: float = 3.0
beta_alpha_max: float = 8.0            # Beta distribution shape
beta_beta_max: float = 8.0

# ── Attention (defaults; may be overridden per archetype) ─────────────────
default_attention_strategy: str = "LinearDecay"
default_attention_decay_rate: float = 0.15
default_attention_recovery: str = "full"
default_attention_recovery_rate: float = 0.5

# ── Attendance (defaults; may be overridden per archetype) ────────────────
default_attendance_strategy: str = "LogisticAttendance"
sat_ewma_alpha: float = 0.4
attend_recency_bonus: float = 0.3
attend_recency_penalty: float = 0.1
attend_noise_scale: float = 0.2

# ── Population mix ────────────────────────────────────────────────────────
archetype_mix: dict[str, float] = field(default_factory=lambda: {"casual": 1.0})
```

---

## 11. New and Modified Modules

| Module | Status | Description |
|---|---|---|
| `sim/config.py` | modify | Add fields from §10 |
| `sim/environment.py` | modify | Add `user_pref_item_factors` collection (small MF item side) |
| `sim/recommender.py` | modify | `update_user` accepts `list[tuple[int, float]]` |
| `sim/persona.py` | **new** | `AgentPersona`, `ArchetypeConfig`, `build_persona()`, `build_population()` |
| `sim/archetypes.py` | **new** | Default archetype constants (`CASUAL_ARCHETYPE_DEFAULT`, etc.) |
| `sim/attention.py` | **new** | `AttentionStrategy` ABC + `LinearDecay`, `ExponentialDecay`, `PerRequest` |
| `sim/attendance.py` | **new** | `AttendanceStrategy` ABC + `LogisticAttendance`, `ThresholdAttendance`, `AlwaysAttend`, `BernoulliAttendance` |
| `sim/agents/base.py` | modify | Updated `evaluate` signature; `update` signature |
| `sim/agents/associative.py` | modify | Scoring via `persona.pref_vector` + small-MF item factors |
| `sim/runner.py` | modify | Integrate persona population, attendance gate, `persona.act()`, signal assembly |

---

## 12. Updated `sim/runner.py` Loop

```python
# Init
population = build_population(cfg, env, rng)   # dict[user_id → AgentPersona]
agent = _build_agent(cfg, env)
recommender = Recommender(cfg, env)

for rnd in 1..num_rounds:
    if rnd > 1:
        recommender.retrain()

    for uid in env.eval_users:
        persona = population[uid]

        # ── Attendance gate ──────────────────────────────────────────────
        if not persona.attendance.will_attend(
            persona.baseline_logit,
            persona.recent_signal_ewma,
            persona.rounds_since_last_visit,
            rng,
        ):
            persona.rounds_since_last_visit += 1
            continue  # user skips this round

        # ── Reset per-round budget ────────────────────────────────────────
        # (happens inside restore(), called at end of previous round)

        round_interactions: list[tuple[int, str, float]] = []

        # ── Inner re-request loop ─────────────────────────────────────────
        for req in 1..max_requests_per_round:
            if persona.budget <= 0:
                break

            candidates = recommender.recommend(uid, n=rec_list_size)
            if len(candidates) == 0:
                break

            recommender.mark_sent(uid, candidates)

            # Agent scores candidates using persona.pref_vector
            item_factors = env.get_user_pref_item_factors(candidates.ids())
            ranked = agent.evaluate(candidates, persona, item_factors)

            # Persona samples + selects action
            new_interactions = persona.act(ranked, item_factors)
            persona.budget = persona.attention.deplete(len(candidates), persona.budget)

            round_interactions.extend(new_interactions)

            # Stop if enough acted-on items (excluding "ignore")
            acted = [x for x in round_interactions if x[1] != "ignore"]
            if len(acted) >= accept_k:
                break

        # ── Post-round updates ────────────────────────────────────────────
        acted = [(mid, act, sig) for mid, act, sig in round_interactions if act != "ignore"]

        # Recommender feedback
        rec_signals = [(mid, sig) for mid, _, sig in acted]
        recommender.update_user(uid, rec_signals)

        # Preference vector update
        persona.update_preference(acted, item_factors_for_all_acted)

        # Agent-level update (if the agent type has its own state)
        agent.update(uid, acted)

        # Attendance / attention state
        mean_sig = mean([sig for _, _, sig in acted]) if acted else 0.0
        persona.recent_signal_ewma = persona.attendance.update_ewma(
            persona.recent_signal_ewma, mean_sig, sat_ewma_alpha
        )
        persona.rounds_since_last_visit = 0
        persona.last_attended_round = rnd
        persona.budget = persona.attention.restore(persona.budget, mean_sig)

    recommender.advance_round()
    # log metrics ...
```

---

## 13. Additional MLflow Metrics (per round)

| Metric | Description |
|---|---|
| `attendance_rate` | Fraction of eval users who visited this round |
| `mean_attention_consumed` | `1 - mean(end_budget)` across attending users |
| `mean_requests_per_user` | Average number of re-requests per attending user |
| `action_watch_frac` | Fraction of interactions that were "watch" |
| `action_rate_frac` | Fraction that were "rate" |
| `action_addlist_frac` | Fraction that were "add_to_list" |
| `mean_signal_strength` | Mean signal across all non-ignore interactions |
| `churn_rate` | Fraction of users absent for ≥ 2 consecutive rounds |
| `pref_vector_drift` | Mean cosine distance between each user's pref vector now vs. round 1 |

---

## 14. Test Plan (v2 additions)

### `tests/test_persona.py`
- `build_persona()` returns correct type and all fields populated.
- Traits are sampled differently for distinct archetypes (with fixed seed).
- Preference vector has correct dimensionality.
- `persona.act()` returns list of `(int, str, float)` tuples.
- Action strings are all in `{"watch", "rate", "add_to_list", "ignore"}`.
- Signal strengths are within `[0.0, 5.0]`.
- With `score_floor=100.0` (impossibly high), `act()` returns only `ignore` actions.
- With zero budget, `act()` returns empty list.
- `update_preference()` changes the pref vector.
- `update_preference()` with zero signal leaves vector unchanged (up to normalization).
- Renormalization: pref vector has unit norm after update.

### `tests/test_attention.py`
- Full budget at init.
- `effective_k` ≤ list_size.
- `effective_k` is monotonically non-increasing as budget decreases.
- `deplete()` returns value strictly lower than input.
- Budget is clamped to `[0, 1]`.
- `LinearDecay`, `ExponentialDecay`, `PerRequest` produce distinct depletion curves.
- `restore("full")` returns 1.0 regardless of end budget.
- `restore("partial")` respects `recovery_rate`.
- `restore("satisfaction")` triggers full recovery above threshold only.

### `tests/test_attendance.py`
- `will_attend` returns `bool`.
- With extremely high `baseline_logit`, `LogisticAttendance` attends consistently.
- With extremely low `baseline_logit`, `AlwaysAttend` still attends.
- `BernoulliAttendance(attend_prob=0.0)` always returns `False`.
- `update_ewma` converges toward the true mean after many updates.
- `rounds_since_last_visit` increments correct on absence.

### `tests/test_user_pref_model.py`
- `env.get_user_pref_item_factors()` returns dict with correct dimensionality.
- Item factors have unit norm (if normalized) or consistent scale.
- Collection exists and has entries after `Environment` init.

### `tests/test_recommender_v2.py`
- `update_user(uid, [(mid, signal)])` stores correct signal in `_feedback`.
- Higher-signal interaction produces higher stored rating.
- `retrain()` runs without error with differentiated signals.

---

## 15. Implementation Order

| Phase | Scope | Prerequisite |
|---|---|---|
| **A** | `sim/config.py` — new fields | None |
| **B** | `sim/environment.py` — small MF item factors collection | A |
| **C** | `sim/attention.py` — strategy ABC + 3 built-ins + tests | A |
| **D** | `sim/attendance.py` — strategy ABC + 4 built-ins + tests | A |
| **E** | `sim/persona.py` + `sim/archetypes.py` — persona + 3 archetypes + tests | B, C, D |
| **F** | `sim/recommender.py` — update `update_user` signature + tests | A |
| **G** | `sim/agents/base.py` + `sim/agents/associative.py` — updated signatures | E |
| **H** | `sim/runner.py` — full integration | E, F, G |
| **I** | `tests/` — migrate + extend existing tests, add new suites | H |

---

## 16. Open Design Questions

1. **NMF vs SVD for the small MF:** NMF produces non-negative factors (more interpretable, never negative cosine similarity), while SVD has a cleaner closed-form init. Either works at 5–10 dims. Default to SVD (`TruncatedSVD`) for speed and reproducibility; can be swapped via `SimConfig.user_pref_method`.

2. **`ignore` accounting:** Items where the action is `ignore` are still recorded as `mark_sent`. Should they also go into `_all_seen` via `recommender.update_user`? Probably not — only acted-on items should affect the recommender. But ignoring the item does mean the agent's preference vector doesn't update toward it, which is the desired behaviour.

3. **Cold-start users:** Users absent from the small MF's training set will have a zero-initialized preference vector. A fallback is to use the centroid of item factors as the initial vector, with noise. Should this be configurable?

4. **Population-level emergent metrics:** Beyond per-round aggregates, candidate statistics for studying emergent dynamics include: Gini coefficient of recommendation exposure across eval users, inter-user cosine diversity of accepted item sets (does the population converge on the same items?), and the spectral gap of the user-item interaction graph (captures filter-bubble tightening).

5. **Agent subtype scorings:** `SemanticAgent` and `LLMAgent` stubs may want to **blend** the small-MF preference score with content-based similarity. The scoring interface (§8) allows this since `evaluate()` receives both `persona.pref_vector` and `item_factors`. The blending coefficient could be a persona trait.
