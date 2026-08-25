# Simulation Metrics Audit

**Project:** phd/llm_testing
**Date:** 2026-06-21
**Auditor:** metrics-audit-k7 scout agent

---

> **Provenance and staleness note, added 2026-08-25.**
>
> This document was never committed.
> It was found untracked in a detached-HEAD worktree at `03fd7bd` during branch cleanup and rescued verbatim; the body below is exactly as written on 2026-06-21 and has not been edited.
>
> **It describes the codebase as of 2026-06-21 and several of its premises are now out of date.**
> Read it as a record of the reasoning, not as a description of current state.
>
> - Every metric name below is in the retired flat form. All MLflow keys are now slash-namespaced (`meta/`, `ranking/`, `sim/`, `popularity/`, `score/`, `correlation/`, `error/`, `rating/`). See `docs/METRIC_DICTIONARY.md`.
> - It states that only `AssociativeAgent` is implemented. `ResidualProfileAgent`, `ItemItemNeighborhoodAgent` and `LLMAgent` have since landed. `SemanticAgent` and `Seq2SeqAgent` are still stubs.
> - It notes that no `reports/` or configs directory existed. Both do now.
> - Its KEEP/DROP/CONSOLIDATE recommendations were never applied. The metric set was renamed rather than trimmed, so the redundancies it identifies may still be present.
>
> Its most substantive open claim is the "critical gap": preference vector drift is never measured, despite being the core dynamic of the simulation. That remains true, and a related item sits in the project backlog.

---

## What I Did

Read every metric-emitting line in:
- `sim/runner.py` (full) - `_run_round` and `_run_recommender_only`
- `sim/config.py` - all behavioral parameters
- `sim/persona.py`, `sim/attention.py`, `sim/attendance.py`, `sim/archetypes.py` - behavioral model
- `sim/user_agent.py` and all `sim/agents/` implementations
- `tests/test_runner.py` - to confirm which metrics are consumed downstream

No experiment configs directory or `reports/` directory existed at time of audit.
The only agent implemented is `AssociativeAgent`; `SemanticAgent`, `Seq2SeqAgent`, and `LLMAgent` are stubs that raise `NotImplementedError`.

The research question under evaluation is: **does the choice of simulated user agent change what we learn about the recommender?**

---

## Metric Inventory

### Full Simulation Mode (`_run_round`)

| # | Name | Location | Verdict | Reason |
|---|------|----------|---------|--------|
| F1 | `attendance_rate` | runner.py:298 | **KEEP** | Direct research signal - agent type → different quality scores → different satisfaction EWMAs → different attendance. Varies round to round and across agent types. |
| F2 | `hit_rate` | runner.py:299 | **DROP** | Mislabeled and subsumed by NDCG (see sharp edge #1). Computes `\|rec ∩ held_out\| / \|held_out\|` on first-batch only - that is recall@1batch, not hit rate. NDCG@k strictly subsumes this. |
| F3 | `ndcg_at_{rec_list_size}` | runner.py:300 | **KEEP** | Primary ranking quality metric. Captures both relevance and position. Agent-sensitive once recommender retrains on feedback. |
| F4 | `holdout_recall` | runner.py:301 | **KEEP** | Cumulative coverage - `\|surfaced ∩ held_out\| / \|held_out\|` across rounds. Distinct from NDCG: measures exploration breadth vs. ranking quality. Agent-sensitive via recommender retrain diversity. |
| F5 | `mean_signal_strength` | runner.py:302 | **CONSOLIDATE** | Partially derivable: `watch_frac × 4.5 + addlist_frac × 3.0 + rate_frac × E[rate_signal]`. Not algebraically exact (rate_signal is Beta-sampled) but 80% redundant with action fractions. Keep only if modeling feedback noise. |
| F6 | `mean_attention_consumed` | runner.py:303 | **DROP** | Decay rate is archetype-fixed, and casual + binger both use `"full"` recovery (budget always resets to 1.0). Metric is therefore almost entirely determined by archetype mix and `max_requests_per_round`, not agent type. Adds no signal for the research question. |
| F7 | `action_watch_frac` | runner.py:304 | **KEEP** | Action mix is a direct behavioral readout. Agent type → different relevance scores → different action distributions. |
| F8 | `action_rate_frac` | runner.py:308 | **KEEP** | Independent from watch_frac. Rate interactions also produce noisy feedback signal - qualitatively different from watch. |
| F9 | `action_addlist_frac` | runner.py:312 | **DROP** | **Algebraically derivable**: `1 - action_watch_frac - action_rate_frac`. Always true since the three fractions sum to 1 by construction (runner.py:296-312). Zero independent information. |

### Recommender-Only Mode (`_run_recommender_only`)

| # | Name | Location | Verdict | Reason |
|---|------|----------|---------|--------|
| R1 | `user_count` | runner.py:423 | **DROP** | Pure sanity check. Fixed for a given config. Should be a parameter/tag, not a metric. |
| R2 | `hit_rate` | runner.py:424 | **DROP** | Same issue as F2. Redundant with NDCG. |
| R3 | `ndcg_at_{rec_list_size}` | runner.py:425 | **KEEP** | Baseline ranking quality before any agent feedback. |
| R4 | `fraction_users_with_holdout_hit` | runner.py:428 | **DROP** | Coarser binary version of hit_rate (which is itself dropped). Doubly redundant. |
| R5 | `mean_user_recommended_popularity_mean` | runner.py:431 | **KEEP** | Core signal: popularity bias of raw recommender. Directly answers whether recommender skews toward popular items. |
| R6 | `mean_user_recommended_popularity_std` | runner.py:435 | **DROP** | Within-user variance of recommendation popularity. Not interpretable in isolation; adds noise without signal for the research question. |
| R7 | `mean_user_heldout_popularity_mean` | runner.py:438 | **KEEP** | Needed as the baseline for interpreting `popularity_mean_delta`. Describes test set characteristics, not recommender behavior. |
| R8 | `mean_user_heldout_popularity_std` | runner.py:441 | **DROP** | Std of test set popularity; describes data distribution, not any research question. |
| R9 | `mean_user_comparison_popularity_mean` | runner.py:447 | **DROP** | **Exact duplicate of R7**. Confirmed by test_runner.py:94 - `allclose(heldout_popularity_mean, comparison_popularity_mean)`. Present solely as a backward-compatible alias. |
| R10 | `mean_user_comparison_popularity_std` | runner.py:451 | **DROP** | **Exact duplicate of R8**. Same alias situation. |
| R11 | `mean_user_popularity_mean_delta` | runner.py:457 | **KEEP** | Key research signal: positive = recommender recommends more-popular items than users actually like. Primary measure of popularity bias. |
| R12 | `std_user_popularity_mean_delta` | runner.py:460 | **KEEP** | Cross-user dispersion of popularity bias. Answers whether bias is uniform or user-heterogeneous. |
| R13 | `mean_user_heldout_recommender_score_mean` | runner.py:465 | **KEEP** | Does the recommender's latent space assign high scores to held-out items? Core alignment diagnostic. |
| R14 | `mean_user_heldout_recommender_score_std` | runner.py:469 | **DROP** | Within-user score dispersion on held-out set. Diagnostic of discriminability, not alignment. Low value for the research question. |
| R15 | `mean_user_heldout_internal_score_mean` | runner.py:473 | **KEEP** | Baseline: how highly does the user's own internal model score held-out items? Needed to compare recommender vs. user alignment. |
| R16 | `mean_user_heldout_internal_score_std` | runner.py:477 | **DROP** | Same rationale as R14. Drop. |
| R17 | `mean_user_heldout_score_mean_gap` | runner.py:485 | **KEEP** | `recommender_score_mean - internal_score_mean` per user. Central alignment metric: positive = recommender's scoring agrees with user internal model on held-out items. |
| R18 | `std_user_heldout_score_mean_gap` | runner.py:489 | **KEEP** | Cross-user dispersion of alignment. Important: if some users get high alignment and others get none, mean alone is misleading. |
| R19 | `fraction_users_recommender_score_mean_gt_internal` | runner.py:495 | **DROP** | Coarser direction indicator already captured by sign of R17 (`mean_user_heldout_score_mean_gap`). |
| R20 | `fraction_users_recommended_mean_gt_comparison_mean` | runner.py:503 | **DROP** | Coarser binary of R11 (`mean_user_popularity_mean_delta`). |

---

## Keep List - Minimal Research Signal Set

### Full simulation mode (per round, step = round number)

```
attendance_rate            # engagement loop dynamics
ndcg_at_{rec_list_size}    # recommender ranking quality (first batch)
holdout_recall             # cumulative held-out coverage across rounds
action_watch_frac          # behavioral readout - action quality distribution
action_rate_frac           # behavioral readout - rating propensity
mean_signal_strength       # optional: keep if feedback noise is a research variable
```

`mean_signal_strength` (F5) is borderline - keep only if the project intends to study how feedback noise (from noisy `rate` interactions) affects recommender convergence. Otherwise, drop it: the action fractions already encode the mix.

### Recommender-only mode (one-shot snapshot)

```
ndcg_at_{rec_list_size}                  # baseline ranking quality
mean_user_recommended_popularity_mean    # popularity bias level
mean_user_heldout_popularity_mean        # test-set baseline (for delta interpretation)
mean_user_popularity_mean_delta          # net popularity bias
std_user_popularity_mean_delta           # heterogeneity of popularity bias
mean_user_heldout_recommender_score_mean # recommender-user alignment
mean_user_heldout_internal_score_mean    # user internal model alignment baseline
mean_user_heldout_score_mean_gap         # alignment delta (key diagnostic)
std_user_heldout_score_mean_gap          # heterogeneity of alignment
```

---

## Drop / Consolidate List

| Metric | Reason |
|--------|--------|
| F2 `hit_rate` (full mode) | Mislabeled (is recall@1batch), redundant with NDCG |
| F6 `mean_attention_consumed` | Archetype-fixed; insensitive to agent type; no research signal |
| F9 `action_addlist_frac` | Algebraically = 1 - watch_frac - rate_frac; zero independent information |
| R1 `user_count` | Constant; should be a tag/parameter |
| R2 `hit_rate` (rec-only) | Same issue as F2 |
| R4 `fraction_users_with_holdout_hit` | Binary coarsen of hit_rate (itself dropped) |
| R6 `mean_user_recommended_popularity_std` | Within-user variance; low interpretive value |
| R8 `mean_user_heldout_popularity_std` | Describes test set, not behavior |
| R9 `mean_user_comparison_popularity_mean` | Exact duplicate of R7 (`heldout_popularity_mean`) |
| R10 `mean_user_comparison_popularity_std` | Exact duplicate of R8 (`heldout_popularity_std`) |
| R14 `mean_user_heldout_recommender_score_std` | Within-user score dispersion; not a research signal |
| R16 `mean_user_heldout_internal_score_std` | Same as R14 |
| R19 `fraction_users_recommender_score_mean_gt_internal` | Direction already in sign of mean gap |
| R20 `fraction_users_recommended_mean_gt_comparison_mean` | Direction already in sign of popularity delta |

**Consolidation note**: The `comparison_*` aliases (R9, R10) in `user_df` artifact should be removed from `_run_recommender_only` at runner.py:403-407. They are documented as "backward-compatible aliases" but actively bloat the per-user parquet artifact and were presumably introduced during a rename - the old name should be fully retired.

---

## Gaps - Missing Metrics Worth Adding

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| `pref_vector_drift` | Cosine distance of `persona.pref_vector` from round 0 to current round | The preference vector is the core dynamic of the agent model. Without tracking its movement, we cannot distinguish "agent caused preference drift" from "agent caused no drift". This is arguably the most important missing metric. |
| `mean_ndcg_at_k` for multiple k | NDCG evaluated at k=1, k=3, k=10 in addition to rec_list_size | With rec_list_size=6 fixed, we lose positional detail. NDCG@1 (top pick quality) vs NDCG@6 (list quality) can diverge significantly. |
| `archetype_stratified_watch_frac` | `action_watch_frac` computed separately per archetype | Bingers are designed to watch more; critics to rate more. Aggregate fractions mix archetypes and may show no change across agent types even when within-archetype behavior changes significantly. |
| `archetype_stratified_attendance_rate` | `attendance_rate` per archetype | Archetype-level engagement dynamics are invisible in the aggregate. ThresholdAttendance (critics) behaves very differently from LogisticAttendance (casual/binger). |
| `requests_per_attending_user` | Mean number of recommendation requests made before reaching `accept_k` interactions | Shows how selective the agent is. A more discerning agent may need more re-requests. Currently this loop state is lost after `_run_request_loop`. |
| `intra_list_diversity` | Mean pairwise cosine distance between recommended items' content embeddings | Whether the recommender converges toward repetitive recommendations over rounds. Agent-sensitive via feedback diversity. |
| `feedback_noise_ratio` | Fraction of total interactions that are `rate` (noisy Beta signal) vs. deterministic (`watch`, `add_to_list`) | Different agents may produce radically different rate/watch splits (e.g. critics rate everything). This determines how noisy the signal fed back to the recommender is, affecting its convergence. (This is action_rate_frac, but framing it as "noise ratio" draws attention to its modeling significance.) |

---

## Sharp Edges in Metric Computation

### 1. `hit_rate` is actually `recall@first_batch` (mislabeled, full mode)

**File:** runner.py:70-74, runner.py:278-279

```python
def _hit_rate(recs: ItemList, held_out_ids: set[int]) -> float:
    ...
    return len(rec_ids & held_ids) / len(held_out_ids)  # denominator is held_out size
```

The standard definition of hit rate is binary: did any recommended item hit? Or sometimes `|rec ∩ relevant| / |rec|` (precision). This implementation divides by `|held_out|`, which is recall. With `rec_list_size=6` and a typical held-out set of 10-50 items, max possible value is 0.12-0.60. Researchers reading "hit_rate=0.03" will likely misinterpret this as "only 3% of recommendation slots are hits" when it actually means "3% of held-out items were covered by the first 6 recommendations." Rename to `holdout_recall_at_{rec_list_size}` or fix to the standard denominator.

### 2. NDCG and hit_rate computed only on request 1 (full mode)

**File:** runner.py:272-279

```python
first_batch = ItemList(
    item_ids=np.array(
        [r["movieId"] for r in recs_rows if r["request"] == 1],  # request==1 only
        dtype=np.int64,
    )
)
hit_rates.append(_hit_rate(first_batch, held_ids))
ndcgs.append(_ndcg_at_k(first_batch, held_ids, cfg.rec_list_size))
```

Users can make up to `max_requests_per_round=3` requests. NDCG and hit_rate ignore subsequent requests entirely. Meanwhile, `holdout_recall` correctly tracks cumulative coverage across all requests (runner.py:595). This asymmetry means NDCG reflects only the initial recommender ranking quality before any within-round feedback, which may actually be intentional - but it is undocumented and could be surprising.

### 3. `action_addlist_frac` is exactly redundant (always sum-to-one)

**File:** runner.py:296-312

```python
total_actions = sum(action_counts.values())
metrics = {
    ...
    "action_watch_frac": action_counts["watch"] / total_actions if total_actions else 0.0,
    "action_rate_frac": action_counts["rate"] / total_actions if total_actions else 0.0,
    "action_addlist_frac": action_counts["add_to_list"] / total_actions if total_actions else 0.0,
}
```

Three fractions that sum to 1, all three logged. One is always derivable. Not harmful but wastes MLflow space and can confuse analysts who see "three independent behavioral dimensions" when there are only two degrees of freedom.

### 4. Backward-compatible aliases double the user artifact (rec-only mode)

**File:** runner.py:403-407

```python
# Backward-compatible aliases for earlier artifact names.
"comparison_item_count": len(cmp_ids),
"comparison_popularity_mean": cmp_mean,
"comparison_popularity_std": cmp_std,
```

`comparison_*` are confirmed by test_runner.py:94 to be numerically identical to `heldout_*`. The parquet artifact carries both. These should be retired: if old notebooks reference `comparison_popularity_mean`, update them; don't carry dead weight in every artifact.

### 5. `ndcg_eval_ks` does not exist in `SimConfig`

The audit task referenced `ndcg_eval_ks` as a config parameter. It does not exist in `sim/config.py`. The only NDCG k is implicit in `rec_list_size`. There is no multi-k NDCG evaluation. This may be a planned parameter that was never implemented.

### 6. Action fraction denominator is volume-weighted, not user-weighted

**File:** runner.py:296-312

`action_counts` aggregates across all attending users. A user who interacts 10 times (binger) has 10x the weight of a user who interacts once. The result is a volume-weighted population average, not a user-level average. For a 70/20/10 casual/binger/critic mix where bingers interact most, the aggregate fractions will be binger-dominated even if bingers are only 20% of users. Archetype stratification would reveal this distortion.

### 7. Attention consumption metric is circular for full-recovery archetypes

**File:** runner.py:596, attention.py:119-127

For archetypes with `recovery="full"` (casual, binger - 90% of the default population), `attention.restore()` always returns 1.0. Budget always starts at 1.0. `mean_attention_consumed` therefore measures `max(0, 1.0 - end_budget_after_requests)`, which is fully determined by `decay_rate × (list_size × num_requests)`. Since these are archetype-fixed, the metric is nearly constant for a given archetype mix and provides no signal about agent type.

---

## Summary

The current metric set has three structural problems:

1. **Three confirmed redundancies**: `action_addlist_frac` is always `1 - watch - rate`; `comparison_*` are identical aliases of `heldout_*`; `fraction_users_*` metrics are coarser versions of their mean equivalents.

2. **One mislabeled metric**: `hit_rate` in full mode computes `recall@first_batch` not hit rate. The denominator bug will produce misleadingly small values and incorrect interpretation.

3. **One critical gap**: preference vector drift is never measured, yet it is the primary internal state that changes between rounds and constitutes the core dynamic of the simulation. Without it, we cannot distinguish "agent type X changed user preferences" from "agent type X changed recommender behavior directly."

The minimal keep set above reduces from 9 full-mode metrics to 5 (or 6 with signal_strength) and from 21 rec-only metrics to 9, cutting MLflow cardinality roughly in half while preserving all distinct research signals.
