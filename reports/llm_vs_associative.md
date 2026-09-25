# LLM agent vs associative baseline: held-out rating prediction

Experiment: `llm-agent-comparison` (`sqlite:///mlflow.db`)
Script: `experiments/llm_vs_associative.py`
LLM arms run 2026-06-26. Corrected baselines run 2026-08-25. Bias-only null run 2026-09-24.
Last updated: 2026-09-24

---

## Summary

> **Read this first (2026-09-24).**
> A bias-only null, which predicts `global + user + item bias` and nothing else, beats **every** arm in this report, the associative baseline included.
> On the matched recent-5 pairs it scores 0.7359 MAE at 128 users and 0.6951 at 2566 users.
> So the honest reading is not "the associative baseline beats the LLM" but "neither arm adds anything over a bias table, and both are worse than one".
> The associative arm loses because its dot term is on the wrong scale and adds a systematic +0.39 stars, not because latent factors are uninformative.
> See [The bias-only null](#the-bias-only-null) below.
> The comparisons further down are still correct as comparisons between those arms; what changed is what they mean.

Five LLM prompt variants and an associative latent-factor baseline were asked the same question: given a user's rating history, predict the rating they gave to a held-out movie.

On a matched evaluation set the associative baseline beats every LLM arm, by **0.122 MAE** against the best of them.

That gap is smaller than it first appeared.
The raw MLflow table showed a 0.209 gap, but the two sides were not scored on the same items, and roughly 42% of the apparent difference was an artefact of that.
The remaining difference is real and large enough to survive the noise at this sample size.

Two things this report does **not** establish.
The ranking among the five LLM arms is not resolvable: they span 0.072 MAE and a single arm's 95% interval is about that wide.
The recency effect described below has a point estimate of 0.047 MAE but a confidence interval spanning zero, so it is a direction worth re-testing, not a measured bias.

---

## What was measured

Both agents predict a rating in `[1, 5]` and are scored by MAE and RMSE against the rating the user actually gave.

The associative prediction is a `bias + dot` reconstruction:
`env.get_rating_bias(uid, mid) + dot(pref_vector, item_factor)`, clipped to `[1, 5]`.
This is the decomposition the model was fitted on, rather than the affine `a * dot + b` formula.

The LLM receives `k` examples of movies the user rated, each with title, genres, overview and the rating given, and predicts a rating for the held-out item.
**No archetype or persona information appears in the prompt.**
This is deliberate: it matches the information boundary of the associative agent, which sees only a preference vector derived from training-set ratings.
The LLM therefore infers preferences from content alone.

Model: `Qwen2.5-7B-Instruct-4bit`, run locally through `mlx-lm`, greedy decoding.

---

## The confound, and why the original table could not be read

`evaluate_llm` sliced `held_ids[:max_items_per_user]` and the sweep ran with `--max-items 5`.
`evaluate_associative` had no cap and scored the full held-out set.
So the baseline was scored on 5659 items and each LLM arm on 640, and the MAE gap between them was partly a difference in evaluation set rather than a difference in agent.

Both arms now select items through a single function, `select_held_items`, so a given `--max-items` produces identical `(user, item)` pairs on both sides by construction.
Every run logs the pairs it scored as a `scored_pairs.csv` artifact, because a matching `meta/item_count` does not prove the pairs match.

**Never report a MAE from this experiment without its `meta/item_count` and `meta/user_count` beside it.**
Reporting the item count alone is what would have caught this originally, and the user count is what catches the weighting artefact described below.

---

## Results

All rows below are 128 evaluation users, `eval_user_frac=0.001`, seed 42.

| Arm | `error/mae` | `error/rmse` | `meta/item_count` | Selection | Comparable? |
|---|---|---|---|---|---|
| associative-baseline | 0.7045 | 0.9478 | 5659 | all | No, different item set |
| associative-baseline-capped-random | 0.7449 | 1.0001 | 640 | random-5 | To each other only |
| **associative-baseline-capped** | **0.7916** | **1.0547** | **640** | first-5 | **Yes** |
| llm-top_rated-k2 | 0.9136 | 1.1999 | 640 | first-5 | Yes |
| llm-recent-k3 | 0.9161 | 1.1477 | 640 | first-5 | Yes |
| llm-top_rated-k5 | 0.9331 | 1.2279 | 640 | first-5 | Yes |
| llm-polarized-k3-no-fewshot | 0.9391 | 1.2177 | 640 | first-5 | Yes |
| llm-polarized-k2 | 0.9852 | 1.2115 | 640 | first-5 | Yes |

### The corrected comparison

The like-for-like rows are `associative-baseline-capped` and the five LLM arms.
All six scored the same 640 `(user, item)` pairs: each user's five most recent held-out ratings.

**Baseline 0.7916 against best LLM arm 0.9136, a gap of 0.122 MAE in favour of the baseline.**

The 0.209 gap implied by the raw table was inflated by the item-set mismatch.
About 42% of it was artefact and 58% was real.

The baseline's 95% interval, clustering by user, is roughly `+/- 0.072`.
The 0.122 gap is comfortably outside that, so the direction of the result is not a sampling accident.
A properly paired test against the LLM arms is not possible from the stored runs, since the 2026-06-26 runs logged aggregate metrics only and not per-item predictions.
Future arms should log per-item errors so this can be tested directly rather than argued from one side's interval.

### The uncapped row is a weighting artefact, not a better baseline

`associative-baseline` at 0.7045 looks like the strongest result in the table.
It is not comparable to anything else, for two independent reasons.

First, it scores a different and larger item set.

Second, it is a micro-average over items, so users with many held-out ratings dominate it.
Heavy raters are slightly easier to predict, `corr(held-out count, per-user MAE) = -0.115`, and held-out counts are very uneven: minimum 10, median 26, maximum 310.
Re-weighting that same run so every user counts equally gives **0.7412**, which is essentially what random-5 recovers at 0.7449.

So the drop from 0.7045 to 0.7916 is two separate effects stacked, user weighting and item recency, not one.

### The recency slice

Held-out rows are sorted by timestamp descending, so `[:5]` is each user's five **most recent** held-out ratings rather than an arbitrary five.

| Selection | MAE | 95% interval (clustered by user) |
|---|---|---|
| first-5 (recent) | 0.7916 | +/- 0.072 |
| random-5 | 0.7449 | +/- 0.060 |

The point estimate says recent items are **harder** to predict, by 0.047 MAE.

**This is not a significant result.**
Users are the independent sampling unit, and the paired per-user difference is `+0.0467` with `SE 0.0306`, giving a 95% interval of `[-0.013, +0.107]` and `t = 1.53`.
At 128 users the effect cannot be distinguished from zero.

What this means in practice.
The headline comparison is unaffected, because both sides were scored on the same recent items, so the recency slice cancels out of the gap.
What it limits is generalisation: the result is established on recent held-out ratings, and whether it holds on a uniformly sampled subset is untested.
Re-running one LLM arm under `--item-selection random` would settle it, and is the cheapest next experiment if this comparison ends up load-bearing.

### The LLM arms cannot be ranked

The five arms span 0.9136 to 0.9852, a range of 0.072.
A single arm's 95% interval at this sample size is about the same width.

So `top_rated-k2` leading and `polarized-k2` trailing is **suggestive, not established**, and adjacent arms such as `top_rated-k2` at 0.9136 and `recent-k3` at 0.9161 are separated by 0.0025 and should not be ranked against each other at all.

The one comparison with any room in it is best against worst, `top_rated-k2` against `polarized-k2`, at 0.072.
Even that sits right at the interval width.

If the arm ranking matters, the sample has to grow.
It is 128 users and 640 items, and the differences being chased are an order of magnitude smaller than the baseline-to-LLM gap.

---

## The bias-only null

Script: `experiments/bias_only_null.py`.
Per-item predictions and the full summary: `reports/bias_only_null/`.

The null predicts `env.get_rating_bias(uid, mid)` for every pair, a debiased residual of exactly zero.
It models no user-item interaction, so an arm that does not beat it is reproducing a lookup table rather than representing preference.

It is scored on exactly the published pairs, through `select_held_items` with `--max-items 5 --item-selection first`.
The script re-scores the associative arm in the same rebuilt environment and refuses to continue unless it reproduces the published MLflow values to 1e-4.
All three did, exactly: 0.704546 and 0.791589 at 128 users, 0.719526 at 2566 users.
So the null is computed against the same bias model the published arms used.

Intervals are 95%, clustered by user.
With every user at the five-item cap, the item micro-average and the user-weighted mean are identical.

### Does each published arm beat the null?

| Arm | Users | MAE | Arm minus null | Beats the null? |
|---|---|---|---|---|
| **bias-only null** | 128 | **0.7359** (0.670 to 0.802) | | |
| associative-baseline-capped | 128 | 0.7916 | +0.056, paired CI 0.017 to 0.095, t = 2.8 | **No, worse** |
| llm-top_rated-k2 | 128 | 0.9136 | +0.178 | **No, worse** |
| llm-recent-k3 | 128 | 0.9161 | +0.180 | **No, worse** |
| llm-top_rated-k5 | 128 | 0.9331 | +0.197 | **No, worse** |
| llm-polarized-k3-no-fewshot | 128 | 0.9391 | +0.203 | **No, worse** |
| llm-polarized-k2 | 128 | 0.9852 | +0.249 | **No, worse** |
| **bias-only null** | 2566 | **0.6951** (0.681 to 0.710) | | |
| associative, re-scored capped | 2566 | 0.7415 | +0.046, paired CI 0.037 to 0.055, t = 10.0 | **No, worse** |
| llm-top_rated-k2 | 2566 | 0.8934 | +0.198 | **No, worse** |

No arm beats the null, at either scale.

The associative rows are paired tests on identical pairs.
The LLM rows cannot be paired, because the 2026-06-26 runs logged aggregates only, so they are read against the null's own interval.
The smallest LLM gap, 0.178 at 128 users, is 5.3 null standard errors; at 2566 users the gap is 0.198 against a null SE of 0.0074.
Even allowing the LLM arm an interval as wide again as the null's, neither is close.

The 2566-user associative row is new.
That sweep's baseline was never re-scored capped, and this is the first matched number for it: 0.7415 on the same 12830 pairs as the LLM arm.

Clamping the null to `[1, 5]` changes nothing material: 0.7350 and 0.6938, against unclamped 0.7359 and 0.6951.
Unclamped is primary because the benchmark measures debiased residuals.
Only 4 of 640 and 54 of 12830 null predictions fall outside the range.

### It is not an artefact of the recent-5 slice

Associative minus null, paired by user, on the other selections:

| Users | Selection | Null | Associative | Difference | t |
|---|---|---|---|---|---|
| 128 | all held-out | 0.6579 | 0.7045 | +0.043 | 3.1 |
| 128 | random-5 | 0.7037 | 0.7449 | +0.041 | 2.2 |
| 2566 | all held-out | 0.6524 | 0.7195 | +0.057 | 17.0 |
| 2566 | random-5 | 0.6744 | 0.7382 | +0.064 | 14.1 |

The associative arm is worse than the null on every selection.
The uncapped `associative-baseline` at 0.7045, the lowest MAE in the original table, loses to the null on the same 5659 items, which scores 0.6579.

### Why the associative arm loses: its dot term is in the wrong units

The associative prediction is `bias + dot(pref_vector, item_factor)`.
It can only lose to `bias` alone if the dot term hurts more than it helps, and it does, for a reason that is structural rather than statistical.

The preference space is a `TruncatedSVD` fitted on **raw** training ratings, not debiased residuals (`sim/environment.py`, `_setup_user_pref_embeddings`), and both sides are L2-normalised.
So the dot term is a cosine similarity in `[-1, 1]`, not a rating residual.
Raw ratings are all positive, so the leading singular direction is shared by everyone and the cosine is positive almost everywhere.

Measured on the recent-5 pairs:

| | 128 users | 2566 users |
|---|---|---|
| Mean of associative minus null prediction | +0.402 | +0.384 |
| Share of pairs where it is positive | 96.6% | 94.7% |
| Mean signed error, null | +0.007 | -0.030 |
| Mean signed error, associative | +0.409 | +0.355 |
| Correlation of dot term with the true residual | 0.021 | 0.068 |
| Least-squares scale of the dot term | 0.008 | 0.125 |

The null is unbiased.
The associative arm over-predicts by about 0.4 stars, and the dot term carries almost no information about the residual it is added to.
A correctly-scaled term would have a least-squares coefficient near 1.

This contradicts the module docstring of `experiments/llm_vs_associative.py`, which describes `bias + dot` as "the same decomposition the model was fitted on".
It is not: the bias model and the SVD were fitted separately, on different targets.
It is a live example of the project's known trap that the associative and LLM agents return different units from the same interface.

### What this changes

- **The headline.** "The associative baseline beats every LLM arm by 0.122 MAE" is still true as a statement about those two arms. But both are worse than a bias table, so it does not say that latent factors beat content-based LLM prediction. It says a miscalibrated latent-factor arm beats a worse LLM arm.
- **The LLM result.** The LLM arms are 0.18 to 0.25 MAE worse than the null. Given `k` rated examples, Qwen2.5-7B predicts ratings worse than the user's and item's average ratings do. That is the cleanest finding in this report.
- **The floor for SASRec and every later backend is the null, not associative.** An arm has to beat 0.6951 at 2566 users, first-5, to show it represents preference at all.
- **The associative arm needs fixing before it is used as a baseline again.** That is issue #27, kept separate because the same preference space drives the simulation's personas.

---

## A second sweep exists

The `llm-agent-comparison` experiment also holds an earlier, larger sweep at `eval_user_frac=0.02`, 2566 users.

| Arm | `error/mae` | `meta/item_count` | `meta/user_count` |
|---|---|---|---|
| associative-baseline | 0.7195 | 116922 | 2566 |
| llm-top_rated-k2 | 0.8934 | 12830 | 2566 |

It is excluded from the headline result for two reasons.
It carries the identical item-count confound, `12830 = 2566 x 5` against an uncapped baseline, and it has not been re-scored.
More importantly only one LLM arm ever ran at that scale, so it cannot rank variants and cannot support the prompt-strategy question this experiment exists to ask.

Its baseline has now been re-scored on the matched pairs, as part of the bias-only null run: 0.7415 on the same 12830 pairs as the LLM arm, a gap of 0.152.
See [The bias-only null](#the-bias-only-null).

It is worth noting that it points the same way, and on 20x the users, which is mild independent support for the direction of the headline result.

---

## What this does not establish

- **One model.** `Qwen2.5-7B-Instruct-4bit`, 4-bit quantised, greedy decoding. Nothing here generalises to larger models, other families, or unquantised weights.
- **One dataset.** MovieLens-32M, held-out split, users with at least 50 ratings.
- **One prompt family.** Five variants over history selection and few-shot presence. No chain-of-thought, no structured decoding, no persona conditioning.
- **Rating prediction, not ranking.** MAE on held-out ratings says nothing directly about NDCG or hit rate in the simulation loop, which is what the recommender actually consumes.
- **No persona information.** The prompts deliberately exclude archetype and persona, to match the associative agent's information boundary. An LLM agent given persona context is a different and untested proposition.
- **A small sample.** 128 users, 640 items per arm. Sufficient to establish the baseline-to-LLM gap, insufficient to rank the arms or resolve the recency effect.

---

## Reproducing

```bash
# Bias-only null on both published sweeps, with the associative re-score,
# selection robustness and dot-term diagnostics. About a minute.
uv run python experiments/bias_only_null.py

# Capped baseline, matched to the LLM arms. No LLM calls, about 15 s on a warm
# embedding cache.
uv run python experiments/llm_vs_associative.py --baseline-only --max-items 5

# Same, under uniform sampling instead of the recent-5 slice.
uv run python experiments/llm_vs_associative.py --baseline-only --max-items 5 \
    --item-selection random

# One LLM arm. Hours, not seconds.
uv run python experiments/llm_vs_associative.py --variant llm-top_rated-k2 --max-items 5
```

Interval and significance figures in this report come from clustering absolute errors by user and taking the standard error of the per-user means, which is the correct unit here because items within a user are not independent.
