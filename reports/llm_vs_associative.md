# LLM agent vs associative baseline: held-out rating prediction

Experiment: `llm-agent-comparison` (`sqlite:///mlflow.db`)
Script: `experiments/llm_vs_associative.py`
LLM arms run 2026-06-26. Corrected baselines run 2026-08-25.
Last updated: 2026-08-25

---

## Summary

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

## A second sweep exists

The `llm-agent-comparison` experiment also holds an earlier, larger sweep at `eval_user_frac=0.02`, 2566 users.

| Arm | `error/mae` | `meta/item_count` | `meta/user_count` |
|---|---|---|---|
| associative-baseline | 0.7195 | 116922 | 2566 |
| llm-top_rated-k2 | 0.8934 | 12830 | 2566 |

It is excluded from the headline result for two reasons.
It carries the identical item-count confound, `12830 = 2566 x 5` against an uncapped baseline, and it has not been re-scored.
More importantly only one LLM arm ever ran at that scale, so it cannot rank variants and cannot support the prompt-strategy question this experiment exists to ask.

It is worth noting that it points the same way, and on 20x the users, which is mild independent support for the direction of the headline result.
Re-scoring its baseline with `--baseline-only --max-items 5` would make it directly quotable and costs one environment build.

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
