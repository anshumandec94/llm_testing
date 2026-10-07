# Running on CARC

How to get a comparison run going on the CARC server and its results back.

## What a clone carries

A `git clone` carries code only.
These are gitignored and must be copied or rebuilt:

| Path | Size | If absent |
|---|---|---|
| `data/ml-32m/` (`ratings.csv`, `movies.csv`, `movie_overviews.csv`) | ~0.9 GB | Copy it. A missing `movie_overviews.csv` does not error: `Environment` only warns, and LLM prompts lose their overview text. |
| `embeddings/chroma/` | ~3.5 GB | Rebuilt on the first `Environment()`: about 27 minutes and 9.5 GB RSS. Copy it to skip that. |
| `mlflow.db`, `mlruns/`, `mlartifacts/` | varies | Created on first run. Each machine has its own store. |
| HuggingFace model cache | ~15 GB | Downloaded on first use by the torch LLM path only. |

## The one rule: keep `data_dir` relative

Put the data at `data/ml-32m/` inside the clone, the same relative path as on the laptop.
`SimConfig.split_cache_key()` hashes `data_dir` exactly as written, so an absolute path gives a different split fingerprint for the same split.
`scripts/merge_runs.py` refuses to merge arms whose fingerprints differ.

## Setup

```bash
git clone git@github.com:anshumandec94/llm_testing.git && cd llm_testing
uv sync
rsync -a laptop:~/Projects/phd/llm_testing/data/ml-32m/ data/ml-32m/
rsync -a laptop:~/Projects/phd/llm_testing/embeddings/chroma/ embeddings/chroma/   # optional, skips the rebuild
uv run python scripts/preflight.py --sweep u2566
```

The preflight prints one PASS or FAIL line per check and exits non-zero if any fail.
If `cuda` fails on a GPU node, torch was installed without CUDA: add a `[[tool.uv.index]]` entry for the matching PyTorch CUDA index.

Runs go to `MLFLOW_TRACKING_URI` if set, else `mlflow.db` at the repo root; the preflight prints which.

## Launch

Run inside `tmux` so the job survives a dropped connection:

```bash
tmux new -s sasrec
uv run python experiments/run_sasrec_eval.py --sweep u2566
# detach with Ctrl-b d; reattach with: tmux attach -t sasrec
```

Rerunning resumes training from the last checkpoint.

## Bringing results back

Do not copy `mlflow.db`: it records artifact paths on the server.
Download each run's artifacts on the server, copy the folder, and merge on the laptop:

```bash
# on the server
uv run mlflow artifacts download --run-id <run_id> --dst-path runs/export/sasrec-u2566
# on the laptop
rsync -a carc:~/llm_testing/runs/export/sasrec-u2566 runs/export/
uv run python scripts/merge_runs.py \
    --run sqlite:///$PWD/mlflow.db <local_run_id> \
    --artifacts runs/export/sasrec-u2566 \
    --out reports/backend_comparison
```

The merge refuses arms that scored different pairs or used a different split.
