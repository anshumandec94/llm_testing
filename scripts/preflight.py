"""
Check that this machine is ready for a comparison run, before starting one.

A fresh clone carries code only. This reports, one PASS or FAIL line each,
the things that otherwise fail slowly, expensively or silently, and exits
non-zero if any check fails. It runs nothing.

    cuda          torch sees a GPU (device name, CUDA version torch was built for)
    data          ratings.csv, movies.csv and movie_overviews.csv under data_dir,
                  with ML-32M's exact row counts. A missing overviews file is a
                  failure: Environment only warns, and the LLM arm's prompts
                  silently lose their overview text.
    data_dir      is the default relative data/ml-32m. split_cache_key() hashes
                  the path as written, and merge_runs.py refuses arms whose keys
                  differ, so every machine keeps the same relative path.
    embeddings    the ChromaDB collections and factor caches for this sweep's
                  config exist. Otherwise the first Environment() rebuilds them:
                  about 27 minutes and 9.5 GB RSS.
    disk          free space against REQUIRED_FREE_GB.
    mlflow        the tracking URI runs will write to, as an absolute path.

Usage:
    uv run python scripts/preflight.py            # the u2566 sweep
    uv run python scripts/preflight.py --sweep u128
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.bias_only_null import SWEEPS  # noqa: E402
from experiments.compare_backends import sweep_config  # noqa: E402
from sim.config import REPO_ROOT, SimConfig  # noqa: E402
from sim.environment import embedding_collection_names, embedding_factor_paths  # noqa: E402

# Data rows (header excluded) in the ML-32M release; overviews are TMDB-derived
# and only need to exist and be non-empty.
EXPECTED_ROWS = {"ratings.csv": 32_000_204, "movies.csv": 87_585, "movie_overviews.csv": None}
DEFAULT_DATA_DIR = SimConfig().data_dir
# Embedding caches (3.5 GB) plus the torch LLM weights (#15, ~15 GB on first use).
REQUIRED_FREE_GB = 20


@dataclass
class Check:
    """One line of the report."""

    name: str
    passed: bool
    detail: str


def check_cuda() -> Check:
    """torch can use a CUDA device."""
    import torch

    if not torch.cuda.is_available():
        return Check("cuda", False, f"torch {torch.__version__} sees no CUDA device (built for CUDA {torch.version.cuda})")
    return Check("cuda", True, f"{torch.cuda.get_device_name(0)}, torch built for CUDA {torch.version.cuda}")


def _count_rows(path: Path) -> int:
    """Lines after the header, read in blocks so a 900 MB file costs a second or two."""
    with path.open("rb") as handle:
        newlines = sum(block.count(b"\n") for block in iter(lambda: handle.read(1 << 20), b""))
    return newlines - 1


def check_data(cfg: SimConfig) -> list[Check]:
    """Each data file exists, with ML-32M's row count where that is fixed."""
    checks = []
    for name, expected in EXPECTED_ROWS.items():
        path = Path(cfg.data_dir) / name
        if not path.is_file():
            checks.append(Check(f"data {name}", False, f"{path} is missing"))
            continue
        rows = _count_rows(path)
        ok = rows == expected if expected is not None else rows > 0
        want = f", expected {expected:,}" if expected is not None and not ok else ""
        checks.append(Check(f"data {name}", ok, f"{rows:,} rows{want}"))
    return checks


def check_data_dir(cfg: SimConfig) -> Check:
    """data_dir is spelled as the default, so split fingerprints match across machines."""
    ok = Path(cfg.data_dir) == DEFAULT_DATA_DIR
    detail = str(cfg.data_dir) if ok else f"{cfg.data_dir} is not {DEFAULT_DATA_DIR}; split_cache_key will not match other machines"
    return Check("data_dir", ok, detail)


def check_embeddings(cfg: SimConfig) -> Check:
    """Every cache this config reads exists, so Environment() will not rebuild."""
    embeddings_dir = Path(cfg.embeddings_dir)
    rebuild = "Environment() will rebuild (~27 min, ~9.5 GB RSS)"
    if not embeddings_dir.is_dir():
        return Check("embeddings", False, f"{embeddings_dir} does not exist; {rebuild}")
    import chromadb

    client = chromadb.PersistentClient(path=str(embeddings_dir))
    existing = {collection.name for collection in client.list_collections()}
    missing = [name for name in embedding_collection_names(cfg) if name not in existing]
    missing += [path.name for path in embedding_factor_paths(cfg) if not path.is_file()]
    if missing:
        return Check("embeddings", False, f"missing {', '.join(missing)}; {rebuild}")
    return Check("embeddings", True, f"all caches present in {embeddings_dir}")


def check_disk(path: Path = REPO_ROOT) -> Check:
    """Free space on the repo's filesystem."""
    free_gb = shutil.disk_usage(path).free / 1e9
    return Check("disk", free_gb >= REQUIRED_FREE_GB, f"{free_gb:.1f} GB free, {REQUIRED_FREE_GB} GB required")


def check_mlflow(cfg: SimConfig) -> Check:
    """Name the store runs will land in; a local one's directory must exist."""
    uri = cfg.mlflow_tracking_uri
    parsed = urlparse(uri)
    if parsed.scheme != "sqlite":
        return Check("mlflow", True, uri)
    store = Path(parsed.path[1:])
    return Check("mlflow", store.parent.is_dir(), f"{store}" + ("" if store.parent.is_dir() else " (directory missing)"))


def run_checks(cfg: SimConfig) -> list[Check]:
    """Every check, in report order."""
    return [check_cuda(), *check_data(cfg), check_data_dir(cfg), check_embeddings(cfg), check_disk(), check_mlflow(cfg)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Report whether this machine is ready for a comparison run.")
    parser.add_argument(
        "--sweep", choices=sorted(SWEEPS), default="u2566",
        help="The sweep the run will use; embedding caches are keyed by it. Default u2566.",
    )
    args = parser.parse_args(argv)

    checks = run_checks(sweep_config(args.sweep))
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'}  {check.name:<26} {check.detail}")
    failed = [check.name for check in checks if not check.passed]
    if failed:
        sys.exit(f"\nnot ready: {len(failed)} check(s) failed: {', '.join(failed)}")
    print("\nready")


if __name__ == "__main__":
    main()
