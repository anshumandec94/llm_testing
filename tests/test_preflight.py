"""
tests/test_preflight.py - the server readiness report (issue #21).

Pins the checks that would otherwise fail silently or slowly: a missing
overviews file, a truncated ratings copy, a data_dir that breaks split
fingerprints, and absent embedding caches. Synthetic fixtures only.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

import scripts.preflight as preflight
from scripts.preflight import Check, check_data, check_data_dir, check_embeddings, main


def _write_rows(path: Path, rows: int) -> None:
    path.write_text("header\n" + "row\n" * rows)


@pytest.fixture
def ml32m_dir(tmp_path, monkeypatch) -> Path:
    """A data dir with the expected row counts, scaled down."""
    monkeypatch.setattr(preflight, "EXPECTED_ROWS", {"ratings.csv": 5, "movies.csv": 2, "movie_overviews.csv": None})
    _write_rows(tmp_path / "ratings.csv", 5)
    _write_rows(tmp_path / "movies.csv", 2)
    _write_rows(tmp_path / "movie_overviews.csv", 2)
    return tmp_path


class TestData:

    def test_complete_data_passes(self, tiny_config, ml32m_dir):
        checks = check_data(dataclasses.replace(tiny_config, data_dir=ml32m_dir))
        assert all(check.passed for check in checks), checks

    def test_missing_overviews_is_a_failure_not_a_warning(self, tiny_config, ml32m_dir):
        (ml32m_dir / "movie_overviews.csv").unlink()
        failed = [c.name for c in check_data(dataclasses.replace(tiny_config, data_dir=ml32m_dir)) if not c.passed]
        assert failed == ["data movie_overviews.csv"]

    def test_truncated_ratings_fail(self, tiny_config, ml32m_dir):
        _write_rows(ml32m_dir / "ratings.csv", 4)
        failed = [c for c in check_data(dataclasses.replace(tiny_config, data_dir=ml32m_dir)) if not c.passed]
        assert [c.name for c in failed] == ["data ratings.csv"]
        assert "expected 5" in failed[0].detail

    def test_a_non_default_data_dir_fails(self, tiny_config):
        assert check_data_dir(dataclasses.replace(tiny_config, data_dir=Path("data/ml-32m"))).passed
        assert not check_data_dir(dataclasses.replace(tiny_config, data_dir=Path("data/ml-32m").resolve())).passed


class TestEmbeddings:

    def test_built_caches_pass(self, tiny_config, env):
        assert check_embeddings(tiny_config).passed

    def test_a_config_with_unbuilt_caches_fails(self, tiny_config, env):
        check = check_embeddings(dataclasses.replace(tiny_config, mf_features=tiny_config.mf_features + 1))
        assert not check.passed
        assert "rebuild" in check.detail


def test_main_exits_non_zero_when_any_check_fails(monkeypatch, capsys):
    monkeypatch.setattr(preflight, "run_checks", lambda cfg: [Check("a", True, ""), Check("b", False, "why")])
    with pytest.raises(SystemExit, match="1 check"):
        main(["--sweep", "u128"])
    out = capsys.readouterr().out
    assert "PASS  a" in out and "FAIL  b" in out
