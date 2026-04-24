"""
tests/test_attendance.py — tests for AttendanceStrategy implementations.

Covers: will_attend return type, AlwaysAttend/BernoulliAttendance extremes,
LogisticAttendance responsiveness, ThresholdAttendance threshold logic,
and update_ewma convergence.
"""
from __future__ import annotations

import numpy as np
import pytest

from sim.attendance import (
    ATTENDANCE_REGISTRY,
    AlwaysAttend,
    BernoulliAttendance,
    LogisticAttendance,
    ThresholdAttendance,
)


@pytest.fixture
def rng():
    return np.random.default_rng(99)


class TestRegistry:
    def test_all_builtins_in_registry(self):
        for name in ("LogisticAttendance", "ThresholdAttendance", "AlwaysAttend", "BernoulliAttendance"):
            assert name in ATTENDANCE_REGISTRY, f"{name!r} missing from ATTENDANCE_REGISTRY"


class TestAlwaysAttend:
    def test_always_returns_true(self, rng):
        s = AlwaysAttend()
        for _ in range(20):
            assert s.will_attend(0.0, 0.0, 0, rng) is True

    def test_update_ewma_basic(self, rng):
        s = AlwaysAttend()
        new_ewma = s.update_ewma(0.0, 3.0, 0.4)
        assert new_ewma == pytest.approx(3.0 * 0.4)


class TestBernoulliAttendance:
    def test_zero_prob_always_false(self, rng):
        s = BernoulliAttendance(attend_prob=0.0)
        for _ in range(20):
            assert s.will_attend(0.0, 0.0, 0, rng) is False

    def test_one_prob_always_true(self, rng):
        s = BernoulliAttendance(attend_prob=1.0)
        for _ in range(20):
            assert s.will_attend(0.0, 0.0, 0, rng) is True

    def test_returns_bool(self, rng):
        s = BernoulliAttendance(attend_prob=0.5)
        result = s.will_attend(0.0, 0.0, 0, rng)
        assert isinstance(result, bool)

    def test_moderate_prob_sometimes_true_sometimes_false(self):
        rng = np.random.default_rng(0)
        s = BernoulliAttendance(attend_prob=0.5)
        outcomes = [s.will_attend(0.0, 0.0, 0, rng) for _ in range(200)]
        assert any(outcomes) and not all(outcomes)


class TestLogisticAttendance:
    def test_returns_bool(self, rng):
        s = LogisticAttendance()
        result = s.will_attend(0.0, 0.0, 0, rng)
        assert isinstance(result, bool)

    def test_high_baseline_attends_more_often(self):
        rng = np.random.default_rng(1)
        high = LogisticAttendance()
        low = LogisticAttendance()
        n = 200
        high_count = sum(high.will_attend(5.0, 0.0, 0, rng) for _ in range(n))
        rng = np.random.default_rng(1)
        low_count = sum(low.will_attend(-5.0, 0.0, 0, rng) for _ in range(n))
        assert high_count > low_count

    def test_high_ewma_increases_attendance(self):
        rng_hi = np.random.default_rng(2)
        rng_lo = np.random.default_rng(2)
        s = LogisticAttendance(beta_sat=2.0)
        n = 200
        hi_count = sum(s.will_attend(0.0, 3.0, 0, rng_hi) for _ in range(n))
        lo_count = sum(s.will_attend(0.0, 0.0, 0, rng_lo) for _ in range(n))
        assert hi_count >= lo_count

    def test_update_ewma_smoothing(self):
        s = LogisticAttendance()
        ewma = s.update_ewma(0.0, 4.0, 0.4)
        expected = 0.6 * 0.0 + 0.4 * 4.0
        assert ewma == pytest.approx(expected)

    def test_update_ewma_convergence(self):
        """After many steps with constant signal, EWMA should converge to signal."""
        s = LogisticAttendance()
        ewma = 0.0
        for _ in range(200):
            ewma = s.update_ewma(ewma, 3.0, 0.3)
        assert abs(ewma - 3.0) < 0.1


class TestThresholdAttendance:
    def test_above_threshold_attends(self, rng):
        s = ThresholdAttendance(threshold=1.0, fallback_prob=0.0)
        # ewma well above threshold → should always attend
        outcomes = [s.will_attend(0.0, 3.0, 0, rng) for _ in range(20)]
        assert all(outcomes)

    def test_below_threshold_uses_fallback(self, rng):
        s = ThresholdAttendance(threshold=5.0, fallback_prob=0.0)
        outcomes = [s.will_attend(0.0, 0.0, 0, rng) for _ in range(20)]
        assert not any(outcomes)
