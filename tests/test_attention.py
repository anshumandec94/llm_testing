"""
tests/test_attention.py — tests for AttentionStrategy implementations.

Covers: effective_k bounds, budget depletion, recovery modes, and
behavioural differences between strategies.
"""
from __future__ import annotations

import pytest

from sim.attention import (
    ATTENTION_REGISTRY,
    ExponentialDecayAttention,
    LinearDecayAttention,
    PerRequestAttention,
)


class TestRegistry:
    def test_all_builtins_in_registry(self):
        for name in ("LinearDecay", "ExponentialDecay", "PerRequest"):
            assert name in ATTENTION_REGISTRY, f"{name!r} missing from ATTENTION_REGISTRY"

    def test_registry_values_are_classes(self):
        for name, cls in ATTENTION_REGISTRY.items():
            assert isinstance(cls, type), f"{name!r} value is not a class"


class TestLinearDecayAttention:
    def _make(self, decay_rate=0.15, recovery="full"):
        return LinearDecayAttention(decay_rate=decay_rate, recovery=recovery)

    # effective_k
    def test_effective_k_zero_budget(self):
        s = self._make()
        assert s.effective_k(10, 0.0) == 0

    def test_effective_k_at_most_list_size(self):
        s = self._make()
        k = s.effective_k(10, 1.0)
        assert k <= 10

    def test_effective_k_at_least_one_when_budget_positive(self):
        s = self._make()
        k = s.effective_k(10, 0.01)
        assert k >= 1

    # deplete
    def test_deplete_reduces_budget(self):
        s = self._make(decay_rate=0.15)
        new_b = s.deplete(5, 1.0)
        assert new_b < 1.0

    def test_deplete_clamped_to_zero(self):
        s = self._make(decay_rate=1.0)
        new_b = s.deplete(10, 0.5)
        assert new_b == pytest.approx(0.0)

    def test_deplete_clamped_at_one(self):
        s = self._make(decay_rate=0.0)
        new_b = s.deplete(5, 1.0)
        assert new_b == pytest.approx(1.0)

    # restore — full recovery
    def test_restore_full_returns_one(self):
        s = self._make(recovery="full")
        assert s.restore(0.2, 0.0) == pytest.approx(1.0)

    # restore — partial recovery
    def test_restore_partial_increases_budget(self):
        s = LinearDecayAttention(decay_rate=0.15, recovery="partial", recovery_rate=0.5)
        new_b = s.restore(0.4, 0.0)
        assert new_b > 0.4

    def test_restore_partial_clamped(self):
        s = LinearDecayAttention(decay_rate=0.15, recovery="partial", recovery_rate=1.0)
        new_b = s.restore(0.9, 0.0)
        assert new_b <= 1.0 + 1e-9

    # restore — satisfaction
    def test_restore_satisfaction_high_signal_restores_full(self):
        s = LinearDecayAttention(
            decay_rate=0.15, recovery="satisfaction", recovery_rate=0.3, sat_threshold=3.5
        )
        new_b = s.restore(0.1, 4.0)   # signal above threshold
        assert new_b == pytest.approx(1.0)

    def test_restore_satisfaction_low_signal_partial(self):
        s = LinearDecayAttention(
            decay_rate=0.15, recovery="satisfaction", recovery_rate=0.3, sat_threshold=3.5
        )
        new_b = s.restore(0.2, 1.0)   # signal below threshold
        assert 0.2 < new_b < 1.0


class TestExponentialDecayAttention:
    def _make(self, decay_rate=0.3):
        return ExponentialDecayAttention(decay_rate=decay_rate)

    def test_deplete_reduces_budget(self):
        s = self._make()
        new_b = s.deplete(5, 1.0)
        assert new_b < 1.0

    def test_deplete_result_positive(self):
        s = self._make(decay_rate=0.5)
        new_b = s.deplete(10, 1.0)
        assert new_b >= 0.0

    def test_different_decay_rates_produce_different_results(self):
        slow = ExponentialDecayAttention(decay_rate=0.1)
        fast = ExponentialDecayAttention(decay_rate=0.5)
        assert slow.deplete(5, 1.0) > fast.deplete(5, 1.0)


class TestPerRequestAttention:
    def _make(self, decay_rate=0.2):
        return PerRequestAttention(decay_rate=decay_rate)

    def test_deplete_ignores_list_size(self):
        s = self._make(decay_rate=0.2)
        b1 = s.deplete(1, 1.0)
        b2 = s.deplete(100, 1.0)
        assert b1 == pytest.approx(b2)

    def test_deplete_reduces_by_flat_amount(self):
        s = self._make(decay_rate=0.2)
        new_b = s.deplete(10, 1.0)
        assert new_b == pytest.approx(0.8)


class TestStrategyContrast:
    """Verify the three strategies behave distinctly for the same input."""

    def test_linear_vs_exponential_differ(self):
        lin = LinearDecayAttention(decay_rate=0.1)
        exp = ExponentialDecayAttention(decay_rate=0.1)
        assert lin.deplete(5, 1.0) != pytest.approx(exp.deplete(5, 1.0))

    def test_per_request_vs_linear_differ(self):
        lin = LinearDecayAttention(decay_rate=0.1)
        pr = PerRequestAttention(decay_rate=0.2)
        assert lin.deplete(5, 1.0) != pytest.approx(pr.deplete(5, 1.0))
