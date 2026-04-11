"""Tests for conditional constraint types."""

from __future__ import annotations

import pytest

from imads_hpo.constraints import (
    CompositeConstraint,
    ConditionalConstraint,
    DynamicBoundConstraint,
    GpuMemoryConstraint,
)


class TestConditionalConstraint:
    def test_active_when_predicate_true(self):
        inner = GpuMemoryConstraint(max_gb=8.0)
        cc = ConditionalConstraint(predicate=lambda p: p.get("use_gpu", False), inner=inner)
        assert cc(10.0, {"use_gpu": True}) == pytest.approx(2.0)

    def test_feasible_when_active(self):
        inner = GpuMemoryConstraint(max_gb=8.0)
        cc = ConditionalConstraint(predicate=lambda p: p.get("use_gpu", False), inner=inner)
        assert cc(6.0, {"use_gpu": True}) == pytest.approx(-2.0)

    def test_inactive_returns_default(self):
        inner = GpuMemoryConstraint(max_gb=8.0)
        cc = ConditionalConstraint(predicate=lambda p: p.get("use_gpu", False), inner=inner)
        assert cc(100.0, {"use_gpu": False}) == pytest.approx(-1.0)

    def test_custom_inactive_value(self):
        inner = GpuMemoryConstraint(max_gb=8.0)
        cc = ConditionalConstraint(predicate=lambda p: False, inner=inner, inactive_value=-5.0)
        assert cc(100.0, {}) == pytest.approx(-5.0)


class TestDynamicBoundConstraint:
    def test_infeasible(self):
        dbc = DynamicBoundConstraint(bound_fn=lambda p: p["batch_size"] * 0.5)
        assert dbc(10.0, {"batch_size": 16}) == pytest.approx(2.0)

    def test_feasible(self):
        dbc = DynamicBoundConstraint(bound_fn=lambda p: p["batch_size"] * 0.5)
        assert dbc(3.0, {"batch_size": 16}) == pytest.approx(-5.0)

    def test_exact_bound(self):
        dbc = DynamicBoundConstraint(bound_fn=lambda p: 10.0)
        assert dbc(10.0, {}) == pytest.approx(0.0)


class TestCompositeConstraint:
    def test_all_feasible(self):
        c1 = DynamicBoundConstraint(bound_fn=lambda p: 10.0)
        c2 = DynamicBoundConstraint(bound_fn=lambda p: 20.0)
        comp = CompositeConstraint(constraints=(c1, c2))
        assert comp([5.0, 10.0], {}) == pytest.approx(-5.0)

    def test_one_infeasible(self):
        c1 = DynamicBoundConstraint(bound_fn=lambda p: 10.0)
        c2 = DynamicBoundConstraint(bound_fn=lambda p: 5.0)
        comp = CompositeConstraint(constraints=(c1, c2))
        assert comp([8.0, 8.0], {}) == pytest.approx(3.0)

    def test_single_constraint(self):
        c = DynamicBoundConstraint(bound_fn=lambda p: 5.0)
        comp = CompositeConstraint(constraints=(c,))
        assert comp([3.0], {}) == pytest.approx(-2.0)
