import pytest

"""Unit tests for cross_phase_naturalness.py"""
import numpy as np

from backend.core.cross_phase_naturalness import (
    CrossPhaseTracker,
    estimate_band_effects,
    get_tracker,
    guard_stage,
)

_SR = 48000


@pytest.mark.unit
class TestCrossPhaseTracker:
    def test_01_init(self):
        t = CrossPhaseTracker()
        assert t is not None

    def test_02_record_band(self):
        t = CrossPhaseTracker()
        t.record("test", {"presence": 3.0})
        r = t.get_band_report()
        assert r["presence"]["cumulative_gain_db"] == 3.0

    def test_03_can_process_below_limit(self):
        t = CrossPhaseTracker()
        assert t.can_process("presence", 2.0)

    def test_04_cannot_process_above_limit(self):
        t = CrossPhaseTracker()
        t.record("p1", {"presence": 7.0})
        assert not t.can_process("presence", 2.0)

    def test_05_cannot_process_too_many_phases(self):
        t = CrossPhaseTracker()
        t.record("p1", {"presence": 1.0})
        t.record("p2", {"presence": 1.0})
        t.record("p3", {"presence": 1.0})
        assert not t.can_process("presence", 0.5)

    def test_06_suggest_scale(self):
        t = CrossPhaseTracker()
        assert t.suggest_scale(["presence"]) == 1.0
        t.record("p1", {"presence": 6.0})
        assert t.suggest_scale(["presence"]) < 1.0

    def test_07_reset(self):
        t = CrossPhaseTracker()
        t.record("p1", {"presence": 3.0})
        t.reset()
        r = t.get_band_report()
        assert r["presence"]["cumulative_gain_db"] == 0.0

    def test_08_singleton(self):
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2

    def test_09_estimate_band_effects(self):
        a = np.random.randn(_SR).astype(np.float32) * 0.3
        b = a * 1.1
        e = estimate_band_effects(a, b, _SR)
        assert isinstance(e, dict)

    def test_10_guard_stage_no_change(self):
        a = np.random.randn(_SR).astype(np.float32) * 0.3
        r, s = guard_stage("test", a, a.copy(), _SR)
        assert s == 1.0
