"""
E2E-Test: Watchdog + PipelineGuard + SpecConstitution integration.
Verifies v10 infrastructure end-to-end.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.mark.e2e
class TestWatchdogPipelineE2E:
    def test_preflight_detects_nan(self):
        from backend.core.watchdog_monitor import WatchdogMonitor

        wd = WatchdogMonitor()
        ok, issues = wd.pre_flight_check(np.full(48000, np.nan, dtype=np.float32), 48000)
        assert not ok and any("NaN" in i for i in issues)

    def test_preflight_passes_clean(self):
        from backend.core.watchdog_monitor import WatchdogMonitor

        wd = WatchdogMonitor()
        a = (np.sin(2 * np.pi * 440 * np.arange(48000) / 48000) * 0.3).astype(np.float32)
        assert wd.pre_flight_check(a, 48000)[0]

    def test_phase_tracking(self):
        from backend.core.watchdog_monitor import WatchdogMonitor

        wd = WatchdogMonitor()
        a = (np.sin(2 * np.pi * 440 * np.arange(48000) / 48000) * 0.3).astype(np.float32)
        wd.pre_flight_check(a, 48000)
        wd.on_phase_start("test")
        wd.on_phase_end("test", a, 48000)
        r = wd.post_flight_validity(a, 48000)
        assert len(r.phase_watches) > 0

    def test_pipeline_guard_full_cycle(self):
        from backend.core.pipeline_guard import PipelineGuard

        g = PipelineGuard()
        a = (np.sin(2 * np.pi * 440 * np.arange(48000) / 48000) * 0.3).astype(np.float32)
        ok, _ = g.pre_flight(a, 48000, "vinyl")
        assert ok
        g.phase_start("test")
        g.phase_end("test", a, 48000)
        r = g.post_flight(a, 48000)
        assert "watchdog" in r and "adaptive_goals" in r

    def test_spec_constitution_shield(self):
        from backend.core.spec_constitution import get_constitution

        c = get_constitution()
        assert c.goal_count == 15 and c.forbidden_count >= 20
        assert c.get_shield_thresholds()["artifact_freedom_min"] == 0.95

    def test_gpu_backend_enum(self):
        from backend.core.ml_device_manager import GPUBackend

        assert {b.value for b in GPUBackend} == {"cuda", "rocm", "directml", "none"}

    def test_clp_zones(self):
        from backend.core.critical_listening_points import CLP_ZONES

        assert len(CLP_ZONES) == 6 and "Praesenz" in {z.name for z in CLP_ZONES}

    def test_no_venv_rocm_refs(self):
        from pathlib import Path

        from backend.core.runtime_env_selector import _candidate_paths

        ps = [str(p) for p in _candidate_paths(Path("."))]
        assert not any(".venv_rocm" in p for p in ps), f"Stale refs: {ps}"
