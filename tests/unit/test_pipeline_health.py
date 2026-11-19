from __future__ import annotations

"""
test_pipeline_health.py — Pre-Pipeline Health Check Tests
==========================================================

Verifiziert, dass run_health_checks() funktioniert und alle C1-C5 Checks durchführt.
"""


class TestPipelineHealthCheck:
    """Pre-Pipeline Health Verification."""

    def test_01_health_check_runs_without_crash(self):
        """run_health_checks() läuft ohne Exception."""
        from backend.core.pipeline_health_check import run_health_checks

        report = run_health_checks(audio_duration_s=60.0)
        assert report is not None
        assert len(report.checks) >= 4, f"Nur {len(report.checks)} Checks, erwartet >=4"

    def test_02_all_checks_have_results(self):
        """Jeder Check hat name, passed, duration_ms."""
        from backend.core.pipeline_health_check import run_health_checks

        report = run_health_checks()
        for check in report.checks:
            assert check.name, "Check ohne Namen"
            assert check.duration_ms >= 0, f"{check.name}: duration_ms negativ"
            assert isinstance(check.passed, bool), f"{check.name}: passed kein bool"

    def test_03_summary_includes_all_checks(self):
        """Summary enthält alle Check-Namen."""
        from backend.core.pipeline_health_check import run_health_checks

        report = run_health_checks()
        summary = report.summary()
        for check in report.checks:
            assert check.name in summary, f"{check.name} fehlt in Summary"

    def test_04_numpy_scipy_available(self):
        """C1: numpy und scipy sind verfügbar."""
        import numpy as np
        from scipy import signal

        assert np is not None
        assert signal is not None

    def test_05_dsp_modules_importable(self):
        """C2: Kritische DSP-Module sind importierbar."""
        from backend.core.audio_utils import compute_gated_rms_linear

        assert callable(compute_gated_rms_linear)

    def test_06_configuration_files_exist(self):
        """C4: Erforderliche Konfigurationsdateien existieren."""
        import os

        assert os.path.exists("pytest.ini"), "pytest.ini fehlt"
        assert os.path.exists(".github/specs/01_musical_goals.md"), "Spec 01 fehlt"


class TestPipelineHealthMonitorClock:
    """Wall-Time-Akkumulator des Health-Monitors: beide Seiten dieselbe Uhr.

    Regression zu .github/VERBOTEN.md „Wall-Time-Referenz-Mismatch": Mit
    time.time() als Startwert und time.monotonic() im Vergleich ergibt die
    Differenz ~-1,76e9 s — das 2-Stunden-Limit feuerte nie (Breaker inoperativ)
    und summary() meldete eine unsinnige Pipeline-Dauer.
    """

    def test_01_pipeline_budget_breaker_fires(self):
        import time

        from backend.core.pipeline_health_monitor import (
            _MAX_PIPELINE_DURATION_S,
            PipelineHealthMonitor,
        )

        monitor = PipelineHealthMonitor()
        assert monitor.check_circuit_breaker() is True
        monitor._health.pipeline_start_time = time.monotonic() - (_MAX_PIPELINE_DURATION_S + 1.0)
        assert monitor.check_circuit_breaker() is False, "Wall-Time-Limit muss greifen"
        assert monitor._health.circuit_breaker_triggered is True

    def test_02_summary_duration_is_plausible(self):
        from backend.core.pipeline_health_monitor import PipelineHealthMonitor

        monitor = PipelineHealthMonitor()
        duration = monitor.summary()["pipeline_duration_s"]
        assert 0.0 <= duration < 60.0, f"unsinnige Pipeline-Dauer: {duration}"

    def test_03_phase_duration_uses_same_clock(self):
        import time

        from backend.core.pipeline_health_monitor import PipelineHealthMonitor

        monitor = PipelineHealthMonitor()
        started = monitor.record_phase_start("phase_01_demo")
        time.sleep(0.01)
        monitor.record_phase_end("phase_01_demo", started, retries=0, success=True)
        duration = monitor._health.phase_durations["phase_01_demo"]
        assert 0.0 < duration < 5.0, f"Phasendauer unplausibel: {duration}"
