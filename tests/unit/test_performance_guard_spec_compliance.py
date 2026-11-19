from __future__ import annotations

import time

import pytest

from backend.core.performance_guard import PerformanceGuard, QualityMode


@pytest.mark.unit
class TestPerformanceGuardSpecCompliance:
    """Spec-Compliance für PerformanceGuard (§RT8/2026)."""

    def test_limits_match_2026_spec(self) -> None:
        assert pytest.approx(8.0) == PerformanceGuard.LIMIT_FAST
        assert pytest.approx(32.0) == PerformanceGuard.LIMIT_BALANCED  # v10.0.0: 32× RT (§PerformanceGuard)
        assert pytest.approx(32.0) == PerformanceGuard.LIMIT_QUALITY  # v10.0.0: 32× RT (Restoration)
        assert pytest.approx(32.0) == PerformanceGuard.LIMIT_MAXIMUM  # v10.0.0: 32× RT (Studio-2026-ML-Chain)
        assert pytest.approx(32.0) == PerformanceGuard.RT8_EXCELLENCE_BUDGET  # v10.0.0: 32× RT

    def test_target_mapping_uses_quality_budget(self) -> None:
        fast_guard = PerformanceGuard(mode=QualityMode.FAST, enforce_limit=True, enable_adaptive_skipping=True)
        balanced_guard = PerformanceGuard(mode=QualityMode.BALANCED, enforce_limit=True, enable_adaptive_skipping=True)
        quality_guard = PerformanceGuard(mode=QualityMode.QUALITY, enforce_limit=True, enable_adaptive_skipping=True)

        assert fast_guard.target_rt_factor == pytest.approx(8.0)
        assert balanced_guard.target_rt_factor == pytest.approx(32.0)  # v10.0.0: 32× RT
        assert quality_guard.target_rt_factor == pytest.approx(32.0)  # v10.0.0: 32× RT

    def test_start_phase_uses_monotonic_clock(self) -> None:
        guard = PerformanceGuard(mode=QualityMode.BALANCED, enforce_limit=True, enable_adaptive_skipping=True)
        guard.start_monitoring(10.0)
        phase_start = guard.start_phase("denoise")
        now = time.perf_counter()
        assert phase_start <= now
        assert now - phase_start < 1.0

    def test_quality_mode_can_skip_low_priority_near_budget(self) -> None:
        guard = PerformanceGuard(mode=QualityMode.QUALITY, enforce_limit=True, enable_adaptive_skipping=True)
        guard.start_monitoring(10.0)
        # Simulate that ~31.5x RT are already spent on a 10s file (near LIMIT_QUALITY=32.0).
        guard.start_time = time.perf_counter() - 315.0
        guard.audio_duration = 10.0

        # Lightweight phase (0.15× RT marginal) should NOT be skipped —
        # only heavy ML phases (> 0.5× RT marginal) get deferred to KMV Stufe 2.
        should_skip_light = guard.should_skip_phase(
            phase_id="metadata_embedding",
            estimated_time_seconds=1.5,
            remaining_phases=2,
        )
        assert should_skip_light is False

        # Heavy ML phase (0.8× RT marginal = 8s on 10s audio) SHOULD be skipped/deferred.
        # M6 (2026-11-17): unprofilierte Phasen sind CRITICAL (nie skipbar) —
        # die Skip-Probe nutzt daher eine PROFILIERTE MEDIUM-Phase.
        should_skip_heavy = guard.should_skip_phase(
            phase_id="stereo_enhancement",
            estimated_time_seconds=8.0,
            remaining_phases=2,
        )
        assert should_skip_heavy is True

    def test_quality_mode_rt_exceeded_no_early_exit_below_30min(self) -> None:
        """§2.38 KMV: RT-Limit-Überschreitung → Warnung, KEIN Abbruch.

        Nur das 30-Minuten-Absolutlimit darf die Pipeline abbrechen.
        """
        guard = PerformanceGuard(mode=QualityMode.QUALITY, enforce_limit=True, enable_adaptive_skipping=True)
        guard.start_monitoring(10.0)
        guard.current_rt_factor = 10.1

        # RT not yet exceeded (10.1 < LIMIT_QUALITY=32.0), <90min absolute → pipeline continues
        assert guard.check_early_exit(remaining_phases=3) is False

    def test_absolute_90min_limit_triggers_early_exit(self) -> None:
        """§9.5: 90-Minuten-Absolutlimit (5400s) ist der einzige harte Abbruchgrund (v10.0.0)."""
        guard = PerformanceGuard(mode=QualityMode.QUALITY, enforce_limit=True, enable_adaptive_skipping=True)
        guard.start_monitoring(10.0)
        # Simulate 5401 seconds elapsed (>90min)
        guard.start_time = time.perf_counter() - 5401.0
        guard.current_rt_factor = 540.0

        assert guard.check_early_exit(remaining_phases=3) is True

    def test_musical_excellence_phase_never_skipped_even_near_limit(self) -> None:
        guard = PerformanceGuard(mode=QualityMode.BALANCED, enforce_limit=True, enable_adaptive_skipping=True)
        guard.start_monitoring(10.0)
        guard.start_time = time.perf_counter() - 79.0
        guard.audio_duration = 10.0

        should_skip = guard.should_skip_phase(
            phase_id="excellence_optimizer",
            estimated_time_seconds=3.0,
            remaining_phases=1,
        )
        assert should_skip is False

    def test_runtime_mandatory_phase_never_skipped_under_rt_pressure(self) -> None:
        guard = PerformanceGuard(mode=QualityMode.QUALITY, enforce_limit=True, enable_adaptive_skipping=True)
        guard.start_monitoring(10.0)
        guard.start_time = time.perf_counter() - 315.0
        guard.audio_duration = 10.0

        # Baseline: heavy non-mandatory phase is deferred near the RT budget.
        # M6 (2026-11-17): unprofilierte Phasen sind CRITICAL (nie skipbar) —
        # die Baseline-Probe nutzt eine PROFILIERTE MEDIUM-Phase.
        should_skip_non_mandatory = guard.should_skip_phase(
            phase_id="stereo_enhancement",
            estimated_time_seconds=8.0,
            remaining_phases=2,
        )
        assert should_skip_non_mandatory is True

        # §6.2a runtime guard: material mandatory phase must never be skipped.
        guard.set_never_skip_phases(["phase_09_crackle_removal"])
        should_skip_mandatory = guard.should_skip_phase(
            phase_id="phase_09_crackle_removal",
            estimated_time_seconds=8.0,
            remaining_phases=2,
        )
        assert should_skip_mandatory is False


class TestDK337BudgetWahrheit:
    """D-K3-37 (geschlossen 2026-11-07): Die 32×-Norm ist ehrlich deklariert.

    Durchsetzung (`enforce_limit=True`) ist Opt-in; der Standardpfad ist
    MONITORING (budget_violations, kein Abbruch) mit dem UI-Watchdog als
    Enforcement-Oberfläche. Aussagen über Laufzeit-Begrenzung sind NUR bei
    aktivem enforce_limit belegt.
    """

    def test_guard_docstring_deklariert_monitoring_wahrheit(self):
        import pathlib

        src = (pathlib.Path(__file__).resolve().parents[2] / "backend" / "core" / "performance_guard.py").read_text(
            encoding="utf-8"
        )
        assert "D-K3-37" in src
        assert "MONITORING-Instrument" in src
        assert "NUR bei aktivem enforce_limit belegt" in src

    def test_config_kommentar_nennt_budget_eigentuemer(self):
        import pathlib

        src = (pathlib.Path(__file__).resolve().parents[2] / "backend" / "core" / "unified_restorer_v3.py").read_text(
            encoding="utf-8"
        )
        assert "RT-Budget-Durchsetzung (32×-Norm §2.38 KMV)" in src
        assert "Budget-Eigentümer" in src
        assert 'Flag-Name "3x" ist Legacy' in src
