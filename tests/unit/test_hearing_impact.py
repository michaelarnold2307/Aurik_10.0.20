"""§PERF-R R7 (2026-09-18) — Hör-Impact-Schätzer + Guard-Integration.

Der PerformanceGuard deferiert unter Budgetdruck künftig hör-bewusst:
Null-Impact-Phasen (kein hörbarer Ziel-Defekt auf diesem Audio) werden
vorgezogen deferred (reine Laufzeit-Ersparnis, PSY-A1-Passthrough),
High-Impact-Phasen (hörbarer Ziel-Defekt, Score ≥ 0.6) werden geschützt.
Ohne Scores bleibt die bisherige Fix-Priorität bit-identisch aktiv.
"""

from __future__ import annotations

import time

import pytest

from backend.core.dsp.hearing_impact import (
    HIGH_IMPACT_THRESHOLD,
    ZERO_IMPACT_THRESHOLD,
    classify_hearing_impact,
    estimate_phase_hearing_impact,
)
from backend.core.performance_guard import PerformanceGuard, QualityMode


@pytest.mark.unit
def test_estimate_impact_uses_max_mapped_score():
    """R7: Impact = max Score der der Phase zugeordneten DefectTypes."""
    from backend.core.defect_scanner import DefectType

    scores = {DefectType.CLICKS: 0.8, DefectType.TRANSIENT_SMEARING: 0.3}
    # CLICKS → phase_01 (primär); TRANSIENT_SMEARING → phase_36
    impact = estimate_phase_hearing_impact("phase_01_click_removal", scores)
    assert impact == pytest.approx(0.8)


@pytest.mark.unit
def test_estimate_impact_string_keys_normalized():
    """R7: String-Schlüssel (Scanner-Format) werden auf DefectType normalisiert."""
    impact = estimate_phase_hearing_impact("phase_36_transient_shaper", {"transient_smearing": 0.55})
    assert impact == pytest.approx(0.55)


@pytest.mark.unit
def test_estimate_impact_zero_score_mapped():
    """R7: Zugeordneter Defekt mit Score 0.0 ⇒ Impact 0.0 (Null-Impact-Kandidat)."""
    impact = estimate_phase_hearing_impact("phase_36_transient_shaper", {"transient_smearing": 0.0})
    assert impact == pytest.approx(0.0)


@pytest.mark.unit
def test_estimate_impact_unmapped_phase_returns_none():
    """R7: Phase ohne Zuordnung ⇒ None (Fix-Priorität bleibt aktiv)."""
    from backend.core.defect_scanner import DefectType

    impact = estimate_phase_hearing_impact("phase_99_keine_zuordnung", {DefectType.CLICKS: 0.9})
    assert impact is None


@pytest.mark.unit
def test_estimate_impact_empty_scores_returns_none():
    """R7: Keine Scores ⇒ None — kein R7-Eingriff (bit-identisch zum Status quo)."""
    assert estimate_phase_hearing_impact("phase_01_click_removal", {}) is None
    assert estimate_phase_hearing_impact("phase_01_click_removal", None) is None


@pytest.mark.unit
def test_classify():
    assert classify_hearing_impact(None) == "unknown"
    assert classify_hearing_impact(0.0) == "zero"
    assert classify_hearing_impact(0.05) == "zero"
    assert classify_hearing_impact(0.3) == "normal"
    assert classify_hearing_impact(0.6) == "high"
    assert classify_hearing_impact(1.0) == "high"


@pytest.mark.unit
def test_guard_zero_impact_deferral_under_pressure():
    """R7: Budgetdruck + Null-Impact ⇒ vorgezogenes Deferral (auch für
    Phasen, die die Fix-Prioritäts-Logik NICHT skippen würde)."""
    guard = PerformanceGuard(mode=QualityMode.QUALITY, enable_adaptive_skipping=True)
    guard.start_monitoring(30.0)
    guard.set_defect_scores({"transient_smearing": 0.0})
    # Simuliere 0.6× Target-Druck: elapsed ≈ 19.2 s … Ziel 32× → 0.60× = 19.2
    guard.start_time = time.perf_counter() - (0.60 * guard.target_rt_factor * 30.0) - 5.0
    assert guard.should_skip_phase("phase_36_transient_shaper", estimated_time_seconds=1.0, remaining_phases=10)
    assert "phase_36_transient_shaper" in guard.skipped_phases


@pytest.mark.unit
def test_guard_zero_impact_no_pressure_no_deferral():
    """R7: Ohne Budgetdruck bleibt der Null-Impact-Pfad aktiv (kein vorzeitiger Skip)."""
    guard = PerformanceGuard(mode=QualityMode.QUALITY, enable_adaptive_skipping=True)
    guard.start_monitoring(30.0)
    guard.set_defect_scores({"transient_smearing": 0.0})
    # frisch gestartet ⇒ current_rt ≈ 0
    assert not guard.should_skip_phase("phase_36_transient_shaper", estimated_time_seconds=1.0, remaining_phases=10)


@pytest.mark.unit
def test_guard_high_impact_protection():
    """R7: Fix-Priorität würde skippen (Projektion über Schwelle), aber der
    hörbare Ziel-Defekt (Impact ≥ 0.6) schützt die Phase."""
    guard = PerformanceGuard(mode=QualityMode.QUALITY, enable_adaptive_skipping=True)
    guard.start_monitoring(30.0)
    guard.set_defect_scores({"transient_smearing": 0.8})
    # current_rt nahe am Skip-Threshold (MEDIUM ⇒ 0.93× 32 = 29.76) — elapsed 860 s ⇒ 28.7×
    guard.start_time = time.perf_counter() - 860.0
    # estimated_time groß ⇒ Projektion (860 + 120 + 3)/30 = 32.8 > 29.76 ⇒ Skip der Fix-Logik
    assert not guard.should_skip_phase(
        "phase_36_transient_shaper", estimated_time_seconds=120.0, remaining_phases=10
    ), "High-Impact-Phase muss unter Budgetdruck geschützt werden"


@pytest.mark.unit
def test_guard_without_scores_keeps_fixed_priority():
    """R7: Ohne Scores skippt die Fix-Prioritäts-Logik wie bisher (gleiche Bedingungen)."""
    guard = PerformanceGuard(mode=QualityMode.QUALITY, enable_adaptive_skipping=True)
    guard.start_monitoring(30.0)
    guard.start_time = time.perf_counter() - 860.0
    # Ohne R7-Schutz ⇒ Fix-Logik greift (Projektion über Schwelle).
    assert guard.should_skip_phase("phase_36_transient_shaper", estimated_time_seconds=120.0, remaining_phases=10)


@pytest.mark.unit
def test_guard_critical_phase_never_deferred_even_zero_impact():
    """R7: Kritische Phasen (Priorität ≥ 9) bleiben trotz Null-Impact geschützt."""
    guard = PerformanceGuard(mode=QualityMode.QUALITY, enable_adaptive_skipping=True)
    guard.start_monitoring(30.0)
    guard.set_defect_scores({"clicks": 0.0})  # phase_01 ist CRITICAL (10)
    guard.start_time = time.perf_counter() - (0.60 * guard.target_rt_factor * 30.0) - 5.0
    assert not guard.should_skip_phase("phase_01_click_removal", estimated_time_seconds=10.0, remaining_phases=10)


@pytest.mark.unit
def test_thresholds_conservative():
    """R7: Schwellen bleiben konservativ (zwischen zero/high: Fix-Priorität)."""
    assert 0.0 < ZERO_IMPACT_THRESHOLD <= 0.05
    assert HIGH_IMPACT_THRESHOLD >= 0.6
