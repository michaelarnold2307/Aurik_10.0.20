import numpy as np
import pytest

from backend.core.calibration_matrix import (
    blend_targets_with_confidence,
    compute_cpb,
    compute_export_reliability,
    compute_goal_coverage_index,
    compute_ibs,
    compute_recovery_pressure_index,
    compute_reference_confidence,
    compute_retry_temperature,
    compute_tcci,
)


@pytest.mark.unit
def test_compute_tcci_is_bounded_and_chain_sensitive():
    shallow = compute_tcci(["cd_digital"])
    deep_lossy = compute_tcci(["shellac", "reel_tape", "cassette", "cd_digital", "mp3_low"])
    assert 0.0 <= shallow <= 1.0
    assert 0.0 <= deep_lossy <= 1.0
    assert deep_lossy > shallow


def test_compute_ibs_is_bounded():
    value = compute_ibs(restorability=35.0, defect_severity_mean=0.8, tcci=0.7)
    assert 0.15 <= value <= 0.95


def test_blend_targets_with_confidence_moves_toward_canonical_when_low_conf():
    canonical = {"brillanz": 0.78, "transparenz": 0.82}
    song = {"brillanz": 0.90, "transparenz": 0.90}
    low_conf = blend_targets_with_confidence(canonical, song, 0.1, 0.1, 0.1)
    high_conf = blend_targets_with_confidence(canonical, song, 0.9, 0.9, 0.9)

    assert canonical["brillanz"] <= low_conf["brillanz"] <= high_conf["brillanz"] <= song["brillanz"]
    assert canonical["transparenz"] <= low_conf["transparenz"] <= high_conf["transparenz"] <= song["transparenz"]


def test_compute_cpb_restoration_is_more_conservative_than_studio():
    rest = compute_cpb(material_ceiling=16.0, current_value=13.0, mode="restoration")
    studio = compute_cpb(material_ceiling=16.0, current_value=13.0, mode="studio_2026")
    assert 0.0 <= studio <= 16.0
    assert 0.0 <= rest <= 16.0
    assert rest >= studio


def test_retry_temperature_is_bounded():
    t = compute_retry_temperature(restorability=20.0, tcci=0.8, artifact_freedom_score=0.9)
    assert 0.0 <= t <= 1.0


def test_export_reliability_is_bounded_and_improves_with_better_inputs():
    low = compute_export_reliability(
        hpi=0.2,
        artifact_freedom=0.8,
        passed_goals=6,
        total_goals=14,
        reference_confidence=0.5,
    )
    high = compute_export_reliability(
        hpi=0.6,
        artifact_freedom=0.96,
        passed_goals=11,
        total_goals=14,
        reference_confidence=0.8,
    )

    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low
    assert np.isfinite(high)


def test_goal_coverage_index_is_weighted_and_bounded():
    poor = compute_goal_coverage_index(
        {
            "natuerlichkeit": False,
            "authentizitaet": False,
            "brillanz": True,
            "raumtiefe": True,
        }
    )
    good = compute_goal_coverage_index(
        {
            "natuerlichkeit": True,
            "authentizitaet": True,
            "brillanz": False,
            "raumtiefe": False,
        }
    )
    assert 0.0 <= poor <= 1.0
    assert 0.0 <= good <= 1.0
    assert good > poor


def test_reference_confidence_penalizes_complexity_and_heavy_carrier_recovery():
    stable = compute_reference_confidence(target_confidence=0.85, tcci=0.10, carrier_chain_recovery_ratio=0.05)
    unstable = compute_reference_confidence(target_confidence=0.85, tcci=0.80, carrier_chain_recovery_ratio=0.40)
    assert 0.0 <= stable <= 1.0
    assert 0.0 <= unstable <= 1.0
    assert stable > unstable


def test_recovery_pressure_index_grows_with_attempts_rollbacks_and_deficit():
    low = compute_recovery_pressure_index(fallback_attempts=0, rollback_count=0, goal_deficit_ratio=0.1)
    high = compute_recovery_pressure_index(fallback_attempts=2, rollback_count=6, goal_deficit_ratio=0.7)
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low
