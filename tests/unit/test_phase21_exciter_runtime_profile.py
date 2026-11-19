from __future__ import annotations

"""Unit tests for Phase 21 _compute_exciter_runtime_profile (§2.54 adaptive constants)."""


import pytest


@pytest.fixture(scope="module")
def phase21():
    from backend.core.phases.phase_21_exciter import Exciter

    return Exciter()


# ---------------------------------------------------------------------------
# 1. Output-Bound-Check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "material,mode,rest",
    [
        ("cd_digital", "balanced", 75.0),
        ("shellac", "fast", 20.0),
        ("mp3_low", "quality", 90.0),
        ("reel_tape", "maximum", 50.0),
        ("vinyl", "balanced", 0.0),
        ("streaming", "fast", 100.0),
    ],
)
def test_profile_bounds(phase21, material, mode, rest):
    p = phase21._compute_exciter_runtime_profile(material, mode, rest)
    assert 1.80 <= p["saturation_drive"] <= 4.00, f"saturation_drive out of range: {p}"
    assert 0.18 <= p["odd_partial_blend"] <= 0.45, f"odd_partial_blend out of range: {p}"
    assert 0.45 <= p["harmonic_output_scale"] <= 0.85, f"harmonic_output_scale out of range: {p}"


# ---------------------------------------------------------------------------
# 2. Fast-vs-quality directional check
# ---------------------------------------------------------------------------


def test_fast_more_conservative_than_quality(phase21):
    fast = phase21._compute_exciter_runtime_profile("vinyl", "fast", 75.0)
    qual = phase21._compute_exciter_runtime_profile("vinyl", "quality", 75.0)
    assert fast["saturation_drive"] < qual["saturation_drive"]
    assert fast["odd_partial_blend"] < qual["odd_partial_blend"]
    assert fast["harmonic_output_scale"] < qual["harmonic_output_scale"]


# ---------------------------------------------------------------------------
# 3. Low-vs-high restorability directional check
# ---------------------------------------------------------------------------


def test_low_restorability_more_conservative(phase21):
    low = phase21._compute_exciter_runtime_profile("vinyl", "balanced", 10.0)
    high = phase21._compute_exciter_runtime_profile("vinyl", "balanced", 90.0)
    assert low["saturation_drive"] <= high["saturation_drive"]
    assert low["harmonic_output_scale"] <= high["harmonic_output_scale"]
