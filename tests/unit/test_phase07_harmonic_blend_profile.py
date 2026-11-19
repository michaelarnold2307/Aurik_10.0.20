from __future__ import annotations

"""Unit tests for Phase 07 _compute_harmonic_blend_profile (§2.54 adaptive constants)."""


import pytest


@pytest.fixture(scope="module")
def phase07():
    from backend.core.phases.phase_07_harmonic_restoration import HarmonicRestorationPhase

    return HarmonicRestorationPhase()


# ---------------------------------------------------------------------------
# 1. Output-Bound-Check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "material,mode,rest",
    [
        ("vinyl", "balanced", 75.0),
        ("shellac", "fast", 20.0),
        ("mp3_low", "quality", 90.0),
        ("reel_tape", "maximum", 50.0),
        ("digital", "balanced", 0.0),
        ("unknown", "fast", 100.0),
    ],
)
def test_profile_bounds(phase07, material, mode, rest):
    p = phase07._compute_harmonic_blend_profile(material, mode, rest)
    assert 0.30 <= p["ddsp_blend_factor"] <= 0.65, f"ddsp_blend_factor out of range: {p}"
    assert 0.20 <= p["ddsp_wet_cap"] <= 0.55, f"ddsp_wet_cap out of range: {p}"
    assert 0.25 <= p["fill_gain_factor"] <= 0.58, f"fill_gain_factor out of range: {p}"


# ---------------------------------------------------------------------------
# 2. Fast-vs-quality directional check
# ---------------------------------------------------------------------------


def test_fast_more_conservative_than_quality(phase07):
    fast = phase07._compute_harmonic_blend_profile("vinyl", "fast", 75.0)
    qual = phase07._compute_harmonic_blend_profile("vinyl", "quality", 75.0)
    assert fast["ddsp_blend_factor"] < qual["ddsp_blend_factor"]
    assert fast["fill_gain_factor"] < qual["fill_gain_factor"]
    assert fast["ddsp_wet_cap"] < qual["ddsp_wet_cap"]


# ---------------------------------------------------------------------------
# 3. Low-vs-high restorability directional check
# ---------------------------------------------------------------------------


def test_low_restorability_more_conservative(phase07):
    low = phase07._compute_harmonic_blend_profile("vinyl", "balanced", 10.0)
    high = phase07._compute_harmonic_blend_profile("vinyl", "balanced", 90.0)
    assert low["ddsp_blend_factor"] <= high["ddsp_blend_factor"]
    assert low["fill_gain_factor"] <= high["fill_gain_factor"]
