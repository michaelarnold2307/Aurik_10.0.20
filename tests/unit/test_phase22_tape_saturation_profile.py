from __future__ import annotations

"""Tests für _compute_tape_saturation_profile (§2.54 Phase 22)."""


import numpy as np
import pytest

from backend.core.phases.phase_22_tape_saturation import TapeSaturation


@pytest.fixture()
def phase() -> TapeSaturation:
    return TapeSaturation()


# ---------------------------------------------------------------------------
# 1. Bounds
# ---------------------------------------------------------------------------
MATERIALS = [
    "shellac",
    "vinyl",
    "tape",
    "reel_tape",
    "cassette",
    "cd_digital",
    "streaming",
    "mp3_low",
    "mp3_medium",
]
QUALITY_MODES = ["fast", "balanced", "quality", "maximum"]


@pytest.mark.parametrize("mat", MATERIALS)
@pytest.mark.parametrize("qm", QUALITY_MODES)
@pytest.mark.parametrize("rest", [0.0, 50.0, 100.0])
def test_profile_bounds(mat, qm, rest):
    p = TapeSaturation._compute_tape_saturation_profile(mat, qm, rest)
    assert 4.0 <= p["drive_gain_scalar"] <= 14.0
    assert 0.050 <= p["h2_scale"] <= 0.200
    assert 0.025 <= p["h3_scale"] <= 0.100
    assert 0.010 <= p["h4_scale"] <= 0.050
    assert 0.15 <= p["side_drive_fraction"] <= 0.50


# ---------------------------------------------------------------------------
# 2. Quality-mode directional: fast < quality for harmonic richness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mat", ["vinyl", "tape", "cd_digital"])
def test_profile_fast_less_than_quality(mat):
    fast = TapeSaturation._compute_tape_saturation_profile(mat, "fast", 75.0)
    qual = TapeSaturation._compute_tape_saturation_profile(mat, "quality", 75.0)
    assert fast["drive_gain_scalar"] < qual["drive_gain_scalar"]
    assert fast["h2_scale"] < qual["h2_scale"]
    assert fast["h3_scale"] < qual["h3_scale"]


# ---------------------------------------------------------------------------
# 3. Restorability directional: low restorability → less drive
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mat", ["vinyl", "shellac", "reel_tape"])
def test_profile_low_rest_less_drive(mat):
    low = TapeSaturation._compute_tape_saturation_profile(mat, "balanced", 10.0)
    high = TapeSaturation._compute_tape_saturation_profile(mat, "balanced", 90.0)
    assert low["drive_gain_scalar"] < high["drive_gain_scalar"]
    assert low["h2_scale"] < high["h2_scale"]


# ---------------------------------------------------------------------------
# 4. Vintage analog has higher baseline than lossy
# ---------------------------------------------------------------------------
def test_vintage_higher_drive_than_lossy():
    vintage = TapeSaturation._compute_tape_saturation_profile("reel_tape", "balanced", 75.0)
    lossy = TapeSaturation._compute_tape_saturation_profile("mp3_low", "balanced", 75.0)
    assert vintage["drive_gain_scalar"] > lossy["drive_gain_scalar"]


# ---------------------------------------------------------------------------
# 5. NaN/Inf safety
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rest", [-10.0, 0.0, 50.0, 100.0, 150.0])
def test_profile_no_nan(rest):
    p = TapeSaturation._compute_tape_saturation_profile("vinyl", "balanced", rest)
    for v in p.values():
        assert np.isfinite(v), f"Non-finite value for restorability={rest}: {v}"


# ---------------------------------------------------------------------------
# 6. Integration: profile keys present in PhaseResult metadata
# ---------------------------------------------------------------------------
def test_profile_in_metadata(phase):
    from backend.core.defect_scanner import MaterialType

    audio = np.random.uniform(-0.1, 0.1, 48000).astype(np.float32)
    result = phase.process(
        audio,
        48000,
        material=MaterialType.VINYL,
        quality_mode="balanced",
        restorability_score=75.0,
        strength=0.5,
    )
    assert result.success
    assert "tape_saturation_profile" in result.metadata
    p = result.metadata["tape_saturation_profile"]
    for key in ("drive_gain_scalar", "h2_scale", "h3_scale", "h4_scale", "side_drive_fraction"):
        assert key in p


# ---------------------------------------------------------------------------
# 7. Integration stereo: side_drive_fraction propagated correctly
# ---------------------------------------------------------------------------
def test_stereo_profile_applied(phase):
    from backend.core.defect_scanner import MaterialType

    audio = np.random.uniform(-0.1, 0.1, (48000, 2)).astype(np.float32)
    result = phase.process(
        audio,
        48000,
        material=MaterialType.REEL_TAPE,
        quality_mode="maximum",
        restorability_score=80.0,
        strength=0.7,
    )
    assert result.success
    assert result.audio.shape == (48000, 2)
