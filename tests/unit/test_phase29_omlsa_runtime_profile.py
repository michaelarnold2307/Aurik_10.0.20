from __future__ import annotations

"""Tests für _compute_omlsa_runtime_profile (§2.54 Phase 29)."""


import numpy as np
import pytest

from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase

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
    "wax_cylinder",
]
QUALITY_MODES = ["fast", "balanced", "quality", "maximum"]


@pytest.mark.parametrize("mat", MATERIALS)
@pytest.mark.parametrize("qm", QUALITY_MODES)
@pytest.mark.parametrize("rest", [0.0, 50.0, 100.0])
def test_profile_bounds(mat, qm, rest):
    p = TapeHissReductionPhase._compute_omlsa_runtime_profile(mat, qm, rest)
    assert 1.40 <= p["imcra_b_min"] <= 1.90
    assert 0.75 <= p["imcra_alpha_g"] <= 0.92
    assert 0.35 <= p["omlsa_q"] <= 0.65
    assert 0.30 <= p["hf_floor_scale"] <= 0.65


# ---------------------------------------------------------------------------
# 2. Quality-mode directional: fast < maximum for smoothing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mat", ["tape", "vinyl", "shellac"])
def test_fast_less_smooth_than_maximum(mat):
    fast = TapeHissReductionPhase._compute_omlsa_runtime_profile(mat, "fast", 75.0)
    maxq = TapeHissReductionPhase._compute_omlsa_runtime_profile(mat, "maximum", 75.0)
    assert fast["imcra_alpha_g"] < maxq["imcra_alpha_g"]
    assert fast["imcra_b_min"] < maxq["imcra_b_min"]


# ---------------------------------------------------------------------------
# 3. Restorability directional: low rest → more smoothing (higher alpha_g)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mat", ["tape", "reel_tape", "shellac"])
def test_low_rest_more_smoothing(mat):
    low = TapeHissReductionPhase._compute_omlsa_runtime_profile(mat, "balanced", 10.0)
    high = TapeHissReductionPhase._compute_omlsa_runtime_profile(mat, "balanced", 90.0)
    assert low["imcra_alpha_g"] > high["imcra_alpha_g"]
    assert low["imcra_b_min"] > high["imcra_b_min"]


# ---------------------------------------------------------------------------
# 4. Heavy-hiss material has higher q (more aggressive)
# ---------------------------------------------------------------------------
def test_shellac_higher_q_than_streaming():
    shellac = TapeHissReductionPhase._compute_omlsa_runtime_profile("shellac", "balanced", 75.0)
    stream = TapeHissReductionPhase._compute_omlsa_runtime_profile("streaming", "balanced", 75.0)
    assert shellac["omlsa_q"] > stream["omlsa_q"]


# ---------------------------------------------------------------------------
# 5. NaN/Inf safety
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rest", [-10.0, 0.0, 50.0, 100.0, 150.0])
def test_profile_no_nan(rest):
    p = TapeHissReductionPhase._compute_omlsa_runtime_profile("tape", "balanced", rest)
    for v in p.values():
        assert np.isfinite(v)


# ---------------------------------------------------------------------------
# 6. Integration: profile keys in metadata (tape material)
# ---------------------------------------------------------------------------
def test_profile_in_metadata():
    from backend.core.defect_scanner import MaterialType

    phase = TapeHissReductionPhase()
    audio = np.random.uniform(-0.05, 0.05, 48000).astype(np.float32)
    result = phase.process(
        audio,
        48000,
        material=MaterialType.TAPE,
        quality_mode="balanced",
        restorability_score=70.0,
        strength=0.5,
    )
    assert result.success
    assert "omlsa_runtime_profile" in result.metadata
    p = result.metadata["omlsa_runtime_profile"]
    for key in ("imcra_b_min", "imcra_alpha_g", "omlsa_q", "hf_floor_scale"):
        assert key in p


# ---------------------------------------------------------------------------
# 7. Integration: stereo passes through without crash
# ---------------------------------------------------------------------------
def test_stereo_no_crash():
    from backend.core.defect_scanner import MaterialType

    phase = TapeHissReductionPhase()
    audio = np.random.uniform(-0.05, 0.05, (48000, 2)).astype(np.float32)
    result = phase.process(
        audio,
        48000,
        material=MaterialType.REEL_TAPE,
        quality_mode="quality",
        restorability_score=60.0,
        strength=0.6,
    )
    assert result.success
    assert result.audio.shape == (48000, 2)


def test_phase29_uses_npd_singleton_accessor(monkeypatch):
    from backend.core.defect_scanner import MaterialType

    phase = TapeHissReductionPhase()
    audio = np.random.uniform(-0.05, 0.05, 48000).astype(np.float32)
    calls = {"count": 0}

    class _NpdStub:
        def detect(self, _audio, _sr):
            calls["count"] += 1
            return None

    monkeypatch.setattr(
        "backend.core.phases.phase_29_tape_hiss_reduction._get_phase29_npd",
        lambda: _NpdStub(),
    )

    result = phase.process(
        audio,
        48000,
        material=MaterialType.TAPE,
        quality_mode="balanced",
        restorability_score=70.0,
        strength=0.5,
    )

    assert result.success
    assert calls["count"] >= 1
