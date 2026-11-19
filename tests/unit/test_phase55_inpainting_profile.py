import pytest

"""Unit-Tests: DiffusionInpaintingPhase._compute_inpainting_profile() (§2.56)."""

import numpy as np

from backend.core.phases.phase_55_diffusion_inpainting import DiffusionInpaintingPhase


def _profile(material: str, qm: str = "balanced", rest: float = 50.0) -> dict:
    return DiffusionInpaintingPhase._compute_inpainting_profile(material, qm, rest)


@pytest.mark.unit
def test_analog_uses_higher_min_gap_than_digital():
    wax = _profile("wax_cylinder")
    cd = _profile("cd_digital")
    assert wax["min_gap_ms"] > cd["min_gap_ms"]


def test_quality_mode_relaxes_gate_and_extends_budget():
    base = _profile("tape", "balanced", 60.0)
    quality = _profile("tape", "quality", 60.0)
    assert quality["min_gap_ms"] < base["min_gap_ms"]
    assert quality["wall_budget_seconds"] > base["wall_budget_seconds"]


def test_fast_mode_tightens_runtime_budget():
    base = _profile("tape", "balanced", 60.0)
    fast = _profile("tape", "fast", 60.0)
    assert fast["min_gap_ms"] > base["min_gap_ms"]
    assert fast["wall_budget_seconds"] < base["wall_budget_seconds"]


def test_profile_bounds():
    p = _profile("unknown", "maximum", 10.0)
    assert 8.0 <= p["min_gap_ms"] <= 80.0
    assert 40.0 <= p["wall_budget_seconds"] <= 240.0


def test_process_metadata_contains_inpainting_profile(monkeypatch):
    phase = DiffusionInpaintingPhase()
    audio = np.zeros(2048, dtype=np.float32)

    monkeypatch.setattr(
        "backend.core.phases.phase_55_diffusion_inpainting._detect_gaps",
        lambda *_args, **_kwargs: [],
    )

    result = phase.process(
        audio,
        sample_rate=48000,
        quality_mode="quality",
        restorability_score=35.0,
        material_type="tape",
        strength=0.5,
    )

    assert result.success
    assert "inpainting_profile" in result.metadata
    assert "min_gap_ms" in result.metadata
    assert "wall_budget_seconds" in result.metadata
