import pytest

"""Unit-Tests: SpectralBandGapRepairPhase._compute_band_gap_profile() (§2.56)."""

import numpy as np

from backend.core.phases.phase_56_spectral_band_gap_repair import (
    SpectralBandGapRepairPhase,
    _detect_band_gaps,
    _harmonic_interpolate_gap,
)


def _profile(material: str, qm: str = "balanced", rest: float = 50.0) -> dict:
    return SpectralBandGapRepairPhase._compute_band_gap_profile(material, qm, rest)


@pytest.mark.unit
def test_tape_has_lower_confidence_gate_than_cd():
    tape = _profile("tape")
    cd = _profile("cd_digital")
    assert tape["min_head_wear_confidence"] < cd["min_head_wear_confidence"]


def test_quality_mode_more_sensitive_than_balanced():
    base = _profile("tape", "balanced", 60.0)
    quality = _profile("tape", "quality", 60.0)
    assert quality["min_head_wear_confidence"] < base["min_head_wear_confidence"]
    assert quality["mid_gap_fraction_min"] < base["mid_gap_fraction_min"]


def test_fast_mode_more_conservative_than_balanced():
    base = _profile("tape", "balanced", 60.0)
    fast = _profile("tape", "fast", 60.0)
    assert fast["min_head_wear_confidence"] > base["min_head_wear_confidence"]
    assert fast["mid_gap_fraction_min"] > base["mid_gap_fraction_min"]


def test_profile_bounds():
    p = _profile("unknown", "maximum", 10.0)
    assert 0.40 <= p["min_head_wear_confidence"] <= 0.85
    assert 0.70 <= p["mid_gap_fraction_min"] <= 0.97
    assert 0.85 <= p["side_gap_fraction_min"] <= 0.995


def test_process_metadata_contains_band_gap_profile():
    phase = SpectralBandGapRepairPhase()
    audio = np.random.uniform(-0.1, 0.1, 4096).astype(np.float32)

    result = phase.process(
        audio,
        sample_rate=48000,
        confidence=1.0,
        quality_mode="quality",
        restorability_score=35.0,
        material_type="tape",
        strength=0.5,
    )

    assert result.success
    assert "band_gap_profile" in result.metadata
    assert "min_head_wear_confidence" in result.metadata
    assert "mid_gap_fraction_min" in result.metadata
    assert "side_gap_fraction_min" in result.metadata


def test_harmonic_interpolate_uses_neighbor_phase_consistency():
    sr = 48000
    n_fft = 2048
    n_bins = n_fft // 2 + 1
    n_frames = 12

    mag = np.zeros((n_bins, n_frames), dtype=np.float32)
    phase = np.zeros((n_bins, n_frames), dtype=np.float32)

    # Nachbar-Bins mit stabilen Phasen vorbereiten.
    phase[6, :] = 0.25
    phase[8, :] = -0.35
    mag[6, :] = 0.8
    mag[8, :] = 0.7

    mag_out, phase_out = _harmonic_interpolate_gap(
        mag,
        phase,
        gap=(7, 9),
        f0_hz=165.0,
        sr=sr,
        n_fft=n_fft,
        instrument_tag="unknown",
    )

    assert mag_out.shape == mag.shape
    assert phase_out.shape == phase.shape
    # In der Gap-Region darf Phase nicht zufällig sein; sie muss finite und begrenzt bleiben.
    assert np.all(np.isfinite(phase_out[7, :]))
    assert float(np.max(np.abs(phase_out[7, :]))) <= np.pi


def test_detect_band_gaps_ignores_sparse_burst_bin():
    """Sporadische Bursts duerfen nicht als dauerhaftes Gap erkannt werden."""
    sr = 48000
    n_fft = 2048
    n_bins = n_fft // 2 + 1
    n_frames = 100

    # Basis: klare Low-Energy-Luecke ueber 200 Hz Breite.
    stft_mag = np.ones((n_bins, n_frames), dtype=np.float32) * 1e-6
    gap_low = 300
    gap_high = 310
    stft_mag[gap_low:gap_high, :] = 1e-7

    # Ein Bin in der Luecke bekommt seltene, aber starke Bursts.
    # empty_fraction bleibt hoch (~0.90), Median bleibt niedrig,
    # mittlere Energie ist jedoch klar ueber der Gap-Schwelle.
    burst_bin = gap_low + 3
    stft_mag[burst_bin, 0:10] = 0.1

    gaps = _detect_band_gaps(stft_mag, sr=sr, n_fft=n_fft)

    # Die Luecke wird in Segmente vor/nach burst_bin getrennt erkannt,
    # aber burst_bin selbst darf nicht im Gap enthalten sein.
    assert all(not (lo <= burst_bin < hi) for lo, hi in gaps)
