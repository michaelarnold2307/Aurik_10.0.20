"""
Tests for Professional Audio Metering
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.dsp.professional_meters import (
    LUFSMeter,
    MeterV9,
    PhaseCorrelationMeter,
    SpectrumAnalyzer,
    TruePeakDetector,
    meter_audio,
)


@pytest.fixture
def test_audio_mono():
    """Generate 1s test tone (1 kHz, -20 dBFS)"""
    sr = 48000
    duration = 1.0
    freq = 1000
    t = np.linspace(0, duration, int(sr * duration))

    # -20 dBFS amplitude
    amplitude = 10 ** (-20 / 20)
    audio = amplitude * np.sin(2 * np.pi * freq * t)

    return audio.astype(np.float32), sr


@pytest.fixture
def test_audio_stereo():
    """Generate 1s stereo test tone"""
    sr = 48000
    duration = 1.0
    freq = 1000
    t = np.linspace(0, duration, int(sr * duration))

    amplitude = 10 ** (-20 / 20)
    left = amplitude * np.sin(2 * np.pi * freq * t)
    right = amplitude * np.sin(2 * np.pi * freq * t + np.pi / 4)  # 45° phase shift

    audio = np.stack([left, right])
    return audio.astype(np.float32), sr


def test_lufs_meter_mono(test_audio_mono):
    """Test LUFS measurement on mono signal"""
    audio, sr = test_audio_mono

    meter = LUFSMeter(sr)
    result = meter.measure(audio, sr, gating=False)

    assert "integrated_lufs" in result
    assert -30 < result["integrated_lufs"] < -10  # Should be around -20 LUFS
    print(f"\nMono LUFS: {result['integrated_lufs']:.2f}")


def test_lufs_meter_stereo(test_audio_stereo):
    """Test LUFS measurement on stereo signal"""
    audio, sr = test_audio_stereo

    meter = LUFSMeter(sr)
    result = meter.measure(audio, sr, gating=False)

    assert "integrated_lufs" in result
    assert -30 < result["integrated_lufs"] < -10
    print(f"\nStereo LUFS: {result['integrated_lufs']:.2f}")


def test_lufs_meter_with_gating(test_audio_mono):
    """Test LUFS with gating"""
    audio, sr = test_audio_mono

    meter = LUFSMeter(sr)
    result = meter.measure(audio, sr, gating=True)

    assert "integrated_lufs" in result
    print(f"\nGated LUFS: {result['integrated_lufs']:.2f}")


def test_true_peak_detector(test_audio_mono):
    """Test True Peak detection"""
    audio, sr = test_audio_mono

    detector = TruePeakDetector(sr)
    result = detector.measure(audio, sr)

    assert "true_peak_db" in result
    assert "sample_peak_db" in result
    assert "true_peak_exceeded" in result

    # For sine wave, true peak should be close to -20 dB
    assert -25 < result["true_peak_db"] < -15
    print(f"\nTrue Peak: {result['true_peak_db']:.2f} dBTP")
    print(f"Sample Peak: {result['sample_peak_db']:.2f} dBFS")


def test_phase_correlation_mono(test_audio_mono):
    """Test phase correlation on mono (should return None)"""
    audio, sr = test_audio_mono

    meter = PhaseCorrelationMeter()
    result = meter.measure(audio, sr)

    assert result["phase_correlation"] is None


def test_phase_correlation_stereo(test_audio_stereo):
    """Test phase correlation on stereo"""
    audio, sr = test_audio_stereo

    meter = PhaseCorrelationMeter()
    result = meter.measure(audio, sr)

    assert "phase_correlation" in result
    assert result["phase_correlation"] is not None
    assert -1.0 <= result["phase_correlation"] <= 1.0
    assert 0.0 <= result["phase_coherence"] <= 1.0

    print(f"\nPhase Correlation: {result['phase_correlation']:.3f}")
    print(f"Phase Coherence: {result['phase_coherence']:.3f}")


def test_spectrum_analyzer(test_audio_mono):
    """Test spectrum analysis"""
    audio, sr = test_audio_mono

    analyzer = SpectrumAnalyzer(sr)
    result = analyzer.analyze(audio, sr)

    assert "freqs" in result
    assert "power_db" in result
    assert "spectral_centroid" in result
    assert "spectral_rolloff" in result

    # For 1 kHz tone, centroid should be near 1 kHz
    assert 900 < result["spectral_centroid"] < 1100
    print(f"\nSpectral Centroid: {result['spectral_centroid']:.1f} Hz")
    print(f"Spectral Rolloff: {result['spectral_rolloff']:.1f} Hz")


def test_professional_meter_complete(test_audio_stereo):
    """Test complete professional metering"""
    audio, sr = test_audio_stereo

    meter = MeterV9(sr)
    result = meter.analyze(audio, sr, verbose=False)

    # Check all attributes exist
    assert hasattr(result, "integrated_lufs")
    assert hasattr(result, "true_peak_db")
    assert hasattr(result, "sample_peak_db")
    assert hasattr(result, "phase_correlation")
    assert hasattr(result, "spectral_centroid")

    # Validate ranges
    assert result.integrated_lufs > -100
    assert -30 < result.true_peak_db < 0  # Should be negative for -20 dBFS signal
    assert result.phase_correlation is not None


def test_meter_audio_convenience(test_audio_mono):
    """Test convenience function"""
    audio, sr = test_audio_mono

    result = meter_audio(audio, sr, verbose=False)

    assert result is not None
    assert hasattr(result, "integrated_lufs")


def test_peak_limiter_detection():
    """Test detection of peaks exceeding -1 dBTP"""
    sr = 48000
    duration = 0.1
    t = np.linspace(0, duration, int(sr * duration))

    # Create signal with peak at 0 dBFS (exceeds -1 dBTP)
    audio = np.sin(2 * np.pi * 1000 * t)

    detector = TruePeakDetector(sr)
    result = detector.measure(audio, sr)

    assert result["true_peak_exceeded"] is True
    print(f"\nPeak exceeded test: {result['true_peak_db']:.2f} dBTP")


def test_anti_phase_detection():
    """Test detection of anti-phase stereo content"""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Create perfect anti-phase stereo (L = -R)
    left = np.sin(2 * np.pi * 1000 * t)
    right = -left
    audio = np.stack([left, right]) * 0.5

    meter = PhaseCorrelationMeter()
    result = meter.measure(audio, sr)

    # Should be close to -1.0
    assert result["phase_correlation"] < -0.9
    print(f"\nAnti-phase correlation: {result['phase_correlation']:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
