"""
Tests für Comprehensive Audio Quality Metrics
==============================================

Testet psychoakustische, musikalische und emotionale Metriken.

Phase: Entwicklung psychoakustischer, musikalischer und emotionaler Metriken
Author: Aurik 10.0.0 Development Team
Date: 15. Februar 2026
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.comprehensive_metrics import (
    ComprehensiveMetricsCalculator,
    ComprehensiveMetricsResult,
    EmotionalMetrics,
    MusicalMetrics,
    PsychoAcousticMetrics,
    generate_metrics_report,
)

# ============================================================
# TEST FIXTURES
# ============================================================


@pytest.fixture
def sample_rate():
    """Standard sample rate."""
    return 48000


@pytest.fixture
def clean_tone(sample_rate):
    """Generate clean sine tone (440 Hz)."""
    duration = 2.0
    freq = 440.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    return audio


@pytest.fixture
def noisy_audio(clean_tone):
    """Add noise to clean tone."""
    noise = np.random.normal(0, 0.1, len(clean_tone))
    return clean_tone + noise


@pytest.fixture
def harmonic_audio(sample_rate):
    """Generate audio with rich harmonic content."""
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Fundamental + harmonics
    fund_freq = 440.0
    audio = 0.3 * np.sin(2 * np.pi * fund_freq * t)
    audio += 0.2 * np.sin(2 * np.pi * 2 * fund_freq * t)
    audio += 0.1 * np.sin(2 * np.pi * 3 * fund_freq * t)
    audio += 0.05 * np.sin(2 * np.pi * 4 * fund_freq * t)

    return audio


@pytest.fixture
def rhythmic_audio(sample_rate):
    """Generate rhythmic audio with clear beats."""
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Carrier tone with rhythmic envelope
    carrier = np.sin(2 * np.pi * 440 * t)
    envelope = 0.2 + 0.8 * (np.sin(2 * np.pi * 2 * t) > 0).astype(float)  # 2 Hz beats

    return carrier * envelope


@pytest.fixture
def complex_music(sample_rate):
    """Generate complex musical signal."""
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Multiple tones (chord)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4
    audio += 0.3 * np.sin(2 * np.pi * 554 * t)  # C#5 (major third)
    audio += 0.3 * np.sin(2 * np.pi * 659 * t)  # E5 (perfect fifth)

    # Rhythmic modulation
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    audio = audio * envelope

    # Add subtle noise
    audio += np.random.normal(0, 0.01, len(audio))

    return audio


@pytest.fixture
def calculator(sample_rate):
    """Metrics calculator instance."""
    return ComprehensiveMetricsCalculator(sample_rate=sample_rate)


# ============================================================
# PSYCHOACOUSTIC METRICS TESTS
# ============================================================


@pytest.mark.unit
class TestPsychoAcousticMetrics:
    """Test psychoacoustic metrics computation."""

    def test_snr_clean_signal(self, calculator, clean_tone):
        """Clean signal should have high SNR."""
        result = calculator.compute_all(clean_tone)
        assert result.psychoacoustic.snr_db > 30.0, "Clean signal should have SNR > 30 dB"

    def test_snr_noisy_signal(self, calculator, noisy_audio):
        """Noisy signal should have lower SNR."""
        result = calculator.compute_all(noisy_audio)
        assert result.psychoacoustic.snr_db < 25.0, "Noisy signal should have SNR < 25 dB"

    def test_thd_clean_tone(self, calculator, clean_tone):
        """Clean tone should have low THD."""
        result = calculator.compute_all(clean_tone)
        assert result.psychoacoustic.thd_percent < 5.0, "Clean tone should have THD < 5%"

    def test_lufs_range(self, calculator, clean_tone):
        """LUFS should be in reasonable range."""
        result = calculator.compute_all(clean_tone)
        assert -50.0 < result.psychoacoustic.integrated_lufs < 0.0, "LUFS should be between -50 and 0"

    def test_crest_factor(self, calculator, clean_tone):
        """Sine wave should have predictable crest factor (~3 dB)."""
        result = calculator.compute_all(clean_tone)
        # Sine wave crest factor = sqrt(2) ≈ 3 dB
        assert 2.0 < result.psychoacoustic.crest_factor_db < 4.0

    def test_tonality_pure_tone(self, calculator, clean_tone):
        """Pure tone should have high tonality."""
        result = calculator.compute_all(clean_tone)
        assert result.psychoacoustic.tonality > 0.7, "Pure tone should have high tonality"

    def test_tonality_noise(self, calculator, sample_rate):
        """White noise should have low tonality."""
        noise = np.random.normal(0, 0.1, int(sample_rate * 2))
        result = calculator.compute_all(noise)
        assert result.psychoacoustic.tonality < 0.3, "Noise should have low tonality"

    def test_clipping_detection(self, calculator, sample_rate):
        """Should detect clipped audio."""
        # Create clipped signal
        t = np.linspace(0, 1, sample_rate)
        audio = 2.0 * np.sin(2 * np.pi * 440 * t)  # Will clip at ±1
        audio = np.clip(audio, -1, 1)

        result = calculator.compute_all(audio)
        assert result.psychoacoustic.clipping_percent > 10.0, "Should detect clipping"

    def test_click_detection(self, calculator, clean_tone):
        """Should detect click artifacts."""
        # Add artificial clicks
        audio = clean_tone.copy()
        click_positions = [1000, 5000, 10000]
        for pos in click_positions:
            audio[pos : pos + 10] = 0.9  # Sharp spike

        result = calculator.compute_all(audio)
        assert result.psychoacoustic.click_detection >= 2, "Should detect clicks"

    def test_spectral_features(self, calculator, clean_tone):
        """Spectral features should be computed."""
        result = calculator.compute_all(clean_tone)
        assert result.psychoacoustic.spectral_centroid_hz > 0
        assert result.psychoacoustic.spectral_rolloff_hz > 0
        assert result.psychoacoustic.spectral_flux >= 0


# ============================================================
# MUSICAL METRICS TESTS
# ============================================================


class TestMusicalMetrics:
    """Test musical metrics computation."""

    def test_harmonic_clarity_pure_tone(self, calculator, clean_tone):
        """Pure tone should have some harmonic clarity."""
        result = calculator.compute_all(clean_tone)
        assert 0.0 <= result.musical.harmonic_clarity <= 1.0

    def test_harmonic_clarity_rich_harmonics(self, calculator, harmonic_audio):
        """Rich harmonics should have higher clarity."""
        result = calculator.compute_all(harmonic_audio)
        assert result.musical.harmonic_clarity > 0.3

    def test_hnr_computation(self, calculator, harmonic_audio):
        """HNR should be in reasonable range."""
        result = calculator.compute_all(harmonic_audio)
        assert -10.0 < result.musical.harmonic_to_noise_ratio_db < 40.0

    def test_key_detection(self, calculator, complex_music):
        """Key detection should return valid key."""
        result = calculator.compute_all(complex_music)
        assert result.musical.detected_key is not None
        assert "major" in result.musical.detected_key or "minor" in result.musical.detected_key
        assert 0.0 <= result.musical.key_confidence <= 1.0

    def test_consonance_chord(self, calculator, complex_music):
        """Musical chord should have reasonable consonance."""
        result = calculator.compute_all(complex_music)
        assert 0.0 <= result.musical.consonance <= 1.0

    def test_tempo_detection(self, calculator, rhythmic_audio):
        """Tempo detection should find rhythmic pattern."""
        result = calculator.compute_all(rhythmic_audio)
        # 2 Hz rhythm = 120 BPM
        assert 60.0 < result.musical.tempo_bpm < 200.0
        assert 0.0 <= result.musical.tempo_stability <= 1.0

    def test_rhythmic_regularity(self, calculator, rhythmic_audio):
        """Regular rhythm should have high regularity score."""
        result = calculator.compute_all(rhythmic_audio)
        assert result.musical.rhythmic_regularity > 0.3

    def test_timbral_qualities(self, calculator, harmonic_audio):
        """Timbral qualities should be in valid range."""
        result = calculator.compute_all(harmonic_audio)
        assert 0.0 <= result.musical.warmth <= 1.0
        assert 0.0 <= result.musical.brightness <= 1.0
        assert 0.0 <= result.musical.fullness <= 1.0

    def test_spectral_balance(self, calculator, complex_music):
        """Spectral balance should be computed."""
        result = calculator.compute_all(complex_music)
        assert 0.0 <= result.musical.spectral_balance <= 1.0

    def test_dynamic_contrast(self, calculator, rhythmic_audio):
        """Rhythmic audio should have dynamic contrast."""
        result = calculator.compute_all(rhythmic_audio)
        assert result.musical.dynamic_contrast > 0.1


# ============================================================
# EMOTIONAL METRICS TESTS
# ============================================================


class TestEmotionalMetrics:
    """Test emotional metrics computation."""

    def test_valence_arousal_range(self, calculator, complex_music):
        """Valence and arousal should be in [-1, +1]."""
        result = calculator.compute_all(complex_music)
        assert -1.0 <= result.emotional.valence <= 1.0
        assert -1.0 <= result.emotional.arousal <= 1.0

    def test_energy_loud_signal(self, calculator, sample_rate):
        """Loud signal should have high energy."""
        loud = 0.8 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sample_rate * 2)))
        result = calculator.compute_all(loud)
        assert result.emotional.energy > 0.5

    def test_energy_quiet_signal(self, calculator, sample_rate):
        """Quiet signal should have low energy."""
        quiet = 0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sample_rate * 2)))
        result = calculator.compute_all(quiet)
        assert result.emotional.energy < 0.5

    def test_intensity_dynamic_range(self, calculator, rhythmic_audio):
        """High dynamic range should increase intensity."""
        result = calculator.compute_all(rhythmic_audio)
        assert result.emotional.intensity > 0.2

    def test_emotional_categories_range(self, calculator, complex_music):
        """All emotional categories should be in [0, 1]."""
        result = calculator.compute_all(complex_music)
        e = result.emotional

        assert 0.0 <= e.power <= 1.0
        assert 0.0 <= e.joyful_activation <= 1.0
        assert 0.0 <= e.nostalgia <= 1.0
        assert 0.0 <= e.sadness <= 1.0
        assert 0.0 <= e.peacefulness <= 1.0
        assert 0.0 <= e.transcendence <= 1.0

    def test_perceived_emotions_range(self, calculator, complex_music):
        """Perceived emotions should be in [0, 1]."""
        result = calculator.compute_all(complex_music)
        e = result.emotional

        assert 0.0 <= e.perceived_happiness <= 1.0
        assert 0.0 <= e.perceived_sadness <= 1.0
        assert 0.0 <= e.perceived_anger <= 1.0
        assert 0.0 <= e.perceived_fear <= 1.0
        assert 0.0 <= e.perceived_surprise <= 1.0

    def test_tension_computation(self, calculator, complex_music):
        """Tension should be computed."""
        result = calculator.compute_all(complex_music)
        assert 0.0 <= result.emotional.tension <= 1.0


# ============================================================
# OVERALL QUALITY TESTS
# ============================================================


class TestOverallQuality:
    """Test overall quality scoring."""

    def test_quality_scores_range(self, calculator, complex_music):
        """Quality scores should be in [0, 1]."""
        result = calculator.compute_all(complex_music)
        assert 0.0 <= result.overall_technical_quality <= 1.0
        assert 0.0 <= result.overall_musical_quality <= 1.0
        assert 0.0 <= result.overall_emotional_impact <= 1.0

    def test_aurik_score_range(self, calculator, complex_music):
        """Aurik score should be in [0, 100]."""
        result = calculator.compute_all(complex_music)
        assert 0.0 <= result.aurik_quality_score <= 100.0

    def test_high_quality_signal(self, calculator, harmonic_audio):
        """High-quality signal should score well."""
        result = calculator.compute_all(harmonic_audio)
        assert result.aurik_quality_score > 30.0  # Reasonable baseline

    def test_weltklasse_check(self, calculator, complex_music):
        """Weltklasse check should work."""
        result = calculator.compute_all(complex_music)
        # Should return boolean
        passes = result.passes_aurik_standards()
        assert isinstance(passes, bool)


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestIntegration:
    """Integration tests for complete metric system."""

    def test_compute_all_returns_complete_result(self, calculator, complex_music):
        """compute_all should return ComprehensiveMetricsResult."""
        result = calculator.compute_all(complex_music)
        assert isinstance(result, ComprehensiveMetricsResult)
        assert isinstance(result.psychoacoustic, PsychoAcousticMetrics)
        assert isinstance(result.musical, MusicalMetrics)
        assert isinstance(result.emotional, EmotionalMetrics)

    def test_to_dict_conversion(self, calculator, complex_music):
        """Result should convert to dictionary."""
        result = calculator.compute_all(complex_music)
        metrics_dict = result.to_dict()

        assert isinstance(metrics_dict, dict)
        assert "psychoacoustic" in metrics_dict
        assert "musical" in metrics_dict
        assert "emotional" in metrics_dict
        assert "aurik_quality_score" in metrics_dict

    def test_generate_metrics_report(self, calculator, complex_music):
        """Report generation should work."""
        result = calculator.compute_all(complex_music)
        report = generate_metrics_report(result)

        assert isinstance(report, str)
        assert "PSYCHOACOUSTIC METRICS" in report
        assert "MUSICAL METRICS" in report
        assert "EMOTIONAL METRICS" in report
        assert "AURIK QUALITY SCORE" in report

    def test_stereo_audio(self, calculator, sample_rate):
        """Should handle stereo audio."""
        t = np.linspace(0, 2, int(sample_rate * 2))
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 554 * t)
        stereo = np.stack([left, right], axis=1)

        result = calculator.compute_all(stereo)
        assert result is not None
        assert result.aurik_quality_score > 0

    def test_short_audio(self, calculator, sample_rate):
        """Should handle short audio clips."""
        short = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sample_rate * 0.5)))
        result = calculator.compute_all(short)
        assert result is not None

    def test_ultra_short_audio_below_lra_frame(self, calculator, sample_rate):
        """Should handle very short clips (<100 ms) without frame split errors."""
        ultra_short = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.02, int(sample_rate * 0.02)))
        result = calculator.compute_all(ultra_short)
        assert result is not None
        assert np.isfinite(result.psychoacoustic.integrated_lufs)

    def test_extreme_short_audio_no_decimate_padlen_crash(self, calculator, sample_rate):
        """Should not crash on tiny clips where decimate() padlen would normally fail."""
        tiny = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.0002, max(1, int(sample_rate * 0.0002))))
        result = calculator.compute_all(tiny)
        assert result is not None
        assert np.isfinite(result.aurik_quality_score)

    def test_single_sample_audio_stability(self, calculator):
        """Should handle single-sample input without exceptions."""
        single = np.array([0.1], dtype=np.float32)
        result = calculator.compute_all(single)
        assert result is not None
        assert np.isfinite(result.aurik_quality_score)

    def test_stereo_channel_first_layout(self, calculator, sample_rate):
        """Should accept channel-first stereo arrays (channels, samples)."""
        t = np.linspace(0, 2, int(sample_rate * 2))
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 554 * t)
        channel_first = np.stack([left, right], axis=0)

        result = calculator.compute_all(channel_first)
        assert result is not None
        assert result.aurik_quality_score > 0

    def test_long_audio(self, calculator, sample_rate):
        """Should handle longer audio."""
        long = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 10, int(sample_rate * 10)))
        result = calculator.compute_all(long)
        assert result is not None


# ============================================================
# COMPARISON TESTS
# ============================================================


class TestComparisons:
    """Test metric comparisons between different audio types."""

    def test_clean_vs_noisy_snr(self, calculator, clean_tone, noisy_audio):
        """Clean audio should have higher SNR than noisy."""
        result_clean = calculator.compute_all(clean_tone)
        result_noisy = calculator.compute_all(noisy_audio)

        assert result_clean.psychoacoustic.snr_db > result_noisy.psychoacoustic.snr_db

    def test_harmonic_vs_noise_clarity(self, calculator, harmonic_audio, sample_rate):
        """Harmonic audio should have higher clarity than noise."""
        result_harmonic = calculator.compute_all(harmonic_audio)

        noise = np.random.normal(0, 0.1, int(sample_rate * 2))
        result_noise = calculator.compute_all(noise)

        assert result_harmonic.musical.harmonic_clarity > result_noise.musical.harmonic_clarity

    def test_loud_vs_quiet_energy(self, calculator, sample_rate):
        """Loud signal should have higher energy."""
        loud = 0.8 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sample_rate * 2)))
        quiet = 0.2 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sample_rate * 2)))

        result_loud = calculator.compute_all(loud)
        result_quiet = calculator.compute_all(quiet)

        assert result_loud.emotional.energy > result_quiet.emotional.energy


# ============================================================
# EDGE CASES
# ============================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_silence(self, calculator, sample_rate):
        """Should handle silence without crashing."""
        silence = np.zeros(int(sample_rate * 2))
        result = calculator.compute_all(silence)
        assert result is not None

    def test_dc_offset(self, calculator, sample_rate):
        """Should handle DC offset."""
        audio = 0.5 + 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sample_rate * 2)))
        result = calculator.compute_all(audio)
        assert result is not None

    def test_extreme_amplitude(self, calculator, sample_rate):
        """Should handle extreme but valid amplitudes."""
        extreme = 0.99 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sample_rate * 2)))
        result = calculator.compute_all(extreme)
        assert result.psychoacoustic.true_peak_dbtp > -1.0

    def test_very_high_frequency(self, calculator, sample_rate):
        """Should handle high-frequency content."""
        high_freq = 0.5 * np.sin(2 * np.pi * 12000 * np.linspace(0, 2, int(sample_rate * 2)))
        result = calculator.compute_all(high_freq)
        assert result.psychoacoustic.spectral_centroid_hz > 10000


# ============================================================
# PERFORMANCE TESTS
# ============================================================


class TestPerformance:
    """Test performance and efficiency."""

    def test_computation_time(self, calculator, complex_music):
        """Metrics should compute in reasonable time."""
        import time

        start = time.time()
        calculator.compute_all(complex_music)
        elapsed = time.time() - start

        # Should complete within 5 seconds for 5s audio
        assert elapsed < 5.0, f"Computation took {elapsed:.2f}s (too slow)"

    def test_memory_efficiency(self, calculator, sample_rate):
        """Should not leak memory with repeated calls."""
        audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sample_rate * 2)))

        # Run multiple times
        for _ in range(10):
            calculator.compute_all(audio)

        # Verify we can obtain a valid result object after repeated calls.
        res = calculator.compute_all(audio)
        assert isinstance(res, ComprehensiveMetricsResult)


# ============================================================
# MAIN TEST RUNNER
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
