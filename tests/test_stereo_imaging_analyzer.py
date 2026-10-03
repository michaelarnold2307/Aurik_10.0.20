"""
Tests for Stereo Imaging Analyzer & Fixer (GAP #21)

Tests:
- Phase correlation analysis
- Stereo width computation
- Balance correction
- Mid/Side encoding/decoding
- Width adjustment
- Phase cancellation fix
- Auto-correction
- Edge cases
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.dsp.stereo_imaging_analyzer import StereoImagingAnalyzer, StereoImagingFixer


@pytest.mark.unit
class TestPhaseCorrelationAnalysis:
    """Test phase correlation analysis."""

    def test_perfect_correlation_mono(self):
        """Mono signal should have correlation near +1."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Identical left and right (mono)
        mono = 0.5 * np.sin(2 * np.pi * 440 * t)
        left = mono.copy()
        right = mono.copy()

        analyzer = StereoImagingAnalyzer()
        metrics = analyzer.analyze_phase_correlation(left, right, sr)

        # Should be close to +1
        assert metrics["phase_correlation_mean"] > 0.95
        assert metrics["problematic_frames_ratio"] == 0.0

    def test_no_correlation_stereo(self):
        """Uncorrelated channels should have correlation near 0."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Completely different signals
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 880 * t)

        analyzer = StereoImagingAnalyzer()
        metrics = analyzer.analyze_phase_correlation(left, right, sr)

        # Should be close to 0
        assert -0.3 < metrics["phase_correlation_mean"] < 0.3

    def test_anti_correlation_phase_reversed(self):
        """Phase-reversed signal should have correlation near -1."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Right is inverted left
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = -left

        analyzer = StereoImagingAnalyzer()
        metrics = analyzer.analyze_phase_correlation(left, right, sr)

        # Should be close to -1
        assert metrics["phase_correlation_mean"] < -0.95
        assert metrics["problematic_frames_ratio"] > 0.9  # Most frames problematic

    def test_detects_problematic_frames(self):
        """Should detect frames with phase cancellation."""
        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # First half: good correlation, second half: phase reversed
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = left.copy()
        right[len(right) // 2 :] *= -1  # Invert second half

        analyzer = StereoImagingAnalyzer()
        metrics = analyzer.analyze_phase_correlation(left, right, sr)

        # Should detect some problematic frames
        assert 0.3 < metrics["problematic_frames_ratio"] < 0.7


class TestStereoWidthComputation:
    """Test stereo width computation."""

    def test_mono_has_zero_width(self):
        """Mono signal should have width near 0."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        mono = 0.5 * np.sin(2 * np.pi * 440 * t)
        left = mono.copy()
        right = mono.copy()

        analyzer = StereoImagingAnalyzer()
        width = analyzer.compute_stereo_width(left, right)

        # Should be very small (near 0)
        assert width < 0.1

    def test_normal_stereo_width(self):
        """Normal stereo should have width around 1."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Different frequencies in L/R
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 880 * t)

        analyzer = StereoImagingAnalyzer()
        width = analyzer.compute_stereo_width(left, right)

        # Should be measurable width
        assert width > 0.3

    def test_enhanced_stereo_width(self):
        """Enhanced stereo should have width > 1."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Mid signal
        mid = 0.3 * np.sin(2 * np.pi * 440 * t)
        # Strong side signal
        side = 0.7 * np.sin(2 * np.pi * 1000 * t)

        left = mid + side
        right = mid - side

        analyzer = StereoImagingAnalyzer()
        width = analyzer.compute_stereo_width(left, right)

        # Should have enhanced width
        assert width > 1.5


class TestBalanceComputation:
    """Test left/right balance computation."""

    def test_perfect_balance(self):
        """Equal L/R should have balance near 0 dB."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 880 * t)

        analyzer = StereoImagingAnalyzer()
        balance = analyzer.compute_balance(left, right)

        # Should be close to 0 dB
        assert abs(balance["balance_db"]) < 0.5

    def test_left_louder(self):
        """Left louder should have positive balance."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        left = 0.8 * np.sin(2 * np.pi * 440 * t)
        right = 0.4 * np.sin(2 * np.pi * 880 * t)

        analyzer = StereoImagingAnalyzer()
        balance = analyzer.compute_balance(left, right)

        # Left is ~6 dB louder (2× amplitude)
        assert balance["balance_db"] > 5.0
        assert balance["balance_ratio"] > 1.5

    def test_right_louder(self):
        """Right louder should have negative balance."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        left = 0.3 * np.sin(2 * np.pi * 440 * t)
        right = 0.6 * np.sin(2 * np.pi * 880 * t)

        analyzer = StereoImagingAnalyzer()
        balance = analyzer.compute_balance(left, right)

        # Right is louder
        assert balance["balance_db"] < -5.0


class TestMidSideEncoding:
    """Test Mid/Side encoding and decoding."""

    def test_encode_decode_roundtrip(self):
        """Encode + decode should preserve signal."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 880 * t)

        fixer = StereoImagingFixer()

        # Encode
        mid, side = fixer.encode_mid_side(left, right)

        # Decode
        left_decoded, right_decoded = fixer.decode_mid_side(mid, side)

        # Should match original
        np.testing.assert_allclose(left_decoded, left, rtol=1e-10)
        np.testing.assert_allclose(right_decoded, right, rtol=1e-10)

    def test_mono_has_no_side(self):
        """Mono signal should have zero side content."""
        mono = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 48000))

        fixer = StereoImagingFixer()
        mid, side = fixer.encode_mid_side(mono, mono)

        # Mid should equal mono
        np.testing.assert_allclose(mid, mono, rtol=1e-10)

        # Side should be zero
        np.testing.assert_allclose(side, 0.0, atol=1e-10)


class TestWidthAdjustment:
    """Test stereo width adjustment."""

    def test_reduce_width(self):
        """Should reduce stereo width when factor < 1."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create wide stereo
        mid = 0.5 * np.sin(2 * np.pi * 440 * t)
        side = 0.5 * np.sin(2 * np.pi * 1000 * t)
        left = mid + side
        right = mid - side

        # Measure original width
        analyzer = StereoImagingAnalyzer()
        width_before = analyzer.compute_stereo_width(left, right)

        # Reduce width
        fixer = StereoImagingFixer()
        left_narrow, right_narrow = fixer.adjust_width(left, right, width_factor=0.5)

        # Measure new width
        width_after = analyzer.compute_stereo_width(left_narrow, right_narrow)

        # Should be reduced
        assert width_after < width_before

    def test_enhance_width(self):
        """Should enhance stereo width when factor > 1."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create narrow stereo
        mid = 0.7 * np.sin(2 * np.pi * 440 * t)
        side = 0.2 * np.sin(2 * np.pi * 1000 * t)
        left = mid + side
        right = mid - side

        # Measure original width
        analyzer = StereoImagingAnalyzer()
        width_before = analyzer.compute_stereo_width(left, right)

        # Enhance width
        fixer = StereoImagingFixer()
        left_wide, right_wide = fixer.adjust_width(left, right, width_factor=2.0)

        # Measure new width
        width_after = analyzer.compute_stereo_width(left_wide, right_wide)

        # Should be enhanced
        assert width_after > width_before

    def test_preserves_mono_when_width_zero(self):
        """Width factor of 0 should create mono."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 880 * t)

        fixer = StereoImagingFixer()
        left_mono, right_mono = fixer.adjust_width(left, right, width_factor=0.0)

        # Should be identical (mono)
        np.testing.assert_allclose(left_mono, right_mono, rtol=1e-5)


class TestBalanceCorrection:
    """Test balance correction."""

    def test_corrects_left_bias(self):
        """Should correct left-biased audio."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Left much louder
        left = 0.8 * np.sin(2 * np.pi * 440 * t)
        right = 0.4 * np.sin(2 * np.pi * 880 * t)

        analyzer = StereoImagingAnalyzer()
        balance_before = analyzer.compute_balance(left, right)

        # Correct to balanced
        fixer = StereoImagingFixer()
        left_corrected, right_corrected = fixer.correct_balance(left, right, target_balance_db=0.0)

        balance_after = analyzer.compute_balance(left_corrected, right_corrected)

        # Should be closer to 0 dB
        assert abs(balance_after["balance_db"]) < abs(balance_before["balance_db"])
        assert abs(balance_after["balance_db"]) < 1.0

    def test_no_clipping_after_correction(self):
        """Should not introduce clipping."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Imbalanced but with headroom
        left = 0.7 * np.sin(2 * np.pi * 440 * t)
        right = 0.3 * np.sin(2 * np.pi * 880 * t)

        fixer = StereoImagingFixer()
        left_corrected, right_corrected = fixer.correct_balance(left, right, target_balance_db=0.0)

        # Should not clip
        assert np.max(np.abs(left_corrected)) <= 1.0
        assert np.max(np.abs(right_corrected)) <= 1.0


class TestFullProcessing:
    """Test full auto-correction pipeline."""

    def test_auto_corrects_multiple_issues(self):
        """Should auto-correct width, balance, and phase."""
        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create audio with multiple problems:
        # - Narrow stereo
        # - Imbalanced (left louder)
        mono_content = 0.7 * np.sin(2 * np.pi * 440 * t)
        left = mono_content * 1.5 + 0.05 * np.random.randn(len(mono_content))
        right = mono_content * 0.8 + 0.02 * np.random.randn(len(mono_content))

        audio = np.column_stack([left, right])

        # Process with auto-correction
        fixer = StereoImagingFixer(target_width=1.0, balance_tolerance_db=1.0)
        audio_fixed, metrics = fixer.process(audio, sr, auto_correct=True)

        # Should have applied corrections
        assert metrics["num_corrections"] > 0

        # Balance should be improved
        if abs(metrics["before"]["balance_db"]) > 2.0:
            assert abs(metrics["after"]["balance_db"]) < abs(metrics["before"]["balance_db"])

    def test_no_correction_for_good_audio(self):
        """Should not correct already good audio."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Good stereo audio
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 880 * t)

        audio = np.column_stack([left, right])

        fixer = StereoImagingFixer(target_width=1.0, balance_tolerance_db=1.0)
        audio_fixed, metrics = fixer.process(audio, sr, auto_correct=True)

        # Should apply few or no corrections
        assert metrics["num_corrections"] <= 1


class TestStereoFormatSupport:
    """Test support for different stereo formats."""

    def test_channels_first_format(self):
        """Should handle (channels, samples) format."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 880 * t)

        audio = np.vstack([left, right])  # (2, samples)

        analyzer = StereoImagingAnalyzer()
        metrics = analyzer.analyze(audio, sr)

        # Should analyze correctly
        assert "phase_correlation_mean" in metrics
        assert "stereo_width" in metrics

    def test_channels_last_format(self):
        """Should handle (samples, channels) format."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 880 * t)

        audio = np.column_stack([left, right])  # (samples, 2)

        analyzer = StereoImagingAnalyzer()
        metrics = analyzer.analyze(audio, sr)

        # Should analyze correctly
        assert "phase_correlation_mean" in metrics
        assert "stereo_width" in metrics

    def test_preserves_input_format(self):
        """Should return audio in same format as input."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        left = 0.5 * np.sin(2 * np.pi * 440 * t) * 1.5
        right = 0.5 * np.sin(2 * np.pi * 880 * t) * 0.8

        # Test channels-first
        audio_cf = np.vstack([left, right])
        fixer = StereoImagingFixer()
        processed_cf, _ = fixer.process(audio_cf, sr)
        assert processed_cf.shape == audio_cf.shape
        assert processed_cf.shape[0] == 2

        # Test channels-last
        audio_cl = np.column_stack([left, right])
        processed_cl, _ = fixer.process(audio_cl, sr)
        assert processed_cl.shape == audio_cl.shape
        assert processed_cl.shape[1] == 2


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_handles_silent_audio(self):
        """Should handle silent audio without errors."""
        sr = 48000
        duration = 1.0

        silent = np.zeros((int(sr * duration), 2))

        analyzer = StereoImagingAnalyzer()
        metrics = analyzer.analyze(silent, sr)

        # Should not crash
        assert "stereo_width" in metrics

    def test_handles_very_short_audio(self):
        """Should handle very short audio clips."""
        sr = 48000
        duration = 0.05  # 50ms
        t = np.linspace(0, duration, int(sr * duration))

        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 880 * t)

        audio = np.column_stack([left, right])

        analyzer = StereoImagingAnalyzer()
        metrics = analyzer.analyze(audio, sr)

        # Should not crash
        assert "stereo_width" in metrics

    def test_phase_correction_preserves_length(self):
        """Phase correction should preserve audio length."""
        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = -left  # Phase reversed

        fixer = StereoImagingFixer(target_phase_correlation_min=-0.3)
        left_fixed, right_fixed = fixer.fix_phase_cancellation(left, right, sr)

        # Length should be preserved (or close)
        assert abs(len(left_fixed) - len(left)) <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
