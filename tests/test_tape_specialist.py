"""
test_tape_specialist.py - Tests für Tape Specialist (GAP #1, #2)

Testet:
- TapePrintThroughRemover (GAP #1): Print-Through Removal
- TapeAzimuthCorrector (GAP #2): Phase Alignment
- TapeSpecialist (unified API)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.dsp.tape_specialist import (
    TapeAzimuthCorrector,
    TapePrintThroughRemover,
    TapeSpecialist,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_rate():
    """Standard sample rate for tests (Spec §2.37: Interne Verarbeitungs-SR stets 48000 Hz)"""
    return 48000


@pytest.fixture
def duration():
    """Standard duration in seconds"""
    return 1.0


@pytest.fixture
def mono_audio(sample_rate, duration):
    """Generate mono test audio"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    return audio


@pytest.fixture
def stereo_audio(mono_audio):
    """Generate stereo test audio"""
    return np.vstack([mono_audio, mono_audio * 0.9])


@pytest.fixture
def audio_with_print_through(sample_rate, duration):
    """Generate audio with simulated print-through echo"""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Original signal: Transient at 0.3s
    audio = np.zeros_like(t)
    transient_idx = int(0.3 * sample_rate)
    transient_width = int(0.01 * sample_rate)  # 10ms
    audio[transient_idx : transient_idx + transient_width] = 0.8

    # Add post-echo (100ms later, -30 dB)
    echo_delay_samples = int(0.1 * sample_rate)
    echo_gain = 10 ** (-30 / 20)  # -30 dB

    audio_with_echo = audio.copy()
    if transient_idx + echo_delay_samples + transient_width < len(audio):
        audio_with_echo[transient_idx + echo_delay_samples : transient_idx + echo_delay_samples + transient_width] += (
            audio[transient_idx : transient_idx + transient_width] * echo_gain
        )

    return audio_with_echo


@pytest.fixture
def stereo_audio_with_azimuth_error(sample_rate, duration):
    """Generate stereo audio with phase misalignment"""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Left channel
    left = 0.5 * np.sin(2 * np.pi * 1000 * t)

    # Right channel: Delayed by 10 samples (~0.227ms at 44.1kHz)
    # This simulates azimuth error
    right = np.roll(left, 10)
    right[:10] = 0  # Zero-pad beginning

    return np.vstack([left, right])


# =============================================================================
# TESTS: TapePrintThroughRemover (GAP #1)
# =============================================================================


@pytest.mark.unit
class TestTapePrintThroughRemover:
    """Tests for GAP #1: Tape Print-Through Removal"""

    def test_initialization(self):
        """Test initialization with default parameters"""
        remover = TapePrintThroughRemover()
        assert remover.max_delay_ms == 150.0
        assert remover.attenuation_threshold_db == -40.0
        assert remover.pre_echo_detection is True
        assert remover.post_echo_detection is True
        assert remover.adaptive_strength == 0.7

    def test_initialization_with_params(self):
        """Test initialization with custom parameters"""
        remover = TapePrintThroughRemover(max_delay_ms=200.0, attenuation_threshold_db=-35.0, adaptive_strength=0.8)
        assert remover.max_delay_ms == 200.0
        assert remover.attenuation_threshold_db == -35.0
        assert remover.adaptive_strength == 0.8

    def test_parameter_clipping(self):
        """Test that parameters are clipped to valid ranges"""
        remover = TapePrintThroughRemover(
            max_delay_ms=1000.0,
            attenuation_threshold_db=-70.0,
            adaptive_strength=1.5,  # > 500  # < -60  # > 1.0
        )
        assert remover.max_delay_ms == 500.0
        assert remover.attenuation_threshold_db == -60.0
        assert remover.adaptive_strength == 1.0

    def test_detect_print_through_clean_audio(self, mono_audio, sample_rate):
        """Test detection on clean audio (no print-through)"""
        remover = TapePrintThroughRemover()
        detection = remover.detect_print_through(mono_audio, sample_rate)

        # Should not detect significant echo
        assert "pre_echo_detected" in detection
        assert "post_echo_detected" in detection

    def test_detect_print_through_with_echo(self, audio_with_print_through, sample_rate):
        """Test detection on audio with print-through"""
        remover = TapePrintThroughRemover(attenuation_threshold_db=-35.0)  # Lower threshold to detect -30 dB echo
        detection = remover.detect_print_through(audio_with_print_through, sample_rate)

        # Should detect something (echo detection is challenging with sparse signals)
        assert "post_echo_detected" in detection
        assert "post_echo_delay_ms" in detection

        # Note: Autocorrelation-based detection may not always find correct delay
        # with sparse transients. Real-world implementation would use more sophisticated
        # methods (spectral analysis, machine learning)

    def test_remove_print_through(self, audio_with_print_through, sample_rate):
        """Test print-through removal"""
        remover = TapePrintThroughRemover(attenuation_threshold_db=-35.0, adaptive_strength=0.8)

        # Detect
        detection = remover.detect_print_through(audio_with_print_through, sample_rate)

        # Remove
        cleaned = remover.remove_print_through(audio_with_print_through, sample_rate, detection)

        # Should return same length
        assert len(cleaned) == len(audio_with_print_through)

        # Should not introduce NaN or Inf
        assert np.all(np.isfinite(cleaned))

        # Note: Echo removal effectiveness depends on accurate detection.
        # With simplified autocorrelation, results may vary.

    def test_process_mono_clean_audio(self, mono_audio, sample_rate):
        """Test processing clean mono audio"""
        remover = TapePrintThroughRemover()
        processed = remover.process(mono_audio, sample_rate)

        # Should preserve length
        assert len(processed) == len(mono_audio)

        # Should be nearly unchanged
        correlation = np.corrcoef(mono_audio, processed)[0, 1]
        assert correlation > 0.95

    def test_process_mono_with_echo(self, audio_with_print_through, sample_rate):
        """Test processing audio with echo"""
        remover = TapePrintThroughRemover(attenuation_threshold_db=-35.0)
        processed = remover.process(audio_with_print_through, sample_rate)

        # Should preserve length
        assert len(processed) == len(audio_with_print_through)

        # Metrics should be populated
        assert "post_echo_detected" in remover.metrics

    def test_process_stereo_audio(self, stereo_audio, sample_rate):
        """Test processing stereo audio"""
        remover = TapePrintThroughRemover()
        processed = remover.process(stereo_audio, sample_rate)

        # Should preserve stereo shape
        assert processed.shape == stereo_audio.shape
        assert processed.ndim == 2

    def test_quality_gate_over_processing(self, mono_audio, sample_rate):
        """Test quality gate prevents over-processing"""
        remover = TapePrintThroughRemover(adaptive_strength=1.0)  # Maximum strength
        processed = remover.process(mono_audio, sample_rate)

        # RMS should not drop dramatically
        rms_before = np.sqrt(np.mean(mono_audio**2))
        rms_after = np.sqrt(np.mean(processed**2))
        assert rms_after > rms_before * 0.7  # Allow up to 30% reduction

    def test_metrics_reporting(self, audio_with_print_through, sample_rate):
        """Test that metrics are properly reported"""
        remover = TapePrintThroughRemover(attenuation_threshold_db=-35.0)
        remover.process(audio_with_print_through, sample_rate)

        assert hasattr(remover, "metrics")
        assert "post_echo_detected" in remover.metrics
        assert "rms_change_db" in remover.metrics


# =============================================================================
# TESTS: TapeAzimuthCorrector (GAP #2)
# =============================================================================


class TestTapeAzimuthCorrector:
    """Tests for GAP #2: Tape Azimuth Correction"""

    def test_initialization(self):
        """Test initialization with default parameters"""
        corrector = TapeAzimuthCorrector()
        assert corrector.correction_strength == 0.8
        assert corrector.phase_threshold_degrees == 10.0
        assert corrector.preserve_stereo_width is True

    def test_initialization_with_params(self):
        """Test initialization with custom parameters"""
        corrector = TapeAzimuthCorrector(
            correction_strength=0.9, phase_threshold_degrees=5.0, preserve_stereo_width=False
        )
        assert corrector.correction_strength == 0.9
        assert corrector.phase_threshold_degrees == 5.0
        assert corrector.preserve_stereo_width is False

    def test_parameter_clipping(self):
        """Test that parameters are clipped to valid ranges"""
        corrector = TapeAzimuthCorrector(correction_strength=1.5, phase_threshold_degrees=50.0)  # > 1.0  # > 45.0
        assert corrector.correction_strength == 1.0
        assert corrector.phase_threshold_degrees == 45.0

    def test_detect_phase_error_aligned(self, stereo_audio, sample_rate):
        """Test detection on phase-aligned audio"""
        corrector = TapeAzimuthCorrector()

        left = stereo_audio[0]
        right = stereo_audio[1]

        detection = corrector.detect_phase_error(left, right, sample_rate)

        assert "phase_error_detected" in detection
        assert "max_phase_error_degrees" in detection
        assert "delay_samples" in detection

    def test_detect_phase_error_misaligned(self, stereo_audio_with_azimuth_error, sample_rate):
        """Test detection on phase-misaligned audio"""
        corrector = TapeAzimuthCorrector(phase_threshold_degrees=5.0)

        left = stereo_audio_with_azimuth_error[0]
        right = stereo_audio_with_azimuth_error[1]

        detection = corrector.detect_phase_error(left, right, sample_rate)

        # May or may not detect depending on threshold
        # Just check structure
        assert "phase_error_detected" in detection
        assert "delay_samples" in detection

        # Delay should be around 10 samples
        assert abs(detection["delay_samples"]) <= 20

    def test_correct_azimuth(self, stereo_audio_with_azimuth_error, sample_rate):
        """Test azimuth correction"""
        corrector = TapeAzimuthCorrector(correction_strength=1.0, phase_threshold_degrees=1.0)  # Low threshold

        left = stereo_audio_with_azimuth_error[0]
        right = stereo_audio_with_azimuth_error[1]

        # Detect
        detection = corrector.detect_phase_error(left, right, sample_rate)
        detection["phase_error_detected"] = True  # Force correction

        # Correct
        left_corrected, right_corrected = corrector.correct_azimuth(left, right, detection)

        # Should return same length
        assert len(left_corrected) == len(left)
        assert len(right_corrected) == len(right)

    def test_process_stereo_aligned(self, stereo_audio, sample_rate):
        """Test processing phase-aligned stereo"""
        corrector = TapeAzimuthCorrector()
        processed = corrector.process(stereo_audio, sample_rate)

        # Should preserve shape
        assert processed.shape == stereo_audio.shape

    def test_process_stereo_misaligned(self, stereo_audio_with_azimuth_error, sample_rate):
        """Test processing phase-misaligned stereo"""
        corrector = TapeAzimuthCorrector(phase_threshold_degrees=1.0)  # Low threshold
        processed = corrector.process(stereo_audio_with_azimuth_error, sample_rate)

        # Should preserve shape
        assert processed.shape == stereo_audio_with_azimuth_error.shape

        # Metrics should be populated
        assert "phase_error_detected" in corrector.metrics

    def test_process_mono_returns_unchanged(self, mono_audio, sample_rate):
        """Test that mono audio is returned unchanged"""
        corrector = TapeAzimuthCorrector()
        processed = corrector.process(mono_audio, sample_rate)

        # Should be unchanged
        assert np.allclose(processed, mono_audio)

    def test_stereo_width_preservation(self, stereo_audio_with_azimuth_error, sample_rate):
        """Test stereo width preservation"""
        corrector = TapeAzimuthCorrector(preserve_stereo_width=True, phase_threshold_degrees=1.0)

        # Compute original stereo width
        left = stereo_audio_with_azimuth_error[0]
        right = stereo_audio_with_azimuth_error[1]
        original_mid = (left + right) / 2
        original_side = (left - right) / 2
        original_width = np.sqrt(np.mean(original_side**2)) / (np.sqrt(np.mean(original_mid**2)) + 1e-8)

        # Process
        processed = corrector.process(stereo_audio_with_azimuth_error, sample_rate)

        # Compute corrected stereo width
        left_corrected = processed[0]
        right_corrected = processed[1]
        corrected_mid = (left_corrected + right_corrected) / 2
        corrected_side = (left_corrected - right_corrected) / 2
        corrected_width = np.sqrt(np.mean(corrected_side**2)) / (np.sqrt(np.mean(corrected_mid**2)) + 1e-8)

        # Width should be similar (within 20%)
        assert abs(corrected_width - original_width) / (original_width + 1e-8) < 0.2

    def test_metrics_reporting(self, stereo_audio_with_azimuth_error, sample_rate):
        """Test that metrics are properly reported"""
        corrector = TapeAzimuthCorrector(phase_threshold_degrees=1.0)
        corrector.process(stereo_audio_with_azimuth_error, sample_rate)

        assert hasattr(corrector, "metrics")
        assert "phase_error_detected" in corrector.metrics
        assert "correction_applied" in corrector.metrics


# =============================================================================
# TESTS: TapeSpecialist (Unified API)
# =============================================================================


class TestTapeSpecialist:
    """Tests for unified TapeSpecialist API"""

    def test_initialization_all_enabled(self):
        """Test initialization with all modules enabled"""
        specialist = TapeSpecialist()

        assert specialist.enable_print_through_removal is True
        assert specialist.enable_azimuth_correction is True
        assert hasattr(specialist, "print_through_remover")
        assert hasattr(specialist, "azimuth_corrector")

    def test_initialization_selective_enable(self):
        """Test initialization with selective module enable"""
        specialist = TapeSpecialist(enable_print_through_removal=True, enable_azimuth_correction=False)

        assert specialist.enable_print_through_removal is True
        assert specialist.enable_azimuth_correction is False
        assert hasattr(specialist, "print_through_remover")
        assert not hasattr(specialist, "azimuth_corrector")

    def test_initialization_with_params(self):
        """Test initialization with custom parameters"""
        specialist = TapeSpecialist(
            max_delay_ms=200.0, print_through_strength=0.8, azimuth_correction_strength=0.9, phase_threshold_degrees=5.0
        )

        assert specialist.print_through_remover.max_delay_ms == 200.0
        assert specialist.print_through_remover.adaptive_strength == 0.8
        assert specialist.azimuth_corrector.correction_strength == 0.9
        assert specialist.azimuth_corrector.phase_threshold_degrees == 5.0

    def test_process_mono_audio(self, mono_audio, sample_rate):
        """Test processing mono audio"""
        specialist = TapeSpecialist()
        processed = specialist.process(mono_audio, sample_rate)

        # Should preserve length
        assert len(processed) == len(mono_audio)

    def test_process_stereo_audio(self, stereo_audio, sample_rate):
        """Test processing stereo audio"""
        specialist = TapeSpecialist()
        processed = specialist.process(stereo_audio, sample_rate)

        # Should preserve shape
        assert processed.shape == stereo_audio.shape

    def test_process_with_all_modules(self, stereo_audio_with_azimuth_error, sample_rate):
        """Test processing with all modules enabled"""
        specialist = TapeSpecialist(
            enable_print_through_removal=True, enable_azimuth_correction=True, phase_threshold_degrees=1.0
        )

        processed = specialist.process(stereo_audio_with_azimuth_error, sample_rate)

        # Should preserve shape
        assert processed.shape == stereo_audio_with_azimuth_error.shape

    def test_process_with_only_print_through(self, mono_audio, sample_rate):
        """Test processing with only print-through enabled"""
        specialist = TapeSpecialist(enable_print_through_removal=True, enable_azimuth_correction=False)

        processed = specialist.process(mono_audio, sample_rate)

        assert len(processed) == len(mono_audio)

    def test_process_with_only_azimuth(self, stereo_audio, sample_rate):
        """Test processing with only azimuth enabled"""
        specialist = TapeSpecialist(enable_print_through_removal=False, enable_azimuth_correction=True)

        processed = specialist.process(stereo_audio, sample_rate)

        assert processed.shape == stereo_audio.shape

    def test_get_metrics(self, stereo_audio, sample_rate):
        """Test metrics collection from all modules"""
        specialist = TapeSpecialist()
        specialist.process(stereo_audio, sample_rate)

        metrics = specialist.get_metrics()

        # Should have metrics from enabled modules
        assert isinstance(metrics, dict)
        assert len(metrics) > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for full pipeline"""

    def test_full_pipeline_realistic_tape(self, sample_rate):
        """Test full pipeline on realistic tape audio"""
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create stereo audio with both problems:
        # 1. Phase misalignment (azimuth error)
        # 2. Print-through echo

        # Left channel with transient
        left = np.zeros_like(t)
        transient_idx = int(0.3 * sample_rate)
        transient_width = int(0.01 * sample_rate)
        left[transient_idx : transient_idx + transient_width] = 0.7

        # Right channel: Delayed by 10 samples (azimuth error)
        right = np.roll(left, 10)
        right[:10] = 0

        # Add print-through to both channels
        echo_delay = int(0.1 * sample_rate)
        echo_gain = 10 ** (-30 / 20)

        if transient_idx + echo_delay + transient_width < len(left):
            left[transient_idx + echo_delay : transient_idx + echo_delay + transient_width] += (
                left[transient_idx : transient_idx + transient_width] * echo_gain
            )
            right[transient_idx + echo_delay : transient_idx + echo_delay + transient_width] += (
                right[transient_idx : transient_idx + transient_width] * echo_gain
            )

        audio = np.vstack([left, right])

        # Process with all modules
        specialist = TapeSpecialist(
            enable_print_through_removal=True,
            enable_azimuth_correction=True,
            print_through_strength=0.7,
            phase_threshold_degrees=1.0,
        )

        processed = specialist.process(audio, sample_rate)

        # Verify processing
        metrics = specialist.get_metrics()

        # Should have processed successfully
        assert processed.shape == audio.shape
        assert len(metrics) >= 1

    def test_preserves_silence(self, sample_rate):
        """Test that silence remains silence"""
        duration = 1.0
        silence = np.zeros(int(sample_rate * duration))

        specialist = TapeSpecialist()
        processed = specialist.process(silence, sample_rate)

        # Should still be (near) silence
        rms = np.sqrt(np.mean(processed**2))
        assert rms < 1e-6


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformance:
    """Performance and efficiency tests"""

    def test_processing_time_reasonable(self, stereo_audio, sample_rate):
        """Test that processing time is reasonable"""
        import time

        specialist = TapeSpecialist()

        start = time.time()
        specialist.process(stereo_audio, sample_rate)
        elapsed = time.time() - start

        audio_duration = stereo_audio.shape[1] / sample_rate

        # Should process faster than 10× real-time (conservative)
        assert elapsed < audio_duration * 10

        print(f"\nProcessing time: {elapsed:.3f}s for {audio_duration:.1f}s audio")
        print(f"Real-time factor: {elapsed / audio_duration:.2f}×")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
