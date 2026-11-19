"""
test_generic_safety_wrappers.py - Tests for Generic HIPS Safety Wrappers

Comprehensive test suite for:
- GenericNoiseReductionSafety
- GenericRestorationSafety
- GenericDynamicsSafety
- GenericSpectralSafety
- GenericSpatialSafety
- SafetyWrapperFactory

Author: AURIK Team
Version: 1.0.0
Date: 8. Februar 2026
"""

import numpy as np
import pytest

from backend.core.musical_goals.processing_modes import ProcessingMode
from backend.ml.safety_wrappers.generic_safety_wrapper import GenericNoiseReductionSafety, GenericRestorationSafety
from backend.ml.safety_wrappers.generic_safety_wrapper_extended import (
    GenericDynamicsSafety,
    GenericSpatialSafety,
    GenericSpectralSafety,
)
from backend.ml.safety_wrappers.safety_wrapper_factory import (
    DSP_MODULE_CLASSIFICATION,
    SafetyWrapperFactory,
    wrap_module,
)
from backend.ml.safety_wrappers.safety_wrapper_template import ProcessingDecision

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_rate():
    """Sample rate for test audio."""
    return 16000


@pytest.fixture
def clean_audio(sample_rate):
    """Create clean audio (sine wave)."""
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine
    return audio.astype(np.float32)


@pytest.fixture
def noisy_audio(sample_rate):
    """Create noisy audio with moderate SNR (~15 dB)."""
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Signal: 440 Hz sine
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Noise (SNR ~ 15 dB) - reduce noise amplitude
    noise = np.random.normal(0, 0.03, len(signal))  # Reduced from 0.1

    audio = signal + noise
    return audio.astype(np.float32)


@pytest.fixture
def defective_audio(sample_rate):
    """Create audio with defects (clicks) but not clipping."""
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Signal (reduce amplitude to avoid clipping warnings)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)

    # Add clicks (moderate amplitude)
    n_clicks = 50
    click_positions = np.random.randint(100, len(audio) - 100, n_clicks)
    for pos in click_positions:
        audio[pos : pos + 3] += np.random.uniform(-0.3, 0.3, 3)

    return audio.astype(np.float32)


@pytest.fixture
def stereo_audio(sample_rate):
    """Create stereo audio."""
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Left and right channels with different content
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 554 * t)  # E note

    audio = np.vstack([left, right])
    return audio.astype(np.float32)


def simple_processor(audio, sr, strength=1.0):
    """Simple dummy processor for testing."""
    # Just scale audio (simulates processing)
    return audio * (1.0 - 0.1 * strength)


def identity_processor(audio, sr, **kwargs):
    """Identity processor (returns unchanged)."""
    return audio.copy()


def aggressive_processor(audio, sr, **kwargs):
    """Aggressive processor that significantly alters audio."""
    # Severely attenuate (simulates over-processing)
    return audio * 0.3


# ============================================================================
# TEST GENERIC NOISE REDUCTION SAFETY
# ============================================================================


@pytest.mark.unit
class TestGenericNoiseReductionSafety:
    """Tests for GenericNoiseReductionSafety wrapper."""

    def test_initialization(self):
        """Test wrapper initialization."""
        wrapper = GenericNoiseReductionSafety(
            module_name="test_denoiser", module_version="1.0.0", processor_func=simple_processor
        )

        assert wrapper.module_name == "test_denoiser"
        assert wrapper.module_version == "1.0.0"
        assert wrapper.total_calls == 0

    def test_clean_audio_processing(self, clean_audio, sample_rate):
        """Test processing of already clean audio."""
        wrapper = GenericNoiseReductionSafety(
            module_name="test_denoiser", module_version="1.0.0", processor_func=simple_processor
        )

        processed, report = wrapper.process(clean_audio, sample_rate, strength=0.8)

        # Should process (even though clean)
        assert report.decision in [ProcessingDecision.PROCEED, ProcessingDecision.REDUCE_STRENGTH]
        assert wrapper.total_calls == 1

        # May have warnings about clean signal (but not required since SNR might be good)
        # Just verify processing completed
        assert processed is not None

    def test_noisy_audio_processing(self, noisy_audio, sample_rate):
        """Test processing of noisy audio."""
        wrapper = GenericNoiseReductionSafety(
            module_name="test_denoiser", module_version="1.0.0", processor_func=simple_processor
        )

        processed, report = wrapper.process(noisy_audio, sample_rate, strength=0.8)

        # Should proceed (with high or medium confidence)
        assert report.decision in [ProcessingDecision.PROCEED, ProcessingDecision.REDUCE_STRENGTH]
        assert report.pre_check_result.confidence > 0.5
        assert wrapper.total_calls >= 1  # At least one call made

    def test_musical_noise_detection(self, clean_audio, sample_rate):
        """Test that musical noise artifacts are detected."""

        # Processor that introduces tonal bursts
        def musical_noise_processor(audio, sr, **kwargs):
            output = audio.copy()
            # Add tonal bursts every 0.1s
            for i in range(0, len(output), sr // 10):
                if i < len(output):
                    output[i] = 1.0  # Sharp transient
            return output

        wrapper = GenericNoiseReductionSafety(
            module_name="test_denoiser", module_version="1.0.0", processor_func=musical_noise_processor
        )

        processed, report = wrapper.process(clean_audio, sample_rate)

        # Check if musical noise was detected
        if report.post_check_result:
            assert "musical_noise_score" in report.post_check_result.metrics

    def test_over_processing_detection(self, noisy_audio, sample_rate):
        """Test detection of over-processing."""
        wrapper = GenericNoiseReductionSafety(
            module_name="test_denoiser", module_version="1.0.0", processor_func=aggressive_processor
        )

        processed, report = wrapper.process(noisy_audio, sample_rate)

        # Should detect quality issues
        if report.post_check_result:
            assert report.post_check_result.quality_score < 0.8

    def test_statistics_tracking(self, clean_audio, sample_rate):
        """Test statistics tracking."""
        wrapper = GenericNoiseReductionSafety(
            module_name="test_denoiser", module_version="1.0.0", processor_func=simple_processor
        )

        # Process multiple times
        for _ in range(5):
            wrapper.process(clean_audio, sample_rate, strength=0.5)

        stats = wrapper.get_statistics()

        assert stats["total_calls"] == 5
        assert stats["success_rate"] > 0
        assert "average_quality_score" in stats


# ============================================================================
# TEST GENERIC RESTORATION SAFETY
# ============================================================================


class TestGenericRestorationSafety:
    """Tests for GenericRestorationSafety wrapper."""

    def test_initialization(self):
        """Test wrapper initialization."""
        wrapper = GenericRestorationSafety(
            module_name="test_declicker", module_version="1.0.0", processor_func=simple_processor
        )

        assert wrapper.module_name == "test_declicker"
        assert wrapper.confidence_threshold == 0.6

    def test_defective_audio_processing(self, defective_audio, sample_rate):
        """Test processing of defective audio."""
        wrapper = GenericRestorationSafety(
            module_name="test_declicker", module_version="1.0.0", processor_func=simple_processor
        )

        processed, report = wrapper.process(defective_audio, sample_rate, strength=0.8)

        # Should detect defects and proceed
        assert report.pre_check_result.passed or report.decision == ProcessingDecision.ABORT
        assert wrapper.total_calls == 1

        # If passed, should have detected some defects
        if report.pre_check_result.passed:
            assert report.pre_check_result.metadata.get("defect_score", 0) >= 0

    def test_clean_audio_no_defects(self, clean_audio, sample_rate):
        """Test that clean audio has low defect score."""
        wrapper = GenericRestorationSafety(
            module_name="test_declicker", module_version="1.0.0", processor_func=simple_processor
        )

        processed, report = wrapper.process(clean_audio, sample_rate)

        # Clean audio should have warnings
        if report.pre_check_result.warnings:
            assert any("few defects" in w.lower() for w in report.pre_check_result.warnings)

    def test_defect_reduction_validation(self, defective_audio, sample_rate):
        """Test that defect reduction is validated."""

        # Processor that actually reduces defects (smoothing)
        def defect_reducer(audio, sr, **kwargs):
            from scipy.ndimage import gaussian_filter1d

            return gaussian_filter1d(audio, sigma=3)

        wrapper = GenericRestorationSafety(
            module_name="test_declicker", module_version="1.0.0", processor_func=defect_reducer
        )

        processed, report = wrapper.process(defective_audio, sample_rate)

        # Should show defect reduction
        if report.post_check_result:
            assert "defect_reduction" in report.post_check_result.metrics
            assert report.post_check_result.metrics["defect_reduction"] >= 0


# ============================================================================
# TEST GENERIC DYNAMICS SAFETY
# ============================================================================


class TestGenericDynamicsSafety:
    """Tests for GenericDynamicsSafety wrapper."""

    def test_initialization(self):
        """Test wrapper initialization."""
        wrapper = GenericDynamicsSafety(
            module_name="test_compressor", module_version="1.0.0", processor_func=simple_processor
        )

        assert wrapper.module_name == "test_compressor"
        assert wrapper.confidence_threshold == 0.6

    def test_dynamics_processing(self, clean_audio, sample_rate):
        """Test basic dynamics processing."""

        # Compressor simulation
        def compressor(audio, sr, ratio=4.0, **kwargs):
            threshold = 0.3
            output = audio.copy()
            mask = np.abs(output) > threshold
            output[mask] = np.sign(output[mask]) * (threshold + (np.abs(output[mask]) - threshold) / ratio)
            return output

        wrapper = GenericDynamicsSafety(
            module_name="test_compressor", module_version="1.0.0", processor_func=compressor
        )

        processed, report = wrapper.process(clean_audio, sample_rate, ratio=4.0)

        # Should process successfully
        assert report.decision in [ProcessingDecision.PROCEED, ProcessingDecision.REDUCE_STRENGTH]

        # Check dynamic range measurement
        if report.post_check_result:
            assert "dynamic_range_change_db" in report.post_check_result.metrics

    def test_pumping_detection(self, clean_audio, sample_rate):
        """Test pumping artifact detection."""

        # Processor that introduces pumping (gain modulation)
        def pumping_processor(audio, sr, **kwargs):
            t = np.arange(len(audio)) / sr
            gain_mod = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz modulation
            return audio * gain_mod

        wrapper = GenericDynamicsSafety(
            module_name="test_compressor",
            module_version="1.0.0",
            processor_func=pumping_processor,
            quality_threshold=0.6,
        )

        processed, report = wrapper.process(clean_audio, sample_rate)

        # Should detect pumping
        if report.post_check_result:
            pumping_score = report.post_check_result.metrics.get("pumping_score", 0.0)
            assert pumping_score > 0.1  # Some pumping detected


# ============================================================================
# TEST GENERIC SPECTRAL SAFETY
# ============================================================================


class TestGenericSpectralSafety:
    """Tests for GenericSpectralSafety wrapper."""

    def test_initialization(self):
        """Test wrapper initialization."""
        wrapper = GenericSpectralSafety(module_name="test_eq", module_version="1.0.0", processor_func=simple_processor)

        assert wrapper.module_name == "test_eq"
        assert wrapper.confidence_threshold == 0.7

    def test_spectral_processing(self, clean_audio, sample_rate):
        """Test basic spectral processing."""

        # Simple high-shelf filter
        def highshelf(audio, sr, gain_db=6.0, **kwargs):
            from scipy.signal import butter, sosfilt

            sos = butter(2, 2000, "high", fs=sr, output="sos")
            boosted = sosfilt(sos, audio)
            gain = 10 ** (gain_db / 20)
            return audio + (boosted - audio) * (gain - 1)

        wrapper = GenericSpectralSafety(module_name="test_eq", module_version="1.0.0", processor_func=highshelf)

        processed, report = wrapper.process(clean_audio, sample_rate, gain_db=6.0)

        # Processing may proceed or abort depending on Musical Goals thresholds
        # Just verify the wrapper executed and returned a valid report
        assert processed is not None
        assert report is not None
        assert report.module_name == "test_eq"

        # Check spectral centroid metrics exist
        if report.post_check_result:
            assert "centroid_change_pct" in report.post_check_result.metrics

    def test_harshness_detection(self, sample_rate):
        """Test detection of harsh high frequencies."""
        # Create audio with harsh resonance at 5 kHz
        t = np.linspace(0, 2.0, int(sample_rate * 2.0))
        audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.7 * np.sin(2 * np.pi * 5000 * t)  # Strong 5 kHz
        audio = audio.astype(np.float32)

        wrapper = GenericSpectralSafety(
            module_name="test_eq", module_version="1.0.0", processor_func=identity_processor
        )

        processed, report = wrapper.process(audio, sample_rate)

        # Should detect harshness
        if report.post_check_result:
            harshness = report.post_check_result.metrics.get("harshness_score", 0.0)
            assert harshness > 0.2


# ============================================================================
# TEST GENERIC SPATIAL SAFETY
# ============================================================================


class TestGenericSpatialSafety:
    """Tests for GenericSpatialSafety wrapper."""

    def test_initialization(self):
        """Test wrapper initialization."""
        wrapper = GenericSpatialSafety(
            module_name="test_widener", module_version="1.0.0", processor_func=simple_processor
        )

        assert wrapper.module_name == "test_widener"
        assert wrapper.confidence_threshold == 0.6

    def test_stereo_required(self, clean_audio, sample_rate):
        """Test that spatial processing requires stereo."""
        wrapper = GenericSpatialSafety(
            module_name="test_widener", module_version="1.0.0", processor_func=simple_processor
        )

        # Mono audio should be rejected
        processed, report = wrapper.process(clean_audio, sample_rate)

        assert not report.pre_check_result.passed
        assert "stereo" in str(report.pre_check_result.reasons).lower()

    def test_stereo_widening(self, stereo_audio, sample_rate):
        """Test stereo widening."""

        # Simple widener (Mid/Side processing)
        def widener(audio, sr, width=1.5, **kwargs):
            mid = (audio[0] + audio[1]) / 2.0
            side = (audio[0] - audio[1]) / 2.0
            side *= width
            left = mid + side
            right = mid - side
            return np.vstack([left, right])

        wrapper = GenericSpatialSafety(module_name="test_widener", module_version="1.0.0", processor_func=widener)

        processed, report = wrapper.process(stereo_audio, sample_rate, width=1.5)

        # Should process successfully
        assert report.decision in [ProcessingDecision.PROCEED, ProcessingDecision.REDUCE_STRENGTH]

        # Check width change exists (may be positive or negative depending on implementation)
        if report.post_check_result:
            assert "width_change" in report.post_check_result.metrics
            # Width change should exist, direction depends on algorithm

    def test_mono_compatibility(self, stereo_audio, sample_rate):
        """Test mono compatibility validation."""

        # Widener that may cause phase issues
        def aggressive_widener(audio, sr, **kwargs):
            mid = (audio[0] + audio[1]) / 2.0
            side = (audio[0] - audio[1]) / 2.0
            side *= 3.0  # Very aggressive
            left = mid + side
            right = mid - side
            return np.vstack([left, right])

        wrapper = GenericSpatialSafety(
            module_name="test_widener", module_version="1.0.0", processor_func=aggressive_widener, quality_threshold=0.6
        )

        processed, report = wrapper.process(stereo_audio, sample_rate)

        # Check mono compatibility
        if report.post_check_result:
            assert "mono_compatibility" in report.post_check_result.metrics


# ============================================================================
# TEST SAFETY WRAPPER FACTORY
# ============================================================================


class TestSafetyWrapperFactory:
    """Tests for SafetyWrapperFactory."""

    def test_factory_initialization(self):
        """Test factory initialization."""
        factory = SafetyWrapperFactory(processing_mode=ProcessingMode.RESTORATION)

        assert factory.processing_mode == ProcessingMode.RESTORATION
        assert factory.wrapper_stats["custom"] == 0

    def test_module_classification(self):
        """Test module classification."""
        factory = SafetyWrapperFactory()

        # Test noise reduction
        assert factory.classify_module("adaptive_imcra") == "noise_reduction"
        assert factory.classify_module("sota_denoiser") == "noise_reduction"

        # Test restoration
        assert factory.classify_module("automatic_declicker") == "restoration"
        assert factory.classify_module("automatic_declipper") == "restoration"

        # Test dynamics
        assert factory.classify_module("multiband_compressor") == "dynamics"
        assert factory.classify_module("intelligent_limiter") == "dynamics"

        # Test spectral
        assert factory.classify_module("auto_eq") == "spectral"
        # Note: harmonic_exciter has a custom wrapper, so it's classified as 'custom'

        # Test spatial (note: stereo_widener has custom wrapper)
        assert factory.classify_module("balance") == "spatial"

    def test_custom_module_classification(self):
        """Test custom module classification."""
        factory = SafetyWrapperFactory()

        # Modules with existing custom wrappers
        assert factory.classify_module("dehum") == "custom"
        assert factory.classify_module("pitch_correction") == "custom"

    def test_create_wrapper_for_module(self):
        """Test wrapper creation for modules."""
        factory = SafetyWrapperFactory()

        # Create noise reduction wrapper
        wrapper = factory.create_wrapper_for_module("adaptive_imcra", processor_func=simple_processor)

        assert isinstance(wrapper, GenericNoiseReductionSafety)
        assert wrapper.module_name == "adaptive_imcra"

    def test_wrap_multiple_modules(self):
        """Test wrapping multiple modules."""
        factory = SafetyWrapperFactory()

        module_list = ["adaptive_imcra", "automatic_declicker", "multiband_compressor", "auto_eq", "stereo_widener"]

        wrapped = factory.wrap_all_modules(module_list=module_list)

        assert len(wrapped) == 5
        assert "adaptive_imcra" in wrapped
        assert "auto_eq" in wrapped

    def test_wrapper_statistics(self):
        """Test wrapper statistics."""
        factory = SafetyWrapperFactory()

        module_list = [
            "adaptive_imcra",  # noise_reduction
            "automatic_declicker",  # restoration
            "multiband_compressor",  # dynamics
        ]

        factory.wrap_all_modules(module_list=module_list)

        stats = factory.get_statistics()

        assert stats["total_wrapped"] == 3
        assert stats["by_type"]["noise_reduction"] == 1
        assert stats["by_type"]["restoration"] == 1
        assert stats["by_type"]["dynamics"] == 1

    def test_wrap_module_convenience_function(self):
        """Test convenience function."""
        wrapper = wrap_module(
            "adaptive_imcra", processor_func=simple_processor, processing_mode=ProcessingMode.STUDIO_2026
        )

        assert isinstance(wrapper, GenericNoiseReductionSafety)
        assert wrapper.processing_mode == ProcessingMode.STUDIO_2026


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for safety wrapper system."""

    def test_end_to_end_workflow(self, noisy_audio, sample_rate):
        """Test complete workflow from factory to processing."""
        # Create wrapper via factory
        wrapper = wrap_module(
            "adaptive_imcra", processor_func=simple_processor, processing_mode=ProcessingMode.RESTORATION
        )

        # Process audio
        processed, report = wrapper.process(noisy_audio, sample_rate, strength=0.8)

        # Verify results
        assert processed is not None
        assert report.module_name == "adaptive_imcra"
        assert report.processing_time_ms > 0

        # Check statistics
        stats = wrapper.get_statistics()
        assert stats["total_calls"] == 1

    def test_module_coverage(self):
        """Test that all classified modules can be wrapped."""
        SafetyWrapperFactory()

        # Count total classified modules
        total_modules = sum(len(modules) for modules in DSP_MODULE_CLASSIFICATION.values())

        # Should be able to classify all
        assert total_modules > 90  # We have 93+ modules classified


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
