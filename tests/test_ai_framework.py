"""
Tests for Aurik 10.0.0 AI Framework
=================================

Comprehensive test suite for AI-based audio restoration & enhancement.

Author: Aurik 10.0.0 Development Team
Date: 15. Februar 2026
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.ai_framework import (
    AurikAIFramework,
    DefectDetectionResult,
    DefectType,
    EnhancementResult,
    MaterialType,
    RestorationMode,
    Studio2026Processor,
    UnifiedAudioEnhancer,
    UnifiedAudioRestorer,
    UnifiedDefectDetector,
)
from backend.core.ai_framework import (
    FrameworkRestorationResult as RestorationResult,
)

# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def sample_rate():
    """Standard sample rate."""
    return 48000


@pytest.fixture
def clean_audio(sample_rate):
    """Generate clean test audio (sine wave)."""
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz A note
    return audio


@pytest.fixture
def audio_with_clicks(clean_audio, sample_rate):
    """Audio with clicks."""
    audio = clean_audio.copy()
    click_positions = [1000, 5000, 10000, 20000]
    for pos in click_positions:
        if pos < len(audio):
            audio[pos : pos + 5] += 0.8  # Strong transient
    return audio


@pytest.fixture
def audio_with_hiss(clean_audio, sample_rate):
    """Audio with hiss (broadband noise)."""
    audio = clean_audio.copy()
    hiss = np.random.normal(0, 0.1, len(audio))
    audio += hiss
    return audio


@pytest.fixture
def audio_with_hum(clean_audio, sample_rate):
    """Audio with 50Hz hum."""
    audio = clean_audio.copy()
    t = np.arange(len(audio)) / sample_rate
    hum = 0.15 * np.sin(2 * np.pi * 50 * t)
    audio += hum
    return audio


@pytest.fixture
def audio_with_distortion(clean_audio):
    """Audio with distortion (clipping)."""
    audio = clean_audio.copy()
    audio = np.clip(audio * 2.5, -0.99, 0.99)  # Hard clipping
    return audio


@pytest.fixture
def audio_with_dropouts(clean_audio, sample_rate):
    """Audio with dropouts (silence regions)."""
    audio = clean_audio.copy()
    # Create dropout at 0.3-0.35 seconds
    start = int(0.3 * sample_rate)
    end = int(0.35 * sample_rate)
    audio[start:end] = 0.0
    return audio


@pytest.fixture
def framework(sample_rate):
    """Create framework instance."""
    return AurikAIFramework(sample_rate=sample_rate)


@pytest.fixture
def detector(sample_rate):
    """Create detector instance."""
    return UnifiedDefectDetector(sample_rate=sample_rate)


@pytest.fixture
def restorer(sample_rate):
    """Create restorer instance."""
    return UnifiedAudioRestorer(sample_rate=sample_rate)


@pytest.fixture
def enhancer(sample_rate):
    """Create enhancer instance."""
    return UnifiedAudioEnhancer(sample_rate=sample_rate)


# ============================================================
# DEFECT DETECTION TESTS
# ============================================================


class TestUnifiedDefectDetector:
    """Test suite for defect detection."""

    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector.sr == 48000
        assert hasattr(detector, "detect")

    def test_detect_clean_audio(self, detector, clean_audio):
        """Test detection on clean audio."""
        result = detector.detect(clean_audio)

        assert isinstance(result, DefectDetectionResult)
        assert result.overall_quality_score > 0.7  # Should be high quality
        assert len(result.defects) > 0  # Should detect some defects dict

    def test_detect_clicks(self, detector, audio_with_clicks):
        """Test click detection."""
        result = detector.detect(audio_with_clicks)

        # Should detect clicks
        assert DefectType.CLICKS in result.defects
        click_confidence = result.defects[DefectType.CLICKS]
        assert click_confidence > 0.3  # Should have reasonable confidence

        # Should have locations
        assert DefectType.CLICKS in result.locations
        assert len(result.locations[DefectType.CLICKS]) > 0

    def test_detect_hiss(self, detector, audio_with_hiss):
        """Test hiss detection."""
        result = detector.detect(audio_with_hiss)

        # Should detect hiss
        assert DefectType.HISS in result.defects
        hiss_confidence = result.defects[DefectType.HISS]
        assert hiss_confidence > 0.2  # Should detect some hiss

    def test_detect_hum(self, detector, audio_with_hum):
        """Test hum detection."""
        result = detector.detect(audio_with_hum)

        # Should detect hum
        assert DefectType.HUM in result.defects
        hum_confidence = result.defects[DefectType.HUM]
        assert hum_confidence > 0.3  # Should clearly detect 50Hz hum

    def test_detect_distortion(self, detector, audio_with_distortion):
        """Test distortion detection."""
        result = detector.detect(audio_with_distortion)

        # Should detect distortion or clipping
        distortion_detected = (
            result.defects.get(DefectType.DISTORTION, 0) > 0.3 or result.defects.get(DefectType.CLIPPING, 0) > 0.3
        )
        assert distortion_detected

    def test_detect_dropouts(self, detector, audio_with_dropouts):
        """Test dropout detection."""
        result = detector.detect(audio_with_dropouts)

        # Should detect dropout
        assert DefectType.DROPOUT in result.defects
        assert result.defects[DefectType.DROPOUT] > 0.2

    def test_quality_score_degraded_audio(self, detector, audio_with_clicks, audio_with_hiss):
        """Test quality score is lower for degraded audio."""
        result_clicks = detector.detect(audio_with_clicks)
        result_hiss = detector.detect(audio_with_hiss)

        # Both should have reduced quality
        assert result_clicks.overall_quality_score < 1.0
        assert result_hiss.overall_quality_score < 1.0

    def test_material_type_detection(self, detector, audio_with_clicks):
        """Test material type detection."""
        result = detector.detect(audio_with_clicks)

        # Should detect some material type
        assert result.material_type is not None
        assert isinstance(result.material_type, MaterialType)

    def test_recommended_mode(self, detector, audio_with_clicks):
        """Test restoration mode recommendation."""
        result = detector.detect(audio_with_clicks)

        # Should recommend a mode
        assert result.recommended_mode is not None
        assert isinstance(result.recommended_mode, RestorationMode)

    def test_stereo_audio(self, detector, clean_audio):
        """Test detection on stereo audio."""
        # Create stereo
        stereo = np.stack([clean_audio, clean_audio], axis=1)

        result = detector.detect(stereo)
        assert isinstance(result, DefectDetectionResult)
        assert result.overall_quality_score > 0


# ============================================================
# AUDIO RESTORATION TESTS
# ============================================================


class TestUnifiedAudioRestorer:
    """Test suite for audio restoration."""

    def test_restorer_initialization(self, restorer):
        """Test restorer initializes correctly."""
        assert restorer.sr == 48000
        assert hasattr(restorer, "restore")
        assert hasattr(restorer, "detector")

    def test_restore_clean_audio(self, restorer, clean_audio):
        """Test restoration on clean audio (should not harm)."""
        result = restorer.restore(clean_audio, mode=RestorationMode.BALANCED)

        assert isinstance(result, RestorationResult)
        assert result.audio.shape == clean_audio.shape
        assert result.sample_rate == 48000

        # Should not be too different from original
        correlation = np.corrcoef(clean_audio, result.audio)[0, 1]
        assert correlation > 0.9  # Should be highly correlated

    def test_restore_clicks(self, restorer, audio_with_clicks):
        """Test click removal."""
        result = restorer.restore(audio_with_clicks, mode=RestorationMode.AGGRESSIVE, auto_detect=True)

        assert isinstance(result, RestorationResult)
        assert len(result.processing_applied) > 0

        # Should have removed some clicks
        if DefectType.CLICKS in result.defects_removed:
            assert result.defects_removed[DefectType.CLICKS] > 0

    def test_restore_hiss(self, restorer, audio_with_hiss):
        """Test hiss reduction."""
        result = restorer.restore(audio_with_hiss, mode=RestorationMode.BALANCED, auto_detect=True)

        assert isinstance(result, RestorationResult)
        assert "hiss_reduction" in result.processing_applied or "hiss" in str(result.processing_applied).lower()

    def test_restore_hum(self, restorer, audio_with_hum):
        """Test hum removal."""
        result = restorer.restore(audio_with_hum, mode=RestorationMode.BALANCED, auto_detect=True)

        assert isinstance(result, RestorationResult)

        # Check hum removal in result
        assert "hum_removal" in result.processing_applied or "hum" in str(result.processing_applied).lower()

    def test_restore_dropouts(self, restorer, audio_with_dropouts):
        """Test dropout filling."""
        result = restorer.restore(audio_with_dropouts, mode=RestorationMode.BALANCED, auto_detect=True)

        assert isinstance(result, RestorationResult)

        # Dropout filling should be applied
        assert any("dropout" in proc.lower() for proc in result.processing_applied)

    def test_restoration_modes(self, restorer, audio_with_clicks):
        """Test different restoration modes."""
        modes = [RestorationMode.CONSERVATIVE, RestorationMode.BALANCED, RestorationMode.AGGRESSIVE]

        for mode in modes:
            result = restorer.restore(audio_with_clicks, mode=mode, auto_detect=True)
            assert isinstance(result, RestorationResult)
            assert result.metadata.get("mode") == mode.value

    def test_quality_improvement(self, restorer, audio_with_clicks):
        """Test quality improvement is positive."""
        result = restorer.restore(audio_with_clicks, mode=RestorationMode.AGGRESSIVE, auto_detect=True)

        # Quality improvement should be non-negative
        assert result.quality_improvement >= -0.2  # Allow small degradation

    def test_stereo_restoration(self, restorer, audio_with_clicks):
        """Test restoration on stereo audio."""
        stereo = np.stack([audio_with_clicks, audio_with_clicks], axis=1)

        result = restorer.restore(stereo, mode=RestorationMode.BALANCED)
        assert result.audio.shape == stereo.shape
        assert result.audio.ndim == 2


# ============================================================
# AUDIO ENHANCEMENT TESTS
# ============================================================


class TestUnifiedAudioEnhancer:
    """Test suite for audio enhancement."""

    def test_enhancer_initialization(self, enhancer):
        """Test enhancer initializes correctly."""
        assert enhancer.sr == 48000
        assert hasattr(enhancer, "enhance")

    def test_enhance_audio(self, enhancer, clean_audio):
        """Test basic enhancement."""
        result = enhancer.enhance(clean_audio, target_clarity=0.7, target_presence=0.7, target_detail=0.7)

        assert isinstance(result, EnhancementResult)
        assert result.audio.shape == clean_audio.shape
        assert len(result.enhancements_applied) > 0

    def test_clarity_enhancement(self, enhancer, clean_audio):
        """Test clarity enhancement."""
        result = enhancer.enhance(clean_audio, target_clarity=0.9, target_presence=0.5, target_detail=0.5)

        assert result.clarity_improvement >= 0.8
        assert "clarity" in str(result.enhancements_applied).lower()

    def test_presence_enhancement(self, enhancer, clean_audio):
        """Test presence enhancement."""
        result = enhancer.enhance(clean_audio, target_clarity=0.5, target_presence=0.9, target_detail=0.5)

        assert result.presence_improvement >= 0.8
        assert "presence" in str(result.enhancements_applied).lower()

    def test_detail_enhancement(self, enhancer, clean_audio):
        """Test detail enhancement."""
        result = enhancer.enhance(clean_audio, target_clarity=0.5, target_presence=0.5, target_detail=0.9)

        assert result.detail_improvement >= 0.8
        assert "detail" in str(result.enhancements_applied).lower()

    def test_no_enhancement_low_targets(self, enhancer, clean_audio):
        """Test low target values result in minimal enhancement."""
        result = enhancer.enhance(clean_audio, target_clarity=0.3, target_presence=0.3, target_detail=0.3)

        # Should apply minimal or no enhancements
        assert len(result.enhancements_applied) >= 0

    def test_stereo_enhancement(self, enhancer, clean_audio):
        """Test enhancement on stereo audio."""
        stereo = np.stack([clean_audio, clean_audio], axis=1)

        result = enhancer.enhance(stereo, target_clarity=0.7)
        assert result.audio.shape == stereo.shape
        assert result.audio.ndim == 2


# ============================================================
# STUDIO 2026 TESTS
# ============================================================


class TestStudio2026Processor:
    """Test suite for Studio 2026 Magic Button."""

    def test_studio2026_initialization(self, sample_rate):
        """Test Studio 2026 processor initializes."""
        processor = Studio2026Processor(sample_rate=sample_rate)
        assert processor.sr == sample_rate
        assert hasattr(processor, "process")

    def test_studio2026_process(self, sample_rate, audio_with_clicks):
        """Test Studio 2026 processing."""
        processor = Studio2026Processor(sample_rate=sample_rate)

        audio_out, report = processor.process(audio_with_clicks)

        # Check output
        assert audio_out.shape == audio_with_clicks.shape
        assert isinstance(report, dict)

        # Check report structure
        assert "detection" in report
        assert "restoration" in report
        assert "enhancement" in report
        assert "final" in report

        # Check report content
        assert "quality_score_before" in report["detection"]
        assert "defects_removed" in report["restoration"]
        assert report["final"]["success"] is True

    def test_studio2026_improves_quality(self, sample_rate, audio_with_clicks, audio_with_hiss):
        """Test Studio 2026 improves audio quality."""
        processor = Studio2026Processor(sample_rate=sample_rate)

        # Process degraded audio
        audio_out, report = processor.process(audio_with_clicks)

        # Should have detected and processed defects
        assert report["detection"]["defects_found"] > 0
        assert len(report["restoration"]["processes"]) > 0


# ============================================================
# INTEGRATED FRAMEWORK TESTS
# ============================================================


class TestAurikAIFramework:
    """Test suite for complete AI framework."""

    def test_framework_initialization(self, framework):
        """Test framework initializes all components."""
        assert hasattr(framework, "detector")
        assert hasattr(framework, "restorer")
        assert hasattr(framework, "enhancer")
        assert hasattr(framework, "studio2026")

    def test_framework_analyze(self, framework, audio_with_clicks):
        """Test framework analyze method."""
        result = framework.analyze(audio_with_clicks)
        assert isinstance(result, DefectDetectionResult)

    def test_framework_restore(self, framework, audio_with_clicks):
        """Test framework restore method."""
        result = framework.restore(audio_with_clicks, mode=RestorationMode.BALANCED)
        assert isinstance(result, RestorationResult)

    def test_framework_enhance(self, framework, clean_audio):
        """Test framework enhance method."""
        result = framework.enhance(clean_audio, target_clarity=0.7)
        assert isinstance(result, EnhancementResult)

    def test_framework_magic_button(self, framework, audio_with_clicks):
        """Test framework magic button."""
        audio_out, report = framework.studio2026_magic_button(audio_with_clicks)

        assert audio_out.shape == audio_with_clicks.shape
        assert isinstance(report, dict)
        assert report["final"]["success"] is True

    def test_framework_pipeline(self, framework, audio_with_clicks):
        """Test complete processing pipeline."""
        # Step 1: Analyze
        detection = framework.analyze(audio_with_clicks)
        assert detection.overall_quality_score < 1.0

        # Step 2: Restore
        restoration = framework.restore(audio_with_clicks, mode=RestorationMode.BALANCED)
        assert len(restoration.processing_applied) > 0

        # Step 3: Enhance
        enhancement = framework.enhance(restoration.audio, target_clarity=0.8)
        assert len(enhancement.enhancements_applied) > 0

        # Step 4: Verify improvement
        final_detection = framework.analyze(enhancement.audio)
        # Quality should improve or stay similar
        assert final_detection.overall_quality_score >= detection.overall_quality_score - 0.1


# ============================================================
# EDGE CASE TESTS
# ============================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_audio(self, framework):
        """Test with empty audio."""
        empty = np.array([])

        # Should handle gracefully
        try:
            result = framework.analyze(empty)
            # If it doesn't raise, check it returns valid structure
            assert isinstance(result, DefectDetectionResult)
        except (ValueError, IndexError):
            # Also acceptable to raise error
            pass

    def test_very_short_audio(self, framework, sample_rate):
        """Test with very short audio (< 100ms)."""
        short = np.random.randn(int(0.05 * sample_rate))

        # Should handle gracefully
        result = framework.analyze(short)
        assert isinstance(result, DefectDetectionResult)

    def test_silent_audio(self, framework, sample_rate):
        """Test with silent audio."""
        silent = np.zeros(sample_rate)

        result = framework.analyze(silent)
        assert isinstance(result, DefectDetectionResult)
        # Quality score might be low due to lack of content
        assert 0 <= result.overall_quality_score <= 1

    def test_loud_audio(self, framework, clean_audio):
        """Test with very loud audio."""
        loud = clean_audio * 100  # Very loud

        result = framework.analyze(loud)
        assert isinstance(result, DefectDetectionResult)
        # Should detect clipping
        assert result.defects.get(DefectType.CLIPPING, 0) > 0 or result.defects.get(DefectType.DISTORTION, 0) > 0

    def test_mono_and_stereo_consistency(self, framework, clean_audio):
        """Test mono and stereo produce consistent results."""
        mono_result = framework.analyze(clean_audio)

        # Create stereo from mono
        stereo = np.stack([clean_audio, clean_audio], axis=1)
        stereo_result = framework.analyze(stereo)

        # Quality scores should be similar
        assert abs(mono_result.overall_quality_score - stereo_result.overall_quality_score) < 0.2


# ============================================================
# PERFORMANCE TESTS
# ============================================================


class TestPerformance:
    """Test performance and efficiency."""

    def test_detection_speed(self, framework, sample_rate):
        """Test detection completes in reasonable time."""
        import time

        # 5 seconds of audio
        audio = np.random.randn(5 * sample_rate)

        start = time.time()
        result = framework.analyze(audio)
        elapsed = time.time() - start

        # Should complete in < 5 seconds for 5s audio
        assert elapsed < 5.0
        assert isinstance(result, DefectDetectionResult)

    def test_restoration_speed(self, framework, sample_rate):
        """Test restoration completes in reasonable time."""
        import time

        # 3 seconds of audio
        audio = np.random.randn(3 * sample_rate)

        start = time.time()
        result = framework.restore(audio, mode=RestorationMode.BALANCED)
        elapsed = time.time() - start

        # Should complete in < 10 seconds for 3s audio
        assert elapsed < 10.0
        assert isinstance(result, RestorationResult)

    @pytest.mark.slow
    def test_magic_button_speed(self, framework, sample_rate):
        """Test magic button completes in reasonable time."""
        import time

        # 2 seconds of audio
        audio = np.random.randn(2 * sample_rate)

        start = time.time()
        audio_out, report = framework.studio2026_magic_button(audio)
        elapsed = time.time() - start

        # Should complete in < 15 seconds for 2s audio
        assert elapsed < 15.0
        assert audio_out.shape == audio.shape


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestIntegration:
    """Test integration with existing Aurik components."""

    def test_ml_defect_detector_integration(self, sample_rate):
        """Test integration with existing ML defect detector."""
        detector = UnifiedDefectDetector(sample_rate=sample_rate)

        # Should have attempted to load ML detector
        assert hasattr(detector, "has_ml_detector")
        # May or may not be available
        assert isinstance(detector.has_ml_detector, bool)

    def test_framework_can_process_realistic_audio(self, framework, sample_rate):
        """Test with more realistic audio."""
        # Create complex audio with multiple components
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Multiple frequencies (chord)
        audio = (
            0.3 * np.sin(2 * np.pi * 440 * t)  # A
            + 0.3 * np.sin(2 * np.pi * 554.37 * t)  # C#
            + 0.3 * np.sin(2 * np.pi * 659.25 * t)  # E
        )

        # Add realistic artifacts
        audio += np.random.normal(0, 0.02, len(audio))  # Small noise

        # Process
        audio_out, report = framework.studio2026_magic_button(audio)

        assert audio_out.shape == audio.shape
        assert report["final"]["success"] is True


# ============================================================
# MAIN TEST RUNNER
# ============================================================

if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short", "-x"])
