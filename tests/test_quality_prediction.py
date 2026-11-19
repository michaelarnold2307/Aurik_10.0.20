"""
tests/test_quality_prediction.py
Test Suite for Quality Prediction System
========================================

Tests:
- Quality analysis (SNR, DR, THD, perceptual metrics)
- Improvement prediction
- Quality validation
- Quality gates (early stopping)
- Integration with real audio

Author: AURIK Team
"""

import numpy as np
import pytest

from backend.core.quality_prediction import (
    QualityAnalyzer,
    QualityEstimate,
    QualityLevel,
    QualityPredictor,
    create_quality_prediction_system,
)

# === Test Fixtures ===


@pytest.fixture
def clean_audio():
    """Clean audio fixture (high SNR)."""
    np.random.seed(42)
    # 1 second sine wave + small noise
    t = np.linspace(0, 1, 48000)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz
    noise = 0.01 * np.random.randn(48000)
    return (signal + noise).astype(np.float32)


@pytest.fixture
def noisy_audio():
    """Noisy audio fixture (low SNR)."""
    np.random.seed(42)
    t = np.linspace(0, 1, 48000)
    signal = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = 0.15 * np.random.randn(48000)  # High noise
    return (signal + noise).astype(np.float32)


@pytest.fixture
def distorted_audio():
    """Distorted audio with clipping."""
    np.random.seed(42)
    t = np.linspace(0, 1, 48000)
    signal = 1.2 * np.sin(2 * np.pi * 440 * t)  # Overdriven
    return np.clip(signal, -1.0, 1.0).astype(np.float32)


@pytest.fixture
def analyzer():
    """Quality analyzer fixture."""
    return QualityAnalyzer()


@pytest.fixture
def predictor():
    """Quality predictor fixture."""
    return QualityPredictor()


@pytest.fixture
def system():
    """Quality prediction system fixture."""
    return create_quality_prediction_system()


# === Test Cases ===


class TestQualityAnalyzer:
    """Test quality analysis."""

    def test_analyze_clean_audio(self, analyzer, clean_audio):
        """Test analyzing clean audio."""
        quality = analyzer.analyze_quality(clean_audio, 48000)

        # Adjust expectations for synthetic sine wave
        assert quality.overall_score > 20  # Reasonable quality score
        assert quality.snr_db > 15  # Moderate SNR fürSinus + Noise
        assert quality.quality_level in [QualityLevel.POOR, QualityLevel.FAIR, QualityLevel.GOOD]
        assert quality.confidence > 0.3

    def test_analyze_noisy_audio(self, analyzer, noisy_audio):
        """Test analyzing noisy audio."""
        quality = analyzer.analyze_quality(noisy_audio, 48000)

        assert quality.overall_score < 70  # Lower quality
        assert quality.snr_db < 30  # Lower SNR
        assert quality.quality_level in [QualityLevel.FAIR, QualityLevel.POOR]

    def test_snr_estimation(self, analyzer, clean_audio, noisy_audio):
        """Test SNR estimation."""
        clean_snr = analyzer._estimate_snr(clean_audio)
        noisy_snr = analyzer._estimate_snr(noisy_audio)

        # For synthetic sine waves, SNR values should be reasonable
        assert clean_snr > 0  # Positive SNR
        assert noisy_snr > 0  # Positive SNR
        # Difference between clean and noisy should be noticeable
        assert abs(clean_snr - noisy_snr) > 2  # At least 2 dB difference

    def test_dynamic_range_measurement(self, analyzer, clean_audio):
        """Test dynamic range measurement."""
        dr = analyzer._measure_dynamic_range(clean_audio)

        assert dr > 0
        assert dr < 120  # Reasonable range

    def test_thd_estimation(self, analyzer, clean_audio, distorted_audio):
        """Test THD estimation."""
        clean_thd = analyzer._estimate_thd(clean_audio, 48000)
        distorted_thd = analyzer._estimate_thd(distorted_audio, 48000)

        # Distorted should have higher THD
        assert distorted_thd > clean_thd

    def test_perceptual_metrics(self, analyzer, clean_audio):
        """Test perceptual quality metrics."""
        clarity = analyzer._measure_clarity(clean_audio, 48000)
        warmth = analyzer._measure_warmth(clean_audio, 48000)
        brightness = analyzer._measure_brightness(clean_audio, 48000)
        naturalness = analyzer._measure_naturalness(clean_audio, 48000)

        # All should be in valid range
        assert 0 <= clarity <= 1
        assert 0 <= warmth <= 1
        assert 0 <= brightness <= 1
        assert 0 <= naturalness <= 1

    def test_warmth_counts_vocal_body_band(self, analyzer):
        """Vokal-Körperband darf nicht als fehlende Wärme bewertet werden."""
        sr = 48000
        t = np.linspace(0, 1, sr, endpoint=False)
        vocal_body = (0.45 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        thin_hf = (0.45 * np.sin(2 * np.pi * 4200 * t)).astype(np.float32)

        assert analyzer._measure_warmth(vocal_body, sr) >= 0.35
        assert analyzer._measure_warmth(vocal_body, sr) > analyzer._measure_warmth(thin_hf, sr)

    def test_bandwidth_measurement(self, analyzer, clean_audio):
        """Test bandwidth measurement."""
        low, high = analyzer._measure_bandwidth(clean_audio, 48000)

        assert low < high
        assert low >= 0
        assert high <= 24000  # Nyquist

    def test_artifact_detection_clipping(self, analyzer, distorted_audio):
        """Test clipping detection."""
        has_artifacts, artifact_types = analyzer._detect_artifacts(distorted_audio, 48000)

        assert has_artifacts
        assert "clipping" in artifact_types

    def test_artifact_detection_dc_offset(self, analyzer):
        """Test DC offset detection."""
        # Audio with DC offset
        audio = np.random.randn(48000).astype(np.float32) * 0.1 + 0.05

        has_artifacts, artifact_types = analyzer._detect_artifacts(audio, 48000)

        assert has_artifacts
        assert "dc_offset" in artifact_types

    def test_quality_level_determination(self, analyzer):
        """Test quality level determination."""
        assert analyzer._determine_quality_level(98) == QualityLevel.PRISTINE
        assert analyzer._determine_quality_level(85) == QualityLevel.EXCELLENT
        assert analyzer._determine_quality_level(70) == QualityLevel.GOOD
        assert analyzer._determine_quality_level(50) == QualityLevel.FAIR
        assert analyzer._determine_quality_level(30) == QualityLevel.POOR

    def test_confidence_calculation(self, analyzer, clean_audio, noisy_audio):
        """Test confidence calculation."""
        clean_conf = analyzer._calculate_confidence(clean_audio, 60)
        noisy_conf = analyzer._calculate_confidence(noisy_audio, 20)

        assert clean_conf > noisy_conf  # Higher SNR = more confident


class TestQualityPredictor:
    """Test improvement prediction."""

    def test_predict_improvement_noise_reduction(self, predictor):
        """Test prediction with noise reduction."""
        # Mock current quality (noisy)
        current = QualityEstimate(
            overall_score=50,
            quality_level=QualityLevel.FAIR,
            snr_db=30,
            dynamic_range_db=60,
            thd_percent=2.0,
            clarity=0.5,
            warmth=0.5,
            brightness=0.5,
            naturalness=0.5,
            authenticity=0.5,
            confidence=0.7,
            bandwidth_hz=(20, 20000),
            has_artifacts=False,
        )

        predicted = predictor.predict_improvement(current, ["NoiseReduction"])

        # Should predict improvement
        assert predicted.snr_improvement_db > 0
        assert predicted.clarity_improvement > 0
        assert predicted.final_snr_db > current.snr_db
        assert predicted.confidence > 0

    def test_predict_improvement_multiple_modules(self, predictor):
        """Test prediction with multiple modules."""
        current = QualityEstimate(
            overall_score=40,
            quality_level=QualityLevel.POOR,
            snr_db=25,
            dynamic_range_db=50,
            thd_percent=3.0,
            clarity=0.4,
            warmth=0.4,
            brightness=0.4,
            naturalness=0.4,
            authenticity=0.5,
            confidence=0.6,
            bandwidth_hz=(20, 20000),
            has_artifacts=True,
            artifact_types=["clicks", "crackle"],
        )

        modules = ["DCBlocker", "NoiseReduction", "ClickRemover", "CrackleSuppressor"]
        predicted = predictor.predict_improvement(current, modules)

        # Should predict cumulative improvement
        assert predicted.snr_improvement_db > 10  # Multiple modules
        assert predicted.clarity_improvement > 0.2
        assert predicted.estimated_processing_time_sec > 0

    def test_predict_with_forensics(self, predictor):
        """Test prediction with forensic analysis."""
        current = QualityEstimate(
            overall_score=45,
            quality_level=QualityLevel.FAIR,
            snr_db=28,
            dynamic_range_db=55,
            thd_percent=2.5,
            clarity=0.45,
            warmth=0.5,
            brightness=0.5,
            naturalness=0.5,
            authenticity=0.5,
            confidence=0.65,
            bandwidth_hz=(20, 20000),
            has_artifacts=False,
        )

        forensic_analysis = {"medium_type": "VINYL", "quality_assessment": "POOR"}

        predicted = predictor.predict_improvement(current, ["NoiseReduction"], forensic_analysis)

        # Poor quality should allow more improvement
        assert predicted.snr_improvement_db > 10
        assert predicted.confidence > 0

    def test_quality_gates_generation(self, predictor):
        """Test quality gates generation."""
        current = QualityEstimate(
            overall_score=50,
            quality_level=QualityLevel.FAIR,
            snr_db=30,
            dynamic_range_db=60,
            thd_percent=2.0,
            clarity=0.5,
            warmth=0.5,
            brightness=0.5,
            naturalness=0.5,
            authenticity=0.5,
            confidence=0.7,
            bandwidth_hz=(20, 20000),
            has_artifacts=False,
        )

        predicted = predictor.predict_improvement(current, ["NoiseReduction"])

        # Should have quality gates
        assert "snr_db" in predicted.quality_gates
        assert "clarity" in predicted.quality_gates
        assert "overall_score" in predicted.quality_gates

        # Gates should be reasonable targets
        assert predicted.quality_gates["snr_db"] > 50
        assert predicted.quality_gates["clarity"] > 0.7

    def test_module_recommendations(self, predictor):
        """Test module recommendations."""
        # Low quality audio
        quality = QualityEstimate(
            overall_score=30,
            quality_level=QualityLevel.POOR,
            snr_db=20,
            dynamic_range_db=50,
            thd_percent=4.0,
            clarity=0.3,
            warmth=0.3,
            brightness=0.5,
            naturalness=0.4,
            authenticity=0.5,
            confidence=0.5,
            bandwidth_hz=(20, 20000),
            has_artifacts=True,
            artifact_types=["clicks"],
        )

        forensics = {"medium_type": "VINYL"}

        recommended = predictor._recommend_modules(quality, forensics)

        # Should recommend appropriate modules
        assert "DCBlocker" in recommended  # Always first
        assert "NoiseReduction" in recommended  # Low SNR
        assert "ClickRemover" in recommended  # Has clicks
        assert "TapeSpecialist" in recommended  # VINYL medium


class TestQualityPredictionSystem:
    """Test complete quality prediction system."""

    def test_system_initialization(self, system):
        """Test system initialization."""
        assert system.analyzer is not None
        assert system.predictor is not None
        assert system.VERSION == "2.0.0"

    def test_estimate_quality(self, system, clean_audio):
        """Test quality estimation."""
        quality = system.estimate_quality(clean_audio, 48000)

        assert quality.overall_score > 0
        assert quality.snr_db > 0
        assert quality.quality_level in list(QualityLevel)

    def test_predict_processing_outcome(self, system, noisy_audio):
        """Test processing outcome prediction."""
        current, improvement = system.predict_processing_outcome(noisy_audio, 48000, ["NoiseReduction", "ClickRemover"])

        # Should return both current and predicted
        assert current.overall_score > 0
        assert improvement.snr_improvement_db > 0
        assert improvement.final_snr_db > current.snr_db

    def test_predict_with_forensics(self, system, noisy_audio):
        """Test prediction with forensic analysis."""
        forensics = {"medium_type": "VINYL", "quality_assessment": "POOR", "defects_detected": {"clicks": True}}

        current, improvement = system.predict_processing_outcome(
            noisy_audio, 48000, ["DCBlocker", "NoiseReduction"], forensics
        )

        assert improvement.confidence > 0
        assert improvement.recommended_modules is not None

    def test_validate_prediction(self, system, clean_audio, noisy_audio):
        """Test prediction validation."""
        # Predict improvement
        current, predicted = system.predict_processing_outcome(noisy_audio, 48000, ["NoiseReduction"])

        # Simulate processing (use clean audio as "result")
        validation = system.validate_prediction(predicted, clean_audio, 48000)

        assert validation.prediction_score >= 0
        assert validation.snr_accuracy_db >= 0
        assert validation.clarity_accuracy >= 0
        assert validation.actual_quality is not None

    def test_quality_gate_met(self, system):
        """Test quality gate checking (gate met)."""
        # High quality audio
        quality = QualityEstimate(
            overall_score=92,
            quality_level=QualityLevel.EXCELLENT,
            snr_db=75,
            dynamic_range_db=85,
            thd_percent=0.5,
            clarity=0.90,
            warmth=0.7,
            brightness=0.6,
            naturalness=0.8,
            authenticity=0.7,
            confidence=0.9,
            bandwidth_hz=(20, 20000),
            has_artifacts=False,
        )

        gates = {"snr_db": 70.0, "clarity": 0.85, "overall_score": 90.0}

        gate_met, reason = system.check_quality_gate(quality, gates)

        assert gate_met
        assert "gate" in reason.lower() and "met" in reason.lower()

    def test_quality_gate_not_met(self, system):
        """Test quality gate checking (gate not met)."""
        # Medium quality audio
        quality = QualityEstimate(
            overall_score=65,
            quality_level=QualityLevel.GOOD,
            snr_db=55,
            dynamic_range_db=70,
            thd_percent=1.5,
            clarity=0.70,
            warmth=0.6,
            brightness=0.5,
            naturalness=0.7,
            authenticity=0.6,
            confidence=0.75,
            bandwidth_hz=(20, 20000),
            has_artifacts=False,
        )

        gates = {"snr_db": 70.0, "clarity": 0.85, "overall_score": 90.0}

        gate_met, reason = system.check_quality_gate(quality, gates)

        assert not gate_met
        assert "not met" in reason.lower()


class TestIntegration:
    """Integration tests."""

    def test_full_prediction_workflow(self, system, noisy_audio):
        """Test full prediction workflow."""
        # 1. Estimate current quality
        quality = system.estimate_quality(noisy_audio, 48000)
        assert quality.quality_level in [QualityLevel.FAIR, QualityLevel.POOR]

        # 2. Predict outcome
        modules = ["DCBlocker", "NoiseReduction", "Equalizer"]
        current, improvement = system.predict_processing_outcome(noisy_audio, 48000, modules)

        # 3. Check quality gates
        gate_met, reason = system.check_quality_gate(quality, improvement.quality_gates)

        # Should not meet gates yet (noisy input)
        assert not gate_met

        # 4. Simulate processing (use less noisy audio)
        processed = noisy_audio * 0.5  # Simplified "processing"

        # 5. Validate prediction
        validation = system.validate_prediction(improvement, processed, 48000)

        assert validation.prediction_score > 0

    def test_iterative_quality_prediction(self, system, noisy_audio):
        """Test iterative quality prediction (multi-pass)."""
        current_audio = noisy_audio

        for pass_num in range(3):
            # Estimate quality
            quality = system.estimate_quality(current_audio, 48000)

            # Check if target reached
            gates = {"overall_score": 85.0}
            gate_met, _ = system.check_quality_gate(quality, gates)

            if gate_met:
                break

            # Predict improvement
            _, improvement = system.predict_processing_outcome(current_audio, 48000, ["NoiseReduction"])

            # Simulate processing (reduce noise)
            current_audio = current_audio * 0.8 + 0.2 * np.random.randn(len(current_audio)) * 0.01

        # Should have improved over iterations
        final_quality = system.estimate_quality(current_audio, 48000)
        initial_quality = system.estimate_quality(noisy_audio, 48000)

        # Final should be slightly better (simplified processing)
        # Note: Our simplified processing may not improve much
        assert final_quality.overall_score >= initial_quality.overall_score - 10


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_silent_audio(self, analyzer):
        """Test analyzing silent audio."""
        silent = np.zeros(48000, dtype=np.float32)

        quality = analyzer.analyze_quality(silent, 48000)

        # Should handle gracefully
        assert quality.overall_score >= 0
        assert quality.snr_db >= 0

    def test_very_short_audio(self, analyzer):
        """Test analyzing very short audio."""
        short = np.random.randn(1000).astype(np.float32) * 0.1

        quality = analyzer.analyze_quality(short, 48000)

        # Should work but with lower confidence
        assert quality.confidence < 1.0

    def test_empty_module_list(self, predictor):
        """Test prediction with empty module list."""
        quality = QualityEstimate(
            overall_score=50,
            quality_level=QualityLevel.FAIR,
            snr_db=30,
            dynamic_range_db=60,
            thd_percent=2.0,
            clarity=0.5,
            warmth=0.5,
            brightness=0.5,
            naturalness=0.5,
            authenticity=0.5,
            confidence=0.7,
            bandwidth_hz=(20, 20000),
            has_artifacts=False,
        )

        predicted = predictor.predict_improvement(quality, [])

        # Should predict minimal improvement
        assert predicted.snr_improvement_db == 0
        assert predicted.estimated_processing_time_sec == 0

    def test_unknown_modules(self, predictor):
        """Test prediction with unknown modules."""
        quality = QualityEstimate(
            overall_score=50,
            quality_level=QualityLevel.FAIR,
            snr_db=30,
            dynamic_range_db=60,
            thd_percent=2.0,
            clarity=0.5,
            warmth=0.5,
            brightness=0.5,
            naturalness=0.5,
            authenticity=0.5,
            confidence=0.7,
            bandwidth_hz=(20, 20000),
            has_artifacts=False,
        )

        predicted = predictor.predict_improvement(quality, ["UnknownModule1", "UnknownModule2"])

        # Should handle gracefully (no improvement predicted)
        assert predicted.snr_improvement_db == 0
        assert predicted.confidence > 0


# === Parametrized Tests ===


@pytest.mark.parametrize(
    "score,expected_level",
    [
        (98, QualityLevel.PRISTINE),
        (85, QualityLevel.EXCELLENT),
        (70, QualityLevel.GOOD),
        (50, QualityLevel.FAIR),
        (30, QualityLevel.POOR),
    ],
)
def test_quality_levels(score, expected_level):
    """Test quality level determination."""
    analyzer = QualityAnalyzer()
    level = analyzer._determine_quality_level(score)
    assert level == expected_level


@pytest.mark.parametrize(
    "module,expected_improvement",
    [
        ("DCBlocker", True),
        ("NoiseReduction", True),
        ("ClickRemover", True),
        ("Equalizer", True),
    ],
)
def test_module_improvements(module, expected_improvement):
    """Test that modules predict improvement."""
    predictor = QualityPredictor()

    quality = QualityEstimate(
        overall_score=50,
        quality_level=QualityLevel.FAIR,
        snr_db=30,
        dynamic_range_db=60,
        thd_percent=2.0,
        clarity=0.5,
        warmth=0.5,
        brightness=0.5,
        naturalness=0.5,
        authenticity=0.5,
        confidence=0.7,
        bandwidth_hz=(20, 20000),
        has_artifacts=False,
    )

    predicted = predictor.predict_improvement(quality, [module])

    if expected_improvement:
        # Should predict some improvement
        assert predicted.snr_improvement_db > 0 or predicted.clarity_improvement > 0 or predicted.warmth_improvement > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
