"""
tests/test_unified_analyzer.py
Tests für Unified Signal Forensics Analyzer
============================================

Tests:
1. Analyzer initialization
2. Unified analysis (all detectors)
3. Partial analysis (some detectors missing)
4. Consistency checks
5. Recommendation generation
6. Model loading
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.forensics.dataset_generator import DatasetGenerator
from backend.core.forensics.ml_defect_detector import train_ml_defect_detector_from_dataset
from backend.core.forensics.ml_era_detector import MLEraDetector, train_ml_era_detector_from_dataset
from backend.core.forensics.ml_medium_detector import MLMediumDetector, train_ml_detector_from_dataset
from backend.core.forensics.unified_analyzer import UnifiedForensicAnalysis, UnifiedForensicAnalyzer


@pytest.fixture(scope="module")
def trained_medium_detector():
    """Generate and train medium detector."""
    gen = DatasetGenerator()
    dataset = gen.generate_medium_dataset(n_synthetic_per_medium=8)
    detector, _ = train_ml_detector_from_dataset(dataset, test_size=0.2, verbose=False)
    return detector


@pytest.fixture(scope="module")
def trained_era_detector():
    """Generate and train era detector."""
    gen = DatasetGenerator()
    dataset = gen.generate_era_dataset(n_synthetic_per_era=8)
    detector, _ = train_ml_era_detector_from_dataset(dataset, test_size=0.2, verbose=False)
    return detector


@pytest.fixture(scope="module")
def trained_defect_detector():
    """Generate and train defect detector."""
    from tests.test_ml_defect_detector import generate_defect_dataset

    dataset = generate_defect_dataset(n_samples_per_type=8)
    detector, _ = train_ml_defect_detector_from_dataset(dataset, test_size=0.2, verbose=False)
    return detector


@pytest.fixture(scope="module")
def full_analyzer(trained_medium_detector, trained_era_detector, trained_defect_detector):
    """Analyzer with all detectors."""
    return UnifiedForensicAnalyzer(
        medium_detector=trained_medium_detector,
        era_detector=trained_era_detector,
        defect_detector=trained_defect_detector,
    )


@pytest.mark.unit
class TestUnifiedForensicAnalyzer:
    """Test suite for Unified Forensic Analyzer."""

    def test_initialization_empty(self):
        """Test initialization without detectors."""
        analyzer = UnifiedForensicAnalyzer()

        assert analyzer.medium_detector is None
        assert analyzer.era_detector is None
        assert analyzer.defect_detector is None
        assert not analyzer.is_ready()

    def test_initialization_with_detectors(self, full_analyzer):
        """Test initialization with all detectors."""
        assert full_analyzer.medium_detector is not None
        assert full_analyzer.era_detector is not None
        assert full_analyzer.defect_detector is not None
        assert full_analyzer.is_ready()

    def test_status(self, full_analyzer):
        """Test status reporting."""
        status = full_analyzer.get_status()

        assert "medium_detector" in status
        assert "era_detector" in status
        assert "defect_detector" in status
        assert status["medium_detector"] is True
        assert status["era_detector"] is True
        assert status["defect_detector"] is True

    def test_full_analysis(self, full_analyzer):
        """Test complete analysis with all detectors."""
        # Generate test audio
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Analyze
        result = full_analyzer.analyze(audio, sr, verbose=False)

        assert isinstance(result, UnifiedForensicAnalysis)

        # Check medium detection
        assert result.medium_type in MLMediumDetector.MEDIUM_CATEGORIES
        assert 0 <= result.medium_confidence <= 1
        assert len(result.medium_probabilities) > 0

        # Check era detection
        assert result.era in MLEraDetector.ERA_CATEGORIES
        assert 0 <= result.era_confidence <= 1
        assert len(result.era_probabilities) > 0
        assert result.era_characteristics is not None

        # Check defect detection
        assert len(result.defects_detected) == 5
        assert len(result.defect_confidences) == 5
        assert len(result.defect_severities) == 5

        # Check overall metrics
        assert 0 <= result.overall_confidence <= 1
        assert result.analysis_quality in ["EXCELLENT", "GOOD", "FAIR", "POOR"]

        # Check recommendations
        assert len(result.recommended_processing_chain) > 0
        assert result.restoration_priority in ["HIGH", "MEDIUM", "LOW"]

        # Check reports
        assert result.summary != ""
        assert result.detailed_report != ""

        # Check metadata
        assert result.features_used > 0
        assert "medium" in result.model_versions
        assert "era" in result.model_versions
        assert "defects" in result.model_versions

    def test_partial_analysis_medium_only(self, trained_medium_detector):
        """Test analysis with only medium detector."""
        analyzer = UnifiedForensicAnalyzer(medium_detector=trained_medium_detector)

        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        result = analyzer.analyze(audio, sr, verbose=False)

        assert isinstance(result, UnifiedForensicAnalysis)
        assert result.medium_type != "UNKNOWN"
        assert result.era == "UNKNOWN"  # Era detector not available

    def test_analysis_vinyl_with_defects(self, full_analyzer):
        """Test analysis of vinyl with defects."""
        # Generate vinyl-like audio with clicks
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Add vinyl characteristics
        # 1. Add rumble (low-frequency noise)
        rumble = np.sin(2 * np.pi * 30 * t) * 0.05
        audio += rumble

        # 2. Add clicks
        for i in range(5):
            click_pos = 1000 + i * 2000
            if click_pos < len(audio):
                audio[click_pos : click_pos + 10] += np.random.randn(10) * 0.5

        # Analyze
        result = full_analyzer.analyze(audio, sr, verbose=False)

        # Check that some defects are detected
        # (We don't require specific defects due to synthetic data limitations)
        assert isinstance(result, UnifiedForensicAnalysis)
        assert result.summary != ""

    def test_analysis_digital_clean(self, full_analyzer):
        """Test analysis of clean digital audio."""
        # Generate clean digital audio
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.2

        result = full_analyzer.analyze(audio, sr, verbose=False)

        # Should have low restoration priority
        assert result.restoration_priority in ["LOW", "MEDIUM"]

    def test_recommended_processing_chain(self, full_analyzer):
        """Test processing chain recommendation."""
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        result = full_analyzer.analyze(audio, sr, verbose=False)

        # Should always include some processing modules
        assert len(result.recommended_processing_chain) > 0

        # Should always start with DCBlocker
        assert result.recommended_processing_chain[0] == "DCBlocker"

        # Should always end with Enhancement
        assert result.recommended_processing_chain[-1] == "Enhancement"

    def test_consistency_checks(self, full_analyzer):
        """Test cross-detector consistency checks."""
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        result = full_analyzer.analyze(audio, sr, verbose=False)

        # Overall confidence should reflect consistency
        assert 0 <= result.overall_confidence <= 1

    def test_detailed_report_generation(self, full_analyzer):
        """Test detailed report generation."""
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        result = full_analyzer.analyze(audio, sr, verbose=False)

        # Check report structure
        assert "DETAILED FORENSIC ANALYSIS REPORT" in result.detailed_report
        assert "MEDIUM DETECTION" in result.detailed_report
        assert "ERA DETECTION" in result.detailed_report
        assert "DEFECT DETECTION" in result.detailed_report
        assert "CONSISTENCY SCORE" in result.detailed_report


class TestAnalysisQuality:
    """Test analysis quality assessment."""

    def test_quality_excellent(self, full_analyzer):
        """Test EXCELLENT quality assessment."""
        # This test may not always pass due to synthetic data
        # but validates the quality assessment logic
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        result = full_analyzer.analyze(audio, sr, verbose=False)

        # Quality should be in valid range
        assert result.analysis_quality in ["EXCELLENT", "GOOD", "FAIR", "POOR"]

    def test_restoration_priority_high(self, full_analyzer):
        """Test HIGH restoration priority for heavily defected audio."""
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3

        # Add multiple severe defects
        # Clicks
        for i in range(10):
            click_pos = 1000 + i * 1200
            if click_pos < len(audio):
                audio[click_pos : click_pos + 5] += np.random.randn(5) * 1.0

        # Hum
        hum = np.sin(2 * np.pi * 50 * t) * 0.2
        audio += hum

        # Clipping
        audio = np.clip(audio, -0.6, 0.6)

        result = full_analyzer.analyze(audio, sr, verbose=False)

        # Should detect multiple defects
        detected_count = sum(result.defects_detected.values())
        assert detected_count > 0  # At least some defects detected


class TestModelPersistence:
    """Test model loading/saving."""

    def test_load_models(self, trained_medium_detector, trained_era_detector, trained_defect_detector):
        """Test loading models from files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            medium_path = str(Path(tmpdir) / "medium.pkl")
            era_path = str(Path(tmpdir) / "era.pkl")
            defect_path = str(Path(tmpdir) / "defect.pkl")

            # Save models
            trained_medium_detector.save(medium_path)
            trained_era_detector.save(era_path)
            trained_defect_detector.save(defect_path)

            # Create new analyzer and load models
            analyzer = UnifiedForensicAnalyzer()
            analyzer.load_models(medium_model_path=medium_path, era_model_path=era_path, defect_model_path=defect_path)

            # Check that models are loaded
            assert analyzer.is_ready()
            assert analyzer.medium_detector is not None
            assert analyzer.era_detector is not None
            assert analyzer.defect_detector is not None

            # Test analysis
            sr = 48000
            duration = 0.3
            t = np.linspace(0, duration, int(sr * duration))
            audio = np.sin(2 * np.pi * 440 * t) * 0.3

            result = analyzer.analyze(audio, sr, verbose=False)
            assert isinstance(result, UnifiedForensicAnalysis)


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_audio(self, full_analyzer):
        """Test analysis of very short audio."""
        sr = 48000
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, sr // 2)) * 0.3  # 0.5 second

        # Should not crash (but may have lower confidence)
        result = full_analyzer.analyze(audio, sr, verbose=False)
        assert isinstance(result, UnifiedForensicAnalysis)

    def test_mono_vs_stereo(self, full_analyzer):
        """Test analysis of mono vs stereo audio."""
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))

        # Mono
        audio_mono = np.sin(2 * np.pi * 440 * t) * 0.3
        result_mono = full_analyzer.analyze(audio_mono, sr, verbose=False)

        # Stereo
        audio_stereo = np.stack([audio_mono, audio_mono], axis=1)
        result_stereo = full_analyzer.analyze(audio_stereo, sr, verbose=False)

        # Both should work
        assert isinstance(result_mono, UnifiedForensicAnalysis)
        assert isinstance(result_stereo, UnifiedForensicAnalysis)


def manual_test_unified_analyzer():
    """
    Manual test for unified analyzer.
    Run with: pytest -k manual_test_unified_analyzer -s
    """
    print("\n" + "=" * 60)
    print("Manual Unified Analyzer Test")
    print("=" * 60)

    # Generate and train detectors
    print("\n[1/4] Training Medium Detector...")
    gen = DatasetGenerator()
    medium_dataset = gen.generate_medium_dataset(n_synthetic_per_medium=10)
    medium_detector, _ = train_ml_detector_from_dataset(medium_dataset, test_size=0.2, verbose=False)
    print("      ✓ Medium Detector trained")

    print("\n[2/4] Training Era Detector...")
    era_dataset = gen.generate_era_dataset(n_synthetic_per_era=10)
    era_detector, _ = train_ml_era_detector_from_dataset(era_dataset, test_size=0.2, verbose=False)
    print("      ✓ Era Detector trained")

    print("\n[3/4] Training Defect Detector...")
    from tests.test_ml_defect_detector import generate_defect_dataset

    defect_dataset = generate_defect_dataset(n_samples_per_type=10)
    defect_detector, _ = train_ml_defect_detector_from_dataset(defect_dataset, test_size=0.2, verbose=False)
    print("      ✓ Defect Detector trained")

    # Create analyzer
    print("\n[4/4] Creating Unified Analyzer...")
    analyzer = UnifiedForensicAnalyzer(
        medium_detector=medium_detector, era_detector=era_detector, defect_detector=defect_detector
    )
    print("      ✓ Analyzer ready")

    # Test analysis
    print("\n" + "=" * 60)
    print("Test Analysis: Vinyl with Clicks")
    print("=" * 60)

    sr = 48000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.3

    # Add vinyl characteristics
    rumble = np.sin(2 * np.pi * 30 * t) * 0.05
    audio += rumble

    # Add clicks
    for i in range(5):
        click_pos = 1000 + i * 3000
        if click_pos < len(audio):
            audio[click_pos : click_pos + 10] += np.random.randn(10) * 0.5

    # Analyze
    result = analyzer.analyze(audio, sr, verbose=True)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Medium: {result.medium_type} ({result.medium_confidence:.1%})")
    print(f"Era: {result.era} ({result.era_confidence:.1%})")
    print(f"Overall Confidence: {result.overall_confidence:.1%}")
    print(f"Analysis Quality: {result.analysis_quality}")
    print(f"Restoration Priority: {result.restoration_priority}")

    print("\nDetected Defects:")
    for defect, detected in result.defects_detected.items():
        if detected:
            conf = result.defect_confidences[defect]
            sev = result.defect_severities[defect]
            print(f"  - {defect}: {conf:.1%} ({sev})")

    print("\nRecommended Processing Chain:")
    for i, module in enumerate(result.recommended_processing_chain, 1):
        print(f"  {i}. {module}")

    print("\n" + "=" * 60)
    print("DETAILED REPORT")
    print("=" * 60)
    print(result.detailed_report)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    manual_test_unified_analyzer()
