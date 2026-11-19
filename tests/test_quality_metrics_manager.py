import pytest

"""Test Quality Metrics Manager Integration."""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.quality_metrics_manager import (
    QualityMetricsManager,
    assess_audio_quality,
)


@pytest.mark.unit
def test_quality_metrics_manager_initialization():
    """Test QualityMetricsManager initialization."""
    manager = QualityMetricsManager(enable_all=False)
    assert manager is not None

    # Lazy loading
    assert manager._cdpam is None
    # _dnsmos entfernt — verboten §4.4+§10.2 (Sprach-Metrik)
    # _nisqa entfernt — verboten §4.4+§10.2 (Sprach-Metrik)
    assert manager._visqol is None


def test_assess_quality_integration():
    """Test comprehensive quality assessment with all non-reference metrics."""
    # Create test audio
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr)
        audio_path = tmp.name

    try:
        # Run assessment
        manager = QualityMetricsManager()
        results = manager.assess_quality(audio_path)

        # Validate structure
        assert "audio_file" in results
        assert "metrics" in results
        assert "aggregate" in results

        # Check for individual metrics
        metrics = results["metrics"]

        # CDPAM
        if "cdpam" in metrics and "score" in metrics["cdpam"]:
            assert 0 <= metrics["cdpam"]["score"] <= 100
            assert "rating" in metrics["cdpam"]
            print(f"✓ CDPAM: {metrics['cdpam']['score']:.2f}/100")

        # DNSMOS/NISQA entfernt — verboten §4.4+§10.2 (Sprach-Metriken)

        # Aggregate
        aggregate = results["aggregate"]
        assert "overall_score" in aggregate
        assert 0 <= aggregate["overall_score"] <= 1
        assert "overall_rating" in aggregate
        print(f"✓ Aggregate: {aggregate['overall_score']:.3f} - {aggregate['overall_rating']}")

    finally:
        Path(audio_path).unlink(missing_ok=True)


def test_assess_visqol_reference_based():
    """Test ViSQOL reference-based assessment."""
    # Create test audio
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Reference: clean sine
    ref_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Degraded: noisy sine
    deg_audio = ref_audio + 0.05 * np.random.randn(len(t)).astype(np.float32)
    deg_audio = np.clip(deg_audio, -1.0, 1.0)

    # Save to temp files
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_ref:
        sf.write(tmp_ref.name, ref_audio, sr)
        ref_path = tmp_ref.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_deg:
        sf.write(tmp_deg.name, deg_audio, sr)
        deg_path = tmp_deg.name

    try:
        manager = QualityMetricsManager()
        result = manager.assess_visqol(ref_path, deg_path, mode="audio")

        # Validate
        assert "ViSQOL_MOS" in result
        assert 1.0 <= result["ViSQOL_MOS"] <= 5.0
        assert "rating" in result
        assert result["mode"] == "audio"

        print(f"✓ ViSQOL: MOS-LQO={result['ViSQOL_MOS']:.2f}/5.0")

    finally:
        Path(ref_path).unlink(missing_ok=True)
        Path(deg_path).unlink(missing_ok=True)


def test_comprehensive_assessment():
    """Test comprehensive assessment with all metrics."""
    # Create test audio
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    ref_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    deg_audio = ref_audio + 0.02 * np.random.randn(len(t)).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_ref:
        sf.write(tmp_ref.name, ref_audio, sr)
        ref_path = tmp_ref.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_deg:
        sf.write(tmp_deg.name, deg_audio, sr)
        deg_path = tmp_deg.name

    try:
        manager = QualityMetricsManager()
        results = manager.assess_comprehensive(deg_path, ref_path)

        # Validate structure
        assert "metrics" in results
        assert "aggregate" in results

        # Should have all metrics including ViSQOL
        metrics = results["metrics"]

        # Check counts
        # dnsmos/nisqa-Zeilen entfernt — verboten §4.4+§10.2 (Sprach-Metriken)
        num_metrics = sum(
            [
                "cdpam" in metrics and "score" in metrics["cdpam"],
                "visqol" in metrics and "ViSQOL_MOS" in metrics["visqol"],
            ]
        )

        assert num_metrics >= 1, "At least one metric should be available"
        print(f"✓ Comprehensive Assessment: {num_metrics} metrics available")

        # Generate report
        report = manager.generate_report(results)
        assert isinstance(report, str)
        assert "QUALITY METRICS REPORT" in report
        print("\n" + report)

    finally:
        Path(ref_path).unlink(missing_ok=True)
        Path(deg_path).unlink(missing_ok=True)


def test_convenience_function():
    """Test convenience function assess_audio_quality."""
    sr = 48000
    t = np.linspace(0, 1.0, sr)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr)
        audio_path = tmp.name

    try:
        results = assess_audio_quality(audio_path)

        assert "metrics" in results
        assert "aggregate" in results

        print(f"✓ Convenience function works: {results['aggregate']['overall_rating']}")

    finally:
        Path(audio_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 80)
    print("QUALITY METRICS MANAGER TESTS")
    print("=" * 80 + "\n")

    print("Test 1: Initialization")
    test_quality_metrics_manager_initialization()

    print("\nTest 2: Assess Quality (Non-Reference)")
    test_assess_quality_integration()

    print("\nTest 3: ViSQOL (Reference-Based)")
    test_assess_visqol_reference_based()

    print("\nTest 4: Comprehensive Assessment")
    test_comprehensive_assessment()

    print("\nTest 5: Convenience Function")
    test_convenience_function()

    print("\n" + "=" * 80)
    print("✓ All tests completed!")
    print("=" * 80 + "\n")
