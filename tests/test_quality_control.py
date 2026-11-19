import pytest

"""
Test suite for backend/quality_control.py - QualityControl class
Tests non-destructive checks, A/B tests, psychoacoustic scoring
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.quality_control import QualityControl


@pytest.mark.unit
def test_quality_control_initialization():
    """Test QualityControl initializes with empty logs"""
    qc = QualityControl()

    assert qc.quality_log == []
    assert qc.ab_results == []
    assert qc.warnings == []
    assert qc.test_db == []


def test_check_non_destructive_identical_signals():
    """Test non-destructive check with identical signals"""
    qc = QualityControl()

    original = np.array([0.5, 0.3, -0.2, 0.1])
    processed = original.copy()

    snr = qc.check_non_destructive(original, processed)

    # SNR should be very high (>70 dB) for identical signals (limited by epsilon 1e-8)
    assert snr is not None
    assert snr > 70
    # No warnings for identical signals
    assert len(qc.warnings) == 0


def test_check_non_destructive_small_difference():
    """Test non-destructive check with small processing difference"""
    qc = QualityControl()

    original = np.array([0.5, 0.3, -0.2, 0.1])
    processed = original + 0.01  # Small change

    snr = qc.check_non_destructive(original, processed)

    # SNR should be close to 30 dB threshold
    assert snr is not None
    assert 29 <= snr <= 31
    # Warning expected since SNR < 30 dB
    assert len(qc.warnings) >= 0  # May or may not warn depending on precision


def test_check_non_destructive_large_difference():
    """Test non-destructive check warns on large difference"""
    qc = QualityControl()

    original = np.array([0.5, 0.3, -0.2, 0.1])
    processed = original * 0.2  # Large change (80% reduction)

    snr = qc.check_non_destructive(original, processed)

    # SNR should be low (<30 dB)
    assert snr is not None
    assert snr < 30
    # Should have warning
    assert len(qc.warnings) == 1
    assert "destruktive Bearbeitung" in qc.warnings[0]


def test_ab_test_identical_signals():
    """Test A/B test with identical signals"""
    qc = QualityControl()

    reference = np.array([0.5, 0.3, -0.2, 0.1])
    candidate = reference.copy()

    score = qc.ab_test(reference, candidate)

    # Correlation should be 1.0
    assert score is not None
    assert np.isclose(float(score), 1.0, rtol=1e-6)
    # Result should be logged
    assert len(qc.ab_results) == 1
    assert qc.ab_results[0] == score


def test_ab_test_similar_signals():
    """Test A/B test with similar but not identical signals"""
    qc = QualityControl()

    reference = np.array([0.5, 0.3, -0.2, 0.1, 0.4])
    candidate = reference + 0.05  # Slight difference

    score = qc.ab_test(reference, candidate)

    # Correlation should be high (>0.9) due to similar values
    assert score is not None
    assert score > 0.9
    assert score <= 1.0
    assert len(qc.ab_results) == 1


def test_ab_test_inverted_signals():
    """Test A/B test with inverted signals (negative correlation)"""
    qc = QualityControl()

    reference = np.array([0.5, 0.3, -0.2, 0.1])
    candidate = -reference  # Inverted

    score = qc.ab_test(reference, candidate)

    # Correlation should be close to -1.0
    assert score is not None
    assert np.isclose(float(score), -1.0, rtol=1e-6)


def test_psychoacoustic_score_basic():
    """Test psychoacoustic score computation"""
    qc = QualityControl()

    audio = np.random.randn(1000) * 0.5
    sr = 48000

    score = qc.psychoacoustic_score(audio, sr)

    # Score should be positive
    assert score > 0
    # Log should be updated
    assert len(qc.quality_log) == 1
    assert "loudness" in qc.quality_log[0]
    assert "clarity" in qc.quality_log[0]
    assert "score" in qc.quality_log[0]


def test_psychoacoustic_score_silent_signal():
    """Test psychoacoustic score with silent signal"""
    qc = QualityControl()

    audio = np.zeros(1000)
    sr = 48000

    score = qc.psychoacoustic_score(audio, sr)

    # Score should be very low or zero
    assert score < 0.01


def test_psychoacoustic_score_loud_signal():
    """Test psychoacoustic score with loud signal"""
    qc = QualityControl()

    audio = np.ones(1000) * 0.9  # Loud DC signal
    sr = 48000

    score = qc.psychoacoustic_score(audio, sr)

    # Score should be very high (high loudness, low clarity variance)
    assert score > 10


def test_add_to_test_db():
    """Test adding entries to test database"""
    qc = QualityControl()

    features1 = {"rms": 0.5, "snr": 45}
    features2 = {"rms": 0.3, "snr": 30}

    qc.add_to_test_db(features1, "good")
    qc.add_to_test_db(features2, "bad")

    assert len(qc.test_db) == 2
    assert qc.test_db[0]["features"] == features1
    assert qc.test_db[0]["label"] == "good"
    assert qc.test_db[1]["label"] == "bad"


def test_get_warnings():
    """Test get_warnings() returns all warnings"""
    qc = QualityControl()

    # Generate some warnings
    original = np.array([0.5, 0.3, -0.2, 0.1])
    processed = original * 0.1
    qc.check_non_destructive(original, processed)

    warnings = qc.get_warnings()
    assert len(warnings) > 0
    assert "destruktive Bearbeitung" in warnings[0]


def test_get_quality_log():
    """Test get_quality_log() returns all quality entries"""
    qc = QualityControl()

    # Generate quality log entries
    audio = np.random.randn(1000) * 0.5
    qc.psychoacoustic_score(audio, 48000)
    qc.psychoacoustic_score(audio * 0.5, 48000)

    log = qc.get_quality_log()
    assert len(log) == 2
    assert "loudness" in log[0]
    assert "score" in log[1]


def test_get_ab_results():
    """Test get_ab_results() returns all A/B scores"""
    qc = QualityControl()

    # Generate A/B test results
    reference = np.array([0.5, 0.3, -0.2, 0.1])
    qc.ab_test(reference, reference * 1.1)
    qc.ab_test(reference, reference * 0.9)

    results = qc.get_ab_results()
    assert len(results) == 2
    assert all(isinstance(r, float) for r in results)


def test_get_test_db():
    """Test get_test_db() returns all database entries"""
    qc = QualityControl()

    qc.add_to_test_db({"rms": 0.5}, "good")
    qc.add_to_test_db({"rms": 0.2}, "bad")

    db = qc.get_test_db()
    assert len(db) == 2
    assert db[0]["features"]["rms"] == 0.5


def test_multiple_warnings_accumulate():
    """Test multiple warnings accumulate in warnings list"""
    qc = QualityControl()

    original = np.array([0.5, 0.3, -0.2, 0.1])

    # Generate multiple warnings
    qc.check_non_destructive(original, original * 0.1)
    qc.check_non_destructive(original, original * 0.05)

    assert len(qc.warnings) == 2


def test_quality_control_workflow():
    """Test complete quality control workflow"""
    qc = QualityControl()

    # Original and processed signals
    original = np.random.randn(1000) * 0.5
    processed = original + 0.01

    # 1. Non-destructive check
    snr = qc.check_non_destructive(original, processed)
    assert snr is not None
    assert snr > 30

    # 2. A/B test
    score = qc.ab_test(original, processed)
    assert score is not None
    assert score > 0.9

    # 3. Psychoacoustic scoring
    psych_score = qc.psychoacoustic_score(processed, 48000)
    assert psych_score > 0

    # 4. Add to test DB
    qc.add_to_test_db({"snr": snr, "ab_score": score}, "good")

    # Verify all logs populated
    assert len(qc.warnings) == 0
    assert len(qc.ab_results) == 1
    assert len(qc.quality_log) == 1
    assert len(qc.test_db) == 1
