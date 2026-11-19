import pytest

"""
Test suite for core/core_utils.py
Tests basic utility functions: compute_rms, compute_loudness, audio_stats
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.core_utils import audio_stats, compute_loudness, compute_rms, log_message


@pytest.mark.unit
def test_compute_rms_simple():
    """Test RMS computation for simple signals"""
    # DC signal
    audio = np.ones(1000) * 0.5
    rms = compute_rms(audio)
    assert np.isclose(rms, 0.5, rtol=1e-6)


def test_compute_loudness():
    """Test loudness computation"""
    # High amplitude signal
    audio = np.ones(1000) * 0.8
    loudness = compute_loudness(audio)
    assert loudness < 0  # Should be in dB, negative for signals < 1.0
    assert loudness > -10  # Should be reasonably loud


def test_audio_stats_comprehensive():
    """Test audio_stats returns all expected keys"""
    audio = np.random.randn(1000) * 0.5
    stats = audio_stats(audio)

    # Check all keys present
    assert "peak" in stats
    assert "rms" in stats
    assert "loudness" in stats

    # Check values are reasonable
    assert stats["peak"] >= 0  # Peak can exceed 1.0 for non-normalized audio
    assert stats["rms"] >= 0


def test_compute_rms_zero_signal():
    """Test RMS of zero signal"""
    audio = np.zeros(1000)
    rms = compute_rms(audio)
    assert rms == 0.0


def test_compute_rms_alternating_signal():
    """Test RMS of alternating +1/-1 signal"""
    audio = np.array([1.0, -1.0] * 500)
    rms = compute_rms(audio)
    assert np.isclose(rms, 1.0, rtol=1e-6)


def test_compute_loudness_zero_signal():
    """Test loudness of zero signal (edge case)"""
    audio = np.zeros(1000)
    loudness = compute_loudness(audio)
    # Should handle zero gracefully (likely returns very low dB)
    assert loudness < -60  # Should be very quiet


def test_audio_stats_stereo_simulation():
    """Test audio_stats with multi-channel simulation"""
    # Create stereo-like signal (2 channels)
    audio_left = np.random.randn(1000) * 0.5
    audio_right = np.random.randn(1000) * 0.3
    # Test both channels separately
    stats_left = audio_stats(audio_left)
    stats_right = audio_stats(audio_right)
    # Left should be louder
    assert stats_left["rms"] > stats_right["rms"]


def test_audio_stats_clipped_signal():
    """Test audio_stats with clipped signal"""
    audio = np.clip(np.random.randn(1000) * 2.0, -1.0, 1.0)
    stats = audio_stats(audio)
    # Peak should be 1.0
    assert np.isclose(stats["peak"], 1.0, rtol=1e-6)
    assert stats["rms"] > 0


def test_log_message_basic():
    """Test log_message writes to file"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as tmp:
        logfile = tmp.name

    try:
        log_message("Test log message", logfile=logfile)
        # Check file exists and contains message
        with open(logfile) as f:
            content = f.read()
        assert "Test log message" in content
    finally:
        # Cleanup
        if os.path.exists(logfile):
            os.remove(logfile)
