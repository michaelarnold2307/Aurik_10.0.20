"""
Tests for RF Interference Remover (GAP #6)

Tests:
- RF interference detection
- Harmonic filtering
- Stereo support
- Auto-detection
- Edge cases
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.dsp.rf_interference_remover import RFInterferenceRemover


@pytest.mark.unit
class TestRFInterferenceDetection:
    """Test RF interference detection."""

    def test_detects_single_interference(self):
        """Should detect single RF interference frequency."""
        # Create test signal
        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Clean signal
        clean = 0.5 * np.sin(2 * np.pi * 440 * t)

        # Add RF interference at 12 kHz
        interference = 0.05 * np.sin(2 * np.pi * 12000 * t)
        contaminated = clean + interference

        # Detect
        remover = RFInterferenceRemover(detection_threshold_db=-65.0)
        detected_freqs = remover.detect_interference_frequencies(contaminated, sr)

        # Should detect around 12 kHz
        assert len(detected_freqs) >= 1
        assert any(11800 < freq < 12200 for freq in detected_freqs)

    def test_detects_multiple_interferences(self):
        """Should detect multiple RF interference frequencies."""
        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        clean = 0.5 * np.sin(2 * np.pi * 440 * t)
        interference1 = 0.05 * np.sin(2 * np.pi * 12000 * t)
        interference2 = 0.03 * np.sin(2 * np.pi * 15000 * t)

        contaminated = clean + interference1 + interference2

        remover = RFInterferenceRemover(detection_threshold_db=-65.0)
        detected_freqs = remover.detect_interference_frequencies(contaminated, sr)

        # Should detect both
        assert len(detected_freqs) >= 2

    def test_ignores_musical_content(self):
        """Should not detect musical content as interference."""
        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Musical signal with harmonics (not interference)
        musical = np.sin(2 * np.pi * 440 * t)
        for harmonic in range(2, 10):
            musical += 0.5**harmonic * np.sin(2 * np.pi * 440 * harmonic * t)

        remover = RFInterferenceRemover(min_interference_freq=5000.0)
        detected_freqs = remover.detect_interference_frequencies(musical, sr)

        # Should detect few or no frequencies (musical harmonics < 5kHz)
        assert len(detected_freqs) <= 2  # May detect some high harmonics

    def test_handles_harmonically_related_interference(self):
        """Should detect fundamental and harmonics."""
        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Fundamental + 2nd harmonic RF interference
        fundamental_freq = 10000
        interference1 = 0.05 * np.sin(2 * np.pi * fundamental_freq * t)
        interference2 = 0.03 * np.sin(2 * np.pi * fundamental_freq * 2 * t)  # 20 kHz

        contaminated = interference1 + interference2

        remover = RFInterferenceRemover(detection_threshold_db=-65.0, harmonic_tolerance=0.05)
        detected_freqs = remover.detect_interference_frequencies(contaminated, sr)

        # Should detect both fundamental and harmonic
        assert len(detected_freqs) >= 1


class TestRFInterferenceRemoval:
    """Test RF interference removal."""

    def test_removes_single_interference(self):
        """Should remove single RF interference."""
        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        clean = 0.5 * np.sin(2 * np.pi * 440 * t)
        interference = 0.05 * np.sin(2 * np.pi * 12000 * t)
        contaminated = clean + interference

        remover = RFInterferenceRemover()
        processed, metrics = remover.process(contaminated, sr)

        # Should have detected and removed interference
        assert metrics["num_interference"] >= 1

        # RMS should be reduced (interference removed)
        rms_contaminated = np.sqrt(np.mean(contaminated**2))
        rms_processed = np.sqrt(np.mean(processed**2))
        assert rms_processed < rms_contaminated

    def test_preserves_clean_signal(self):
        """Should preserve clean audio without interference."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Clean signal (no interference)
        clean = 0.5 * np.sin(2 * np.pi * 440 * t)

        remover = RFInterferenceRemover()
        processed, metrics = remover.process(clean, sr)

        # Should detect no interference
        assert metrics["num_interference"] == 0

        # Should not alter signal
        np.testing.assert_allclose(processed, clean, rtol=0.01)

    def test_manual_frequency_specification(self):
        """Should remove interference at manually specified frequencies."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        clean = 0.5 * np.sin(2 * np.pi * 440 * t)
        interference = 0.05 * np.sin(2 * np.pi * 12000 * t)
        contaminated = clean + interference

        remover = RFInterferenceRemover()
        processed = remover.remove_interference(contaminated, sr, interference_freqs=[12000.0])

        # RMS should be reduced
        rms_contaminated = np.sqrt(np.mean(contaminated**2))
        rms_processed = np.sqrt(np.mean(processed**2))
        assert rms_processed < rms_contaminated


class TestStereoSupport:
    """Test stereo audio support."""

    def test_stereo_channels_first_format(self):
        """Should handle (channels, samples) format."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Stereo with different interference in each channel
        left = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.sin(2 * np.pi * 12000 * t)
        right = 0.5 * np.sin(2 * np.pi * 880 * t) + 0.03 * np.sin(2 * np.pi * 15000 * t)

        audio = np.vstack([left, right])  # (2, samples)

        remover = RFInterferenceRemover()
        processed, metrics = remover.process(audio, sr)

        # Should process both channels
        assert processed.shape == audio.shape
        assert processed.shape[0] == 2

        # Should detect interference
        assert metrics["num_interference"] >= 1

    def test_stereo_channels_last_format(self):
        """Should handle (samples, channels) format."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        left = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.sin(2 * np.pi * 12000 * t)
        right = 0.5 * np.sin(2 * np.pi * 880 * t) + 0.03 * np.sin(2 * np.pi * 15000 * t)

        audio = np.column_stack([left, right])  # (samples, 2)

        remover = RFInterferenceRemover()
        processed, metrics = remover.process(audio, sr)

        # Should process both channels
        assert processed.shape == audio.shape
        assert processed.shape[1] == 2

        # Should detect interference
        assert metrics["num_interference"] >= 1

    def test_mono_signal(self):
        """Should handle mono signals."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        mono = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.sin(2 * np.pi * 12000 * t)

        remover = RFInterferenceRemover()
        processed, metrics = remover.process(mono, sr)

        # Should process mono
        assert processed.shape == mono.shape
        assert processed.ndim == 1


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_handles_silent_audio(self):
        """Should handle silent audio without errors."""
        sr = 48000
        duration = 1.0
        silent = np.zeros(int(sr * duration))

        remover = RFInterferenceRemover()
        processed, metrics = remover.process(silent, sr)

        assert metrics["num_interference"] == 0
        np.testing.assert_array_equal(processed, silent)

    def test_handles_very_short_audio(self):
        """Should handle very short audio clips."""
        sr = 48000
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(sr * duration))

        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        remover = RFInterferenceRemover()
        processed, metrics = remover.process(audio, sr)

        # Should not crash
        assert processed.shape == audio.shape

    def test_handles_near_nyquist_interference(self):
        """Should handle interference near Nyquist frequency."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Interference at 22 kHz (near Nyquist 24 kHz)
        clean = 0.5 * np.sin(2 * np.pi * 440 * t)
        interference = 0.05 * np.sin(2 * np.pi * 22000 * t)
        contaminated = clean + interference

        remover = RFInterferenceRemover(max_interference_freq=23000.0)
        processed, metrics = remover.process(contaminated, sr)

        # Should handle without errors (may or may not detect due to Nyquist limits)
        assert processed.shape == contaminated.shape

    def test_respects_frequency_range_limits(self):
        """Should only detect interference within specified frequency range."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Interference at 3 kHz (below default min_interference_freq=5kHz)
        low_interference = 0.1 * np.sin(2 * np.pi * 3000 * t)

        remover = RFInterferenceRemover(min_interference_freq=5000.0)
        detected = remover.detect_interference_frequencies(low_interference, sr)

        # Should not detect (below minimum frequency)
        assert len(detected) == 0


class TestPerformance:
    """Test performance characteristics."""

    def test_reasonable_processing_time(self):
        """Processing time should be reasonable for typical audio."""
        import time

        sr = 48000
        duration = 10.0  # 10 seconds
        t = np.linspace(0, duration, int(sr * duration))

        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.sin(2 * np.pi * 12000 * t)

        remover = RFInterferenceRemover()

        start = time.time()
        processed, metrics = remover.process(audio, sr)
        elapsed = time.time() - start

        # Should process 10s audio in less than 5 seconds (< 0.5× realtime)
        assert elapsed < 5.0, f"Processing took {elapsed:.2f}s for 10s audio"

    def test_no_clipping_introduced(self):
        """Should not introduce clipping."""
        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Signal with some headroom
        clean = 0.7 * np.sin(2 * np.pi * 440 * t)
        interference = 0.05 * np.sin(2 * np.pi * 12000 * t)
        contaminated = clean + interference

        remover = RFInterferenceRemover()
        processed, metrics = remover.process(contaminated, sr)

        # Should not clip
        assert np.max(np.abs(processed)) <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
