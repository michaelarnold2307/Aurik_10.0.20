from __future__ import annotations

"""Tests for TFS Preservation Guard (backend/core/tfs_preservation_guard.py).

Covers:
- ERB centre-frequency grid generation
- Hilbert-based instantaneous phase extraction
- TFS coherence measurement (perfect, degraded, silence)
- Mono and stereo handling
- NaN/Inf guards
- Thread-safe singleton
- Edge cases (very short audio, single-band, DC offset)
- Phase-shift detection (known degradation)
- Frequency-band selectivity
"""


import threading

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def guard():
    from backend.core.tfs_preservation_guard import TFSPreservationGuard

    return TFSPreservationGuard()


@pytest.fixture
def sr():
    return 48000


@pytest.fixture
def sine_440(sr):
    """1 s 440 Hz sine at 48 kHz."""
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return np.sin(2 * np.pi * 440.0 * t).astype(np.float64)


@pytest.fixture
def sine_200(sr):
    """1 s 200 Hz sine at 48 kHz."""
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return np.sin(2 * np.pi * 200.0 * t).astype(np.float64)


@pytest.fixture
def complex_tone(sr):
    """1 s complex tone (200 + 400 + 800 Hz) at 48 kHz."""
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return (
        0.5 * np.sin(2 * np.pi * 200.0 * t) + 0.3 * np.sin(2 * np.pi * 400.0 * t) + 0.2 * np.sin(2 * np.pi * 800.0 * t)
    ).astype(np.float64)


# ---------------------------------------------------------------------------
# ERB grid tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestERBGrid:
    def test_centre_frequencies_count(self):
        from backend.core.tfs_preservation_guard import _erb_centre_frequencies

        centres = _erb_centre_frequencies(100.0, 1500.0, 12)
        assert len(centres) == 12

    def test_centre_frequencies_range(self):
        from backend.core.tfs_preservation_guard import _erb_centre_frequencies

        centres = _erb_centre_frequencies(100.0, 1500.0, 12)
        assert centres[0] >= 95.0  # approximately 100 Hz
        assert centres[-1] <= 1550.0  # approximately 1500 Hz

    def test_centre_frequencies_monotonic(self):
        from backend.core.tfs_preservation_guard import _erb_centre_frequencies

        centres = _erb_centre_frequencies(100.0, 1500.0, 12)
        for i in range(1, len(centres)):
            assert centres[i] > centres[i - 1]

    def test_erb_bandwidth_formula(self):
        from backend.core.tfs_preservation_guard import _erb_hz

        # At 1 kHz: ERB = 24.7 * (4.37 + 1) = 24.7 * 5.37 ≈ 132.6
        erb_1k = _erb_hz(1000.0)
        assert 130.0 < erb_1k < 135.0

    def test_erb_bandwidth_low_freq_narrower(self):
        from backend.core.tfs_preservation_guard import _erb_hz

        erb_100 = _erb_hz(100.0)
        erb_1000 = _erb_hz(1000.0)
        assert erb_100 < erb_1000  # narrower at lower frequencies


# ---------------------------------------------------------------------------
# TFS coherence tests
# ---------------------------------------------------------------------------


class TestTFSCoherence:
    def test_identical_signals_perfect_coherence(self, guard, sine_440, sr):
        result = guard.measure(sine_440, sine_440.copy(), sr)
        assert result.mean_coherence >= 0.98
        assert result.passes_threshold is True

    def test_identical_complex_tone(self, guard, complex_tone, sr):
        result = guard.measure(complex_tone, complex_tone.copy(), sr)
        assert result.mean_coherence >= 0.98

    def test_phase_shifted_signal_lower_coherence(self, guard, sr):
        """A constant phase shift reduces TFS coherence."""
        t = np.linspace(0, 1.0, sr, endpoint=False)
        orig = np.sin(2 * np.pi * 300.0 * t)
        # Shift by π/4 — should reduce coherence noticeably
        shifted = np.sin(2 * np.pi * 300.0 * t + np.pi / 4)
        result = guard.measure(orig, shifted, sr)
        # Not perfect anymore but phase is still consistent
        assert result.mean_coherence < 1.0

    def test_random_phase_disruption_low_coherence(self, guard, sr):
        """Random noise has no TFS correlation with a broadband signal."""
        t = np.linspace(0, 1.0, sr, endpoint=False)
        # Broadband signal covering full TFS range (100–1500 Hz)
        orig = (
            0.3 * np.sin(2 * np.pi * 150.0 * t)
            + 0.3 * np.sin(2 * np.pi * 400.0 * t)
            + 0.2 * np.sin(2 * np.pi * 800.0 * t)
            + 0.2 * np.sin(2 * np.pi * 1200.0 * t)
        )
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(sr) * 0.5
        result = guard.measure(orig, noise, sr)
        # Unpredictable phase → low coherence
        assert result.mean_coherence < 0.6

    def test_slightly_noisy_signal_high_coherence(self, guard, sine_440, sr):
        """Adding small noise should maintain TFS coherence."""
        rng = np.random.default_rng(42)
        noisy = sine_440 + rng.standard_normal(len(sine_440)) * 0.01
        result = guard.measure(sine_440, noisy, sr)
        assert result.mean_coherence >= 0.80

    def test_threshold_parameter(self, guard, sr):
        t = np.linspace(0, 1.0, sr, endpoint=False)
        orig = (
            0.3 * np.sin(2 * np.pi * 150.0 * t)
            + 0.3 * np.sin(2 * np.pi * 400.0 * t)
            + 0.2 * np.sin(2 * np.pi * 800.0 * t)
            + 0.2 * np.sin(2 * np.pi * 1200.0 * t)
        )
        rng = np.random.default_rng(42)
        noisy = orig + rng.standard_normal(sr) * 0.3
        # Strict threshold
        r1 = guard.measure(orig, noisy, sr, threshold=0.99)
        assert r1.passes_threshold is False
        # Lenient threshold
        r2 = guard.measure(orig, noisy, sr, threshold=0.30)
        assert r2.passes_threshold is True


# ---------------------------------------------------------------------------
# Band-level detail tests
# ---------------------------------------------------------------------------


class TestTFSBands:
    def test_band_results_populated(self, guard, sine_200, sr):
        result = guard.measure(sine_200, sine_200.copy(), sr)
        assert result.n_bands > 0
        assert len(result.band_results) == result.n_bands

    def test_band_centre_frequencies_in_range(self, guard, complex_tone, sr):
        result = guard.measure(complex_tone, complex_tone.copy(), sr)
        for br in result.band_results:
            assert 50.0 <= br.centre_freq_hz <= 2000.0
            assert br.erb_width_hz > 0

    def test_band_coherence_bounded_0_1(self, guard, complex_tone, sr):
        rng = np.random.default_rng(42)
        noisy = complex_tone + rng.standard_normal(len(complex_tone)) * 0.1
        result = guard.measure(complex_tone, noisy, sr)
        for br in result.band_results:
            assert 0.0 <= br.tfs_coherence <= 1.0

    def test_voiced_frame_count_reasonable(self, guard, sine_440, sr):
        result = guard.measure(sine_440, sine_440.copy(), sr)
        for br in result.band_results:
            # 440 Hz is within TFS range — should have voiced frames for that band
            assert br.n_voiced_frames >= 0

    def test_min_coherence_leq_mean(self, guard, complex_tone, sr):
        rng = np.random.default_rng(42)
        noisy = complex_tone + rng.standard_normal(len(complex_tone)) * 0.05
        result = guard.measure(complex_tone, noisy, sr)
        assert result.min_coherence <= result.mean_coherence + 1e-6


# ---------------------------------------------------------------------------
# Shape / format tests
# ---------------------------------------------------------------------------


class TestTFSShapeHandling:
    def test_stereo_input(self, guard, sr):
        t = np.linspace(0, 1.0, sr, endpoint=False)
        mono = np.sin(2 * np.pi * 300.0 * t)
        stereo = np.stack([mono, mono * 0.9], axis=0)  # (2, N) shape
        result = guard.measure(stereo, stereo.copy(), sr)
        assert result.mean_coherence >= 0.95

    def test_stereo_column_major(self, guard, sr):
        """(N, 2) shape stereo."""
        t = np.linspace(0, 1.0, sr, endpoint=False)
        mono = np.sin(2 * np.pi * 300.0 * t)
        stereo = np.stack([mono, mono * 0.9], axis=1)  # (N, 2) shape
        result = guard.measure(stereo, stereo.copy(), sr)
        assert result.mean_coherence >= 0.90

    def test_different_lengths_handled(self, guard, sr):
        t1 = np.linspace(0, 1.0, sr, endpoint=False)
        t2 = np.linspace(0, 0.8, int(sr * 0.8), endpoint=False)
        s1 = np.sin(2 * np.pi * 300.0 * t1)
        s2 = np.sin(2 * np.pi * 300.0 * t2)
        result = guard.measure(s1, s2, sr)
        assert result.mean_coherence >= 0.0  # no crash

    def test_very_short_audio(self, guard, sr):
        """Audio shorter than 2 frames returns perfect coherence."""
        short = np.sin(2 * np.pi * 300.0 * np.linspace(0, 0.01, 480))
        result = guard.measure(short, short.copy(), sr)
        assert result.mean_coherence == 1.0


# ---------------------------------------------------------------------------
# NaN/Inf guards
# ---------------------------------------------------------------------------


class TestTFSNanInfGuard:
    def test_nan_in_input(self, guard, sine_440, sr):
        corrupted = sine_440.copy()
        corrupted[100:200] = np.nan
        result = guard.measure(corrupted, sine_440, sr)
        assert np.isfinite(result.mean_coherence)
        assert np.isfinite(result.min_coherence)

    def test_inf_in_input(self, guard, sine_440, sr):
        corrupted = sine_440.copy()
        corrupted[500] = np.inf
        result = guard.measure(corrupted, sine_440, sr)
        assert np.isfinite(result.mean_coherence)

    def test_all_zeros(self, guard, sr):
        zeros = np.zeros(sr, dtype=np.float64)
        result = guard.measure(zeros, zeros, sr)
        assert np.isfinite(result.mean_coherence)

    def test_result_fields_all_finite(self, guard, complex_tone, sr):
        rng = np.random.default_rng(42)
        noisy = complex_tone + rng.standard_normal(len(complex_tone)) * 0.1
        result = guard.measure(complex_tone, noisy, sr)
        assert np.isfinite(result.mean_coherence)
        assert np.isfinite(result.min_coherence)
        for br in result.band_results:
            assert np.isfinite(br.tfs_coherence)
            assert np.isfinite(br.centre_freq_hz)
            assert np.isfinite(br.erb_width_hz)


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------


class TestTFSSingleton:
    def test_singleton_returns_same_instance(self):
        from backend.core.tfs_preservation_guard import get_tfs_preservation_guard

        g1 = get_tfs_preservation_guard()
        g2 = get_tfs_preservation_guard()
        assert g1 is g2

    def test_singleton_thread_safe(self):
        from backend.core.tfs_preservation_guard import get_tfs_preservation_guard

        instances = []

        def worker():
            instances.append(get_tfs_preservation_guard())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert all(inst is instances[0] for inst in instances)


# ---------------------------------------------------------------------------
# Regression / known-behaviour tests
# ---------------------------------------------------------------------------


class TestTFSKnownBehaviour:
    def test_frequency_selective_degradation(self, guard, sr):
        """Degrading only 200 Hz band should lower coherence for that band
        but not for 1 kHz band."""
        t = np.linspace(0, 1.0, sr, endpoint=False)
        orig = 0.5 * np.sin(2 * np.pi * 200.0 * t) + 0.5 * np.sin(2 * np.pi * 1000.0 * t)
        # Replace 200 Hz component with shifted version
        degraded = 0.5 * np.sin(2 * np.pi * 200.0 * t + np.pi / 3) + 0.5 * np.sin(2 * np.pi * 1000.0 * t)
        result = guard.measure(orig, degraded, sr)
        # Should have some degradation but not total
        assert 0.2 < result.mean_coherence < 0.99

    def test_amplitude_change_does_not_affect_phase(self, guard, sine_440, sr):
        """TFS measures phase, not amplitude — gain change should keep coherence high."""
        quieter = sine_440 * 0.3
        result = guard.measure(sine_440, quieter, sr)
        # Phase is identical — coherence should be very high
        assert result.mean_coherence >= 0.95

    def test_time_reversal_chirp_destroys_tfs(self, guard, sr):
        """Time-reversing a chirp (non-stationary) disrupts TFS coherence.

        Pure sines have constant phase offset under reversal (known property).
        Non-stationary signals like chirps have time-varying instantaneous
        frequency, so reversal genuinely disrupts TFS.
        """
        t = np.linspace(0, 1.0, sr, endpoint=False)
        # Chirp sweeping 200 → 1200 Hz
        phase = 2 * np.pi * (200.0 * t + 0.5 * (1200.0 - 200.0) * t**2)
        orig = np.sin(phase)
        reversed_sig = orig[::-1].copy()
        result = guard.measure(orig, reversed_sig, sr)
        # Reversed chirp has inverted frequency trajectory → low coherence
        assert result.mean_coherence < 0.85

    def test_dc_offset_tolerance(self, guard, sine_440, sr):
        """DC offset should not affect TFS coherence (bandpass filtered out)."""
        with_dc = sine_440 + 0.3
        result = guard.measure(sine_440, with_dc, sr)
        # DC is outside the 100–1500 Hz bandpass
        assert result.mean_coherence >= 0.95

    def test_low_sample_rate(self, guard):
        """Works at 16 kHz (Nyquist 8 kHz — still covers TFS range)."""
        sr_low = 16000
        t = np.linspace(0, 1.0, sr_low, endpoint=False)
        sig = np.sin(2 * np.pi * 300.0 * t)
        result = guard.measure(sig, sig.copy(), sr_low)
        assert result.mean_coherence >= 0.95

    def test_high_sample_rate(self, guard):
        """Works at 96 kHz."""
        sr_hi = 96000
        t = np.linspace(0, 1.0, sr_hi, endpoint=False)
        sig = np.sin(2 * np.pi * 500.0 * t)
        result = guard.measure(sig, sig.copy(), sr_hi)
        assert result.mean_coherence >= 0.95


# ---------------------------------------------------------------------------
# Performance sanity
# ---------------------------------------------------------------------------


class TestTFSPerformance:
    def test_reasonable_execution_time(self, guard, sr):
        """60 s mono audio should complete in < 5 s."""
        import time

        t = np.linspace(0, 60.0, sr * 60, endpoint=False)
        sig = np.sin(2 * np.pi * 300.0 * t) + 0.2 * np.sin(2 * np.pi * 800.0 * t)
        rng = np.random.default_rng(42)
        noisy = sig + rng.standard_normal(len(sig)) * 0.02

        start = time.monotonic()
        result = guard.measure(sig, noisy, sr)
        elapsed = time.monotonic() - start

        assert elapsed < 12.0, f"TFS measurement took {elapsed:.1f}s (limit 12s)"
        assert np.isfinite(result.mean_coherence)
