from __future__ import annotations

"""Tests for ERB Auditory Masking Model (backend/core/erb_auditory_masking.py).

Covers:
- ERB bandwidth formula (Glasberg & Moore 1990)
- ERB-rate scale conversion (Hz ↔ Cams)
- Spreading function asymmetry (upward > downward masking)
- Temporal masking decay (forward / backward, 3:1 ratio)
- Frequency-dependent masking thresholds
- Tonality-based informational masking
- Salience estimation (masked vs exposed defects)
- NaN/Inf guards
- Thread-safe singleton
- Edge cases (silence, very short, DC-only)
- Mono and stereo input handling
"""


import threading

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model():
    from backend.core.erb_auditory_masking import ERBAuditoryMaskingModel

    return ERBAuditoryMaskingModel()


@pytest.fixture
def sr():
    return 48000


@pytest.fixture
def loud_sine(sr):
    """2 s loud 1 kHz sine (amplitude 0.8)."""
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    return (0.8 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float64)


@pytest.fixture
def quiet_click_in_loud(sr):
    """2 s loud sine with a tiny click at t=1.0 s (masked scenario)."""
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    audio = 0.8 * np.sin(2 * np.pi * 1000.0 * t)
    # Tiny click at 1.0 s
    click_pos = sr  # sample index for t=1.0 s
    audio[click_pos : click_pos + 10] += 0.01
    return audio.astype(np.float64)


@pytest.fixture
def silence_with_click(sr):
    """2 s silence with a click at t=1.0 s (exposed scenario)."""
    audio = np.zeros(sr * 2, dtype=np.float64)
    click_pos = sr
    audio[click_pos : click_pos + 10] = 0.3
    return audio


# ---------------------------------------------------------------------------
# ERB formula tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestERBFormulas:
    def test_erb_hz_at_1khz(self):
        from backend.core.erb_auditory_masking import erb_hz

        # ERB(1000) = 24.7 * (4.37 + 1) = 24.7 * 5.37 ≈ 132.6
        result = erb_hz(1000.0)
        assert abs(result - 132.6) < 1.0

    def test_erb_hz_increases_with_frequency(self):
        from backend.core.erb_auditory_masking import erb_hz

        assert erb_hz(100.0) < erb_hz(500.0) < erb_hz(2000.0) < erb_hz(8000.0)

    def test_erb_hz_at_zero(self):
        from backend.core.erb_auditory_masking import erb_hz

        # ERB(0) = 24.7 * 1 = 24.7
        assert abs(erb_hz(0.0) - 24.7) < 0.01

    def test_erb_rate_monotonic(self):
        from backend.core.erb_auditory_masking import erb_rate

        freqs = [100, 200, 500, 1000, 2000, 4000, 8000]
        rates = [erb_rate(f) for f in freqs]
        for i in range(1, len(rates)):
            assert rates[i] > rates[i - 1]

    def test_erb_rate_roundtrip(self):
        from backend.core.erb_auditory_masking import erb_rate, erb_rate_to_hz

        for f in [100.0, 500.0, 1000.0, 4000.0, 10000.0]:
            n = erb_rate(f)
            f_back = erb_rate_to_hz(n)
            assert abs(f_back - f) < 0.01, f"Roundtrip failed: {f} → {n} → {f_back}"


# ---------------------------------------------------------------------------
# Spreading function tests
# ---------------------------------------------------------------------------


class TestSpreadingFunction:
    def test_same_band_full_masking(self):
        from backend.core.erb_auditory_masking import _spreading_function_db

        assert _spreading_function_db(1000.0, 1000.0) == 0.0

    def test_upward_masking_stronger(self):
        """Masking from low to high should be weaker attenuation (more masking)
        than from high to low — upward masking is asymmetric."""
        from backend.core.erb_auditory_masking import _spreading_function_db

        # Masker at 500 Hz, signal at 1000 Hz (upward — lower skirt -10 dB/ERB)
        upward = _spreading_function_db(500.0, 1000.0)
        # Masker at 1000 Hz, signal at 500 Hz (downward — upper skirt -24 dB/ERB)
        downward = _spreading_function_db(1000.0, 500.0)
        # Upward masking should be stronger (less negative = more masking)
        assert upward > downward

    def test_distant_bands_weak_masking(self):
        from backend.core.erb_auditory_masking import _spreading_function_db

        # Very distant bands — nearly no masking
        spread = _spreading_function_db(200.0, 8000.0)
        assert spread < -50.0  # very weak


# ---------------------------------------------------------------------------
# Temporal masking tests
# ---------------------------------------------------------------------------


class TestTemporalMasking:
    def test_forward_decay_at_zero(self):
        from backend.core.erb_auditory_masking import _forward_masking_decay_db

        assert _forward_masking_decay_db(0.0) == 0.0

    def test_forward_decay_increases_with_time(self):
        from backend.core.erb_auditory_masking import _forward_masking_decay_db

        d10 = _forward_masking_decay_db(10.0)
        d50 = _forward_masking_decay_db(50.0)
        d100 = _forward_masking_decay_db(100.0)
        assert d10 > d50 > d100  # less negative = less decay

    def test_forward_decay_beyond_200ms(self):
        from backend.core.erb_auditory_masking import _forward_masking_decay_db

        assert _forward_masking_decay_db(201.0) == -100.0

    def test_backward_decay_at_zero(self):
        from backend.core.erb_auditory_masking import _backward_masking_decay_db

        assert _backward_masking_decay_db(0.0) == 0.0

    def test_backward_shorter_than_forward(self):
        """Backward masking has shorter effective range (20 ms vs 200 ms)."""
        from backend.core.erb_auditory_masking import (
            _backward_masking_decay_db,
            _forward_masking_decay_db,
        )

        # At 15 ms: backward still active, forward very strong
        fwd_15 = _forward_masking_decay_db(15.0)
        bwd_15 = _backward_masking_decay_db(15.0)
        # Both active, but backward decays faster
        assert bwd_15 < fwd_15

    def test_backward_beyond_20ms(self):
        from backend.core.erb_auditory_masking import _backward_masking_decay_db

        assert _backward_masking_decay_db(21.0) == -100.0


# ---------------------------------------------------------------------------
# Masking threshold computation
# ---------------------------------------------------------------------------


class TestMaskingThresholds:
    def test_loud_context_high_threshold(self, model, loud_sine, sr):
        """Defect during loud sine should have high masking threshold → low salience."""
        result = model.compute_masking_threshold(
            loud_sine,
            sr,
            defect_start_s=0.8,
            defect_end_s=0.82,
        )
        assert result.salience < 0.8

    def test_silent_context_low_threshold(self, model, silence_with_click, sr):
        """Defect in silence should have low masking threshold → high salience."""
        result = model.compute_masking_threshold(
            silence_with_click,
            sr,
            defect_start_s=0.95,
            defect_end_s=1.05,
        )
        assert result.salience > 0.3

    def test_band_thresholds_populated(self, model, loud_sine, sr):
        result = model.compute_masking_threshold(
            loud_sine,
            sr,
            defect_start_s=0.5,
            defect_end_s=0.52,
        )
        assert len(result.band_thresholds) > 0
        for bt in result.band_thresholds:
            assert bt.centre_freq_hz > 0
            assert bt.erb_width_hz > 0
            assert np.isfinite(bt.threshold_db)

    def test_frequency_range_filter(self, model, loud_sine, sr):
        """Providing defect_freq_range should limit bands evaluated."""
        result_all = model.compute_masking_threshold(
            loud_sine,
            sr,
            0.5,
            0.52,
        )
        result_narrow = model.compute_masking_threshold(
            loud_sine,
            sr,
            0.5,
            0.52,
            defect_freq_range=(200.0, 400.0),
        )
        assert len(result_narrow.band_thresholds) <= len(result_all.band_thresholds)

    def test_dominant_masking_type(self, model, loud_sine, sr):
        result = model.compute_masking_threshold(
            loud_sine,
            sr,
            0.5,
            0.52,
        )
        assert result.dominant_masking_type in {
            "simultaneous",
            "temporal_forward",
            "temporal_backward",
            "combined",
            "none",
        }


# ---------------------------------------------------------------------------
# Tonality / informational masking
# ---------------------------------------------------------------------------


class TestTonalityMasking:
    def test_tonal_signal_higher_masking(self, model, sr):
        """Pure tone should produce informational masking bonus."""
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        tonal = 0.8 * np.sin(2 * np.pi * 500.0 * t)
        result = model.compute_masking_threshold(
            tonal,
            sr,
            0.8,
            0.82,
        )
        # Should have some informational bonus
        bonuses = [bt.informational_bonus_db for bt in result.band_thresholds]
        assert any(b > 0 for b in bonuses)

    def test_noise_signal_no_bonus(self, model, sr):
        """White noise should NOT produce informational masking bonus."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(sr * 2) * 0.3
        result = model.compute_masking_threshold(
            noise,
            sr,
            0.8,
            0.82,
        )
        bonuses = [bt.informational_bonus_db for bt in result.band_thresholds]
        # Noise has low tonality → minimal bonus
        avg_bonus = np.mean(bonuses) if bonuses else 0.0
        assert avg_bonus < 1.5  # noise shouldn't get large bonus


# ---------------------------------------------------------------------------
# Convenience salience API
# ---------------------------------------------------------------------------


class TestSalienceConvenience:
    def test_estimate_salience_returns_float(self, model, loud_sine, sr):
        sal = model.estimate_salience(loud_sine, sr, 0.5, 0.52)
        assert isinstance(sal, float)
        assert 0.0 <= sal <= 1.0

    def test_salience_bounded(self, model, sr):
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(sr * 2) * 0.3
        sal = model.estimate_salience(audio, sr, 0.5, 0.6)
        assert 0.0 <= sal <= 1.0


# ---------------------------------------------------------------------------
# Shape / format tests
# ---------------------------------------------------------------------------


class TestERBShapeHandling:
    def test_stereo_input(self, model, sr):
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        mono = 0.5 * np.sin(2 * np.pi * 800.0 * t)
        stereo = np.stack([mono, mono * 0.9], axis=0)
        result = model.compute_masking_threshold(
            stereo,
            sr,
            0.5,
            0.52,
        )
        assert len(result.band_thresholds) > 0

    def test_stereo_column_major(self, model, sr):
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        mono = 0.5 * np.sin(2 * np.pi * 800.0 * t)
        stereo = np.stack([mono, mono * 0.9], axis=1)
        result = model.compute_masking_threshold(
            stereo,
            sr,
            0.5,
            0.52,
        )
        assert len(result.band_thresholds) > 0


# ---------------------------------------------------------------------------
# NaN/Inf guards
# ---------------------------------------------------------------------------


class TestERBNanInfGuard:
    def test_nan_in_audio(self, model, sr):
        audio = np.zeros(sr * 2, dtype=np.float64)
        audio[1000:1100] = np.nan
        result = model.compute_masking_threshold(audio, sr, 0.5, 0.52)
        assert np.isfinite(result.salience)
        assert np.isfinite(result.mean_threshold_db)

    def test_inf_in_audio(self, model, sr):
        audio = np.zeros(sr * 2, dtype=np.float64)
        audio[5000] = np.inf
        result = model.compute_masking_threshold(audio, sr, 0.5, 0.52)
        assert np.isfinite(result.salience)

    def test_all_zeros(self, model, sr):
        zeros = np.zeros(sr * 2, dtype=np.float64)
        result = model.compute_masking_threshold(zeros, sr, 0.5, 0.52)
        assert np.isfinite(result.salience)

    def test_all_band_thresholds_finite(self, model, loud_sine, sr):
        result = model.compute_masking_threshold(loud_sine, sr, 0.5, 0.52)
        for bt in result.band_thresholds:
            assert np.isfinite(bt.threshold_db)
            assert np.isfinite(bt.informational_bonus_db)
            assert np.isfinite(bt.centre_freq_hz)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestERBEdgeCases:
    def test_defect_at_start(self, model, loud_sine, sr):
        result = model.compute_masking_threshold(loud_sine, sr, 0.0, 0.02)
        assert np.isfinite(result.salience)

    def test_defect_at_end(self, model, loud_sine, sr):
        result = model.compute_masking_threshold(loud_sine, sr, 1.98, 2.0)
        assert np.isfinite(result.salience)

    def test_very_short_defect(self, model, loud_sine, sr):
        result = model.compute_masking_threshold(loud_sine, sr, 1.0, 1.001)
        assert np.isfinite(result.salience)

    def test_defect_longer_than_context(self, model, loud_sine, sr):
        result = model.compute_masking_threshold(loud_sine, sr, 0.1, 1.9)
        assert np.isfinite(result.salience)

    def test_short_audio(self, model, sr):
        """256 samples — shorter than FFT window."""
        audio = np.sin(2 * np.pi * 500.0 * np.linspace(0, 0.005, 256))
        result = model.compute_masking_threshold(audio, sr, 0.001, 0.004)
        assert np.isfinite(result.salience)

    def test_low_sample_rate(self, model):
        """8 kHz sample rate — limited Nyquist."""
        sr_low = 8000
        t = np.linspace(0, 2.0, sr_low * 2, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 500.0 * t)
        result = model.compute_masking_threshold(audio, sr_low, 0.5, 0.52)
        assert np.isfinite(result.salience)
        assert len(result.band_thresholds) > 0


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------


class TestERBSingleton:
    def test_singleton_same_instance(self):
        from backend.core.erb_auditory_masking import get_erb_auditory_masking_model

        m1 = get_erb_auditory_masking_model()
        m2 = get_erb_auditory_masking_model()
        assert m1 is m2

    def test_singleton_thread_safe(self):
        from backend.core.erb_auditory_masking import get_erb_auditory_masking_model

        instances = []

        def worker():
            instances.append(get_erb_auditory_masking_model())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert all(inst is instances[0] for inst in instances)


# ---------------------------------------------------------------------------
# Integration compatibility
# ---------------------------------------------------------------------------


class TestERBIntegration:
    def test_result_can_replace_fixed_salience(self, model, loud_sine, sr):
        """ERBMaskingResult.salience can replace the fixed-threshold salience
        used in PerceptualSalienceEstimator."""
        result = model.compute_masking_threshold(loud_sine, sr, 0.5, 0.52)
        # Salience is in [0, 1] — compatible with existing severity scaling:
        # adjusted_severity = severity * (0.3 + 0.7 * salience)
        adjusted = 0.8 * (0.3 + 0.7 * result.salience)
        assert 0.0 <= adjusted <= 1.0
