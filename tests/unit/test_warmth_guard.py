from __future__ import annotations

"""Unit-Tests für §WBG (V25) warmth_guard.py.

Testet measure_warmth_band_delta() und WarmthBandResult.
"""


import numpy as np
import pytest

SR = 48000
_N = 48000  # 1 s


def _make_noise(n: int = _N, amp: float = 0.2, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(n)).astype(np.float32)


def _make_warmth_signal(n: int = _N) -> np.ndarray:
    """Signal mit starker 200–800 Hz Energie."""
    t = np.linspace(0.0, n / SR, n, endpoint=False)
    audio = (
        0.3 * np.sin(2 * np.pi * 300.0 * t) + 0.25 * np.sin(2 * np.pi * 500.0 * t) + 0.2 * np.sin(2 * np.pi * 700.0 * t)
    )
    return np.asarray(audio, dtype=np.float32)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWarmthGuardImport:
    def test_import_function(self):
        from backend.core.dsp.warmth_guard import measure_warmth_band_delta

        assert callable(measure_warmth_band_delta)

    def test_import_result_class(self):
        from backend.core.dsp.warmth_guard import WarmthBandResult

        assert WarmthBandResult is not None

    def test_import_threshold(self):
        from backend.core.dsp.warmth_guard import WARMTH_LOSS_THRESHOLD_DB

        assert pytest.approx(2.5) == WARMTH_LOSS_THRESHOLD_DB


# ---------------------------------------------------------------------------
# WarmthBandResult Dataclass
# ---------------------------------------------------------------------------


class TestWarmthBandResult:
    def test_fields_exist(self):
        from backend.core.dsp.warmth_guard import WarmthBandResult

        r = WarmthBandResult(
            loss_db=1.2,
            gain_db=0.0,
            ok=True,
            warmth_blend_factor=0.9,
        )
        assert r.loss_db == pytest.approx(1.2)
        assert r.gain_db == 0.0
        assert r.ok is True
        assert r.warmth_blend_factor == pytest.approx(0.9)

    def test_default_blend_factor(self):
        from backend.core.dsp.warmth_guard import WarmthBandResult

        r = WarmthBandResult(loss_db=0.0, gain_db=0.0, ok=True)
        assert r.warmth_blend_factor == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Identisches Signal → kein Verlust
# ---------------------------------------------------------------------------


class TestWarmthBandIdentical:
    def test_identical_no_loss(self):
        from backend.core.dsp.warmth_guard import measure_warmth_band_delta

        audio = _make_warmth_signal()
        result = measure_warmth_band_delta(audio, audio.copy(), SR)
        assert result.loss_db <= 0.1  # minimal numerisches Rauschen erlaubt

    def test_identical_ok_true(self):
        from backend.core.dsp.warmth_guard import measure_warmth_band_delta

        audio = _make_noise()
        result = measure_warmth_band_delta(audio, audio, SR)
        assert result.ok is True


# ---------------------------------------------------------------------------
# Signal ohne Wärmeband-Energie (HP-gefiltert) → Verlust
# ---------------------------------------------------------------------------


class TestWarmthBandLoss:
    def test_warmth_loss_detected(self):
        """Wenn alle Wärmeband-Energie entfernt wird → loss_db > 0."""
        from scipy.signal import butter, sosfiltfilt

        from backend.core.dsp.warmth_guard import measure_warmth_band_delta

        audio = _make_warmth_signal()
        # High-Pass bei 2 kHz → Wärmeband (200–800 Hz) komplett entfernt
        sos = butter(6, 2000.0 / (SR / 2.0), btype="high", output="sos")
        post = sosfiltfilt(sos, audio).astype(np.float32)
        result = measure_warmth_band_delta(audio, post, SR)
        assert result.loss_db > 2.0, f"Erwartete Wärmeverlust > 2 dB, erhalten: {result.loss_db}"

    def test_ok_false_on_large_loss(self):
        """Großer Verlust → ok=False."""
        from scipy.signal import butter, sosfiltfilt

        from backend.core.dsp.warmth_guard import measure_warmth_band_delta

        audio = _make_warmth_signal()
        sos = butter(8, 2000.0 / (SR / 2.0), btype="high", output="sos")
        post = sosfiltfilt(sos, audio).astype(np.float32)
        result = measure_warmth_band_delta(audio, post, SR)
        assert result.ok is False


# ---------------------------------------------------------------------------
# Blend-Faktor bei kumulativem Verlust
# ---------------------------------------------------------------------------


class TestWarmthBlendFactor:
    def test_cumulative_loss_reduces_blend(self):
        from backend.core.dsp.warmth_guard import WARMTH_LOSS_THRESHOLD_DB, measure_warmth_band_delta

        audio = _make_warmth_signal()
        # Kumulativer Verlust über Schwellwert → Blend-Faktor < 1.0
        result = measure_warmth_band_delta(
            audio,
            audio.copy(),
            SR,
            cumulative_loss_db=WARMTH_LOSS_THRESHOLD_DB + 1.0,
        )
        assert result.warmth_blend_factor < 1.0

    def test_zero_cumulative_loss_full_blend(self):
        from backend.core.dsp.warmth_guard import measure_warmth_band_delta

        audio = _make_warmth_signal()
        result = measure_warmth_band_delta(audio, audio.copy(), SR, cumulative_loss_db=0.0)
        assert result.warmth_blend_factor == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# sr-Guard
# ---------------------------------------------------------------------------


class TestWarmthSrGuard:
    def test_sr_assert_48000(self):
        from backend.core.dsp.warmth_guard import measure_warmth_band_delta

        audio = _make_noise(n=22050)
        with pytest.raises(AssertionError):
            measure_warmth_band_delta(audio, audio.copy(), 44100)


# ---------------------------------------------------------------------------
# Randfall: Stille
# ---------------------------------------------------------------------------


class TestWarmthSilence:
    def test_silence_no_crash(self):
        from backend.core.dsp.warmth_guard import measure_warmth_band_delta

        silence = np.zeros(_N, dtype=np.float32)
        result = measure_warmth_band_delta(silence, silence, SR)
        assert isinstance(result.ok, bool)
        assert result.loss_db >= -0.1  # numerisches Rauschen

    def test_stereo_ok(self):
        from backend.core.dsp.warmth_guard import measure_warmth_band_delta

        stereo = np.stack([_make_warmth_signal(), _make_noise()], axis=0)
        result = measure_warmth_band_delta(stereo, stereo.copy(), SR)
        assert isinstance(result.ok, bool)
