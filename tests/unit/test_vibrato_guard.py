from __future__ import annotations

"""Unit-Tests für §2.72 vibrato_guard.py.

Testet check_vibrato_depth_preservation() und VibratoDepthResult.
"""


import numpy as np
import pytest

SR = 48000
_N = 48000  # 1 s


def _make_vibrato_sine(
    carrier_hz: float = 440.0,
    vibrato_rate_hz: float = 5.5,
    vibrato_depth_hz: float = 8.0,
    n: int = _N,
) -> np.ndarray:
    """Generiert ein Vibrato-Signal: Träger mit FM-Modulation."""
    t = np.linspace(0.0, n / SR, n, endpoint=False)
    instantaneous_freq = carrier_hz + vibrato_depth_hz * np.sin(2.0 * np.pi * vibrato_rate_hz * t)
    phase = np.cumsum(2.0 * np.pi * instantaneous_freq / SR)
    return np.asarray(0.4 * np.sin(phase), dtype=np.float32)


def _make_sine(freq_hz: float = 440.0, n: int = _N) -> np.ndarray:
    t = np.linspace(0.0, n / SR, n, endpoint=False)
    return np.asarray(0.4 * np.sin(2.0 * np.pi * freq_hz * t), dtype=np.float32)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVibratoGuardImport:
    def test_import_function(self):
        from backend.core.dsp.vibrato_guard import check_vibrato_depth_preservation

        assert callable(check_vibrato_depth_preservation)

    def test_import_result_class(self):
        from backend.core.dsp.vibrato_guard import VibratoDepthResult

        assert VibratoDepthResult is not None

    def test_import_threshold(self):
        from backend.core.dsp.vibrato_guard import VIBRATO_MAX_REDUCTION_PCT

        assert VIBRATO_MAX_REDUCTION_PCT == 10.0


# ---------------------------------------------------------------------------
# VibratoDepthResult Dataclass
# ---------------------------------------------------------------------------


class TestVibratoDepthResult:
    def test_fields(self):
        from backend.core.dsp.vibrato_guard import VibratoDepthResult

        r = VibratoDepthResult(
            depth_pre_hz=8.0,
            depth_post_hz=7.5,
            depth_reduction_pct=6.25,
            ok=True,
        )
        assert r.depth_pre_hz == 8.0
        assert r.depth_post_hz == 7.5
        assert r.depth_reduction_pct == pytest.approx(6.25)
        assert r.ok is True

    def test_ok_false_when_reduced(self):
        from backend.core.dsp.vibrato_guard import VibratoDepthResult

        r = VibratoDepthResult(
            depth_pre_hz=8.0,
            depth_post_hz=5.0,
            depth_reduction_pct=37.5,
            ok=False,
        )
        assert r.ok is False


# ---------------------------------------------------------------------------
# Identisches Signal → ok=True
# ---------------------------------------------------------------------------


class TestVibratoDepthIdentical:
    def test_identical_vibrato_ok(self):
        from backend.core.dsp.vibrato_guard import check_vibrato_depth_preservation

        audio = _make_vibrato_sine()
        result = check_vibrato_depth_preservation(audio, audio.copy(), SR)
        assert result.ok is True

    def test_depth_reduction_zero_for_identical(self):
        from backend.core.dsp.vibrato_guard import check_vibrato_depth_preservation

        audio = _make_vibrato_sine()
        result = check_vibrato_depth_preservation(audio, audio.copy(), SR)
        assert result.depth_reduction_pct < 10.0


# ---------------------------------------------------------------------------
# Gain-skaliertes Signal → reduction prüfen
# ---------------------------------------------------------------------------


class TestVibratoDepthGainScaled:
    def test_gain_095_small_reduction(self):
        from backend.core.dsp.vibrato_guard import check_vibrato_depth_preservation

        audio = _make_vibrato_sine()
        post = (audio * 0.95).astype(np.float32)
        result = check_vibrato_depth_preservation(audio, post, SR)
        # Kleiner Gain-Unterschied: Tiefenreduktion minimal
        assert result.depth_reduction_pct < 15.0  # tolerant für DSP-Proxy

    def test_strong_amplitude_compression_may_reduce_depth(self):
        """Starke Kompression kann Vibrato-Tiefe messbar reduzieren."""
        from backend.core.dsp.vibrato_guard import check_vibrato_depth_preservation

        audio = _make_vibrato_sine(vibrato_depth_hz=12.0)
        # Hard-Limiting auf 0.2 → quasi kein Vibrato mehr in Amplitude
        post = np.clip(audio * 3.0, -0.2, 0.2).astype(np.float32)
        result = check_vibrato_depth_preservation(audio, post, SR)
        # Ergebnis kann ok oder not-ok sein; wichtig: kein Crash
        assert isinstance(result.ok, bool)
        assert result.depth_reduction_pct >= 0.0


# ---------------------------------------------------------------------------
# Stille / Randfall
# ---------------------------------------------------------------------------


class TestVibratoDepthEdgeCases:
    def test_silence_no_crash(self):
        from backend.core.dsp.vibrato_guard import check_vibrato_depth_preservation

        silence = np.zeros(_N, dtype=np.float32)
        result = check_vibrato_depth_preservation(silence, silence, SR)
        assert isinstance(result.ok, bool)
        assert result.depth_reduction_pct >= 0.0

    def test_pure_sine_no_vibrato_ok(self):
        """Reine Sinustöne haben keinen Vibrato → ok=True (kein Verlust)."""
        from backend.core.dsp.vibrato_guard import check_vibrato_depth_preservation

        audio = _make_sine(440.0)
        result = check_vibrato_depth_preservation(audio, audio.copy(), SR)
        assert result.ok is True

    def test_return_type_correct(self):
        from backend.core.dsp.vibrato_guard import VibratoDepthResult, check_vibrato_depth_preservation

        audio = _make_vibrato_sine()
        result = check_vibrato_depth_preservation(audio, audio, SR)
        assert isinstance(result, VibratoDepthResult)

    def test_stereo_no_crash(self):
        from backend.core.dsp.vibrato_guard import check_vibrato_depth_preservation

        stereo = np.stack([_make_vibrato_sine(), _make_vibrato_sine(carrier_hz=880.0)], axis=0)
        result = check_vibrato_depth_preservation(stereo, stereo.copy(), SR)
        assert isinstance(result.ok, bool)
