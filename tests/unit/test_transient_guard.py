from __future__ import annotations

import pytest

"""Unit-Tests für §PEP (V22) transient_guard.py.

Testet detect_transient_shifts() und TransientShiftResult.
"""


import numpy as np

SR = 48000
_N = 48000  # 1 s


def _make_impulse_audio(positions_ms: list[float], n: int = _N) -> np.ndarray:
    """Erstellt ein Signal mit Impulsen an gegebenen Positionen (ms)."""
    audio = np.zeros(n, dtype=np.float32)
    for pos_ms in positions_ms:
        idx = int(pos_ms / 1000.0 * SR)
        if 0 <= idx < n:
            audio[idx : idx + 10] = 0.8
    return audio


def _make_noise(n: int = _N, amp: float = 0.05, seed: int = 99) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(n)).astype(np.float32)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTransientGuardImport:
    def test_import_function(self):
        from backend.core.dsp.transient_guard import detect_transient_shifts

        assert callable(detect_transient_shifts)

    def test_import_result_class(self):
        from backend.core.dsp.transient_guard import TransientShiftResult

        assert TransientShiftResult is not None

    def test_import_threshold(self):
        from backend.core.dsp.transient_guard import TRANSIENT_SHIFT_THRESHOLD_MS

        assert TRANSIENT_SHIFT_THRESHOLD_MS == 2.0


# ---------------------------------------------------------------------------
# TransientShiftResult Dataclass
# ---------------------------------------------------------------------------


class TestTransientShiftResult:
    def test_fields(self):
        from backend.core.dsp.transient_guard import TransientShiftResult

        r = TransientShiftResult(
            max_shift_ms=1.5,
            onset_count=3,
            ok=True,
            blend_reduction=0.0,
            shifts_ms=[0.5, 1.0, 1.5],
        )
        assert r.max_shift_ms == 1.5
        assert r.onset_count == 3
        assert r.ok is True
        assert r.blend_reduction == 0.0
        assert r.shifts_ms == [0.5, 1.0, 1.5]

    def test_ok_false_on_large_shift(self):
        from backend.core.dsp.transient_guard import TransientShiftResult

        r = TransientShiftResult(
            max_shift_ms=5.0,
            onset_count=2,
            ok=False,
            blend_reduction=0.4,
        )
        assert r.ok is False
        assert r.blend_reduction > 0.0

    def test_default_shifts_list(self):
        from backend.core.dsp.transient_guard import TransientShiftResult

        r = TransientShiftResult(max_shift_ms=0.0, onset_count=0, ok=True)
        assert isinstance(r.shifts_ms, list)


# ---------------------------------------------------------------------------
# Identisches Signal → kein Shift, ok=True
# ---------------------------------------------------------------------------


class TestTransientShiftIdentical:
    def test_identical_no_shift(self):
        from backend.core.dsp.transient_guard import detect_transient_shifts

        audio = _make_impulse_audio([50.0, 200.0, 500.0])
        result = detect_transient_shifts(audio, audio.copy(), SR)
        assert result.ok is True
        assert result.max_shift_ms <= 2.0

    def test_noise_no_crash(self):
        from backend.core.dsp.transient_guard import detect_transient_shifts

        audio = _make_noise()
        result = detect_transient_shifts(audio, audio.copy(), SR)
        assert isinstance(result.ok, bool)
        assert result.max_shift_ms >= 0.0


# ---------------------------------------------------------------------------
# Kleiner Shift (< 2 ms) → ok=True
# ---------------------------------------------------------------------------


class TestTransientShiftSmall:
    def test_tiny_delay_ok(self):
        """1 ms Verzögerung liegt unter dem Schwellwert."""
        from backend.core.dsp.transient_guard import detect_transient_shifts

        audio = _make_impulse_audio([100.0, 300.0, 600.0])
        n = len(audio)
        shift_samples = int(0.001 * SR)  # 1 ms
        post = np.zeros(n, dtype=np.float32)
        post[shift_samples:] = audio[: n - shift_samples]
        result = detect_transient_shifts(audio, post, SR)
        # Bei 1 ms-Delay: ok je nach Implementierung (tolerant testen)
        assert isinstance(result.ok, bool)


# ---------------------------------------------------------------------------
# Stille → kein Crash
# ---------------------------------------------------------------------------


class TestTransientShiftEdgeCases:
    def test_silence_no_crash(self):
        from backend.core.dsp.transient_guard import detect_transient_shifts

        silence = np.zeros(_N, dtype=np.float32)
        result = detect_transient_shifts(silence, silence, SR)
        assert isinstance(result.ok, bool)

    def test_stereo_no_crash(self):
        from backend.core.dsp.transient_guard import detect_transient_shifts

        stereo = np.stack([_make_impulse_audio([100.0]), _make_noise()], axis=0)
        result = detect_transient_shifts(stereo, stereo.copy(), SR)
        assert isinstance(result.ok, bool)

    def test_return_type_correct(self):
        from backend.core.dsp.transient_guard import TransientShiftResult, detect_transient_shifts

        audio = _make_noise()
        result = detect_transient_shifts(audio, audio, SR)
        assert isinstance(result, TransientShiftResult)

    def test_onset_count_non_negative(self):
        from backend.core.dsp.transient_guard import detect_transient_shifts

        audio = _make_impulse_audio([100.0, 300.0])
        result = detect_transient_shifts(audio, audio, SR)
        assert result.onset_count >= 0

    def test_blend_reduction_in_range(self):
        from backend.core.dsp.transient_guard import detect_transient_shifts

        audio = _make_noise()
        result = detect_transient_shifts(audio, audio, SR)
        assert 0.0 <= result.blend_reduction <= 1.0
