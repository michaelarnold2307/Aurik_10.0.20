import pytest

pytest.importorskip("mutagen")  # CI-Minimal-Umgebung (cross-platform)

"""Unit-tests for backend.exporter — POW-r Type 3 / TPDF dithering.

Spec §DSP-Spezialregeln:
  Dithering Export: POW-r Typ 3 (primär) → TPDF (fallback).
  VERBOTEN: Truncation ohne Dithering.

Tests cover:
  - Shape preservation (mono / stereo)
  - Output stays within [-1.0, 1.0]
  - Dither is actually applied (signal changes)
  - No NaN / Inf in output
  - 32-bit is a no-op
  - TPDF fallback path
  - POW-r primary path (scipy present)
  - Amplitude of dither ≈ 1 LSB (not catastrophically larger)
"""

from typing import cast

import numpy as np

from backend.exporter import (
    _POWR3_COEFFS,
    _apply_powr3_dither,
    _apply_tpdf_dither,
    _export_nuance_guard,
    apply_dither,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine(n: int = 4800, amp: float = 0.5, sr: int = 48000) -> np.ndarray:
    """Mono sine at 1 kHz, float32."""
    t = np.arange(n) / sr
    out = np.asarray((amp * np.sin(2 * np.pi * 1000 * t)), dtype=np.float32)
    return cast(np.ndarray, out)


def _stereo(n: int = 4800) -> np.ndarray:
    """Stereo float32 signal."""
    ch1 = _sine(n, 0.4)
    ch2 = _sine(n, 0.3)
    return np.stack([ch1, ch2], axis=1)


# ---------------------------------------------------------------------------
# _POWR3_COEFFS sanity
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPowR3Coefficients:
    def test_shape(self):
        assert _POWR3_COEFFS.ndim == 1
        assert len(_POWR3_COEFFS) == 9

    def test_dtype(self):
        assert _POWR3_COEFFS.dtype == np.float64

    def test_first_coeff_positive(self):
        # POW-r Type 3 first coefficient is ~2.4 (high-frequency emphasis)
        assert _POWR3_COEFFS[0] > 2.0

    def test_coeffs_not_all_identical(self):
        assert len(set(_POWR3_COEFFS.tolist())) > 1


# ---------------------------------------------------------------------------
# _apply_powr3_dither
# ---------------------------------------------------------------------------


class TestApplyPowR3Dither:
    def test_mono_shape(self):
        a = _sine()
        out = _apply_powr3_dither(a, 16)
        assert out.shape == a.shape

    def test_stereo_shape(self):
        a = _stereo()
        out = _apply_powr3_dither(a, 24)
        assert out.shape == a.shape

    def test_no_nan_mono(self):
        out = _apply_powr3_dither(_sine(), 16)
        assert np.isfinite(out).all()

    def test_no_nan_stereo(self):
        out = _apply_powr3_dither(_stereo(), 24)
        assert np.isfinite(out).all()

    def test_clipped_to_unity(self):
        out = _apply_powr3_dither(_sine(), 16)
        assert np.max(np.abs(out)) <= 1.0

    def test_dither_applied_mono(self):
        a = _sine()
        out = _apply_powr3_dither(a, 16)
        assert not np.array_equal(out, a)

    def test_dither_applied_stereo(self):
        a = _stereo()
        out = _apply_powr3_dither(a, 24)
        assert not np.array_equal(out, a)

    def test_noop_for_32bit(self):
        a = _sine()
        out = _apply_powr3_dither(a, 32)
        np.testing.assert_array_equal(out, a)

    def test_dither_amplitude_reasonable_16bit(self):
        # POW-r Type 3 noise shaping can amplify TPDF peaks up to ~15–20× LSB
        # (sum of |coeffs| ≈ 17.8 → worst-case headroom).  Anything below 32× is
        # still well within audibility threshold and does not constitute clipping.
        a = np.zeros(48000, dtype=np.float32)
        out = _apply_powr3_dither(a, 16)
        lsb_16 = 2.0 / (2**16)
        assert np.max(np.abs(out)) < lsb_16 * 32  # shaped dither — noise-shaped headroom

    def test_dtype_float32(self):
        out = _apply_powr3_dither(_sine(), 16)
        assert out.dtype == np.float32

    def test_silent_signal_no_clip(self):
        a = np.zeros(2048, dtype=np.float32)
        out = _apply_powr3_dither(a, 16)
        assert np.max(np.abs(out)) <= 1.0

    def test_near_full_scale_no_clip(self):
        a = np.full(2048, 0.9999, dtype=np.float32)
        out = _apply_powr3_dither(a, 16)
        assert np.max(np.abs(out)) <= 1.0


# ---------------------------------------------------------------------------
# _apply_tpdf_dither
# ---------------------------------------------------------------------------


class TestApplyTpdfDither:
    def test_mono_shape(self):
        a = _sine()
        out = _apply_tpdf_dither(a, 16)
        assert out.shape == a.shape

    def test_stereo_shape(self):
        a = _stereo()
        out = _apply_tpdf_dither(a, 16)
        assert out.shape == a.shape

    def test_no_nan(self):
        out = _apply_tpdf_dither(_sine(), 16)
        assert np.isfinite(out).all()

    def test_clipped_to_unity(self):
        out = _apply_tpdf_dither(_sine(), 16)
        assert np.max(np.abs(out)) <= 1.0

    def test_dither_applied(self):
        a = _sine()
        out = _apply_tpdf_dither(a, 16)
        assert not np.array_equal(out, a)

    def test_noop_for_32bit(self):
        a = _sine()
        out = _apply_tpdf_dither(a, 32)
        np.testing.assert_array_equal(out, a)

    def test_dtype_float32(self):
        out = _apply_tpdf_dither(_sine(), 16)
        assert out.dtype == np.float32

    def test_amplitude_bounded_24bit(self):
        a = np.zeros(4800, dtype=np.float32)
        out = _apply_tpdf_dither(a, 24)
        lsb_24 = 2.0 / (2**24)
        assert np.max(np.abs(out)) <= lsb_24 * 2 + 1e-9


# ---------------------------------------------------------------------------
# apply_dither (public API — primary + fallback routing)
# ---------------------------------------------------------------------------


class TestApplyDither:
    def test_mono_shape(self):
        a = _sine()
        out = apply_dither(a, 16)
        assert out.shape == a.shape

    def test_stereo_shape(self):
        a = _stereo()
        out = apply_dither(a, 24)
        assert out.shape == a.shape

    def test_no_nan_mono(self):
        out = apply_dither(_sine(), 16)
        assert np.isfinite(out).all()

    def test_no_nan_stereo(self):
        out = apply_dither(_stereo(), 16)
        assert np.isfinite(out).all()

    def test_clipped_to_unity(self):
        out = apply_dither(_sine(amp=0.99), 16)
        assert np.max(np.abs(out)) <= 1.0

    def test_dither_changes_signal(self):
        a = _sine()
        out = apply_dither(a, 16)
        assert not np.array_equal(out, a)

    def test_noop_for_32bit(self):
        a = _sine()
        out = apply_dither(a, 32)
        np.testing.assert_array_equal(out, a)

    def test_noop_for_33bit(self):
        a = _sine()
        out = apply_dither(a, 33)
        np.testing.assert_array_equal(out, a)

    def test_dtype_float32(self):
        out = apply_dither(_sine(), 16)
        assert out.dtype == np.float32

    def test_two_calls_not_identical(self):
        # Dithering is stochastic — two independent calls should differ
        a = _sine()
        out1 = apply_dither(a, 16)
        out2 = apply_dither(a, 16)
        assert not np.array_equal(out1, out2)

    def test_fallback_when_scipy_absent(self, monkeypatch):
        """When scipy is unavailable, route through TPDF fallback."""
        import backend.exporter as _mod

        monkeypatch.setattr(_mod, "_SCIPY_AVAILABLE", False)
        a = _sine()
        out = _mod.apply_dither(a, 16)
        assert out.shape == a.shape
        assert np.isfinite(out).all()
        assert np.max(np.abs(out)) <= 1.0
        assert not np.array_equal(out, a)

    def test_primary_uses_powr3_when_scipy_present(self, monkeypatch):
        """When scipy IS available, _apply_powr3_dither must be called."""
        import backend.exporter as _mod

        called = []

        def _fake_powr3(audio, bit_depth, **kwargs):
            called.append(bit_depth)
            return audio

        monkeypatch.setattr(_mod, "_SCIPY_AVAILABLE", True)
        monkeypatch.setattr(_mod, "_apply_powr3_dither", _fake_powr3)
        _mod.apply_dither(_sine(), 16)
        assert called == [16]

    def test_16bit_and_24bit_differ(self):
        """16-bit dither has larger LSB step than 24-bit."""
        np.random.seed(0)
        a = _sine()
        lsb_16 = 2.0 / (2**16)
        lsb_24 = 2.0 / (2**24)
        assert lsb_16 > lsb_24
        diff16 = np.mean(np.abs(apply_dither(a, 16) - a))
        diff24 = np.mean(np.abs(apply_dither(a, 24) - a))
        # 16-bit dither must add more noise energy than 24-bit
        assert diff16 > diff24 * 10


class TestExportNuanceGuard:
    def test_balance_guard_never_boosts_quieter_channel(self):
        left = _sine(n=48000, amp=0.40)
        right = _sine(n=48000, amp=0.08)
        stereo = np.stack([left, right], axis=1).astype(np.float32)
        stereo_in = stereo.copy()

        out = _export_nuance_guard(stereo, 48000)

        centre = slice(1024, -1024)
        in_left_rms = float(np.sqrt(np.mean(stereo_in[centre, 0].astype(np.float64) ** 2) + 1e-12))
        in_right_rms = float(np.sqrt(np.mean(stereo_in[centre, 1].astype(np.float64) ** 2) + 1e-12))
        out_left_rms = float(np.sqrt(np.mean(out[centre, 0].astype(np.float64) ** 2) + 1e-12))
        out_right_rms = float(np.sqrt(np.mean(out[centre, 1].astype(np.float64) ** 2) + 1e-12))

        assert out_left_rms < in_left_rms
        assert out_right_rms <= in_right_rms * 1.001


class TestDK364DitherKaskade:
    """D-K3-64 (geschlossen 2026-11-07): Der EINE Dither ist der Exporter.

    Ein Restore-seitiger TPDF-Vordither (``apply_final_polish``) erzeugte
    zusammen mit dem Export-``apply_dither`` eine Dither-Kaskade
    (doppelte Quantisierung, §V5 (copilot-instructions.md), §IV).
    """

    def test_final_polish_default_dither_aus(self):
        import inspect

        from backend.core.dsp.final_polish import apply_final_polish

        sig = inspect.signature(apply_final_polish)
        assert sig.parameters["dither"].default is False

    def test_unified_restorer_uebergibt_dither_false(self):
        import pathlib

        src = pathlib.Path("backend/core/unified_restorer_v3.py").read_text(encoding="utf-8")
        assert "dither=False,  # §D-K3-64: der EINE Dither ist der Exporter" in src
        assert src.count("apply_final_polish(") + src.count("_afp_song(") >= 2
        assert src.count("dither=False,  # §D-K3-64") == 2  # beide Call-Sites gepinnt

    def test_exporter_bleibt_der_eine_dither(self):
        import pathlib

        src = pathlib.Path("backend/exporter.py").read_text(encoding="utf-8")
        # Genau EINE Dither-Aufrufstelle im Export-Quantisierungspfad.
        assert src.count("apply_dither(audio") == 1
        assert "cd_active=True" in src  # §G4 (GEBOTE.md)-CD-Rauschprofil-Kopplung bleibt
