"""Unit-Tests für den report-only Musik-VAD-Gate-Offset-Orakel (KP-4, kein Training).

Vertrag (§III.11 (copilot-instructions.md), Hörordnung §8, W-11): das Orakel
übt NIE ein Veto aus, flippt keine Flags und ändert das Gate-Verhalten nicht —
es berechnet und protokolliert nur einen musiknativen Offset aus der
PANNs-Singing-Evidenz. Die Tests prüfen Determinismus (§G5 (copilot-instructions.md)),
Fallback-Pfade (§V6 (copilot-instructions.md)),
Offset-Skalierung (0…−15 dB) und Stereo-Layout-Handling (§11 Spec 02) — ohne
`try/except: pass`-Hüllen (Lehre D-K3-20).
"""

from __future__ import annotations

import numpy as np
import pytest

import backend.core.dsp.music_vad_oracle as mvo


def _sine(n: int = 4000, freq: float = 440.0, sr: int = 44100, seed: int = 42) -> np.ndarray:
    t = np.linspace(0, n / sr, n, endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ── Offset-Skalierung ──────────────────────────────────────────────────────


def test_offset_zero_for_zero_singing() -> None:
    w = mvo.compute_music_vad_offset_db(0.0)
    assert w.applied is False
    assert w.panns_singing == 0.0
    assert w.offset_db == 0.0


def test_offset_full_band_for_full_singing() -> None:
    w = mvo.compute_music_vad_offset_db(1.0)
    assert w.applied is False
    assert w.offset_db == pytest.approx(-15.0, abs=1e-4)


def test_offset_mid_scales_linearly() -> None:
    w = mvo.compute_music_vad_offset_db(0.5)
    assert w.offset_db == pytest.approx(-7.5, abs=1e-4)


def test_offset_negative_semantics_lower_threshold() -> None:
    """Offset ist negativ → senkt Gate-Schwelle → Gate hält länger offen."""
    w_hi = mvo.compute_music_vad_offset_db(0.9)
    w_lo = mvo.compute_music_vad_offset_db(0.1)
    assert w_hi.offset_db < w_lo.offset_db


def test_offset_deterministic() -> None:
    a = mvo.compute_music_vad_offset_db(0.7)
    b = mvo.compute_music_vad_offset_db(0.7)
    assert a.offset_db == b.offset_db
    assert a.panns_singing == b.panns_singing


# ── Konservativ-Schwelle (§G9, copilot-instructions.md) ───────────────────────


def test_below_min_confidence_no_offset() -> None:
    w = mvo.compute_music_vad_offset_db(0.24)
    assert w.offset_db == 0.0
    assert w.reasons, "konservative Begründung muss gesetzt sein"


def test_at_min_confidence_offset_applied() -> None:
    w = mvo.compute_music_vad_offset_db(0.25)
    assert w.offset_db == pytest.approx(-15.0 * 0.25, abs=1e-4)


# ── None-/§V6-Fallbacks ────────────────────────────────────────────────────


def test_none_singing_neutral_offset() -> None:
    w = mvo.compute_music_vad_offset_db(None)
    assert w.applied is False
    assert w.panns_singing is None
    assert w.offset_db == 0.0
    assert w.reasons, "None muss eine Begründung tragen (§V6, copilot-instructions.md)"


def test_non_numeric_singing_neutral() -> None:
    w = mvo.compute_music_vad_offset_db("gesang?")
    assert w.offset_db == 0.0
    assert w.panns_singing is None
    assert w.reasons


def test_nan_singing_neutral() -> None:
    w = mvo.compute_music_vad_offset_db(float("nan"))
    assert w.offset_db == 0.0
    assert w.panns_singing is None


def test_out_of_range_clipped() -> None:
    w = mvo.compute_music_vad_offset_db(1.5)
    assert w.offset_db == pytest.approx(-15.0, abs=1e-4)


# ── panns_tags-Fallback ────────────────────────────────────────────────────


def test_tags_fallback_singing_voice() -> None:
    w = mvo.compute_music_vad_offset_db(None, {"Singing voice": 0.6})
    assert w.offset_db == pytest.approx(-15.0 * 0.6, abs=1e-4)
    assert w.panns_singing == pytest.approx(0.6, abs=1e-4)


def test_tags_fallback_singing_key() -> None:
    w = mvo.compute_music_vad_offset_db(None, {"Singing": 0.4})
    assert w.offset_db == pytest.approx(-15.0 * 0.4, abs=1e-4)


def test_tags_fallback_none_when_missing() -> None:
    w = mvo.compute_music_vad_offset_db(None, {"Drum": 0.9})
    assert w.offset_db == 0.0
    assert w.panns_singing is None
    assert w.reasons


def test_singing_value_preferred_over_tags() -> None:
    # Expliziter panns_singing-Wert hat Vorrang vor den Tags.
    w = mvo.compute_music_vad_offset_db(0.8, {"Singing voice": 0.1})
    assert w.offset_db == pytest.approx(-15.0 * 0.8, abs=1e-4)


# ── Stereo-Layout ──────────────────────────────────────────────────────────


def test_downmix_channels_first_and_last() -> None:
    n = 1000
    left = _sine(n)
    right = 0.5 * np.sin(2 * np.pi * 880.0 * np.linspace(0, n / 44100, n, endpoint=False)).astype(np.float32)
    cf = np.stack([left, right])  # (2, N)
    cl = np.stack([left, right], axis=1)  # (N, 2)
    mono_cf = mvo._downmix(cf)
    mono_cl = mvo._downmix(cl)
    assert mono_cf.shape == (n,)
    assert mono_cl.shape == (n,)
    np.testing.assert_allclose(mono_cf, mono_cl, rtol=1e-6, atol=1e-6)


def test_downmix_mono_identity() -> None:
    mono = _sine()
    out = mvo._downmix(mono)
    assert out.shape == mono.shape


# ── report-only Invariante ─────────────────────────────────────────────────


def test_applied_always_false() -> None:
    for val in (0.0, 0.5, 1.0, None):
        assert mvo.compute_music_vad_offset_db(val).applied is False


def test_witness_log_string() -> None:
    w = mvo.compute_music_vad_offset_db(0.5)
    line = mvo.music_vad_witness_log(w)
    assert "report-only" in line
    assert "offset_db" in line
    assert "applied=False" in line
