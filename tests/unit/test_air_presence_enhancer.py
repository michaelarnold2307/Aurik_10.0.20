"""Air-Presence-Enhancer Unit-Tests (§v10.19-Paket, D-K3-98).

Prüft: Passthrough bei strength=0 (bit-identisch), Finite-Ausgabe,
No-Gain unterhalb des Luftbands (Tiefband unverändert), Gain nur bei
Energie oberhalb des Noise-Floors, Determinismus, Layout-Erhalt.
D-K3-98 zusätzlich: COLA-Rekonstruktion ohne Ripple (identisches periodisches
Fenster in Analyse und Synthese), Eingangs-Decke statt Hard-Clip.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from backend.core.dsp.air_presence_enhancer import (
    _istft_frames,
    _stft_frames,
    enhance_air_presence,
)

SR = 48000


def _air_signal(duration_s: float = 2.0, with_air: bool = True) -> np.ndarray:
    rng = np.random.default_rng(7)
    n = int(SR * duration_s)
    t = np.arange(n) / SR
    base = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
    if with_air:
        base = base + 0.05 * np.sin(2 * np.pi * 12000 * t)
    base = base + 0.01 * rng.standard_normal(n)
    return np.tanh(base * 1.5).astype(np.float32)


def test_strength_zero_is_bit_identical_passthrough():
    audio = _air_signal()
    out = enhance_air_presence(audio, SR, strength=0.0)
    np.testing.assert_array_equal(out, audio)


def test_output_finite_and_clipped():
    audio = _air_signal()
    out = enhance_air_presence(audio, SR, strength=0.3)
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0
    assert out.shape == audio.shape
    assert out.dtype == np.float32


def test_low_band_unchanged_energy():
    """Unterhalb des Luftbands darf keine Energieänderung auftreten."""
    audio = _air_signal(with_air=True)
    out = enhance_air_presence(audio, SR, strength=0.25)
    low_in = float(np.mean(audio[:SR] ** 2))
    low_out = float(np.mean(out[:SR] ** 2))
    assert low_out == pytest.approx(low_in, rel=0.05)


def test_no_gain_on_noise_only_signal():
    """Reines Rauschen unter dem Noise-Floor wird nicht angehoben."""
    rng = np.random.default_rng(3)
    noise = (rng.standard_normal(SR) * 1e-5).astype(np.float32)  # −100 dBFS ≪ −80 dB Floor
    out = enhance_air_presence(noise, SR, strength=0.3)
    assert float(np.mean(out**2)) <= float(np.mean(noise**2)) * 1.2


def test_deterministic():
    audio = _air_signal()
    out_a = enhance_air_presence(audio, SR, strength=0.2)
    out_b = enhance_air_presence(audio, SR, strength=0.2)
    np.testing.assert_array_equal(out_a, out_b)


def test_stereo_layout_preserved():
    mono = _air_signal()
    stereo = np.stack([mono, mono], axis=0).astype(np.float32)
    out = enhance_air_presence(stereo, SR, strength=0.2)
    assert out.shape == (2, len(mono))
    np.testing.assert_array_equal(out[0], out[1])


def _sine(freq_hz: float, duration_s: float = 1.0, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(SR * duration_s)) / SR
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_cola_rekonstruktion_ohne_ripple():
    """D-K3-98a/b: Hin- und Rücktransform mit demselben periodischen Fenster
    muss den Sinus exakt rekonstruieren (vorher 8,9 % Relativfehler + NOLA-Warnung)."""
    audio = _sine(1000.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _spec = _stft_frames(audio, 2048, 512)
        rec = _istft_frames(_spec, 2048, 512, len(audio))
    nola = [str(w.message) for w in caught if "NOLA" in str(w.message)]
    assert nola == []
    rel_err = float(np.linalg.norm(rec - audio) / np.linalg.norm(audio))
    assert rel_err < 1e-6


def test_kein_luftband_ist_bitnah_passthrough():
    """Ein 440-Hz-Ton hat keine Luftband-Energie ⇒ der Enhancer darf ihn bis auf
    Gleitkomma-Rauschen unverändert lassen (vorher: 0,25 = volle Amplitude)."""
    audio = _sine(440.0)
    out = enhance_air_presence(audio, SR, strength=0.15)
    assert float(np.max(np.abs(out - audio))) < 1e-6


def test_eingangsdecke_statt_hardclip():
    """D-K3-98c: lautes Material wird nicht bei ±1,0 hart geklippt; der Zusatz
    wird zurückgenommen, der Ausgangspeak bleibt an der Eingangs-Decke."""
    audio = _sine(12000.0, duration_s=1.0, amp=0.99)
    out = enhance_air_presence(audio, SR, strength=1.0)
    peak_in = float(np.max(np.abs(audio)))
    assert float(np.max(np.abs(out))) <= peak_in + 1e-6
    assert int(np.sum(np.abs(out) >= 0.99999)) == 0


def test_sehr_kurzes_signal_bleibt_finite():
    """Kürzer als ein STFT-Fenster (2048) darf nicht crashen
    (§V6 (copilot-instructions.md))."""
    audio = np.zeros(500, dtype=np.float32)
    audio[200] = 0.5
    out = enhance_air_presence(audio, SR, strength=0.2)
    assert out.shape == audio.shape
    assert np.isfinite(out).all()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
