"""Air-Presence-Enhancer Unit-Tests (§v10.19-Paket).

Prüft: Passthrough bei strength=0 (bit-identisch), Finite-Ausgabe,
No-Gain unterhalb des Luftbands (Tiefband unverändert), Gain nur bei
Energie oberhalb des Noise-Floors, Determinismus, Layout-Erhalt.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.dsp.air_presence_enhancer import enhance_air_presence

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


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
