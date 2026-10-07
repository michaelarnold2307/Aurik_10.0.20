"""D-K3-43: `_dsp_hf_rolloff` — Real-Musik-Kalibrierung (Register D-K3-43).

Pinnt die Kalibrier-Matrix des Fusion-Fixes (2026-10-07):

  - **Weißrauschen & Weiß-LP bleiben unverändert** (3k-LP→~2,8k; 4k-LP→~3,7k):
    die Konsistenz-Bedingung (max/min ≤ 2,6) schützt die LP-Skirt-Kalibrierung.
  - **LF-dominante Musik/pink kollabieren nicht mehr auf E90** (~2–6 kHz):
    Konvergenz-Cluster bzw. Band-Decay-Beweis (E(10–20 kHz) ≥ E(1–3 kHz) − 6 dB)
    übernehmen; reale Musik (E90 ≈ 2 kHz) und pink (E90 ≈ 6 kHz) sind die
    dokumentierten Kollaps-Fälle.
  - **Determinismus** (§G5 (copilot-instructions.md)): identische Eingabe ⇒ identisches Ergebnis.

Real-Fixture-Checks laufen nur, wenn die test_audio-WAVs vorhanden sind
(`*.wav` ist gitignored) — sonst Skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from backend.core.era_classifier import _dsp_hf_rolloff

SR = 48000
N = 10 * SR
_PROJECT = Path(__file__).resolve().parents[2]


def _white(n: int, seed: int = 42) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(n) * 0.1).astype(np.float32)


def _lp(x: np.ndarray, cut: float) -> np.ndarray:
    return sosfilt(butter(6, cut, "lowpass", fs=SR, output="sos"), x).astype(np.float32)


def _pink(n: int, seed: int = 42) -> np.ndarray:
    w = np.random.default_rng(seed).standard_normal(n)
    x = np.fft.irfft(np.fft.rfft(w) / np.sqrt(np.maximum(np.fft.rfftfreq(n, 1 / SR), 1.0)), n)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.3).astype(np.float32)


def _music_mock(n: int, seed: int = 42) -> np.ndarray:
    t = np.arange(n) / SR
    x = np.zeros(n)
    for f, a in [(80, 0.30), (160, 0.25), (320, 0.20), (640, 0.10)]:
        x += a * np.sin(2 * np.pi * f * t)
    for f, a in [(1000, 0.03), (2000, 0.02), (3000, 0.02), (9000, 0.01), (12000, 0.008), (14000, 0.006)]:
        x += a * np.sin(2 * np.pi * f * t)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.5).astype(np.float32)


def test_white_fullband_stays_fullband() -> None:
    assert _dsp_hf_rolloff(_white(N), SR) >= 20000.0


def test_white_lp3k_keeps_lp_calibration() -> None:
    """LP-Skirt-Schutz: der Band-Decay/Cluster darf nicht auf den Skirt (5k/19k) springen."""
    assert 2200.0 <= _dsp_hf_rolloff(_lp(_white(N), 3000), SR) <= 3600.0


def test_white_lp4k_keeps_lp_calibration() -> None:
    assert 3000.0 <= _dsp_hf_rolloff(_lp(_white(N), 4000), SR) <= 4600.0


def test_pink_noise_is_not_collapsed_to_e90() -> None:
    """Kollaps-Fall 1: pinkes Rauschen — E90 ≈ 6 kHz, Energie bis Nyquist vorhanden."""
    assert _dsp_hf_rolloff(_pink(N), SR) >= 15000.0


def test_music_with_hf_is_not_collapsed_to_e90() -> None:
    """Kollaps-Fall 2: LF-dominante Musik (E90 ≈ 0,3 kHz) mit HF-Anteil bis 14 kHz."""
    assert _dsp_hf_rolloff(_music_mock(N), SR) >= 9000.0


def test_rolloff_is_deterministic() -> None:
    x = _music_mock(N, seed=7)
    assert _dsp_hf_rolloff(x, SR) == _dsp_hf_rolloff(x, SR)


def test_real_fixtures_if_present() -> None:
    """Real-Audio-Nachweis (nur mit vorhandenen test_audio-WAVs, gitignored)."""
    jazz = _PROJECT / "test_audio" / "vinyl" / "jazz_1950s_scratched.wav"
    elke = (
        _PROJECT
        / "test_audio"
        / ("Elke Best - Hey Kleiner, mit Dir spielt wohl keiner (1977) HD audio 320 kbps german (original).wav")
    )
    if not jazz.exists() or not elke.exists():
        pytest.skip("test_audio-Fixtures nicht vorhanden (gitignored) — Real-Check übersprungen")
    import soundfile as sf

    x, sr = sf.read(str(jazz), always_2d=True)
    assert _dsp_hf_rolloff(x.mean(axis=1).astype(np.float32), sr) >= 8000.0, (
        "1950s-Jazz-Fixture trägt HF bis ~20k (E(10–20k) nur −3 dB unter E(1–3k)) — kein 1,3-kHz-Kollaps"
    )
    y, sr_y = sf.read(str(elke), always_2d=True)
    win = y[: 10 * sr_y].mean(axis=1).astype(np.float32)
    assert _dsp_hf_rolloff(win, sr_y) >= 12000.0, "echter 1977er Song: kein 2-kHz-Kollaps"
