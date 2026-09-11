#!/usr/bin/env python3
"""backend/core/dsp/temporal_loudness.py — Zeitvariantes Lautheitsmodell (DLM-Basis).

Ehrlicher Scope: Reduziertes, deterministisches dynamisches Lautheitsmodell
nach Moore-Glasberg-Prinzip (Moore, Glasberg & Baer 1997; ITU-R BS.1770-
Verwandtschaft für die Mittelohr-Korrektur), KEINE vollständige ISO 532-3-
Implementierung:

  1. ERB-Filterbank (Glasberg & Moore 1990) — 28 Bänder, 50 Hz–15 kHz
  2. Mittelohr-Übertragung: vereinfachte 400-Hz-Betonung + Hochpass
     (deterministische IIR-Kette, dokumentierte Näherung)
  3. Erregung → spezifische Lautheit: Kompressions-Exponent 0.23,
     Ruhehörschwellen-Floor pro Band
  4. Short-Term-Loudness (STL): Attack ~10 ms / Release ~100 ms
  5. Long-Term-Loudness (LTL): Release ~2 s

Produktiver Zweck: Pumpen-freie Dynamiksteuerung. `stl_adaptive_smoothing()`
liefert aus der STL-Kurve adaptive Zeitkonstanten (laut = schnell, leise =
langsam) für Gain-Enveloppen in Loudness-/Mastering-Phasen — die klassische
„NR/Limiter atmet"-Ursache wird damit deterministisch gedämpft.

Determinismus (§G5 copilot-instructions.md): reine Filterung/EMA — keine
Zufallsgrößen. Layout (§V7 (copilot-instructions.md)): mono_mix-Eingang, kein Layout-Annehmen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy import signal as _spsig

from backend.core.audio_layout import mono_mix

logger = logging.getLogger(__name__)

_N_BANDS: int = 28
_F_LO_HZ: float = 50.0
_F_HI_HZ: float = 15000.0
_COMPRESSION_EXP: float = 0.23
_STL_ATTACK_S: float = 0.010
_STL_RELEASE_S: float = 0.100
_LTL_RELEASE_S: float = 2.0


@dataclass
class TemporalLoudnessResult:
    """STL/LTL-Kurven (Sone) eines Signals."""

    stl_sone: np.ndarray  # pro Sample
    ltl_sone: np.ndarray  # pro Sample
    peak_stl_sone: float
    integrated_ltl_sone: float

    def as_dict(self) -> dict[str, object]:
        return {
            "peak_stl_sone": round(float(self.peak_stl_sone), 3),
            "integrated_ltl_sone": round(float(self.integrated_ltl_sone), 3),
            "stl_range_sone": round(float(np.ptp(self.stl_sone)), 3),
        }


def _erbb_scale(freq_hz: float) -> float:
    """ERB-Rate nach Glasberg & Moore (1990), Eq. 3 (vereinfacht)."""
    return float(21.4 * np.log10(4.37 * freq_hz / 1000.0 + 1.0))


def _freq_from_erbb(erbb: float) -> float:
    return float((10.0 ** (erbb / 21.4) - 1.0) * 1000.0 / 4.37)


def _band_centers_hz() -> list[float]:
    lo = _erbb_scale(_F_LO_HZ)
    hi = _erbb_scale(_F_HI_HZ)
    rates = np.linspace(lo, hi, _N_BANDS)
    return [float(_freq_from_erbb(r)) for r in rates]


def _middle_ear_sos(sr: int) -> np.ndarray | None:
    """Vereinfachte Mittelohr-Übertragung: 400-Hz-Resonanz + Hochpass 40 Hz.

    Dokumentierte Näherung (kein vollständiges Moore-Glasberg-Outer/Middle-
    Ear-Modell) — deterministisch, biquad-Kette.
    """
    sos_high = _spsig.butter(2, max(20.0, 40.0) / (sr / 2.0), btype="high", output="sos")
    sos_res = _spsig.butter(2, [250.0 / (sr / 2.0), 600.0 / (sr / 2.0)], btype="band", output="sos")
    sos_peq = _spsig.butter(2, 400.0 / (sr / 2.0), btype="low", output="sos")
    return cast(np.ndarray, np.vstack([sos_high, sos_res, sos_peq]).astype(np.float64))


def temporal_loudness(audio: np.ndarray, sr: int) -> TemporalLoudnessResult:
    """Berechnet STL/LTL-Kurven (Sone) eines Mono- oder Stereo-Signals."""
    mono = mono_mix(np.asarray(audio, dtype=np.float64))
    if mono.size < max(sr // 10, 256):
        empty = np.zeros(max(mono.size, 1), dtype=np.float64)
        return TemporalLoudnessResult(empty, empty, 0.0, 0.0)

    # Mittelohr-Kette
    sos_me = _middle_ear_sos(sr)
    x = _spsig.sosfiltfilt(sos_me, mono).astype(np.float64)

    # ERB-Erregung
    specific = np.zeros_like(x)
    for fc in _band_centers_hz():
        erb = 24.7 * (4.37 * fc / 1000.0 + 1.0)
        lo = max(20.0, fc - erb / 2.0)
        hi = min(sr / 2.0 - 1.0, fc + erb / 2.0)
        if hi <= lo:
            continue
        try:
            sos = _spsig.butter(4, [lo / (sr / 2.0), hi / (sr / 2.0)], btype="band", output="sos")
            band = _spsig.sosfiltfilt(sos, x)
        except ValueError:
            continue
        exc = band**2
        # Ruhehörschwelle-Floor: relative Schwelle steigt zu den Bandrändern
        thr = 1e-10 * (1.0 + max(0.0, (1000.0 - fc) / 1000.0) ** 2 + max(0.0, (fc - 8000.0) / 7000.0) ** 2)
        specific += np.maximum(exc, thr) ** _COMPRESSION_EXP

    specific = np.clip(specific, 0.0, None)

    # Short-Term-Loudness: Attack schnell, Release langsamer (EMA, deterministisch)
    a_stl = float(np.exp(-1.0 / max(_STL_ATTACK_S * sr, 1.0)))
    r_stl = float(np.exp(-1.0 / max(_STL_RELEASE_S * sr, 1.0)))
    stl = np.empty_like(specific)
    prev = 0.0
    for i in range(len(specific)):
        c = a_stl if specific[i] > prev else r_stl
        prev = c * prev + (1.0 - c) * specific[i]
        stl[i] = prev

    # Long-Term-Loudness
    a_ltl = float(np.exp(-1.0 / max(_LTL_RELEASE_S * sr, 1.0)))
    ltl = np.empty_like(stl)
    prev = 0.0
    for i in range(len(stl)):
        prev = a_ltl * prev + (1.0 - a_ltl) * stl[i]
        ltl[i] = prev

    return TemporalLoudnessResult(
        stl_sone=stl.astype(np.float64),
        ltl_sone=ltl.astype(np.float64),
        peak_stl_sone=float(np.max(stl)),
        integrated_ltl_sone=float(np.mean(ltl[len(ltl) // 2 :])),
    )


def stl_adaptive_smoothing(
    gain_db: np.ndarray,
    audio: np.ndarray,
    sr: int,
    attack_ms: float = 50.0,
    release_ms: float = 200.0,
) -> np.ndarray:
    """Glättet eine Gain-Kurve (dB) mit STL-adaptiven Zeitkonstanten.

    Laut (STL hoch) → schnellere Reaktion (Attack); leise Passagen → längere
    Release-Zeit — verhindert hörbares Pumpen, ohne musikalische Transienten
    zu verschleifen. Deterministisch, NaN/Inf-sicher, nie größer als Input-Max.
    """
    g = np.asarray(gain_db, dtype=np.float64)
    g = np.clip(np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), -24.0, 24.0)
    if g.size < 4:
        return cast(np.ndarray, g.astype(np.float64))
    try:
        tl = temporal_loudness(audio, sr)
        stl = tl.stl_sone
        if stl.size != g.size:
            stl = np.interp(np.linspace(0, 1, g.size), np.linspace(0, 1, stl.size), stl)
        stl_ref = max(float(np.median(stl)), 1e-9)
        loudness_ratio = np.clip(stl / stl_ref, 0.05, 20.0)
    except Exception as exc:
        logger.debug("STL-Berechnung fehlgeschlagen — konstante Zeitkonstanten: %s", exc)
        loudness_ratio = np.ones(g.size, dtype=np.float64)

    out = np.empty_like(g)
    prev = float(g[0])
    for i in range(g.size):
        local_attack = attack_ms / 1000.0 * np.clip(1.0 / loudness_ratio[i], 0.3, 2.0)
        local_release = release_ms / 1000.0 * np.clip(loudness_ratio[i], 0.3, 3.0)
        rising = g[i] > prev
        tau = local_attack if rising else local_release
        c = float(np.exp(-1.0 / max(tau * sr, 1.0)))
        prev = c * prev + (1.0 - c) * g[i]
        out[i] = prev
    return cast(np.ndarray, np.clip(out, -24.0, 24.0).astype(np.float64))


__all__ = [
    "TemporalLoudnessResult",
    "temporal_loudness",
    "stl_adaptive_smoothing",
]
