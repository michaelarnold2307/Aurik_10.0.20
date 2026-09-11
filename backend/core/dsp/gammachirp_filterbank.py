#!/usr/bin/env python3
"""backend/core/dsp/gammachirp_filterbank.py — Auditorische Peripherie (SOTA-Basis).

Ehrlicher Scope: Deterministische Gammachirp-Filterbank nach Irino & Patterson
(1997, 2001) in der vereinfachten Kaskadenform: Gammatone-Impulsantwort
(Patterson-Holdsworth) mit Frequenz-Chirp-Korrektur (asymmetrische Flanken),
Energie-Detektion pro Band. KEIN vollständiges dynamisches
Compressive-Gammachirp-Modell (dcGC) mit Level-abhängiger Schärfe — das wäre
ein eigenes Modul; die statische Form ist für NSIM-/Defekt-Ranking-Metriken
die etablierte Referenz.

Quellen:
  - Irino, T. & Patterson, R. D. (1997): „A time-domain, level-dependent
    auditory filter: The gammachirp".
  - Patterson, R. D., Nimmo-Smith, I., Holdsworth, J. & Rice, P. (1988):
    gammatone impulse response.
  - Glasberg & Moore (1990): ERB-Bandbreiten.

Determinismus (§G5 (GEBOTE.md)): FIR-Faltung mit deterministisch abgeschnittener
Impulsantwort — keine Zufallsgrößen.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

# Chirp-Faktor: 0 = reines Gammatone; typische Literaturwerte 1–3 für
# asymmetrische Flanken (Irino & Patterson 1997). Konservativ: 2.0.
_CHIRP_C: float = 2.0
_N_BANDS: int = 32
_F_LO_HZ: float = 80.0
_F_HI_HZ: float = 8000.0


def _erbb(freq_hz: float) -> float:
    return 24.7 * (4.37 * freq_hz / 1000.0 + 1.0)


def _band_centers_hz() -> list[float]:
    lo = 21.4 * np.log10(4.37 * _F_LO_HZ / 1000.0 + 1.0)
    hi = 21.4 * np.log10(4.37 * _F_HI_HZ / 1000.0 + 1.0)
    rates = np.linspace(lo, hi, _N_BANDS)
    return [float((10.0 ** (r / 21.4) - 1.0) * 1000.0 / 4.37) for r in rates]


def gammachirp_impulse_response(
    fc_hz: float,
    sr: int,
    order: int = 4,
    chirp_c: float = _CHIRP_C,
    level_db: float | None = None,
) -> np.ndarray:
    """Gammachirp-Impulsantwort (Irino & Patterson 1997), deterministisch.

    h(t) = a·t^(n-1)·exp(-2π·b·ERB(fc)·t)·cos(2π·fc·t + c·ln(t+ε))
    mit b = 1.019 (Glasberg & Moore 1990), c = Chirp-Faktor.

    dcGC-Basis (Level 3): mit ``level_db`` wird der Chirp-Faktor
    level-abhängig skaliert — bei höherem Pegel verbreitert sich das
    auditorische Filter (kompressives Gammachirp, Irino & Patterson 2001):
        c_eff = c · (0.6 + 0.4 · clip((level+60)/60, 0, 1))
    """
    if level_db is not None:
        _lvl = float(np.clip((level_db + 60.0) / 60.0, 0.0, 1.0))
        chirp_c = float(chirp_c) * (0.6 + 0.4 * _lvl)
    erb = _erbb(fc_hz)
    b = 1.019
    # Länge: genug Zeit für −60 dB Abklingen der Hüllkurve
    t_peak = (order - 1) / (2.0 * np.pi * b * erb + 1e-9)
    t_max = t_peak * 8.0
    n = int(round(t_max * sr))
    n = max(256, min(n, int(0.1 * sr)))
    t = np.arange(n, dtype=np.float64) / sr
    env = t ** (order - 1) * np.exp(-2.0 * np.pi * b * erb * t)
    phase = 2.0 * np.pi * fc_hz * t + chirp_c * np.log(t + 1e-6)
    h = env * np.cos(phase)
    h -= h.mean()  # DC-frei (Bandpass-Charakter)
    norm = np.sqrt(np.dot(h, h)) + 1e-12
    return cast(np.ndarray, (h / norm).astype(np.float32))


def gammachirp_spectrogram(
    audio: np.ndarray,
    sr: int,
    hop_ms: float = 10.0,
    n_bands: int = _N_BANDS,
    level_db: float | None = None,
) -> np.ndarray:
    """Band-Energien (bands, frames) — FIR-Konvolution, deterministisch.

    Layout: Mono (1-D) oder Stereo (beliebiges Layout) → mono_mix via
    audio_layout (§V7 (copilot-instructions.md)). Frames mit 10 ms Hop (Default), Hanning-frei —
    Energie-Envelope über gleitende Fenster der gefilterten Signale.
    """
    from backend.core.audio_layout import mono_mix

    x = mono_mix(np.asarray(audio, dtype=np.float64))
    if x.size < sr // 20:
        return cast(np.ndarray, np.zeros((n_bands, 1), dtype=np.float32))
    hop = max(1, int(round(hop_ms / 1000.0 * sr)))
    frame = hop * 2
    n_frames = max(1, (len(x) - frame) // hop + 1)
    out = np.zeros((n_bands, n_frames), dtype=np.float32)
    for b_i, fc in enumerate(_band_centers_hz()[:n_bands]):
        h = gammachirp_impulse_response(fc, sr, level_db=level_db)
        y = np.convolve(x, h.astype(np.float64), mode="same")
        env = y**2
        for f_i in range(n_frames):
            s = f_i * hop
            out[b_i, f_i] = float(np.sqrt(np.mean(env[s : s + frame]) + 1e-18))
    # Band-normalisieren (relative Excitation — robust gegen Pegel)
    band_norms = np.sqrt((out**2).sum(axis=1, keepdims=True)) + 1e-12
    return cast(np.ndarray, (out / band_norms).astype(np.float32))


def nsim_gammachirp(reference: np.ndarray, degraded: np.ndarray, sr: int) -> float:
    """Gammachirp-NSIM: Ähnlichkeit der Band-Enveloppen (Korrelation, 0..1).

    Deterministisches Pendant zu gammatone-NSIM-Metriken (PQS) — ehrlicher
    DSP-Front-End mit Gammachirp-Flanken.
    """
    spec_ref = gammachirp_spectrogram(reference, sr)
    spec_deg = gammachirp_spectrogram(degraded, sr)
    n_f = min(spec_ref.shape[1], spec_deg.shape[1])
    if n_f < 2:
        return 1.0
    sims: list[float] = []
    for b_i in range(spec_ref.shape[0]):
        a = spec_ref[b_i, :n_f]
        b = spec_deg[b_i, :n_f]
        if float(np.std(a)) < 1e-9 and float(np.std(b)) < 1e-9:
            sims.append(1.0)
            continue
        denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
        sims.append(float(np.dot(a, b) / (denom + 1e-12)))
    return float(np.clip(np.mean(sims), 0.0, 1.0))


__all__ = [
    "gammachirp_impulse_response",
    "gammachirp_spectrogram",
    "nsim_gammachirp",
]
