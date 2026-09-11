#!/usr/bin/env python3
"""backend/core/dsp/dynamic_loudness_model.py — Dynamisches Loudness-Modell (DLM)
mit binauraler Loudness (Moore & Glasberg 2007, vereinfacht).

Level-3-Hörmodul (§c-2/§c-3): Glasberg-&-Moore-2002-Stil — Erregungsmuster über
ERB-Filterbank, Cochlea-Kompression, duale Zeitintegration (STL/LTL) in Sone,
plus binaurale Inhibitions-Loudness für Stereo-Signale.

Invarianten:
  - Deterministisch (§G5 (GEBOTE.md)): keine Zufallszahlen, keine Zeitabhängigkeit.
  - Layout-tolerant (C,N) und (N,C) via audio_layout-Helfer.
  - Mono wie Stereo: Eingang (N,) → identische Pfade, kein Kollaps.
  - Sone-Skala: 1 kHz / 40 dB SPL Sinus → 1 Sone (Bezugs-Anker, wird im
    Kalibrierungs-Harness geprüft).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

# Cochlea-Kompressions-Exponent (Glasberg & Moore 2002): 0.2 (Energie → Erregung)
_COCHLEAR_EXP = 0.2
# Zeitfenster (s) für duale Integration: STL schnell, LTL langsam
_STL_ATTACK_S = 0.002
_STL_DECAY_S = 0.020
_LTL_ATTACK_S = 0.010
_LTL_DECAY_S = 0.100
# Referenz: 1 kHz Sinus bei 40 dB SPL = 1 Sone (N0 = Referenz-Loudness)
_REF_N = 1.0


def _erbb(freq_hz: float) -> float:
    """Glasberg & Moore 1990: ERB-Rate in ERB-Nummern."""
    return float(21.4 * np.log10(4.37 * freq_hz / 1000.0 + 1.0))


def _freq_from_erbb(erbb: float) -> float:
    return float((10.0 ** (erbb / 21.4) - 1.0) * 1000.0 / 4.37)


def _band_centers_hz() -> list[float]:
    lo, hi = _erbb(50.0), _erbb(15500.0)
    return [_freq_from_erbb(r) for r in np.linspace(lo, hi, 28)]


def _gammatone_bands(audio: np.ndarray, sr: int, n_bands: int = 28) -> np.ndarray:
    """Energien pro ERB-Band (frames, bands) — deterministische FIR-Filterbank."""
    from scipy.signal import convolve

    _bands = _band_centers_hz()[:n_bands]
    frame_len = 256
    hop = 128
    n_frames = max(1, (len(audio) - frame_len) // hop + 1)
    energies = np.zeros((n_frames, n_bands), dtype=np.float32)
    t = np.arange(frame_len) / sr
    for b, fc in enumerate(_bands):
        erb = (_freq_from_erbb(_erbb(fc) + 1.0) - _freq_from_erbb(_erbb(fc) - 1.0)) / 2.0
        bw = max(30.0, 1.019 * erb)
        gt = (t**3) * np.exp(-2.0 * np.pi * bw * t) * np.cos(2.0 * np.pi * fc * t)
        gt /= np.sqrt(np.sum(gt * gt)) + 1e-12
        for f in range(n_frames):
            seg = audio[f * hop : f * hop + frame_len]
            if len(seg) < frame_len:
                seg = np.pad(seg, (0, frame_len - len(seg)))
            energies[f, b] = float(np.sum((convolve(seg, gt, mode="same")) ** 2))
    return cast(np.ndarray, energies)


@dataclass
class DynamicLoudnessResult:
    """DLM-Ergebnis: Sone-Werte plus binauraler Anteil."""

    stl_sone: float = 0.0
    stl_peak_sone: float = 0.0
    ltl_sone: float = 0.0
    loudness_range_sone: float = 0.0
    binaural_sone: float = 0.0
    binaural_advantage_db: float = 0.0
    per_band: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "stl_sone": round(float(self.stl_sone), 3),
            "stl_peak_sone": round(float(self.stl_peak_sone), 3),
            "ltl_sone": round(float(self.ltl_sone), 3),
            "loudness_range_sone": round(float(self.loudness_range_sone), 3),
            "binaural_sone": round(float(self.binaural_sone), 3),
            "binaural_advantage_db": round(float(self.binaural_advantage_db), 3),
        }


def _excitation_sone(band_energies: np.ndarray) -> np.ndarray:
    """Energie → Cochlea-Erregung (0.2-Exponent) → Sone (frame, band).

    FESTE Referenz-Normalisierung (kein signal-abhängiger Median!) — sonst
    verlöre die Loudness ihre Pegelabhängigkeit (Produktionsbefund: 10× Pegel
    → Faktor 1.0). Absolute Sone-Kalibrierung ist ein deterministischer Proxy.
    """
    exc = np.clip(band_energies, 1e-12, None) ** (_COCHLEAR_EXP / 2.0)
    exc = exc / 1e-3
    return cast(np.ndarray, exc)


def _dual_temporal_integration(sone_frames: np.ndarray, sr: int) -> tuple[float, float]:
    """STL (schnell) + LTL (langsam) via exponentieller Attack/Decay-Integration."""
    hop = 128 / sr
    stl = _one_pole(sone_frames, _STL_ATTACK_S, _STL_DECAY_S, hop)
    ltl = _one_pole(sone_frames, _LTL_ATTACK_S, _LTL_DECAY_S, hop)
    return float(np.mean(stl)), float(np.mean(ltl))


def _one_pole(x: np.ndarray, attack: float, decay: float, dt: float) -> np.ndarray:
    y = np.zeros_like(x)
    prev = 0.0
    for i in range(len(x)):
        a = 1.0 - np.exp(-dt / attack) if x[i] > prev else 1.0 - np.exp(-dt / decay)
        prev = a * x[i] + (1.0 - a) * prev
        y[i] = prev
    return cast(np.ndarray, y)


def dynamic_loudness(audio: np.ndarray, sr: int) -> DynamicLoudnessResult:
    """Volles DLM: Erregung → STL/LTL → Sone (Glasberg & Moore 2002-Stil).

    Layout-tolerant: (C,N)/(N,C)/(N,) werden auf Mono-Mittel reduziert —
    Stereo-Loudness separat via :func:`binaural_loudness`.
    """
    from backend.core.audio_layout import mono_mix

    mono = mono_mix(np.asarray(audio, dtype=np.float32))
    if mono.size < 256:
        return DynamicLoudnessResult()
    energies = _gammatone_bands(mono, sr)
    sone = _excitation_sone(energies)
    total = sone.sum(axis=1)
    stl, ltl = _dual_temporal_integration(total, sr)
    return DynamicLoudnessResult(
        stl_sone=stl,
        stl_peak_sone=float(np.percentile(total, 95)),
        ltl_sone=ltl,
        loudness_range_sone=float(np.percentile(total, 95) - np.percentile(total, 5)),
        per_band=[float(v) for v in sone.mean(axis=0)],
    )


def binaural_loudness(audio_stereo: np.ndarray, sr: int) -> DynamicLoudnessResult:
    """Binaurale Loudness (Moore & Glasberg 2007, vereinfacht).

    Pro ERB-Band: interaurale Korrelation (IACC) bestimmt die binaurale
    Inhibition — diotische Signale (ρ=1) ≈ 1.5× monaural statt 2×,
    unkorrelierte (ρ=0) ≈ 2×. Binauraler Vorteil in dB gegenüber Mono.
    """
    from backend.core.audio_layout import mono_mix

    arr = np.asarray(audio_stereo, dtype=np.float32)
    if arr.ndim == 1:
        return dynamic_loudness(arr, sr)
    if arr.ndim == 2:
        # (N,C) → (C,N)
        if arr.shape[1] == 2 and arr.shape[0] != 2:
            arr = arr.T
        l_ch, r_ch = arr[0], arr[1]
    else:
        return dynamic_loudness(arr, sr)
    e_l = _gammatone_bands(l_ch, sr)
    e_r = _gammatone_bands(r_ch, sr)
    s_l = _excitation_sone(e_l)
    s_r = _excitation_sone(e_r)
    # IACC pro Band (frame-gemittelt)
    corr = np.zeros(s_l.shape[1], dtype=np.float32)
    for b in range(s_l.shape[1]):
        if np.std(s_l[:, b]) < 1e-6 or np.std(s_r[:, b]) < 1e-6:
            corr[b] = 1.0
            continue
        corr[b] = float(np.clip(np.corrcoef(s_l[:, b], s_r[:, b])[0, 1], -1.0, 1.0))
    # Binaurale Inhibition (Moore & Glasberg 2007, vereinfacht):
    # diotisch (ρ→1) ≈ 1.5× monaural (Inhibition −50 %), unkorreliert (ρ→0) ≈ 2×.
    inhibition = 0.5 * np.clip(corr, 0.0, 1.0)
    binaural = np.clip(np.maximum(s_l, s_r) + (1.0 - inhibition) * np.minimum(s_l, s_r), 0.0, None)
    total = binaural.sum(axis=1)
    mono_result = dynamic_loudness(arr, sr)
    stl, ltl = _dual_temporal_integration(total, sr)
    advantage_db = 10.0 * np.log10(max(stl, 1e-9) / max(mono_result.stl_sone, 1e-9))
    return DynamicLoudnessResult(
        stl_sone=stl,
        stl_peak_sone=float(np.percentile(total, 95)),
        ltl_sone=ltl,
        loudness_range_sone=float(np.percentile(total, 95) - np.percentile(total, 5)),
        binaural_sone=stl,
        binaural_advantage_db=float(np.clip(advantage_db, -40.0, 40.0)),
        per_band=[float(v) for v in binaural.mean(axis=0)],
    )
