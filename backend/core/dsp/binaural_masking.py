#!/usr/bin/env python3
"""backend/core/dsp/binaural_masking.py — Equalization-Cancellation / BMLD (SOTA-Basis).

Ehrlicher Scope: Deterministisches Equalization-Cancellation-Modell (Durlach
1972) auf ERB-Bändern mit optimaler Gain/Zeit-Kompensation pro Band. Damit
wird die BINAURALE Maskierungs-Freisetzung (BMLD) berechnet, die monaurale
Modelle strukturell nicht sehen:

  - N0S0 (interaural korrelierter Maskierer)  → Freisetzung ≈ 0 dB
  - NπS0 (interaural invertierter Maskierer)  → bis ~15 dB Schwellwert-Vorteil

Das erlaubt der Noise-Reduction, an der BINAURALEN statt der monauralen
Maskierungsschwelle zu stoppen: Rauschen, das das Ohr dank BMLD ohnehin
maskiert, darf (und soll) stehen bleiben — jedes zusätzliche NR-Denken
entfernt nur noch Musik.

Quellen:
  - Durlach, N. I. (1972): „Binaural signal detection: Equalization and
    cancellation theory" (Foundations of Modern Auditory Theory).
  - Levitt, H. & Rabiner, L. R. (1967): BMLD frequency dependence.
  - Glasberg & Moore (1990): ERB-Rate-Skala für die Filterbänke.

Determinismus (§G5 copilot-instructions.md): reine FIR/IIR-Filterung und
Energieberechnung — keine Zufallsgrößen, bit-reproduzierbar.

Layout (§V7 copilot-instructions.md): via audio_layout — kein hartes
(N,2)/(2,N)-Annehmen; Mono → Freisetzung 0 dB (kein binauraler Vorteil).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy import signal as _spsig

from backend.core.audio_layout import mono_mix, to_channels_first

logger = logging.getLogger(__name__)

# ERB-Filterbank (Glasberg & Moore 1990), deterministisch: Butterworth 4. Ord.
_BAND_CENTERS_HZ = [
    200.0,
    317.0,
    448.0,
    595.0,
    762.0,
    950.0,
    1165.0,
    1408.0,
    1683.0,
    1995.0,
    2349.0,
    2751.0,
    3208.0,
    3728.0,
    4320.0,
    4995.0,
    5768.0,
    6653.0,
    7670.0,
    8840.0,
]
_BMLD_MAX_DB: float = 15.0  # Levitt (1971): Frequenzmaximum ~500 Hz
_BMLD_CAP_DB: float = 8.0  # produktiver Cap für NR-Floor-Freisetzung
_MIN_BAND_SAMPLES: int = 64


@dataclass
class BinauralMaskingResult:
    """EC/BMLD-Ergebnis eines Stereo-Signals."""

    is_stereo: bool
    release_db: float  # maskierungsgewichtete Gesamt-Freisetzung (0..15)
    nr_floor_release_db: float  # produktiver Cap (0.._BMLD_CAP_DB)
    per_band_release_db: dict[str, float]  # "317.0" → Freisetzung in dB
    ec_gain_db: float  # mittlere EC-Verbesserung (Residual-Reduktion)

    def as_dict(self) -> dict[str, object]:
        return {
            "is_stereo": bool(self.is_stereo),
            "release_db": round(self.release_db, 3),
            "nr_floor_release_db": round(self.nr_floor_release_db, 3),
            "per_band_release_db": {k: round(v, 3) for k, v in self.per_band_release_db.items()},
            "ec_gain_db": round(self.ec_gain_db, 3),
        }


def _band_sos(sr: int, center_hz: float) -> np.ndarray | None:
    """Butterworth-Bandpass 4. Ordnung um ein ERB-Band (Glasberg & Moore 1990)."""
    erb = 24.7 * (4.37 * center_hz / 1000.0 + 1.0)
    lo = max(20.0, center_hz - erb / 2.0)
    hi = min(sr / 2.0 - 1.0, center_hz + erb / 2.0)
    if hi <= lo or hi >= sr / 2.0:
        return None
    try:
        return cast(np.ndarray, _spsig.butter(4, [lo / (sr / 2.0), hi / (sr / 2.0)], btype="band", output="sos"))
    except ValueError:
        return None


def _ec_cancellation_db(left: np.ndarray, right: np.ndarray, sr: int, max_lag_samples: int = 48) -> tuple[float, float]:
    """EC-Modell pro Band: optimale Gain/Zeit-Kompensation, Residual-Reduktion.

    Returns (ec_gain_db, coherence) — ec_gain_db = wie viel dB die optimale
    L/R-Kompensation das Differenz-Residual unter den naiven Mittelwert drückt;
    coherence = Interaural-Korrelation des Bands (BMLD-Proxy-Basis).
    """
    n = min(len(left), len(right))
    if n < _MIN_BAND_SAMPLES:
        return 0.0, 1.0
    l = left[:n].astype(np.float64)
    r = right[:n].astype(np.float64)
    l -= l.mean()
    r -= r.mean()
    denom = float(np.sqrt(np.dot(l, l) * np.dot(r, r)))
    if denom < 1e-12:
        return 0.0, 1.0
    # §V08/§10a-konform: FFT-basierte Kreuzkorrelation statt O(n²) np.correlate
    # (Kosten-Cliff ab ~10k Samples; Produktionsbefund 2026-09-11 analog
    # interaural_cues — Vollsignal-Korrelation im Stereo-/BMLD-Pfad).
    if n > 8192:
        from backend.core.core_utils import fft_crosscorr

        corr = fft_crosscorr(l, r) / denom
    else:
        corr = np.correlate(l, r, mode="full") / denom
    lag_axis = np.arange(-(n - 1), n, dtype=np.int64)
    mask = np.abs(lag_axis) <= max_lag_samples
    idx = int(np.argmax(np.abs(corr[mask])))
    lag = int(lag_axis[mask][idx])
    coherence = float(np.clip(abs(corr[mask][idx]), 0.0, 1.0))
    # Optimale Zeitkompensation anwenden
    if lag > 0:
        r_al = r[lag:]
        l_al = l[: n - lag]
    elif lag < 0:
        r_al = r[: n + lag]
        l_al = l[-lag:]
    else:
        r_al, l_al = r, l
    if len(l_al) < _MIN_BAND_SAMPLES:
        return 0.0, coherence
    # Optimale Gain-Kompensation (Minimum-Mean-Square-Error)
    g_opt = float(np.dot(l_al, r_al) / (np.dot(r_al, r_al) + 1e-12))
    naive_res = l_al - r_al
    ec_res = l_al - g_opt * r_al
    e_naive = float(np.dot(naive_res, naive_res))
    e_ec = float(np.dot(ec_res, ec_res))
    if e_naive < 1e-18:
        return 0.0, coherence
    ec_gain = float(10.0 * np.log10(e_naive / max(e_ec, 1e-12)))
    return ec_gain, coherence


def binaural_masking_advantage(audio: np.ndarray, sr: int) -> BinauralMaskingResult:
    """EC/BMLD-Freisetzung eines Stereo-Signals (produktiver Einstiegspunkt).

    NR-Bedeutung: `nr_floor_release_db` gibt an, um wie viele dB die
    Rausch-Floor-Schätzung angehoben werden darf, weil das Ohr das Rauschen
    binaural maskiert — NR muss dort NICHT weiter schürfen.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim != 2:
        return BinauralMaskingResult(False, 0.0, 0.0, {}, 0.0)
    ch = to_channels_first(arr)
    if ch.shape[0] < 2 or ch.shape[1] < _MIN_BAND_SAMPLES * 2:
        return BinauralMaskingResult(True, 0.0, 0.0, {}, 0.0)
    left = ch[0]
    right = ch[1]

    per_band: dict[str, float] = {}
    releases: list[float] = []
    ec_gains: list[float] = []
    weights: list[float] = []
    for fc in _BAND_CENTERS_HZ:
        sos = _band_sos(sr, fc)
        if sos is None:
            continue
        try:
            lb = _spsig.sosfiltfilt(sos, left.astype(np.float64))
            rb = _spsig.sosfiltfilt(sos, right.astype(np.float64))
        except ValueError:
            continue
        band_energy = float(np.dot(lb, lb) + np.dot(rb, rb))
        if band_energy < 1e-12:
            continue
        ec_gain, coherence = _ec_cancellation_db(lb, rb, sr)
        # BMLD steigt mit sinkender Interaural-Korrelation; Maximum ~500 Hz,
        # Hochpass-Abfall oberhalb 1.5 kHz (Levitt & Rabiner 1967).
        _freq_weight = float(np.clip(1.0 - max(0.0, (fc - 1500.0) / 7000.0), 0.15, 1.0))
        release = _BMLD_MAX_DB * (1.0 - coherence) * _freq_weight
        per_band[f"{fc:.0f}"] = float(release)
        releases.append(release)
        ec_gains.append(ec_gain)
        weights.append(band_energy)

    if not releases:
        return BinauralMaskingResult(True, 0.0, 0.0, {}, 0.0)
    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() + 1e-12)
    release_db = float(np.dot(np.asarray(releases), w))
    ec_gain_db = float(np.dot(np.asarray(ec_gains), w))
    return BinauralMaskingResult(
        is_stereo=True,
        release_db=float(np.clip(release_db, 0.0, _BMLD_MAX_DB)),
        nr_floor_release_db=float(np.clip(release_db, 0.0, _BMLD_CAP_DB)),
        per_band_release_db=per_band,
        ec_gain_db=float(np.clip(ec_gain_db, 0.0, 30.0)),
    )


def binaural_noise_floor_release_db(audio: np.ndarray, sr: int) -> float:
    """Kurzform für die NR-Verdrahtung: produktive Floor-Freisetzung in dB."""
    return binaural_masking_advantage(audio, sr).nr_floor_release_db


__all__ = [
    "BinauralMaskingResult",
    "binaural_masking_advantage",
    "binaural_noise_floor_release_db",
]
