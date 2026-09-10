"""§v10.753 (2026-09-09): STFT-IF-basierte Wow&Flutter-Schätzung mit Harmonischen-Konsens.

Prinzip (klassische FM-Demodulation, blind auf Musik angewendet):
1. STFT → pro Frame die Momentanfrequenz IF(t, k) über Phasen-Differenzbildung
   (Frame-zu-Frame-Phasenvorschub vs. erwarteten Bin-Vorschub).
2. Harmonischen-Konsens: Die W&F-Abweichung ist für ALLE Teiltöne eines Klangs
   identisch → gewichtete Mittelung der IF-Abweichungen über ganzzahlige
   Vielfache des dominanten F0 (Gewicht = |X|²) — robust gegen Einzelton-Rauschen.
3. Kalman-Glättung (langsame Drift, konstantes Zustandsmodell) → W&F-Kurve.

Deterministisch, vollständig vektorisiert, kein ML, §G5. Ergänzt den
frame-lokalen Pitch-Tracking-Solver in Phase 12 (Polyphonie-robust).

Mathematik: dphi = arg(Z[t+1]·conj(Z[t])); IF = f_bin + dphi/(2π·hop)·sr.
Deviation in Cents: 1200·log2(IF/f_bin).
"""

from __future__ import annotations

import numpy as np

_NFFT = 4096
_HOP = 1024
_MAX_HARMONICS = 12
_KALMAN_Q = 3e-4  # Prozessrauschen (langsame Drift, bidirektional geglättet)
_KALMAN_R = 0.25  # Messrauschen (Cents²)


def _dominant_f0_hps(mag: np.ndarray, sr: int, bin_hz: float) -> np.ndarray:
    """Harmonic-Product-Spectrum F0-Schätzung pro Frame (vektorisiert, log-Domäne)."""
    # mag: (F, T); Produkt über Downsampling-Faktoren 1..4
    hps = np.log(mag + 1e-10)
    for d in (2, 3, 4):
        hps[: mag.shape[0] // d] += np.log(mag[::d] + 1e-10)[: hps.shape[0] // d]
    # Suchbereich: 50–400 Hz
    lo = max(1, int(50 / bin_hz))
    hi = min(mag.shape[0] - 1, int(400 / bin_hz))
    idx = np.argmax(hps[lo:hi], axis=0) + lo
    return idx.astype(np.float32) * bin_hz  # (T,)


def estimate_wow_flutter(
    audio: np.ndarray,
    sr: int,
    *,
    f0_hz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Schätzt die W&F-Abweichungskurve.

    Returns:
        deviation_cents: (T,) Abweichung in Cents pro Frame (0 = kein W&F)
        frame_times:     (T,) Zeitpunkte in Sekunden
        confidence:      (T,) Konsens-Gewicht (0..1, niedrig bei Stille/Rauschen)
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    win = np.hanning(_NFFT).astype(np.float32)
    from scipy.signal import stft as _stft  # pylint: disable=import-outside-toplevel

    _, _, Z = _stft(
        audio,
        fs=sr,
        window=win,
        nperseg=_NFFT,
        noverlap=_NFFT - _HOP,
        boundary=None,
        padded=False,
        return_onesided=True,
    )
    Z = np.asarray(Z, dtype=np.complex64)  # (F, T)
    T = Z.shape[1]
    if T < 3:
        return np.zeros(T, np.float32), np.zeros(T, np.float32), np.zeros(T, np.float32)

    # ── 1. Phasen-Differenz → Momentanfrequenz-Abweichung je Bin ───────────
    dphi_raw = np.angle(Z[:, 1:] * np.conj(Z[:, :-1]))  # (−π, π]
    bin_hz = sr / _NFFT
    expected = 2.0 * np.pi * _HOP / _NFFT  # erwarteter Vorschub je Bin
    # §v10.753 Phasen-Unwrapping: ±50 Cent bei hohen Harmonischen liegen weit
    # über ±π — ganzzahlige 2π-Korrekturen stellen den wahren Vorschub her.
    dphi = dphi_raw + 2.0 * np.pi * np.round((expected - dphi_raw) / (2.0 * np.pi))
    df_hz = (dphi - expected) / (2.0 * np.pi * _HOP) * sr  # (F, T−1)
    f_bin = np.arange(Z.shape[0], dtype=np.float32)[:, None] * bin_hz
    with np.errstate(divide="ignore", invalid="ignore"):
        cents = 1200.0 * np.log2(1.0 + df_hz / (f_bin + 1e-9))  # (F, T−1)

    # ── 2. Dominantes F0 (HPS) ────────────────────────────────────────────
    mag = np.abs(Z)  # (F, T)
    f0 = _dominant_f0_hps(mag, sr, bin_hz)  # (T,)
    f0_median = float(np.median(f0[f0 > 0])) if np.any(f0 > 0) else 100.0

    # ── 3. Harmonischen-Konsens: gewichtete Abweichung über h·F0 ──────────
    consensus = np.zeros(T - 1, np.float64)
    weight_sum = np.zeros(T - 1, np.float64)
    for h in range(1, _MAX_HARMONICS + 1):
        center_hz = h * f0_median
        k = int(round(center_hz / bin_hz))
        if 1 <= k < Z.shape[0] - 1:
            w = (mag[k - 1, 1:] ** 2 + mag[k, 1:] ** 2 + mag[k + 1, 1:] ** 2)
            d = (cents[k - 1] * mag[k - 1, 1:] ** 2 + cents[k] * mag[k, 1:] ** 2 + cents[k + 1] * mag[k + 1, 1:] ** 2)
            consensus += d
            weight_sum += w
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.where(weight_sum > 1e-12, consensus / np.maximum(weight_sum, 1e-12), 0.0)
    total_energy = (mag[:, 1:] ** 2).sum(0)
    conf = np.clip(weight_sum / (total_energy + 1e-12), 0.0, 1.0)

    # ── 4. Bidirektionale Kalman-Glättung (lag-frei) ──────────────────────
    def _kalman_run(signal: np.ndarray) -> np.ndarray:
        x = 0.0
        p = 1.0
        out = np.empty_like(signal)
        for t in range(signal.shape[0]):
            p += _KALMAN_Q
            k_gain = p / (p + _KALMAN_R)
            x += k_gain * (signal[t] - x)
            p *= 1.0 - k_gain
            out[t] = x
        return out

    fwd = _kalman_run(dev)
    bwd = _kalman_run(dev[::-1])[::-1]
    smooth = 0.5 * (fwd + bwd)

    frame_times = (np.arange(T - 1) * _HOP + _NFFT // 2) / sr
    return smooth.astype(np.float32), frame_times.astype(np.float32), conf.astype(np.float32)


def correct_wow_flutter(audio: np.ndarray, sr: int, deviation_cents: np.ndarray, frame_times: np.ndarray) -> np.ndarray:
    """Zeitvariante Resampling-Korrektur entlang der Abweichungskurve.

    deviation_cents(t) > 0 = Ton zu HOCH (Wiedergabe zu schnell) → dehnen.
    Implementation: zeitvarianter Zeitstempel via kumulativer Korrektur +
    np.interp (linear — für langsame W&F hinreichend glatt, §G5-deterministisch).
    """
    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    t_orig = np.arange(n) / sr
    # Korrektur-Faktor (Samples pro Original-Sample): 2^(−cents/1200)
    rate = np.power(2.0, -deviation_cents / 1200.0)
    rate_t = np.interp(t_orig, frame_times, rate, left=rate[0], right=rate[-1]).astype(np.float64)
    t_new = np.cumsum(rate_t) / sr
    t_new -= t_new[0]
    # Ziel-Zeitachse = gleichmäßig: invertiere via Interpolation
    t_target = np.linspace(0.0, t_new[-1], n)
    corrected = np.interp(t_target, t_new, audio.astype(np.float64))
    return corrected.astype(np.float32)
