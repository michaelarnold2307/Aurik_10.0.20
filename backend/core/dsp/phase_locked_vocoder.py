"""§v10.758-Rebuild (2026-09-09): Phasen-verriegelter Vocoder.

SOTA-Anforderungen (aus dem Prototyp-Review):
1. Manuelle OLA mit expliziter Gewicht-Normalisierung (beliebiger Synthese-Hop).
2. Identity-Phase-Locking: Nur Spektral-Peaks tragen die propagierte Phase;
   Nicht-Peak-Bins übernehmen die Phase des nächstgelegenen Peaks.
3. Transienten-Phase-Reset: Onset-Frames behalten die Analyse-Phase.

Deterministisch, vektorisiert, kein ML, §G5.
"""

from __future__ import annotations

import numpy as np

_NFFT = 2048
_HOP = 512


def phase_locked_stretch(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    if abs(rate - 1.0) < 1e-6:
        return np.asarray(audio, dtype=np.float32)
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
    F, T = Z.shape
    if T < 4:
        return audio

    H_a = _HOP
    H_s = max(1, int(round(H_a * rate)))  # rate>1 = langsamer → größerer Hop
    mag = np.abs(Z)
    phase = np.angle(Z)

    # ── Spektral-Peaks je Frame (lokale Maxima, ±2 Bins) ──────────────────
    is_peak = np.zeros((F, T), dtype=bool)
    for off in (-2, -1, 1, 2):
        _s = np.zeros_like(mag)
        _s[max(0, -off) : F - max(0, off)] = mag[max(0, off) : F + min(0, off)]
        is_peak |= mag > _s
    is_peak[0, :] = False
    is_peak[-1, :] = False

    # ── Transienten-Erkennung (Spektral-Flux) ──────────────────────────────
    flux = np.zeros(T, dtype=np.float32)
    flux[1:] = np.abs(mag[:, 1:] - mag[:, :-1]).sum(0)
    is_onset = flux > 2.0 * float(np.median(flux[flux > 0])) + 1e-6

    # ── Phasen-Propagation der Peaks ───────────────────────────────────────
    freq_bins = np.arange(F, dtype=np.float64)[:, None] * (2.0 * np.pi / _NFFT)
    expected = freq_bins * H_a
    dphi = np.diff(phase, axis=1)
    dev = (dphi - expected + np.pi) % (2.0 * np.pi) - np.pi
    true_freq = expected + dev  # (F, T-1) rad/Frame

    n_out = int(T * rate)
    src_idx = np.minimum((np.arange(n_out) * H_a // H_s).astype(int), T - 2)

    # Peak-Phase propagiert; Nicht-Peaks übernehmen den nächstgelegenen Peak
    peak_phase = np.zeros((F, n_out), dtype=np.float64)
    for t in range(n_out):
        s = src_idx[t]
        if t == 0:
            peak_phase[:, 0] = phase[:, 0]
            continue
        advance = true_freq[:, max(s - 1, 0)] * H_s
        peak_phase[:, t] = peak_phase[:, t - 1] + advance
        if is_onset[s]:
            peak_phase[:, t] = phase[:, s]  # Transienten-Reset

    # Identity-Locking: je Bin die propagierte Phase des nächstgelegenen Peaks
    # PLUS die in der Analyse erhaltene Phasen-Differenz zu diesem Peak — das
    # erhält die relative Phasenstruktur (sonst kollabiert das Spektrum).
    out_phase = np.zeros((F, n_out), dtype=np.float64)
    for t in range(n_out):
        s = src_idx[t]
        peaks = np.where(is_peak[:, s])[0]
        if peaks.size == 0:
            out_phase[:, t] = peak_phase[:, t]
            continue
        nearest = peaks[np.argmin(np.abs(np.arange(F)[:, None] - peaks[None, :]), axis=1)]
        rel = phase[:, s] - phase[nearest, s]
        out_phase[:, t] = peak_phase[nearest, t] + rel
        # Peaks selbst behalten ihre propagierte Phase (rel==0 dort)
        out_phase[is_peak[:, s], t] = peak_phase[is_peak[:, s], t]

    Z_out = (mag[:, src_idx] * np.exp(1j * out_phase)).astype(np.complex64)

    # ── Manuelle OLA mit Gewicht-Normalisierung ────────────────────────────
    # Die Fensterung steckt BEREITS in den STFT-Koeffizienten (Analyse-Fenster)
    # → Synthese-Frames NICHT erneut fenstern; normalisieren mit Σw² des
    # Analyse-Fensters über das Synthese-Hop-Gitter.
    out_len = (n_out - 1) * H_s + _NFFT
    acc = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)
    for t in range(n_out):
        seg = np.fft.irfft(Z_out[:, t], n=_NFFT)
        start = t * H_s
        acc[start : start + _NFFT] += seg
        wsum[start : start + _NFFT] += win.astype(np.float64) ** 2
    y = acc / np.maximum(wsum, 1e-9)
    return y.astype(np.float32)
