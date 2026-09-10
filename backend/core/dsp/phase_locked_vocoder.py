"""§v10.758-Rebuild (2026-09-09): Phasen-verriegelter Vocoder.

SOTA-Anforderungen (aus dem Prototyp-Review):
1. Manuelle OLA mit expliziter Gewicht-Normalisierung (beliebiger Synthese-Hop).
2. Identity-Phase-Locking: Nur Spektral-Peaks tragen die propagierte Phase;
   Nicht-Peak-Bins übernehmen die Phase des nächstgelegenen Peaks.
3. Transienten-Phase-Reset: Onset-Frames behalten die Analyse-Phase.

§v10.759 (2026-09-09): STFT/OLA vollständig manuell (np.fft.rfft) statt
scipy.signal.stft — dessen versteckte Fenster-Normalisierung (w/sum(w),
scaling='density') hatte die Energie-Kalibrierung der OLA unkontrollierbar
gemacht. Jetzt gilt explizit: Z[t] = rfft(x·w), OLA-Norm = Σ_m w über das
Synthese-Gitter, Phasen-Advance in Frame-Einheiten (true_freq·H_s/H_a).

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

    # ── Manuelle STFT: Z[t] = rfft(x·w) — keine versteckte Normalisierung ──
    n = len(audio)
    n_frames = 1 + max(0, (n - _NFFT) // _HOP)
    if n_frames < 4:
        return audio
    F = _NFFT // 2 + 1
    Z = np.empty((F, n_frames), dtype=np.complex64)
    for t in range(n_frames):
        seg = audio[t * _HOP : t * _HOP + _NFFT] * win
        Z[:, t] = np.fft.rfft(seg, n=_NFFT).astype(np.complex64)
    T = n_frames

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
    true_freq = expected + dev  # (F, T-1) rad je Analyse-Hop (H_a Samples)

    # Korrekte Ausgabeframing: Gesamtlänge ≈ (T-1)·H_a·rate + N
    n_out = 1 + int(round((T - 1) * H_a * rate / H_s))
    src_idx = np.minimum((np.arange(n_out) * H_a // H_s).astype(int), T - 2)

    # Peak-Phase propagiert; Nicht-Peaks übernehmen den nächstgelegenen Peak
    peak_phase = np.zeros((F, n_out), dtype=np.float64)
    for t in range(n_out):
        s = src_idx[t]
        if t == 0:
            peak_phase[:, 0] = phase[:, 0]
            continue
        # Advance in Frame-Einheiten: true_freq ist rad je ANALYSE-Hop (H_a)
        # Samples → für den Synthese-Hop H_s mit H_s/H_a skalieren (§v10.759).
        advance = true_freq[:, max(s - 1, 0)] * (H_s / H_a)
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

    # ── Manuelle OLA (LSEE) mit Gewicht-Normalisierung ─────────────────────
    # Synthese-Fenster ANWENDEN und mit Σ_m w² über das Synthese-Gitter
    # normalisieren: Σw² hat für Hann (DC=0.375) bei BELIEBIGEM Hop einen
    # strikt positiven Boden (keine Nullstellen wie Σw) → keine Blow-ups,
    # exakte Einheitsverstärkung (§v10.759).
    out_len = (n_out - 1) * H_s + _NFFT
    acc = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)
    win2 = win.astype(np.float64) ** 2
    for t in range(n_out):
        seg = np.fft.irfft(Z_out[:, t], n=_NFFT) * win
        start = t * H_s
        acc[start : start + _NFFT] += seg
        wsum[start : start + _NFFT] += win2
    y = acc / np.maximum(wsum, 1e-9)
    # Degenerationszone am Signalende abschneiden: wo nur noch ein Fenster
    # überlappt (wsum→0), wird ε/w zum Spike. Trunkieren auf die Region mit
    # wsum > 1e-3 (natürlicher Fenster-Fade bis w≈0.03, unhörbar).
    valid = np.where(wsum > 1e-3)[0]
    if valid.size:
        y = y[: valid[-1] + 1]
    return y.astype(np.float32)
