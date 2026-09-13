"""WF-V2: F0-unabhängige Warp-Schätzung über harmonische Spektral-Korrelation.

Wow/Flutter ist eine gemeinsame Zeit-Warp-Funktion für ALLE Frequenzen —
die F0-zentrische Schätzung (pYIN/CREPE) versagt in rein instrumentalen,
dichten oder perkussiven Passagen. Dieser Schätzer (Capstan-Prinzip)
vergleicht jedes Frame-Spektrum mit einer robusten Referenz in der
Log-Frequenz-Domäne: eine Verschiebung in log f entspricht direkt dem
Warp-Verhältnis des Frames — ohne jegliche F0-Schätzung.

Deterministisch, rein numpy.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_N_FFT = 4096
_HOP = 1024


def spectral_warp_estimate(
    audio: np.ndarray,
    sr: int,
    n_fft: int = _N_FFT,
    hop: int = _HOP,
    ref_quantile: float = 0.5,
    max_shift_bins: int = 40,
    min_freq_hz: float = 100.0,
    max_freq_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Schätzt die Warp-Ratio je Frame (log-f-Spektral-Korrelation vs. Referenz).

    Returns:
        (times_s, warp_ratio, corr_quality) —
        warp_ratio ≈ 1.0 für unwarped, >1 = Frame ist schneller (höher).
        corr_quality ∈ [0, 1]: Peak-Steigung der Kreuzkorrelation — niedrig
        bei Rauschen/unstrukturiertem Spektrum (fürs Konsens-Gate).
    """
    audio_m = np.asarray(audio, dtype=np.float32).ravel()
    if len(audio_m) < n_fft:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = max(1, (len(audio_m) - n_fft) // hop + 1)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = audio_m[idx] * win
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)).astype(np.float64)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    lo_mask = freqs >= min_freq_hz
    hi_mask = freqs < (max_freq_hz if max_freq_hz is not None else sr / 2.0)
    band = lo_mask & hi_mask
    log_f = np.log(freqs[band] + 1e-9)
    n_bins = int(np.count_nonzero(band))
    if n_bins < 64:
        return np.zeros(n_frames), np.zeros(n_frames), np.zeros(n_frames)

    # Log-Frequenz-Raster mit Interpolation (kubisch) auf gleichmäßiges Gitter.
    log_edges = np.linspace(log_f[0], log_f[-1], n_bins)
    log_spec = np.zeros((n_frames, n_bins), dtype=np.float64)
    for f in range(n_frames):
        log_spec[f] = np.interp(log_edges, log_f, np.log(spec[f, band] + 1e-9))

    # Robuste Referenz: Median-Spektrum über die ruhigsten (kohärenten) Frames.
    ref = np.median(log_spec, axis=0)

    warp = np.ones(n_frames, dtype=np.float64)
    quality = np.zeros(n_frames, dtype=np.float64)
    shifts = np.arange(-max_shift_bins, max_shift_bins + 1)
    for f in range(n_frames):
        frame_s = log_spec[f] - np.median(log_spec[f])
        ref_s = ref - np.median(ref)
        # Effiziente Verschiebungs-Korrelation (Norm aus den verschobenen Slices).
        best = 0.0
        best_shift = 0
        for sh in shifts:
            if sh < 0:
                a, b = frame_s[-sh:], ref_s[:sh]
            elif sh > 0:
                a, b = frame_s[:-sh], ref_s[sh:]
            else:
                a, b = frame_s, ref_s
            if len(a) < 64:
                continue
            denom_c = np.sqrt(np.sum(a**2) * np.sum(b**2)) + 1e-12
            c = float(np.dot(a, b) / denom_c)
            if c > best:
                best, best_shift = c, int(sh)
        # Shift in log-f-Bins → Warp-Ratio: log f_shift = log(ratio).
        dlog = (log_edges[-1] - log_edges[0]) / max(n_bins - 1, 1)
        warp[f] = float(np.exp(-best_shift * dlog))
        quality[f] = float(np.clip(best, 0.0, 1.0))

    times = (np.arange(n_frames) * hop + n_fft / 2.0) / sr
    return times, warp, quality


def consensus_warp(
    traj_a: np.ndarray,
    times_a: np.ndarray | None,
    traj_b: np.ndarray,
    times_b: np.ndarray | None,
    quality_b: np.ndarray | None = None,
    tol: float = 0.005,
    min_quality: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Konsens-Gate zweier Warp-Trajektorien (pYIN vs. Spektral-Schätzer).

    Nimmt traj_a (F0-basiert) als Referenzraster; traj_b (Spektral) wird auf
    dessen Raster interpoliert. Wo beide übereinstimmen (|Δ| ≤ tol) und der
    Spektral-Schätzer Vertrauen hat (quality ≥ min_quality), wird der Mittel-
    wert genommen; sonst bleibt traj_a. Liefert zusätzlich die agreement-Maske.

    Returns:
        (consolidated_trajectory, agreement_mask_bool)
    """
    a = np.asarray(traj_a, dtype=np.float64).ravel()
    b = np.asarray(traj_b, dtype=np.float64).ravel()
    if times_a is None:
        t_a = np.arange(len(a), dtype=np.float64)
    else:
        t_a = np.asarray(times_a, dtype=np.float64).ravel()
    if times_b is None:
        t_b = np.arange(len(b), dtype=np.float64)
    else:
        t_b = np.asarray(times_b, dtype=np.float64).ravel()

    b_on_a = np.interp(t_a, t_b, b)
    if quality_b is not None:
        q = np.asarray(quality_b, dtype=np.float64).ravel()
        q_on_a = np.interp(t_a, t_b, q)
    else:
        q_on_a = np.ones_like(t_a)

    agree = (np.abs(b_on_a - a) <= tol) & (q_on_a >= min_quality)
    consolidated = np.where(agree, 0.5 * (a + b_on_a), a)
    return consolidated.astype(np.float64), agree
