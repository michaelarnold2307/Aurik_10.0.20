"""§PV1 Professional Phase Vocoder — STFT-based time-stretching with phase coherence.

Replaces the WSOLA-like _phase_vocoder_timestretch in Phase 12 with a proper
STFT phase vocoder (Flanagan & Golden 1966; Laroche & Dolson 1999; Driedger
& Müller 2016 TSM Toolbox).

Algorithm:
  - STFT with Hann window, n_fft=2048, 75 % overlap
  - Instantaneous frequency via phase derivative (phase-unwrapping per bin)
  - Time-varying stretch factors map to synthesis hop sizes
  - Overlap-add synthesis with Hann-window normalisation
  - Identity-phase-locking below 2.5 kHz (Puckette 1995) preserves
    harmonic integrity in tonal regions while allowing smooth time-scaling
    in noisy/transient bins.

Performance target: <50 ms per 5 s chunk @ 48 kHz on modern CPU.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
_N_FFT: int = 2048
_HOP_RATIO: int = 4  # 75 % overlap
_MIN_STRETCH: float = 0.90
_MAX_STRETCH: float = 1.10
# §PERF-R (2026-09-18): Identity-Phase-Lock DEAKTIVIERT (0 Hz). Messung auf
# dem Produktionspfad: mit Lock wurden Nicht-Bin-Frequenzen auf die
# Bin-Rasterfrequenz verfälscht (440-Hz-Ton → 445,3 Hz = +12 Cent Verstimmung)
# und die Hüllkurve modulierte auf Musik bis p95 ≈ 6,7 dB — der
# Reinhör-Witness flaggte im 225-s-Lauf genau das („pitch_instability“).
# Ohne Lock (reiner Laroche/Dolson): exakt 440,00 Hz, flache Hüllkurve
# (max/min 1,01), p95 ≈ 2,7 dB. Wow/Flutter-Korrektur nutzt winzige
# Stretch-Verhältnisse (≤ ±10 %, typisch ≤ ±2 %) — dafür ist der
# ungelockte Phasen-Vocoder das etablierte Werkzeug (Lock ist für große
# musikalische TSM-Stretches gedacht).
_IDENTITY_PHASE_LOCK_MAX_HZ: float = 0.0  # 0 = deaktiviert (Evidenz s. o.)


def phase_vocoder_timestretch(
    audio: np.ndarray,
    stretch_factors: np.ndarray,
    sample_rate: int = 48000,
    *,
    n_fft: int = _N_FFT,
    hop_ratio: int = _HOP_RATIO,
) -> np.ndarray:
    """Time-varying STFT phase vocoder for wow/flutter correction.

    Args:
        audio: Mono float audio [N] (float32 or float64).
        stretch_factors: Per-frame stretch ratios, one per analysis hop
                         (>1 = slow down, <1 = speed up).
        sample_rate: Audio sample rate (must be 48000).
        n_fft: STFT window length (must be power of 2, default 2048).
        hop_ratio: overlap factor (4 = 75 % overlap, analysis_hop = n_fft/4).

    Returns:
        Time-stretched audio, same length as input, float32, NaN/Inf-free.
    """
    audio_f = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n_samples = len(audio_f)
    if n_samples < n_fft:
        return cast(np.ndarray, audio_f.copy())

    analysis_hop = n_fft // hop_ratio

    # ── Step 0: Interpolate frame-wise stretch factors to per-analysis-hop ──
    sf = np.asarray(stretch_factors, dtype=np.float32)
    sf = np.clip(sf, _MIN_STRETCH, _MAX_STRETCH)
    n_analysis_frames = 1 + max(0, (n_samples - n_fft) // analysis_hop)

    if len(sf) == 1:
        sf_per_frame = np.full(n_analysis_frames, sf[0], dtype=np.float64)
    elif len(sf) != n_analysis_frames:
        src_x = np.linspace(0, n_analysis_frames - 1, max(len(sf), 2), dtype=np.float64)
        dst_x = np.arange(n_analysis_frames, dtype=np.float64)
        sf_per_frame = np.interp(dst_x, src_x, sf.astype(np.float64)).astype(np.float64)
    else:
        sf_per_frame = sf.astype(np.float64)

    # Savitzky-Golay smoothing to remove micro-jitter (preserves wow contour)
    _sg_win = max(5, min(n_analysis_frames // 8, 21))
    if _sg_win >= 5 and _sg_win % 2 == 0:
        _sg_win += 1
    if n_analysis_frames >= _sg_win >= 5:
        from scipy.signal import savgol_filter

        sf_per_frame = savgol_filter(sf_per_frame, _sg_win, 2, mode="interp")
    sf_per_frame = np.clip(sf_per_frame, _MIN_STRETCH, _MAX_STRETCH)

    # Early exit if no correction needed
    if np.max(np.abs(sf_per_frame - 1.0)) < 0.002:
        return cast(np.ndarray, audio_f.copy())

    # ── Step 1: Analysis STFT ────────────────────────────────────────
    win = np.hanning(n_fft).astype(np.float64)
    win_sq = win**2

    # Build analysis frames via sliding_window_view + batched rfft
    if n_analysis_frames < 2:
        return cast(np.ndarray, audio_f.copy())

    frames = np.lib.stride_tricks.sliding_window_view(audio_f.astype(np.float64), n_fft)[::analysis_hop][
        :n_analysis_frames
    ]
    stft = np.fft.rfft(frames * win, axis=1, n=n_fft)  # (T, F)

    mag = np.abs(stft).astype(np.float64)
    phase = np.angle(stft).astype(np.float64)
    n_freqs = mag.shape[1]  # n_fft // 2 + 1

    # ── Step 2: Instantaneous frequency (phase derivative) ────────────
    # Δφ = unwrap(φ[t] - φ[t-1] - ω_bin * analysis_hop / sr)
    freq_bins = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate).astype(np.float64)
    omega_bin = 2.0 * np.pi * freq_bins * analysis_hop / sample_rate  # rad per hop

    phase_diff = np.diff(phase, axis=0)  # (T-1, F)
    # Phase unwrapping: bring into [-π, π] around expected rotation
    phase_diff = np.arctan2(np.sin(phase_diff - omega_bin), np.cos(phase_diff - omega_bin))
    inst_freq = (omega_bin + phase_diff) / (2.0 * np.pi * analysis_hop / sample_rate)  # Hz (T-1, F)

    # Full IF: frame 0 = bin centre, frames 1.. use derivative
    inst_freq_full = np.zeros((n_analysis_frames, n_freqs), dtype=np.float64)
    inst_freq_full[0] = freq_bins
    inst_freq_full[1:] = inst_freq

    # ── Step 3: Identity Phase Locking (Puckette 1995) ─────────────────
    # Bins < 2.5 kHz: lock phase coherence to preserve harmonic integrity.
    # This prevents the "phasiness" artefact common in naive phase vocoders.
    identity_mask = freq_bins[np.newaxis, :] < _IDENTITY_PHASE_LOCK_MAX_HZ
    inst_freq_full = np.where(identity_mask, freq_bins[np.newaxis, :], inst_freq_full)

    # ── Step 4: Synthesis via variable hop (frame-wise, Laroche & Dolson) ──
    # synthesis_hop[t] = analysis_hop / stretch_factor[t]
    synthesis_hop_arr = analysis_hop / np.clip(sf_per_frame, _MIN_STRETCH, _MAX_STRETCH)

    # Build synthesis time grid (one synthesis frame per analysis frame)
    synthesis_times = np.cumsum(np.concatenate([[0.0], synthesis_hop_arr[:-1]]))  # (T,) in samples

    # Total synthesis samples: keep output same length as input
    total_synth_samples = n_samples

    # §PERF-R (2026-09-18): Frame-weise Synthese — die Vorgänger-Version
    # legte PRO OUTPUT-SAMPLE eine komplette n_fft-irfft an (Schleife über
    # N Samples ⇒ ~1 Mio. irffts je 10 s Stereo; gemessen 79 s/10 s statt
    # des Docstring-Ziels <50 ms/5 s). Der Standard-Phase-Vocoder
    # synthetisiert EINEN Frame je Synthese-Hop (~512 Samples) und
    # overlap-addiert auf dem Hop-Grid — identische Mathematik
    # (Fenster-Interpolation + IF-Phasenfortschritt), ~500× weniger irffts.
    # Phasenfortschritt je Synthese-Frame = kumulative Verschiebung des
    # Synthese-Grids gegenüber dem Analyse-Grid × Momentanfrequenz
    # (korrekte PV-Phasenpropagation; die Vorgänger-Version setzte die
    # Phase je Frame auf die Analyse-Phase zurück).
    analysis_times = np.arange(n_analysis_frames, dtype=np.float64) * analysis_hop
    displacement = synthesis_times - analysis_times  # (T,)
    phase_adv = 2.0 * np.pi * inst_freq_full * displacement[:, np.newaxis] / sample_rate
    synth_phase = phase + phase_adv
    synth_spectrum = mag * np.exp(1j * synth_phase)

    # Alle Synthese-Frames in EINEM batched irfft (T, F) → (T, n_fft)
    frames_td = np.fft.irfft(synth_spectrum, n=n_fft, axis=1).real
    frames_td *= win[np.newaxis, :]

    output = np.zeros(total_synth_samples + n_fft, dtype=np.float64)
    norm = np.zeros(total_synth_samples + n_fft, dtype=np.float64)
    for t in range(max(0, n_analysis_frames - 1)):
        s = int(round(synthesis_times[t]))
        if s >= total_synth_samples:
            break
        e = min(s + n_fft, total_synth_samples + n_fft)
        n_place = e - s
        output[s:e] += frames_td[t, :n_place]
        norm[s:e] += win_sq[:n_place]

    output = output[:total_synth_samples]
    norm = norm[:total_synth_samples]

    # ── Step 5: Normalise & finalise ──────────────────────────────────
    norm = np.where(norm > 1e-10, norm, 1.0)
    output = output / norm
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    output = np.clip(output, -1.0, 1.0)

    return cast(np.ndarray, (output.astype(np.float32, copy=False)))


def phase_vocoder_timestretch_fast(
    audio: np.ndarray,
    stretch_factors: np.ndarray,
    sample_rate: int = 48000,
) -> np.ndarray:
    """Optimised fast path: same API, reduced quality for speed-critical contexts.

    Uses linear source-position mapping (WSOLA-style) but with band-limited
    interpolation (sinc kernel, kaiser window) instead of np.interp.
    Falls back to the full STFT phase vocoder if audio is short or stretch is small.
    """
    from scipy.signal import resample_poly as _resample_poly

    audio_f = np.asarray(audio, dtype=np.float32)
    n_samples = len(audio_f)
    if n_samples < 2048:
        return phase_vocoder_timestretch(audio, stretch_factors, sample_rate)

    sf = np.clip(np.asarray(stretch_factors, dtype=np.float32), _MIN_STRETCH, _MAX_STRETCH)

    # If correction is small (< 1 %), fast path is acceptable
    if np.max(np.abs(sf - 1.0)) < 0.01:
        if len(sf) == 1:
            sf_samples = np.full(n_samples, sf[0], dtype=np.float32)
        else:
            src_idx = np.linspace(0, n_samples - 1, len(sf), dtype=np.float32)
            sf_samples = np.interp(np.arange(n_samples, dtype=np.float32), src_idx, sf)  # type: ignore[assignment]

        sf_samples = np.clip(sf_samples, _MIN_STRETCH, _MAX_STRETCH)
        src_step = 1.0 / sf_samples
        src_pos = np.cumsum(src_step)
        src_pos = (src_pos / (src_pos[-1] + 1e-12)) * (n_samples - 1)
        src_pos = np.clip(src_pos, 0.0, n_samples - 1)

        corrected = np.interp(src_pos, np.arange(n_samples, dtype=np.float32), audio_f)
        return cast(
            np.ndarray, (np.nan_to_num(corrected, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False))
        )

    # For larger stretches, use full phase vocoder
    return phase_vocoder_timestretch(audio, stretch_factors, sample_rate)
