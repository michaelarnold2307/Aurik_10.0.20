"""
backend/core/micro_temporal_envelope_fidelity.py
Aurik 10.0.0 — Micro-Temporal Envelope Fidelity (MTEF)

Bridges the temporal gap between TFS preservation (<2 ms) and MDEM (400 ms)
by preserving envelope shape at syllable/note scale (10–100 ms).

Scientific basis:
    - Hilbert analytic envelope (Gabor 1946) — instantaneous amplitude
    - Syllable-rate modulation (Rosen 1992): 4–25 Hz modulation rate carries
      intelligibility and rhythmic information
    - Attack preservation (Grey & Gordon 1978): onset shape determines timbre
      perception at 10–50 ms scale

Algorithm:
    1. Compute Hilbert analytic signal → instantaneous amplitude envelope
    2. Multi-scale smoothing: 15 ms (attack detail), 40 ms (syllable), 80 ms (note)
    3. Per-scale Pearson correlation between original and restored
    4. Where correlation degrades, apply proportional morphing correction
    5. Cross-scale weighted fusion with safety-limited gain (±3 dB)

Comparable to: iZotope RX Breath Control envelope matching, CEDAR envelope
preservation in DNS Two.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert

logger = logging.getLogger(__name__)


# ── Scale definitions ─────────────────────────────────────────────────────────
# Each scale captures a different temporal granularity of envelope modulation.
# Window sizes in seconds; hop = window/2 for 50% overlap.
_SCALES: tuple[tuple[str, float, float], ...] = (
    # (name, window_s, weight_in_fusion)
    ("attack", 0.015, 0.40),  # 15 ms: attack transient detail
    ("syllable", 0.040, 0.35),  # 40 ms: syllable/phoneme rate
    ("note", 0.080, 0.25),  # 80 ms: note onset/decay
)

# Maximum gain correction per sample (in linear amplitude ratio).
# ±3 dB = factor 1.413 (upper) / 0.708 (lower).
_MAX_GAIN_LINEAR: float = 1.413
_MIN_GAIN_LINEAR: float = 0.708

# Minimum level below which we skip correction (silence / near-silence).
_SILENCE_THRESHOLD: float = 1e-6  # ~ -120 dBFS


@dataclass
class MTEFResult:
    """Result of MTEF envelope fidelity measurement/correction."""

    pearson_attack: float  # Correlation at 15 ms scale
    pearson_syllable: float  # Correlation at 40 ms scale
    pearson_note: float  # Correlation at 80 ms scale
    fidelity_score: float  # Weighted average [0, 1]
    max_gain_applied_db: float  # Largest correction applied (dB)
    corrected: bool  # Whether correction was applied


def _hilbert_envelope(audio: np.ndarray) -> np.ndarray:
    """Berechnet instantaneous amplitude envelope via Hilbert transform.

    Uses scipy.signal.hilbert on the analytic signal. For very long audio
    (>10 min at 48 kHz), processes in chunks to limit memory.
    """
    if audio.ndim == 2:
        audio = audio.mean(axis=-1) if audio.shape[-1] <= 2 else audio.mean(axis=0)

    n = len(audio)
    # Chunk processing for memory efficiency (max ~30 s chunks)
    max_chunk = 48000 * 30
    if n <= max_chunk:
        analytic = hilbert(audio.astype(np.float64))
        return np.abs(analytic).astype(np.float32)  # type: ignore[no-any-return]

    envelope = np.empty(n, dtype=np.float32)
    overlap = 4800  # 100 ms overlap for smooth stitching
    pos = 0
    while pos < n:
        end = min(pos + max_chunk, n)
        chunk_start = max(0, pos - overlap)
        chunk = audio[chunk_start:end].astype(np.float64)
        analytic_chunk = np.abs(hilbert(chunk)).astype(np.float32)

        if pos == 0:
            envelope[:end] = analytic_chunk[: end - chunk_start]
        else:
            # Crossfade in overlap region
            write_start = pos
            data_offset = pos - chunk_start
            envelope[write_start:end] = analytic_chunk[data_offset:]
        pos = end - overlap if end < n else n

    return envelope  # type: ignore[no-any-return]


def _smooth_envelope(env: np.ndarray, sr: int, window_s: float) -> np.ndarray:
    """Glättet envelope with a Hanning window of given duration."""
    win_samples = max(3, int(window_s * sr) | 1)  # ensure odd
    kernel = np.hanning(win_samples).astype(np.float32)
    kernel /= kernel.sum()
    # Use 'same' mode to keep alignment
    return np.convolve(env, kernel, mode="same").astype(np.float32)  # type: ignore[no-any-return]


def _frame_pearson(env_orig: np.ndarray, env_rest: np.ndarray, sr: int, window_s: float) -> float:
    """Berechnet Pearson correlation between two envelopes at a given scale."""
    # Smooth both envelopes at the target scale
    s_orig = _smooth_envelope(env_orig, sr, window_s)
    s_rest = _smooth_envelope(env_rest, sr, window_s)

    # Only consider frames above silence threshold
    mask = (s_orig > _SILENCE_THRESHOLD) | (s_rest > _SILENCE_THRESHOLD)
    if mask.sum() < 100:
        return 1.0  # not enough data to measure

    o = s_orig[mask].astype(np.float64)
    r = s_rest[mask].astype(np.float64)

    o_mean = o.mean()
    r_mean = r.mean()
    o_std = o.std()
    r_std = r.std()

    if o_std < 1e-10 or r_std < 1e-10:
        return 1.0  # constant signal → perfect correlation

    pearson = float(np.mean((o - o_mean) * (r - r_mean)) / (o_std * r_std))
    return max(-1.0, min(1.0, pearson))


def measure(
    original: np.ndarray,
    restored: np.ndarray,
    sr: int = 48000,
) -> MTEFResult:
    """Misst micro-temporal envelope fidelity between original and restored.

    Args:
        original: Original audio (mono or stereo).
        restored: Restored audio (same shape).
        sr: Sample rate (expected 48000 Hz).

    Returns:
        MTEFResult with per-scale Pearson correlations and weighted fidelity.
    """
    # Convert to mono if needed
    orig_mono = original.mean(axis=-1) if original.ndim == 2 else original.copy()
    rest_mono = restored.mean(axis=-1) if restored.ndim == 2 else restored.copy()

    # Align lengths
    min_len = min(len(orig_mono), len(rest_mono))
    orig_mono = orig_mono[:min_len]
    rest_mono = rest_mono[:min_len]

    # Compute Hilbert envelopes
    env_orig = _hilbert_envelope(orig_mono)
    env_rest = _hilbert_envelope(rest_mono)

    # Per-scale Pearson correlation
    correlations: dict[str, float] = {}
    for name, window_s, _weight in _SCALES:
        correlations[name] = _frame_pearson(env_orig, env_rest, sr, window_s)

    # Weighted fidelity score
    fidelity = sum(correlations[name] * weight for name, _ws, weight in _SCALES)
    fidelity = max(0.0, min(1.0, fidelity))

    return MTEFResult(
        pearson_attack=correlations["attack"],
        pearson_syllable=correlations["syllable"],
        pearson_note=correlations["note"],
        fidelity_score=fidelity,
        max_gain_applied_db=0.0,
        corrected=False,
    )


def morph(
    original: np.ndarray,
    restored: np.ndarray,
    sr: int = 48000,
    *,
    mode: str = "restoration",
) -> tuple[np.ndarray, MTEFResult]:
    """Misst and correct micro-temporal envelope fidelity.

    When fidelity drops below threshold, applies proportional envelope
    morphing at each scale to recover the original's temporal dynamics.

    Args:
        original: Original audio (mono or stereo).
        restored: Restored audio (same shape).
        sr: Sample rate.
        mode: 'restoration' (conservative ±2 dB) or 'studio' (±3 dB).

    Returns:
        (corrected_audio, MTEFResult)
    """
    # Mode-adaptive gain limit
    max_db = 2.0 if mode == "restoration" else 3.0
    max_gain = 10.0 ** (max_db / 20.0)
    min_gain = 10.0 ** (-max_db / 20.0)

    # Measure first
    result = measure(original, restored, sr)

    # Skip correction if fidelity is already good
    if result.fidelity_score >= 0.95:
        return restored.copy(), result

    is_stereo = restored.ndim == 2

    # Work in mono for envelope computation, apply gain to all channels
    orig_mono = original.mean(axis=-1) if is_stereo else original.copy()
    rest_mono = restored.mean(axis=-1) if is_stereo else restored.copy()

    min_len = min(len(orig_mono), len(rest_mono))
    orig_mono = orig_mono[:min_len]
    rest_mono = rest_mono[:min_len]

    env_orig = _hilbert_envelope(orig_mono)
    env_rest = _hilbert_envelope(rest_mono)

    # Multi-scale gain computation
    gain = np.ones(min_len, dtype=np.float32)

    for name, window_s, weight in _SCALES:
        s_orig = _smooth_envelope(env_orig, sr, window_s)
        s_rest = _smooth_envelope(env_rest, sr, window_s)

        # Compute per-sample correction ratio
        ratio = np.ones(min_len, dtype=np.float32)
        active = s_rest > _SILENCE_THRESHOLD
        ratio[active] = s_orig[active] / (s_rest[active] + 1e-10)

        # Soften: blend toward unity proportional to how bad this scale is
        scale_corr = _frame_pearson(env_orig, env_rest, sr, window_s)
        blend = max(0.0, 1.0 - scale_corr)  # 0=perfect, 1=zero correlation
        blend = min(blend, 0.5)  # cap at 50% correction per scale

        ratio_softened = 1.0 + blend * (ratio - 1.0)

        # Apply weighted contribution
        gain *= ratio_softened**weight

    # Safety clamp gain
    gain = np.clip(gain, min_gain, max_gain)

    # Smooth gain curve to prevent click artifacts (2 ms window)
    smooth_win = max(3, int(0.002 * sr) | 1)
    kernel = np.hanning(smooth_win).astype(np.float32)
    kernel /= kernel.sum()
    gain = np.convolve(gain, kernel, mode="same").astype(np.float32)
    gain = np.clip(gain, min_gain, max_gain)

    # Apply gain to audio
    output = restored.copy()
    if is_stereo:
        for ch in range(min(output.shape[-1], 2)):
            output[:min_len, ch] *= gain
    else:
        output[:min_len] *= gain

    # Final clip & NaN guard
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    output = np.clip(output, -1.0, 1.0)

    # Measure post-correction fidelity
    post_result = measure(original, output, sr)
    max_gain_db = float(20.0 * np.log10(np.max(np.abs(gain)) + 1e-10))

    return output, MTEFResult(
        pearson_attack=post_result.pearson_attack,
        pearson_syllable=post_result.pearson_syllable,
        pearson_note=post_result.pearson_note,
        fidelity_score=post_result.fidelity_score,
        max_gain_applied_db=max_gain_db,
        corrected=True,
    )


# ── Singleton Pattern ─────────────────────────────────────────────────────────


class MicroTemporalEnvelopeFidelity:
    """Singleton-Wrapper für MTEF-Mess-/Morph-Operationen."""

    def measure(self, original: np.ndarray, restored: np.ndarray, sr: int = 48000) -> MTEFResult:
        return measure(original, restored, sr)

    def morph(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int = 48000,
        *,
        mode: str = "restoration",
    ) -> tuple[np.ndarray, MTEFResult]:
        return morph(original, restored, sr, mode=mode)


_instance: MicroTemporalEnvelopeFidelity | None = None
_lock = threading.Lock()


def get_mtef() -> MicroTemporalEnvelopeFidelity:
    """Thread-safe singleton access to MTEF."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MicroTemporalEnvelopeFidelity()
    return _instance
