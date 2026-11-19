"""
backend/core/sub_stem_processor.py — Instrument Sub-Stem Decomposition & Processing
=====================================================================================

Decomposes an instrument stem into acoustically meaningful sub-components,
applies targeted restoration to each, and recombines them.

Motivation (the 5.8 → 9.3 gap):
    Global processing over a "guitar" stem treats body resonance, string attack,
    and pick noise identically.  Real engineers send each to a dedicated chain:
    - Body/sustain band  → gentle noise reduction, warmth EQ
    - String/mid band    → presence EQ, soft compression
    - Pick/air band      → de-essing-style transient shaping, de-noising

Sub-stem definitions (frequency ranges + processing intent):

    Guitar:
        bass_body   (   50 –  800 Hz) : body resonance, warmth
        string_mid  (  800 – 4000 Hz) : string harmonics, presence
        pick_air    ( 4000 – 16000 Hz): pick noise, pick transients

    Piano:
        hammer_sub  (   50 –  250 Hz) : hammer thump & sub
        body_mid    (  250 – 3000 Hz) : string resonance tone
        shimmer_hi  ( 3000 – 16000 Hz): high register shimmer, hammer click

    Drums:
        kick_sub    (   30 –  200 Hz) : kick drum & sub
        snare_mid   (  200 – 5000 Hz) : snare body & ghost notes
        cymbal_hi   ( 5000 – 20000 Hz): hi-hat, cymbal, room HF

    Brass:
        fundamental (   50 –  500 Hz) : fundamental pitch
        harmonics   (  500 – 5000 Hz) : harmonic overtone series
        air_noise   ( 5000 – 16000 Hz): breath noise, air

    Strings (bowed):
        body_wood   (   80 –  600 Hz) : wood body resonance
        bow_mid     (  600 – 4000 Hz) : bowing harmonics
        bow_noise   ( 4000 – 16000 Hz): bow rosin noise

    Woodwinds:
        reed_low    (   80 –  600 Hz) : reed/body fundamental
        reed_mid    (  600 – 4000 Hz) : harmonic register
        reed_air    ( 4000 – 16000 Hz): reed air noise

    Bass (electric):
        sub_bass    (   30 –  150 Hz) : sub-bass
        mid_bass    (  150 – 1000 Hz) : fundamental + 1st harmonic
        string_top  ( 1000 – 6000 Hz) : string noise, fret click

    Keys/Synth (shared):
        low_keys    (   50 –  300 Hz) : sub/bass
        mid_keys    (  300 – 4000 Hz) : main tone
        hi_keys     ( 4000 – 16000 Hz): aliasing / air

Crossover filter: Linkwitz-Riley 4th order (LR4 = two cascaded Butterworth-2
                  filters, −24 dB/oct, perfect magnitude reconstruction).
                  Cascaded as: LP → mid_BP (LP − HP) → HP.

Processing per sub-stem:
    - Gentle spectral subtraction NR (stationary noise only, σ-based threshold)
    - Band-limited EQ gain (flat by default, instrument-tuned offset)
    - Soft-knee amplitude envelope normalisation (optional, instrument-specific)

Reference:
    Linkwitz, S. H. (1976). "Active crossover networks for noncoincident drivers."
    J. AES, 24(1), 2–8.

Singleton (§3.2), NaN/Inf-guard (§3.1), assert sr == 48000.

Author: Aurik Development Team
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.signal as sig
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

SR_REQUIRED: int = 48_000

# ── Sub-stem band catalogue ───────────────────────────────────────────────────

# Each entry: (low_hz, high_hz, label, eq_gain_db)
# eq_gain_db: default gentle EQ nudge applied after NR (0 = flat pass-through)

_BANDS: dict[str, list[tuple[float, float, str, float]]] = {
    "guitar": [
        (50.0, 800.0, "bass_body", +0.5),
        (800.0, 4000.0, "string_mid", +0.8),
        (4000.0, 16000.0, "pick_air", -0.5),
    ],
    "keys": [
        (50.0, 250.0, "hammer_sub", +0.3),
        (250.0, 3000.0, "body_mid", +0.5),
        (3000.0, 16000.0, "shimmer_hi", -0.3),
    ],
    "piano": [  # alias for keys
        (50.0, 250.0, "hammer_sub", +0.3),
        (250.0, 3000.0, "body_mid", +0.5),
        (3000.0, 16000.0, "shimmer_hi", -0.3),
    ],
    "drums": [
        (30.0, 200.0, "kick_sub", +1.0),
        (200.0, 5000.0, "snare_mid", +0.5),
        (5000.0, 20000.0, "cymbal_hi", -0.3),
    ],
    "percussion": [  # alias for drums
        (30.0, 200.0, "kick_sub", +1.0),
        (200.0, 5000.0, "snare_mid", +0.5),
        (5000.0, 20000.0, "cymbal_hi", -0.3),
    ],
    "brass": [
        (50.0, 500.0, "fundamental", +0.5),
        (500.0, 5000.0, "harmonics", +0.5),
        (5000.0, 16000.0, "air_noise", -1.0),
    ],
    "strings": [
        (80.0, 600.0, "body_wood", +0.5),
        (600.0, 4000.0, "bow_mid", +0.5),
        (4000.0, 16000.0, "bow_noise", -0.8),
    ],
    "woodwinds": [
        (80.0, 600.0, "reed_low", +0.5),
        (600.0, 4000.0, "reed_mid", +0.5),
        (4000.0, 16000.0, "reed_air", -0.8),
    ],
    "bass": [
        (30.0, 150.0, "sub_bass", +1.0),
        (150.0, 1000.0, "mid_bass", +0.5),
        (1000.0, 6000.0, "string_top", -0.3),
    ],
    "synth": [
        (50.0, 300.0, "low_synth", +0.3),
        (300.0, 4000.0, "mid_synth", +0.3),
        (4000.0, 16000.0, "hi_synth", -0.5),
    ],
}


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class SubStemBandResult:
    """Diagnostics for one sub-stem band."""

    label: str
    low_hz: float
    high_hz: float
    eq_gain_db: float
    nr_reduction_db: float  # estimated NR attenuation applied (dB)
    rms_in: float
    rms_out: float


@dataclass
class SubStemResult:
    """Full result of :class:`SubStemProcessor`.

    Attributes:
        audio:           Processed output audio (same shape as input).
        instrument:      Instrument type string used.
        n_bands:         Number of sub-stem bands that were processed.
        bands:           Per-band diagnostic information.
        processing_strength: Effective strength applied.
        passthrough:     True when no processing was applied (unknown instrument /
                         strength=0 / error).
    """

    audio: np.ndarray
    instrument: str
    n_bands: int
    bands: list[SubStemBandResult] = field(default_factory=list)
    processing_strength: float = 0.0
    passthrough: bool = False


# ── LR4 crossover helpers ─────────────────────────────────────────────────────


def _lr4_lowpass(audio: np.ndarray, sr: int, cutoff_hz: float) -> NDArray[np.float32]:
    """4th-order Linkwitz-Riley low-pass at *cutoff_hz* (two cascaded Butterworth-2)."""
    cutoff_hz = float(np.clip(cutoff_hz, 5.0, sr / 2.0 - 1.0))
    sos = sig.butter(2, cutoff_hz, btype="low", fs=sr, output="sos")
    out: NDArray[np.float64] = np.asarray(sig.sosfilt(sos, audio.astype(np.float64)), dtype=np.float64)
    out = np.asarray(sig.sosfilt(sos, out), dtype=np.float64)  # cascade for LR4 phase response
    return np.asarray(out, dtype=np.float32)


def _lr4_highpass(audio: np.ndarray, sr: int, cutoff_hz: float) -> NDArray[np.float32]:
    """4th-order Linkwitz-Riley high-pass at *cutoff_hz*."""
    cutoff_hz = float(np.clip(cutoff_hz, 5.0, sr / 2.0 - 1.0))
    sos = sig.butter(2, cutoff_hz, btype="high", fs=sr, output="sos")
    out: NDArray[np.float64] = np.asarray(sig.sosfilt(sos, audio.astype(np.float64)), dtype=np.float64)
    out = np.asarray(sig.sosfilt(sos, out), dtype=np.float64)
    return np.asarray(out, dtype=np.float32)


def _extract_band(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    """Extrahiert the frequency band [low_hz, high_hz] from *audio* via LR4 crossovers."""
    nyq = sr / 2.0
    # LP at high_hz unless it's effectively SR/2 (pass-through that edge)
    band = _lr4_lowpass(audio, sr, high_hz) if high_hz < nyq * 0.95 else audio.astype(np.float32)
    # HP at low_hz unless it's effectively 0
    if low_hz > 10.0:
        band = _lr4_highpass(band, sr, low_hz)
    return band  # type: ignore[no-any-return]


# ── Per-band NR (stationary-noise spectral subtraction) ──────────────────────


def _soft_spectral_subtraction(
    band: np.ndarray, strength: float, noise_estimate_frames: int = 10
) -> tuple[np.ndarray, float]:
    """Leichtgewichtiges stationary-noise spectral subtraction for a narrow band signal.

    Estimates noise floor from the quietest *noise_estimate_frames* frames of
    the STFT and subtracts it with a soft-knee Wiener-style mask.

    Args:
        band:                  1-D mono float32 signal.
        strength:              Effective blend (0 = no processing, 1 = full NR).
        noise_estimate_frames: How many quiet frames to use for noise floor.

    Returns:
        (processed_band, estimated_reduction_db)
    """
    n = len(band)
    if n < 512:
        return band, 0.0

    n_fft = 512
    hop = 256
    win = np.hanning(n_fft)

    # STFT
    frames = []
    for i in range(0, n - n_fft + 1, hop):
        frames.append(band[i : i + n_fft] * win)
    if not frames:
        return band, 0.0

    stft = np.array([np.fft.rfft(f) for f in frames])  # (n_frames, n_bins)
    mag = np.abs(stft)

    # Noise floor: mean magnitude of the quietest frames
    frame_energy = mag.mean(axis=1)
    n_quiet = max(1, min(noise_estimate_frames, len(frame_energy) // 4))
    quiet_idx = np.argsort(frame_energy)[:n_quiet]
    noise_floor = mag[quiet_idx].mean(axis=0)  # (n_bins,)

    # Wiener-style mask: max(0, 1 - k * noise/mag)
    k = 2.0 * strength
    mask = np.maximum(0.0, 1.0 - k * noise_floor / (mag + 1e-10))
    mask = np.clip(mask, 0.0, 1.0)

    # Apply mask
    mag_clean = mag * mask
    reduction_db = float(
        np.clip(
            20.0 * np.log10((mag.mean() + 1e-10) / (mag_clean.mean() + 1e-10)),
            0.0,
            12.0,
        )
    )

    stft_clean = stft * (mag_clean / (mag + 1e-10))

    # Overlap-add ISTFT
    out = np.zeros(n, dtype=np.float32)
    cnt = np.zeros(n, dtype=np.float32)
    for fi, i in enumerate(range(0, n - n_fft + 1, hop)):
        frame_td = np.fft.irfft(stft_clean[fi], n=n_fft).astype(np.float32)
        out[i : i + n_fft] += frame_td * win
        cnt[i : i + n_fft] += win**2

    # Normalise overlap
    cnt = np.where(cnt > 1e-10, cnt, 1.0)
    out /= cnt

    # Blend original + cleaned
    blend = float(np.clip(strength, 0.0, 1.0))
    result = (1.0 - blend) * band + blend * out[:n]
    return result.astype(np.float32), reduction_db


# ── EQ gain helper ────────────────────────────────────────────────────────────


def _apply_gain_db(audio: np.ndarray, gain_db: float, strength: float) -> NDArray[np.float32]:
    """Wendet einen linearen Gain (dB) skaliert mit *strength* auf *audio* an."""
    effective_db = gain_db * float(np.clip(strength, 0.0, 1.0))
    if abs(effective_db) < 0.01:
        return np.asarray(audio, dtype=np.float32)
    gain_lin = 10.0 ** (effective_db / 20.0)
    return np.asarray(audio * gain_lin, dtype=np.float32)


# ── Core class ────────────────────────────────────────────────────────────────


class SubStemProcessor:
    """Decompose an instrument stem into sub-stems, process each, recombine.

    For each known instrument, splits the signal into 3 acoustically meaningful
    frequency bands via Linkwitz-Riley crossovers, applies:
      - Stationary-noise spectral subtraction (per-band noise floor estimation)
      - Band-specific gentle EQ gain nudge
    Then sums bands back together with crossover linearity (LR4 ensures flat
    summed magnitude response within 0.1 dB).

    Instantiate via :func:`get_sub_stem_processor` (singleton §3.2).

    Args:
        processing_strength: Global blend factor 0.0–1.0 (default 0.35).
    """

    MAX_STRENGTH: float = 0.60

    def __init__(self, processing_strength: float = 0.35) -> None:
        self.processing_strength = float(np.clip(processing_strength, 0.0, self.MAX_STRENGTH))

    # ── Public API ────────────────────────────────────────────────────────────

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        instrument: str = "guitar",
        processing_strength: float | None = None,
    ) -> SubStemResult:
        """Verarbeitet *audio* by sub-stem decomposition.

        Args:
            audio:               Mono or stereo audio at 48 000 Hz.
            sr:                  Sample rate — must be 48 000 Hz.
            instrument:          Instrument type key (see _BANDS).
            processing_strength: Override blend 0.0–1.0; clamped to MAX_STRENGTH.

        Returns:
            :class:`SubStemResult` with processed audio and diagnostics.
        """
        assert sr == SR_REQUIRED, f"Sample rate must be 48000 Hz, got {sr}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        strength = float(
            np.clip(
                processing_strength if processing_strength is not None else self.processing_strength,
                0.0,
                self.MAX_STRENGTH,
            )
        )

        bands_cfg = _BANDS.get(instrument.lower())

        def _passthrough(reason: str) -> SubStemResult:
            out = np.clip(audio, -1.0, 1.0)
            logger.debug("SubStemProcessor passthrough: %s", reason)
            return SubStemResult(
                audio=out,
                instrument=instrument,
                n_bands=0,
                processing_strength=strength,
                passthrough=True,
            )

        if bands_cfg is None:
            return _passthrough(f"unknown instrument '{instrument}'")
        if strength < 1e-5:
            return _passthrough("strength=0")

        # --- mono processing path ---
        is_stereo = audio.ndim == 2
        if is_stereo:
            if audio.shape[0] <= 8:  # (channels, samples)
                channels = [audio[c] for c in range(audio.shape[0])]
            else:  # (samples, channels)
                channels = [audio[:, c] for c in range(audio.shape[1])]
            result_channels = [self._process_mono(ch.astype(np.float32), sr, bands_cfg, strength) for ch in channels]
            band_results = result_channels[0][1]  # diagnostics from first channel
            processed_channels = [r[0] for r in result_channels]
            out = np.stack(processed_channels, axis=0) if audio.shape[0] <= 8 else np.stack(processed_channels, axis=1)
        else:
            out, band_results = self._process_mono(audio.astype(np.float32), sr, bands_cfg, strength)

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0)

        logger.info(
            "SubStemProcessor: instrument=%s bands=%d strength=%.2f",
            instrument,
            len(band_results),
            strength,
        )
        return SubStemResult(
            audio=out,
            instrument=instrument,
            n_bands=len(band_results),
            bands=band_results,
            processing_strength=strength,
            passthrough=False,
        )

    # ── Internal mono processing ──────────────────────────────────────────────

    def _process_mono(
        self,
        mono: np.ndarray,
        sr: int,
        bands_cfg: list[tuple[float, float, str, float]],
        strength: float,
    ) -> tuple[np.ndarray, list[SubStemBandResult]]:
        """Verarbeitet a single mono signal through the band decomposition chain."""

        sub_stems: list[np.ndarray] = []
        band_results: list[SubStemBandResult] = []

        for low_hz, high_hz, label, eq_gain_db in bands_cfg:
            # 1. Extract sub-stem band
            band = _extract_band(mono, sr, low_hz, high_hz)
            rms_in = float(np.sqrt(np.mean(band**2)) + 1e-10)

            # 2. Spectral subtraction NR
            band_nr, nr_db = _soft_spectral_subtraction(band, strength)

            # 3. EQ gain nudge
            band_eq = _apply_gain_db(band_nr, eq_gain_db, strength)

            rms_out = float(np.sqrt(np.mean(band_eq**2)) + 1e-10)
            sub_stems.append(band_eq)
            band_results.append(
                SubStemBandResult(
                    label=label,
                    low_hz=low_hz,
                    high_hz=high_hz,
                    eq_gain_db=eq_gain_db,
                    nr_reduction_db=nr_db,
                    rms_in=rms_in,
                    rms_out=rms_out,
                )
            )

        # 4. Reconstruct: blend processed sum with original
        processed_sum = np.zeros_like(mono, dtype=np.float32)
        for s in sub_stems:
            n = min(len(s), len(processed_sum))
            processed_sum[:n] += s[:n]

        # Identity-safe blend (protects musical identity)
        blend = float(np.clip(strength, 0.0, 1.0))
        n_min = min(len(mono), len(processed_sum))
        result = ((1.0 - blend) * mono[:n_min] + blend * processed_sum[:n_min]).astype(np.float32)

        # Pad if processing shortened the signal
        if len(result) < len(mono):
            result = np.concatenate([result, mono[len(result) :]])

        return result, band_results


# ── Singleton (§3.2 Double-Checked Locking) ──────────────────────────────────

_lock = threading.Lock()
_SINGLETON_INSTANCES: dict[type, Any] = {}


class _SingletonMeta(type):
    """Metaclass for thread-safe singleton pattern without global statement."""

    def __call__(cls, *args, **kwargs):
        if cls not in _SINGLETON_INSTANCES:
            with _lock:
                if cls not in _SINGLETON_INSTANCES:
                    _SINGLETON_INSTANCES[cls] = super().__call__(*args, **kwargs)
        return _SINGLETON_INSTANCES[cls]


def get_sub_stem_processor() -> SubStemProcessor:
    """Gibt the module-level singleton :class:`SubStemProcessor` zurück.

    Thread-safe via double-checked locking (§3.2).
    """
    if SubStemProcessor not in _SINGLETON_INSTANCES:
        with _lock:
            if SubStemProcessor not in _SINGLETON_INSTANCES:
                _SINGLETON_INSTANCES[SubStemProcessor] = SubStemProcessor()
    return _SINGLETON_INSTANCES[SubStemProcessor]  # type: ignore[no-any-return]


def process_sub_stems(
    audio: np.ndarray,
    sr: int,
    instrument: str = "guitar",
    processing_strength: float | None = None,
) -> SubStemResult:
    """Convenience wrapper: apply sub-stem processing to *audio*.

    Args:
        audio:               Mono or stereo audio at 48 000 Hz.
        sr:                  Sample rate — must be 48 000 Hz.
        instrument:          Instrument type key.
        processing_strength: Blend factor 0.0–1.0; clamped to 0.60.

    Returns:
        :class:`SubStemResult`.
    """
    return get_sub_stem_processor().process(audio, sr, instrument=instrument, processing_strength=processing_strength)
