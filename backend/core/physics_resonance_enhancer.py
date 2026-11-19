"""
backend/core/physics_resonance_enhancer.py — Physics-Inspired Instrument Body Resonance
=========================================================================================

Adds instrument-body resonance coloration using cascaded biquad peak-EQ filters
whose parameters are derived from measured acoustic literature data.

Motivation:
    After noise reduction, stem separation and drift correction, the processed
    instrument signal is spectrally "flat" compared to the original recording.
    Real instrument bodies act as mechanical resonators: a classical guitar body
    has a dominant air resonance (Helmholtz) ~100–120 Hz and a top-plate
    resonance ~170–220 Hz; a piano soundboard peaks at ~60–90 Hz; a trumpet bell
    has bell-bore resonances at each harmonic partial.
    Reintroducing these physically motivated colorations restores timbral
    authenticity — the gap between a "clean but lifeless" restoration and a
    "warm and alive" one.

Per-instrument resonance models (Biquad Peak-EQ chains):

    Guitar (acoustic, classical / steel-string; McIntyre & Woodhouse 1978, Christensen 1982):
        Peak 1: f0 ≈ 102 Hz,  Q=8.0,  gain = +2.5 dB  (Helmholtz air resonance)
        Peak 2: f0 ≈ 195 Hz,  Q=6.0,  gain = +2.0 dB  (top-plate resonance T(1,1))
        Peak 3: f0 ≈ 400 Hz,  Q=4.0,  gain = +1.2 dB  (back-plate coupling)
        Peak 4: f0 ≈ 2500 Hz, Q=3.0,  gain = +0.8 dB  (upper-bout bridge resonance)

    Piano (Benade 1976, Fletcher & Rossing 1998):
        Peak 1: f0 ≈  75 Hz,  Q=5.0,  gain = +2.0 dB  (soundboard bass resonance)
        Peak 2: f0 ≈ 180 Hz,  Q=5.0,  gain = +1.5 dB  (mid-range board resonance)
        Peak 3: f0 ≈ 550 Hz,  Q=3.5,  gain = +1.0 dB  (bridge resonance)
        Peak 4: f0 ≈ 3000 Hz, Q=2.5,  gain = +0.6 dB  (soundboard high mode)

    Trumpet/Brass (Benade 1976, Hirschberg et al. 1994):
        Peak 1: f0 ≈ 233 Hz,  Q=10.0, gain = +2.5 dB  (bell bore 1st formant)
        Peak 2: f0 ≈ 466 Hz,  Q=8.0,  gain = +2.0 dB  (bell bore 2nd formant)
        Peak 3: f0 ≈ 932 Hz,  Q=6.0,  gain = +1.5 dB  (bell bore 3rd formant)
        Peak 4: f0 ≈ 1500 Hz, Q=4.0,  gain = +0.8 dB  (bell flare resonance)

    Drums/Kick (Rossing 2000 — Acoustics of Percussion):
        Peak 1: f0 ≈  60 Hz,  Q=4.0,  gain = +3.0 dB  (primary drumhead resonance)
        Peak 2: f0 ≈ 120 Hz,  Q=3.5,  gain = +1.5 dB  (shell resonance)
        Peak 3: f0 ≈ 250 Hz,  Q=3.0,  gain = +0.8 dB  (overtone ring)
        Peak 4: f0 ≈ 5000 Hz, Q=2.0,  gain = +0.5 dB  (beater click attack)

    Violin/Strings (Hutchins 1983 — Catgut Acoustical Society):
        Peak 1: f0 ≈ 290 Hz,  Q=10.0, gain = +3.0 dB  (air resonance A0)
        Peak 2: f0 ≈ 430 Hz,  Q=8.0,  gain = +2.5 dB  (top-plate resonance T1)
        Peak 3: f0 ≈ 560 Hz,  Q=6.0,  gain = +1.5 dB  (back-plate resonance B1)
        Peak 4: f0 ≈ 1800 Hz, Q=4.0,  gain = +0.8 dB  (bridge resonance)

    Woodwinds/Oboe (Nederveen 1969 — Acoustical Aspects of Woodwind Instruments):
        Peak 1: f0 ≈ 350 Hz,  Q=8.0,  gain = +2.0 dB  (bore resonance F1)
        Peak 2: f0 ≈ 700 Hz,  Q=6.0,  gain = +1.5 dB  (bore resonance F2)
        Peak 3: f0 ≈ 1200 Hz, Q=5.0,  gain = +1.0 dB  (bore resonance F3)
        Peak 4: f0 ≈ 2500 Hz, Q=3.0,  gain = +0.5 dB  (tone-hole register)

    Bass guitar (electric; Roberts 1990):
        Peak 1: f0 ≈  85 Hz,  Q=5.0,  gain = +2.5 dB  (body resonance)
        Peak 2: f0 ≈ 200 Hz,  Q=4.0,  gain = +1.5 dB  (neck+body coupling)
        Peak 3: f0 ≈ 600 Hz,  Q=3.5,  gain = +1.0 dB  (pickup resonance)
        Peak 4: f0 ≈ 1500 Hz, Q=2.5,  gain = +0.5 dB  (bridge/nut reflection)

    Synth (minimal body coloration — slight air/warmth):
        Peak 1: f0 ≈  80 Hz,  Q=2.0,  gain = +1.0 dB  (warmth sub)
        Peak 2: f0 ≈ 350 Hz,  Q=1.5,  gain = +0.5 dB  (body mid)
        Peak 3: f0 ≈ 3000 Hz, Q=1.5,  gain = +0.3 dB  (presence)
        Peak 4: f0 ≈ 8000 Hz, Q=1.0,  gain = +0.2 dB  (air)

Biquad implementation:
    Audio-EQ-Cookbook (Zölzer, 2008).  Peaking EQ:
        H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)
        A  = 10^(gain_dB/40)
        w0 = 2π·f0/fs
        α  = sin(w0)/(2·Q)
        b0 =  1 + α·A,   b1 = −2·cos(w0),  b2 =  1 − α·A
        a0 =  1 + α/A,   a1 = −2·cos(w0),  a2 =  1 − α/A

    All gains are scaled by `enhancement_strength` (0.0–1.0, default 0.40)
    to preserve musical identity.  Gain ceiling: 4.0 dB per peak.
    Cascaded filter order: linear convolution (scipy.signal.lfilter, sequential).

Singleton (§3.2 Double-Checked Locking), NaN/Inf-guard (§3.1),
assert sr == 48000, full PEP 484 type annotations.

Author: Aurik Development Team
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np
import scipy.signal as sig

logger = logging.getLogger(__name__)

SR_REQUIRED: int = 48_000
MAX_GAIN_DB: float = 4.0  # hard ceiling per peak
MAX_STRENGTH: float = 1.0  # blend ceiling
DEFAULT_STRENGTH: float = 0.40  # default enhancement strength


# ── Resonance model catalogue ─────────────────────────────────────────────────
# Each entry: (f0_hz, Q, gain_db_at_strength_1)
# gain is scaled by enhancement_strength at runtime

_RESONANCES: dict[str, list[tuple[float, float, float]]] = {
    "guitar": [
        (102.0, 8.0, 2.5),
        (195.0, 6.0, 2.0),
        (400.0, 4.0, 1.2),
        (2500.0, 3.0, 0.8),
    ],
    "keys": [
        (75.0, 5.0, 2.0),
        (180.0, 5.0, 1.5),
        (550.0, 3.5, 1.0),
        (3000.0, 2.5, 0.6),
    ],
    "piano": [  # alias
        (75.0, 5.0, 2.0),
        (180.0, 5.0, 1.5),
        (550.0, 3.5, 1.0),
        (3000.0, 2.5, 0.6),
    ],
    "brass": [
        (233.0, 10.0, 2.5),
        (466.0, 8.0, 2.0),
        (932.0, 6.0, 1.5),
        (1500.0, 4.0, 0.8),
    ],
    "drums": [
        (60.0, 4.0, 3.0),
        (120.0, 3.5, 1.5),
        (250.0, 3.0, 0.8),
        (5000.0, 2.0, 0.5),
    ],
    "percussion": [  # alias
        (60.0, 4.0, 3.0),
        (120.0, 3.5, 1.5),
        (250.0, 3.0, 0.8),
        (5000.0, 2.0, 0.5),
    ],
    "strings": [
        (290.0, 10.0, 3.0),
        (430.0, 8.0, 2.5),
        (560.0, 6.0, 1.5),
        (1800.0, 4.0, 0.8),
    ],
    "woodwinds": [
        (350.0, 8.0, 2.0),
        (700.0, 6.0, 1.5),
        (1200.0, 5.0, 1.0),
        (2500.0, 3.0, 0.5),
    ],
    "bass": [
        (85.0, 5.0, 2.5),
        (200.0, 4.0, 1.5),
        (600.0, 3.5, 1.0),
        (1500.0, 2.5, 0.5),
    ],
    "synth": [
        (80.0, 2.0, 1.0),
        (350.0, 1.5, 0.5),
        (3000.0, 1.5, 0.3),
        (8000.0, 1.0, 0.2),
    ],
}


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class ResonancePeakResult:
    """Diagnostics for one applied resonance peak."""

    f0_hz: float
    q: float
    gain_db_nominal: float
    gain_db_applied: float  # after strength scaling
    b_coeffs: tuple[float, float, float]
    a_coeffs: tuple[float, float, float]


@dataclass
class PhysicsResonanceResult:
    """Full result of :class:`PhysicsResonanceEnhancer`.

    Attributes:
        audio:               Enhanced output audio (same shape as input).
        instrument:          Instrument type key used.
        n_peaks:             Number of resonance peaks applied.
        peaks:               Per-peak diagnostic information.
        enhancement_strength: Effective blend factor used.
        passthrough:         True when no processing applied.
    """

    audio: np.ndarray
    instrument: str
    n_peaks: int
    peaks: list[ResonancePeakResult] = field(default_factory=list)
    enhancement_strength: float = 0.0
    passthrough: bool = False


# ── Biquad helpers ────────────────────────────────────────────────────────────


def _peak_eq_coeffs(f0_hz: float, q: float, gain_db: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute biquad peaking-EQ coefficients (Audio-EQ-Cookbook / Zölzer).

    Args:
        f0_hz:   Center frequency (Hz).
        q:       Quality factor.
        gain_db: Boost in dB (positive = boost).
        sr:      Sample rate (Hz).

    Returns:
        (b, a) — numerator/denominator coefficient arrays shape (3,).
    """
    f0_hz = float(np.clip(f0_hz, 1.0, sr / 2.0 - 1.0))
    q = float(np.clip(q, 0.1, 100.0))
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0_hz / sr
    alpha = np.sin(w0) / (2.0 * q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def _apply_peak(audio: np.ndarray, sr: int, f0: float, q: float, gain_db: float) -> np.ndarray:
    """Wendet einen einzelnen Biquad-Peak-EQ auf Mono-*audio* an."""
    if abs(gain_db) < 0.01:
        return audio
    b, a = _peak_eq_coeffs(f0, q, gain_db, sr)
    out = sig.lfilter(b, a, audio.astype(np.float64))
    return out.astype(np.float32)  # type: ignore[no-any-return]


# ── Core class ────────────────────────────────────────────────────────────────


class PhysicsResonanceEnhancer:
    """Wendet an: physics-derived body resonance coloration to instrument audio.

    Uses cascaded biquad peak-EQ filters whose center frequencies, Q values
    and gains are sourced from acoustic literature for each instrument class.

    Instantiate via :func:`get_physics_resonance_enhancer` (singleton §3.2).

    Args:
        enhancement_strength: Global blend factor 0.0–1.0 (default 0.40).
                              Scales all peak gains proportionally.
    """

    def __init__(self, enhancement_strength: float = DEFAULT_STRENGTH) -> None:
        self.enhancement_strength = float(np.clip(enhancement_strength, 0.0, MAX_STRENGTH))

    # ── Public API ────────────────────────────────────────────────────────────

    def enhance(
        self,
        audio: np.ndarray,
        sr: int,
        instrument: str = "guitar",
        enhancement_strength: float | None = None,
    ) -> PhysicsResonanceResult:
        """Wendet an: instrument body resonance coloration to *audio*.

        Args:
            audio:                Mono or stereo audio at 48 000 Hz.
            sr:                   Sample rate — must be 48 000 Hz.
            instrument:           Instrument type key (see _RESONANCES).
            enhancement_strength: Override blend 0.0–1.0.

        Returns:
            :class:`PhysicsResonanceResult` with enhanced audio and diagnostics.
        """
        assert sr == SR_REQUIRED, f"Sample rate must be 48000 Hz, got {sr}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        strength = float(
            np.clip(
                enhancement_strength if enhancement_strength is not None else self.enhancement_strength,
                0.0,
                MAX_STRENGTH,
            )
        )

        resonances = _RESONANCES.get(instrument.lower())

        def _passthrough(reason: str) -> PhysicsResonanceResult:
            out = np.clip(audio, -1.0, 1.0)
            logger.debug("PhysicsResonanceEnhancer passthrough: %s", reason)
            return PhysicsResonanceResult(
                audio=out,
                instrument=instrument,
                n_peaks=0,
                enhancement_strength=strength,
                passthrough=True,
            )

        if resonances is None:
            return _passthrough(f"unknown instrument '{instrument}'")
        if strength < 1e-5:
            return _passthrough("strength=0")

        # ── Process per channel ───────────────────────────────────────────────
        is_stereo = audio.ndim == 2
        if is_stereo:
            # Determine layout: (channels, samples) vs (samples, channels)
            if audio.shape[0] <= 8:
                channels = [audio[c].astype(np.float32) for c in range(audio.shape[0])]
            else:
                channels = [audio[:, c].astype(np.float32) for c in range(audio.shape[1])]

            processed_channels, peak_results = [], []
            for idx, ch in enumerate(channels):
                ch_out, prs = self._process_mono(ch, sr, resonances, strength)
                processed_channels.append(ch_out)
                if idx == 0:
                    peak_results = prs  # diagnostics from first channel

            out = np.stack(processed_channels, axis=0) if audio.shape[0] <= 8 else np.stack(processed_channels, axis=1)
        else:
            out, peak_results = self._process_mono(audio.astype(np.float32), sr, resonances, strength)

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0)

        logger.info(
            "PhysicsResonanceEnhancer: instrument=%s peaks=%d strength=%.2f",
            instrument,
            len(peak_results),
            strength,
        )
        return PhysicsResonanceResult(
            audio=out,
            instrument=instrument,
            n_peaks=len(peak_results),
            peaks=peak_results,
            enhancement_strength=strength,
            passthrough=False,
        )

    # ── Internal mono processing ──────────────────────────────────────────────

    def _process_mono(
        self,
        mono: np.ndarray,
        sr: int,
        resonances: list[tuple[float, float, float]],
        strength: float,
    ) -> tuple[np.ndarray, list[ResonancePeakResult]]:
        """Wendet kaskadierte Biquad-Peaks auf einen einzelnen Mono-Kanal an."""
        enhanced = mono.copy()
        peak_results: list[ResonancePeakResult] = []

        for f0, q, gain_nominal in resonances:
            # Scale gain by strength, clamp to MAX_GAIN_DB
            gain_applied = float(np.clip(gain_nominal * strength, -MAX_GAIN_DB, MAX_GAIN_DB))
            b, a = _peak_eq_coeffs(f0, q, gain_applied, sr)
            enhanced = _apply_peak(enhanced, sr, f0, q, gain_applied)
            peak_results.append(
                ResonancePeakResult(
                    f0_hz=f0,
                    q=q,
                    gain_db_nominal=gain_nominal,
                    gain_db_applied=gain_applied,
                    b_coeffs=(float(b[0]), float(b[1]), float(b[2])),
                    a_coeffs=(float(a[0]), float(a[1]), float(a[2])),
                )
            )

        # Identity-safe blend: wet = enhanced, dry = original
        result = ((1.0 - strength) * mono + strength * enhanced).astype(np.float32)

        return result, peak_results


# ── Singleton (§3.2 Double-Checked Locking) ──────────────────────────────────

_instance: PhysicsResonanceEnhancer | None = None
_lock = threading.Lock()


def get_physics_resonance_enhancer() -> PhysicsResonanceEnhancer:
    """Gibt the module-level singleton :class:`PhysicsResonanceEnhancer` zurück.

    Thread-safe via double-checked locking (§3.2).
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PhysicsResonanceEnhancer()
    return _instance


def enhance_physics_resonance(
    audio: np.ndarray,
    sr: int,
    instrument: str = "guitar",
    enhancement_strength: float | None = None,
) -> PhysicsResonanceResult:
    """Convenience wrapper: apply body resonance coloration to *audio*.

    Args:
        audio:                Mono or stereo audio at 48 000 Hz.
        sr:                   Sample rate — must be 48 000 Hz.
        instrument:           Instrument type key.
        enhancement_strength: Blend 0.0–1.0; max 1.0.

    Returns:
        :class:`PhysicsResonanceResult`.
    """
    return get_physics_resonance_enhancer().enhance(
        audio, sr, instrument=instrument, enhancement_strength=enhancement_strength
    )
