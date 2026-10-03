"""
dsp/pesto_pitch.py — PESTO-inspired Chromagram Pitch Tracker (DSP variant)

PESTO (Riou et al. ISMIR 2023): "Efficient Pitch Estimation with Self-Supervised
Transposition-equivariant Objective". This DSP implementation captures the core idea:
CQT-based log-frequency representation + harmonic-weighted peak picking, without
the learned transposition-equivariant head.

Role in Aurik §4.4 pitch-tracking hierarchy (DSP tier, penultimate fallback):
    FCPE (ML, primary) → RMVPE (ML) → PESTO (DSP) → pYIN (last resort)

VERBOTEN as primary tracker — FCPE / RMVPE accuracy is superior for music.

Key properties vs pYIN:
    - ~8–20× faster (no YIN difference function across all lags)
    - Better on polyphonic content with dominant tonal component
    - Weaker on breathy/noisy material (use pYIN fallback there)
    - No voicing confidence output → uses energy heuristic instead

Reference:
    Riou et al. "PESTO: Pitch Estimation with Self-Supervised Transposition-equivariant
    Objective" — ISMIR 2023, https://arxiv.org/abs/2309.02396

Singleton: use get_pesto_estimator() for shared instances.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SR_TARGET: int = 48_000  # Aurik canonical SR (assert at entry)
_F0_MIN_HZ: float = 40.0  # C1 ≈ 32.7 Hz — practical floor for music
_F0_MAX_HZ: float = 2100.0  # C7 ≈ 2093 Hz  — soprano / violin ceiling
_N_BINS_PER_OCTAVE: int = 48  # 4 bins/semitone → ~25 cent resolution
_N_OCTAVES: int = 6  # 40 … 2560 Hz (padded above _F0_MAX_HZ)
_HOP_SAMPLES: int = 512  # 10.67 ms @ 48 kHz (≈ FCPE frame hop)
_WINDOW_SAMPLES: int = 4096  # 85 ms window for sub-bass energy
_N_HARMONICS: int = 5  # harmonic summation depth
_ENERGY_FLOOR: float = 1e-7  # unvoiced frame gate
_VOICING_RATIO_MIN: float = 2.5  # peak-to-median ratio for voiced decision

# Semitone → Hz mapping
_N_BINS_TOTAL: int = _N_BINS_PER_OCTAVE * _N_OCTAVES
_F_REF: float = _F0_MIN_HZ
_BIN_FREQS: np.ndarray = _F_REF * 2.0 ** (np.arange(_N_BINS_TOTAL, dtype=np.float64) / _N_BINS_PER_OCTAVE)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PestoResult:
    """Output of PESTO pitch estimation.

    Attributes:
        f0:         Estimated F0 per frame [Hz]; NaN = unvoiced.
        times:      Frame centre times [s].
        voiced:     Boolean mask — True if frame is considered voiced.
        confidence: Normalised peak-to-median ratio ∈ [0, 1] (heuristic).
        f0_mean:    Mean F0 of voiced frames [Hz].
        f0_std:     Std of F0 of voiced frames [Hz].
        model_used: Always "pesto_dsp".
    """

    f0: np.ndarray
    times: np.ndarray
    voiced: np.ndarray
    confidence: np.ndarray
    f0_mean: float = 0.0
    f0_std: float = 0.0
    model_used: str = "pesto_dsp"
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------


class PestoPitchEstimator:
    """CQT-based pitch estimator (PESTO-inspired, no neural network).

    Computes a pseudo-CQT salience function via STFT + log-frequency
    aggregation, applies harmonic-weighted summing (Klapuri 2003), and
    finds the F0 via parabolic sub-bin interpolation (Brown & Puckette 1993).

    Thread-safe after construction (no mutable state in estimate()).
    """

    def __init__(self) -> None:
        self._bin_freqs = _BIN_FREQS.copy()
        self._build_fft_mapping()
        logger.debug("PestoPitchEstimator ready: %d bins, %.0f–%.0f Hz", _N_BINS_TOTAL, _F_REF, self._bin_freqs[-1])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_fft_mapping(self) -> None:
        """Pre-compute FFT bin → CQT bin weight matrix (sparse, float32)."""
        fft_freqs = np.fft.rfftfreq(_WINDOW_SAMPLES, d=1.0 / _SR_TARGET)
        n_fft_bins = len(fft_freqs)

        # For each CQT bin k, collect FFT bins within ±0.5 semitone bandwidth.
        # bandwidth = f_centre * (2^(1/(2*bpo)) - 2^(-1/(2*bpo)))
        half_bw_ratio = 2.0 ** (1.0 / (2.0 * _N_BINS_PER_OCTAVE)) - 1.0
        rows, cols, vals = [], [], []
        for k, fc in enumerate(self._bin_freqs):
            bw = fc * half_bw_ratio
            lo, hi = fc - bw, fc + bw
            mask = (fft_freqs >= lo) & (fft_freqs <= hi)
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            # Triangular weight within bandwidth
            w = 1.0 - np.abs(fft_freqs[idx] - fc) / bw
            w = np.clip(w, 0.0, 1.0).astype(np.float32)
            rows.extend([k] * len(idx))
            cols.extend(idx.tolist())
            vals.extend(w.tolist())

        self._map_rows = np.array(rows, dtype=np.int32)
        self._map_cols = np.array(cols, dtype=np.int32)
        self._map_vals = np.array(vals, dtype=np.float32)
        self._n_fft_bins = n_fft_bins

    def _stft_magnitude(self, frame: np.ndarray) -> np.ndarray:
        """Gibt FFT magnitude for one zero-padded frame zurück."""
        win = np.hanning(len(frame)).astype(np.float64)
        spec = np.abs(np.fft.rfft(frame * win, n=_WINDOW_SAMPLES))
        return np.asarray(spec.astype(np.float32))  # type: ignore[no-any-return]

    def _salience(self, mag: np.ndarray) -> np.ndarray:
        """Map FFT magnitude → CQT salience via pre-built weight matrix."""
        sal = np.zeros(_N_BINS_TOTAL, dtype=np.float32)
        np.add.at(sal, self._map_rows, self._map_vals * mag[self._map_cols])
        return np.asarray(sal)  # type: ignore[no-any-return]

    def _harmonic_summation(self, sal: np.ndarray) -> np.ndarray:
        """Sum harmonics 1..N_HARMONICS using log-frequency shift (Klapuri 2003)."""
        harmonic_sal = sal.copy()
        for harmonic in range(2, _N_HARMONICS + 1):
            shift = round(_N_BINS_PER_OCTAVE * math.log2(harmonic))
            if shift < _N_BINS_TOTAL:
                n = _N_BINS_TOTAL - shift
                harmonic_sal[:n] += sal[shift : shift + n] * (1.0 / harmonic)
        return harmonic_sal

    def _pick_peak(self, harm_sal: np.ndarray) -> tuple[float, float]:
        """Parabolic interpolation around argmax → (f0_hz, confidence).

        confidence = peak / median (heuristic voicing proxy).
        Returns (nan, 0.0) if energy or peak-ratio too low.
        """
        if harm_sal.max() < _ENERGY_FLOOR:
            return float("nan"), 0.0

        median_sal = float(np.median(harm_sal[harm_sal > 0]) + 1e-9) if np.any(harm_sal > 0) else 1e-9
        peak_idx = int(np.argmax(harm_sal))
        peak_val = float(harm_sal[peak_idx])
        ratio = peak_val / median_sal

        if ratio < _VOICING_RATIO_MIN:
            return float("nan"), float(np.clip(ratio / _VOICING_RATIO_MIN, 0.0, 1.0))

        # Parabolic sub-bin interpolation (Brown & Puckette 1993)
        if 0 < peak_idx < _N_BINS_TOTAL - 1:
            alpha = float(harm_sal[peak_idx - 1])
            beta = float(harm_sal[peak_idx])
            gamma = float(harm_sal[peak_idx + 1])
            denom = alpha - 2.0 * beta + gamma
            sub = (alpha - gamma) / (2.0 * denom) if abs(denom) > 1e-9 else 0.0
            sub = float(np.clip(sub, -0.5, 0.5))
        else:
            sub = 0.0

        f0_hz = _F_REF * 2.0 ** ((peak_idx + sub) / _N_BINS_PER_OCTAVE)
        confidence = float(np.clip((ratio - _VOICING_RATIO_MIN) / max(_VOICING_RATIO_MIN * 4.0, 1e-9), 0.0, 1.0))
        return f0_hz, confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        voiced_threshold: float = 0.15,
    ) -> PestoResult:
        """Schätzt F0 trajectory via PESTO-inspired CQT salience.

        Args:
            audio:            Mono float32/64 audio, any length ≥ 1 sample.
            sr:               Sample rate — MUST be 48000 (Aurik invariant).
            voiced_threshold: Confidence above which a frame is voiced.

        Returns:
            PestoResult with per-frame F0, voicing mask, and confidence.
        """
        assert sr == _SR_TARGET, f"PestoPitchEstimator: sr={sr} ≠ {_SR_TARGET}"
        audio = np.asarray(audio, dtype=np.float64).ravel()
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        n_samples = len(audio)
        hop = _HOP_SAMPLES
        n_frames = max(1, (n_samples - _WINDOW_SAMPLES) // hop + 1)

        f0_arr = np.full(n_frames, float("nan"), dtype=np.float32)
        conf_arr = np.zeros(n_frames, dtype=np.float32)
        times_arr = np.arange(n_frames, dtype=np.float32) * hop / sr

        for i in range(n_frames):
            start = i * hop
            end = start + _WINDOW_SAMPLES
            if end > n_samples:
                frame_raw = np.zeros(_WINDOW_SAMPLES, dtype=np.float64)
                frame_raw[: n_samples - start] = audio[start:]
            else:
                frame_raw = audio[start:end]

            mag = self._stft_magnitude(frame_raw)
            sal = self._salience(mag)
            harm_sal = self._harmonic_summation(sal)
            f0, conf = self._pick_peak(harm_sal)
            f0_arr[i] = f0
            conf_arr[i] = conf

        voiced = conf_arr >= voiced_threshold
        f0_arr = np.where(voiced, f0_arr, float("nan"))

        voiced_f0 = f0_arr[voiced & np.isfinite(f0_arr)]
        f0_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        f0_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0

        return PestoResult(
            f0=f0_arr,
            times=times_arr,
            voiced=voiced,
            confidence=conf_arr,
            f0_mean=f0_mean,
            f0_std=f0_std,
            model_used="pesto_dsp",
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: PestoPitchEstimator | None = None
_lock = threading.Lock()


def get_pesto_estimator() -> PestoPitchEstimator:
    """Gibt shared PestoPitchEstimator instance (thread-safe lazy init) zurück."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PestoPitchEstimator()
    return _instance


def estimate_pitch(audio: np.ndarray, sr: int, **kwargs) -> PestoResult:
    """Convenience wrapper — estimates F0 for mono audio at 48 kHz.

    Returns PestoResult. On any error, returns a silent result with all NaN.
    """
    try:
        return get_pesto_estimator().estimate(audio, sr, **kwargs)
    except Exception as exc:  # pragma: no cover
        logger.warning("PestoPitchEstimator fehlgeschlagen: %s — stilles Ergebnis wird zurückgegeben", exc)
        n = max(1, len(audio) // _HOP_SAMPLES)
        return PestoResult(
            f0=np.full(n, float("nan"), dtype=np.float32),
            times=np.arange(n, dtype=np.float32) * _HOP_SAMPLES / sr,
            voiced=np.zeros(n, dtype=bool),
            confidence=np.zeros(n, dtype=np.float32),
            model_used="pesto_dsp_error",
        )
