"""
core/attack_type_classifier.py — Instrument Attack-Type Classification
=======================================================================

Classifies the *physical attack mechanism* of an instrument onset into one of
five acoustic categories using purely DSP features — no ML model required.

Attack types (analogous to phoneme classes in vocal processing):

    PICK   — Sharp plucked/picked attack:  guitar pick, harpsichord, banjo
             Signature: massive HF transient (>5 kHz), rise time < 5 ms,
             spectral centroid > 0.45, spectral flatness < 0.4.

    BOW    — Gradual bowed/drawn attack:   violin, cello, contrabass, hurdy-gurdy
             Signature: slow ramp (50–120 ms), dominant low-mid energy (<5 kHz),
             centroid < 0.25, flatness < 0.25.

    MALLET — Medium-speed mallet/hammer:   xylophone, marimba, vibraphone, piano key,
                                           orchestral timpani
             Signature: attack 10–50 ms, mid-high centroid (0.25–0.55),
             high harmonic content (low flatness).

    STRIKE — Very sharp, broadband hit:    crash cymbal, tam-tam, snare rimshot,
                                           castanets
             Signature: rise time < 3 ms, spectral flatness > 0.55 (noise-like),
             broadband energy distribution.

    BREATH — Breath/tongue articulation:   flute, recorder, saxophone tonguing,
                                           trumpet flutter-tongue
             Signature: ZCR > 0.35, energy concentration 1–8 kHz,
             noisy onset (flatness > 0.40).

Scientific foundation:
    Bello et al. (2005): "A Tutorial on Onset Detection in Music Signals"
    Masri (1996): "Computer Modeling of Sound ... Musical Signals"
    Scheirer (1998): "Tempo and Beat Analysis of Acoustic Musical Signals"
    Leveau et al. (2004): "Methodology and Tools for the Evaluation of
                           Automatic Onset Detection Algorithms in Music"

Algorithm per onset:
    1. Extract a 50 ms analysis window around the onset.
    2. Compute four features:
       - Spectral Centroid (amplitude-weighted, normalized 0–1 over 0–24 kHz)
       - Spectral Flatness  (Wiener entropy, 0 = tonal, 1 = noise)
       - Zero-Crossing Rate (normalized per sample)
       - Envelope Rise Time (10%→90% of peak RMS, in ms)
    3. Rule-based decision tree ordered by discriminating power.
    4. Confidence = 1 − (normalized distance to decision boundary).

Singleton pattern (§3.2 Double-Checked Locking), NaN/Inf-guard (§3.1),
full PEP 484 type annotations, assert sample_rate == 48000.

Author: Aurik Development Team
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

SR_REQUIRED: int = 48_000

# Analysis window around onset (ms)
ANALYSIS_WINDOW_MS: float = 50.0
# Onset search window used when no onset is supplied (first 100 ms)
ONSET_SEARCH_MS: float = 100.0

# Feature thresholds (tuned on McIntyre/Woodhouse, Benade reference recordings)
_CENTROID_BOW_MAX: float = 0.25  # centroid ≤ → BOW candidate
_CENTROID_PICK_MIN: float = 0.42  # centroid ≥ → PICK / STRIKE candidate
_FLATNESS_STRIKE_MIN: float = 0.55  # flatness ≥ → STRIKE candidate
_FLATNESS_BREATH_MIN: float = 0.38  # flatness ≥ (+ ZCR) → BREATH candidate
_ZCR_BREATH_MIN: float = 0.34  # ZCR ≥ (+ flatness) → BREATH candidate
_RISE_PICK_MAX_MS: float = 6.0  # rise time ≤ → PICK (not MALLET)
_RISE_BOW_MIN_MS: float = 40.0  # rise time ≥ → confirms BOW

# ── Result dataclass ─────────────────────────────────────────────────────────


@dataclass
class AttackTypeResult:
    """Result of :class:`AttackTypeClassifier`.

    Attributes:
        attack_type:        One of ``'pick'`` | ``'bow'`` | ``'mallet'`` |
                            ``'strike'`` | ``'breath'`` | ``'unknown'``.
        confidence:         Scalar in [0, 1].  Higher = more certain.
        onset_sample:       Sample index of the detected onset (−1 if none found).
        spectral_centroid:  Normalized spectral centroid used for classification.
        spectral_flatness:  Spectral flatness (Wiener entropy) of the onset window.
        zcr:                Normalized zero-crossing rate of the onset window.
        rise_time_ms:       Envelope rise time (10 %→90 % of peak), in ms.
        features:           Dict copy of the four scalar features for logging.
    """

    attack_type: str
    confidence: float
    onset_sample: int
    spectral_centroid: float
    spectral_flatness: float
    zcr: float
    rise_time_ms: float
    features: dict = field(default_factory=dict)


# ── Core class ───────────────────────────────────────────────────────────────


class AttackTypeClassifier:
    """Classify the physical attack type of a musical instrument onset.

    Instantiate once via :func:`get_attack_type_classifier` (singleton).

    Usage::

        clf = get_attack_type_classifier()
        result = clf.classify(audio, sr=48000)
        # log result: logger.debug("attack=%s conf=%.2f", result.attack_type, result.confidence)
    """

    # ── Feature extraction helpers ────────────────────────────────────────────

    @staticmethod
    def _mono(audio: np.ndarray) -> np.ndarray:
        """Gibt mono view (averaged channels, float64) zurück."""
        a = audio.astype(np.float64)
        if a.ndim == 2:
            return np.mean(a, axis=0) if a.shape[0] < a.shape[1] else np.mean(a, axis=1)  # type: ignore[no-any-return]
        return a  # type: ignore[no-any-return]

    @staticmethod
    def _spectral_centroid(frame: np.ndarray, sr: int) -> float:
        """Amplitude-weighted spectral centroid, normalized 0–1 over 0–sr/2 Hz."""
        mag = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        total = mag.sum() + 1e-12
        freqs = np.linspace(0.0, 1.0, len(mag))
        return float((freqs * mag).sum() / total)

    @staticmethod
    def _spectral_flatness(frame: np.ndarray) -> float:
        """Wiener entropy: geometric mean / arithmetic mean of |FFT|, in [0, 1]."""
        mag = np.abs(np.fft.rfft(frame * np.hanning(len(frame)))) + 1e-12
        log_mean = np.mean(np.log(mag))
        arith_mean = np.mean(mag)
        flatness = math.exp(log_mean) / (arith_mean + 1e-12)
        return float(np.clip(flatness, 0.0, 1.0))

    @staticmethod
    def _zcr(frame: np.ndarray) -> float:
        """Zero-crossing rate normalized per sample, in [0, 1]."""
        zc: int = int(np.sum(np.diff(np.sign(frame)) != 0))
        return float(zc / max(len(frame) - 1, 1) / 2.0)

    @staticmethod
    def _envelope_rise_time_ms(frame: np.ndarray, sr: int) -> float:
        """RMS-envelope rise time from 10 % to 90 % of peak, in ms.

        Uses a 1 ms hop RMS envelope.  Returns ANALYSIS_WINDOW_MS if the
        envelope never crosses the thresholds (i.e. no clear onset in frame).
        """
        hop = max(1, sr // 1000)  # 1 ms hop
        n_frames = len(frame) // hop
        if n_frames < 2:
            return ANALYSIS_WINDOW_MS
        rms = np.array([np.sqrt(np.mean(frame[i * hop : (i + 1) * hop] ** 2)) for i in range(n_frames)])
        peak = rms.max() + 1e-12
        lo_idx = np.argmax(rms >= 0.10 * peak)
        hi_idx = np.argmax(rms >= 0.90 * peak)
        if hi_idx <= lo_idx:
            # Peak not reached within window → slow onset
            return ANALYSIS_WINDOW_MS
        return float((hi_idx - lo_idx) * 1.0)  # already in ms (1 ms hop)

    # ── Onset detection ───────────────────────────────────────────────────────

    @staticmethod
    def _find_onset(mono: np.ndarray, sr: int) -> int:
        """Erkennt the first prominent onset within the first ONSET_SEARCH_MS.

        Uses a simple spectral flux onset detector (Bello et al. 2005):
        per-bin positive flux summed across the spectrum, peak-picked.

        Returns sample index of onset, or 0 if none found.
        """
        search_samples = int(ONSET_SEARCH_MS / 1000.0 * sr)
        audio_slice = mono[:search_samples]
        hop = max(1, sr // 200)  # 5 ms hop
        win = min(len(audio_slice), hop * 4)
        if len(audio_slice) < win:
            return 0

        prev_mag = np.zeros(win // 2 + 1)
        flux_values: list[float] = []
        for start in range(0, len(audio_slice) - win, hop):
            frame = audio_slice[start : start + win] * np.hanning(win)
            mag = np.abs(np.fft.rfft(frame))
            flux = float(np.sum(np.maximum(mag - prev_mag, 0.0)))
            flux_values.append(flux)
            prev_mag = mag

        if not flux_values:
            return 0

        flux_arr = np.array(flux_values)
        # Threshold: mean + 1.5 std
        threshold = flux_arr.mean() + 1.5 * flux_arr.std()
        peaks = np.where(flux_arr > threshold)[0]
        if len(peaks) == 0:
            return 0
        onset_frame = int(peaks[0])
        return min(onset_frame * hop, len(mono) - 1)

    # ── Decision tree ─────────────────────────────────────────────────────────

    @staticmethod
    def _decide(
        centroid: float,
        flatness: float,
        zcr: float,
        rise_ms: float,
    ) -> tuple[str, float]:
        """Map features to (attack_type, confidence) via rule-based decision tree.

        Ordered by discriminative power (most separable first):

        1. STRIKE  — broadband noise + very sharp rise
        2. BREATH  — noisy + high ZCR  (wind articulation)
        3. BOW     — very low centroid + slow rise
        4. PICK    — high centroid + fast rise
        5. MALLET  — everything else with moderate rise
        6. UNKNOWN — fallback
        """
        # 1. STRIKE: noise-like spectrum + sharp onset
        if flatness >= _FLATNESS_STRIKE_MIN and rise_ms <= 8.0:
            conf = float(
                np.clip(
                    0.5
                    + (flatness - _FLATNESS_STRIKE_MIN) / (1.0 - _FLATNESS_STRIKE_MIN) * 0.4
                    + (8.0 - rise_ms) / 8.0 * 0.1,
                    0.0,
                    1.0,
                )
            )
            return "strike", conf

        # 2. BREATH: noisy but not broadband, high ZCR
        if flatness >= _FLATNESS_BREATH_MIN and zcr >= _ZCR_BREATH_MIN:
            conf = float(np.clip(0.45 + zcr * 0.3 + flatness * 0.25, 0.0, 1.0))
            return "breath", conf

        # 3. BOW: low centroid and slow rise
        if centroid <= _CENTROID_BOW_MAX and rise_ms >= _RISE_BOW_MIN_MS:
            conf = float(
                np.clip(
                    0.5 + (_CENTROID_BOW_MAX - centroid) / _CENTROID_BOW_MAX * 0.3 + min(rise_ms, 120.0) / 120.0 * 0.2,
                    0.0,
                    1.0,
                )
            )
            return "bow", conf

        # 4. PICK: high centroid and fast rise
        if centroid >= _CENTROID_PICK_MIN and rise_ms <= _RISE_PICK_MAX_MS:
            conf = float(
                np.clip(
                    0.5
                    + (centroid - _CENTROID_PICK_MIN) / (1.0 - _CENTROID_PICK_MIN) * 0.3
                    + (_RISE_PICK_MAX_MS - rise_ms) / _RISE_PICK_MAX_MS * 0.2,
                    0.0,
                    1.0,
                )
            )
            return "pick", conf

        # 5. MALLET: medium rise, moderate to high centroid
        if 8.0 < rise_ms < 60.0:
            conf = float(np.clip(0.45 + centroid * 0.2 + (1.0 - flatness) * 0.15, 0.0, 1.0))
            return "mallet", conf

        return "unknown", 0.30

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(
        self,
        audio: np.ndarray,
        sr: int,
        onset_sample: int | None = None,
    ) -> AttackTypeResult:
        """Classify the physical attack type of the dominant onset in *audio*.

        Args:
            audio:         Mono or stereo audio at 48 000 Hz.
            sr:            Sample rate — must be 48 000 Hz.
            onset_sample:  Optional pre-computed onset position (sample index).
                           If *None*, the onset is detected automatically.

        Returns:
            :class:`AttackTypeResult` with ``attack_type``, ``confidence``,
            and all intermediate feature values.
        """
        assert sr == SR_REQUIRED, f"Sample rate must be 48000 Hz, got {sr}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        mono = self._mono(audio)

        if len(mono) == 0:
            return AttackTypeResult(
                attack_type="unknown",
                confidence=0.0,
                onset_sample=-1,
                spectral_centroid=0.0,
                spectral_flatness=0.0,
                zcr=0.0,
                rise_time_ms=0.0,
                features={},
            )

        # Detect or use supplied onset
        if onset_sample is None:
            onset_sample = self._find_onset(mono, sr)

        # Extract analysis window (50 ms) around onset
        win_samples = int(ANALYSIS_WINDOW_MS / 1000.0 * sr)
        t0 = int(np.clip(onset_sample, 0, max(len(mono) - 1, 0)))
        t1 = min(t0 + win_samples, len(mono))
        frame = mono[t0:t1]

        if len(frame) < 8:
            return AttackTypeResult(
                attack_type="unknown",
                confidence=0.0,
                onset_sample=int(onset_sample),
                spectral_centroid=0.0,
                spectral_flatness=0.0,
                zcr=0.0,
                rise_time_ms=0.0,
                features={},
            )

        # Pad to win_samples for FFT stability
        if len(frame) < win_samples:
            frame = np.pad(frame, (0, win_samples - len(frame)))

        # Compute features
        centroid = self._spectral_centroid(frame, sr)
        flatness = self._spectral_flatness(frame)
        zcr = self._zcr(frame)
        rise_ms = self._envelope_rise_time_ms(frame, sr)

        # Guard against NaN/Inf in features
        if not all(math.isfinite(v) for v in (centroid, flatness, zcr, rise_ms)):
            centroid = float(np.nan_to_num(centroid))
            flatness = float(np.nan_to_num(flatness))
            zcr = float(np.nan_to_num(zcr))
            rise_ms = ANALYSIS_WINDOW_MS

        attack_type, confidence = self._decide(centroid, flatness, zcr, rise_ms)

        features = {
            "spectral_centroid": centroid,
            "spectral_flatness": flatness,
            "zcr": zcr,
            "rise_time_ms": rise_ms,
        }

        logger.debug(
            "AttackTypeClassifier: type=%s conf=%.2f centroid=%.3f flat=%.3f zcr=%.3f rise=%.1fms",
            attack_type,
            confidence,
            centroid,
            flatness,
            zcr,
            rise_ms,
        )

        return AttackTypeResult(
            attack_type=attack_type,
            confidence=confidence,
            onset_sample=int(onset_sample),
            spectral_centroid=centroid,
            spectral_flatness=flatness,
            zcr=zcr,
            rise_time_ms=rise_ms,
            features=features,
        )

    def classify_batch(
        self,
        audio: np.ndarray,
        sr: int,
        onset_samples: list[int],
    ) -> list[AttackTypeResult]:
        """Classify multiple onsets in *audio* at the given sample positions.

        Args:
            audio:          Mono or stereo audio at 48 000 Hz.
            sr:             Sample rate — must be 48 000 Hz.
            onset_samples:  List of onset sample positions.

        Returns:
            List of :class:`AttackTypeResult`, one per onset.
        """
        return [self.classify(audio, sr, onset_sample=s) for s in onset_samples]


# ── Singleton (§3.2 Double-Checked Locking) ──────────────────────────────────

_instance: AttackTypeClassifier | None = None
_lock = threading.Lock()


def get_attack_type_classifier() -> AttackTypeClassifier:
    """Gibt the module-level singleton :class:`AttackTypeClassifier` zurück.

    Thread-safe via double-checked locking (§3.2).
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = AttackTypeClassifier()
    return _instance


def classify_attack_type(
    audio: np.ndarray,
    sr: int,
    onset_sample: int | None = None,
) -> AttackTypeResult:
    """Convenience wrapper: classify attack type of *audio*.

    Args:
        audio:         Mono or stereo audio at 48 000 Hz.
        sr:            Sample rate — must be 48 000 Hz.
        onset_sample:  Optional pre-computed onset position.

    Returns:
        :class:`AttackTypeResult`.
    """
    return get_attack_type_classifier().classify(audio, sr, onset_sample=onset_sample)
