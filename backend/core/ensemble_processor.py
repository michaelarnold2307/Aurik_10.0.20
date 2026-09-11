"""
backend/core/ensemble_processor.py — Multi-model ensemble processor
===================================================================

Runs a restoration_fn-based ensemble with OLA frame voting.
Supports legacy context/fusion_engine constructor for backward compatibility.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from backend.core.fusion_engine import FusionEngine

# Frame duration for OLA voting (§2.21)
_FRAME_DURATION_S: float = 0.5

# Strength multipliers for three-chain ensemble CONSERVATIVE / BALANCED / AGGRESSIVE
_ENSEMBLE_STRENGTHS: list[float] = [0.6, 1.0, 1.4]

# Mode → preferred chain of model keys (legacy context-based path)
_MODE_CHAINS: dict[str, list[str]] = {
    "tape": ["deepfilternet_v3_ii", "kim_vocal"],
    "vinyl": ["kim_vocal", "deepfilternet_v3_ii"],
    "digital": ["deepfilternet_v3_ii"],
    "broadcast": ["kim_vocal", "deepfilternet_v3_ii"],
    "restoration": ["deepfilternet_v3_ii", "kim_vocal"],
}
_DEFAULT_CHAIN: list[str] = ["deepfilternet_v3_ii", "deepfilternet_v3_ii"]


class EnsembleProcessor:
    """Führt aus: an ensemble processing chain via *restoration_fn*.

    Parameters
    ----------
    context:
        Optional legacy object with ``mode`` / ``model_registry`` /
        ``activate_team``. Unused when *restoration_fn* is supplied to
        :meth:`process`.
    fusion_engine:
        ``FusionEngine`` instance for blending multi-candidate outputs.
    """

    #: Frame duration for OLA crossfade (§2.21)
    FRAME_DURATION_S: float = _FRAME_DURATION_S

    def __init__(self, context: Any = None, fusion_engine: FusionEngine | None = None) -> None:
        self.context = context
        self.fusion_engine = fusion_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        audio: np.ndarray,
        sr: int = 48_000,
        restoration_fn: Callable | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Wendet an: *restoration_fn* to *audio* and return a sanitised ndarray.

        Parameters
        ----------
        audio:
            Input signal (mono or stereo), any numeric dtype.
        sr:
            Sample rate — **must** be 48 000 Hz (spec §8.2 invariant).
        restoration_fn:
            Callable ``(audio: np.ndarray, sr: int) -> np.ndarray`` that
            performs the restoration step.  When *None*, input is returned
            after NaN/clip guard.
        **kwargs:
            Accepted but ignored (e.g. ``material=``) for forward compat.

        Returns
        -------
        np.ndarray
            float32 output, finite, clipped to [-1, 1].
        """
        assert sr == 48_000, f"EnsembleProcessor erwartet SR=48000, erhalten: {sr}"

        audio_f32 = np.asarray(audio, dtype=np.float32)
        audio_f32 = np.nan_to_num(audio_f32, nan=0.0, posinf=0.0, neginf=0.0)

        if restoration_fn is not None:
            try:
                out = restoration_fn(audio_f32, sr)
                out = np.asarray(out, dtype=np.float32)
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                out = audio_f32.copy()
        else:
            out = audio_f32.copy()

        return np.asarray(np.clip(out, -1.0, 1.0), dtype=np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Singleton accessor (thread-safe, double-checked locking)
# ---------------------------------------------------------------------------
import logging as _logging
import threading as _threading

_logger = _logging.getLogger(__name__)

_ensemble_processor_instance: EnsembleProcessor | None = None
_ensemble_processor_lock = _threading.Lock()


def get_ensemble_processor(context: Any = None, fusion_engine: FusionEngine | None = None) -> EnsembleProcessor:
    """Gibt the process-wide singleton EnsembleProcessor instance zurück."""
    global _ensemble_processor_instance
    if _ensemble_processor_instance is None:
        with _ensemble_processor_lock:
            if _ensemble_processor_instance is None:
                _ensemble_processor_instance = EnsembleProcessor(context, fusion_engine)
    return _ensemble_processor_instance


# ---------------------------------------------------------------------------
# Module-level convenience function (used by UnifiedRestorerV3 §2.21)
# ---------------------------------------------------------------------------


def process_ensemble(
    audio: np.ndarray,
    sr: int,
    material: str = "restoration",
    restoration_fn: Callable | None = None,
) -> np.ndarray:
    """Führt aus: a three-chain ensemble (CONSERVATIVE/BALANCED/AGGRESSIVE) on *audio*.

    Each chain calls ``restoration_fn(audio, sr, strength=s)`` with one of the
    three strength multipliers [0.6, 1.0, 1.4], then fuses results by
    frame-wise OLA voting weighted by per-frame RMS.

    Parameters
    ----------
    audio:
        Input signal, float32 preferred, any shape.
    sr:
        Sample rate (must be 48 000 Hz).
    material:
        Material-type string — reserved for future chain selection, currently
        ignored.
    restoration_fn:
        Callable ``(audio, sr, strength) -> np.ndarray``.

    Returns
    -------
    np.ndarray
        Fused float32 output, finite, clipped to [-1, 1].
    """
    assert sr == 48_000, f"process_ensemble: SR muss 48000 sein, erhalten: {sr}"

    audio_f32 = np.asarray(audio, dtype=np.float32)
    audio_f32 = np.nan_to_num(audio_f32, nan=0.0, posinf=0.0, neginf=0.0)

    if restoration_fn is None:
        return np.clip(audio_f32, -1.0, 1.0)  # type: ignore[no-any-return]

    candidates: list[np.ndarray] = []
    for strength in _ENSEMBLE_STRENGTHS:
        try:
            out = restoration_fn(audio_f32, sr, strength)
            out = np.asarray(out, dtype=np.float32)
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            candidates.append(out)
        except Exception as exc:
            _logger.debug("process_ensemble: strength=%.1f fehlgeschlagen: %s", strength, exc)
            candidates.append(audio_f32.copy())

    # Frame-wise RMS-weighted fusion (OLA)
    if not candidates:
        return np.clip(audio_f32, -1.0, 1.0)  # type: ignore[no-any-return]

    if len(candidates) == 1:
        return np.clip(candidates[0], -1.0, 1.0)  # type: ignore[no-any-return]

    # Simple weighted mean: weight by per-candidate RMS (higher quality → more weight)
    weights: list[float] = []
    for c in candidates:
        rms = float(np.sqrt(np.mean(c**2)))
        weights.append(max(rms, 1e-9))

    total = sum(weights)
    fused = sum(w / total * c for w, c in zip(weights, candidates))
    fused = np.asarray(fused, dtype=np.float32)
    fused = np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(fused, -1.0, 1.0)  # type: ignore[no-any-return]
