"""
backend/core/fusion_engine.py — Multi-model output fusion
=========================================================

Blends audio outputs from multiple restoration models using
weighted average or adaptive signal-level fusion.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class FusionEngine:
    """Mischt multiple audio outputs into one.

    Parameters
    ----------
    strategy:
        ``"mean"``    — equal-weight average.
        ``"weighted"`` — weighted by signal quality.
        ``"adaptive"`` — picks best-quality output per frame.
    """

    def __init__(self, strategy: str = "mean") -> None:
        self.strategy = strategy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        outputs: Sequence[np.ndarray],
        weights: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Mischt *outputs* into a single audio signal.

        Parameters
        ----------
        outputs:
            List of audio arrays, all the same length and dtype.
        weights:
            Optional per-output weights (will be L1-normalised).
            Ignored when ``strategy="mean"``.

        Returns
        -------
        Fused audio as float32 ndarray.
        """
        if not outputs:
            return np.zeros(1, dtype=np.float32)  # type: ignore[no-any-return]

        arrays = [np.asarray(o, dtype=np.float32) for o in outputs]
        # Ensure equal length
        min_len = min(len(a) for a in arrays)
        arrays = [a[:min_len] for a in arrays]

        if self.strategy == "mean" or weights is None:
            result = np.mean(np.stack(arrays, axis=0), axis=0)
        elif self.strategy in ("weighted", "adaptive"):
            w = np.asarray(weights, dtype=np.float64)
            w = w / (w.sum() + 1e-12)
            result = np.zeros(min_len, dtype=np.float64)
            for arr, weight in zip(arrays, w):
                result += weight * arr.astype(np.float64)
            result = result.astype(np.float32)
        else:
            result = arrays[0]

        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Singleton accessor (thread-safe, double-checked locking)
# ---------------------------------------------------------------------------
import threading as _threading

_fusion_engine_instance = None
_fusion_engine_lock = _threading.Lock()


def get_fusion_engine() -> FusionEngine:
    """Gibt the process-wide singleton FusionEngine instance zurück."""
    global _fusion_engine_instance
    if _fusion_engine_instance is None:
        with _fusion_engine_lock:
            if _fusion_engine_instance is None:
                _fusion_engine_instance = FusionEngine()
    return _fusion_engine_instance
