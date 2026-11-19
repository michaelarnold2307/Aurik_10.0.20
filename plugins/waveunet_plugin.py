"""WaveUNetPlugin legacy adapter for routed vocal/instrumental separation.

The historical WaveUNet model is not bundled.  This module keeps the legacy
API but delegates first to the central SOTA vocal router, then uses HPSS as an
honest DSP fallback.
"""

# Optional DSP dependencies are imported lazily inside fallback paths.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_inst: WaveUNetPlugin | None = None


class WaveUNetPlugin:
    """Legacy WaveUNet-compatible adapter backed by the central vocal router."""

    def __init__(self) -> None:
        self._last_route_metadata: dict[str, object] = {
            "model_used": "uninitialized",
            "capability_status": "unavailable",
            "fallback_chain": [],
        }
        logger.info("WaveUNetPlugin: legacy adapter via SOTA vocal router.")

    @property
    def route_metadata(self) -> dict[str, object]:
        """Gibt metadata for the most recent routing or DSP fallback path zurück."""
        return dict(self._last_route_metadata)

    def separate(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """Separate vocal and instrumental stems via SOTA router or HPSS fallback."""
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        reference = np.asarray(audio, dtype=np.float32)
        try:
            from backend.core.dsp.sota_vocal_model_router import (  # pylint: disable=import-outside-toplevel
                get_sota_vocal_model_router,
            )

            routed = get_sota_vocal_model_router().separate_vocal_instrumental(
                reference,
                sr,
                panns_singing=0.5,
                ctx={"legacy_adapter": "waveunet"},
            )
            self._last_route_metadata = {
                "model_used": routed.model_used,
                "success": routed.success,
                "capability_status": routed.metadata.get("capability_status", "unknown"),
                "fallback_chain": list(routed.fallback_chain),
                "legacy_adapter": "waveunet",
            }
            if routed.success:
                return np.clip(routed.vocal, -1.0, 1.0), np.clip(routed.instrumental, -1.0, 1.0)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("WaveUNet legacy adapter router path unavailable: %s", exc)
        if audio.ndim == 2:
            # Handle (2, N) channels-first (UV3) and (N, 2) samples-first
            mono = (
                audio.mean(axis=0) if (audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]) else audio.mean(axis=1)
            )
        else:
            mono = audio
        mono = mono.astype(np.float32)
        try:
            import librosa

            harm, perc = librosa.effects.hpss(mono)  # type: ignore[attr-defined]
            harm = np.nan_to_num(harm, nan=0.0, posinf=0.0, neginf=0.0)
            perc = np.nan_to_num(perc, nan=0.0, posinf=0.0, neginf=0.0)
        except ImportError:
            harm = mono.copy()
            perc = np.zeros_like(mono)
        self._last_route_metadata = {
            "model_used": "hpss_dsp_fallback",
            "success": True,
            "capability_status": "dsp_fallback",
            "fallback_chain": ["sota_vocal_router:unavailable"],
            "legacy_adapter": "waveunet",
        }
        return np.clip(harm, -1.0, 1.0), np.clip(perc, -1.0, 1.0)


def get_waveunet_plugin() -> WaveUNetPlugin:
    """Gibt the thread-safe WaveUNet legacy adapter singleton zurück."""
    global _inst  # pylint: disable=global-statement
    if _inst is None:
        with _lock:
            if _inst is None:
                _inst = WaveUNetPlugin()
    return _inst


def separate(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper for WaveUNet-compatible separation."""
    return get_waveunet_plugin().separate(audio, sr)
