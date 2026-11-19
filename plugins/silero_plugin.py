"""SileroPlugin — VAD via lokales ONNX (kein Docker/HF).

Modell : models/silero/silero_en_v5.onnx
ONNX   : input[batch,samples] -> output[batch,frames,999]
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_inst: SileroPlugin | None = None
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL = os.path.join(_ROOT, "models", "silero", "silero_en_v5.onnx")
_SR = 16_000
_CHUNK = 512  # 32 ms @ 16 kHz


class SileroPlugin:
    def __init__(self, model_path: str | None = None) -> None:
        self._session: Any = None
        self._threshold = 0.5
        self._try_load(model_path or _MODEL)

    def _try_load(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning("Silero VAD fehlt: %s -- Energie-Ersatzpfad.", path)
            return
        try:
            import onnxruntime as ort

            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc

                if not _try_alloc("SileroVAD", size_gb=0.11):
                    logger.warning("SileroVAD: ML-Grenze erschöpft — Energie-Ersatzpfad.")
                    return
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            # §v10.40c: Registry-konsultierte Provider-Wahl statt hartem CPU.
            from backend.core.gpu_model_registry import get_onnx_providers

            self._session = ort.InferenceSession(path, sess_options=opts, providers=get_onnx_providers(path))
            logger.info("Silero VAD ONNX geladen: %s", path)
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                _reg_plm("SileroVAD", size_gb=0.11, unload_fn=lambda s=self: setattr(s, "_session", None))  # type: ignore[misc]
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
        except Exception as exc:
            logger.warning("Silero Ladefehler: %s -- Energie-Ersatzpfad.", exc)
            try:
                from backend.core.ml_memory_budget import release as _rel

                _rel("SileroVAD")
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

    def is_speech(self, audio: np.ndarray, sr: int) -> float:
        """Gibt Sprach-Wahrscheinlichkeit [0,1] zurueck."""
        if audio.ndim == 2:
            # Handle (2, N) channels-first (UV3) and (N, 2) samples-first
            mono = (
                audio.mean(axis=0) if (audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]) else audio.mean(axis=1)
            )
        else:
            mono = audio
        mono = _resamp(mono, sr, _SR).astype(np.float32)
        if self._session:
            return self._vad_onnx(mono)
        return self._energy_vad(mono)

    def get_speech_mask(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Gibt bool-Array zurueck (True = Sprach-Segment).

        Uses a single ONNX call for the entire audio to avoid repeated small
        inference calls that can destabilise the ONNX runtime.
        """
        if audio.ndim == 2:
            # Handle (2, N) channels-first (UV3) and (N, 2) samples-first
            mono = (
                audio.mean(axis=0) if (audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]) else audio.mean(axis=1)
            )
        else:
            mono = audio
        mono16 = _resamp(mono, sr, _SR).astype(np.float32)
        n16 = len(mono16)

        if self._session is not None:
            try:
                mask16 = self._vad_mask_single_call(mono16)
            except Exception as exc:
                logger.warning("Silero VAD single-call fehlgeschlagen (%s), using energy Ersatzpfad", exc)
                mask16 = self._energy_mask(mono16)
        else:
            mask16 = self._energy_mask(mono16)

        # Upsample mask to original SR via nearest-neighbour (contiguous ranges)
        if sr != _SR and len(mono) != n16:
            indices = np.arange(len(mono))
            src_idx = np.clip((indices * n16 / len(mono)).astype(int), 0, n16 - 1)
            return mask16[src_idx]  # type: ignore[no-any-return]
        return mask16[: len(mono)]  # type: ignore[no-any-return]

    def _vad_mask_single_call(self, mono16: np.ndarray) -> np.ndarray:
        """Führt aus: ONNX model once on entire audio and derive per-sample bool mask."""
        _plm = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

            _plm = get_plugin_lifecycle_manager()
            _plm.set_active("SileroVAD", True)
        except Exception:
            logger.warning("silero_plugin.py::_vad_mask_single_call Ersatzpfad", exc_info=True)
        try:
            inp = mono16[None].astype(np.float32)  # [1, n_samples]
            out = self._session.run(None, {"input": inp})[0]  # [1, frames, 999]
            out = np.nan_to_num(np.asarray(out, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            probs = out[0]  # [frames, 999]
            # Per-frame speech probability: max over non-silence classes
            if probs.shape[-1] > 1:
                frame_probs = probs[:, 1:].max(axis=-1)  # [frames]
            else:
                frame_probs = np.full(probs.shape[0], 0.5, dtype=np.float32)
            frame_probs = np.clip(frame_probs, 0.0, 1.0)
            # Expand frame-level decisions to sample-level mask
            n_frames = len(frame_probs)
            n_samples = len(mono16)
            if n_frames < 1:
                return np.ones(n_samples, dtype=bool)  # type: ignore[no-any-return]
            # Map each sample to its frame
            sample_indices = np.arange(n_samples)
            frame_indices = np.clip((sample_indices * n_frames / n_samples).astype(int), 0, n_frames - 1)
            return frame_probs[frame_indices] >= self._threshold  # type: ignore[no-any-return]
        finally:
            if _plm is not None:
                try:
                    _plm.set_active("SileroVAD", False)
                except Exception:
                    logger.warning("silero_plugin.py::_vad_mask_single_call Ersatzpfad", exc_info=True)

    def _energy_mask(self, mono16: np.ndarray) -> np.ndarray:
        """Energy-based VAD fallback: chunk-wise RMS."""
        mask = np.zeros(len(mono16), dtype=bool)
        for s in range(0, len(mono16), _CHUNK):
            e = min(s + _CHUNK, len(mono16))
            if self._energy_vad(mono16[s:e]) >= self._threshold:
                mask[s:e] = True
        return mask  # type: ignore[no-any-return]

    def _vad_onnx(self, chunk: np.ndarray) -> float:
        if len(chunk) < 1:
            return 0.0
        _plm = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

            _plm = get_plugin_lifecycle_manager()
            _plm.set_active("SileroVAD", True)
        except Exception:
            logger.warning("silero_plugin.py::_vad_onnx Ersatzpfad", exc_info=True)
        try:
            inp = chunk[None].astype(np.float32)
            try:
                out = self._session.run(None, {"input": inp})[0]  # [1,frames,999]
                out = np.nan_to_num(np.asarray(out, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                probs = out[0]  # [frames, 999]
                speech_prob = float(probs[:, 1:].max(axis=-1).mean()) if probs.shape[-1] > 1 else 0.5
                return min(max(speech_prob, 0.0), 1.0)
            except Exception as exc:
                logger.debug("Silero VAD ONNX Ausfuehrung Fehler: %s", exc)
                return self._energy_vad(chunk)
        finally:
            if _plm is not None:
                try:
                    _plm.set_active("SileroVAD", False)
                except Exception:
                    logger.warning("silero_plugin.py::_vad_onnx Ersatzpfad", exc_info=True)

    @staticmethod
    def _energy_vad(chunk: np.ndarray, threshold: float = 0.01) -> float:
        rms = float(np.sqrt(np.mean(np.nan_to_num(chunk, nan=0.0) ** 2)))
        return 1.0 if rms > threshold else 0.0


def _resamp(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(src, dst)
    return resample_poly(x, dst // g, src // g).astype(np.float32)  # type: ignore[no-any-return]


def get_silero_plugin() -> SileroPlugin:
    global _inst
    if _inst is None:
        with _lock:
            if _inst is None:
                _inst = SileroPlugin()
    return _inst


def is_speech(audio: np.ndarray, sr: int) -> float:
    """Convenience-Wrapper: Sprach-Wahrscheinlichkeit [0,1]."""
    return get_silero_plugin().is_speech(audio, sr)


# synthesize() Stub — TTS-Interface für Kompatibilität
def _silero_synthesize(self, text: str, speaker: str = "default", sr: int = 48000, **kwargs) -> bytes:
    """TTS-Stub: gibt leeres bytes-Objekt zurück (kein echtes TTS ohne Modell)."""
    import numpy as np

    samples = np.zeros(sr, dtype=np.float32)  # 1 s Stille
    return samples.tobytes()  # type: ignore[no-any-return]


SileroPlugin.synthesize = _silero_synthesize  # type: ignore[attr-defined]


# Backward-compat alias (Spec §11.3)
SileroVADPlugin = SileroPlugin
