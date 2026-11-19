"""
UVR_MDXNetPlugin — Instrumentaltrennung via lokale ONNX-Modelle (kein Docker).

Modelle: models/uvr_mdx_net/uvr_mdx_net_inst_hq_{1..4}.onnx
ONNX-Interface: input[batch,4,3072,256] → output[batch,4,3072,256]
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_inst: UVRMDXNetPlugin | None = None

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = os.path.join(_ROOT, "models", "uvr_mdx_net")
_SR = 44_100
_N_FFT = 6144
_HOP = 1024
_BINS = 3072
_CHUNK = 256  # Zeitschritte pro Chunk


class UVRMDXNetPlugin:
    """UVR MDX-Net Instrumental-Trennung — 4 HQ-Modelle, ONNX lokal."""

    MODEL_FILES = [
        "uvr_mdx_net_inst_hq_1.onnx",
        "uvr_mdx_net_inst_hq_2.onnx",
        "uvr_mdx_net_inst_hq_3.onnx",
        "uvr_mdx_net_inst_hq_4.onnx",
    ]

    def __init__(self, model_dir: str | None = None) -> None:
        d = model_dir or _MODEL_DIR
        self._sessions: list = []
        self._try_load(d)

    def _try_load(self, d: str) -> None:
        try:
            import onnxruntime as ort

            # ML-Budget-Guard: 4 UVR-MDX-Net-Modelle zusammen ~1.2 GB
            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc

                if not _try_alloc("UVR_MDXNet", size_gb=1.20):
                    logger.warning("UVR MDX-Net: ML-Budget erschöpft — DSP-Fallback.")
                    return
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            try:
                from backend.core.ml_device_manager import get_ort_providers as _get_prov

                prov = _get_prov("MDXNet")
            except Exception:
                prov = ["CPUExecutionProvider"]
            for mf in self.MODEL_FILES:
                mp = os.path.join(d, mf)
                if not os.path.exists(mp):
                    continue
                try:
                    self._sessions.append(ort.InferenceSession(mp, sess_options=opts, providers=prov))
                    logger.info("UVR MDX-Net geladen: %s", mf)
                except Exception as exc:
                    logger.debug("UVR Modell-Ladefehler %s: %s", mf, exc)
            if not self._sessions:
                logger.warning("Keine UVR-Modelle in: %s — DSP-Fallback.", d)
                try:
                    from backend.core.ml_memory_budget import release as _release

                    _release("UVR_MDXNet")
                except Exception as _exc:
                    logger.debug("Operation failed (non-critical): %s", _exc)
            else:
                try:
                    from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                    _reg_plm("UVR_MDXNet", size_gb=1.20, unload_fn=lambda s=self: setattr(s, "_sessions", []))  # type: ignore[misc]
                except Exception as _exc:
                    logger.debug("Operation failed (non-critical): %s", _exc)
        except Exception as exc:
            logger.warning("UVR ONNX-Ladefehler: %s — DSP-Fallback.", exc)
            try:
                from backend.core.ml_memory_budget import release as _release

                _release("UVR_MDXNet")
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

    # ── Public ──────────────────────────────────────────────────────────────

    def separate(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """(vocals, instrumental) aus Audio trennen."""
        audio = np.nan_to_num(audio.astype(np.float32))
        if audio.ndim == 1:
            mono = audio
        elif audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:  # (2, N) channels-first (UV3)
            mono = audio.mean(axis=0)
        else:  # (N, 2) samples-first
            mono = audio.mean(axis=1)
        inst = self._run_ensemble(mono, sr) if self._sessions else self._hpss_fallback(mono)
        # §NaN-Guard: ONNX inference and HPSS fallback can produce NaN/Inf in edge cases.
        # Replace non-finite values before computing vocals to keep subtraction stable.
        inst = np.nan_to_num(inst, nan=0.0, posinf=1.0, neginf=-1.0)
        inst = np.clip(inst, -1.0, 1.0)
        voc = np.clip(mono - inst, -1.0, 1.0)
        return voc.astype(np.float32), inst.astype(np.float32)

    # ── Internal ────────────────────────────────────────────────────────────

    def _stft_mag(self, mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        win = np.hanning(_N_FFT).astype(np.float32)
        n = len(mono)
        n_frames = max(1, (n + _HOP - 1) // _HOP)
        padded = np.zeros(n_frames * _HOP + _N_FFT, dtype=np.float32)
        padded[:n] = mono
        specs = []
        for i in range(n_frames):
            frame = padded[i * _HOP : i * _HOP + _N_FFT] * win
            specs.append(np.fft.rfft(frame)[:_BINS])
        spec = np.array(specs, dtype=np.complex64).T  # [3072, frames]
        return np.abs(spec), spec  # (mag, complex)

    def _istft(self, spec: np.ndarray, n_orig: int) -> np.ndarray:
        win = np.hanning(_N_FFT).astype(np.float32)
        n_frames = spec.shape[1]
        n_out = n_frames * _HOP + _N_FFT
        out = np.zeros(n_out, dtype=np.float32)
        ws = np.zeros(n_out, dtype=np.float32)
        full = np.zeros((_N_FFT // 2 + 1, n_frames), dtype=np.complex64)
        full[:_BINS] = spec[:_BINS]
        for i in range(n_frames):
            frame = np.fft.irfft(full[:, i], n=_N_FFT).real.astype(np.float32)
            out[i * _HOP : i * _HOP + _N_FFT] += frame * win
            ws[i * _HOP : i * _HOP + _N_FFT] += win**2
        ws = np.where(ws < 1e-8, 1.0, ws)
        return (out / ws)[:n_orig].astype(np.float32)  # type: ignore[no-any-return]

    def _run_ensemble(self, mono: np.ndarray, sr: int) -> np.ndarray:
        from scipy.signal import resample_poly

        # Resample zu Modell-SR
        if sr != _SR:
            from math import gcd

            g = gcd(sr, _SR)
            mono = resample_poly(mono, _SR // g, sr // g).astype(np.float32)

        # Snapshot sessions to avoid races with lifecycle eviction while processing.
        sessions = tuple(self._sessions)
        if not sessions:
            return self._hpss_fallback(mono)

        # §4.6b PLM-Active-Guard: verhindert Emergency-Eviction während Inferenz.
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_uvr

            _plm_uvr = _get_plm_uvr()
            _plm_uvr.set_active("UVR_MDXNet", True)
        except Exception as _exc:
            logger.debug("UVR MDX-Net: PLM set_active failed: %s", _exc)
            _plm_uvr = None
        try:
            n = len(mono)
            mag, cplx = self._stft_mag(mono)  # [3072, frames]
            n_frames = mag.shape[1]
            masks_sum = np.zeros_like(mag)

            for sess in sessions:
                mask_out = np.zeros_like(mag)
                for s in range(0, n_frames, _CHUNK):
                    e = min(s + _CHUNK, n_frames)
                    T = e - s
                    seg = mag[:, s:e]
                    # Pad zu [batch=1, 4, 3072, 256]
                    inp = np.zeros((1, 4, _BINS, _CHUNK), dtype=np.float32)
                    inp[0, 0, :, :T] = seg
                    inp[0, 1, :, :T] = seg  # 4 Kanäle mit gleicher Magnitude
                    inp[0, 2, :, :T] = seg
                    inp[0, 3, :, :T] = seg
                    try:
                        out = sess.run(None, {"input": inp})[0]  # [1,4,3072,256]
                        mask_out[:, s:e] = np.clip(out[0, 0, :, :T], 0, 1)
                    except Exception as exc:
                        logger.debug("UVR chunk-Fehler: %s", exc)
                        mask_out[:, s:e] = 0.5
                masks_sum += mask_out

            # §G-DIV0: max(len,1) prevents divide-by-zero when PLM evicts sessions
            mask = np.clip(masks_sum / float(max(len(sessions), 1)), 0, 1)
            inst_spec = mask * np.abs(cplx) * np.exp(1j * np.angle(cplx))
            inst = self._istft(inst_spec, n)
            # Zurück auf Eingangs-SR
            if sr != _SR:
                from math import gcd

                g = gcd(_SR, sr)
                inst = resample_poly(inst, sr // g, _SR // g).astype(np.float32)
            mn, mx = len(inst), len(mono)
            if mx > mn:
                return np.pad(inst, (0, mx - mn)).astype(np.float32)  # type: ignore[no-any-return]
            return inst[:mx].astype(np.float32)  # type: ignore[no-any-return]
        finally:
            try:
                if _plm_uvr is not None:
                    _plm_uvr.set_active("UVR_MDXNet", False)
            except Exception as _exc:
                logger.debug("UVR MDX-Net: PLM unset_active failed: %s", _exc)

    @staticmethod
    def _hpss_fallback(mono: np.ndarray) -> np.ndarray:
        try:
            import librosa

            _H, P = librosa.effects.hpss(mono)  # type: ignore[attr-defined]
            return P.astype(np.float32)  # type: ignore[no-any-return]
        except Exception:
            logger.warning("uvr_mdxnet_plugin.py::_hpss_fallback fallback", exc_info=True)
            return (mono * 0.7).astype(np.float32)  # type: ignore[no-any-return]


def get_uvr_mdxnet_plugin() -> UVRMDXNetPlugin:
    """Thread-sicherer Singleton."""
    global _inst
    if _inst is None:
        with _lock:
            if _inst is None:
                _inst = UVRMDXNetPlugin()
    return _inst


def separate_instrumental(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: (vocals, instrumental)."""
    return get_uvr_mdxnet_plugin().separate(audio, sr)


# Convenience-Alias
import numpy as _np


def separate_vocals_uvr(audio: _np.ndarray, sr: int = 48000):
    """Gibt (vocals, instrumental) Tuple zurück."""
    return separate_instrumental(audio, sr)
