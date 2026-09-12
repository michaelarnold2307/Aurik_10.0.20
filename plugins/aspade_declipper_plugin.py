"""A-SPADE-Declipper-Plugin — neuronaler Declip-Zweig für Phase 07.

DECLIPPER_SOTA_PLAN.md Slice B (2026-09-12): A-SPADE-Klasse (Gaultier et al.,
IEEE/ACM TASLP 2023 — unrolled ADMM, 1-D-CNN, deterministische Feed-Forward-
Inferenz) als Quality-Zweig für starkes Clipping, wo PCHIP nichts bewirkt.

Vertrag:
    - Modell-Pfad: models/aspade/aspade_declipper.onnx (Waveform→Waveform,
      float32, time-domain).
    - Kein Modell vorhanden → Plugin meldet `is_available() == False`;
      der Aufrufer fällt auf DSP/CQT-Diff zurück (§V6 (copilot-instructions.md):
      logger.warning + Begründung).
    - Deterministisch (§G5 (copilot-instructions.md)): kein Sampling, feste
      Thread-Zahlen, chunked Inferenz mit Hann-Overlap-Add.
    - Never-worsen bleibt beim Aufrufer (Phase 07, Harmonik-Proxy).

Kein Netzwerk, kein Docker — lokales ONNX über ml_device_manager/Providers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_SUBPATH = "models/aspade/aspade_declipper.onnx"
TARGET_SR = 48_000
CHUNK_SEC = 10.0
OVERLAP_SEC = 0.5
_PEAK_NORM = 0.95


# ── APPLADE-Pfad (Gaultier et al., ICASSP 2022 — aus dem Paper implementiert) ──
# DGT: Hann 1024, Hop 256, FFT 1024, kanonisch-tightes Fenster (deterministisch).
# PnP-ADMM: x ← Π_Γ(F.H(v−u)); v ← T_θ(Fx+u); u ← Fx−v+u; λ = 0.3·Clip-Prozent.
_APL_W, _APL_A, _APL_M = 1024, 256, 1024


def _apl_tight_window() -> np.ndarray:
    g = np.hanning(_APL_W)
    s = np.zeros(_APL_W)
    for n in range(_APL_W):
        acc = 0.0
        for k in range(-4, 5):
            idx = n - k * _APL_A
            if 0 <= idx < _APL_W:
                acc += g[idx] ** 2
        s[n] = acc
    return (g / np.sqrt(s)).astype(np.float32)  # type: ignore[no-any-return]


_APL_GT = _apl_tight_window()


def _apl_dgt(x: np.ndarray) -> np.ndarray:
    n = len(x)
    n_frames = (n + _APL_A - 1) // _APL_A
    out = np.zeros((_APL_M // 2 + 1, n_frames), dtype=np.complex128)
    for m in range(n_frames):
        seg = np.zeros(_APL_W, dtype=np.float64)
        s0 = m * _APL_A
        ln = min(_APL_W, n - s0)
        if ln > 0:
            seg[:ln] = x[s0 : s0 + ln]
        out[:, m] = np.fft.rfft(seg * _APL_GT)
    return out  # type: ignore[no-any-return]


def _apl_idgt(x_cmplx: np.ndarray, n: int) -> np.ndarray:
    n_frames = x_cmplx.shape[1]
    out = np.zeros(n + _APL_W, dtype=np.float64)
    for m in range(n_frames):
        seg = np.fft.irfft(x_cmplx[:, m], n=_APL_W) * _APL_GT
        out[m * _APL_A : m * _APL_A + _APL_W] += seg
    return out[:n]  # type: ignore[no-any-return]


def _apl_pi_gamma(x: np.ndarray, idx: dict, clipped: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[idx["R"]] = clipped[idx["R"]]
    y[idx["H"]] = np.maximum(y[idx["H"]], clipped[idx["H"]])
    y[idx["L"]] = np.minimum(y[idx["L"]], clipped[idx["L"]])
    return y


class AspadeDeclipperPlugin:
    """Lädt A-SPADE-ONNX und de-clippt deterministisch (Waveform→Waveform)."""

    def __init__(self, model_dir: str | None = None) -> None:
        self._session = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._model_ok: bool = False
        if model_dir is not None:
            self._model_path = Path(model_dir) / "aspade_declipper.onnx"
        else:
            self._model_path = Path(__file__).parent.parent / DEFAULT_MODEL_SUBPATH
        self._try_load_model()

    # ------------------------------------------------------------------

    def _try_load_model(self) -> None:
        if not self._model_path.exists():
            logger.warning(
                "A-SPADE-ONNX nicht gefunden: %s — Phase 07 nutzt CQT-Diff/PCHIP-Ersatzpfad",
                self._model_path,
            )
            return
        try:
            import onnxruntime as ort

            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc

                if not _try_alloc("AspadeDeclipper", size_gb=0.60):
                    logger.warning("AspadeDeclipper: ML-Grenze erschöpft — Ersatzpfad.")
                    return
            except Exception as _exc:
                logger.debug("Plugin operation fehlgeschlagen (unkritisch): %s", _exc)

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            try:
                from backend.core.ml_device_manager import get_ort_providers as _get_prov

                _providers = _get_prov("AspadeDeclipper")
            except Exception:
                _providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(
                str(self._model_path),
                sess_options=opts,
                providers=_providers,
            )
            self._input_name = self._session.get_inputs()[0].name  # type: ignore[attr-defined]
            self._output_name = self._session.get_outputs()[0].name  # type: ignore[attr-defined]
            self._model_ok = True
            logger.info(
                "A-SPADE ONNX geladen: %s | In=%s | Out=%s", self._model_path, self._input_name, self._output_name
            )
        except Exception as _exc:
            logger.warning("A-SPADE-ONNX nicht ladbar (%s) — Ersatzpfad.", _exc)

    def is_available(self) -> bool:
        return self._model_ok and self._session is not None

    def declip(self, audio: np.ndarray, sr: int, clip_threshold: float) -> np.ndarray | None:
        """Neuronaler Declip eines Mono-Signals; None, wenn nicht verfügbar.

        Args:
            audio: Mono-Waveform (N,) float32.
            sr: Abtastrate (48 kHz erwartet; sonst None).
            clip_threshold: Detektionsschwelle — wird dem Aufrufer überlassen;
                das Modell rekonstruiert unabhängig davon die Wellenform.

        Returns:
            Degeclipptes Mono-Signal (N,) float32 — oder None (Ersatzpfad).
        """
        if not self.is_available():
            return None
        if int(sr) != TARGET_SR:
            logger.warning("A-SPADE: sr=%d ≠ 48 kHz — Ersatzpfad.", sr)
            return None
        return self._declip_applade(audio, sr, clip_threshold)
        x = np.asarray(audio, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(x)
        if n < 1024:
            return x.copy()  # type: ignore[no-any-return]

        chunk = int(CHUNK_SEC * TARGET_SR)
        hop = max(1, chunk - int(OVERLAP_SEC * TARGET_SR))
        win = np.hanning(chunk).astype(np.float32)
        out = np.zeros(n, dtype=np.float32)
        norm_acc = np.zeros(n, dtype=np.float32)

        peak = float(np.max(np.abs(x))) + 1e-9
        x_norm = (x * (_PEAK_NORM / peak)).astype(np.float32)

        if n <= chunk:
            # Einzel-Segment: padden, inferieren, zurückschneiden (deterministisch).
            seg_pad = np.zeros(chunk, dtype=np.float32)
            seg_pad[:n] = x_norm
            rec = self._infer(seg_pad) * (peak / _PEAK_NORM)
            out[:] = rec[:n]
            norm_acc[:] = 1.0
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(out, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

        for start in range(0, n - chunk + 1, hop):
            seg = x_norm[start : start + chunk]
            rec = self._infer(seg)
            rec = rec * (peak / _PEAK_NORM)
            out[start : start + chunk] += (rec * win).astype(np.float32)
            norm_acc[start : start + chunk] += win * win
        # Reststück ohne Overlap verarbeiten (deterministisch, gepaddet).
        tail_start = n - chunk
        if tail_start > 0:
            seg_pad = np.zeros(chunk, dtype=np.float32)
            seg_pad[: n - tail_start] = x_norm[tail_start:]
            rec = self._infer(seg_pad) * (peak / _PEAK_NORM)
            _len = n - tail_start
            out[tail_start:] += (rec[:_len] * win[:_len]).astype(np.float32)
            norm_acc[tail_start:] += win[:_len] * win[:_len]

        norm_acc = np.maximum(norm_acc, 1e-9)
        out = out / norm_acc
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(out, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    def _declip_applade(self, audio: np.ndarray, sr: int, clip_threshold: float) -> np.ndarray | None:
        """APPLADE-PnP-ADMM um das ONNX-DNN (16-kHz-Sprachmodell, Papier-Setup).

        Deterministisch; 16 384-Sample-Blöcke (64 DGT-Frames); Rückmischung
        nur in den geclippten Regionen (Masken-Blend, 48-kHz-Treue sonst).
        """
        from scipy.signal import resample_poly  # pylint: disable=import-outside-toplevel

        try:
            x = np.asarray(audio, dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            theta = float(np.clip(clip_threshold, 0.05, 0.999))
            mask = np.abs(x) >= theta  # 48-kHz-Domäne

            x16 = resample_poly(x.astype(np.float64), 1, 3).astype(np.float64)
            idx = {
                "H": (x16 > theta) | (np.abs(x16 - theta) < 1e-9),
                "L": x16 < -theta,
            }
            idx["R"] = ~(idx["H"] | idx["L"])
            if not idx["H"].any() and not idx["L"].any():
                return x.copy()  # type: ignore[no-any-return]
            pct = float(np.mean(~idx["R"]) * 100.0)
            lam = 30.0 * pct / 100.0
            weight = ((np.arange(1, _APL_M // 2 + 2, dtype=np.float64) / (_APL_M // 2 + 1)) ** 2).reshape(-1, 1)

            rec16 = np.zeros_like(x16)
            block = 16384
            assert self._session is not None
            for s0 in range(0, len(x16), block):
                blk = x16[s0 : s0 + block]
                if len(blk) < 512:
                    rec16[s0:] = blk
                    break
                orig_len = len(blk)
                if orig_len < block:
                    blk = np.pad(blk, (0, block - orig_len))
                # Block-lokale Masks (idx ist block-relativ)
                _bidx = {
                    "H": (blk > theta) | (np.abs(blk - theta) < 1e-9),
                    "L": blk < -theta,
                }
                _bidx["R"] = ~(_bidx["H"] | _bidx["L"])
                _v = _apl_dgt(blk)
                _u = np.zeros_like(_v)
                _x = blk.copy()
                for _ in range(100):
                    _x = _apl_pi_gamma(_apl_idgt(_v - _u, len(_x)), _bidx, blk)
                    _fx = _apl_dgt(_x)
                    _vin = _fx + _u
                    _mag = np.abs(_vin)
                    _feed = np.zeros((1, 1, _APL_M // 2, _mag.shape[1]), dtype=np.float32)
                    _feed[0, 0, :, :] = _mag[:-1, :]
                    _out = self._session.run([self._output_name], {self._input_name: _feed})[0]
                    _den = np.vstack([np.asarray(_out, dtype=np.float64)[0, 0], np.zeros((1, _mag.shape[1]))])
                    _v = np.maximum(0.0, 1.0 - lam * weight / (_den + 1e-6) ** 2) * _vin
                    _u = _fx - _v + _u
                _x[_bidx["R"]] = blk[_bidx["R"]]
                rec16[s0 : s0 + orig_len] = _x[:orig_len]

            rec48 = resample_poly(rec16, 3, 1)[: len(x)].astype(np.float32)
            fade = int(0.005 * sr)
            win = 0.5 * (1 - np.cos(np.pi * np.arange(fade) / max(fade, 1)))
            blend = np.zeros(len(x), dtype=np.float32)
            edges = np.diff(mask.astype(np.int8))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            for st in starts:
                a0 = max(0, st - fade)
                blend[a0:st] = np.maximum(blend[a0:st], win[: st - a0])
            for en in ends:
                b1 = min(len(x), en + fade)
                blend[en:b1] = np.maximum(blend[en:b1], win[::-1][: b1 - en])
            blend[mask] = 1.0
            out = blend * rec48 + (1.0 - blend) * x
            return np.clip(out, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]
        except Exception as _exc:
            logger.warning("A-SPADE-APPLADE-Pfad fehlgeschlagen (%s) — Ersatzpfad.", _exc)
            return None

    def _infer(self, seg: np.ndarray) -> np.ndarray:
        """Ein Segment (chunk,) → Modell-Inferenz → (chunk,) float32."""
        assert self._session is not None
        n = len(seg)
        # Robust gegen beide üblichen A-SPADE-Export-Layouts: (1, 1, T) und (1, T).
        try:
            feed = np.zeros((1, 1, n), dtype=np.float32)
            feed[0, 0, :] = seg
            res = self._session.run([self._output_name], {self._input_name: feed})[0]
        except Exception:
            feed = np.zeros((1, n), dtype=np.float32)
            feed[0, :] = seg
            res = self._session.run([self._output_name], {self._input_name: feed})[0]
        arr = np.asarray(res, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0, 0, :]
        elif arr.ndim == 2:
            arr = arr[0, :]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if len(arr) >= n:
            return arr[:n]
        rec = np.zeros(n, dtype=np.float32)
        rec[: len(arr)] = arr
        return rec


_plugin_singleton: AspadeDeclipperPlugin | None = None


def get_aspade_declipper_plugin() -> AspadeDeclipperPlugin:
    """Prozess-weite Singleton-Instanz (lazy, deterministisch)."""
    global _plugin_singleton
    if _plugin_singleton is None:
        _plugin_singleton = AspadeDeclipperPlugin()
    return _plugin_singleton
