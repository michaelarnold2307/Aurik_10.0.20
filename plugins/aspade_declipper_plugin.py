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
