#!/usr/bin/env python3
"""AERO-Plugin — Challenger-Kandidat: Bandbreiten-Extension 12 kHz → 48 kHz.

Bewertung gegen den Incumbent (FlashSR) über scripts/challenger_round.py auf
dem goldenen Hör-Set. NICHT in die Produktions-Routing-Pipeline verdrahtet —
Aufnahme erst nach bestandener Challenger-Runde (ADOPT).

§v10-AERO-ONNX (2026-09-10): Primär ONNX (models/aero/aero_12_48.onnx,
exportiert via scripts/export_aero_onnx.py, Parität rel≈2e-5, statisches
Produktions-Shape 120000 Samples = 10 s @ 12 kHz). GPU-Policy über
backend/core/gpu_model_registry (§v10.40c, verdict=rocm). Eager-.th bleibt
als §V6-Fallback (Warnung + Grund bei jedem Fallback).

Shape-Konstanz: Das ONNX-Artefakt ist am Shape 120000 gebacken (BLSTM-
Framing/LocalState-Trace-Konstanten). Kürzere Endsegmente werden deshalb auf
120000 ge-null-padded und das 4-fach verlängerte Ergebnis auf die reale
Segmentlänge getrimmt (Receptive-Field-Effekte am gepaddeten Rand werden
weggetrimmt).

Quelle: slp-rl/aero (MIT, vendored unter plugins/_vendor_aero/ mit LICENSE).
Checkpoint: models/aero/checkpoint_12-48_hl256.th (offizieller Google-Drive-Link
aus dem Upstream-README).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
_VENDOR = _ROOT / "_vendor_aero"
_MODEL_DIR = _ROOT.parent / "models" / "aero"
_CHECKPOINT = _MODEL_DIR / "checkpoint_12-48_hl256.th"
_ONNX_PATH = _MODEL_DIR / "aero_12_48.onnx"
_LR_SR = 12000
_HR_SR = 48000
_SEGMENT_S = 10
_SEGMENT = _SEGMENT_S * _LR_SR  # 120000 — statisches ONNX-Produktions-Shape

_inst: AeroPlugin | None = None


class AeroPlugin:
    """AERO Super-Resolution (12 kHz → 48 kHz) als Challenger-Kandidat."""

    def __init__(
        self,
        checkpoint: Path | None = None,
        device: str = "cpu",
        onnx_path: Path | None = None,
    ) -> None:
        self._model: Any = None  # eager-.th-Fallback (§V6 (copilot-instructions.md))
        self._session: Any = None  # ONNX-Session (primär)
        self._device = device
        self.checkpoint = Path(checkpoint) if checkpoint else _CHECKPOINT
        self._onnx_path = Path(onnx_path) if onnx_path else _ONNX_PATH
        self._try_load()

    def _try_load(self) -> None:
        if not self.checkpoint.exists() and not self._onnx_path.exists():
            logger.warning(
                "AERO-Modelle fehlen (%s / %s) — Plugin ohne Modell (Challenger nicht lauffähig).",
                self.checkpoint.name,
                self._onnx_path.name,
            )
            return
        # ── Primär: ONNX (mit §v10.40c-GPU-Policy) ──
        if self._onnx_path.exists():
            try:
                import onnxruntime as ort  # pylint: disable=import-outside-toplevel

                from backend.core.gpu_model_registry import apply_gpu_policy

                _gpu = self._device != "cpu"
                _requested = ["ROCMExecutionProvider", "CPUExecutionProvider"] if _gpu else ["CPUExecutionProvider"]
                _providers = apply_gpu_policy(_requested, self._onnx_path)
                self._session = ort.InferenceSession(str(self._onnx_path), providers=_providers)
                logger.info(
                    "AERO ONNX geladen (%s, provider=%s)",
                    self._onnx_path.name,
                    self._session.get_providers()[0],
                )
                return
            except Exception as exc:  # pylint: disable=broad-except
                # §V6 (copilot-instructions.md): kein Silent-Failure — Warnung + Grund, dann eager-Fallback.
                logger.warning("AERO-ONNX nicht ladbar (%s) — eager-.th-Ersatzpfad", exc)
                self._session = None
        # ── §V6-Fallback: eager .th ──
        if not self.checkpoint.exists():
            logger.warning("AERO-.th-Checkpoint fehlt — Plugin ohne Modell.")
            return
        if str(_VENDOR) not in sys.path:
            sys.path.insert(0, str(_VENDOR))
        try:
            import torch  # pylint: disable=import-outside-toplevel
            from src.models.aero import Aero
        except Exception as exc:
            logger.warning("AERO-Vendor-Import fehlgeschlagen: %s", exc)
            return
        try:
            import inspect  # pylint: disable=import-outside-toplevel

            package = torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
            kwargs = dict(package["models"]["generator"].get("kwargs") or {})
            sig = inspect.signature(Aero.__init__).parameters
            kwargs = {k: v for k, v in kwargs.items() if k in sig and k != "self"}
            model = Aero(**kwargs)
            model.load_state_dict(package["models"]["generator"]["state"])
            model.eval()
            model.to(self._device)
            self._model = model
            logger.info("AERO eager geladen (%s, device=%s, Ersatzpfad)", self.checkpoint.name, self._device)
        except Exception as exc:
            logger.warning("AERO-Ladevorgang fehlgeschlagen: %s", exc)
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._session is not None or self._model is not None

    def enhance(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        """12 kHz → 48 kHz Bandbreiten-Extension. None bei fehlendem Modell."""
        if self._session is None and self._model is None:
            return None

        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != _LR_SR:
            from scipy.signal import resample_poly  # pylint: disable=import-outside-toplevel

            g = int(np.gcd(sr, _LR_SR))
            audio = resample_poly(audio, _LR_SR // g, sr // g).astype(np.float32)

        chunks = [audio[i : i + _SEGMENT] for i in range(0, len(audio), _SEGMENT)]
        outs: list[np.ndarray] = []
        for ch in chunks:
            real_len = len(ch)
            out_len = real_len * (_HR_SR // _LR_SR)
            if self._session is not None:
                # Statisches ONNX-Shape: Endsegment auf 120000 padden, Output trimmen.
                if real_len < _SEGMENT:
                    ch = np.pad(ch, (0, _SEGMENT - real_len), mode="constant")
                x = ch.astype(np.float32)[None, None, :]
                out = self._session.run(None, {"mix": x})[0]
                out = out[0, 0, :out_len]
            else:
                import torch  # pylint: disable=import-outside-toplevel

                with torch.no_grad():
                    mono = torch.from_numpy(ch.astype(np.float32))
                    x = mono.unsqueeze(0).unsqueeze(0).to(self._device)
                    out = self._model(x).squeeze(0).squeeze(0).cpu().numpy()
                out = out[:out_len]
            outs.append(np.asarray(out, dtype=np.float32))

        out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
        peak = float(np.max(np.abs(out))) if out.size else 1.0
        if peak > 1.0:
            out = out / peak
        return cast(np.ndarray | None, out[: int(round(len(audio) * (_HR_SR / _LR_SR)))])


def get_aero_plugin(device: str = "cpu") -> AeroPlugin:
    """Thread-sicherer Singleton für Challenger-Runs."""
    global _inst
    if _inst is None:
        _inst = AeroPlugin(device=device)
    return _inst
