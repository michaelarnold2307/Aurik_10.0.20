"""EAR-VAE-Denoiser-Plugin — musik-finetuneter VAE-Denoiser (44,1 kHz, Stereo).

Kandidat für den Quality-Mode-Denoise-Pfad: Encoder→Latent→Decoder
(ONNX, CPU). Die Absicherung übernimmt die H1-Masking-Fusion in
backend/core/dsp/hybrid_denoise_fusion.py (Never-worsen, Bark-Schwellen)
plus H2 Musical-Noise-Gate — dieses Plugin liefert nur den Kandidaten.

§V6 (copilot-instructions.md): Fehlt das Modell oder schlägt die Inferenz
fehl, gibt denoise() None zurück (logger.warning + Begründung) — der
Aufrufer behält den DSP-Pfad.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

_SR_TARGET = 44100
_MODEL_DIR = "models/ear_vae"


class EarVaeDenoiser:
    """ONNX-Inferenz (Encoder + Decoder) für den musik-finetuneten EAR-VAE."""

    def __init__(self) -> None:
        self._enc: Any = None
        self._dec: Any = None
        self._load_error: str | None = None
        self._init()

    def _init(self) -> None:
        try:
            import onnxruntime as ort  # pylint: disable=import-outside-toplevel

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            self._enc = ort.InferenceSession(
                f"{_MODEL_DIR}/encoder.onnx", providers=["CPUExecutionProvider"], sess_options=opts
            )
            self._dec = ort.InferenceSession(
                f"{_MODEL_DIR}/decoder.onnx", providers=["CPUExecutionProvider"], sess_options=opts
            )
            logger.info("EAR-VAE-Denoiser geladen (%s/encoder.onnx + decoder.onnx)", _MODEL_DIR)
        except Exception as _exc:  # pylint: disable=broad-except
            self._load_error = str(_exc)
            self._enc = None
            self._dec = None
            logger.warning(
                "§V6 (copilot-instructions.md) EAR-VAE-Denoiser nicht ladbar (%s) — DSP-Pfad bleibt aktiv", _exc
            )

    @property
    def available(self) -> bool:
        return self._enc is not None and self._dec is not None

    def denoise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
        """Denoisiert (C, N) oder (N,); None bei Fehler (DSP-Pfad bleibt)."""
        if not self.available:
            return None
        audio = np.asarray(audio, dtype=np.float32)
        was_1d = audio.ndim == 1
        if was_1d:
            audio = audio[None, :]
        elif audio.shape[0] > 2 and audio.shape[1] <= 2:
            audio = audio.T  # (N, C) → (C, N)

        n_ch = audio.shape[0]
        # Mono → Stereo duplizieren (Modellkontrakt ist Stereo).
        if n_ch == 1:
            audio = np.repeat(audio, 2, axis=0)

        try:
            from scipy.signal import resample_poly  # pylint: disable=import-outside-toplevel

            x = (
                resample_poly(audio, _SR_TARGET, sample_rate, axis=1).astype(np.float32)
                if sample_rate != _SR_TARGET
                else audio
            )
            lat = self._enc.run(None, {"audio": x[None]})[0]
            out = self._dec.run(None, {"latent": lat})[0][0]  # (2, N')
            n_out = min(out.shape[1], x.shape[1])
            out = out[:, :n_out]
            # Auf Original-Länge trimmen.
            n_orig = audio.shape[1]
            if n_out > n_orig:
                out = out[:, :n_orig]
            elif n_out < n_orig:
                out = np.pad(out, ((0, 0), (0, n_orig - n_out)))
            if sample_rate != _SR_TARGET:
                out = resample_poly(out, sample_rate, _SR_TARGET, axis=1)[:, :n_orig].astype(np.float32)
            if n_ch == 1:
                out = out[:1]
            if was_1d:
                out = out[0]
            return cast(np.ndarray, out)
        except Exception as _exc:  # pylint: disable=broad-except
            logger.warning(
                "§V6 (copilot-instructions.md) EAR-VAE-Inferenz fehlgeschlagen (%s) — DSP-Pfad bleibt aktiv", _exc
            )
            return None


_INSTANCE: EarVaeDenoiser | None = None


def get_ear_vae_denoiser() -> EarVaeDenoiser:
    """Singleton-Zugriff (lazy load)."""
    global _INSTANCE  # pylint: disable=global-statement
    if _INSTANCE is None:
        _INSTANCE = EarVaeDenoiser()
    return _INSTANCE
