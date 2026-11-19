"""
masking_remover.py - Masking-Remover (AI) für Aurik 6.0
masking_remover.py - SOTA-Masking-Remover (AI) für Aurik 6.0

Dieses Modul reduziert psychoakustisches Maskieren AI-gestützt.
Kombiniert klassische Spectral-Contrast-Enhancement und Deep-Learning (ML-ready).
"""

import logging

import numpy as np
from scipy.signal import istft, stft

_logger = logging.getLogger(__name__)


class MaskingRemover:
    """
    SOTA-Masking-Remover: Kombiniert klassische Spectral-Contrast-Enhancement und Deep-Learning (ML-ready). Robust, adaptiv, Fallback-fähig.
    """

    def __init__(self, model_path: str | None = None, contrast: float = 1.2):
        self.model_path = model_path
        self.model = None
        self.contrast = contrast

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Normkonform: Quality-Gate, Audit-Logging, robuste Fehlerbehandlung, DL-Inferenz-Platzhalter, Doku als Code
        Reduziert psychoakustisches Maskieren. Zuerst klassische Spectral-Contrast-Enhancement, dann optional Deep-Learning-Inferenz (wenn Modell geladen).
        """
        self._log_contract()
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
                raise ValueError("Ungültige Eingabe für MaskingRemover")
            _f, _t, Zxx = stft(audio, fs=sr, nperseg=1024)
            mag = np.abs(Zxx)
            mag_mean = np.mean(mag, axis=1, keepdims=True)
            mag_enh = mag_mean + self.contrast * (mag - mag_mean)
            mag_enh = np.clip(mag_enh, 0.0, None)
            # PGHI phase reconstruction (§4.5 — modifiziertes Betragsspektrum erfordert PGHI)
            try:
                from backend.core.dsp.pghi import pghi_reconstruct

                audio_out = pghi_reconstruct(mag_enh, sr=sr, win_size=1024, hop=512)
            except Exception as pghi_err:
                _logger.warning(
                    "PGHI fehlgeschlagen (%s) — iSTFT mit Originalphase (kein Griffin-Lim, §4.5)",
                    pghi_err,
                )
                # Use original phase from Zxx — never Griffin-Lim (§4.5 forbids it)
                _, audio_out = istft(mag_enh * np.exp(1j * np.angle(Zxx)), fs=sr, nperseg=1024, noverlap=512)
            audio_out = audio_out[: len(audio)]
            # ML-Inferenz via ONNX (wenn Modell geladen)
            if self.model is not None:
                try:
                    _in_name = self.model.get_inputs()[0].name
                    _inp = audio_out[np.newaxis, :].astype(np.float32)
                    _raw = self.model.run(None, {_in_name: _inp})[0].squeeze()
                    _raw = np.nan_to_num(_raw.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                    audio_out = np.clip(_raw, -1.0, 1.0)
                except Exception as _onnx_err:
                    _logger.warning(
                        "MaskingRemover: ONNX-Inferenz fehlgeschlagen (%s) — Spectral-Contrast-DSP-Fallback aktiv.",
                        _onnx_err,
                    )
            self._audit_log({"shape": audio_out.shape, "success": True})
            return np.asarray(np.clip(np.nan_to_num(audio_out, nan=0.0), -1.0, 1.0).astype(audio.dtype))  # type: ignore[no-any-return]
        except Exception as e:
            _logger.error("MaskingRemover: Fehler bei Verarbeitung: %s", e)
            self._audit_log({"error": str(e)})
            return audio

    def _log_contract(self) -> None:
        _logger.debug("[Contract][MaskingRemover] process(audio, sr) -> np.ndarray")

    def _audit_log(self, result: dict) -> None:
        _logger.debug("[AuditLog][MaskingRemover] %s", result)
