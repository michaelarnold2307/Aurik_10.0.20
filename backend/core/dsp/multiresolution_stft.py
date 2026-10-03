"""
SOTA-Maximum DSP-Modul für Aurik 6.0: AdaptiveSTFT & AdaptiveMelSpectrogram
Ermöglicht adaptive, dynamisch optimierte STFT/ISTFT und Mel-Spektrogramme für maximale Produktivität.
"""

import logging
from typing import Any

import librosa
import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveSTFT:
    """Adaptive STFT/ISTFT-Klasse mit automatischer Parameteroptimierung."""

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.win_length = win_length or n_fft
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

    def stft(self, y: np.ndarray, sr: int | None = None, **kwargs: Any) -> np.ndarray:  # pylint: disable=unused-argument
        """
        Normkonform: Quality-Gate, Audit-Logging, robuste Fehlerbehandlung
        Berechnet die STFT adaptiv mit aktuellen Parametern.
        """
        self._log_contract("stft")
        try:
            if not isinstance(y, np.ndarray) or y.size == 0:
                raise ValueError("Ungültige Eingabe für AdaptiveSTFT.stft")
            # Guard: n_fft must not exceed signal length; clamp to next-lower power of 2.
            raw_n_fft = kwargs.get("n_fft", self.n_fft)
            safe_n_fft = min(raw_n_fft, max(32, 1 << int(np.floor(np.log2(max(1, len(y)))))))
            result = librosa.stft(
                y,
                n_fft=safe_n_fft,
                hop_length=kwargs.get("hop_length", self.hop_length),
                win_length=kwargs.get("win_length", min(kwargs.get("win_length", self.win_length), safe_n_fft)),
                window=kwargs.get("window", self.window),
                center=kwargs.get("center", self.center),
                pad_mode=kwargs.get("pad_mode", self.pad_mode),
            )
            self._audit_log({"func": "stft", "shape": result.shape, "success": True})
            return np.asarray(result)  # type: ignore[no-any-return]
        except Exception as e:
            logger.error("[AdaptiveSTFT][Fehler][stft] %s", e)
            self._audit_log({"func": "stft", "error": str(e)})
            return np.asarray(np.zeros((self.n_fft // 2 + 1, 1)))  # type: ignore[no-any-return]

    def istft(self, D: np.ndarray, sr: int | None = None, **kwargs: Any) -> np.ndarray:  # pylint: disable=unused-argument
        """
        Normkonform: Quality-Gate, Audit-Logging, robuste Fehlerbehandlung
        Inverse STFT mit adaptiven Parametern.
        """
        self._log_contract("istft")
        try:
            if not isinstance(D, np.ndarray) or D.size == 0:
                raise ValueError("Ungültige Eingabe für AdaptiveSTFT.istft")
            result = librosa.istft(
                D,
                hop_length=kwargs.get("hop_length", self.hop_length),
                win_length=kwargs.get("win_length", self.win_length),
                window=kwargs.get("window", self.window),
                center=kwargs.get("center", self.center),
                length=kwargs.get("length"),
            )
            self._audit_log({"func": "istft", "shape": result.shape, "success": True})
            return np.asarray(result)  # type: ignore[no-any-return]
        except Exception as e:
            logger.error("[AdaptiveSTFT][Fehler][istft] %s", e)
            self._audit_log({"func": "istft", "error": str(e)})
            return np.asarray(np.zeros((1,)))  # type: ignore[no-any-return]

    def _log_contract(self, func: str):
        logger.info("[Contract][AdaptiveSTFT] %s(...) -> np.ndarray", func)

    def _audit_log(self, result: dict[str, Any]):
        logger.info("[AuditLog][AdaptiveSTFT] Ergebnis: %s", result)

    def auto_optimize(self, y: np.ndarray, sr: int) -> None:  # pylint: disable=unused-argument
        """Automatische Anpassung der Parameter je nach Signal (SOTA-Ansatz)."""
        # Beispiel: Passe n_fft an die Signal-Länge an
        if len(y) < 4096:
            self.n_fft = 512
            self.hop_length = 128
        elif len(y) < 16384:
            self.n_fft = 1024
            self.hop_length = 256
        else:
            self.n_fft = 2048
            self.hop_length = 512
        # win_length immer konsistent zu n_fft setzen
        self.win_length = self.n_fft


class AdaptiveMelSpectrogram:
    """Adaptive Mel-Spektrogramm-Klasse mit automatischer Parameteroptimierung."""

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int | None = None,
        n_mels: int = 128,
        sr: int = 22050,
        fmin: int = 0,
        fmax: int | None = None,
        power: float = 2.0,
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.n_mels = n_mels
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.power = power

    def mel_spectrogram(self, y: np.ndarray, sr: int | None = None, **kwargs: Any) -> np.ndarray:
        """
        Normkonform: Quality-Gate, Audit-Logging, robuste Fehlerbehandlung
        Berechnet ein adaptives Mel-Spektrogramm.
        """
        self._log_contract("mel_spectrogram")
        try:
            if not isinstance(y, np.ndarray) or y.size == 0:
                raise ValueError("Ungültige Eingabe für AdaptiveMelSpectrogram.mel_spectrogram")
            result = librosa.feature.melspectrogram(
                y=y,
                sr=sr or self.sr,
                n_fft=kwargs.get("n_fft", self.n_fft),
                hop_length=kwargs.get("hop_length", self.hop_length),
                n_mels=kwargs.get("n_mels", self.n_mels),
                fmin=kwargs.get("fmin", self.fmin),
                fmax=kwargs.get("fmax", self.fmax),
                power=kwargs.get("power", self.power),
            )
            self._audit_log({"func": "mel_spectrogram", "shape": result.shape, "success": True})
            return np.asarray(result)  # type: ignore[no-any-return]
        except Exception as e:
            logger.error("[AdaptiveMelSpectrogram][Fehler][mel_spectrogram] %s", e)
            self._audit_log({"func": "mel_spectrogram", "error": str(e)})
            return np.asarray(np.zeros((self.n_mels, 1)))  # type: ignore[no-any-return]

    def _log_contract(self, func: str):
        logger.info("[Contract][AdaptiveMelSpectrogram] %s(...) -> np.ndarray", func)

    def _audit_log(self, result: dict[str, Any]):
        logger.info("[AuditLog][AdaptiveMelSpectrogram] Ergebnis: %s", result)

    def transform(self, y: np.ndarray, sr: int | None = None, **kwargs: Any) -> np.ndarray:
        """Alias für mel_spectrogram — ermöglicht einheitliche transform()-Schnittstelle."""
        return self.mel_spectrogram(y, sr=sr, **kwargs)

    def auto_optimize(self, y: np.ndarray, sr: int) -> None:  # pylint: disable=unused-argument
        """Automatische Anpassung der Mel-Parameter je nach Signal."""
        if sr < 16000:
            self.n_mels = 64
        elif sr < 32000:
            self.n_mels = 128
        else:
            self.n_mels = 256
