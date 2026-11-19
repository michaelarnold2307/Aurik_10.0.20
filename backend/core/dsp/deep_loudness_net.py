import logging

import numpy as np

logger = logging.getLogger(__name__)


# DeepLoudnessNet → LUFSNormalizer — Klassenname verboten §4.4+§10.2. ITU-R BS.1770 Implementierung bleibt erhalten.
class LUFSNormalizer:
    """
    ITU-R BS.1770 Loudness-Normalisierung (LUFS, Studio-Standard)
    """

    def __init__(self, target_lufs=-14.0):
        self.target_lufs = target_lufs

    def integrated_lufs(self, audio, sr):
        # K-Weighting-Filter (vereinfachte Version)
        b = [1.0, -2.0, 1.0]
        a = [1.0, -1.99004745483398, 0.99007225036621]
        weighted = np.convolve(audio, b, mode="same") / np.convolve(np.ones_like(audio), a, mode="same")
        rms = np.sqrt(np.mean(weighted**2))
        lufs = 20 * np.log10(rms + 1e-12)
        return lufs

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        lufs = self.integrated_lufs(audio, sr)
        # Loudness-DPN Challenger (hinter use_loudness_dpn, Default AUS).
        lufs = _loudness_dpn_challenger(audio, sr, lufs)
        gain = 10 ** ((self.target_lufs - lufs) / 20)
        return np.asarray(audio * gain)  # type: ignore[no-any-return]


# Loudness-DPN (Challenger, hinter use_loudness_dpn, Default AUS). Gelernte LUFS/DR-
# Regression statt vereinfachter K-Weighting-Naeherung (MAE LUFS 2.46 dB, gemessen).
# §V6 (copilot-instructions.md): bei Fehlern/ohne Modell → fallback (integrated_lufs).
_LOUDNESS_DPN_MODEL = None
_LOUDNESS_DPN_LOADED = False


def _load_loudness_dpn():
    global _LOUDNESS_DPN_MODEL, _LOUDNESS_DPN_LOADED
    if _LOUDNESS_DPN_LOADED:
        return _LOUDNESS_DPN_MODEL
    _LOUDNESS_DPN_LOADED = True
    try:
        from pathlib import Path as _P

        import torch

        from backend.models.loudness_dpn.train_loudness_dpn import LoudnessDPN

        _p = _P("models/loudness_dpn.pt")
        if not _p.exists():
            return None
        _ck = torch.load(_p, map_location="cpu", weights_only=False)
        _m = LoudnessDPN()
        _m.load_state_dict(_ck["state_dict"])
        _m.eval()
        _LOUDNESS_DPN_MODEL = _m
    except Exception as _e:  # §V6 (copilot-instructions.md)
        logger.warning("Loudness-DPN nicht geladen: %s", _e)
        _LOUDNESS_DPN_MODEL = None
    return _LOUDNESS_DPN_MODEL


def _loudness_dpn_lufs(audio: np.ndarray, sr: int) -> float | None:
    """DPN-LUFS-Vorhersage. None bei Fehlern (§V6 (copilot-instructions.md))."""
    try:
        import torch

        from backend.models.loudness_dpn.train_loudness_dpn import _logmel

        _m = _load_loudness_dpn()
        if _m is None:
            return None
        _mono = np.mean(audio, axis=-1 if audio.shape[-1] <= 2 else 0) if audio.ndim > 1 else audio
        _x = _logmel(_mono, sr)
        with torch.no_grad():
            _out = _m(torch.tensor(_x[None, None], dtype=torch.float32))[0]
        return float((_out[0].item() - 1.0) * 70.0)  # denorm: LUFS_norm → LUFS
    except Exception:  # §V6 (copilot-instructions.md)
        return None


def _loudness_dpn_challenger(audio: np.ndarray, sr: int, fallback_lufs: float) -> float:
    """Loudness-DPN Challenger (hinter use_loudness_dpn, Default AUS).

    Nutzt die gelernte LUFS-Regression statt der Naeherung. §V6 (copilot-instructions.md):
    bei Fehlern/ohne Modell → fallback_lufs.
    Default AUS = bit-identisch.
    """
    try:
        from backend.core.music_model_flags import use_loudness_dpn as _dpn_on

        if not _dpn_on:
            return fallback_lufs
        _pred = _loudness_dpn_lufs(audio, sr)
        return fallback_lufs if _pred is None else _pred
    except Exception:  # §V6 (copilot-instructions.md)
        return fallback_lufs
