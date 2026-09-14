"""F2-Gabor-Shim: eigene NumPy-DGT für den GaCELA-Upstream-Datenpfad.

Der Upstream (models/gacela_upstream/data/audioLoader.py, MIT) importiert
`tifresi` (GaussTruncTF) — dessen Forward-Pfad hängt an `ltfatpy.dgtreal`
(C-Extension, kein Binary-Wheel, Build-Blocker 2026-09-14). tifresi hat
zudem eine unklare Lizenzlage (MIT/GPLv3 gemischt) → KEIN Vendoring.

Dieses Modul implementiert den für das TRAINING nötigen Vorwärtsweg
(Echt-Signal-Gabor-Transformation, nur Betrags-Spektrogramm) eigenständig
in NumPy und wird dem Upstream per sys.modules-Shim untergeschoben
(scripts/train_gacela_vocal_inpaint.py). Die Inversion (PGHI/gabdual) wird
vom TrainDataset nicht benötigt und ist bewusst nicht implementiert.

Konventionen (identisch zur Upstream-Nutzung, sr 22.050):
  hop_size=256, stft_channels=1024, min_height=1e-4 (Truncated Gaussian),
  log_spectrogram mit dynamic_range_dB=50, danach /(dr/2)+1 (AudioLoader),
  preprocess_signal: librosa-trim + Null-Pad auf Vielfache von 1024.
"""

from __future__ import annotations

import sys
import types

import numpy as np


def truncated_gauss_window(hop_size: int, stft_channels: int, min_height: float = 1e-4) -> np.ndarray:
    """Truncated-Gaussian-Analysefenster (Formel der tifresi-GaussTruncTF-Klasse)."""
    lg_true = np.sqrt(-4.0 * hop_size * stft_channels * np.log(min_height) / np.pi)
    lg_long = int(np.ceil(lg_true / stft_channels) * stft_channels)
    x = (1.0 / lg_true) * np.concatenate([np.arange(0.5 * lg_long), np.arange(-0.5 * lg_long, 0)])
    g = np.exp(4.0 * np.log(min_height) * (x**2))
    g = g / np.linalg.norm(g)
    return g.astype(np.float64)


def dgt_real_mag(x: np.ndarray, g: np.ndarray, hop_size: int, stft_channels: int) -> np.ndarray:
    """Vorwärts-DGT eines reellen Signals — Betragsspektrogramm (513, N).

    Zero-Padding jenseits des Signals (Frame-weise Faltung mit dem Fenster),
    M-Punkt-rFFT. Die LTFAT-Phasenkonvention entfällt, weil nur |·| genutzt wird.
    """
    x = np.asarray(x, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    lg = len(g)
    n_frames = len(x) // hop_size
    out = np.zeros((stft_channels // 2 + 1, n_frames), dtype=np.float64)
    for n in range(n_frames):
        start = n * hop_size
        seg = np.zeros(lg, dtype=np.float64)
        seg_len = min(lg, len(x) - start)
        seg[:seg_len] = x[start : start + seg_len]
        spec = np.fft.rfft(seg * g, n=stft_channels)
        out[:, n] = np.abs(spec)
    return out


def log_spectrogram(spectrogram: np.ndarray, dynamic_range_dB: float = 50.0) -> np.ndarray:
    """Log-Spektrogramm (Formel der tifresi-transforms-Signatur)."""
    spec = np.abs(np.asarray(spectrogram, dtype=np.float64))
    peak = float(np.max(spec)) + 1e-12
    minimum_relative = peak / 10 ** (dynamic_range_dB / 10)
    return 10.0 * np.log10(np.clip(spec, a_min=minimum_relative, a_max=None))


def preprocess_signal(y: np.ndarray, m: int = 1024) -> np.ndarray:
    """librosa-trim + Null-Pad auf Vielfache von m (tifresi-utils-Äquivalent)."""
    import librosa

    y = np.asarray(y, dtype=np.float64)
    y_trimmed, _ = librosa.effects.trim(y)
    left_over = int(np.mod(len(y_trimmed), m))
    if left_over:
        y_trimmed = np.pad(y_trimmed, (0, m - left_over))
    return y_trimmed


class GaussTruncTFShim:
    """Drop-in-Ersatz für tifresi.stft.GaussTruncTF (Forward-Pfad, Betrag)."""

    def __init__(self, hop_size: int = 256, stft_channels: int = 1024, min_height: float = 1e-4) -> None:
        assert np.mod(stft_channels, 2) == 0, "stft_channels muss gerade sein"
        self.hop_size = hop_size
        self.stft_channels = stft_channels
        self._g = truncated_gauss_window(hop_size, stft_channels, min_height)

    def spectrogram(self, time_signal: np.ndarray, normalize: bool = True, **kwargs) -> np.ndarray:  # type: ignore[no-untyped-def]
        mag = dgt_real_mag(time_signal, self._g, self.hop_size, self.stft_channels)
        if normalize and np.max(mag) > 0.0:
            mag = mag / np.max(mag)
        return mag


def install_tifresi_shim() -> None:
    """Legt Fake-Pakete tifresi/tifresi.stft/transforms/utils in sys.modules.

    Danach importiert `data.audioLoader` im Upstream fehlerfrei und nutzt
    diese NumPy-Implementierung statt ltfatpy.
    """
    if "tifresi" in sys.modules:
        return
    pkg = types.ModuleType("tifresi")
    stft_mod = types.ModuleType("tifresi.stft")
    stft_mod.GaussTruncTF = GaussTruncTFShim  # type: ignore[attr-defined]
    transforms_mod = types.ModuleType("tifresi.transforms")
    transforms_mod.log_spectrogram = log_spectrogram  # type: ignore[attr-defined]
    utils_mod = types.ModuleType("tifresi.utils")
    utils_mod.preprocess_signal = preprocess_signal  # type: ignore[attr-defined]
    pkg.stft = stft_mod  # type: ignore[attr-defined]
    pkg.transforms = transforms_mod  # type: ignore[attr-defined]
    pkg.utils = utils_mod  # type: ignore[attr-defined]
    sys.modules["tifresi"] = pkg
    sys.modules["tifresi.stft"] = stft_mod
    sys.modules["tifresi.transforms"] = transforms_mod
    sys.modules["tifresi.utils"] = utils_mod
