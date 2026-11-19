"""
visqol_plugin.py — DSP-Ersatz fuer ViSQOL v3 (kein Docker, kein Download).

NSIM-basierter Audio-Qualitaetsscore als Proxy fuer ViSQOL --audio Mode.
Gammatone-aehnliche Bark-Filterbank + SSIM-Variante -> MOS [1.0-5.0].
"""

from __future__ import annotations

import itertools
import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_inst: VisqolPlugin | None = None

_BARK_EDGES_HZ = [
    20,
    100,
    200,
    300,
    400,
    510,
    630,
    770,
    920,
    1080,
    1270,
    1480,
    1720,
    2000,
    2320,
    2700,
    3150,
    3700,
    4400,
    5300,
    6400,
    7700,
    9500,
    12000,
    15500,
]


class VisqolPlugin:
    """NSIM-basierter ViSQOL-Proxy (rein DSP, kein ML-Modell benoetigt).

    Ref: Chinen et al. (2020) ViSQOL v3 -- Musik/Vollband-Erweiterung.
    Aurik: nur --audio Mode (Vollband 20-20000 Hz), kein Speech-Mode.
    """

    N_FFT = 2048
    HOP = 512

    def score(self, reference: np.ndarray, degraded: np.ndarray, sr: int = 48000) -> float:
        """MOS-Proxy via NSIM auf Bark-Energie-Matrizen, [1.0-5.0].

        Algorithmus:
            1. Mono-Konvertierung, Laengenanpassung per zero-pad
            2. STFT (Hanning, N_FFT=2048, hop=512)
            3. Bark-Band-Energie pro Frame (24 Baender)
            4. NSIM zwischen Referenz- und degradierter Bark-Matrix
            5. NSIM -> MOS via Sigmoid-Streckung
        """
        ref = self._to_mono(reference)
        deg = self._to_mono(degraded)
        n = max(len(ref), len(deg))
        ref = np.pad(ref, (0, max(0, n - len(ref))))
        deg = np.pad(deg, (0, max(0, n - len(deg))))
        ref_bark = self._bark_matrix(ref, sr)
        deg_bark = self._bark_matrix(deg, sr)
        nsim = self._nsim(ref_bark, deg_bark)
        mos = 1.0 + 4.0 / (1.0 + np.exp(-(nsim - 0.5) * 8.0))
        return float(np.clip(mos, 1.0, 5.0))

    def score_absolute(self, audio: np.ndarray, sr: int = 48000) -> float:
        """Referenzfreier Score via Spektralflachheit-Proxy, [1.0-5.0]."""
        mono = self._to_mono(audio)
        from scipy.signal import stft as _stft

        _, _, Z = _stft(mono, fs=sr, window="hann", nperseg=self.N_FFT, noverlap=self.N_FFT - self.HOP)
        mag = np.abs(Z) + 1e-9
        geom = np.exp(np.log(mag).mean(axis=0))
        arith = mag.mean(axis=0)
        flat = float(np.clip((geom / arith).mean(), 0.0, 1.0))
        # Hohe Flachheit = Rauschen/schlechte Qualitaet
        mos = 1.0 + 4.0 * (1.0 - flat**0.5)
        return float(np.clip(mos, 1.0, 5.0))

    # ------------------------------------------------------------------
    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        a = np.array(audio, dtype=np.float32)
        if a.ndim == 2:
            a = a.mean(axis=0) if a.shape[0] <= 8 else a.mean(axis=1)
        return np.nan_to_num(a, nan=0.0)  # type: ignore[no-any-return]

    def _bark_matrix(self, mono: np.ndarray, sr: int) -> np.ndarray:
        from scipy.signal import stft as _stft

        _, _, Z = _stft(mono, fs=sr, window="hann", nperseg=self.N_FFT, noverlap=self.N_FFT - self.HOP)
        mag2 = np.abs(Z) ** 2
        freqs = np.fft.rfftfreq(self.N_FFT, 1.0 / sr)
        bands = []
        for lo, hi in itertools.pairwise(_BARK_EDGES_HZ):
            mask = (freqs >= lo) & (freqs < hi)
            if mask.any():
                bands.append(mag2[mask].mean(axis=0))
            else:
                bands.append(np.zeros(mag2.shape[1], dtype=np.float32))
        return np.stack(bands, axis=0)  # type: ignore[no-any-return]

    def _nsim(self, ref: np.ndarray, deg: np.ndarray, k1: float = 0.01, k2: float = 0.03) -> float:
        L = max(float(ref.max()), float(deg.max()), 1e-9)
        c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
        mu_r, mu_d = ref.mean(), deg.mean()
        sig_r, sig_d = ref.var(), deg.var()
        sig_rd = float(np.mean((ref - mu_r) * (deg - mu_d)))
        num = (2 * mu_r * mu_d + c1) * (2 * sig_rd + c2)
        den = (mu_r**2 + mu_d**2 + c1) * (sig_r + sig_d + c2)
        return float(np.clip(num / (den + 1e-12), 0.0, 1.0))


def get_visqol_plugin() -> VisqolPlugin:
    global _inst
    if _inst is None:
        with _lock:
            if _inst is None:
                _inst = VisqolPlugin()
    return _inst


def get_loaded_visqol_plugin() -> VisqolPlugin | None:
    """Gibt nur eine bereits geladene ViSQOL-Instanz zurück, ohne Lazy-Load."""
    return _inst


def score_audio(reference: np.ndarray, degraded: np.ndarray, sr: int = 48000) -> float:
    """Convenience-Wrapper. Gibt MOS-Proxy in [1.0, 5.0] zurueck."""
    return get_visqol_plugin().score(reference, degraded, sr)


# calculate() mit mode/ref_wav Signatur (Kompatibilität mit Test-Interface)
def _visqol_calculate(self, audio, sr=48000, mode="audio", ref_wav=None, **kwargs):
    """Kompatibilitäts-Wrapper: mode='audio' oder ref_wav Parameter."""
    if ref_wav is not None:
        return self.score(ref_wav, audio, sr)
    return self.score_absolute(audio, sr)


VisqolPlugin.calculate = _visqol_calculate  # type: ignore[attr-defined]

# Alias für Rückwärtskompatibilität
ViSQOLPlugin = VisqolPlugin
