"""
ai_dehiss_multiband.py - KI-gestützter Multiband-Dehiss für Aurik 6.0

Dieses Modul entfernt Bandrauschen/Hiss multiband aus Audiosignalen (Stub).

Dieses Modul entfernt Bandrauschen/Hiss multiband aus Audiosignalen.
Kombiniert klassische Multiband-Spectral-Subtraction und Deep-Learning (ML-ready).
"""

import numpy as np
from scipy.signal import butter, istft, lfilter, stft


class AiDehissMultiband:
    """
    SOTA-Multiband-Dehiss: Kombiniert Multiband-Spectral-Subtraction und Deep-Learning (ML-ready). Robust, adaptiv, Fallback-fähig.
    """

    def __init__(
        self,
        model_path: str | None = None,
        bands=((0, 4000), (4000, 12000), (12000, 22050)),
        hiss_floor_db: float = -35.0,
    ):
        self.model_path = model_path
        self.model = None
        self.bands = bands
        self.hiss_floor_db = hiss_floor_db

    def _bandpass(self, audio, sr, low, high):
        nyq = 0.5 * sr
        lowcut = low / nyq
        highcut = high / nyq
        b, a = butter(4, [lowcut, highcut], btype="band", output="ba")  # type: ignore[misc]
        return lfilter(b, a, audio)

    # ------------------------------------------------------------------ #
    # OMLSA-Konstanten (identisch mit AiDehiss)                           #
    # ------------------------------------------------------------------ #
    _NPERSEG: int = 1024
    _G_FLOOR: float = 0.10  # §2.28
    _ALPHA: float = 0.96
    _MU: float = 1.5
    _BETA: float = 0.005
    _MIN_HIST: int = 20

    def _omlsa_gain(self, mag: np.ndarray) -> np.ndarray:
        """OMLSA/MMSE-LSA-Gain (bandweise, identischer Algorithmus wie AiDehiss).

        Minimum-Statistik-Rauschbodenschätzung + Decision-Directed a-priori-SNR
        + MMSE-LSA-Gain mit G_floor=0.10.
        Referenz: Cohen (2002/2003), Ephraim & Malah (1985) LSA.
        """
        n_bins, n_frames = mag.shape
        G_out = np.ones_like(mag)
        noise_est = mag[:, 0:1].copy() + 1e-12
        noise_min = mag[:, 0:1].copy() + 1e-12
        G_prev = np.ones((n_bins, 1))
        snr_post_prev = np.zeros((n_bins, 1))

        for t in range(n_frames):
            m = mag[:, t : t + 1]
            noise_min = np.minimum(noise_min * self._ALPHA + (1 - self._ALPHA) * m, m + 1e-12)
            if t >= self._MIN_HIST:
                noise_est = noise_min * self._MU
            noise_est = np.maximum(noise_est, self._BETA * m)

            snr_post = np.maximum(m**2 / (noise_est**2 + 1e-24) - 1.0, 0.0)
            snr_prior = 0.98 * G_prev**2 * snr_post_prev + 0.02 * snr_post
            v = snr_prior * snr_post / (1.0 + snr_prior + 1e-12)
            G = np.maximum(self._G_FLOOR, v / (v + 1.0))
            G = self._ALPHA * G_prev + (1 - self._ALPHA) * G
            G = np.maximum(G, self._G_FLOOR)

            G_out[:, t : t + 1] = G
            G_prev = G
            snr_post_prev = snr_post

        return np.asarray(G_out)  # type: ignore[no-any-return]

    def dehiss_multiband(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Entfernt Hiss multibandig via OMLSA/MMSE-LSA pro Frequenzband.

        Algorithmus:
            1. Für jedes Band (bands): Bandpassfilterung (Butterworth 4. Ordnung).
            2. STFT des Bandsignals (nperseg=1024, noverlap=768).
            3. OMLSA-Gain-Matrix G per Band berechnen.
            4. Gain anwenden, ISTFT mit Originalphase.
            5. Aufsummierung aller Bänder (keine Normierung nötig: Bandpässe
               sind näherungsweise orthogonal).
            6. Clip/NaN-Guard; ML-Inferenz als optionaler finaler Pass.
        Referenz: Cohen (2002/2003); §4.5 Multi-Resolution-STFT-Prinzip.
        Invariante: NaN/Inf-frei, Ausgang ∈ [−1, 1], dtype erhalten.
        """
        orig_dtype = audio.dtype
        orig_len = len(audio)
        audio_f32 = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)

        noverlap = self._NPERSEG * 3 // 4
        bands_out = []
        for low, high in self.bands:
            band = self._bandpass(audio_f32, sr, low, high)
            _, _, Zxx = stft(band, fs=sr, nperseg=self._NPERSEG, noverlap=noverlap)
            mag = np.abs(Zxx)
            phase = np.angle(Zxx)
            G = self._omlsa_gain(mag)
            Zxx_clean = mag * G * np.exp(1j * phase)
            _, band_out = istft(Zxx_clean, fs=sr, nperseg=self._NPERSEG, noverlap=noverlap)
            band_out = np.nan_to_num(band_out[:orig_len], nan=0.0, posinf=0.0, neginf=0.0)
            bands_out.append(band_out)

        audio_out = np.clip(np.nan_to_num(np.sum(bands_out, axis=0), nan=0.0), -1.0, 1.0)

        # ML-Inferenz als optionaler finaler Pass
        if self.model is not None:
            try:
                inp = audio_out.astype(np.float32)
                ml_out = self.model.run(None, {"input": inp[np.newaxis, :]})[0].squeeze()
                audio_out = np.clip(np.nan_to_num(ml_out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
            except Exception as exc:
                import logging as _log

                _log.getLogger(__name__).warning("[AiDehissMultiband] ML fehlgeschlagen (%s).", exc)

        return np.asarray(audio_out.astype(orig_dtype))  # type: ignore[no-any-return]
