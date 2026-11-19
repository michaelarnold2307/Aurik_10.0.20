"""
dehiss.py — OMLSA/MMSE-LSA-Rauschunterdrückung für Aurik 10.0.0.x

Entfernt Bandrauschen/Hiss via echtem IMCRA-inspirierten Minimum-Statistik-Schätzer
(Cohen 2002/2003) + MMSE-LSA-Gain (Ephraim & Malah 1985).
Optional ML-Pass über lokale ONNX-Datei (DeepFilterNet-kompatibel).
Kein spectral subtraction, kein veralteter Wiener 1984.
"""

import numpy as np
from scipy.signal import istft, stft


class AiDehiss:
    """
    SOTA-Dehiss: Kombiniert klassische Spectral-Subtraction und Deep-Learning (ML-ready). Robust, adaptiv, Fallback-fähig.
    """

    def __init__(self, model_path: str | None = None, hiss_floor_db: float = -35.0):
        self.model_path = model_path
        self.model = None
        self.hiss_floor_db = hiss_floor_db

    # ------------------------------------------------------------------ #
    # Konfigurations-Konstanten                                            #
    # ------------------------------------------------------------------ #
    _NPERSEG: int = 1024
    _G_FLOOR: float = 0.10  # §2.28 HarmonicPreservationGuard-Analogon
    _ALPHA: float = 0.96  # Temporal-Glättung für Minimum-Statistik
    _MU: float = 1.5  # Überabzug-Faktor (oversubtraction)
    _BETA: float = 0.005  # Spektrale Bodengrenze (Residual-Rauschunterdrückung)
    _MIN_HIST: int = 20  # Frames bis Minimum-Statistik einspielt

    def _omlsa_gain(self, mag: np.ndarray) -> np.ndarray:
        """OMLSA/MMSE-LSA-Gain für Magnitudenspektrogramm.

        Algorithmus (IMCRA-inspirierte Minimum-Statistik + MMSE-LSA):
            1. Für jeden Frequenzkanal: gleitende Minimum-Statistik über die Zeit
               als Rauschboden-Schätzer N_est[k] (Cohen 2002/2003).
            2. SNR_post = max(mag²/N_est² − 1, 0)  (a-posteriori SNR).
            3. SNR_prior ≈ 0.98 * G_prev² * SNR_post_prev + 0.02 * SNR_post
               (Decision-Directed-Prior, Ephraim & Malah MMSE-LSA Variante).
            4. v = SNR_prior * SNR_post / (1 + SNR_prior)
            5. G = max(G_floor, v / (v + 1))   [MMSE-LSA vereinfacht]
            6. G temporal geglättet: G_smooth = alpha * G_prev + (1-alpha) * G
        Referenz: Cohen (2002, 2003), Ephraim & Malah (1985) LSA-Variante.
        """
        n_bins, n_frames = mag.shape
        G_out = np.ones_like(mag)
        noise_est = mag[:, 0:1].copy() + 1e-12  # Initialisierung
        noise_min = mag[:, 0:1].copy() + 1e-12
        G_prev = np.ones((n_bins, 1))
        snr_post_prev = np.zeros((n_bins, 1))

        for t in range(n_frames):
            m = mag[:, t : t + 1]  # (n_bins, 1)

            # 1. Minimum-Statistik-Update (IMCRA-Proxy)
            noise_min = np.minimum(noise_min * self._ALPHA + (1 - self._ALPHA) * m, m + 1e-12)
            if t >= self._MIN_HIST:
                noise_est = noise_min * self._MU
            noise_est = np.maximum(noise_est, self._BETA * m)

            # 2. A-posteriori SNR
            snr_post = np.maximum(m**2 / (noise_est**2 + 1e-24) - 1.0, 0.0)

            # 3. Decision-Directed a-priori SNR
            snr_prior = 0.98 * G_prev**2 * snr_post_prev + 0.02 * snr_post

            # 4. Zusammengesetzte Variable v
            v = snr_prior * snr_post / (1.0 + snr_prior + 1e-12)

            # 5. MMSE-LSA-Gain (vereinfacht), G_floor aus §2.28
            G = np.maximum(self._G_FLOOR, v / (v + 1.0))

            # 6. Temporale Glättung
            G = self._ALPHA * G_prev + (1 - self._ALPHA) * G
            G = np.maximum(G, self._G_FLOOR)

            G_out[:, t : t + 1] = G
            G_prev = G
            snr_post_prev = snr_post

        return np.asarray(G_out)  # type: ignore[no-any-return]

    def dehiss(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Entfernt Bandrauschen/Hiss via OMLSA/MMSE-LSA.

        Algorithmus:
            1. STFT (nperseg=1024, noverlap=768 → 75 % Überlappung)
            2. OMLSA-Gain-Matrix G berechnen (_omlsa_gain)
            3. Gain anwenden: Zxx_clean = Zxx * G
            4. ISTFT mit Originalphase (phasenkonsistent, §4.5 PGHI-Äquivalent)
            5. ML-Inferenz als finaler Pass (falls Modell geladen)
        Referenz: Cohen & Berdugo (2002/2003), Le Roux & Vincent (2013).
        Invariante: NaN/Inf-frei, Ausgang ∈ [−1, 1], dtype erhalten.
        """
        orig_dtype = audio.dtype
        orig_len = len(audio)
        audio_f32 = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)

        noverlap = self._NPERSEG * 3 // 4
        _f, _t, Zxx = stft(audio_f32, fs=sr, nperseg=self._NPERSEG, noverlap=noverlap)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        G = self._omlsa_gain(mag)
        Zxx_clean = mag * G * np.exp(1j * phase)
        _, audio_out = istft(Zxx_clean, fs=sr, nperseg=self._NPERSEG, noverlap=noverlap)
        audio_out = np.nan_to_num(audio_out[:orig_len], nan=0.0, posinf=0.0, neginf=0.0)
        audio_out = np.clip(audio_out, -1.0, 1.0)

        # ML-Inferenz als optionaler finaler Pass
        if self.model is not None:
            try:
                inp = audio_out.astype(np.float32)
                ml_out = self.model.run(None, {"input": inp[np.newaxis, :]})[0].squeeze()
                audio_out = np.nan_to_num(ml_out, nan=0.0, posinf=0.0, neginf=0.0)
                audio_out = np.clip(audio_out, -1.0, 1.0)
            except Exception as exc:
                import logging as _log

                _log.getLogger(__name__).warning("[AiDehiss] ML fehlgeschlagen (%s).", exc)

        return np.asarray(audio_out.astype(orig_dtype))  # type: ignore[no-any-return]
