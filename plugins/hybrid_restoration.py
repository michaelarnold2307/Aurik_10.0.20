"""
HybridRestorationPlugin – NaN-sicherer Orchestrierungs-Wrapper (§3.1, §4.4).

Dieses Modul kombiniert mehrere Restaurierungsverfahren in einer robusten
Pipeline: breitbandige Rauschunterdrückung (DeepFilterNet wenn verfügbar,
ansonsten OMLSA-inspirierte spektrale Subtraktion), NaN/Inf-Bereinigung
(§3.1) und True-Peak-Begrenzung.

Algorithm:
    1. NaN/Inf guard (§3.1): nan_to_num → clip to [-1, 1]
    2. Mono/Stereo split — jeder Kanal einzeln verarbeitet
    3. Spectral subtraction (DSP-Fallback):
       - Rauschboden-Schätzung aus erstem Stille-Segment (100 ms)
       - STFT (n_fft=1024, hop=256)
       - Gain-Funktion: G(t,f) = max(sqrt(|S|² - α|N|²) / max(|S|, ε), G_floor)
         mit α=1.0, G_floor=0.10 (OMLSA-konform, §4.5)
       - PGHI nicht verfügbar in DSP-only → phase carry-forward (ISTFT phase)
    4. Optional: DeepFilterNet v3 wenn dfn-pytorch installiert
    5. clip(output, -1.0, 1.0), float32-Ausgabe

Invariants (§3.1):
    - Nil NaN/Inf in output — np.nan_to_num + clip immer angewendet
    - G_floor ≥ 0.10 (kein komplettes Auslöschen)
    - Laufzeit-Budget: ≤ 5 s pro Minute Audio (DSP-only Mode)
    - Thread-safe singleton

References:
    §3.1 Numerische Robustheit
    §3.2 Singleton-Pattern
    §4.4 SOTA-Tabelle: DeepFilterNet v3 (primär) / OMLSA+IMCRA (Fallback)
    §4.5 Pflicht-Algorithmus Rauschunterdrückung: G_floor ≥ 0.10
    Cohen & Berdugo (2002) IMCRA / OMLSA
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------
G_FLOOR: float = 0.10  # §4.5 — Minimum gain (harmonische Bins über HPG)
NOISE_ESTIMATION_MS: float = 100.0  # Erstes Fenster für Rauschboden-Schätzung
ALPHA_NR: float = 1.0  # Over-subtraction factor
N_FFT: int = 1024
HOP_LENGTH: int = 256
TARGET_SR: int = 48_000


# ---------------------------------------------------------------------------
# Core plugin
# ---------------------------------------------------------------------------
class HybridRestorationPlugin:
    """NaN-sicherer Restaurierungs-Wrapper (§3.1/§3.2/§4.4).

    Versucht beim ersten Aufruf DeepFilterNet v3 zu laden; fällt bei
    Nicht-Verfügbarkeit automatisch auf OMLSA-inspirierte Spektral-
    Subtraktion (post-2018 DSP, §4.5) zurück — kein Absturz, kein pass.

    Usage:
        plugin = HybridRestorationPlugin()
        restored = plugin.restore(audio, sr)
    """

    def __init__(self) -> None:
        self._dfn_model: object | None = None
        self._dfn_available: bool = False
        self._initialized: bool = False
        self._init_lock = threading.Lock()
        logger.debug("HybridRestorationPlugin: created, lazy-init pending")

    # ------------------------------------------------------------------
    # Lazy initializer
    # ------------------------------------------------------------------
    def _ensure_initialized(self) -> None:
        """Lazily attempt to load DeepFilterNet v3 (thread-safe)."""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            try:
                from df.enhance import enhance, init_df  # type: ignore[import]

                self._dfn_model = (init_df, enhance)
                self._dfn_available = True
                logger.info("HybridRestorationPlugin: DeepFilterNet v3 geladen ✓")
            except (ImportError, Exception) as exc:
                self._dfn_available = False
                logger.debug(
                    "HybridRestorationPlugin: DeepFilterNet nicht verfügbar (%s), nutze OMLSA-DSP-Fallback",
                    exc,
                )
            self._initialized = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(
        self,
        audio: npt.NDArray[np.float32],
        sr: int,
    ) -> npt.NDArray[np.float32]:
        """Restauriert Audio mit NaN-Schutz und adaptiver Rauschunterdrückung.

        Args:
            audio: float32/64 ndarray mono oder stereo
            sr: Sample-Rate in Hz

        Returns:
            Restauriertes Audio (float32, selbe Shape wie input, clip [-1, 1])
        """
        self._ensure_initialized()

        # --- § 3.1 NaN/Inf guard ---
        audio_f = np.nan_to_num(
            np.asarray(audio, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        audio_f = np.clip(audio_f, -1.0, 1.0)

        if audio_f.size == 0:
            return audio_f

        # --- Try DeepFilterNet v3 (primär, §4.4) ---
        if self._dfn_available and self._dfn_model is not None:
            try:
                result = self._apply_deepfilternet(audio_f, sr)
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                return np.clip(result, -1.0, 1.0).astype(np.float32)
            except Exception as exc:  # pragma: no cover
                logger.warning("DeepFilterNet-Inferenz fehlgeschlagen (%s), nutze DSP-Fallback", exc)

        # --- OMLSA-DSP-Fallback (§4.5) ---
        result = self._apply_spectral_subtraction(audio_f, sr)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(result, -1.0, 1.0).astype(np.float32)

    def restore(
        self,
        audio: npt.NDArray[np.float32],
        sr: int,
    ) -> npt.NDArray[np.float32]:
        """Alias für process() — Drop-in für Legacy-Aufrufe."""
        return self.process(audio, sr)

    # ------------------------------------------------------------------
    # Private: DeepFilterNet v3
    # ------------------------------------------------------------------
    def _apply_deepfilternet(
        self,
        audio: npt.NDArray[np.float32],
        sr: int,
    ) -> npt.NDArray[np.float32]:
        """Wendet DeepFilterNet v3 an (energy_bias = -4 dB für Musik, §4.4)."""
        from typing import Any

        import torch

        _dfn: Any = self._dfn_model
        init_df: Any
        enhance: Any
        init_df, enhance = _dfn

        model, df_state, _ = init_df()
        # For music: reduce energy_bias to preserve harmonics (§4.4 DeepFilterNet row)
        if hasattr(df_state, "set_atten_lim"):
            df_state.set_atten_lim(30.0)  # 30 dB max attenuation (music-safe)

        mono = audio if audio.ndim == 1 else audio.mean(axis=-1)
        tensor_in = torch.from_numpy(mono[None]).float()
        enhanced = enhance(model, df_state, tensor_in)
        out_mono: npt.NDArray[np.float32] = np.asarray(enhanced.squeeze().numpy(), dtype=np.float32)

        if audio.ndim == 1:
            return out_mono
        # Stereo: apply same NR gain to both channels via the mono mask
        gain = np.where(
            np.abs(mono) > 1e-8,
            np.clip(np.abs(out_mono) / np.maximum(np.abs(mono), 1e-8), G_FLOOR, 1.0),
            1.0,
        )
        stereo = audio * gain[:, None] if audio.ndim > 1 else audio * gain
        return np.asarray(stereo, dtype=np.float32)

    # ------------------------------------------------------------------
    # Private: Spectral subtraction (OMLSA-inspired DSP fallback)
    # ------------------------------------------------------------------
    def _apply_spectral_subtraction(
        self,
        audio: npt.NDArray[np.float32],
        sr: int,
    ) -> npt.NDArray[np.float32]:
        """OMLSA-inspirierte spektrale Subtraktion (§4.5 DSP-Fallback).

        Algorithm:
            1. Noise power estimation from first NOISE_ESTIMATION_MS ms
            2. STFT (n_fft=1024, hop=256) — Hanning window
            3. G(t,f) = max(sqrt(|S|² − α·|N|²), 0) / |S| ≥ G_floor
            4. Apply gain to STFT; ISTFT with overlap-add
        """
        stereo = audio.ndim == 2
        channels = [audio[:, c] for c in range(audio.shape[1])] if stereo else [audio]
        outputs = []

        for ch in channels:
            outputs.append(self._denoise_channel(ch, sr))

        if stereo:
            return np.stack(outputs, axis=-1).astype(np.float32)
        return outputs[0].astype(np.float32)

    @staticmethod
    def _denoise_channel(
        mono: npt.NDArray[np.float32],
        sr: int,
    ) -> npt.NDArray[np.float32]:
        """Spektrale Subtraktion auf einem Mono-Kanal."""
        n_fft = N_FFT
        hop = HOP_LENGTH
        n_noise = int(NOISE_ESTIMATION_MS * sr / 1000)
        n_noise = max(n_noise, n_fft)

        window = np.hanning(n_fft).astype(np.float32)

        def stft(x: npt.NDArray[np.float32]):
            frames = []
            for start in range(0, len(x) - n_fft + 1, hop):
                frames.append(np.fft.rfft(x[start : start + n_fft] * window))
            return np.array(frames) if frames else np.zeros((1, n_fft // 2 + 1), dtype=complex)

        def istft(S: np.ndarray, orig_len: int) -> npt.NDArray[np.float32]:
            out = np.zeros(orig_len + n_fft, dtype=np.float64)
            win_sq = np.zeros(orig_len + n_fft, dtype=np.float64)
            w = window.astype(np.float64)
            for i, frame in enumerate(S):
                start = i * hop
                reconstructed = np.fft.irfft(frame, n=n_fft)
                out[start : start + n_fft] += reconstructed * w
                win_sq[start : start + n_fft] += w**2
            # OLA normalization
            mask = win_sq > 1e-12
            out[mask] /= win_sq[mask]
            return out[:orig_len].astype(np.float32)

        # Pad for STFT
        pad_len = n_fft
        padded = np.concatenate([np.zeros(pad_len // 2, dtype=np.float32), mono, np.zeros(n_fft, dtype=np.float32)])
        S = stft(padded)

        # Estimate noise PSD from first n_noise samples
        noise_seg = padded[: n_noise + n_fft]
        S_noise = stft(noise_seg)
        noise_psd = np.mean(np.abs(S_noise) ** 2, axis=0, keepdims=True)
        noise_psd = np.maximum(noise_psd, 1e-30)

        # Gain function: Wiener-style with G_floor
        sig_psd = np.abs(S) ** 2
        gain = np.sqrt(np.maximum(sig_psd - ALPHA_NR * noise_psd, 0.0) / np.maximum(sig_psd, 1e-30))
        gain = np.maximum(gain, G_FLOOR)

        # Apply gain to STFT (keep phase)
        S_enhanced = S * gain

        # ISTFT
        orig_with_pad = len(padded)
        reconstructed = istft(S_enhanced, orig_with_pad)
        # Trim padding
        start = pad_len // 2
        result = reconstructed[start : start + len(mono)]
        # Ensure exact same length
        if len(result) < len(mono):
            result = np.pad(result, (0, len(mono) - len(result)))
        elif len(result) > len(mono):
            result = result[: len(mono)]
        return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Singleton (§3.2)
# ---------------------------------------------------------------------------
_instance: HybridRestorationPlugin | None = None
_lock = threading.Lock()


def get_hybrid_restoration_plugin() -> HybridRestorationPlugin:
    """Thread-safe singleton (Double-Checked Locking, §3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = HybridRestorationPlugin()
    return _instance


def hybrid_restore(
    audio: npt.NDArray[np.float32],
    sr: int,
) -> npt.NDArray[np.float32]:
    """Convenience wrapper — NaN-sichere Restaurierung ohne Klassen-Instantiierung.

    Args:
        audio: float32 audio ndarray (mono or stereo)
        sr: Sample rate in Hz

    Returns:
        Restauriertes Audio float32, clip [-1, 1], NaN-frei.
    """
    return get_hybrid_restoration_plugin().restore(audio, sr)
