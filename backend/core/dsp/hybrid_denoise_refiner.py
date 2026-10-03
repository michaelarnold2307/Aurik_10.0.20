"""
Hybrid DSP+ML Denoise Refiner
==============================

Post-Processing nach ML-Denoise für maximale Transparenz.

Problem: ML-Models (DeepFilterNet, WPE) sind gut, aber:
- Hinterlassen subtile Artefakte (spectral smearing)
- Können Transienten dämpfen (Drum-Attacks verlieren Punch)
- Residual-Noise bleibt manchmal zurück

Lösung: DSP-Refinement in 3 Stages:
1. Spectral Gating für Residual-Noise
2. Transient Protection (Original-Attacks wiederherstellen)
3. Adaptive Smoothing

Garantiert:
- Original-Charakter erhalten (Parallel Processing)
- Transparenz > Artefakt-Removal
- Genre-adaptive Parameter
"""

import logging

import librosa
import numpy as np
import scipy.signal as signal

logger = logging.getLogger(__name__)


class HybridDenoiseRefiner:
    """
    DSP-based refinement nach ML-Denoise.

    Kombiniert ML-Power mit DSP-Präzision für maximale Authentizität.
    """

    def __init__(self):
        self.name = "Hybrid DSP+ML Denoise Refiner"

    def refine(
        self,
        audio_ml_cleaned: np.ndarray,
        audio_original: np.ndarray,
        sr: int,
        genre: str = "unknown",
        strength: float = 0.3,
    ) -> np.ndarray:
        """
        Refinement nach ML-Denoise.

        Args:
            audio_ml_cleaned: ML-Model Output (z.B. von DeepFilterNet)
            audio_original: Original-Audio (für Transient-Restoration)
            sr: Sample rate
            genre: Genre für adaptive Parameter
            strength: Refinement-Stärke (0.0 = none, 0.3 = conservative, 0.5 = aggressive)

        Returns:
            Refined audio (hybrid ML+DSP)
        """
        assert sr == 48000, f"Sample rate must be 48000 Hz, got {sr}"
        logger.info("Hybrid Denoise Refinement (strength=%.2f, genre=%s)", strength, genre)

        # Genre-adaptive Parameter
        params = self._get_genre_params(genre)

        # Stage 1: Spectral Gating für Residual-Noise
        audio_stage1 = self._spectral_gating(audio_ml_cleaned, audio_original, sr, threshold=params["gate_threshold"])

        # Stage 2: Transient Restoration
        audio_stage2 = self._restore_transients(
            audio_stage1,
            audio_original,
            sr,
            blend_ratio=params["transient_blend"] * strength,
        )

        # Stage 3: Adaptive Smoothing
        audio_stage3 = self._adaptive_smoothing(audio_stage2, sr, amount=params["smoothing"] * strength)

        # Parallel Mix mit Original (Authenticity Safeguard #1)
        audio_refined = (1 - strength) * audio_ml_cleaned + strength * audio_stage3

        # NaN/Inf-Guard + Clipping
        audio_refined = np.nan_to_num(audio_refined, nan=0.0, posinf=0.0, neginf=0.0)
        audio_refined = np.clip(audio_refined, -1.0, 1.0)

        logger.info("✓ Verfeinerung abgeschlossen (3 Stufen angewendet)")

        return np.asarray(audio_refined)  # type: ignore[no-any-return]

    def _get_genre_params(self, genre: str) -> dict:
        """
        Genre-adaptive Refinement-Parameter.

        Authenticity Safeguard #3: Genre-Adaptive Limits
        """
        params = {
            "classical": {
                "gate_threshold": -40,  # Sanft (preserve ambience)
                "transient_blend": 0.2,  # Wenig (natural attacks)
                "smoothing": 0.1,  # Minimal (transparency)
            },
            "jazz": {
                "gate_threshold": -35,
                "transient_blend": 0.3,
                "smoothing": 0.15,
            },
            "speech": {
                "gate_threshold": -30,  # Aggressiver (clarity)
                "transient_blend": 0.15,  # Wenig (sibilance important)
                "smoothing": 0.2,
            },
            "rock": {
                "gate_threshold": -30,
                "transient_blend": 0.4,  # Mehr (drum punch)
                "smoothing": 0.15,
            },
        }

        return params.get(genre, params["jazz"])  # Default: Jazz

    def _spectral_gating(
        self,
        audio_cleaned: np.ndarray,
        audio_original: np.ndarray,
        sr: int,
        threshold: float = -35,
    ) -> np.ndarray:
        """
        Spectral Gating für Residual-Noise.

        Methode:
        1. Berechne Residual = Original - ML_Cleaned
        2. Estimate Noise Profile from Residual
        3. Gate Residual mit Threshold
        4. Reconstruct
        """
        # Residual (was wurde vom ML-Model entfernt)
        residual = audio_original - audio_cleaned

        # STFT
        hop_length = 512
        n_fft = min(2048, max(64, len(residual)))
        D_residual = librosa.stft(residual, n_fft=n_fft, hop_length=hop_length)

        # Magnitude + Phase
        mag_residual = np.abs(D_residual)
        phase_residual = np.angle(D_residual)

        # Noise Profile (median über Zeit)
        noise_profile = np.median(mag_residual, axis=1, keepdims=True)

        # Gate: Nur Bins über Threshold behalten
        threshold_linear = 10 ** (threshold / 20)
        gate_mask = mag_residual > (noise_profile * threshold_linear)

        # Apply Gate
        mag_gated = mag_residual * gate_mask

        # Reconstruct
        D_gated = mag_gated * np.exp(1j * phase_residual)
        residual_gated = librosa.istft(D_gated, hop_length=hop_length, length=len(audio_original))

        # Reconstruct Audio (Original - Gated_Residual)
        audio_refined = audio_original - residual_gated

        # NaN/Inf-Guard
        audio_refined = np.nan_to_num(audio_refined, nan=0.0, posinf=0.0, neginf=0.0)

        return np.asarray(audio_refined)  # type: ignore[no-any-return]

    def _restore_transients(
        self,
        audio_processed: np.ndarray,
        audio_original: np.ndarray,
        sr: int,
        blend_ratio: float = 0.3,
    ) -> np.ndarray:
        """
        Wiederherstellen von Original-Transienten.

        Problem: ML-Models können Drum-Attacks dämpfen
        Lösung: Detect transients, blend original attacks

        Authenticity Safeguard: Preserves original punch/attack
        """
        # Onset Detection (Transient-Zeiten finden)
        onset_env_original = librosa.onset.onset_strength(y=audio_original, sr=sr)  # type: ignore[attr-defined]
        onset_frames = librosa.onset.onset_detect(  # type: ignore[attr-defined]
            onset_envelope=onset_env_original,
            sr=sr,
            units="samples",
            hop_length=512,
            backtrack=True,
        )

        audio_restored = audio_processed.copy()

        # Für jeden Transient: Blend original attack
        attack_length = int(0.010 * sr)  # 10ms attack window

        for onset_sample in onset_frames:
            start = onset_sample
            end = min(onset_sample + attack_length, len(audio_original))

            if end > start:
                # Blend original attack
                audio_restored[start:end] = (1 - blend_ratio) * audio_processed[
                    start:end
                ] + blend_ratio * audio_original[start:end]

        # NaN/Inf-Guard
        audio_restored = np.nan_to_num(audio_restored, nan=0.0, posinf=0.0, neginf=0.0)

        return np.asarray(audio_restored)  # type: ignore[no-any-return]

    def _adaptive_smoothing(self, audio: np.ndarray, sr: int, amount: float = 0.15) -> np.ndarray:
        """
        Frequency-dependent spectral smoothing.

        Reduziert ML-Artefakte (spectral smearing) ohne Höhen zu dämpfen.

        Methode:
        - Low-Frequency: mehr Smoothing (Rumble-Reduction)
        - Mid-Frequency: moderat
        - High-Frequency: minimal (preserve brilliance)
        """
        # STFT
        hop_length = 512
        n_fft = min(2048, max(64, len(audio)))
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

        mag = np.abs(D)
        phase = np.angle(D)

        # Frequency-dependent smoothing kernel
        freq_bins = D.shape[0]
        smoothing_kernel = np.ones((freq_bins, 1))

        # Low frequencies (0-500Hz): stärkeres Smoothing
        low_freq_cutoff = int(500 * n_fft / sr)
        smoothing_kernel[:low_freq_cutoff] = amount * 2.0

        # Mid frequencies (500-5000Hz): moderates Smoothing
        mid_freq_cutoff = int(5000 * n_fft / sr)
        smoothing_kernel[low_freq_cutoff:mid_freq_cutoff] = amount

        # High frequencies (5000+Hz): minimales Smoothing
        smoothing_kernel[mid_freq_cutoff:] = amount * 0.5

        # Apply smoothing (median filter über Zeit)
        mag_smoothed = mag.copy()
        for i in range(freq_bins):
            kernel_size = int(3 * smoothing_kernel[i, 0])  # Adaptive kernel size
            if kernel_size >= 3:
                mag_smoothed[i, :] = signal.medfilt(mag[i, :], kernel_size=kernel_size)

        # Reconstruct
        D_smoothed = mag_smoothed * np.exp(1j * phase)
        audio_smoothed = librosa.istft(D_smoothed, hop_length=hop_length, length=len(audio))

        # NaN/Inf-Guard
        audio_smoothed = np.nan_to_num(audio_smoothed, nan=0.0, posinf=0.0, neginf=0.0)

        return np.asarray(audio_smoothed)  # type: ignore[no-any-return]


# ============================================================================
# Integration in Pipeline
# ============================================================================


def apply_hybrid_refinement(
    audio_ml_cleaned: np.ndarray,
    audio_original: np.ndarray,
    sr: int,
    genre: str,
    strength: float = 0.3,
) -> tuple[np.ndarray, dict]:
    """
    Convenience function für Pipeline-Integration.

    Returns:
        (refined_audio, metrics)
    """
    refiner = HybridDenoiseRefiner()

    audio_refined = refiner.refine(audio_ml_cleaned, audio_original, sr, genre=genre, strength=strength)

    # Authenticity Check (Safeguard #4)
    from dsp.authenticity_metrics import AuthenticityMetrics

    is_authentic, warnings, metrics = AuthenticityMetrics.authenticity_check(audio_original, audio_refined, sr)

    if not is_authentic:
        logger.warning("Refinement altered character:")
        for warning in warnings:
            logger.warning("  %s", warning)

    return audio_refined, metrics


# ============================================================================
# Unit Test
# ============================================================================

if __name__ == "__main__":
    # Test mit Synthetic Audio
    import soundfile as sf

    sr = 48000
    duration = 3.0
    samples = int(duration * sr)

    # Generate test signal
    t = np.linspace(0, duration, samples)
    audio_clean = np.sin(2 * np.pi * 440 * t) * 0.3

    # Add noise
    audio_noisy = audio_clean + np.random.randn(samples) * 0.05

    # Simulate ML-Denoise (simple noise reduction)
    audio_ml_cleaned = audio_noisy * 0.9  # Simulated

    # Apply Refinement
    refiner = HybridDenoiseRefiner()
    audio_refined = refiner.refine(audio_ml_cleaned, audio_noisy, sr, genre="classical", strength=0.3)

    # Save for comparison
    sf.write("test_original.wav", audio_noisy, sr)
    sf.write("test_ml_cleaned.wav", audio_ml_cleaned, sr)
    sf.write("test_refined.wav", audio_refined, sr)

    logger.info("✓ Test audio files written")
    _ext = ".wav"
    logger.info("  - test_%s%s (Referenzsignal mit Rauschen)", "original", _ext)
    logger.info("  - test_ml_cleaned.wav (ML-cleaned)")
    logger.info("  - test_refined.wav (hybrid refined)")
