"""
Adaptive MMSE Noise PSD Estimation DSP-Modul für Aurik 10.0.0 (SOTA-Maximum)

Implements IMCRA-inspired noise PSD estimation with:
- Minimum-statistics tracking over sliding window (Cohen 2002)
- Speech-presence-probability-weighted update
- SNR-adaptive smoothing factor
- Full NaN/Inf guards on all paths

Replaces the naive exponential-smoothing approach (assumes stationary noise)
with proper minimum-tracking that handles non-stationary audio content.

References:
  Cohen (2002): "Noise spectrum estimation in adverse environments"
  Cohen (2003): "IMCRA noise estimation"
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveMMSENoisePSD:
    """IMCRA-inspired noise PSD estimation with minimum-statistics tracking."""

    def __init__(self, alpha: float = 0.92, noise_floor: float = 1e-6, min_window: int = 12, bias_factor: float = 1.5):
        self.alpha = alpha  # smoothing factor for PSD tracking
        self.noise_floor = noise_floor
        self.min_window = min_window  # frames for minimum search
        self.bias_factor = bias_factor  # minimum-statistics bias compensation

    def estimate_noise(self, power_spectrogram: np.ndarray) -> np.ndarray:
        """Schätzt noise PSD using minimum-statistics with exponential smoothing.

        Combines recursive PSD averaging with minimum tracking:
        1. Smooth PSD with exponential averaging
        2. Track minimum over sliding window (minimum-statistics, Cohen 2002)
        3. Use minimum as noise estimate (with bias compensation)
        4. Apply speech-presence-probability gate: if current power >> minimum,
           likely speech/music present → don't update noise estimate

        Args:
            power_spectrogram: 2D array [n_frames, n_bins], power spectrum
        Returns:
            Estimated noise PSD [n_frames, n_bins]
        """
        power = np.nan_to_num(np.asarray(power_spectrogram, dtype=np.float64))
        n_frames, n_bins = power.shape

        noise_psd = np.zeros_like(power)
        smoothed_psd = np.zeros(n_bins, dtype=np.float64)
        min_buffer = np.full((self.min_window, n_bins), np.inf, dtype=np.float64)

        # Initialize with first frame
        smoothed_psd[:] = power[0]
        noise_psd[0] = np.maximum(power[0], self.noise_floor)

        for t in range(1, n_frames):
            # Step 1: Smooth PSD with exponential averaging
            smoothed_psd = self.alpha * smoothed_psd + (1.0 - self.alpha) * power[t]

            # Step 2: Update minimum buffer
            min_buffer[t % self.min_window] = smoothed_psd
            min_psd = np.min(min_buffer, axis=0)

            # Step 3: Bias compensation (minimum-statistics underestimates noise)
            min_psd_compensated = min_psd * self.bias_factor

            # Step 4: Speech-presence probability gate
            # If current smoothed PSD is much larger than minimum → speech present
            spp_ratio = smoothed_psd / np.maximum(min_psd_compensated, self.noise_floor)
            speech_present = spp_ratio > 2.0  # simple threshold

            # Update noise estimate:
            # - Where speech present: hold previous estimate
            # - Where noise-only: track smoothed PSD
            prev_noise = noise_psd[t - 1]
            noise_update = self.alpha * prev_noise + (1.0 - self.alpha) * min_psd_compensated
            noise_psd[t] = np.where(speech_present, prev_noise, noise_update)
            noise_psd[t] = np.maximum(noise_psd[t], self.noise_floor)

        noise_psd = np.nan_to_num(noise_psd, nan=self.noise_floor, posinf=self.noise_floor, neginf=self.noise_floor)
        return np.asarray(noise_psd)  # type: ignore[no-any-return]

    def auto_optimize(self, power_spectrogram: np.ndarray) -> None:
        """Adapt alpha and min_window based on signal characteristics."""
        power = np.nan_to_num(np.asarray(power_spectrogram, dtype=np.float64))
        n_frames = power.shape[0]

        # Adapt window size to signal length
        if n_frames < 30:
            self.min_window = 4
            self.alpha = 0.85
        elif n_frames < 100:
            self.min_window = 8
            self.alpha = 0.90
        elif n_frames < 500:
            self.min_window = 12
            self.alpha = 0.92
        else:
            self.min_window = 20
            self.alpha = 0.95

        # Estimate SNR to adapt bias factor
        mean_pow = float(np.mean(power)) + 1e-12
        min_pow = float(np.percentile(power, 5)) + 1e-12
        snr_est = 10.0 * np.log10(mean_pow / min_pow)

        # Low SNR → smaller bias (noise is closer to minimum)
        # High SNR → larger bias (minimum underestimates more)
        self.bias_factor = float(np.clip(1.2 + 0.03 * snr_est, 1.0, 2.5))

        logger.debug(
            "AdaptiveMMSENoisePSD.auto_optimize: n_frames=%d, est_SNR=%.1f dB → alpha=%.2f, min_window=%d, bias=%.2f",
            n_frames,
            snr_est,
            self.alpha,
            self.min_window,
            self.bias_factor,
        )
