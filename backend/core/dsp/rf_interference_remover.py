"""
RF (Radio Frequency) Interference Removal Module

Implements GAP #6 from TIEFENANALYSE_MUSIKRESTAURATION_PROBLEME.md

Entfernt hochfrequente Radio-Interferenzen aus Audio-Aufnahmen, die häufig durch:
- AM/FM Radio-Signale
- Mobilfunk-Interferenzen
- Computer-/Netzgerät-Störungen
- Drahtlose Mikrofon-Störungen verursacht werden

Technik:
- Harmonische Rauschprofile-Analyse
- Adaptive Notch-Filterung
- Spectral Gating für schmalbandige Störungen
- Frequency-Domain Processing mit STFT

Author: AURIK Development Team
Date: 9. Februar 2026
Version: 1.0.0
"""

import logging

import numpy as np
import scipy.signal as signal

logger = logging.getLogger("aurik.dsp.rf_interference_remover")
logger.setLevel(logging.INFO)


class RFInterferenceRemover:
    """
    Entfernt radio frequency (RF) interference from audio signals.

    RF interference typically manifests as:
    - Continuous tones (carrier frequencies)
    - Amplitude-modulated noise
    - Harmonically-related peaks in the spectrum

    This module detects and removes such interference using adaptive
    spectral filtering and notch filtering techniques.
    """

    def __init__(
        self,
        detection_threshold_db: float = -60.0,
        min_interference_freq: float = 5000.0,
        max_interference_freq: float = 20000.0,
        notch_q_factor: float = 30.0,
        harmonic_tolerance: float = 0.02,
        min_duration_sec: float = 0.1,
    ):
        """
        Initialisiert RF Interference Remover.

        Args:
            detection_threshold_db: Threshold for detecting interference peaks (dB)
            min_interference_freq: Minimum frequency for RF detection (Hz)
            max_interference_freq: Maximum frequency for RF detection (Hz)
            notch_q_factor: Quality factor for notch filters (higher = narrower)
            harmonic_tolerance: Tolerance for harmonic detection (fraction)
            min_duration_sec: Minimum duration for consistent interference (seconds)
        """
        self.detection_threshold_db = detection_threshold_db
        self.min_interference_freq = min_interference_freq
        self.max_interference_freq = max_interference_freq
        self.notch_q_factor = notch_q_factor
        self.harmonic_tolerance = harmonic_tolerance
        self.min_duration_sec = min_duration_sec

    def detect_interference_frequencies(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> list[float]:
        """
        Erkennt RF interference frequencies in the audio signal.

        Uses spectral analysis to find narrow-band peaks that are likely
        RFinterference rather than musical content.

        Args:
            audio: Audio signal (mono)
            sr: Sample rate

        Returns:
            List of detected interference frequencies (Hz)
        """
        assert sr == 48000, f"Sample rate must be 48000 Hz, got {sr}"
        # STFT parameters
        nperseg = min(4096, len(audio))
        noverlap = nperseg // 2

        # Compute spectrogram
        f, _t, Sxx = signal.spectrogram(
            audio, fs=sr, nperseg=nperseg, noverlap=noverlap, window="hann", scaling="density"
        )

        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Frequency range mask
        freq_mask = (f >= self.min_interference_freq) & (f <= self.max_interference_freq)

        # Mean power across time (averaged spectrum)
        mean_spectrum_db = np.mean(Sxx_db[freq_mask], axis=1)
        f_range = f[freq_mask]

        # Detect peaks that are significantly above the detection threshold
        # and persistent across time (indicating continuous interference)

        # Time persistence check: interference should be present in >90% of frames
        time_presence = np.mean(Sxx_db[freq_mask] > self.detection_threshold_db, axis=1)
        persistent_mask = time_presence > 0.9

        # Find local maxima in the spectrum
        peaks, _properties = signal.find_peaks(
            mean_spectrum_db,
            prominence=10.0,  # At least 10 dB above surrounding
            distance=max(1, int(50 / (sr / nperseg))),  # At least 50 Hz apart (minimum 1)
        )

        # Filter peaks by persistence
        persistent_peaks = peaks[persistent_mask[peaks]]

        # Get interference frequencies
        interference_freqs = f_range[persistent_peaks].tolist()

        # Check for harmonic relationships (fundamental + harmonics = interference)
        interference_freqs = self._filter_harmonics(interference_freqs)

        return interference_freqs

    def _filter_harmonics(self, frequencies: list[float]) -> list[float]:
        """
        Identify fundamental frequencies and their harmonics.

        If multiple frequencies are harmonically related, only keep
        the fundamental and explicit harmonics.

        Args:
            frequencies: List of detected frequencies

        Returns:
            Filtered list with fundamental and harmonic frequencies
        """
        if len(frequencies) <= 1:
            return frequencies

        frequencies = sorted(frequencies)
        fundamentals = []

        for i, f1 in enumerate(frequencies):
            is_harmonic = False

            # Check if f1 is a harmonic of any lower frequency
            for f0 in frequencies[:i]:
                ratio = f1 / f0
                # Check if ratio is close to an integer (harmonic relationship)
                closest_integer = round(ratio)
                if abs(ratio - closest_integer) / closest_integer < self.harmonic_tolerance:
                    is_harmonic = True
                    break

            if not is_harmonic:
                fundamentals.append(f1)

                # Add explicit harmonics of this fundamental
                for n in range(2, 6):  # Up to 5th harmonic
                    harmonic_freq = f1 * n
                    if harmonic_freq <= self.max_interference_freq:
                        # Check if this harmonic is in the original list
                        for f_detected in frequencies:
                            if abs(f_detected - harmonic_freq) / harmonic_freq < self.harmonic_tolerance:
                                if f_detected not in fundamentals:
                                    fundamentals.append(f_detected)
                                break

        return sorted(fundamentals)

    def remove_interference(
        self,
        audio: np.ndarray,
        sr: int,
        interference_freqs: list[float] | None = None,
    ) -> np.ndarray:
        """
        Entfernt RF interference from audio signal.

        Args:
            audio: Audio signal (mono or stereo)
            sr: Sample rate
            interference_freqs: List of interference frequencies to remove.
                               If None, auto-detect.

        Returns:
            Processed audio with interference removed
        """
        # Handle stereo and mono with shape (samples, 1) or (1, samples)
        if audio.ndim == 2:
            # Mono mit shape (samples, 1)
            if audio.shape[1] == 1:
                return self.remove_interference(audio[:, 0], sr, interference_freqs)
            # Mono mit shape (1, samples)
            if audio.shape[0] == 1:
                return self.remove_interference(audio[0], sr, interference_freqs)
            # Stereo oder mehr
            if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 32:
                # (channels, samples) format
                left = self.remove_interference(audio[0], sr, interference_freqs)
                right = self.remove_interference(audio[1], sr, interference_freqs)
                return np.asarray(np.vstack([left, right]))  # type: ignore[no-any-return]
            else:
                # (samples, channels) format - AURIK standard
                left = self.remove_interference(audio[:, 0], sr, interference_freqs)
                right = self.remove_interference(audio[:, 1], sr, interference_freqs)
                return np.asarray(np.column_stack([left, right]))  # type: ignore[no-any-return]

        # Mono processing
        if interference_freqs is None:
            interference_freqs = self.detect_interference_frequencies(audio, sr)

        if len(interference_freqs) == 0:
            return audio.copy()

        # Apply cascaded notch filters for each interference frequency
        processed: np.ndarray = audio.copy()

        for freq in interference_freqs:
            # Design notch filter
            # Ensure frequency is within Nyquist limits
            nyquist = sr / 2
            if freq >= nyquist * 0.95:
                continue

            # IIR notch filter
            b, a = signal.iirnotch(freq, self.notch_q_factor, sr)

            # Apply filter (forward-backward to avoid phase shift)
            processed = signal.filtfilt(b, a, processed)  # Merge-Rest 2026-10-03, typ-seitig geklärt

            # NaN/Inf-Guard after filtering
            processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)

        # Final clip
        processed = np.clip(processed, -1.0, 1.0)

        return np.asarray(processed)  # type: ignore[no-any-return]

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        auto_detect: bool = True,
        audit_log: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """
        Full processing pipeline: detect and remove RF interference.
        Quality Gate, Audit-Logging, robuste Fehlerbehandlung
        Args:
            audio: Input audio (mono or stereo)
            sr: Sample rate
            auto_detect: Whether to auto-detect interference frequencies
            audit_log: Audit-Logging aktivieren
        Returns:
            Tuple of (processed_audio, metrics)
        """
        # Quality Gate: Input-Checks
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            logger.error("Ungültiges Audio-Array (leer oder falscher Typ)")
            raise ValueError("Ungültiges Audio-Array (leer oder falscher Typ)")
        if np.isnan(audio).any():
            logger.error("Audio enthält NaN-Werte")
            raise ValueError("Audio enthält NaN-Werte")
        if np.max(np.abs(audio)) > 1e6:
            logger.warning("Audio möglicherweise nicht normiert (max > 1e6)")

        try:
            # Detect interference
            if auto_detect:
                # Use first channel for detection if stereo
                detection_audio = audio
                if audio.ndim == 2:
                    if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 32:
                        detection_audio = audio[0]  # (channels, samples)
                    else:
                        detection_audio = audio[:, 0]  # (samples, channels)

                interference_freqs = self.detect_interference_frequencies(detection_audio, sr)
            else:
                interference_freqs = []

            # Remove interference
            processed = self.remove_interference(audio, sr, interference_freqs)

            # Metrics
            metrics = {
                "interference_freqs": interference_freqs,
                "num_interference": len(interference_freqs),
                "frequency_range": (
                    min(interference_freqs) if interference_freqs else 0,
                    max(interference_freqs) if interference_freqs else 0,
                ),
            }
        except Exception as e:
            logger.error("Fehler bei RF-Interferenz-Entfernung: %s", e)
            processed = audio.copy()
            metrics = {"interference_freqs": [], "num_interference": 0, "frequency_range": (0, 0)}

        if audit_log:
            logger.info(
                f"RFInterferenceRemover: num_interference={metrics['num_interference']}, range={metrics['frequency_range']}"
            )
        return processed, metrics


if __name__ == "__main__":
    # Example usage
    pass

    # Create test signal with RF interference
    sr = 48000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))

    # Clean audio: sine wave at 440 Hz
    clean = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Add RF interference at 12 kHz and 15.5 kHz
    interference1 = 0.05 * np.sin(2 * np.pi * 12000 * t)
    interference2 = 0.03 * np.sin(2 * np.pi * 15500 * t)

    # Add modulated RF interference (AM radio)
    carrier_freq = 18000
    modulation_freq = 50
    am_interference = 0.02 * np.sin(2 * np.pi * carrier_freq * t) * (1 + 0.5 * np.sin(2 * np.pi * modulation_freq * t))

    # Contaminated signal
    contaminated = clean + interference1 + interference2 + am_interference

    # Process
    remover = RFInterferenceRemover()
    processed, metrics = remover.process(contaminated, sr)

    logger.info("RF Interference Removal Results:")
    logger.info("  %s Störfrequenzen erkannt:", metrics["num_interference"])
    for freq in metrics["interference_freqs"]:
        logger.info("    - %.1f Hz", freq)
    logger.info("  Frequency range: %.1f - %.1f Hz", metrics["frequency_range"][0], metrics["frequency_range"][1])

    # Compare RMS
    rms_original = np.sqrt(np.mean(contaminated**2))
    rms_processed = np.sqrt(np.mean(processed**2))
    reduction_db = 20 * np.log10(rms_processed / rms_original)
    logger.info("  Overall level reduction: %.2f dB", reduction_db)
