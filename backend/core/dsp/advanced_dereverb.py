"""
Advanced De-Reverb (GAP #54) - AURIK v8

Professional reverb removal for studio recordings using advanced algorithmic techniques.

Unterschied zu RoomDeverberator (Live-Recording):
- RoomDeverberator: RT60-basierte Reduktion für Live-Konzerte
- Advanced De-Reverb: Komplette Reverb-Entfernung für Studio-Aufnahmen

Techniques:
1. Wiener Filtering - Statistical reverb estimation
2. Late Reflection Cancellation - Adaptive filtering
3. Spectral-Temporal Analysis - Direct vs reflected sound separation
4. Multi-band Processing - Frequency-selective control

Status: 0% → 70% (Algorithmic only, ML enhancement later)

Author: AURIK Team
Version: 1.0.0
"""

import logging
import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, istft, sosfilt, stft

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class WienerDereverb:
    """
    Wiener filtering-based dereverberation.

    Statistical approach to separate direct sound from reverb
    based on spectral characteristics.
    """

    def __init__(self, reverb_time_estimate: float = 0.5, strength: float = 0.7):
        """
        Parameters:
        -----------
        reverb_time_estimate : float
            Estimated reverb decay time in seconds (0.3-2.0)
        strength : float
            Processing strength (0.0 = none, 1.0 = maximum)
        """
        self.reverb_time_estimate = reverb_time_estimate
        self.strength = np.clip(strength, 0.0, 1.0)

    def _estimate_reverb_spectrum(self, stft_mag: np.ndarray, _sr: int) -> np.ndarray:
        """
        Schätzt reverb spectral template from late frames.

        Assumption: Late frames are reverb-dominated.
        """
        # Use last 30% of frames as reverb template
        n_frames = stft_mag.shape[1]
        reverb_start = int(n_frames * 0.7)

        # Average reverb spectrum
        reverb_template = np.mean(stft_mag[:, reverb_start:], axis=1, keepdims=True)

        return reverb_template  # type: ignore[no-any-return]

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Wendet an: Wiener filtering for reverb removal.

        Parameters:
        -----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate

        Returns:
        --------
        processed : np.ndarray
            De-reverbed audio
        metrics : dict
            Processing metrics
        """
        if sr <= 0:
            raise ValueError(f"Sample rate must be > 0 Hz, got {sr}")
        input_dtype = audio.dtype
        audio_mono = audio if audio.ndim == 1 else np.mean(audio, axis=1) if audio.ndim == 2 else audio

        # STFT
        nperseg = 2048
        noverlap = nperseg // 2
        _f, _t, Zxx = stft(audio_mono, sr, nperseg=nperseg, noverlap=noverlap)
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Estimate reverb spectrum
        reverb_spectrum = self._estimate_reverb_spectrum(magnitude, sr)

        # Wiener filter gain
        noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
        direct_estimate = np.maximum(magnitude - reverb_spectrum * self.strength, noise_floor)

        # Wiener gain: direct / (direct + reverb)
        wiener_gain = direct_estimate / (magnitude + 1e-10)
        wiener_gain = np.clip(wiener_gain, 0.3, 1.0)  # Limit gain range

        # Apply gain
        magnitude_filtered = magnitude * wiener_gain

        # Reconstruct
        Zxx_filtered = magnitude_filtered * np.exp(1j * phase)
        _, audio_filtered = istft(Zxx_filtered, sr, nperseg=nperseg, noverlap=noverlap)

        # Match length
        if len(audio_filtered) < len(audio_mono):
            audio_filtered = np.pad(audio_filtered, (0, len(audio_mono) - len(audio_filtered)))
        else:
            audio_filtered = audio_filtered[: len(audio_mono)]

        # Stereo reconstruction if needed
        if audio.ndim == 2:
            # Compute scaling factor
            scale = audio_filtered / (audio_mono + 1e-10)
            audio_filtered = np.column_stack([audio[:, 0] * scale, audio[:, 1] * scale])

        # Metrics
        reverb_reduction_db = 20 * np.log10(np.mean(wiener_gain) + 1e-10)

        # NaN/Inf-Guard + Clipping
        audio_filtered = np.nan_to_num(audio_filtered, nan=0.0, posinf=0.0, neginf=0.0)
        if audio_filtered.ndim == 2:
            audio_filtered = np.clip(audio_filtered, -1.0, 1.0)
        else:
            audio_filtered = np.clip(audio_filtered, -1.0, 1.0)

        metrics = {
            "reverb_reduction_db": reverb_reduction_db,
            "wiener_gain_mean": np.mean(wiener_gain),
            "wiener_gain_min": np.min(wiener_gain),
        }

        return audio_filtered.astype(input_dtype), metrics


class LateReflectionCanceller:
    """
    Adaptive filtering to cancel late reflections.

    Uses autocorrelation to detect repetitive patterns (reflections)
    and suppress them adaptively.
    """

    def __init__(self, threshold_lag_ms: float = 50.0, suppression_db: float = 12.0):
        """
        Parameters:
        -----------
        threshold_lag_ms : float
            Minimum lag time to consider as late reflection (50-200ms)
        suppression_db : float
            Suppression amount for detected reflections (6-18dB)
        """
        self.threshold_lag_ms = threshold_lag_ms
        self.suppression_db = suppression_db

    def _detect_reflections(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Erkennt reflection patterns using autocorrelation.

        Returns:
        --------
        reflection_mask : np.ndarray
            Time-frequency mask indicating reflection regions
        """
        # STFT
        nperseg = 2048
        noverlap = nperseg // 2
        _f, _t, Zxx = stft(audio, sr, nperseg=nperseg, noverlap=noverlap)
        magnitude = np.abs(Zxx)

        # Energy envelope per frequency band
        envelope = magnitude**2

        # Detect decay patterns (characteristic of reverb)
        reflection_mask = np.zeros_like(magnitude)

        for i in range(magnitude.shape[0]):
            band_envelope = envelope[i, :]

            # Smooth envelope
            smoothed = uniform_filter1d(band_envelope, size=5, mode="nearest")

            # Detect exponential decay (reflection characteristic)
            # If energy is decreasing: likely reflection
            decay = np.diff(smoothed)
            decay_mask = np.concatenate([decay < 0, np.array([False])])

            reflection_mask[i, :] = decay_mask

        return reflection_mask  # type: ignore[no-any-return]

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Cancel late reflections.

        Parameters:
        -----------
        audio : np.ndarray
            Input audio
        sr : int
            Sample rate

        Returns:
        --------
        processed : np.ndarray
            Audio with suppressed reflections
        metrics : dict
            Processing metrics
        """
        input_dtype = audio.dtype
        audio_mono = audio if audio.ndim == 1 else np.mean(audio, axis=1) if audio.ndim == 2 else audio

        # STFT
        nperseg = 2048
        noverlap = nperseg // 2
        _f, _t, Zxx = stft(audio_mono, sr, nperseg=nperseg, noverlap=noverlap)
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Detect reflection regions
        reflection_mask = self._detect_reflections(audio_mono, sr)

        # Compute suppression gain
        suppression_linear = 10 ** (-self.suppression_db / 20.0)
        gain = np.ones_like(magnitude)
        gain[reflection_mask > 0.5] = suppression_linear

        # Apply gain
        magnitude_suppressed = magnitude * gain

        # Reconstruct
        Zxx_suppressed = magnitude_suppressed * np.exp(1j * phase)
        _, audio_suppressed = istft(Zxx_suppressed, sr, nperseg=nperseg, noverlap=noverlap)

        # Match length
        if len(audio_suppressed) < len(audio_mono):
            audio_suppressed = np.pad(audio_suppressed, (0, len(audio_mono) - len(audio_suppressed)))
        else:
            audio_suppressed = audio_suppressed[: len(audio_mono)]

        # Stereo reconstruction if needed
        if audio.ndim == 2:
            scale = audio_suppressed / (audio_mono + 1e-10)
            audio_suppressed = np.column_stack([audio[:, 0] * scale, audio[:, 1] * scale])

        # Metrics
        reflection_percentage = (np.sum(reflection_mask > 0.5) / reflection_mask.size) * 100

        metrics = {
            "reflection_percentage": reflection_percentage,
            "suppression_db": self.suppression_db,
            "suppressed_regions": np.sum(reflection_mask > 0.5),
        }

        return audio_suppressed.astype(input_dtype), metrics


class SpectralTemporalAnalyzer:
    """
    Analysiert spektral-zeitliche Eigenschaften zur Signaltrennung.
    direct sound from reverb.

    Direct sound: High energy, short duration, localized
    Reverb: Lower energy, long duration, diffuse
    """

    def __init__(self, direct_threshold: float = 0.7):
        """
        Parameters:
        -----------
        direct_threshold : float
            Threshold for direct sound detection (0.5-0.9)
        """
        self.direct_threshold = direct_threshold

    def analyze(self, audio: np.ndarray, sr: int) -> dict:
        """
        Analysiert Direkt- vs. Hall-Inhalt.

        Returns:
        --------
        metrics : dict
            Analysis results
        """
        audio_mono = audio if audio.ndim == 1 else np.mean(audio, axis=1) if audio.ndim == 2 else audio

        # STFT
        nperseg = 2048
        noverlap = nperseg // 2
        _f, _t, Zxx = stft(audio_mono, sr, nperseg=nperseg, noverlap=noverlap)
        magnitude = np.abs(Zxx)

        # Temporal characteristics
        frame_energy = np.sum(magnitude**2, axis=0)
        np.max(frame_energy)
        mean_energy = np.mean(frame_energy)

        # Transient density (high peaks = more direct sound)
        transients = frame_energy > (mean_energy * 3.0)
        transient_density = np.sum(transients) / len(transients)

        # Spectral flatness (low = tonal, high = reverb/noise)
        spectral_flatness = np.mean(
            [
                np.exp(np.mean(np.log(magnitude[:, i] + 1e-10))) / (np.mean(magnitude[:, i]) + 1e-10)
                for i in range(magnitude.shape[1])
            ]
        )

        # Reverb estimate (high flatness + low transient density = reverb)
        reverb_score = (1.0 - transient_density) * spectral_flatness

        # RT60 estimate (simplified)
        envelope = np.abs(np.asarray(hilbert(audio_mono), dtype=np.complex128))
        envelope_db = 20 * np.log10(envelope + 1e-10)
        envelope_db -= np.max(envelope_db)

        # Find -60dB point
        below_60 = np.where(envelope_db < -60)[0]
        rt60_estimate = below_60[0] / sr if len(below_60) > 0 else len(audio_mono) / sr

        return {
            "reverb_score": float(reverb_score),
            "transient_density": float(transient_density),
            "spectral_flatness": float(spectral_flatness),
            "rt60_estimate": float(rt60_estimate),
            "has_significant_reverb": reverb_score > 0.3,
        }


class MultibandDereverb:
    """
    Frequency-selective de-reverb processing.

    Different frequency bands have different reverb characteristics:
    - Low frequencies: Longer decay, more modal
    - Mid frequencies: Speech/music energy, moderate reverb
    - High frequencies: Shorter decay, air absorption
    """

    def __init__(self, low_strength: float = 0.5, mid_strength: float = 0.7, high_strength: float = 0.6):
        """
        Parameters:
        -----------
        low_strength : float
            De-reverb strength for low frequencies (0.0-1.0)
        mid_strength : float
            De-reverb strength for mid frequencies (0.0-1.0)
        high_strength : float
            De-reverb strength for high frequencies (0.0-1.0)
        """
        self.low_strength = np.clip(low_strength, 0.0, 1.0)
        self.mid_strength = np.clip(mid_strength, 0.0, 1.0)
        self.high_strength = np.clip(high_strength, 0.0, 1.0)

        # Band crossover frequencies
        self.low_cutoff = 300
        self.high_cutoff = 3000

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Wendet an: multi-band de-reverb.

        Parameters:
        -----------
        audio : np.ndarray
            Input audio
        sr : int
            Sample rate

        Returns:
        --------
        processed : np.ndarray
            De-reverbed audio
        metrics : dict
            Processing metrics
        """
        input_dtype = audio.dtype
        audio_mono = audio if audio.ndim == 1 else np.mean(audio, axis=1) if audio.ndim == 2 else audio

        # Split into bands
        nyq = sr / 2

        # Low band (< 300 Hz)
        sos_low = butter(4, self.low_cutoff / nyq, btype="low", output="sos")
        audio_low = sosfilt(sos_low, audio_mono)

        # Mid band (300 - 3000 Hz)
        sos_mid = butter(4, [self.low_cutoff / nyq, self.high_cutoff / nyq], btype="band", output="sos")
        audio_mid = sosfilt(sos_mid, audio_mono)

        # High band (> 3000 Hz)
        sos_high = butter(4, self.high_cutoff / nyq, btype="high", output="sos")
        audio_high = sosfilt(sos_high, audio_mono)

        # Process each band
        wiener_low = WienerDereverb(reverb_time_estimate=0.8, strength=self.low_strength)
        wiener_mid = WienerDereverb(reverb_time_estimate=0.5, strength=self.mid_strength)
        wiener_high = WienerDereverb(reverb_time_estimate=0.3, strength=self.high_strength)

        audio_low_proc, _ = wiener_low.process(audio_low, sr)
        audio_mid_proc, _ = wiener_mid.process(audio_mid, sr)
        audio_high_proc, _ = wiener_high.process(audio_high, sr)

        # Recombine
        audio_processed = audio_low_proc + audio_mid_proc + audio_high_proc

        # Stereo reconstruction if needed
        if audio.ndim == 2:
            scale = audio_processed / (audio_mono + 1e-10)
            audio_processed = np.column_stack([audio[:, 0] * scale, audio[:, 1] * scale])

        metrics = {
            "low_strength": self.low_strength,
            "mid_strength": self.mid_strength,
            "high_strength": self.high_strength,
        }

        return audio_processed.astype(input_dtype), metrics


class AdvancedDereverb:
    """
    Complete advanced de-reverb system (GAP #54).

    Combines multiple techniques for professional reverb removal:
    1. Wiener filtering
    2. Late reflection cancellation
    3. Spectral-temporal analysis
    4. Multi-band processing

    Mode options:
    - 'mild': Light reverb reduction (preserve ambience)
    - 'balanced': Moderate reverb reduction
    - 'aggressive': Maximum reverb removal
    """

    def __init__(self, mode: str = "balanced"):
        """
        Parameters:
        -----------
        mode : str
            Processing mode ('mild', 'balanced', 'aggressive')
        """
        self.mode = mode

        # Configure based on mode
        if mode == "mild":
            self.wiener = WienerDereverb(reverb_time_estimate=0.5, strength=0.4)
            self.late_reflection = LateReflectionCanceller(threshold_lag_ms=80.0, suppression_db=6.0)
            self.multiband = MultibandDereverb(low_strength=0.3, mid_strength=0.5, high_strength=0.4)
        elif mode == "aggressive":
            self.wiener = WienerDereverb(reverb_time_estimate=0.5, strength=0.9)
            self.late_reflection = LateReflectionCanceller(threshold_lag_ms=30.0, suppression_db=18.0)
            self.multiband = MultibandDereverb(low_strength=0.7, mid_strength=0.9, high_strength=0.8)
        else:  # balanced
            self.wiener = WienerDereverb(reverb_time_estimate=0.5, strength=0.7)
            self.late_reflection = LateReflectionCanceller(threshold_lag_ms=50.0, suppression_db=12.0)
            self.multiband = MultibandDereverb(low_strength=0.5, mid_strength=0.7, high_strength=0.6)

        self.analyzer = SpectralTemporalAnalyzer()

    def analyze(self, audio: np.ndarray, sr: int) -> dict:
        """
        Analysiert die Hall-Eigenschaften des Audios.

        Returns:
        --------
        metrics : dict
            Reverb analysis
        """
        return self.analyzer.analyze(audio, sr)

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Wendet vollständige Hallentfernung an.

        Parameters:
        -----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate

        Returns:
        --------
        processed : np.ndarray
            De-reverbed audio
        metrics : dict
            Complete processing metrics
        """
        # Analyze first
        analysis = self.analyzer.analyze(audio, sr)

        # If minimal reverb detected, skip processing
        if not analysis["has_significant_reverb"]:
            return audio, {"processed": False, "reason": "Minimal reverb detected", "analysis": analysis}

        # Stage 1: Multi-band de-reverb (main processing)
        audio_stage1, metrics_multiband = self.multiband.process(audio, sr)

        # Stage 2: Wiener filtering (refinement)
        audio_stage2, metrics_wiener = self.wiener.process(audio_stage1, sr)

        # Stage 3: Late reflection cancellation (final polish)
        audio_final, metrics_late = self.late_reflection.process(audio_stage2, sr)

        # Guard: no clipping artefacts from cumulative gain (§0 Primum non nocere)
        audio_final = np.clip(audio_final, -1.0, 1.0)

        # Complete metrics
        metrics = {
            "processed": True,
            "mode": self.mode,
            "analysis": analysis,
            "multiband": metrics_multiband,
            "wiener": metrics_wiener,
            "late_reflection": metrics_late,
        }

        return audio_final, metrics


# CLI interface
if __name__ == "__main__":
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(description="Advanced De-Reverb (GAP #54)")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--output", help="Output file (optional)")
    parser.add_argument(
        "--mode", choices=["mild", "balanced", "aggressive"], default="balanced", help="Processing mode"
    )
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't process")

    args = parser.parse_args()

    # Load audio
    from backend.file_import import load_audio_file

    _res = load_audio_file(args.input)
    assert _res is not None, "load_audio_file returned None"
    _audio, _sr = _res["audio"], int(_res["sr"])
    logger.info("Eingabe: %s (%s Hz, %s)", args.input, _sr, "stereo" if _audio.ndim == 2 else "mono")

    # Create de-reverb system
    dereverb = AdvancedDereverb(mode=args.mode)

    # Analyze
    logger.info("\nAuswertung läuft...")
    _analysis = dereverb.analyze(_audio, _sr)
    logger.info("  Hallwert: %.3f", _analysis["reverb_score"])
    logger.info("  Transient density: %s", format(_analysis["transient_density"], ".2%"))
    logger.info("  RT60 estimate: %.2fs", _analysis["rt60_estimate"])
    logger.info("  Significant reverb: %s", _analysis["has_significant_reverb"])

    if not args.analyze_only:
        # Process
        logger.info("\nVerarbeitung (Betriebsart: %s)...", args.mode)
        _audio_dereverbed, _metrics = dereverb.process(_audio, _sr)

        if _metrics["processed"]:
            logger.info("\n✓ Enthallung abgeschlossen:")
            logger.info("  Wiener reduction: %.1f dB", _metrics["wiener"]["reverb_reduction_db"])
            logger.info("  Reflection suppression: %.1f dB", _metrics["late_reflection"]["suppression_db"])
            logger.info("  Reflection regions: %.1f%%", _metrics["late_reflection"]["reflection_percentage"])
        else:
            logger.info("\n○ %s", _metrics["reason"])

        # Save output
        output_path = args.output or args.input.replace(".wav", "_dereverb.wav")
        sf.write(output_path, _audio_dereverbed, _sr)
        logger.info("\n✓ Gespeichert: %s", output_path)
