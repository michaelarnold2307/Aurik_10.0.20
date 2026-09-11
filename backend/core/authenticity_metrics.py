"""
Authenticity Metrics für AURIK

Quantitative Messung der Authentizitäts-Preservation:
- Breath Detection & Retention Rate
- Plosive Detection & Handling (Preserve in Musik, Remove in Speech)
- Transient Detection & Preservation

Critical für AURIK's USP: Musik-Restauration ohne Artifacts zu entfernen,
die zur Performance gehören.

Author: AURIK Development Team
Version: 1.0
Date: 2026-02-08
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import signal

from backend.core.audio_layout import mono_mix, to_channels_first

logger = logging.getLogger(__name__)

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore


@dataclass
class BreathEvent:
    """Erkanntes Atemereignis."""

    start_sample: int
    end_sample: int
    duration_sec: float
    energy: float
    confidence: float

    def __repr__(self) -> str:
        return f"Breath({self.start_sample}-{self.end_sample}, {self.duration_sec:.3f}s, conf={self.confidence:.2f})"


@dataclass
class PlosiveEvent:
    """Erkanntes Plosiv-Ereignis (P, B, T, D, K, G Laute)."""

    start_sample: int
    peak_sample: int
    end_sample: int
    energy: float
    sharpness: float  # Attack characteristics
    confidence: float

    def __repr__(self) -> str:
        return f"Plosive({self.peak_sample}, sharp={self.sharpness:.2f}, conf={self.confidence:.2f})"


@dataclass
class TransientEvent:
    """Erkannter Transient (Schlagzeugschlag, perkussiver Klang)."""

    peak_sample: int
    attack_time_ms: float
    energy: float
    sharpness: float
    confidence: float

    def __repr__(self) -> str:
        return f"Transient({self.peak_sample}, attack={self.attack_time_ms:.1f}ms, conf={self.confidence:.2f})"


@dataclass
class SibilanceAnalysis:
    """Sibilance characteristic analysis."""

    total_sibilance_energy: float
    sibilance_density: float  # Sibilance energy / total energy
    peak_sibilance_frequency: float  # Hz
    sibilance_duration_sec: float  # Total duration of sibilant content
    natural_level: bool  # True if within natural range

    def __repr__(self) -> str:
        return f"Sibilance(density={self.sibilance_density:.1%}, peak_freq={self.peak_sibilance_frequency:.0f}Hz)"


@dataclass
class RoomToneAnalysis:
    """Room tone/ambience characteristic analysis."""

    ambient_noise_floor_db: float
    room_resonances: list[float]  # Detected room modes (Hz)
    reverb_tail_length_ms: float
    spatial_correlation: float  # Stereo width indicator (0-1)
    naturalness_score: float  # 0-1 (synthetic vs natural)

    def __repr__(self) -> str:
        return f"RoomTone(floor={self.ambient_noise_floor_db:.1f}dB, reverb={self.reverb_tail_length_ms:.0f}ms)"


class BreathDetector:
    """Erkennt breath events in audio."""

    def __init__(
        self,
        min_duration_sec: float = 0.05,
        max_duration_sec: float = 0.5,
        freq_range: tuple[int, int] = (200, 3000),
        energy_threshold_db: float = -40.0,
    ):
        """
        Initialisiert breath detector.

        Args:
            min_duration_sec: Minimum breath duration
            max_duration_sec: Maximum breath duration
            freq_range: Frequency range for breath detection (Hz)
            energy_threshold_db: Energy threshold in dB
        """
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.freq_range = freq_range
        self.energy_threshold_db = energy_threshold_db

    def detect(self, audio: np.ndarray, sr: int) -> list[BreathEvent]:
        """
        Erkennt breath events in audio.

        Breaths are characterized by:
        - Low energy broadband noise
        - Short duration (50-500ms)
        - Primarily in 200-3000 Hz range
        - No tonal content

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            List of detected breath events
        """
        # Bandpass filter for breath frequency range
        sos = signal.butter(4, self.freq_range, "bandpass", fs=sr, output="sos")
        filtered = signal.sosfilt(sos, audio)

        # Compute envelope (RMS in short windows)
        window_size = int(0.01 * sr)  # 10ms windows
        hop_size = window_size // 2
        envelope = np.array(
            [
                np.sqrt(np.mean(filtered[i : i + window_size] ** 2))
                for i in range(0, len(filtered) - window_size, hop_size)
            ]
        )

        # Convert to dB
        envelope_db = 20 * np.log10(envelope + 1e-10)

        # Threshold detection
        threshold = np.max(envelope_db) + self.energy_threshold_db
        above_threshold = envelope_db > threshold

        # Find continuous regions
        breaths = []
        in_breath = False
        start_idx = 0

        for i, is_breath in enumerate(above_threshold):
            if is_breath and not in_breath:
                # Breath starts
                start_idx = i
                in_breath = True
            elif not is_breath and in_breath:
                # Breath ends
                end_idx = i
                duration_frames = end_idx - start_idx
                duration_sec = duration_frames * hop_size / sr

                if self.min_duration_sec <= duration_sec <= self.max_duration_sec:
                    # Valid breath
                    start_sample = start_idx * hop_size
                    end_sample = end_idx * hop_size

                    # Compute energy
                    breath_segment = filtered[start_sample:end_sample]
                    energy = np.sqrt(np.mean(breath_segment**2))

                    # Confidence: based on spectral characteristics
                    # Breaths have high spectral flatness (noise-like).
                    # n_fft must not exceed the segment length — librosa warns and pads
                    # internally when n_fft > len(signal), which produces misleading results.
                    _n_fft_breath = min(2048, max(32, int(2 ** np.floor(np.log2(len(breath_segment))))))
                    if librosa is not None:
                        _stft = np.abs(librosa.stft(breath_segment, n_fft=_n_fft_breath))
                        _sf = float(np.mean(librosa.feature.spectral_flatness(S=_stft)))
                        confidence = float(np.clip(_sf, 0, 1))
                    else:
                        confidence = 0.5

                    breaths.append(
                        BreathEvent(
                            start_sample=start_sample,
                            end_sample=end_sample,
                            duration_sec=duration_sec,
                            energy=float(energy),
                            confidence=confidence,
                        )
                    )

                in_breath = False

        return breaths


class PlosiveDetector:
    """Erkennt plosive events (P, B, T, D, K, G sounds)."""

    def __init__(
        self, min_attack_time_ms: float = 1.0, max_attack_time_ms: float = 20.0, energy_threshold_db: float = -30.0
    ):
        """
        Initialisiert plosive detector.

        Args:
            min_attack_time_ms: Minimum attack time
            max_attack_time_ms: Maximum attack time
            energy_threshold_db: Energy threshold
        """
        self.min_attack_time_ms = min_attack_time_ms
        self.max_attack_time_ms = max_attack_time_ms
        self.energy_threshold_db = energy_threshold_db

    def detect(self, audio: np.ndarray, sr: int) -> list[PlosiveEvent]:
        """
        Erkennt plosive events in audio.

        Plosives are characterized by:
        - Rapid attack (1-20ms)
        - High energy burst
        - Broadband spectrum
        - Short duration

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            List of detected plosive events
        """
        if librosa is None:
            return []
        # Compute onset strength
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)  # type: ignore[attr-defined]

        # Detect onset peaks
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)  # type: ignore[attr-defined]

        plosives = []

        for onset_frame in onset_frames:
            onset_sample = librosa.frames_to_samples(onset_frame)

            # Analyze attack characteristics
            # Look at 50ms window after onset
            window_samples = int(0.05 * sr)
            segment = audio[onset_sample : onset_sample + window_samples]

            if len(segment) < window_samples // 2:
                continue

            # Compute envelope
            envelope = np.abs(signal.hilbert(segment.astype(np.float64)))  # type: ignore[call-overload]

            # Find peak
            peak_idx = np.argmax(envelope)
            peak_sample = onset_sample + peak_idx

            # Attack time: time from onset to peak
            attack_time_ms = peak_idx / sr * 1000

            if not (self.min_attack_time_ms <= attack_time_ms <= self.max_attack_time_ms):
                continue

            # Energy
            energy: float = float(np.max(envelope))
            energy_db = 20 * np.log10(energy + 1e-10)

            if energy_db < self.energy_threshold_db:
                continue

            # Sharpness: rate of energy increase
            attack_segment = envelope[: peak_idx + 1] if peak_idx > 0 else envelope[:1]
            sharpness = float(np.gradient(attack_segment).max()) if len(attack_segment) > 1 else 0.0

            # Confidence based on attack sharpness
            confidence = float(np.clip(sharpness / 0.1, 0, 1))

            plosives.append(
                PlosiveEvent(
                    start_sample=onset_sample,
                    peak_sample=peak_sample,
                    end_sample=onset_sample + window_samples,
                    energy=float(energy),
                    sharpness=sharpness,
                    confidence=confidence,
                )
            )

        return plosives


class TransientDetector:
    """Erkennt transient events (drum hits, percussive sounds)."""

    def __init__(
        self, min_attack_time_ms: float = 0.5, max_attack_time_ms: float = 10.0, energy_threshold_db: float = -20.0
    ):
        """
        Initialisiert transient detector.

        Args:
            min_attack_time_ms: Minimum attack time
            max_attack_time_ms: Maximum attack time
            energy_threshold_db: Energy threshold
        """
        self.min_attack_time_ms = min_attack_time_ms
        self.max_attack_time_ms = max_attack_time_ms
        self.energy_threshold_db = energy_threshold_db

    def detect(self, audio: np.ndarray, sr: int) -> list[TransientEvent]:
        """
        Erkennt transient events in audio.

        Transients are characterized by:
        - Very rapid attack (<10ms)
        - High energy
        - Sharp onset
        - Common in drums, percussion

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            List of detected transient events
        """
        if librosa is None:
            return []
        # Compute onset strength (emphasizes transients)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, aggregate=np.median)  # type: ignore[attr-defined]

        # Detect onsets with high threshold (transients only)
        onset_frames = librosa.onset.onset_detect(  # type: ignore[attr-defined]
            onset_envelope=onset_env,
            sr=sr,
            backtrack=False,
            delta=0.3,  # High threshold for transients only
        )

        transients = []

        for onset_frame in onset_frames:
            onset_sample = librosa.frames_to_samples(onset_frame)

            # Analyze attack (20ms window)
            window_samples = int(0.02 * sr)
            segment = audio[onset_sample : onset_sample + window_samples]

            if len(segment) < window_samples // 2:
                continue

            # Compute envelope
            envelope = np.abs(signal.hilbert(segment.astype(np.float64)))  # type: ignore[call-overload]

            # Find peak
            peak_idx = np.argmax(envelope)
            peak_sample = onset_sample + peak_idx

            # Attack time
            attack_time_ms = peak_idx / sr * 1000

            if not (self.min_attack_time_ms <= attack_time_ms <= self.max_attack_time_ms):
                continue

            # Energy
            energy: float = float(np.max(envelope))
            energy_db = 20 * np.log10(energy + 1e-10)

            if energy_db < self.energy_threshold_db:
                continue

            # Sharpness: very high for drum transients
            attack_segment = envelope[: peak_idx + 1] if peak_idx > 0 else envelope[:1]
            sharpness = float(np.gradient(attack_segment).max()) if len(attack_segment) > 1 else 0.0

            # Confidence: based on attack sharpness and brevity
            confidence = float(np.clip(sharpness / 0.2, 0, 1))

            transients.append(
                TransientEvent(
                    peak_sample=peak_sample,
                    attack_time_ms=attack_time_ms,  # type: ignore[arg-type]
                    energy=float(energy),
                    sharpness=sharpness,
                    confidence=confidence,
                )
            )

        return transients


class SibilanceDetector:
    """Erkennt and analyze sibilance characteristics in audio."""

    def __init__(self, sibilance_range: tuple[int, int] = (4000, 10000), analysis_window_ms: float = 50.0):
        """
        Initialisiert sibilance detector.

        Args:
            sibilance_range: Frequency range for sibilance (Hz)
            analysis_window_ms: Window size for analysis
        """
        self.sibilance_range = sibilance_range
        self.analysis_window_ms = analysis_window_ms

    def analyze(self, audio: np.ndarray, sr: int) -> SibilanceAnalysis:
        """
        Analysiert Sibilanz-Eigenschaften im Audio.

        Sibilance (S, Z, SH sounds) are natural vocal elements that
        should be preserved at natural levels (not over-deessed).

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            SibilanceAnalysis with characteristics
        """
        # Bandpass filter for sibilance range
        sos = signal.butter(4, self.sibilance_range, "bandpass", fs=sr, output="sos")
        sibilance_band = signal.sosfilt(sos, audio)

        # Energy in sibilance band
        sibilance_energy: float = float(np.sum(sibilance_band**2))
        total_energy: float = float(np.sum(audio**2))

        sibilance_density = sibilance_energy / (total_energy + 1e-10)

        # Spectral analysis
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)
        spectrum = np.abs(np.fft.rfft(audio * np.hamming(len(audio))))

        # Find peak sibilance frequency
        sibilance_mask = (freqs >= self.sibilance_range[0]) & (freqs <= self.sibilance_range[1])
        sibilance_spectrum = spectrum[sibilance_mask]
        sibilance_freqs = freqs[sibilance_mask]

        if len(sibilance_spectrum) > 0:
            peak_idx = np.argmax(sibilance_spectrum)
            peak_freq = float(sibilance_freqs[peak_idx])
        else:
            peak_freq = float((self.sibilance_range[0] + self.sibilance_range[1]) / 2)

        # Estimate sibilant content duration
        # Find frames with high sibilance energy
        window_samples = int(self.analysis_window_ms * sr / 1000)
        hop_samples = window_samples // 2

        sibilant_frames = 0
        for i in range(0, len(sibilance_band) - window_samples, hop_samples):
            frame = sibilance_band[i : i + window_samples]
            frame_energy: float = float(np.sum(frame**2))

            # Threshold: significant sibilance energy
            if frame_energy > total_energy / len(audio) * window_samples * 2:
                sibilant_frames += 1

        sibilance_duration = sibilant_frames * hop_samples / sr

        # Natural level check (5-15% sibilance density is natural for vocals)
        natural_level = 0.05 <= sibilance_density <= 0.15

        return SibilanceAnalysis(
            total_sibilance_energy=float(sibilance_energy),
            sibilance_density=float(sibilance_density),
            peak_sibilance_frequency=peak_freq,
            sibilance_duration_sec=float(sibilance_duration),
            natural_level=natural_level,
        )


class RoomToneDetector:
    """Erkennt and analyze natural room tone/ambience."""

    def __init__(self, noise_floor_percentile: float = 10.0, reverb_detection_threshold_db: float = -40.0):
        """
        Initialisiert room tone detector.

        Args:
            noise_floor_percentile: Percentile for noise floor estimation
            reverb_detection_threshold_db: Threshold for reverb tail detection
        """
        self.noise_floor_percentile = noise_floor_percentile
        self.reverb_threshold_db = reverb_detection_threshold_db

    def analyze(self, audio: np.ndarray, sr: int) -> RoomToneAnalysis:
        """
        Analysiert Raumton-Eigenschaften.

        Room tone is the natural acoustic signature of the recording space.
        It should be preserved (not removed by aggressive denoising).

        Args:
            audio: Audio signal (mono or stereo)
            sr: Sample rate

        Returns:
            RoomToneAnalysis with characteristics
        """
        # Convert to mono for analysis — §V7 (copilot-instructions.md):
        # layout-sicher; mean(axis=0) kollabierte (N,2)-Stereo auf (2,).
        if audio.ndim > 1:
            audio_mono = mono_mix(audio)
            is_stereo = True
        else:
            audio_mono = audio
            is_stereo = False

        # 1. Ambient noise floor (quietest 10% of signal)
        sorted_abs = np.sort(np.abs(audio_mono))
        noise_floor_samples = sorted_abs[: int(len(sorted_abs) * self.noise_floor_percentile / 100)]
        noise_floor_rms = np.sqrt(np.mean(noise_floor_samples**2))
        noise_floor_db = 20 * np.log10(noise_floor_rms + 1e-10)

        # 2. Room resonances (modal frequencies)
        # FFT of quiet sections
        quiet_threshold = np.percentile(np.abs(audio_mono), 20)
        quiet_indices = np.where(np.abs(audio_mono) < quiet_threshold)[0]

        room_resonances = []
        if len(quiet_indices) > sr:  # At least 1 second of quiet content
            # Take first second of quiet content
            quiet_segment = audio_mono[quiet_indices[:sr]]

            # FFT
            spectrum = np.abs(np.fft.rfft(quiet_segment * np.hamming(len(quiet_segment))))
            freqs = np.fft.rfftfreq(len(quiet_segment), 1 / sr)

            # Find peaks in low frequency (room modes typically <500 Hz)
            room_mask = freqs < 500
            room_spectrum = spectrum[room_mask]
            room_freqs = freqs[room_mask]

            if len(room_spectrum) > 0:
                peaks, _ = signal.find_peaks(room_spectrum, height=np.max(room_spectrum) * 0.3)
                room_resonances = [float(room_freqs[p]) for p in peaks[:5]]  # Top 5 modes

        # 3. Reverb tail estimation
        # Compute envelope decay time
        envelope = np.abs(signal.hilbert(audio_mono.astype(np.float64)))  # type: ignore[call-overload]

        # Find impulses (sharp peaks)
        impulse_threshold = np.percentile(envelope, 95)
        impulse_indices = np.where(envelope > impulse_threshold)[0]

        reverb_times = []
        for impulse_idx in impulse_indices[:10]:  # Analyze first 10 impulses
            if impulse_idx + int(sr * 0.5) < len(envelope):  # Need 500ms after impulse
                decay_segment = envelope[impulse_idx : impulse_idx + int(sr * 0.5)]

                # Find -60dB decay point
                decay_db = 20 * np.log10(decay_segment / (decay_segment[0] + 1e-10))
                below_threshold = np.where(decay_db < self.reverb_threshold_db)[0]

                if len(below_threshold) > 0:
                    decay_time_ms = below_threshold[0] / sr * 1000
                    reverb_times.append(decay_time_ms)

        reverb_tail_ms = float(np.median(reverb_times)) if reverb_times else 0.0

        # 4. Spatial correlation (stereo width) — §V7 (copilot-instructions.md) layout-sicher
        spatial_correlation = 0.0
        if is_stereo:
            _cf = to_channels_first(audio)
            if _cf.ndim == 2 and _cf.shape[0] >= 2:
                left = _cf[0]
                right = _cf[1]

                # Correlation between channels (NaN-safe: guard against near-constant signals)
                _sl = float(np.std(left))
                _sr = float(np.std(right))
                if _sl > 1e-8 and _sr > 1e-8:
                    _la = left - left.mean()
                    _ra = right - right.mean()
                    _nl = float(np.linalg.norm(_la))
                    _nr = float(np.linalg.norm(_ra))
                    correlation = float(np.dot(_la, _ra) / (_nl * _nr + 1e-10))
                    if not np.isfinite(correlation):
                        correlation = 1.0
                else:
                    correlation = 1.0  # Both constant — mono-equivalent
                spatial_correlation = float(1.0 - abs(correlation))  # 0=mono, 1=wide

        # 5. Naturalness score (heuristic)
        # Natural rooms have:
        # - Moderate noise floor (-60 to -40 dB)
        # - Some room resonances
        # - Short reverb (<300ms typical)
        naturalness = 0.0

        if -60 <= noise_floor_db <= -40:
            naturalness += 0.4
        if len(room_resonances) > 0:
            naturalness += 0.3
        if 10 <= reverb_tail_ms <= 300:
            naturalness += 0.3

        return RoomToneAnalysis(
            ambient_noise_floor_db=float(noise_floor_db),
            room_resonances=room_resonances,
            reverb_tail_length_ms=reverb_tail_ms,
            spatial_correlation=spatial_correlation,
            naturalness_score=float(naturalness),
        )


class AuthenticityMetrics:
    """Berechnet authenticity metrics comparing original vs processed audio."""

    def __init__(self):
        """Initialisiert authenticity metrics computer."""
        self.breath_detector = BreathDetector()
        self.plosive_detector = PlosiveDetector()
        self.transient_detector = TransientDetector()
        self.sibilance_detector = SibilanceDetector()
        self.room_tone_detector = RoomToneDetector()

    def compute_breath_retention(
        self, original: np.ndarray, processed: np.ndarray, sr: int
    ) -> tuple[float, list[BreathEvent], list[BreathEvent]]:
        """
        Berechnet breath retention rate.

        Args:
            original: Original audio
            processed: Processed audio
            sr: Sample rate

        Returns:
            (retention_rate, original_breaths, processed_breaths)
            retention_rate: Percentage of breaths retained (0-1)
        """
        original_breaths = self.breath_detector.detect(original, sr)
        processed_breaths = self.breath_detector.detect(processed, sr)

        if len(original_breaths) == 0:
            return 1.0, original_breaths, processed_breaths  # No breaths to preserve

        # Match breaths by proximity
        matched = 0
        for orig_breath in original_breaths:
            orig_center = (orig_breath.start_sample + orig_breath.end_sample) // 2

            # Find closest breath in processed audio
            for proc_breath in processed_breaths:
                proc_center = (proc_breath.start_sample + proc_breath.end_sample) // 2
                distance_samples = abs(orig_center - proc_center)
                distance_sec = distance_samples / sr

                # If within 100ms, consider it matched
                if distance_sec < 0.1:
                    matched += 1
                    break

        retention_rate = matched / len(original_breaths)
        return retention_rate, original_breaths, processed_breaths

    def compute_plosive_retention(
        self, original: np.ndarray, processed: np.ndarray, sr: int
    ) -> tuple[float, list[PlosiveEvent], list[PlosiveEvent]]:
        """
        Berechnet plosive retention rate.

        Args:
            original: Original audio
            processed: Processed audio
            sr: Sample rate

        Returns:
            (retention_rate, original_plosives, processed_plosives)
        """
        original_plosives = self.plosive_detector.detect(original, sr)
        processed_plosives = self.plosive_detector.detect(processed, sr)

        if len(original_plosives) == 0:
            return 1.0, original_plosives, processed_plosives

        # Match plosives by proximity (within 50ms)
        matched = 0
        for orig_plosive in original_plosives:
            for proc_plosive in processed_plosives:
                distance_samples = abs(orig_plosive.peak_sample - proc_plosive.peak_sample)
                distance_sec = distance_samples / sr

                if distance_sec < 0.05:
                    matched += 1
                    break

        retention_rate = matched / len(original_plosives)
        return retention_rate, original_plosives, processed_plosives

    def compute_transient_preservation(
        self, original: np.ndarray, processed: np.ndarray, sr: int
    ) -> tuple[float, list[TransientEvent], list[TransientEvent]]:
        """
        Berechnet transient preservation rate (sharpness retention).

        Args:
            original: Original audio
            processed: Processed audio
            sr: Sample rate

        Returns:
            (preservation_rate, original_transients, processed_transients)
            preservation_rate: Average sharpness retention (0-1)
        """
        original_transients = self.transient_detector.detect(original, sr)
        processed_transients = self.transient_detector.detect(processed, sr)

        if len(original_transients) == 0:
            return 1.0, original_transients, processed_transients

        # Match transients and compare sharpness
        sharpness_ratios = []

        for orig_trans in original_transients:
            # Find closest transient in processed audio
            best_match = None
            min_distance = float("inf")

            for proc_trans in processed_transients:
                distance_samples = abs(orig_trans.peak_sample - proc_trans.peak_sample)
                distance_sec = distance_samples / sr

                if distance_sec < 0.02 and distance_sec < min_distance:
                    min_distance = distance_sec
                    best_match = proc_trans

            if best_match:
                # Compare sharpness
                if orig_trans.sharpness > 0:
                    ratio = best_match.sharpness / orig_trans.sharpness
                    sharpness_ratios.append(min(ratio, 1.0))  # Cap at 1.0

        if len(sharpness_ratios) == 0:
            return 0.0, original_transients, processed_transients

        preservation_rate = np.mean(sharpness_ratios)
        return float(preservation_rate), original_transients, processed_transients

    def compute_sibilance_retention(
        self, original: np.ndarray, processed: np.ndarray, sr: int
    ) -> tuple[float, SibilanceAnalysis, SibilanceAnalysis]:
        """
        Berechnet sibilance retention rate.

        Measures whether natural sibilance (S, Z, SH sounds) is preserved
        and not over-deessed.

        Args:
            original: Original audio
            processed: Processed audio
            sr: Sample rate

        Returns:
            (retention_rate, original_analysis, processed_analysis)
            retention_rate: Ratio of sibilance energy retained (0-1)
        """
        original_analysis = self.sibilance_detector.analyze(original, sr)
        processed_analysis = self.sibilance_detector.analyze(processed, sr)

        # If no significant sibilance in original, return 1.0
        if original_analysis.sibilance_density < 0.01:
            return 1.0, original_analysis, processed_analysis

        # Compute retention as ratio of densities
        retention_rate = processed_analysis.sibilance_density / original_analysis.sibilance_density

        # Cap at 1.0 (can't have more sibilance than original)
        retention_rate = min(retention_rate, 1.0)

        return float(retention_rate), original_analysis, processed_analysis

    def compute_room_tone_retention(
        self, original: np.ndarray, processed: np.ndarray, sr: int
    ) -> tuple[float, RoomToneAnalysis, RoomToneAnalysis]:
        """
        Berechnet room tone/ambience retention rate.

        Measures whether natural room acoustics are preserved
        and not removed by aggressive denoising.

        Args:
            original: Original audio
            processed: Processed audio
            sr: Sample rate

        Returns:
            (retention_rate, original_analysis, processed_analysis)
            retention_rate: Combined score of room characteristics (0-1)
        """
        original_analysis = self.room_tone_detector.analyze(original, sr)
        processed_analysis = self.room_tone_detector.analyze(processed, sr)

        # Compute retention based on multiple factors
        retention_factors = []

        # 1. Noise floor retention (should not drop too much)
        # If original noise floor is -50dB, processed shouldn't be -80dB
        noise_floor_diff = abs(processed_analysis.ambient_noise_floor_db - original_analysis.ambient_noise_floor_db)

        # Acceptable range: up to 10dB reduction
        if noise_floor_diff <= 10:
            noise_floor_retention = 1.0
        elif noise_floor_diff <= 20:
            noise_floor_retention = 0.5
        else:
            noise_floor_retention = 0.0

        retention_factors.append(noise_floor_retention)

        # 2. Room resonances retention
        if len(original_analysis.room_resonances) > 0:
            # Check how many resonances are still present
            matched_resonances = 0
            for orig_freq in original_analysis.room_resonances:
                for proc_freq in processed_analysis.room_resonances:
                    if abs(orig_freq - proc_freq) < 20:  # Within 20 Hz
                        matched_resonances += 1
                        break

            resonance_retention = matched_resonances / len(original_analysis.room_resonances)
            retention_factors.append(resonance_retention)

        # 3. Reverb tail retention
        if original_analysis.reverb_tail_length_ms > 10:
            reverb_ratio = processed_analysis.reverb_tail_length_ms / original_analysis.reverb_tail_length_ms
            reverb_retention = min(reverb_ratio, 1.0)
            retention_factors.append(reverb_retention)

        # 4. Spatial correlation retention (stereo width)
        if original_analysis.spatial_correlation > 0.1:
            spatial_diff = abs(processed_analysis.spatial_correlation - original_analysis.spatial_correlation)
            spatial_retention = 1.0 - min(spatial_diff, 1.0)
            retention_factors.append(spatial_retention)

        # 5. Naturalness preservation
        naturalness_diff = abs(processed_analysis.naturalness_score - original_analysis.naturalness_score)
        naturalness_retention = 1.0 - min(naturalness_diff, 1.0)
        retention_factors.append(naturalness_retention)

        # Overall retention: weighted average
        if len(retention_factors) > 0:
            retention_rate = np.mean(retention_factors)
        else:
            retention_rate = 1.0  # type: ignore[assignment]  # No room tone to preserve

        return float(retention_rate), original_analysis, processed_analysis


if __name__ == "__main__":
    # Demo
    logger.debug("AURIK Authenticity Metrics")
    logger.debug("=" * 60)
    logger.debug("\nDetectors:")
    logger.debug("  BreathDetector: 50-500ms, 200-3000 Hz")
    logger.debug("  PlosiveDetector: 1-20ms attack")
    logger.debug("  TransientDetector: 0.5-10ms attack")
    logger.debug("  SibilanceDetector: 4-10 kHz, natural level detection")
    logger.debug("  RoomToneDetector: Ambience, resonances, reverb tail")
    logger.debug("\nTarget Metrics:")
    logger.debug("  Breath Retention: >98%%")
    logger.debug("  Vocal Plosive Retention: >95%%")
    logger.debug("  Transient Preservation: >95%%")
    logger.debug("  Sibilance Retention: >95% (not over-deessed)")
    logger.debug("  Room Tone Retention: >90% (natural acoustics)")
