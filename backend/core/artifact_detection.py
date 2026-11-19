"""
Artifact Detection System for AURIK
====================================

Automatically detects processing artifacts in restored audio:
- Pre-ringing / Post-ringing
- Musical noise (from aggressive denoising)
- Frequency smearing
- Temporal smearing
- Clipping artifacts
- Phase distortion
- Spectral holes

Success Criteria (Phase 2D.2.1):
- Artifact Count: <3 audible per minute

Author: AURIK Team
Date: 8. Februar 2026
Phase: 2D.2.1 - Real-World Validation Testing
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


class ArtifactType(Enum):
    """Types of audio processing artifacts."""

    PRE_RINGING = "pre_ringing"  # Gibbs phenomenon before transients
    POST_RINGING = "post_ringing"  # Gibbs phenomenon after transients
    MUSICAL_NOISE = "musical_noise"  # Random tonal artifacts from denoising
    FREQUENCY_SMEARING = "frequency_smearing"  # Loss of frequency resolution
    TEMPORAL_SMEARING = "temporal_smearing"  # Loss of time resolution
    CLIPPING = "clipping"  # Introduced clipping
    PHASE_DISTORTION = "phase_distortion"  # Non-linear phase shifts
    SPECTRAL_HOLES = "spectral_holes"  # Missing frequency bands
    PUMPING_BREATHING = "pumping_breathing"  # Gain fluctuations
    ALIASING = "aliasing"  # High-frequency aliasing


class ArtifactSeverity(Enum):
    """Severity levels for detected artifacts."""

    NONE = 0  # No artifacts detected
    MILD = 1  # Barely noticeable
    MODERATE = 2  # Noticeable but acceptable
    SEVERE = 3  # Clearly audible, problematic
    CRITICAL = 4  # Extremely problematic


@dataclass
class Artifact:
    """Container for detected artifact."""

    artifact_type: ArtifactType
    severity: ArtifactSeverity
    start_time: float  # seconds
    duration: float  # seconds
    confidence: float  # [0, 1]
    description: str
    metadata: dict  # Additional info


@dataclass
class ArtifactAnalysisResult:
    """Results of artifact detection analysis."""

    artifacts: list[Artifact]
    total_count: int
    audible_count: int  # MODERATE or higher
    artifacts_per_minute: float
    overall_severity: ArtifactSeverity
    passes_aurik_standards: bool  # <3 audible per minute

    def get_by_type(self, artifact_type: ArtifactType) -> list[Artifact]:
        """Gibt zurück: all artifacts of specific type."""
        return [a for a in self.artifacts if a.artifact_type == artifact_type]

    def get_by_severity(self, min_severity: ArtifactSeverity) -> list[Artifact]:
        """Gibt zurück: artifacts above minimum severity."""
        return [a for a in self.artifacts if a.severity.value >= min_severity.value]


class RestorationArtifactDetector:
    """
    Erkennt processing artifacts in audio.

    Usage:
        detector = RestorationArtifactDetector()
        result = detector.analyze(
            original_audio,
            restored_audio,
            sr=48000
        )
        logger.debug("Artifacts per minute: %.1f", result.artifacts_per_minute)
    """

    def __init__(
        self,
        sensitivity: float = 0.5,
        frame_size: int = 2048,
        hop_size: int = 512,  # [0, 1], higher = more sensitive
    ):
        self.sensitivity = sensitivity
        self.frame_size = frame_size
        self.hop_size = hop_size

    # ============================================================
    # Main Analysis
    # ============================================================

    def analyze(self, original: np.ndarray, restored: np.ndarray, sr: int = 48000) -> ArtifactAnalysisResult:
        """
        Comprehensive artifact detection.

        Args:
            original: Original audio (pre-restoration)
            restored: Restored audio (post-processing)
            sr: Sample rate

        Returns:
            ArtifactAnalysisResult with all detected artifacts
        """
        # Ensure same length
        min_len = min(len(original), len(restored))
        original = original[:min_len]
        restored = restored[:min_len]

        duration = len(restored) / sr
        artifacts = []

        # Detect each artifact type
        artifacts.extend(self._detect_ringing(restored, sr))
        artifacts.extend(self._detect_musical_noise(original, restored, sr))
        artifacts.extend(self._detect_frequency_smearing(original, restored, sr))
        artifacts.extend(self._detect_temporal_smearing(original, restored, sr))
        artifacts.extend(self._detect_clipping(restored, sr))
        artifacts.extend(self._detect_phase_distortion(original, restored, sr))
        artifacts.extend(self._detect_spectral_holes(original, restored, sr))
        artifacts.extend(self._detect_pumping_breathing(restored, sr))
        artifacts.extend(self._detect_aliasing(restored, sr))

        # Filter by confidence
        artifacts = [a for a in artifacts if a.confidence >= self.sensitivity]

        # Sort by time
        artifacts.sort(key=lambda a: a.start_time)

        # Count audible artifacts (MODERATE or higher)
        audible_count = sum(1 for a in artifacts if a.severity.value >= ArtifactSeverity.MODERATE.value)

        # Compute artifacts per minute
        artifacts_per_minute = audible_count / (duration / 60.0) if duration > 0 else 0

        # Overall severity
        if audible_count == 0:
            overall_severity = ArtifactSeverity.NONE
        else:
            max_severity = max(a.severity.value for a in artifacts)
            overall_severity = ArtifactSeverity(max_severity)

        # AURIK standard: <3 audible per minute
        passes = artifacts_per_minute < 3.0

        return ArtifactAnalysisResult(
            artifacts=artifacts,
            total_count=len(artifacts),
            audible_count=audible_count,
            artifacts_per_minute=artifacts_per_minute,
            overall_severity=overall_severity,
            passes_aurik_standards=passes,
        )

    # ============================================================
    # Individual Detectors
    # ============================================================

    def _detect_ringing(self, audio: np.ndarray, sr: int) -> list[Artifact]:
        """Erkennt pre-ringing and post-ringing artifacts."""
        artifacts = []

        # Detect transients
        transient_indices = self._find_transients(audio, sr)

        for idx in transient_indices:
            # Check pre-ringing (before transient)
            pre_window_start = max(0, idx - int(0.01 * sr))  # 10ms before
            pre_window = audio[pre_window_start:idx]

            if len(pre_window) > 0:
                pre_energy = np.sqrt(np.mean(pre_window**2))
                transient_energy = np.abs(audio[idx])

                # Pre-ringing if energy before transient is >10% of transient
                if pre_energy > 0.1 * transient_energy and transient_energy > 0.1:
                    severity = self._classify_severity(pre_energy / transient_energy, [0.1, 0.2, 0.3, 0.5])

                    artifacts.append(
                        Artifact(
                            artifact_type=ArtifactType.PRE_RINGING,
                            severity=severity,
                            start_time=pre_window_start / sr,
                            duration=(idx - pre_window_start) / sr,
                            confidence=min(1.0, (pre_energy / transient_energy) / 0.5),
                            description=f"Pre-ringing detected before transient at {idx / sr:.3f}s",
                            metadata={"transient_idx": idx, "ratio": pre_energy / transient_energy},
                        )
                    )

            # Check post-ringing (after transient)
            post_window_end = min(len(audio), idx + int(0.02 * sr))  # 20ms after
            post_window = audio[idx + 1 : post_window_end]

            if len(post_window) > 0:
                # Compute decay
                decay = np.sqrt(np.mean(post_window**2))
                transient_energy = np.abs(audio[idx])

                # Post-ringing if decay is too slow
                if decay > 0.15 * transient_energy and transient_energy > 0.1:
                    severity = self._classify_severity(decay / transient_energy, [0.15, 0.25, 0.35, 0.5])

                    artifacts.append(
                        Artifact(
                            artifact_type=ArtifactType.POST_RINGING,
                            severity=severity,
                            start_time=idx / sr,
                            duration=(post_window_end - idx) / sr,
                            confidence=min(1.0, (decay / transient_energy) / 0.5),
                            description=f"Post-ringing detected after transient at {idx / sr:.3f}s",
                            metadata={"transient_idx": idx, "decay_ratio": decay / transient_energy},
                        )
                    )

        return artifacts

    def _detect_musical_noise(self, original: np.ndarray, restored: np.ndarray, sr: int) -> list[Artifact]:
        """Erkennt musical noise artifacts from aggressive denoising."""
        artifacts = []

        # Compute spectrograms
        original_spec = self._compute_spectrogram(original)
        restored_spec = self._compute_spectrogram(restored)

        # Musical noise = isolated spectral peaks that weren't in original
        # Compute spectral variance over time
        restored_variance = np.var(restored_spec, axis=1)
        original_variance = np.var(original_spec, axis=1)

        # High variance in restored but not in original = musical noise
        variance_increase = restored_variance - original_variance

        # Find frames with suspicious variance increase
        threshold = np.percentile(variance_increase, 95)
        suspicious_frames = np.where(variance_increase > threshold)[0]

        # Group adjacent frames
        if len(suspicious_frames) > 0:
            groups = self._group_adjacent_indices(suspicious_frames, max_gap=5)

            for group in groups:
                start_frame = group[0]
                end_frame = group[-1]

                start_time = start_frame * self.hop_size / sr
                duration = (end_frame - start_frame) * self.hop_size / sr

                # Compute severity based on variance increase
                max_variance_increase: float = float(np.max(variance_increase[group]))
                severity = self._classify_severity(
                    float(max_variance_increase / (float(np.mean(original_variance)) + 1e-6)),
                    [0.5, 1.0, 2.0, 5.0],
                )
                confidence = float(np.clip(max_variance_increase / threshold, 0.0, 1.0))

                artifacts.append(
                    Artifact(
                        artifact_type=ArtifactType.MUSICAL_NOISE,
                        severity=severity,
                        start_time=start_time,
                        duration=duration,
                        confidence=confidence,
                        description=f"Musical noise detected at {start_time:.2f}s",
                        metadata={"variance_increase": max_variance_increase},
                    )
                )

        return artifacts

    def _detect_frequency_smearing(self, original: np.ndarray, restored: np.ndarray, sr: int) -> list[Artifact]:
        """Erkennt frequency smearing (loss of frequency resolution)."""
        artifacts = []

        # Compute spectrograms
        original_spec = self._compute_spectrogram(original)
        restored_spec = self._compute_spectrogram(restored)

        # Frequency smearing = reduced spectral resolution
        # Compute spectral flatness (measure of smoothness)
        original_flatness = self._compute_spectral_flatness(original_spec)
        restored_flatness = self._compute_spectral_flatness(restored_spec)

        # Increased flatness = smearing
        flatness_increase = restored_flatness - original_flatness

        # Find frames with significant smearing
        threshold = np.percentile(flatness_increase, 90)
        smeared_frames = np.where(flatness_increase > threshold)[0]

        if len(smeared_frames) > 10:  # Need sustained smearing
            # Report overall smearing
            start_time = smeared_frames[0] * self.hop_size / sr
            duration = len(smeared_frames) * self.hop_size / sr

            severity = self._classify_severity(
                float(np.mean(flatness_increase[smeared_frames])), [0.05, 0.1, 0.15, 0.25]
            )

            artifacts.append(
                Artifact(
                    artifact_type=ArtifactType.FREQUENCY_SMEARING,
                    severity=severity,
                    start_time=start_time,
                    duration=duration,
                    confidence=0.8,
                    description=f"Frequency smearing detected ({len(smeared_frames)} frames)",
                    metadata={"affected_frames": len(smeared_frames)},
                )
            )

        return artifacts

    def _detect_temporal_smearing(self, original: np.ndarray, restored: np.ndarray, sr: int) -> list[Artifact]:
        """Erkennt temporal smearing (loss of time resolution)."""
        artifacts = []

        # Temporal smearing = blurred transients
        # Compute onset strength
        original_onsets = self._compute_onset_strength(original, sr)
        restored_onsets = self._compute_onset_strength(restored, sr)

        # Reduced onset strength = temporal smearing
        onset_reduction = original_onsets - restored_onsets

        # Find locations with significant reduction
        threshold = np.percentile(onset_reduction, 90)
        smeared_indices = np.where(onset_reduction > threshold)[0]

        if len(smeared_indices) > 5:
            groups = self._group_adjacent_indices(smeared_indices, max_gap=10)

            for group in groups:
                start_time = group[0] * self.hop_size / sr
                duration = len(group) * self.hop_size / sr

                severity = self._classify_severity(float(np.mean(onset_reduction[group])), [0.1, 0.2, 0.3, 0.5])

                artifacts.append(
                    Artifact(
                        artifact_type=ArtifactType.TEMPORAL_SMEARING,
                        severity=severity,
                        start_time=start_time,
                        duration=duration,
                        confidence=0.7,
                        description=f"Temporal smearing detected at {start_time:.2f}s",
                        metadata={"affected_transients": len(group)},
                    )
                )

        return artifacts

    def _detect_clipping(self, audio: np.ndarray, sr: int) -> list[Artifact]:
        """Erkennt introduced clipping artifacts."""
        artifacts = []

        # Find clipped samples (near ±1.0)
        clipped_indices = np.where(np.abs(audio) >= 0.99)[0]

        if len(clipped_indices) > 0:
            # Group adjacent clipped samples
            groups = self._group_adjacent_indices(clipped_indices, max_gap=5)

            for group in groups:
                if len(group) >= 3:  # At least 3 consecutive samples
                    start_time = group[0] / sr
                    duration = len(group) / sr

                    # Severity based on duration
                    severity = self._classify_severity(duration * 1000, [0.1, 0.5, 1.0, 5.0])  # Convert to milliseconds

                    artifacts.append(
                        Artifact(
                            artifact_type=ArtifactType.CLIPPING,
                            severity=severity,
                            start_time=start_time,
                            duration=duration,
                            confidence=1.0,
                            description=f"Clipping detected at {start_time:.3f}s ({len(group)} samples)",
                            metadata={"samples_clipped": len(group)},
                        )
                    )

        return artifacts

    def _detect_phase_distortion(self, original: np.ndarray, restored: np.ndarray, sr: int) -> list[Artifact]:
        """Erkennt non-linear phase distortion."""
        artifacts = []

        # Compute instantaneous phase difference
        original_analytic = self._hilbert_transform(original)
        restored_analytic = self._hilbert_transform(restored)

        original_phase = np.angle(original_analytic)
        restored_phase = np.angle(restored_analytic)

        # Phase difference
        phase_diff = np.abs(restored_phase - original_phase)
        phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)  # Wrap to [0, π]

        # Non-linear phase distortion = large phase differences
        threshold = np.pi / 4  # 45 degrees
        distorted_indices = np.where(phase_diff > threshold)[0]

        if len(distorted_indices) > sr // 10:  # >100ms worth
            # Report overall phase distortion
            severity = self._classify_severity(
                float(np.mean(phase_diff[distorted_indices])), [np.pi / 4, np.pi / 3, np.pi / 2, np.pi]
            )

            artifacts.append(
                Artifact(
                    artifact_type=ArtifactType.PHASE_DISTORTION,
                    severity=severity,
                    start_time=0.0,
                    duration=len(restored) / sr,
                    confidence=0.6,
                    description=f"Phase distortion detected ({len(distorted_indices)} samples affected)",
                    metadata={"mean_phase_diff_rad": np.mean(phase_diff[distorted_indices])},
                )
            )

        return artifacts

    def _detect_spectral_holes(self, original: np.ndarray, restored: np.ndarray, sr: int) -> list[Artifact]:
        """Erkennt spectral holes (missing frequency bands)."""
        artifacts = []

        # Compute average spectrum
        original_spectrum = np.abs(np.fft.rfft(original))
        restored_spectrum = np.abs(np.fft.rfft(restored))

        # Spectral holes = regions where restored has much less energy
        ratio = restored_spectrum / (original_spectrum + 1e-10)

        # Find frequency bins with severe attenuation
        threshold = 0.3  # >70% energy loss
        holes = np.where(ratio < threshold)[0]

        if len(holes) > 10:
            # Find continuous holes
            groups = self._group_adjacent_indices(holes, max_gap=5)

            for group in groups:
                if len(group) >= 5:  # At least 5 bins
                    center_freq = (group[len(group) // 2] / len(restored_spectrum)) * (sr / 2)
                    bandwidth = (len(group) / len(restored_spectrum)) * (sr / 2)

                    severity = self._classify_severity(float(np.mean(ratio[group])), [0.3, 0.2, 0.1, 0.05])

                    artifacts.append(
                        Artifact(
                            artifact_type=ArtifactType.SPECTRAL_HOLES,
                            severity=severity,
                            start_time=0.0,
                            duration=len(restored) / sr,
                            confidence=0.8,
                            description=f"Spectral hole detected at {center_freq:.0f}Hz (width: {bandwidth:.0f}Hz)",
                            metadata={"center_freq_hz": center_freq, "bandwidth_hz": bandwidth},
                        )
                    )

        return artifacts

    def _detect_pumping_breathing(self, audio: np.ndarray, sr: int) -> list[Artifact]:
        """Erkennt pumping/breathing artifacts (gain fluctuations)."""
        artifacts = []

        # Compute RMS envelope
        frame_length = int(0.05 * sr)  # 50ms frames
        rms_values: list[float] = []

        for i in range(0, len(audio) - frame_length, frame_length // 2):
            rms = float(np.sqrt(np.mean(np.square(audio[i : i + frame_length]))))
            rms_values.append(rms)

        rms_envelope = np.asarray(rms_values, dtype=np.float64)

        # Pumping = rapid RMS fluctuations
        rms_diff = np.abs(np.diff(rms_envelope))
        threshold = np.percentile(rms_diff, 95)

        fluctuating_frames = np.where(rms_diff > threshold)[0]

        if len(fluctuating_frames) > 5:
            severity = self._classify_severity(
                float(np.max(rms_diff[fluctuating_frames])),
                [float(threshold), float(threshold * 1.5), float(threshold * 2), float(threshold * 3)],
            )

            artifacts.append(
                Artifact(
                    artifact_type=ArtifactType.PUMPING_BREATHING,
                    severity=severity,
                    start_time=0.0,
                    duration=len(audio) / sr,
                    confidence=0.7,
                    description=f"Pumping/breathing detected ({len(fluctuating_frames)} events)",
                    metadata={"num_events": len(fluctuating_frames)},
                )
            )

        return artifacts

    def _detect_aliasing(self, audio: np.ndarray, sr: int) -> list[Artifact]:
        """Erkennt aliasing artifacts."""
        artifacts = []

        # Compute spectrum
        spectrum = np.abs(np.fft.rfft(audio))

        # Aliasing = unexpected energy near Nyquist frequency
        nyquist_idx = int(len(spectrum) * 0.9)  # Above 90% of Nyquist
        nyquist_energy = np.mean(spectrum[nyquist_idx:])
        total_energy = np.mean(spectrum)

        ratio = nyquist_energy / (total_energy + 1e-10)

        if ratio > 0.1:  # >10% energy near Nyquist
            severity = self._classify_severity(float(ratio), [0.1, 0.2, 0.3, 0.5])

            artifacts.append(
                Artifact(
                    artifact_type=ArtifactType.ALIASING,
                    severity=severity,
                    start_time=0.0,
                    duration=len(audio) / sr,
                    confidence=0.8,
                    description=f"Aliasing detected (ratio: {ratio:.2%})",
                    metadata={"energy_ratio": ratio},
                )
            )

        return artifacts

    # ============================================================
    # Helper Methods
    # ============================================================

    def _find_transients(self, audio: np.ndarray, sr: int, threshold: float = 0.1) -> list[int]:
        """Findet Transient-Positionen im Audio."""
        # Compute onset strength
        onset_envelope = self._compute_onset_strength(audio, sr)

        # Find peaks
        transients = []
        for i in range(1, len(onset_envelope) - 1):
            if (
                onset_envelope[i] > threshold
                and onset_envelope[i] > onset_envelope[i - 1]
                and onset_envelope[i] > onset_envelope[i + 1]
            ):
                transients.append(i * self.hop_size)

        return transients

    def _compute_onset_strength(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Berechnet onset strength envelope."""
        spec = self._compute_spectrogram(audio)

        # Onset = increase in spectral energy
        onset_strength = np.maximum(0, np.diff(spec, axis=1))
        onset_strength = np.mean(onset_strength, axis=0)

        # Pad to match original length
        onset_strength = np.concatenate([[0], onset_strength])

        return onset_strength  # type: ignore[no-any-return]

    def _compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Berechnet magnitude spectrogram."""
        hop_length = self.hop_size
        n_fft = self.frame_size

        # Compute STFT
        num_frames = (len(audio) - n_fft) // hop_length + 1
        spec = np.zeros((n_fft // 2 + 1, num_frames))

        for i in range(num_frames):
            start = i * hop_length
            frame = audio[start : start + n_fft]

            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))

            # Apply window
            window = np.hanning(n_fft)
            frame = frame * window

            # FFT
            fft_frame = np.fft.rfft(frame)
            spec[:, i] = np.abs(fft_frame)

        return spec  # type: ignore[no-any-return]

    def _compute_spectral_flatness(self, spec: np.ndarray) -> np.ndarray:
        """Berechnet spectral flatness for each frame."""
        # Spectral flatness = geometric mean / arithmetic mean
        # High flatness = noise-like, Low flatness = tonal

        # Avoid log(0)
        spec_safe = spec + 1e-10

        log_spec = np.asarray(np.log(spec_safe), dtype=np.float64)
        mean_log_spec = np.asarray(np.mean(log_spec, axis=0), dtype=np.float64)
        geometric_mean = np.asarray(np.exp(mean_log_spec), dtype=np.float64)
        arithmetic_mean = np.asarray(np.mean(spec_safe, axis=0), dtype=np.float64)

        flatness = geometric_mean / (arithmetic_mean + 1e-10)

        flatness_array = np.asarray(flatness, dtype=np.float64)
        return cast(np.ndarray, flatness_array)

    def _hilbert_transform(self, audio: np.ndarray) -> np.ndarray:
        """Berechnet analytic signal via Hilbert transform."""
        # Simple Hilbert transform via FFT
        fft = np.fft.fft(audio)
        n = len(audio)

        # Zero out negative frequencies
        h = np.zeros(n)
        h[0] = 1
        h[1 : n // 2] = 2

        analytic: np.ndarray = np.asarray(np.fft.ifft(fft * h))

        analytic_array: np.ndarray = np.asarray(analytic)
        return analytic_array

    def _group_adjacent_indices(self, indices: np.ndarray, max_gap: int = 1) -> list[list[int]]:
        """Group adjacent indices into continuous segments."""
        if len(indices) == 0:
            return []

        groups = []
        current_group = [indices[0]]

        for i in range(1, len(indices)):
            if indices[i] - indices[i - 1] <= max_gap:
                current_group.append(indices[i])
            else:
                groups.append(current_group)
                current_group = [indices[i]]

        groups.append(current_group)

        return groups  # type: ignore[return-value]

    def _classify_severity(
        self,
        value: float,
        thresholds: list[float],  # [mild, moderate, severe, critical]
    ) -> ArtifactSeverity:
        """Classify severity based on value and thresholds."""
        if value < thresholds[0]:
            return ArtifactSeverity.MILD
        elif value < thresholds[1]:
            return ArtifactSeverity.MODERATE
        elif value < thresholds[2]:
            return ArtifactSeverity.SEVERE
        else:
            return ArtifactSeverity.CRITICAL


# ============================================================
# Convenience Functions
# ============================================================


def quick_artifact_check(original: np.ndarray, restored: np.ndarray, sr: int = 48000) -> bool:
    """
    Quick artifact check: passes if <3 artifacts per minute.

    Returns:
        True if passes AURIK standards, False otherwise
    """
    detector = RestorationArtifactDetector()
    result = detector.analyze(original, restored, sr)
    return result.passes_aurik_standards


def generate_artifact_report(result: ArtifactAnalysisResult, audio_duration: float) -> str:
    """Generiert human-readable artifact report."""
    lines = []
    lines.append("=" * 60)
    lines.append("AURIK Artifact Detection Report")
    lines.append("=" * 60)
    lines.append(f"Audio Duration: {audio_duration:.2f}s")
    lines.append("")

    lines.append("Summary:")
    lines.append(f"  Total Artifacts:   {result.total_count}")
    lines.append(f"  Audible Artifacts: {result.audible_count}")
    lines.append(f"  Artifacts/Minute:  {result.artifacts_per_minute:.2f}")
    lines.append(f"  Overall Severity:  {result.overall_severity.name}")
    lines.append("")

    # AURIK standard
    status = "✅ PASSED" if result.passes_aurik_standards else "❌ FAILED"
    lines.append(f"AURIK Standard (<3/min): {status}")
    lines.append("")

    if result.total_count > 0:
        lines.append("Detected Artifacts:")
        lines.append("")

        for art in result.artifacts:
            if art.severity.value >= ArtifactSeverity.MODERATE.value:
                lines.append(f"  [{art.artifact_type.value}] {art.severity.name}")
                lines.append(f"    Time: {art.start_time:.2f}s - {art.start_time + art.duration:.2f}s")
                lines.append(f"    {art.description}")
                lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)
