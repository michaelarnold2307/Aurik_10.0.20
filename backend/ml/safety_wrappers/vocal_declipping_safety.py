"""
vocal_declipping_safety.py - HIPS-Compliant Vocal Declipping Safety Wrapper

Wraps vocal declipping DSP operations with comprehensive safety checks:
- Clipping severity detection (mild/moderate/severe)
- Harmonic structure validation
- Restoration quality verification
- No new artifacts introduced (intermodulation, aliasing)
- Natural voice timbre preservation

This ensures declipping repairs distorted vocals without creating synthetic artifacts.

Author: AURIK Team
Version: 1.0.0
Date: 7. Februar 2026
Phase: 1 Week 5-6
"""

from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal as signal

from .safety_wrapper_template import (
    BaseSafetyWrapper,
    PostCheckResult,
    PreCheckResult,
    compute_energy_ratio,
    compute_spectral_centroid,
    validate_audio_basic,
)

# ============================================================================
# CLIPPING DETECTION UTILITIES
# ============================================================================


def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> tuple[bool, float, dict[str, Any]]:
    """
    Erkennt clipping in audio signal.

    Args:
        audio: Input audio
        threshold: Amplitude threshold for clipping detection

    Returns:
        (is_clipped, severity, metadata): Detection result
    """
    # Flatten if stereo
    audio_flat = audio.flatten() if audio.ndim > 1 else audio

    # Find clipped samples
    clipped_samples = np.abs(audio_flat) >= threshold

    n_clipped = np.sum(clipped_samples)
    total_samples = len(audio_flat)

    clipping_ratio = n_clipped / total_samples

    # Detect consecutive clipped samples (hard clipping)
    consecutive_clips = 0
    max_clip_length = 0
    current_clip_length = 0

    for is_clip in clipped_samples:
        if is_clip:
            current_clip_length += 1
            max_clip_length = max(max_clip_length, current_clip_length)
        else:
            if current_clip_length > 0:
                consecutive_clips += 1
            current_clip_length = 0

    # Severity classification
    if clipping_ratio < 0.001:
        severity = "none"
        severity_score = 0.0
    elif clipping_ratio < 0.01:
        severity = "mild"
        severity_score = 0.3
    elif clipping_ratio < 0.05:
        severity = "moderate"
        severity_score = 0.6
    else:
        severity = "severe"
        severity_score = 1.0

    metadata = {
        "clipped_samples": int(n_clipped),
        "total_samples": int(total_samples),
        "clipping_ratio": float(clipping_ratio),
        "consecutive_clip_regions": int(consecutive_clips),
        "max_clip_length_samples": int(max_clip_length),
        "severity_class": severity,
        "severity_score": severity_score,
    }

    is_clipped = clipping_ratio > 0.001

    return is_clipped, severity_score, metadata


def estimate_thd(audio: np.ndarray, sr: int, fundamental_hz: float | None = None) -> float:
    """
    Schätzt Total Harmonic Distortion (THD).

    Higher THD indicates more distortion (clipping, overdrive, etc.)

    Args:
        audio: Input audio
        sr: Sample rate
        fundamental_hz: Optional fundamental frequency (if known)

    Returns:
        THD percentage (0.0-1.0)
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Compute spectrum
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)

    # Find fundamental (if not provided)
    if fundamental_hz is None:
        # Look in 80-400 Hz range for voice
        f0_band = (freqs >= 80) & (freqs <= 400)
        if np.any(f0_band):
            fundamental_hz = freqs[f0_band][np.argmax(spectrum[f0_band])]
        else:
            return 0.0  # Can't compute THD without fundamental

    if fundamental_hz is None:
        return 0.0

    fundamental_hz_val = float(fundamental_hz)

    # Find harmonics (up to 10th harmonic)
    fundamental_power = 0.0
    harmonic_power = 0.0

    for n in range(1, 11):
        harmonic_freq = fundamental_hz_val * n

        # Find nearest frequency bin
        idx = np.argmin(np.abs(freqs - harmonic_freq))

        if freqs[idx] > sr / 2:
            break

        power = spectrum[idx] ** 2

        if n == 1:
            fundamental_power = power
        else:
            harmonic_power += power

    if fundamental_power == 0:
        return 0.0

    thd = np.sqrt(harmonic_power / fundamental_power)

    return float(np.clip(thd, 0.0, 1.0))


def detect_harmonic_structure(audio: np.ndarray, sr: int) -> tuple[bool, float]:
    """
    Erkennt if audio has clear harmonic structure (periodic signal).

    Voices have harmonic structure. Clipping can destroy it.

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        (has_harmonics, strength): Detection result
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Autocorrelation method to detect periodicity — FFT-based O(N log N)
    from backend.core.core_utils import fft_autocorr

    autocorr = fft_autocorr(audio)

    # Normalize
    autocorr = autocorr / autocorr[0]

    # Find first peak after initial (indicates periodicity)
    # Look for peaks in 80-400 Hz range (voice F0)
    min_lag = int(sr / 400)  # Highest F0
    max_lag = int(sr / 80)  # Lowest F0

    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1

    if min_lag >= max_lag:
        return False, 0.0

    autocorr_region = autocorr[min_lag:max_lag]

    if len(autocorr_region) == 0:
        return False, 0.0

    max_autocorr = np.max(autocorr_region)

    # Strong harmonics if autocorrelation peak > 0.5
    has_harmonics = max_autocorr > 0.5
    strength = float(np.clip(max_autocorr, 0.0, 1.0))

    return has_harmonics, strength


def compute_crest_factor(audio: np.ndarray) -> float:
    """
    Berechnet crest factor (peak-to-RMS ratio).

    Clipped audio has low crest factor.
    Natural voice has high crest factor (15-20 dB).

    Args:
        audio: Input audio

    Returns:
        Crest factor in dB
    """
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))

    if rms == 0:
        return 0.0

    crest_linear = peak / rms
    crest_db = 20 * np.log10(crest_linear + 1e-8)

    return float(crest_db)


# ============================================================================
# VOCAL DECLIPPING SAFETY WRAPPER
# ============================================================================


class VocalDeclippingSafety(BaseSafetyWrapper):
    """
    HIPS-compliant safety wrapper for vocal declipping.

    Ensures:
    - Only processes audio with detected clipping
    - Validates harmonic structure present (can be restored)
    - Verifies restoration quality (no new artifacts)
    - Preserves voice timbre and naturalness
    - Improves crest factor (dynamics)
    """

    def __init__(
        self,
        processor_func,
        enable_logging: bool = True,
        log_dir: Path | None = None,
        min_clipping_severity: float = 0.2,
        max_thd_increase: float = 0.1,
    ):
        """
        Initialisiert Vocal Declipping Safety Wrapper.

        Args:
            processor_func: Declipping function (audio, sr, severity) -> audio
            enable_logging: Enable audit trail
            log_dir: Audit log directory
            min_clipping_severity: Minimum severity to process
            max_thd_increase: Maximum allowed THD increase
        """
        super().__init__(
            module_name="VocalDeclipping",
            module_version="1.0.0",
            processor_func=processor_func,
            enable_logging=enable_logging,
            log_dir=log_dir,
            confidence_threshold=0.5,
            quality_threshold=0.65,
        )

        self.min_clipping_severity = min_clipping_severity
        self.max_thd_increase = max_thd_increase

    def _validate_pre_conditions(self, audio: np.ndarray, sr: int, **params) -> PreCheckResult:
        """Validiert pre-conditions for declipping."""
        # Basic audio validation
        is_valid, errors = validate_audio_basic(audio)

        if not is_valid:
            return PreCheckResult(passed=False, confidence=0.0, reasons=errors)

        warnings = []
        metadata = {}

        # Detect clipping
        is_clipped, severity, clip_metadata = detect_clipping(audio)
        metadata.update(clip_metadata)

        if not is_clipped:
            return PreCheckResult(passed=False, confidence=0.0, reasons=["No clipping detected in audio"])

        if severity < self.min_clipping_severity:
            return PreCheckResult(
                passed=False,
                confidence=severity,
                reasons=[f"Clipping too mild: {severity:.2f} (min {self.min_clipping_severity})"],
            )

        # Check for harmonic structure (needed for restoration)
        has_harmonics, harmonic_strength = detect_harmonic_structure(audio, sr)
        metadata["has_harmonic_structure"] = has_harmonics
        metadata["harmonic_strength"] = harmonic_strength

        if not has_harmonics:
            return PreCheckResult(
                passed=False,
                confidence=severity,
                reasons=["No harmonic structure detected. Cannot reliably restore clipped signal."],
            )

        if harmonic_strength < 0.4:
            warnings.append(f"Weak harmonic structure: {harmonic_strength:.2f}. Restoration quality may be limited.")

        # Compute initial THD
        thd_before = estimate_thd(audio, sr)
        metadata["thd_before"] = thd_before

        if thd_before > 0.5:
            warnings.append(f"High initial THD: {thd_before:.1%}. Audio is heavily distorted.")

        # Compute initial crest factor
        crest_before = compute_crest_factor(audio)
        metadata["crest_factor_before"] = crest_before

        if crest_before < 6:  # Severely compressed/clipped
            warnings.append(f"Very low crest factor: {crest_before:.1f} dB. Indicates severe dynamic range loss.")

        return PreCheckResult(
            passed=True, confidence=severity * harmonic_strength, warnings=warnings, metadata=metadata
        )

    def _assess_epistemic_confidence(self, audio: np.ndarray, sr: int, pre_check: PreCheckResult, **params) -> float:
        """Assess confidence in declipping for this audio."""
        # Base confidence from clipping severity
        severity = pre_check.metadata.get("severity_score", 0.5)

        # Harmonic structure strength
        harmonic_strength = pre_check.metadata.get("harmonic_strength", 0.5)

        # Higher confidence = moderate clipping + strong harmonics
        # Lower confidence = severe clipping (less predictable) or weak harmonics

        # Optimal severity is moderate (0.5-0.7)
        if severity < 0.5:
            severity_factor = severity / 0.5  # Scale up from mild
        else:
            severity_factor = 1.0 - (severity - 0.5)  # Scale down from severe

        confidence = severity_factor * 0.5 + harmonic_strength * 0.5

        return float(np.clip(confidence, 0.0, 1.0))

    def _validate_post_conditions(
        self, original: np.ndarray, processed: np.ndarray, sr: int, **params
    ) -> PostCheckResult:
        """Validiert post-conditions after declipping."""
        issues = []
        side_effects = []
        metrics = {}

        # Ensure same shape
        if original.shape != processed.shape:
            issues.append(f"Shape mismatch: {original.shape} -> {processed.shape}")
            return PostCheckResult(passed=False, quality_score=0.0, issues=issues)

        # 1. Check clipping reduction
        _is_clipped_before, severity_before, _ = detect_clipping(original)
        _is_clipped_after, severity_after, clip_metadata_after = detect_clipping(processed)

        metrics["severity_before"] = severity_before
        metrics["severity_after"] = severity_after
        metrics["clipped_samples_after"] = clip_metadata_after["clipped_samples"]

        if severity_after >= severity_before:
            issues.append(f"Clipping not reduced: {severity_before:.2f} -> {severity_after:.2f}")

        clipping_reduction = severity_before - severity_after
        metrics["clipping_reduction"] = float(clipping_reduction)

        # 2. Check harmonic structure preservation
        _has_harm_before, strength_before = detect_harmonic_structure(original, sr)
        has_harm_after, strength_after = detect_harmonic_structure(processed, sr)

        metrics["harmonic_strength_before"] = strength_before
        metrics["harmonic_strength_after"] = strength_after

        if not has_harm_after:
            issues.append("Harmonic structure lost after processing")
        elif strength_after < strength_before * 0.8:
            side_effects.append(f"Harmonic structure weakened: {strength_before:.2f} -> {strength_after:.2f}")

        # 3. Check THD
        thd_before = estimate_thd(original, sr)
        thd_after = estimate_thd(processed, sr)

        metrics["thd_before"] = thd_before
        metrics["thd_after"] = thd_after

        thd_increase = thd_after - thd_before
        if thd_increase > self.max_thd_increase:
            issues.append(f"THD increased too much: {thd_increase:+.1%} (max {self.max_thd_increase:.1%})")

        # 4. Check crest factor improvement
        crest_before = compute_crest_factor(original)
        crest_after = compute_crest_factor(processed)

        metrics["crest_factor_before"] = crest_before
        metrics["crest_factor_after"] = crest_after

        crest_improvement = crest_after - crest_before
        metrics["crest_improvement_db"] = float(crest_improvement)

        if crest_improvement < 0:
            side_effects.append(f"Crest factor decreased: {crest_improvement:+.1f} dB. Dynamics not improved.")
        elif crest_improvement < 2:
            side_effects.append(f"Minimal crest factor improvement: {crest_improvement:+.1f} dB")

        # 5. Energy preservation
        energy_ratio = compute_energy_ratio(original, processed)
        metrics["energy_ratio"] = energy_ratio

        if energy_ratio < 0.8 or energy_ratio > 1.2:
            side_effects.append(f"Unexpected energy change: {energy_ratio:.2%}")

        # 6. Spectral centroid (timbre check)
        centroid_before = compute_spectral_centroid(original.flatten(), sr)
        centroid_after = compute_spectral_centroid(processed.flatten(), sr)

        metrics["spectral_centroid_before"] = centroid_before
        metrics["spectral_centroid_after"] = centroid_after

        centroid_ratio = centroid_after / centroid_before
        if centroid_ratio > 1.3 or centroid_ratio < 0.7:
            side_effects.append(f"Significant timbre change: {centroid_ratio:.2f}x brightness")

        # 7. Check for new artifacts (high-frequency noise)
        # High-pass filter at 10 kHz and check energy
        sos = signal.butter(4, 10000, "hp", fs=sr, output="sos")
        hf_orig = signal.sosfilt(sos, original.flatten())
        hf_proc = signal.sosfilt(sos, processed.flatten())

        hf_energy_orig = np.mean(hf_orig**2)
        hf_energy_proc = np.mean(hf_proc**2)

        if hf_energy_orig > 0:
            hf_ratio = hf_energy_proc / hf_energy_orig
            metrics["high_freq_energy_ratio"] = float(hf_ratio)

            if hf_ratio > 2.0:
                issues.append(f"Excessive high-frequency energy added: {hf_ratio:.1f}x. Possible artifacts.")

        passed = len(issues) == 0

        return PostCheckResult(
            passed=passed,
            quality_score=0.0,  # Will be computed separately
            issues=issues,
            side_effects=side_effects,
            metrics=metrics,
        )

    def _compute_quality_score(
        self, original: np.ndarray, processed: np.ndarray, sr: int, post_check: PostCheckResult
    ) -> float:
        """Berechnet overall quality score."""
        scores = []

        # Clipping reduction effectiveness
        sev_before = post_check.metrics.get("severity_before", 1.0)
        sev_after = post_check.metrics.get("severity_after", 1.0)

        if sev_before > 0:
            reduction_score = (sev_before - sev_after) / sev_before
            scores.append(np.clip(reduction_score, 0.0, 1.0))

        # Harmonic preservation
        harm_before = post_check.metrics.get("harmonic_strength_before", 1.0)
        harm_after = post_check.metrics.get("harmonic_strength_after", 1.0)

        if harm_before > 0:
            harm_preservation = harm_after / harm_before
            scores.append(np.clip(harm_preservation, 0.0, 1.0))

        # THD improvement (lower is better)
        thd_before = post_check.metrics.get("thd_before", 0.5)
        thd_after = post_check.metrics.get("thd_after", 0.5)

        if thd_before > 0:
            thd_improvement = (thd_before - thd_after) / thd_before
            scores.append(np.clip(thd_improvement, 0.0, 1.0))

        # Crest factor improvement
        crest_improvement = post_check.metrics.get("crest_improvement_db", 0.0)
        # Normalize: 6 dB improvement = 1.0 score
        crest_score = np.clip(crest_improvement / 6.0, 0.0, 1.0)
        scores.append(crest_score)

        # Weighted average (emphasize clipping reduction and THD)
        weights = [0.35, 0.25, 0.25, 0.15]  # reduction, harmonics, THD, crest
        quality = np.average(scores, weights=weights)

        return float(np.clip(quality, 0.0, 1.0))
