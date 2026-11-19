"""
harmonic_exciter_safety.py - HIPS-Compliant Harmonic Exciter Safety Wrapper

Wraps harmonic exciter DSP operations with comprehensive safety checks:
- Harmonic headroom validation (no clipping from added harmonics)
- Musical enhancement vs harshness classification
- Even/odd harmonic balance
- Spectral brightness control
- No intermodulation distortion

This ensures harmonic excitement enhances warmth/presence without harshness.

Author: AURIK Team
Version: 1.0.0
Date: 7. Februar 2026
Phase: 1 Week 5-6
"""

from pathlib import Path

import numpy as np

from .safety_wrapper_template import (
    BaseSafetyWrapper,
    PostCheckResult,
    PreCheckResult,
    compute_correlation,
    compute_energy_ratio,
    compute_spectral_centroid,
    validate_audio_basic,
)

# ============================================================================
# HARMONIC ANALYSIS UTILITIES
# ============================================================================


def measure_harmonic_headroom(audio: np.ndarray) -> float:
    """
    Misst available headroom for harmonic generation.

    Harmonic exciters add energy - need headroom to avoid clipping.
    Uses 99.9th percentile instead of absolute max so that isolated
    impulse artefacts (clicks, crackles) do not artificially reduce the
    measured headroom and prevent legitimate harmonic enhancement
    (§DSP-Invariante Peak-Guards).

    Args:
        audio: Input audio

    Returns:
        Headroom in dB
    """
    # §DSP-Invariante: percentile(99.9) — impulse artefacts must not block gain
    peak = float(np.percentile(np.abs(audio), 99.9)) if audio.size > 0 else 0.0

    if peak == 0:
        return 60.0  # Maximum headroom

    headroom_db = 20 * np.log10(1.0 / peak)

    return float(headroom_db)


def classify_harmonic_content(audio: np.ndarray, sr: int) -> dict[str, float]:
    """
    Classify existing harmonic content.

    Even harmonics: Warmth, tube-like (2nd, 4th, 6th...)
    Odd harmonics: Presence, bite (3rd, 5th, 7th...)

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        Dict with even/odd harmonic ratios
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Find fundamental frequency — FFT-based O(N log N)
    from backend.core.core_utils import fft_autocorr

    autocorr = fft_autocorr(audio)
    autocorr = autocorr / (autocorr[0] + 1e-10)

    # Pitch detection
    min_lag = int(sr / 500)
    max_lag = int(sr / 50)

    if max_lag >= len(autocorr):
        return {"even_ratio": 0.5, "odd_ratio": 0.5, "fundamental_hz": 0}

    search_region = autocorr[min_lag:max_lag]

    if len(search_region) == 0:
        return {"even_ratio": 0.5, "odd_ratio": 0.5, "fundamental_hz": 0}

    peak_idx = np.argmax(search_region) + min_lag
    fundamental_hz = sr / peak_idx

    # Analyze harmonic spectrum
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)

    even_energy = 0.0
    odd_energy = 0.0

    for n in range(2, 11):  # Harmonics 2-10
        harmonic_freq = fundamental_hz * n

        if harmonic_freq > sr / 2:
            break

        idx = np.argmin(np.abs(freqs - harmonic_freq))
        energy = spectrum[idx] ** 2

        if n % 2 == 0:
            even_energy += energy
        else:
            odd_energy += energy

    total_harmonic_energy = even_energy + odd_energy

    if total_harmonic_energy > 0:
        even_ratio = even_energy / total_harmonic_energy
        odd_ratio = odd_energy / total_harmonic_energy
    else:
        even_ratio = 0.5
        odd_ratio = 0.5

    return {"even_ratio": float(even_ratio), "odd_ratio": float(odd_ratio), "fundamental_hz": float(fundamental_hz)}


def detect_harshness(audio: np.ndarray, sr: int) -> tuple[bool, float]:
    """
    Erkennt harshness (excessive high-mid energy).

    Harshness frequency range: 2-6 kHz

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        (is_harsh, severity)
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Compute spectrum
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)

    # Harshness band (2-6 kHz)
    harsh_band = (freqs >= 2000) & (freqs <= 6000)

    # Reference bands
    mid_band = (freqs >= 500) & (freqs < 2000)
    high_band = (freqs > 6000) & (freqs <= 12000)

    harsh_energy = np.mean(spectrum[harsh_band] ** 2) if np.any(harsh_band) else 0
    mid_energy = np.mean(spectrum[mid_band] ** 2) if np.any(mid_band) else 0
    high_energy = np.mean(spectrum[high_band] ** 2) if np.any(high_band) else 0

    reference_energy = (mid_energy + high_energy) / 2

    if reference_energy == 0:
        return False, 0.0

    harshness_ratio = harsh_energy / reference_energy

    # Harsh if 2-6 kHz is > 1.5x reference
    is_harsh = harshness_ratio > 1.5
    severity = min((harshness_ratio - 1.0) / 2.0, 1.0)

    return is_harsh, float(severity)


def measure_brightness(audio: np.ndarray, sr: int) -> float:
    """
    Misst spectral brightness (high-frequency content).

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        Brightness score (0.0-1.0)
    """
    centroid = compute_spectral_centroid(audio.flatten(), sr)

    # Normalize centroid to 0-1 range
    # Typical range: 1000-8000 Hz
    brightness = (centroid - 1000) / 7000
    brightness = np.clip(brightness, 0.0, 1.0)

    return float(brightness)


def detect_intermodulation_distortion(audio: np.ndarray, sr: int) -> float:
    """
    Erkennt intermodulation distortion (IMD).

    IMD creates sum/difference frequencies not in original signal.

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        IMD estimate (0.0-1.0)
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Compute spectrum
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)

    # Find peaks (fundamental + harmonics)
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(spectrum, prominence=np.max(spectrum) * 0.1)

    if len(peaks) < 2:
        return 0.0

    peak_freqs = freqs[peaks]
    peak_energies = spectrum[peaks]

    # Expected harmonics from fundamental
    if len(peak_freqs) > 0:
        fundamental = peak_freqs[0]
        expected_harmonics = {fundamental * n for n in range(1, 11)}

        # Find unexpected peaks (IMD products)
        imd_energy = 0.0
        total_peak_energy = np.sum(peak_energies**2)

        for freq, energy in zip(peak_freqs, peak_energies):
            # Check if this is an expected harmonic
            is_harmonic = any(abs(freq - h) < 20 for h in expected_harmonics)

            if not is_harmonic:
                imd_energy += energy**2

        imd_ratio = imd_energy / total_peak_energy if total_peak_energy > 0 else 0.0
    else:
        imd_ratio = 0.0

    return float(np.clip(imd_ratio, 0.0, 1.0))


# ============================================================================
# HARMONIC EXCITER SAFETY WRAPPER
# ============================================================================


class HarmonicExciterSafety(BaseSafetyWrapper):
    """
    HIPS-compliant safety wrapper for harmonic exciter.

    Ensures:
    - Sufficient headroom (no clipping from added harmonics)
    - Musical enhancement (warmth/presence) not harshness
    - Appropriate even/odd harmonic balance
    - No excessive brightness
    - No intermodulation distortion introduced
    """

    def __init__(
        self,
        processor_func,
        enable_logging: bool = True,
        log_dir: Path | None = None,
        min_headroom_db: float = 6.0,
        max_harshness: float = 0.6,
    ):
        """
        Initialisiert Harmonic Exciter Safety Wrapper.

        Args:
            processor_func: Exciter function (audio, sr, amount, harmonic_mode) -> audio
            enable_logging: Enable audit trail
            log_dir: Audit log directory
            min_headroom_db: Minimum headroom required
            max_harshness: Maximum acceptable harshness
        """
        super().__init__(
            module_name="HarmonicExciter",
            module_version="1.0.0",
            processor_func=processor_func,
            enable_logging=enable_logging,
            log_dir=log_dir,
            confidence_threshold=0.5,
            quality_threshold=0.7,
        )

        self.min_headroom_db = min_headroom_db
        self.max_harshness = max_harshness

    def _validate_pre_conditions(self, audio: np.ndarray, sr: int, **params) -> PreCheckResult:
        """Validiert pre-conditions for harmonic excitement."""
        # Basic audio validation
        is_valid, errors = validate_audio_basic(audio)

        if not is_valid:
            return PreCheckResult(passed=False, confidence=0.0, reasons=errors)

        warnings = []
        metadata = {}

        # Check headroom
        headroom = measure_harmonic_headroom(audio)
        metadata["headroom_db"] = headroom

        if headroom < self.min_headroom_db:
            return PreCheckResult(
                passed=False,
                confidence=0.0,
                reasons=[
                    f"Insufficient headroom: {headroom:.1f} dB "
                    f"(min {self.min_headroom_db} dB). "
                    "Risk of clipping from added harmonics."
                ],
            )

        # Classify existing harmonics
        harmonic_content = classify_harmonic_content(audio, sr)
        metadata.update(harmonic_content)

        # Check for existing harshness
        is_harsh, harshness_severity = detect_harshness(audio, sr)
        metadata["harshness_before"] = harshness_severity

        if is_harsh:
            warnings.append(
                f"Audio already harsh: severity {harshness_severity:.2f}. Harmonic excitement may increase harshness."
            )

        # Measure brightness
        brightness = measure_brightness(audio, sr)
        metadata["brightness_before"] = brightness

        if brightness > 0.7:
            warnings.append(
                f"Already very bright: {brightness:.2f}. Additional harmonics may create excessive sibilance."
            )

        # Check IMD
        imd_before = detect_intermodulation_distortion(audio, sr)
        metadata["imd_before"] = imd_before

        # Validate parameters
        amount = params.get("amount", 0.5)

        if amount > 0.8:
            warnings.append(f"Very high exciter amount: {amount:.2f}. Risk of harsh, unnatural sound.")

        return PreCheckResult(
            passed=True,
            confidence=min(headroom / 12.0, 1.0),  # More headroom = higher confidence
            warnings=warnings,
            metadata=metadata,
        )

    def _assess_epistemic_confidence(self, audio: np.ndarray, sr: int, pre_check: PreCheckResult, **params) -> float:
        """Assess confidence in harmonic excitement for this audio."""
        # Base confidence from headroom
        headroom = pre_check.metadata.get("headroom_db", 6.0)
        headroom_factor = min(headroom / 12.0, 1.0)

        # Penalty for existing harshness
        harshness = pre_check.metadata.get("harshness_before", 0.0)
        harshness_penalty = harshness * 0.3

        # Penalty for high brightness (risk of over-brightening)
        brightness = pre_check.metadata.get("brightness_before", 0.5)
        brightness_penalty = max(0, (brightness - 0.6) / 0.4) * 0.2

        # Penalty for aggressive processing
        amount = params.get("amount", 0.5)
        amount_penalty = max(0, (amount - 0.7) / 0.3) * 0.2

        confidence = headroom_factor * (1.0 - harshness_penalty - brightness_penalty - amount_penalty)

        return float(np.clip(confidence, 0.0, 1.0))

    def _validate_post_conditions(
        self, original: np.ndarray, processed: np.ndarray, sr: int, **params
    ) -> PostCheckResult:
        """Validiert post-conditions after harmonic excitement."""
        issues = []
        side_effects = []
        metrics = {}

        # Ensure same shape
        if original.shape != processed.shape:
            issues.append(f"Shape mismatch: {original.shape} -> {processed.shape}")
            return PostCheckResult(passed=False, quality_score=0.0, issues=issues)

        # 1. Check for clipping
        peak_before = np.max(np.abs(original))
        peak_after = np.max(np.abs(processed))

        metrics["peak_before"] = float(peak_before)
        metrics["peak_after"] = float(peak_after)

        if peak_after > 0.99:
            issues.append(f"Output clipping detected: peak {peak_after:.3f}")

        # 2. Check headroom used
        headroom_before = measure_harmonic_headroom(original)
        headroom_after = measure_harmonic_headroom(processed)

        metrics["headroom_before"] = headroom_before
        metrics["headroom_after"] = headroom_after
        metrics["headroom_used_db"] = float(headroom_before - headroom_after)

        # 3. Check for harshness
        _is_harsh_after, harshness_after = detect_harshness(processed, sr)
        harshness_before = detect_harshness(original, sr)[1]

        metrics["harshness_before"] = harshness_before
        metrics["harshness_after"] = harshness_after

        if harshness_after > self.max_harshness:
            issues.append(f"Excessive harshness: {harshness_after:.2f} (max {self.max_harshness})")

        harshness_increase = harshness_after - harshness_before
        if harshness_increase > 0.3:
            issues.append(f"Harshness increased too much: +{harshness_increase:.2f}")

        # 4. Brightness change
        brightness_before = measure_brightness(original, sr)
        brightness_after = measure_brightness(processed, sr)

        metrics["brightness_before"] = brightness_before
        metrics["brightness_after"] = brightness_after
        metrics["brightness_increase"] = float(brightness_after - brightness_before)

        if brightness_after > 0.85:
            side_effects.append(f"Very bright output: {brightness_after:.2f}. May sound harsh.")

        # 5. Harmonic balance
        harmonic_content_after = classify_harmonic_content(processed, sr)
        metrics["even_ratio_after"] = harmonic_content_after["even_ratio"]
        metrics["odd_ratio_after"] = harmonic_content_after["odd_ratio"]

        # 6. IMD check
        imd_before = detect_intermodulation_distortion(original, sr)
        imd_after = detect_intermodulation_distortion(processed, sr)

        metrics["imd_before"] = imd_before
        metrics["imd_after"] = imd_after

        if imd_after > imd_before + 0.15:
            issues.append(f"Excessive IMD introduced: {imd_before:.2f} -> {imd_after:.2f}")

        # 7. Energy ratio
        energy_ratio = compute_energy_ratio(original, processed)
        metrics["energy_ratio"] = energy_ratio

        if energy_ratio > 1.5:
            side_effects.append(f"Significant energy increase: {energy_ratio:.2f}x")

        # 8. Correlation
        correlation = compute_correlation(original.flatten(), processed.flatten())
        metrics["correlation"] = correlation

        if correlation < 0.85:
            side_effects.append(f"Low correlation: {correlation:.3f}. Significant processing.")

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

        # No clipping
        peak_after = post_check.metrics.get("peak_after", 1.0)
        clipping_score = 1.0 if peak_after < 0.99 else 0.0
        scores.append(clipping_score)

        # Harshness control
        harshness_after = post_check.metrics.get("harshness_after", 0.0)
        harshness_score = 1.0 - (harshness_after / self.max_harshness)
        scores.append(np.clip(harshness_score, 0.0, 1.0))

        # IMD control
        imd_after = post_check.metrics.get("imd_after", 0.0)
        imd_score = 1.0 - imd_after
        scores.append(np.clip(imd_score, 0.0, 1.0))

        # Appropriate brightness increase
        brightness_increase = post_check.metrics.get("brightness_increase", 0.0)
        # Ideal increase: 0.1-0.3
        if brightness_increase < 0.05:
            brightness_score = brightness_increase / 0.05  # Too subtle
        elif brightness_increase > 0.4:
            brightness_score = 1.0 - (brightness_increase - 0.4) / 0.4  # Too much
        else:
            brightness_score = 1.0  # Good range
        scores.append(np.clip(brightness_score, 0.0, 1.0))

        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # clipping, harshness, IMD, brightness
        quality = np.average(scores, weights=weights)

        return float(np.clip(quality, 0.0, 1.0))
