"""
stereo_widener_safety.py - HIPS-Compliant Stereo Widener Safety Wrapper

Wraps stereo widening DSP operations with comprehensive safety checks:
- Mono compatibility validation (no phase cancellation)
- Phase coherence monitoring
- Center content preservation (vocals, bass)
- Spatial balance verification
- No hollow/phasey artifacts

This ensures stereo widening enhances spaciousness without compromising mix integrity.

Author: AURIK Team
Version: 1.0.0
Date: 7. Februar 2026
Phase: 1 Week 5-6
"""

from pathlib import Path

import numpy as np
import scipy.signal as signal

from backend.core.audio_layout import to_channels_first

from .safety_wrapper_template import (
    BaseSafetyWrapper,
    PostCheckResult,
    PreCheckResult,
    compute_energy_ratio,
    validate_audio_basic,
)

# ============================================================================
# STEREO ANALYSIS UTILITIES
# ============================================================================


def check_mono_compatibility(audio: np.ndarray) -> tuple[float, float]:
    """
    Prüft mono compatibility (phase cancellation test).

    When summed to mono, well-produced stereo retains >90% energy.
    Poor phase relationships cause cancellation.

    Args:
        audio: Stereo audio [2, samples]

    Returns:
        (compatibility_score, energy_loss_db)
    """
    # §V7 (copilot-instructions.md): Layout-Normalisierung — shape[0]!=2 ließ
    # channels-last (N,2) fälschlich als Mono durchrutschen.
    _st = to_channels_first(audio)
    if audio.ndim == 1 or _st.ndim != 2 or _st.shape[0] != 2:
        return 1.0, 0.0  # Mono is always compatible

    left = _st[0]
    right = _st[1]

    # Stereo energy
    stereo_energy = np.mean(left**2) + np.mean(right**2)

    # Mono sum
    mono = (left + right) / 2
    mono_energy = np.mean(mono**2)

    # Expected mono energy (if no cancellation)
    expected_mono_energy = stereo_energy / 2

    if expected_mono_energy == 0:
        return 1.0, 0.0

    # Compatibility ratio
    compatibility = mono_energy / expected_mono_energy

    # Energy loss in dB
    energy_loss_db = -10 * np.log10(compatibility + 1e-10)

    compatibility_score = np.clip(compatibility, 0.0, 1.0)

    return float(compatibility_score), float(energy_loss_db)


def measure_stereo_width(audio: np.ndarray) -> float:
    """
    Misst stereo width.

    0.0 = Mono (L == R)
    1.0 = Normal stereo
    >1.0 = Enhanced width

    Args:
        audio: Stereo audio [2, samples]

    Returns:
        Width measure (0.0-2.0)
    """
    _st = to_channels_first(audio)
    if audio.ndim == 1 or _st.ndim != 2 or _st.shape[0] != 2:
        return 0.0  # Mono

    left = _st[0]
    right = _st[1]

    # Mid/Side decomposition
    mid = (left + right) / 2
    side = (left - right) / 2

    mid_energy = np.mean(mid**2)
    side_energy = np.mean(side**2)

    if mid_energy == 0:
        return 2.0  # Maximum width (no center)

    width = side_energy / mid_energy

    # Normalize: 1.0 = equal mid/side (normal stereo)
    return float(np.clip(width, 0.0, 2.0))


def detect_center_content(audio: np.ndarray, sr: int) -> dict[str, float]:
    """
    Erkennt important center content (vocals, bass, kick).

    These should remain in center for mix integrity.

    Args:
        audio: Stereo audio
        sr: Sample rate

    Returns:
        Dict with center content analysis
    """
    _st = to_channels_first(audio)
    if audio.ndim == 1 or _st.ndim != 2 or _st.shape[0] != 2:
        return {"has_center_content": False, "center_energy": 0.0}

    left = _st[0]
    right = _st[1]

    # Mid signal (center content)
    mid = (left + right) / 2

    # Analyze mid signal

    # 1. Low-frequency energy (bass/kick)
    sos_low = signal.butter(4, 150, "lp", fs=sr, output="sos")
    mid_low = signal.sosfilt(sos_low, mid)
    low_energy = np.mean(mid_low**2)

    # 2. Vocal range energy (200-3000 Hz)
    sos_vocal = signal.butter(4, [200, 3000], "bp", fs=sr, output="sos")
    mid_vocal = signal.sosfilt(sos_vocal, mid)
    vocal_energy = np.mean(mid_vocal**2)

    # 3. Total mid energy
    mid_energy = np.mean(mid**2)
    total_energy = (np.mean(left**2) + np.mean(right**2)) / 2

    center_ratio = mid_energy / (total_energy + 1e-10)

    has_center_content = center_ratio > 0.3

    return {
        "has_center_content": has_center_content,
        "center_energy_ratio": float(center_ratio),
        "low_freq_energy": float(low_energy),
        "vocal_range_energy": float(vocal_energy),
    }


def measure_phase_correlation(audio: np.ndarray) -> float:
    """
    Misst phase correlation between L and R.

    +1.0 = Mono (perfect correlation)
    0.0 = Decorrelated (wide stereo)
    -1.0 = Out of phase (cancels in mono)

    Args:
        audio: Stereo audio [2, samples]

    Returns:
        Phase correlation (-1.0 to +1.0)
    """
    _st = to_channels_first(audio)
    if audio.ndim == 1 or _st.ndim != 2 or _st.shape[0] != 2:
        return 1.0  # Mono

    left = _st[0]
    right = _st[1]

    # Normalize
    left_norm = left / (np.sqrt(np.mean(left**2)) + 1e-10)
    right_norm = right / (np.sqrt(np.mean(right**2)) + 1e-10)

    # Correlation
    correlation = np.mean(left_norm * right_norm)

    return float(np.clip(correlation, -1.0, 1.0))


def detect_hollow_artifacts(audio: np.ndarray, sr: int) -> tuple[bool, float]:
    """
    Erkennt "hollow" or "phasey" artifacts from excessive widening.

    Characteristics:
    - Mid-frequency scooping
    - Comb filtering in mid signal

    Args:
        audio: Stereo audio
        sr: Sample rate

    Returns:
        (has_artifacts, severity)
    """
    _st = to_channels_first(audio)
    if audio.ndim == 1 or _st.ndim != 2 or _st.shape[0] != 2:
        return False, 0.0

    left = _st[0]
    right = _st[1]

    # Mid and Side
    mid = (left + right) / 2
    side = (left - right) / 2

    # Frequency analysis
    mid_spectrum = np.abs(np.fft.rfft(mid))
    side_spectrum = np.abs(np.fft.rfft(side))
    freqs = np.fft.rfftfreq(len(mid), 1 / sr)

    # Check mid-frequency range (500-2000 Hz)
    mid_band = (freqs >= 500) & (freqs <= 2000)

    if not np.any(mid_band):
        return False, 0.0

    mid_energy_in_band = np.mean(mid_spectrum[mid_band] ** 2)
    side_energy_in_band = np.mean(side_spectrum[mid_band] ** 2)

    # Hollow sound: mid scooped, side dominant
    ratio = side_energy_in_band / mid_energy_in_band if mid_energy_in_band > 0 else 10.0

    # High ratio = hollowness
    has_artifacts = ratio > 3.0
    severity = min((ratio - 1.0) / 5.0, 1.0)

    return has_artifacts, float(severity)


def measure_spatial_balance(audio: np.ndarray) -> float:
    """
    Misst left-right balance.

    0.5 = Perfect balance
    <0.5 = Left-biased
    >0.5 = Right-biased

    Args:
        audio: Stereo audio [2, samples]

    Returns:
        Balance (0.0-1.0)
    """
    _st = to_channels_first(audio)
    if audio.ndim == 1 or _st.ndim != 2 or _st.shape[0] != 2:
        return 0.5  # Mono = balanced

    left_energy = np.mean(_st[0] ** 2)
    right_energy = np.mean(_st[1] ** 2)

    total_energy = left_energy + right_energy

    if total_energy == 0:
        return 0.5

    balance = left_energy / total_energy

    return float(balance)


# ============================================================================
# STEREO WIDENER SAFETY WRAPPER
# ============================================================================


class StereoWidenerSafety(BaseSafetyWrapper):
    """
    HIPS-compliant safety wrapper for stereo widening.

    Ensures:
    - Mono compatibility (no phase cancellation)
    - Phase coherence maintained
    - Center content preserved (vocals, bass)
    - Spatial balance maintained
    - No hollow/phasey artifacts
    """

    def __init__(
        self,
        processor_func,
        enable_logging: bool = True,
        log_dir: Path | None = None,
        min_mono_compatibility: float = 0.8,
        max_width_increase: float = 0.5,
    ):
        """
        Initialisiert Stereo Widener Safety Wrapper.

        Args:
            processor_func: Widener function (audio, sr, width_amount) -> audio
            enable_logging: Enable audit trail
            log_dir: Audit log directory
            min_mono_compatibility: Minimum mono compatibility score
            max_width_increase: Maximum width increase (relative)
        """
        super().__init__(
            module_name="StereoWidener",
            module_version="1.0.0",
            processor_func=processor_func,
            enable_logging=enable_logging,
            log_dir=log_dir,
            confidence_threshold=0.5,
            quality_threshold=0.7,
        )

        self.min_mono_compatibility = min_mono_compatibility
        self.max_width_increase = max_width_increase

    def _validate_pre_conditions(self, audio: np.ndarray, sr: int, **params) -> PreCheckResult:
        """Validiert pre-conditions for stereo widening."""
        # Basic audio validation
        is_valid, errors = validate_audio_basic(audio)

        if not is_valid:
            return PreCheckResult(passed=False, confidence=0.0, reasons=errors)

        warnings = []
        metadata = {}

        # Check if stereo
        _st = to_channels_first(audio)
        if audio.ndim == 1 or _st.ndim != 2 or _st.shape[0] != 2:
            return PreCheckResult(
                passed=False, confidence=0.0, reasons=["Audio is not stereo. Stereo widening requires stereo input."]
            )

        # Check initial mono compatibility
        compatibility, loss_db = check_mono_compatibility(audio)
        metadata["mono_compatibility_before"] = compatibility
        metadata["mono_loss_db_before"] = loss_db

        if compatibility < 0.7:
            warnings.append(f"Poor initial mono compatibility: {compatibility:.2f}. Mono sum loses {loss_db:.1f} dB.")

        # Measure initial width
        width_before = measure_stereo_width(audio)
        metadata["width_before"] = width_before

        if width_before > 1.5:
            warnings.append(f"Already very wide: {width_before:.2f}. Further widening may cause artifacts.")

        # Detect center content
        center_info = detect_center_content(audio, sr)
        metadata.update(center_info)

        if center_info["has_center_content"]:
            warnings.append(
                f"Important center content detected (ratio: {center_info['center_energy_ratio']:.2f}). "
                "Must be preserved during widening."
            )

        # Phase correlation
        phase_corr = measure_phase_correlation(audio)
        metadata["phase_correlation_before"] = phase_corr

        if phase_corr < 0:
            warnings.append(
                f"Negative phase correlation: {phase_corr:.2f}. Signals are out of phase - widening may worsen this."
            )

        # Spatial balance
        balance = measure_spatial_balance(audio)
        metadata["spatial_balance_before"] = balance

        # Validate parameters
        width_amount = params.get("width_amount", 0.5)

        if width_amount > 0.8:
            warnings.append(f"Very aggressive widening: {width_amount:.2f}. High risk of mono incompatibility.")

        return PreCheckResult(passed=True, confidence=compatibility, warnings=warnings, metadata=metadata)

    def _assess_epistemic_confidence(self, audio: np.ndarray, sr: int, pre_check: PreCheckResult, **params) -> float:
        """Assess confidence in stereo widening for this audio."""
        # Base confidence from mono compatibility
        compatibility = pre_check.metadata.get("mono_compatibility_before", 0.8)

        # Penalty for already wide stereo
        width_before = pre_check.metadata.get("width_before", 1.0)
        width_penalty = max(0, (width_before - 1.3) / 0.7) * 0.3

        # Penalty for important center content
        center_ratio = pre_check.metadata.get("center_energy_ratio", 0.5)
        center_penalty = center_ratio * 0.2  # Up to 20% penalty

        # Penalty for aggressive processing
        width_amount = params.get("width_amount", 0.5)
        amount_penalty = max(0, (width_amount - 0.7) / 0.3) * 0.2

        confidence = compatibility * (1.0 - width_penalty - center_penalty - amount_penalty)

        return float(np.clip(confidence, 0.0, 1.0))

    def _validate_post_conditions(
        self, original: np.ndarray, processed: np.ndarray, sr: int, **params
    ) -> PostCheckResult:
        """Validiert post-conditions after stereo widening."""
        issues = []
        side_effects = []
        metrics = {}

        # Ensure same shape
        if original.shape != processed.shape:
            issues.append(f"Shape mismatch: {original.shape} -> {processed.shape}")
            return PostCheckResult(passed=False, quality_score=0.0, issues=issues)

        # 1. Check mono compatibility
        compatibility_after, loss_db_after = check_mono_compatibility(processed)
        compatibility_before = check_mono_compatibility(original)[0]

        metrics["mono_compatibility_before"] = compatibility_before
        metrics["mono_compatibility_after"] = compatibility_after
        metrics["mono_loss_db_after"] = loss_db_after

        if compatibility_after < self.min_mono_compatibility:
            issues.append(
                f"Poor mono compatibility: {compatibility_after:.2f} "
                f"(min {self.min_mono_compatibility}). "
                f"Mono sum loses {loss_db_after:.1f} dB."
            )

        compatibility_degradation = compatibility_before - compatibility_after
        if compatibility_degradation > 0.15:
            issues.append(f"Mono compatibility degraded: {compatibility_before:.2f} -> {compatibility_after:.2f}")

        # 2. Check width increase
        width_before = measure_stereo_width(original)
        width_after = measure_stereo_width(processed)

        metrics["width_before"] = width_before
        metrics["width_after"] = width_after
        metrics["width_increase"] = float(width_after - width_before)

        width_increase_ratio = (width_after - width_before) / (width_before + 0.1)

        if width_increase_ratio > self.max_width_increase:
            side_effects.append(f"Very large width increase: {width_increase_ratio:.1%}")

        # 3. Center content preservation
        center_before = detect_center_content(original, sr)
        center_after = detect_center_content(processed, sr)

        if center_before["has_center_content"]:
            center_loss = center_before["center_energy_ratio"] - center_after["center_energy_ratio"]

            metrics["center_energy_loss"] = float(center_loss)

            if center_loss > 0.2:
                issues.append(f"Excessive center content loss: {center_loss:.2f}")

        # 4. Phase correlation
        phase_corr_before = measure_phase_correlation(original)
        phase_corr_after = measure_phase_correlation(processed)

        metrics["phase_correlation_before"] = phase_corr_before
        metrics["phase_correlation_after"] = phase_corr_after

        if phase_corr_after < -0.3:
            issues.append(f"Severe phase issues: correlation {phase_corr_after:.2f}")

        # 5. Hollow artifacts
        has_hollow_after, hollow_severity = detect_hollow_artifacts(processed, sr)
        has_hollow_before = detect_hollow_artifacts(original, sr)[0]

        metrics["hollow_artifacts"] = hollow_severity

        if has_hollow_after and not has_hollow_before:
            issues.append(f"Hollow/phasey artifacts introduced: severity {hollow_severity:.2f}")

        # 6. Spatial balance
        balance_before = measure_spatial_balance(original)
        balance_after = measure_spatial_balance(processed)

        metrics["spatial_balance_before"] = balance_before
        metrics["spatial_balance_after"] = balance_after

        balance_shift = abs(balance_after - balance_before)
        if balance_shift > 0.1:
            side_effects.append(f"Spatial balance shifted: {balance_before:.2f} -> {balance_after:.2f}")

        # 7. Energy preservation
        energy_ratio = compute_energy_ratio(original, processed)
        metrics["energy_ratio"] = energy_ratio

        if energy_ratio < 0.9 or energy_ratio > 1.1:
            side_effects.append(f"Energy change: {energy_ratio:.2%}")

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

        # Mono compatibility
        compatibility = post_check.metrics.get("mono_compatibility_after", 0.8)
        scores.append(compatibility)

        # Appropriate width increase (not too much, not too little)
        width_increase = post_check.metrics.get("width_increase", 0.0)
        # Ideal: 0.2-0.4
        if width_increase < 0.1:
            width_score = width_increase / 0.1  # Too subtle
        elif width_increase > 0.5:
            width_score = 1.0 - (width_increase - 0.5) / 0.5  # Too much
        else:
            width_score = 1.0  # Good range
        scores.append(np.clip(width_score, 0.0, 1.0))

        # No hollow artifacts
        hollow_severity = post_check.metrics.get("hollow_artifacts", 0.0)
        hollow_score = 1.0 - hollow_severity
        scores.append(np.clip(hollow_score, 0.0, 1.0))

        # Center content preservation
        center_loss = post_check.metrics.get("center_energy_loss", 0.0)
        center_score = 1.0 - (center_loss / 0.3)  # 30% loss = 0 score
        scores.append(np.clip(center_score, 0.0, 1.0))

        # Weighted average
        weights = [0.35, 0.25, 0.2, 0.2]  # mono, width, hollow, center
        quality = np.average(scores, weights=weights)

        return float(np.clip(quality, 0.0, 1.0))
