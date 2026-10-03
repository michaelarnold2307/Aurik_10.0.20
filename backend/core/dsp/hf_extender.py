"""
High-Frequency Extension
========================

Neural HF-Extension via FlashSR für "Frische" und "Air".

Problem: Alte Aufnahmen haben oft gedämpfte Höhen:
- Tape degradation (HF rolloff)
- Low-quality ADC (Nyquist < 20kHz)
- Resampling artifacts

Lösung: FlashSR (bereits vorhanden!) kann:
- 16kHz → 48kHz upsample (echte HF-Extension)
- Harmonics intelligent re-generate
- Natürliche Höhen ohne Artefakte

Safeguards:
- Parallel Processing (max 30-50% blend)
- Genre-Adaptive Strength (Classical: sanft, Rock: mehr)
- Authenticity Metrics Check
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class HighFrequencyExtender:
    """
    Neural HF-Extension via FlashSR.

    Nutzt SOTA Super-Resolution Model um High-Frequencies zu regenerieren.
    """

    def __init__(self):
        self.name = "HF Extension (FlashSR)"
        self.flashsr_plugin = None

    def _lazy_load_flashsr(self) -> None:
        """Lazy-load FlashSR Plugin (erst bei Verwendung)."""
        if self.flashsr_plugin is None:
            from plugins.flashsr_plugin import FlashSRPlugin

            self.flashsr_plugin = FlashSRPlugin()
            logger.info("FlashSR-Plugin geladen")

    def should_extend(self, sr: int, spectral_rolloff: float | None = None, genre: str = "unknown") -> tuple[bool, str]:
        """
        Entscheidet ob HF-Extension sinnvoll ist.

        Args:
            sr: Sample rate
            spectral_rolloff: Spectral rolloff frequency (Hz, optional)
            genre: Genre

        Returns:
            (should_extend, reason)
        """
        # Regel 1: Sample Rate < 44.1kHz → Extension sinnvoll
        if sr < 44100:
            return True, f"Sample rate {sr}Hz < 44.1kHz (tape/low-quality ADC)"

        # Regel 2: Spectral Rolloff < 15kHz → HF fehlen
        if spectral_rolloff and spectral_rolloff < 15000:
            return (
                True,
                f"Spectral rolloff {spectral_rolloff / 1000:.1f}kHz < 15kHz (tape degradation)",
            )

        # Regel 3: Genre-spezifisch (Classical/Jazz profitieren)
        if genre in ["classical", "jazz", "acoustic"] and sr == 44100:  # CD-Quality, aber könnte besser sein
            return (
                True,
                f"Genre '{genre}' benefits from HF extension (air/brilliance)",
            )

        return False, "No HF extension needed (already high-quality)"

    def extend(
        self,
        audio: np.ndarray,
        sr_in: int,
        sr_target: int = 48000,
        strength: float = 0.3,
        genre: str = "unknown",
    ) -> np.ndarray:
        """
        Extend High-Frequencies neural via FlashSR.

        Args:
            audio: Input audio
            sr_in: Input sample rate
            sr_target: Target sample rate (48000 recommended)
            strength: Extension strength (0.0 = none, 0.3 = conservative, 0.5 = moderate)
            genre: Genre für adaptive Parameter

        Returns:
            HF-extended audio
        """
        assert sr_target == 48000, f"Target SR must be 48000 Hz, got {sr_target}"
        logger.info("HF Extension: %sHz → %sHz (strength=%.2f, genre=%s)", sr_in, sr_target, strength, genre)

        # Genre-adaptive Strength
        strength = self._adjust_strength_for_genre(strength, genre)

        # Lazy-load FlashSR
        self._lazy_load_flashsr()
        assert self.flashsr_plugin is not None, "FlashSR Plugin konnte nicht geladen werden"  # Type narrowing für mypy

        # Process via FlashSR (direkt ndarray — kein Temp-File)
        audio_extended = self.flashsr_plugin.process(audio, sr_in, target_sr=sr_target)

        # Parallel Mix mit Original (Authenticity Safeguard #1)
        # Resample original to match
        import librosa

        audio_original_resampled = librosa.resample(audio, orig_sr=sr_in, target_sr=sr_target)

        # Ensure same length
        min_len = min(len(audio_extended), len(audio_original_resampled))
        audio_extended = audio_extended[:min_len]
        audio_original_resampled = audio_original_resampled[:min_len]

        # Blend
        audio_blended = (1 - strength) * audio_original_resampled + strength * audio_extended

        # NaN/Inf-Guard + Clipping
        audio_blended = np.nan_to_num(audio_blended, nan=0.0, posinf=0.0, neginf=0.0)
        audio_blended = np.clip(audio_blended, -1.0, 1.0)

        logger.info("✓ HF-Extension abgeschlossen: %sHz → %sHz", sr_in, sr_target)

        return np.asarray(audio_blended)  # type: ignore[no-any-return]

    def _adjust_strength_for_genre(self, base_strength: float, genre: str) -> float:
        """
        Genre-adaptive Strength Limits (Safeguard #3).
        """
        genre_limits = {
            "classical": 0.25,  # Sanft (transparency)
            "jazz": 0.35,  # Moderat (air without harshness)
            "acoustic": 0.3,
            "speech": 0.45,  # Mehr OK (intelligibility)
            "rock": 0.4,
            "pop": 0.4,
        }

        max_strength = genre_limits.get(genre, 0.3)  # Default: 30%
        adjusted = min(base_strength, max_strength)

        if adjusted < base_strength:
            logger.info("HF Extension strength limited for genre '%s': %.2f → %.2f", genre, base_strength, adjusted)

        return adjusted

    def extend_with_authenticity_check(
        self,
        audio: np.ndarray,
        sr_in: int,
        sr_target: int = 48000,
        strength: float = 0.3,
        genre: str = "unknown",
    ) -> tuple[np.ndarray, bool, dict]:
        """
        HF-Extension MIT Authenticity Check.

        Returns:
            (audio_extended, is_authentic, metrics)
        """
        # Extend
        audio_extended = self.extend(audio, sr_in, sr_target, strength, genre)

        # Authenticity Check (Safeguard #4)
        # Resample original for comparison
        import librosa

        from backend.core.dsp.authenticity_metrics import AuthenticityMetrics

        audio_original_resampled = librosa.resample(audio, orig_sr=sr_in, target_sr=sr_target)

        # Ensure same length
        min_len = min(len(audio_extended), len(audio_original_resampled))
        audio_extended_check = audio_extended[:min_len]
        audio_original_check = audio_original_resampled[:min_len]

        is_authentic, warnings, metrics = AuthenticityMetrics.authenticity_check(
            audio_original_check, audio_extended_check, sr_target
        )

        if not is_authentic:
            logger.warning("HF Extension altered character:")
            for warning in warnings:
                logger.warning("  %s", warning)

        return audio_extended, is_authentic, metrics


# ============================================================================
# Integration in Policy-Engine
# ============================================================================


def select_hf_extension_strategy(context: dict, goal: dict) -> dict | None:
    """
    Policy-Engine Decision: Soll HF-Extension angewendet werden?

    Args:
        context: Audio context (sr, spectral_rolloff, genre, etc.)
        goal: Processing goals

    Returns:
        hf_params dict or None if not needed
    """
    extender = HighFrequencyExtender()

    # Check if extension needed
    should_extend, reason = extender.should_extend(
        sr=context.get("sample_rate", 48000),
        spectral_rolloff=context.get("spectral_rolloff"),
        genre=context.get("genre", "unknown"),
    )

    if not should_extend:
        logger.info("HF-Extension übersprungen: %s", reason)
        return None

    # Determine target SR based on context
    sr_current = context.get("sample_rate", 48000)
    sr_target = 48000  # Default

    if sr_current <= 16000:
        sr_target = 48000  # Aggressive upsample
    elif sr_current <= 22050 or sr_current <= 44100:
        sr_target = 48000

    # §GEBOT-G08: Adaptive Strength — aus tatsächlichem Bandbreiten-Defizit ableiten
    # Je größer das Defizit, desto höher die Stärke (aber gecapped für Natürlichkeit)
    _bw_deficit = max(0.0, 1.0 - sr_current / max(sr_target, 1))
    if goal.get("quality_level") == "maximal":
        strength = float(np.clip(0.25 + _bw_deficit * 0.50, 0.15, 0.60))
    elif goal.get("priority") == "transparency":
        strength = float(np.clip(0.15 + _bw_deficit * 0.35, 0.10, 0.40))
    else:
        strength = float(np.clip(0.20 + _bw_deficit * 0.40, 0.12, 0.50))

    params = {
        "sr_target": sr_target,
        "strength": strength,
        "genre": context.get("genre", "unknown"),
        "reason": reason,
    }

    logger.info("HF Extension SELECTED: %s (strength=%.2f, target=%sHz)", reason, strength, sr_target)

    return params


# ============================================================================
# Pipeline Integration Example
# ============================================================================


def apply_hf_extension_if_needed(audio: np.ndarray, sr: int, context: dict, goal: dict) -> tuple[np.ndarray, int]:
    """
    Convenience function für Pipeline-Integration.

    Returns:
        (audio_processed, sr_output)
    """
    # Policy Decision
    hf_params = select_hf_extension_strategy(context, goal)

    if hf_params is None:
        # No HF extension needed
        return audio, sr

    # Apply Extension
    extender = HighFrequencyExtender()
    audio_extended, _is_authentic, _metrics = extender.extend_with_authenticity_check(
        audio,
        sr,
        sr_target=hf_params["sr_target"],
        strength=hf_params["strength"],
        genre=hf_params["genre"],
    )

    return audio_extended, hf_params["sr_target"]


# ============================================================================
# Unit Test
# ============================================================================

if __name__ == "__main__":
    # Test HF Extension Decision Logic
    extender = HighFrequencyExtender()

    # Test 1: Low SR (should extend)
    should, reason = extender.should_extend(sr=16000)
    logger.info("Test 1 (16kHz): Should extend = %s, Reason = %s", should, reason)
    assert should, "16kHz should trigger HF extension"

    # Test 2: CD Quality (might not extend)
    should, reason = extender.should_extend(sr=44100, genre="rock")
    logger.info("Test 2 (44.1kHz Rock): Should extend = %s, Reason = %s", should, reason)

    # Test 3: Classical at 44.1kHz (should extend for air)
    should, reason = extender.should_extend(sr=44100, genre="classical")
    logger.info("Test 3 (44.1kHz Classical): Should extend = %s, Reason = %s", should, reason)
    assert should, "Classical at 44.1kHz should extend for air/brilliance"

    # Test 4: High SR (should not extend)
    should, reason = extender.should_extend(sr=96000)
    logger.info("Test 4 (96kHz): Should extend = %s, Reason = %s", should, reason)
    assert not should, "96kHz should NOT extend"

    # Test 5: Policy Decision
    context = {"sample_rate": 22050, "genre": "jazz", "spectral_rolloff": 12000}
    goal = {"quality_level": "maximal"}

    params = select_hf_extension_strategy(context, goal)
    logger.info("\nTest 5 (Policy Decision):")
    logger.info("  Params = %s", params)
    assert params is not None, "Should select HF extension for 22kHz jazz"

    logger.info("\n✓ All tests passed")
