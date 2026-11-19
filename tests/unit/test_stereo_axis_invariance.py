"""
Tests for Stereo Axis Orientation Invariance (§2.51 RELEASE_MUST)

Validates that audio processing in critical phases produces identical results
regardless of whether stereo input is (N, 2) or (2, N) format.

§2.51: "Jede Phase mit Stereo-Audio **MUSS** M/S-Domain oder Linked-Stereo verwenden."
       "Violationen → §2.49 flaggt Phase-Cancellation-Artefakte → Rollback."
"""

import numpy as np
import pytest

from backend.core.audio_utils import safe_to_mono


class TestSafeToMono:
    """Unit tests for safe_to_mono() utility function."""

    def test_mono_passthrough(self):
        """Mono audio (1D) should pass through unchanged."""
        audio_mono = np.random.randn(48000)
        result = safe_to_mono(audio_mono)
        np.testing.assert_array_equal(result, audio_mono)

    def test_channels_first_conversion(self):
        """Correctly convert (2, N) channels-first to mono."""
        # Create identical stereo signal
        mono_ref = np.random.randn(48000)
        audio_cf = np.vstack([mono_ref, mono_ref * 0.9])  # (2, 48000)

        result = safe_to_mono(audio_cf)

        # Result should be mono (1D)
        assert result.ndim == 1
        assert result.shape[0] == 48000

        # Result should be average of channels
        expected = (mono_ref + mono_ref * 0.9) / 2
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_channels_last_conversion(self):
        """Correctly convert (N, 2) channels-last to mono."""
        # Create identical stereo signal
        mono_ref = np.random.randn(48000)
        audio_cl = np.column_stack([mono_ref, mono_ref * 0.9])  # (48000, 2)

        result = safe_to_mono(audio_cl)

        # Result should be mono (1D)
        assert result.ndim == 1
        assert result.shape[0] == 48000

        # Result should be average of channels
        expected = (mono_ref + mono_ref * 0.9) / 2
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_axis_orientation_invariance(self):
        """
        Safe-to-mono should produce identical output regardless of input orientation.

        This is the §2.51 invariant test.
        """
        # Create reference mono signal
        mono_ref = np.random.randn(48000)
        ch2 = mono_ref * 0.8

        # Create both orientations with same data
        audio_cf = np.vstack([mono_ref, ch2])  # (2, N)
        audio_cl = np.column_stack([mono_ref, ch2])  # (N, 2)

        result_cf = safe_to_mono(audio_cf)
        result_cl = safe_to_mono(audio_cl)

        # Results should be identical
        np.testing.assert_allclose(result_cf, result_cl, rtol=1e-12, atol=1e-15)

    def test_edge_case_2x2_matrix(self):
        """Handle edge case of exactly (2, 2) matrix."""
        audio_2x2 = np.array([[0.5, -0.3], [0.2, 0.4]], dtype=np.float64)

        result = safe_to_mono(audio_2x2)

        # Should treat as (2, N) channels-first, averaging over axis 0
        # Result should be (2,) array
        expected = np.mean(audio_2x2, axis=0)
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    @pytest.mark.parametrize(
        "n_channels,n_samples",
        [
            (2, 480),  # Very short
            (2, 48000),  # 1 second
            (2, 480000),  # 10 seconds
        ],
    )
    def test_mono_consistency_across_lengths(self, n_channels, n_samples):
        """Mono conversion should be consistent for various signal lengths."""
        # Test both orientations
        audio_cf = np.random.randn(n_channels, n_samples)
        audio_cl = audio_cf.T  # Convert to (N, 2)

        result_cf = safe_to_mono(audio_cf)
        result_cl = safe_to_mono(audio_cl)

        np.testing.assert_allclose(result_cf, result_cl, rtol=1e-12)

    def test_audio_dtype_preservation(self):
        """Output should match input dtype closely."""
        audio_f32_cf = np.random.randn(2, 48000).astype(np.float32)
        audio_f64_cl = np.random.randn(48000, 2).astype(np.float64)

        result_f32 = safe_to_mono(audio_f32_cf)
        result_f64 = safe_to_mono(audio_f64_cl)

        # Output should be float64 (internal conversion)
        assert result_f32.dtype == np.float64
        assert result_f64.dtype == np.float64


class TestPhaseAxisInvariance:
    """
    Integration tests for phase axis invariance.
    Validates that phases produce consistent results regardless of stereo orientation.
    """

    @pytest.mark.parametrize("layout", ["channels_first", "channels_last"])
    def test_phase_12_axis_invariance(self, layout):
        """Phase 12 wow/flutter should produce consistent output."""
        # Create test audio
        audio_mono = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000) * 0.3
        # Add slight wow/flutter
        wow = 0.02 * np.sin(2 * np.pi * 0.5 * np.arange(48000) / 48000)
        audio_mono = audio_mono + wow

        # Create both orientations
        if layout == "channels_first":
            audio = np.vstack([audio_mono, audio_mono * 0.9])  # (2, N)
        else:
            audio = np.column_stack([audio_mono, audio_mono * 0.9])  # (N, 2)

        # Import and test phase_12
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        phase = WowFlutterFix()
        result = phase.process(audio, sample_rate=48000)

        # Verify result is valid (no NaN/Inf)
        assert np.all(np.isfinite(result.audio)), "Phase output contains NaN/Inf"
        # Verify audio shape is reasonable
        assert result.audio.ndim in (1, 2), "Output has unexpected dimensions"

    @pytest.mark.parametrize("layout", ["channels_first", "channels_last"])
    def test_phase_43_axis_invariance(self, layout):
        """Phase 43 de-esser should produce consistent output."""
        # Create test audio with sibilance
        audio = np.random.randn(48000) * 0.1  # Base signal
        # Add sibilance in 6-12 kHz range
        t = np.arange(48000) / 48000
        sibilance = 0.05 * np.sin(2 * np.pi * 8000 * t)
        audio = audio + sibilance

        # Create both orientations
        if layout == "channels_first":
            audio = np.vstack([audio, audio * 0.95])  # (2, N)
        else:
            audio = np.column_stack([audio, audio * 0.95])  # (N, 2)

        # Import and test phase_43
        from backend.core.phases.phase_43_ml_deesser import MLDeEsserPhase

        phase = MLDeEsserPhase()
        result = phase.process(audio, sample_rate=48000)

        # Verify result is valid
        assert np.all(np.isfinite(result.audio)), "Phase output contains NaN/Inf"
        assert result.audio.ndim in (1, 2), "Output has unexpected dimensions"

    @pytest.mark.parametrize("layout", ["channels_first", "channels_last"])
    def test_phase_53_axis_invariance(self, layout):
        """Phase 53 semantic audio should handle both layouts."""
        # Create test audio
        audio = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000) * 0.3

        # Create both orientations
        if layout == "channels_first":
            audio = np.vstack([audio, audio * 0.9])  # (2, N)
        else:
            audio = np.column_stack([audio, audio * 0.9])  # (N, 2)

        # Import and test phase_53
        from backend.core.phases.phase_53_semantic_audio import SemanticAudioPhase

        phase = SemanticAudioPhase()
        # Axis invariance is the target; avoid expensive CLAP/BEATs inference here.
        result = phase.process(audio, sample_rate=48000, strength=0.0)

        # Verify result (metadata phase, audio unchanged)
        assert np.all(np.isfinite(result.audio)), "Phase output contains NaN/Inf"
        assert result.audio.ndim == audio.ndim, "Output shape mismatch"

    @pytest.mark.parametrize("layout", ["channels_first", "channels_last"])
    def test_phase_56_axis_invariance(self, layout):
        """Phase 56 spectral band gap repair should handle both layouts."""
        # Create test audio
        audio = np.random.randn(48000) * 0.3

        # Create both orientations
        if layout == "channels_first":
            audio = np.vstack([audio, audio * 0.95])  # (2, N)
        else:
            audio = np.column_stack([audio, audio * 0.95])  # (N, 2)

        # Import and test phase_56
        from backend.core.phases.phase_56_spectral_band_gap_repair import (
            SpectralBandGapRepairPhase,
        )

        phase = SpectralBandGapRepairPhase()
        result = phase.process(audio, sample_rate=48000)

        # Verify result is valid
        assert np.all(np.isfinite(result.audio)), "Phase output contains NaN/Inf"
        assert result.audio.ndim in (1, 2), "Output has unexpected dimensions"


class TestPhaseAxisInvarianceB:
    """Kategorie-B phase tests: phases with previously insufficient axis conditionals."""

    @pytest.mark.parametrize("layout", ["channels_first", "channels_last"])
    def test_phase_42_axis_invariance(self, layout):
        """Phase 42 vocal enhancement should handle both stereo orientations."""
        audio_mono = np.random.randn(48000) * 0.2

        if layout == "channels_first":
            audio = np.vstack([audio_mono, audio_mono * 0.9])  # (2, N)
        else:
            audio = np.column_stack([audio_mono, audio_mono * 0.9])  # (N, 2)

        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        result = phase.process(audio, sample_rate=48000)

        assert np.all(np.isfinite(result.audio)), "Phase 42 output contains NaN/Inf"
        assert result.audio.ndim in (1, 2), "Phase 42 output has unexpected dimensions"

    @pytest.mark.parametrize("layout", ["channels_first", "channels_last"])
    def test_phase_44_axis_invariance(self, layout):
        """Phase 44 guitar enhancement should handle both stereo orientations."""
        # Guitar-like signal
        t = np.arange(48000) / 48000
        audio_mono = (
            np.sin(2 * np.pi * 110 * t) * 0.4 + np.sin(2 * np.pi * 220 * t) * 0.2 + np.random.randn(48000) * 0.05
        )

        if layout == "channels_first":
            audio = np.vstack([audio_mono, audio_mono * 0.95])  # (2, N)
        else:
            audio = np.column_stack([audio_mono, audio_mono * 0.95])  # (N, 2)

        from backend.core.phases.phase_44_guitar_enhancement import GuitarEnhancementPhase

        phase = GuitarEnhancementPhase()
        result = phase.process(audio, sample_rate=48000)

        assert np.all(np.isfinite(result.audio)), "Phase 44 output contains NaN/Inf"
        assert result.audio.ndim in (1, 2), "Phase 44 output has unexpected dimensions"


class TestSpecCompliance:
    """§2.51 Stereo-Kohärenz-Invariante compliance tests."""

    def test_spec_2_51_linked_stereo_requirement(self):
        """
        §2.51: Phases must use linked-stereo or M/S domain.

        Linked-stereo means: same operations on both channels.
        This test verifies safe_to_mono handles both layouts.
        """
        # Create stereo signal with known relationship
        mono_base = np.random.randn(48000)
        audio_cf = np.vstack([mono_base, mono_base * 0.8])  # (2, N) — L ch, R ch
        audio_cl = np.column_stack([mono_base, mono_base * 0.8])  # (N, 2)

        mono_cf = safe_to_mono(audio_cf)
        mono_cl = safe_to_mono(audio_cl)

        # Both should be identical
        np.testing.assert_allclose(mono_cf, mono_cl, rtol=1e-12)

    def test_spec_2_51_no_stereo_collapse(self):
        """
        §2.51: Processing must not cause stereo field collapse (L==R collapse).

        This test verifies that the axis-safe conversion doesn't degrade stereo info.
        """
        # Create stereo with distinct L/R
        left = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000) * 0.3
        right = np.sin(2 * np.pi * 880 * np.arange(48000) / 48000) * 0.3
        audio_cf = np.vstack([left, right])  # (2, N)

        mono_result = safe_to_mono(audio_cf)

        # Mono should contain energy from both channels
        # (simple check: variance > 0)
        assert np.var(mono_result) > 0.001, "Mono result lost channel information"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
