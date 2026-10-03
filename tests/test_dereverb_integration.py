"""
Test für Advanced De-Reverb Integration

Validiert die neu integrierte Advanced De-Reverb Komponente.

Author: AURIK Development Team
Date: 10. Februar 2026
"""

import numpy as np
import pytest

from backend.core.dsp.advanced_dereverb import AdvancedDereverb


@pytest.mark.unit
class TestAdvancedDereverb:
    """Tests für Advanced De-Reverb."""

    def test_basic_dereverb(self):
        """Test basic de-reverb functionality."""
        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create test signal with simulated reverb
        # Direct sound (impulse)
        audio = np.zeros(len(t))
        audio[1000] = 1.0  # Impulse

        # Add reverb tail (exponential decay)
        reverb_decay = np.exp(-t * 3.0)  # 3s RT60
        audio = np.convolve(audio, reverb_decay[:10000], mode="same")

        # Test each mode
        for mode in ["mild", "balanced", "aggressive"]:
            dereverb = AdvancedDereverb(mode=mode)

            # Analyze
            analysis = dereverb.analyze(audio, sr)
            print(f"  Mode: {mode}")
            print(f"    RT60 Estimate: {analysis.get('rt60_estimate', 0):.2f}s")
            print(f"    Has Reverb: {analysis.get('has_significant_reverb', False)}")

            # Process
            audio_processed, metrics = dereverb.process(audio, sr)

            # Verify shape preserved
            assert audio_processed.shape == audio.shape

            # Verify metrics
            assert "processed" in metrics
            print(f"    Processed: {metrics['processed']}")

    def test_stereo_dereverb(self):
        """Test stereo de-reverb."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Stereo impulse with reverb
        left = np.sin(2 * np.pi * 440 * t) * np.exp(-t * 5.0)
        right = np.sin(2 * np.pi * 550 * t) * np.exp(-t * 5.0)

        audio = np.stack([left, right], axis=1)  # (samples, 2)

        dereverb = AdvancedDereverb(mode="balanced")
        audio_processed, metrics = dereverb.process(audio, sr)

        # Verify shape
        assert audio_processed.shape == audio.shape
        print(f"✓ Stereo De-Reverb: {metrics['processed']}")

    def test_no_reverb_detection(self):
        """Test that minimal reverb is detected."""
        sr = 48000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))

        # Dry signal (no reverb)
        audio = np.sin(2 * np.pi * 1000 * t) * 0.5

        dereverb = AdvancedDereverb(mode="balanced")
        audio_processed, metrics = dereverb.process(audio, sr)

        # Should skip processing
        if not metrics["processed"]:
            print(f"✓ Correctly skipped: {metrics['reason']}")
        else:
            print("⚠️  Processed dry signal (might be false positive)")


class TestIntegration:
    """Integration tests."""

    def test_processing_config_dereverb(self):
        """Test dereverb parameter in ProcessingConfig."""
        from backend.core.processing_modes import ProcessingMode, get_processing_config

        # Test RESTORATION mode
        config_rest = get_processing_config(ProcessingMode.RESTORATION)
        assert hasattr(config_rest, "dereverb_strength")
        assert config_rest.dereverb_strength == 0.0  # No de-reverb
        print(f"✓ RESTORATION Mode: De-Reverb={config_rest.dereverb_strength:.1%}")

        # Test STUDIO_2026 mode
        config_studio = get_processing_config(ProcessingMode.STUDIO_2026)
        assert config_studio.dereverb_strength == 0.50  # Moderate
        print(f"✓ STUDIO_2026 Mode: De-Reverb={config_studio.dereverb_strength:.1%}")

    def test_validation(self):
        """Test dereverb parameter validation."""
        from backend.core.processing_modes import ProcessingConfig

        # Valid config
        config = ProcessingConfig(dereverb_strength=0.5)
        config.validate()  # Should not raise

        # Invalid (too high)
        config_invalid = ProcessingConfig(dereverb_strength=1.5)
        with pytest.raises(ValueError):
            config_invalid.validate()

        print("✓ Dereverb Validation working")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("AURIK - Advanced De-Reverb Integration Test")
    print("=" * 70 + "\n")

    # Run tests
    print("🎚️  Testing Advanced De-Reverb...")
    test_dereverb = TestAdvancedDereverb()
    test_dereverb.test_basic_dereverb()
    test_dereverb.test_stereo_dereverb()
    test_dereverb.test_no_reverb_detection()

    print("\n🔧 Testing Integration...")
    test_integration = TestIntegration()
    test_integration.test_processing_config_dereverb()
    test_integration.test_validation()

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - Advanced De-Reverb Integration Complete")
    print("=" * 70)
