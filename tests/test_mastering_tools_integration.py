"""
Test für True Peak Limiter und Stereo Width Enhancement Integration

Validiert die neu integrierten Mastering-Tools:
1. True Peak Limiter (ITU-R BS.1770-4 compliant)
2. Stereo Width Enhancer (Mid-Side based)

Author: AURIK Development Team
Date: 10. Februar 2026
"""

import numpy as np
import pytest

from backend.core.dsp.stereo_width_enhancer import StereoWidthEnhancer
from backend.core.dsp.true_peak_limiter import TruePeakLimiter


@pytest.mark.unit
class TestTruePeakLimiter:
    """Tests für True Peak Limiter."""

    def test_basic_limiting(self):
        """Test basic peak limiting functionality."""
        # Create test signal with peaks exceeding ceiling
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Signal with peaks at 0 dBFS (1.0 linear)
        audio = np.sin(2 * np.pi * 1000 * t) * 0.99

        # Add some peaks exceeding -1.0 dBTP
        audio[1000:1100] = 1.0  # Peak at full scale

        limiter = TruePeakLimiter(ceiling_dbtp=-1.0)
        audio_limited, metrics = limiter.process(audio, sr, return_metrics=True)

        # Verify output is below ceiling
        tp_output = limiter.measure_true_peak(audio_limited, sr)
        assert tp_output <= -1.0, f"True Peak exceeds ceiling: {tp_output:.2f} dBTP"

        # Verify metrics
        assert metrics is not None
        assert "true_peak_input_dbtp" in metrics
        assert "true_peak_output_dbtp" in metrics
        assert "gain_reduction_max_db" in metrics

        print(
            f"✓ True Peak Limiter: {metrics['true_peak_input_dbtp']:.2f} → {metrics['true_peak_output_dbtp']:.2f} dBTP"
        )

    def test_stereo_limiting(self):
        """Test stereo signal limiting."""
        sr = 48000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))

        # Stereo signal
        left = np.sin(2 * np.pi * 1000 * t) * 1.05  # Exceeds ceiling
        right = np.sin(2 * np.pi * 1500 * t) * 1.08

        audio_stereo = np.stack([left, right], axis=0)  # (2, samples)

        limiter = TruePeakLimiter(ceiling_dbtp=-1.0)
        audio_limited, metrics = limiter.process(audio_stereo, sr, return_metrics=True)

        # Verify shape preserved
        assert audio_limited.shape == audio_stereo.shape

        # Verify below ceiling
        tp_output = limiter.measure_true_peak(audio_limited, sr)
        assert tp_output <= -1.0, f"True Peak exceeds ceiling: {tp_output:.2f} dBTP (expected <= -1.0 dBTP)"

        assert metrics is not None
        print(f"✓ Stereo True Peak Limiter: {metrics['true_peak_output_dbtp']:.2f} dBTP")

    def test_no_limiting_needed(self):
        """Test when signal already below ceiling."""
        sr = 48000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))

        # Signal well below ceiling
        audio = np.sin(2 * np.pi * 1000 * t) * 0.5  # -6 dBFS

        limiter = TruePeakLimiter(ceiling_dbtp=-1.0)
        audio_limited, metrics = limiter.process(audio, sr, return_metrics=True)

        # Verify no gain reduction applied
        assert metrics is not None
        assert metrics["gain_reduction_max_db"] < 0.1
        assert metrics["samples_limited"] == 0

        print(f"✓ No Limiting Needed: {metrics['true_peak_input_dbtp']:.2f} dBTP already below ceiling")


class TestStereoWidthEnhancer:
    """Tests für Stereo Width Enhancement."""

    def test_width_enhancement(self):
        """Test stereo width enhancement."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create stereo signal with some width
        left = np.sin(2 * np.pi * 1000 * t)
        right = np.sin(2 * np.pi * 1000 * t + np.pi / 4)  # 45° phase shift

        audio = np.stack([left, right], axis=0)  # (2, samples)

        enhancer = StereoWidthEnhancer(width_factor=1.5, safe_mode=True)
        audio_enhanced, metrics = enhancer.process(audio, return_metrics=True)

        # Verify shape preserved
        assert audio_enhanced.shape == audio.shape

        # Verify metrics
        assert metrics is not None
        assert metrics["width_applied"] == 1.5
        assert "phase_correlation_input" in metrics
        assert "phase_correlation_output" in metrics
        assert "mono_compatible" in metrics

        print(f"✓ Stereo Width Enhanced: {metrics['width_applied']:.2f}x")
        print(
            f"  Phase Correlation: {metrics['phase_correlation_input']:.3f} → {metrics['phase_correlation_output']:.3f}"
        )
        print(f"  Mono Compatible: {'Ja' if metrics['mono_compatible'] else 'Nein'} ({metrics['mono_loss_db']:.1f} dB)")

    def test_mono_compatibility(self):
        """Test mono compatibility check."""
        sr = 48000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))

        # Create mono-compatible stereo (identical channels)
        signal = np.sin(2 * np.pi * 1000 * t)
        audio = np.stack([signal, signal], axis=0)

        enhancer = StereoWidthEnhancer(width_factor=1.0)  # No enhancement
        is_compatible, loss_db = enhancer.check_mono_compatibility(audio[0], audio[1])

        # Mono signal should be perfectly compatible
        assert is_compatible
        assert loss_db > -1.0  # Minimal loss

        print(f"✓ Mono Compatibility: {loss_db:.1f} dB loss")

    def test_stereo_field_analysis(self):
        """Test stereo field analysis."""
        sr = 48000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))

        # Create wide stereo signal
        left = np.sin(2 * np.pi * 1000 * t)
        right = np.sin(2 * np.pi * 1200 * t)

        audio = np.stack([left, right], axis=0)

        enhancer = StereoWidthEnhancer()
        analysis = enhancer.analyze_stereo_field(audio)

        # Verify analysis keys
        assert "width_estimate" in analysis
        assert "phase_correlation" in analysis
        assert "mid_energy_db" in analysis
        assert "side_energy_db" in analysis
        assert "mono_compatible" in analysis

        print("✓ Stereo Field Analysis:")
        print(f"  Width Estimate: {analysis['width_estimate']:.2f}")
        print(f"  Phase Correlation: {analysis['phase_correlation']:.3f}")
        print(f"  Mid Energy: {analysis['mid_energy_db']:.1f} dB")
        print(f"  Side Energy: {analysis['side_energy_db']:.1f} dB")

    def test_width_factor_range(self):
        """Test different width factors."""
        sr = 48000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))

        left = np.sin(2 * np.pi * 1000 * t)
        right = np.sin(2 * np.pi * 1000 * t + np.pi / 6)
        audio = np.stack([left, right], axis=0)

        for width_factor in [0.5, 1.0, 1.5, 2.0]:
            enhancer = StereoWidthEnhancer(width_factor=width_factor)
            audio_enhanced, metrics = enhancer.process(audio, return_metrics=True)

            assert metrics is not None
            assert metrics["width_applied"] == width_factor
            print(f"✓ Width Factor {width_factor:.1f}x: Phase Corr {metrics['phase_correlation_output']:.3f}")


class TestIntegration:
    """Integration tests mit ProcessingConfig."""

    def test_processing_config_parameters(self):
        """Test dass neue Parameter in ProcessingConfig existieren."""
        from backend.core.processing_modes import ProcessingMode, get_processing_config

        # Test RESTORATION mode
        config_rest = get_processing_config(ProcessingMode.RESTORATION)
        assert hasattr(config_rest, "stereo_width_factor")
        assert hasattr(config_rest, "true_peak_ceiling_dbtp")
        assert config_rest.stereo_width_factor == 1.0  # Original width
        assert config_rest.true_peak_ceiling_dbtp == -1.0  # EBU R128

        print(
            f"✓ RESTORATION Mode: Width={config_rest.stereo_width_factor:.1f}x, Ceiling={config_rest.true_peak_ceiling_dbtp} dBTP"
        )

        # Test STUDIO_2026 mode
        config_studio = get_processing_config(ProcessingMode.STUDIO_2026)
        assert config_studio.stereo_width_factor == 1.5  # Wider
        assert config_studio.true_peak_ceiling_dbtp == -1.0

        print(
            f"✓ STUDIO_2026 Mode: Width={config_studio.stereo_width_factor:.1f}x, Ceiling={config_studio.true_peak_ceiling_dbtp} dBTP"
        )

    def test_validation(self):
        """Test parameter validation."""
        from backend.core.processing_modes import ProcessingConfig

        # Valid config
        config = ProcessingConfig(stereo_width_factor=1.5, true_peak_ceiling_dbtp=-1.0)
        config.validate()  # Should not raise

        # Invalid width factor (too high)
        config_invalid = ProcessingConfig(stereo_width_factor=5.0)
        with pytest.raises(ValueError):
            config_invalid.validate()

        # Invalid true peak ceiling (positive)
        config_invalid2 = ProcessingConfig(true_peak_ceiling_dbtp=1.0)
        with pytest.raises(ValueError):
            config_invalid2.validate()

        print("✓ Parameter Validation working")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("AURIK - Mastering Tools Integration Test")
    print("=" * 70 + "\n")

    # Run tests
    print("🎚️  Testing True Peak Limiter...")
    test_limiter = TestTruePeakLimiter()
    test_limiter.test_basic_limiting()
    test_limiter.test_stereo_limiting()
    test_limiter.test_no_limiting_needed()

    print("\n🎼 Testing Stereo Width Enhancer...")
    test_stereo = TestStereoWidthEnhancer()
    test_stereo.test_width_enhancement()
    test_stereo.test_mono_compatibility()
    test_stereo.test_stereo_field_analysis()
    test_stereo.test_width_factor_range()

    print("\n🔧 Testing Integration...")
    test_integration = TestIntegration()
    test_integration.test_processing_config_parameters()
    test_integration.test_validation()

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - Mastering Tools Integration Complete")
    print("=" * 70)
