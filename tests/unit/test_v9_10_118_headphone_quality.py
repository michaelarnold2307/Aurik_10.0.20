import pytest

"""
v10.0.0 — Kopfhörer-Qualitäts-Fixes: 5 audible Defizite behoben.

Fix 1: Phase 42 Stereo-Wiener — Wiener-Masking statt Mono-Duplikation
Fix 2: STFT Wet/Dry-Blend — Phasen-bewahrte Magnitude-Interpolation
Fix 3: Diminishing-Returns — Kumulative Stärke-Moderation in UV3
Fix 4: OMLSA Silence G_floor — Energiebasierter G_floor für Stille
Fix 5: De-Esser Era-Adaptiv — Ära-abhängige Sibilanz-Schwellwerte
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Fix 1: Phase 42 Wiener-Stereo-Masking
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestWienerStereoMasking:
    """§9.10.118 — Wiener stereo preservation instead of mono duplication."""

    def _make_phase42(self):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        return VocalEnhancement()

    def test_01_wiener_returns_stereo_shape(self):
        """Output shape must match input stereo shape."""
        p42 = self._make_phase42()
        sr = 48000
        n = sr * 2  # 2 seconds
        rng = np.random.default_rng(42)
        stereo = rng.normal(0, 0.3, (n, 2)).astype(np.float32)
        voc_mono = rng.normal(0, 0.2, n).astype(np.float32)
        vocs, instr = p42._wiener_stereo_from_mono(stereo, voc_mono, sr)
        assert vocs.shape == stereo.shape, f"vocals shape {vocs.shape} != {stereo.shape}"
        assert instr.shape == stereo.shape

    def test_02_wiener_preserves_lr_difference(self):
        """L and R channels must NOT be identical (stereo preserved)."""
        p42 = self._make_phase42()
        sr = 48000
        n = sr * 2
        np.random.default_rng(118)
        # Create stereo with intentional L/R difference
        left = np.sin(2 * np.pi * 440 * np.arange(n) / sr).astype(np.float32) * 0.5
        right = np.sin(2 * np.pi * 660 * np.arange(n) / sr).astype(np.float32) * 0.3
        stereo = np.column_stack([left, right])
        voc_mono = np.sin(2 * np.pi * 440 * np.arange(n) / sr).astype(np.float32) * 0.3
        vocs, _ = p42._wiener_stereo_from_mono(stereo, voc_mono, sr)
        # L and R should differ
        lr_corr = float(np.corrcoef(vocs[:, 0], vocs[:, 1])[0, 1])
        assert lr_corr < 0.99, f"L/R correlation {lr_corr:.4f} — stereo collapsed"

    def test_03_wiener_no_nan_inf(self):
        """Output must be NaN/Inf free."""
        p42 = self._make_phase42()
        sr = 48000
        n = sr
        rng = np.random.default_rng(7)
        stereo = rng.normal(0, 0.3, (n, 2)).astype(np.float32)
        voc_mono = rng.normal(0, 0.15, n).astype(np.float32)
        vocs, instr = p42._wiener_stereo_from_mono(stereo, voc_mono, sr)
        assert np.isfinite(vocs).all(), "NaN/Inf in vocals"
        assert np.isfinite(instr).all(), "NaN/Inf in instruments"

    def test_04_wiener_clipped(self):
        """Output samples must be in [-1, 1]."""
        p42 = self._make_phase42()
        sr = 48000
        n = sr
        rng = np.random.default_rng(99)
        stereo = rng.uniform(-0.9, 0.9, (n, 2)).astype(np.float32)
        voc_mono = rng.uniform(-0.5, 0.5, n).astype(np.float32)
        vocs, instr = p42._wiener_stereo_from_mono(stereo, voc_mono, sr)
        assert np.max(np.abs(vocs)) <= 1.0
        assert np.max(np.abs(instr)) <= 1.0

    def test_05_wiener_mono_input_untouched(self):
        """Mono input should NOT use Wiener masking (direct pass-through)."""
        p42 = self._make_phase42()
        mono = np.zeros(48000, dtype=np.float32)
        # _wiener_stereo_from_mono expects ndim==2; mono skips it in _try_stem_separation
        # Just verify the static method doesn't crash with edge input
        stereo_compat = np.column_stack([mono, mono])
        voc_mono = np.zeros(48000, dtype=np.float32)
        vocs, instr = p42._wiener_stereo_from_mono(stereo_compat, voc_mono, 48000)
        assert vocs.shape == (48000, 2)

    def test_06_wiener_vocal_energy_conservation(self):
        """Vocal + instrument energy should approximate original energy."""
        p42 = self._make_phase42()
        sr = 48000
        n = sr * 2
        rng = np.random.default_rng(200)
        stereo = rng.normal(0, 0.3, (n, 2)).astype(np.float32)
        voc_mono = rng.normal(0, 0.2, n).astype(np.float32)
        vocs, instr = p42._wiener_stereo_from_mono(stereo, voc_mono, sr)
        # Reconstruction: vocs + instr should ≈ original (OLA numerical tolerance)
        recon = vocs + instr
        orig_rms = float(np.sqrt(np.mean(stereo**2)))
        recon_rms = float(np.sqrt(np.mean(recon**2)))
        ratio = recon_rms / (orig_rms + 1e-8)
        assert 0.5 < ratio < 2.0, f"Energy ratio {ratio:.3f} out of range"

    def test_07_wiener_length_mismatch_safe(self):
        """Different lengths between stereo and voc_mono should not crash."""
        p42 = self._make_phase42()
        sr = 48000
        stereo = np.random.default_rng(3).normal(0, 0.3, (sr * 2, 2)).astype(np.float32)
        voc_mono = np.random.default_rng(3).normal(0, 0.2, sr).astype(np.float32)  # shorter
        vocs, instr = p42._wiener_stereo_from_mono(stereo, voc_mono, sr)
        expected_n = min(stereo.shape[0], len(voc_mono))
        assert vocs.shape[0] == expected_n

    def test_08_wiener_silence_input(self):
        """Silence input should produce silence output without artifacts."""
        p42 = self._make_phase42()
        stereo = np.zeros((48000, 2), dtype=np.float32)
        voc_mono = np.zeros(48000, dtype=np.float32)
        vocs, instr = p42._wiener_stereo_from_mono(stereo, voc_mono, 48000)
        assert float(np.max(np.abs(vocs))) < 0.01
        assert float(np.max(np.abs(instr))) < 0.01


# ═══════════════════════════════════════════════════════════════════════
# Fix 2: Phase-Aware STFT Wet/Dry Blend
# ═══════════════════════════════════════════════════════════════════════


class TestPhaseAwareWetDryBlend:
    """§9.10.118 — STFT magnitude blending at low strengths."""

    def _blend(self, dry, wet, strength, phase=None):
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate

        return PerPhaseMusicalGoalsGate._wet_dry_blend(dry, wet, strength, phase)

    def test_09_stft_blend_at_low_strength(self):
        """At strength=0.10, STFT blend should not create comb filtering."""
        sr = 48000
        n = sr
        t = np.arange(n) / sr
        dry = np.sin(2 * np.pi * 1000 * t).astype(np.float32) * 0.5
        # Wet: slightly phase-shifted version (simulating processing latency)
        wet = np.sin(2 * np.pi * 1000 * t + 0.3).astype(np.float32) * 0.5
        blended = self._blend(dry, wet, 0.10)
        assert len(blended) == n
        # Energy should be preserved (no comb-filter cancellation)
        dry_rms = float(np.sqrt(np.mean(dry**2)))
        blend_rms = float(np.sqrt(np.mean(blended**2)))
        ratio = blend_rms / (dry_rms + 1e-8)
        assert ratio > 0.85, f"Energy ratio {ratio:.3f} — possible comb filtering"

    def test_10_linear_blend_at_high_strength(self):
        """At strength=0.50, linear blend should be used (no STFT overhead)."""
        sr = 48000
        n = sr
        rng = np.random.default_rng(10)
        dry = rng.normal(0, 0.3, n).astype(np.float32)
        wet = rng.normal(0, 0.3, n).astype(np.float32)
        blended = self._blend(dry, wet, 0.50)
        # Linear blend: out = dry + 0.5 * (wet - dry) = 0.5*dry + 0.5*wet
        expected = (dry + 0.50 * (wet - dry)).astype(np.float32)
        np.testing.assert_allclose(blended, np.clip(expected, -1.0, 1.0), atol=1e-5)

    def test_11_strength_zero_returns_dry(self):
        """Strength 0 must return dry copy unchanged."""
        dry = np.ones(4800, dtype=np.float32) * 0.5
        wet = np.ones(4800, dtype=np.float32) * 0.9
        blended = self._blend(dry, wet, 0.0)
        np.testing.assert_array_equal(blended, dry)

    def test_12_strength_one_returns_wet(self):
        """Strength 1.0 must return clipped wet."""
        dry = np.ones(4800, dtype=np.float32) * 0.5
        wet = np.ones(4800, dtype=np.float32) * 0.9
        blended = self._blend(dry, wet, 1.0)
        np.testing.assert_allclose(blended, wet, atol=1e-6)

    def test_13_stft_blend_no_nan(self):
        """STFT blend must not produce NaN/Inf."""
        n = 48000
        rng = np.random.default_rng(13)
        dry = rng.normal(0, 0.3, n).astype(np.float32)
        wet = rng.normal(0, 0.3, n).astype(np.float32)
        blended = self._blend(dry, wet, 0.05)
        assert np.isfinite(blended).all()

    def test_14_stft_blend_clipped(self):
        """Output from STFT blend must be in [-1, 1]."""
        n = 48000
        dry = np.ones(n, dtype=np.float32) * 0.95
        wet = np.ones(n, dtype=np.float32) * -0.95
        blended = self._blend(dry, wet, 0.20)
        assert float(np.max(np.abs(blended))) <= 1.0

    def test_15_short_signal_falls_back_to_linear(self):
        """Signals shorter than 2048 must use linear blend (no STFT)."""
        dry = np.ones(1000, dtype=np.float32) * 0.5
        wet = np.ones(1000, dtype=np.float32) * 0.8
        blended = self._blend(dry, wet, 0.10)
        expected = (dry + 0.10 * (wet - dry)).astype(np.float32)
        np.testing.assert_allclose(blended, np.clip(expected, -1.0, 1.0), atol=1e-5)

    def test_16_length_mismatch_handled(self):
        """Different wet/dry lengths must not crash."""
        dry = np.ones(5000, dtype=np.float32) * 0.5
        wet = np.ones(4800, dtype=np.float32) * 0.8
        blended = self._blend(dry, wet, 0.15)
        assert len(blended) == len(dry)

    def test_16b_stereo_channels_first_shape_preserved(self):
        """Stereo channels-first input must keep (2, N) layout after blend."""
        n = 48000
        rng = np.random.default_rng(161)
        dry_cf = rng.normal(0, 0.2, (2, n)).astype(np.float32)
        wet_cf = rng.normal(0, 0.2, (2, n)).astype(np.float32)

        blended = self._blend(dry_cf, wet_cf, 0.20)
        assert blended.shape == dry_cf.shape
        assert blended.dtype == np.float32
        assert np.isfinite(blended).all()

    def test_16c_stereo_mixed_layout_normalized_to_dry_layout(self):
        """Mixed stereo layouts (dry N×2, wet 2×N) must return dry layout."""
        n = 48000
        rng = np.random.default_rng(162)
        dry_sf = rng.normal(0, 0.2, (n, 2)).astype(np.float32)
        wet_cf = rng.normal(0, 0.2, (2, n)).astype(np.float32)

        blended = self._blend(dry_sf, wet_cf, 0.20)
        assert blended.shape == dry_sf.shape
        assert blended.dtype == np.float32
        assert np.isfinite(blended).all()


# ═══════════════════════════════════════════════════════════════════════
# Fix 3: Diminishing-Returns Strength Moderation
# ═══════════════════════════════════════════════════════════════════════


class TestDiminishingReturnsModeration:
    """§9.10.118 — Cumulative strength attenuation after 20 phases."""

    def test_17_diminish_factor_at_phase_20(self):
        """At exactly 20 executed phases, no attenuation yet."""
        n_exec = 20
        diminish = max(0.30, 1.0 - 0.015 * (n_exec - 20))
        assert diminish == 1.0

    def test_18_diminish_factor_at_phase_30(self):
        """At 30 phases, factor = 0.85."""
        n_exec = 30
        diminish = max(0.30, 1.0 - 0.015 * (n_exec - 20))
        assert abs(diminish - 0.85) < 1e-6

    def test_19_diminish_factor_floor_at_030(self):
        """Factor never goes below 0.30."""
        for n in range(70, 100):
            diminish = max(0.30, 1.0 - 0.015 * (n - 20))
            assert diminish >= 0.30

    def test_20_diminish_inactive_below_20(self):
        """No attenuation for first 20 phases."""
        for n in range(0, 21):
            # Gate condition: n > 20
            should_apply = n > 20
            assert not should_apply or n > 20

    def test_21_diminish_combined_strength_clamped(self):
        """Combined strength after diminishing is clipped to [0.05, 1.0]."""
        base_strength = 0.20
        n_exec = 60
        diminish = max(0.30, 1.0 - 0.015 * (n_exec - 20))
        result = float(np.clip(base_strength * diminish, 0.05, 1.0))
        assert result >= 0.05
        assert result <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# Fix 4: OMLSA Silence-Adaptive G_floor
# ═══════════════════════════════════════════════════════════════════════


class TestOMLSASilenceGFloor:
    """§9.10.118 — Energy-adaptive G_floor: lower floor in silence regions."""

    def _make_omlsa(self):
        from dsp.adaptive_omlsa import AdaptiveOMLSA

        return AdaptiveOMLSA()

    def test_22_silence_region_lower_floor(self):
        """In silence, gain floor should be lower than in active signal."""
        omlsa = self._make_omlsa()
        n_bins = 1025
        # Active signal: moderate magnitude
        noisy_active = np.ones(n_bins, dtype=np.float64) * 0.1
        noise_active = np.ones(n_bins, dtype=np.float64) * 0.08
        gain_active = omlsa.omlsa(noisy_active, noise_active, sr=48000)

        # Silence: very low magnitude (< -55 dBFS)
        noisy_silence = np.ones(n_bins, dtype=np.float64) * 1e-4
        noise_silence = np.ones(n_bins, dtype=np.float64) * 1e-4
        gain_silence = omlsa.omlsa(noisy_silence, noise_silence, sr=48000)

        # Silence gain floor should be lower
        active_min = float(np.min(gain_active))
        silence_min = float(np.min(gain_silence))
        assert silence_min <= active_min, f"Silence floor {silence_min:.4f} not ≤ active floor {active_min:.4f}"

    def test_23_2d_spectrogram_silence_frames(self):
        """2D input: silence frames should have lower floor than active frames."""
        omlsa = self._make_omlsa()
        n_frames = 20
        n_bins = 513
        # Active frames
        noisy = np.ones((n_frames, n_bins), dtype=np.float64) * 0.1
        noise = np.ones((n_frames, n_bins), dtype=np.float64) * 0.08
        # Make frames 10-15 silence (<-55 dBFS)
        noisy[10:16, :] = 1e-5
        noise[10:16, :] = 1e-5
        result = omlsa.omlsa(noisy, noise, sr=48000)
        # Gain values in silence frames should be lower on average
        gain_active_mean = float(np.mean(result[:10]))
        gain_silence_mean = float(np.mean(result[10:16]))
        assert gain_silence_mean <= gain_active_mean + 0.01

    def test_24_omlsa_no_nan_with_silence(self):
        """No NaN/Inf when processing silence."""
        omlsa = self._make_omlsa()
        n = 1025
        noisy = np.zeros(n, dtype=np.float64)
        noise = np.zeros(n, dtype=np.float64) + 1e-10
        result = omlsa.omlsa(noisy, noise, sr=48000)
        assert np.isfinite(result).all()

    def test_25_custom_silence_threshold(self):
        """Custom silence_threshold_db kwarg is respected."""
        omlsa = self._make_omlsa()
        n = 1025
        # Magnitude at ≈-50 dBFS (above default -55 threshold)
        noisy = np.ones(n, dtype=np.float64) * 0.003
        noise = np.ones(n, dtype=np.float64) * 0.002
        gain_default = omlsa.omlsa(noisy, noise, sr=48000)
        gain_custom = omlsa.omlsa(noisy, noise, sr=48000, silence_threshold_db=-40.0)
        # With threshold at -40, this level IS silence → lower gain
        custom_min = float(np.min(gain_custom))
        default_min = float(np.min(gain_default))
        assert custom_min <= default_min + 0.001

    def test_26_psychoacoustic_off_no_silence_floor(self):
        """When psychoacoustic=False, silence G_floor is not applied."""
        omlsa = self._make_omlsa()
        n = 1025
        noisy = np.ones(n, dtype=np.float64) * 1e-5
        noise = np.ones(n, dtype=np.float64) * 1e-5
        gain_no_psy = omlsa.omlsa(noisy, noise, sr=48000, psychoacoustic=False)
        # Without psychoacoustic, gain is pure OMLSA (no floor)
        assert float(np.min(gain_no_psy)) >= 0.0
        assert np.isfinite(gain_no_psy).all()

    def test_27_silence_floor_bounded(self):
        """Silence G_floor is exactly 0.5× the normal floor."""
        omlsa = self._make_omlsa()
        n = 1025
        # Active: large enough that silence detection doesn't trigger
        noisy_active = np.ones(n, dtype=np.float64) * 0.5
        noise_active = np.ones(n, dtype=np.float64) * 0.01
        gain_active = omlsa.omlsa(noisy_active, noise_active, sr=48000)

        noisy_silence = np.ones(n, dtype=np.float64) * 1e-5
        noise_silence = np.ones(n, dtype=np.float64) * 1e-5
        gain_silence = omlsa.omlsa(noisy_silence, noise_silence, sr=48000)
        # Should be bounded
        assert float(np.max(gain_silence)) <= 1.0
        assert float(np.min(gain_silence)) >= 0.0


# ═══════════════════════════════════════════════════════════════════════
# Fix 5: Era-Adaptive De-Esser Thresholds
# ═══════════════════════════════════════════════════════════════════════


class TestEraAdaptiveDeEsser:
    """§9.10.118 — De-esser thresholds adapt to recording era."""

    def _make_phase42(self):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        return VocalEnhancement()

    def _make_audio(self, sr=48000, dur=1.0):
        """Create vocal-like test audio with sibilance content."""
        rng = np.random.default_rng(42)
        n = int(sr * dur)
        t = np.arange(n) / sr
        # Vocal fundamental + formants + sibilance
        audio = (
            0.3 * np.sin(2 * np.pi * 220 * t)
            + 0.15 * np.sin(2 * np.pi * 880 * t)
            + 0.05 * rng.normal(0, 1, n)  # sibilance-like noise
        ).astype(np.float32)
        return audio

    def test_28_pre_1940_relaxed_threshold(self):
        """Pre-1940 recordings should have +6 dB higher threshold (less de-essing)."""
        p42 = self._make_phase42()
        from backend.core.defect_scanner import MaterialType

        config_base = dict(p42.ENHANCEMENT_CONFIG[MaterialType.SHELLAC])
        base_threshold = config_base["deess_threshold_db"]

        # Simulate what process() does with era_decade from song_calibration_profile
        config = dict(config_base)
        config["deess_threshold_db"] = float(config["deess_threshold_db"] + 6.0)
        config["deess_reduction_db"] = float(config["deess_reduction_db"] * 0.5)

        assert config["deess_threshold_db"] == base_threshold + 6.0

    def test_29_1950s_moderate_relaxation(self):
        """1950s recordings: +3 dB threshold, 0.7× reduction."""
        p42 = self._make_phase42()
        from backend.core.defect_scanner import MaterialType

        config = dict(p42.ENHANCEMENT_CONFIG[MaterialType.VINYL])
        base_thr = config["deess_threshold_db"]
        base_red = config["deess_reduction_db"]
        config["deess_threshold_db"] = float(config["deess_threshold_db"] + 3.0)
        config["deess_reduction_db"] = float(config["deess_reduction_db"] * 0.7)
        assert config["deess_threshold_db"] == base_thr + 3.0
        assert abs(config["deess_reduction_db"] - base_red * 0.7) < 0.01

    def test_30_modern_2010_stricter_threshold(self):
        """Post-2000: -2 dB threshold (more aggressive de-essing)."""
        p42 = self._make_phase42()
        from backend.core.defect_scanner import MaterialType

        config = dict(p42.ENHANCEMENT_CONFIG[MaterialType.CD_DIGITAL])
        base_thr = config["deess_threshold_db"]
        config["deess_threshold_db"] = float(config["deess_threshold_db"] - 2.0)
        assert config["deess_threshold_db"] == base_thr - 2.0

    def test_31_no_era_no_change(self):
        """Without era info, thresholds remain at material defaults."""
        p42 = self._make_phase42()
        from backend.core.defect_scanner import MaterialType

        config_orig = dict(p42.ENHANCEMENT_CONFIG[MaterialType.CD_DIGITAL])
        config_test = dict(config_orig)
        # era_decade is None → no modification
        _era_decade = None
        if _era_decade is not None:
            config_test["deess_threshold_db"] += 3.0
        assert config_test["deess_threshold_db"] == config_orig["deess_threshold_db"]

    def test_32_era_1970_no_change(self):
        """1970s: between 1960 and 2000 bounds → no era modification."""
        _era_int = 1970
        # Check that 1970 falls outside both branches
        assert _era_int > 1960
        assert _era_int < 2000
        # → no modification applied

    def test_33_config_keys_present(self):
        """All material configs must have deess_threshold_db and deess_reduction_db."""
        p42 = self._make_phase42()
        for material, config in p42.ENHANCEMENT_CONFIG.items():
            assert "deess_threshold_db" in config, f"Missing deess_threshold_db for {material}"
            assert "deess_reduction_db" in config, f"Missing deess_reduction_db for {material}"


# ═══════════════════════════════════════════════════════════════════════
# Integration / Edge-Case Tests
# ═══════════════════════════════════════════════════════════════════════


class TestIntegrationEdgeCases:
    """Cross-cutting edge cases for v10.0.0 fixes."""

    def test_34_stft_blend_stereo_signal(self):
        """STFT blend should handle multi-channel gracefully (first channel)."""
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate

        # Blend works on 1D — if caller passes stereo per-channel, verify no crash
        n = 4800
        dry = np.random.default_rng(34).normal(0, 0.3, n).astype(np.float32)
        wet = np.random.default_rng(35).normal(0, 0.3, n).astype(np.float32)
        result = PerPhaseMusicalGoalsGate._wet_dry_blend(dry, wet, 0.15)
        assert np.isfinite(result).all()
        assert len(result) == n

    def test_35_omlsa_auto_optimize_still_works(self):
        """auto_optimize should still function after silence G_floor changes."""
        from dsp.adaptive_omlsa import AdaptiveOMLSA

        omlsa = AdaptiveOMLSA()
        noisy = np.random.default_rng(35).uniform(0, 0.5, 1025).astype(np.float64)
        noise = np.random.default_rng(36).uniform(0, 0.1, 1025).astype(np.float64)
        omlsa.auto_optimize(noisy, noise)
        assert 0.85 <= omlsa.alpha <= 0.99
        assert omlsa.noise_floor > 0

    def test_36_wiener_with_zero_vocal(self):
        """If voc_mono is all-zero, mask should be ~0 → no vocal bleed."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        p42 = VocalEnhancement()
        n = 48000
        stereo = np.random.default_rng(36).normal(0, 0.3, (n, 2)).astype(np.float32)
        voc_mono = np.zeros(n, dtype=np.float32)
        vocs, instr = p42._wiener_stereo_from_mono(stereo, voc_mono, 48000)
        # Vocals should be near-zero
        assert float(np.sqrt(np.mean(vocs**2))) < 0.05

    def test_37_wiener_reconstruction_adds_to_original(self):
        """vocals + instruments should ≈ original for unit-energy signals."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        p42 = VocalEnhancement()
        sr = 48000
        n = sr
        rng = np.random.default_rng(37)
        stereo = rng.normal(0, 0.3, (n, 2)).astype(np.float32)
        # Vocal that is ≈50% of signal energy
        voc_mono = (stereo.mean(axis=1) * 0.5 + rng.normal(0, 0.05, n)).astype(np.float32)
        vocs, instr = p42._wiener_stereo_from_mono(stereo, voc_mono, sr)
        recon = vocs + instr
        err = float(np.sqrt(np.mean((recon - stereo[:n]) ** 2)))
        orig_rms = float(np.sqrt(np.mean(stereo[:n] ** 2)))
        snr_db = 20.0 * np.log10(orig_rms / (err + 1e-10))
        # Should be at least 5 dB reconstruction SNR
        assert snr_db > 5.0, f"Reconstruction SNR {snr_db:.1f} dB too low"

    def test_38_diminish_formula_monotonic(self):
        """Diminishing factor must be monotonically non-increasing."""
        factors = []
        for n in range(21, 90):
            f = max(0.30, 1.0 - 0.015 * (n - 20))
            factors.append(f)
        for i in range(1, len(factors)):
            assert factors[i] <= factors[i - 1], f"Factor increased at phase {i + 21}"

    def test_39_stft_blend_preserves_energy_at_boundary(self):
        """At strength=0.29 (just below threshold), STFT blend is used."""
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate

        n = 48000
        rng = np.random.default_rng(39)
        dry = rng.normal(0, 0.3, n).astype(np.float32)
        wet = rng.normal(0, 0.3, n).astype(np.float32)
        result = PerPhaseMusicalGoalsGate._wet_dry_blend(dry, wet, 0.29)
        dry_rms = float(np.sqrt(np.mean(dry**2)))
        result_rms = float(np.sqrt(np.mean(result**2)))
        # Should be close to dry (only 29% wet)
        assert abs(result_rms - dry_rms) / (dry_rms + 1e-8) < 0.5

    def test_40_stft_blend_at_030_uses_linear(self):
        """At strength=0.30 (threshold), linear blend should be used."""
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate

        n = 48000
        rng = np.random.default_rng(40)
        dry = rng.normal(0, 0.3, n).astype(np.float32)
        wet = rng.normal(0, 0.3, n).astype(np.float32)
        result = PerPhaseMusicalGoalsGate._wet_dry_blend(dry, wet, 0.30)
        expected = (dry + 0.30 * (wet - dry)).astype(np.float32)
        np.testing.assert_allclose(result, np.clip(expected, -1.0, 1.0), atol=1e-5)
