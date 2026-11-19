from __future__ import annotations

"""Tests for TAPE_HEAD_LEVEL_DIP detection and Phase 12 Tape Level Stabilizer.

Tests:
  - DefectType.TAPE_HEAD_LEVEL_DIP enum exists
  - DefectScanner._detect_tape_head_level_dips() with synthetic dip signals
  - Phase 12 _stabilize_tape_level() repair method
  - CausalDefectReasoner: routing for tape_head_contact_instability
  - Edge-Cases: silence, short files, stereo, NaN/Inf-guards, no-dip passthrough
  - Material gating (only tape/reel_tape/wire_recording trigger detection)
"""


import numpy as np
import pytest

SR = 48_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine(sr: int = SR, duration: float = 5.0, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.asarray(np.sin(2 * np.pi * freq * t) * amp, dtype=np.float32)  # type: ignore[no-any-return]


def _sine_stereo(sr: int = SR, duration: float = 5.0, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    mono = _sine(sr, duration, freq, amp)
    return np.column_stack([mono, mono * 0.9]).astype(np.float32)


def _inject_level_dips(
    audio: np.ndarray,
    sr: int = SR,
    dip_times: list[float] | None = None,
    dip_duration_s: float = 0.2,
    dip_depth_db: float = 15.0,
) -> np.ndarray:
    """Inject gradual level dips into signal (simulates cassette head contact loss).

    Morphology: 80 ms fade-down, hold at minimum, 20 ms snap-back.
    """
    if dip_times is None:
        dip_times = [1.0, 2.8, 4.0]
    result = audio.copy()
    is_stereo = audio.ndim == 2
    dip_samples = int(dip_duration_s * sr)
    fade_down = int(0.080 * sr)  # 80 ms gradual fade
    snap_back = int(0.020 * sr)  # 20 ms sharp recovery
    hold = max(1, dip_samples - fade_down - snap_back)
    linear_depth = 10.0 ** (-dip_depth_db / 20.0)

    for t in dip_times:
        start = int(t * sr)
        if start + dip_samples > audio.shape[0]:
            continue
        # Build gain envelope: 1.0 → linear_depth → 1.0
        env = np.ones(dip_samples, dtype=np.float64)
        # Fade down (Hanning-shaped)
        env[:fade_down] = 1.0 - (1.0 - linear_depth) * (0.5 - 0.5 * np.cos(np.pi * np.arange(fade_down) / fade_down))
        # Hold at minimum
        env[fade_down : fade_down + hold] = linear_depth
        # Snap back (linear ramp)
        env[fade_down + hold :] = np.linspace(linear_depth, 1.0, snap_back)
        if is_stereo:
            result[start : start + dip_samples, 0] *= env.astype(np.float32)
            result[start : start + dip_samples, 1] *= env.astype(np.float32)
        else:
            result[start : start + dip_samples] *= env.astype(np.float32)
    return result


# ===========================================================================
# Class 1: DefectType Enum
# ===========================================================================


@pytest.mark.unit
class TestDefectTypeEnum:
    def test_01_tape_head_level_dip_exists(self):
        from backend.core.defect_scanner import DefectType

        assert hasattr(DefectType, "TAPE_HEAD_LEVEL_DIP")

    def test_02_defect_type_value(self):
        from backend.core.defect_scanner import DefectType

        assert DefectType.TAPE_HEAD_LEVEL_DIP.value == "tape_head_level_dip"

    def test_03_defect_type_count_ge_32(self):
        from backend.core.defect_scanner import DefectType

        assert len(DefectType) >= 32


# ===========================================================================
# Class 2: DefectScanner Detection
# ===========================================================================


class TestDefectScannerDetection:
    def test_10_clean_signal_no_dips(self):
        """Clean sine wave should have zero severity."""
        from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

        scanner = DefectScanner.__new__(DefectScanner)
        scanner.sample_rate = SR
        scanner.material_type = MaterialType.TAPE
        audio = _sine(duration=10.0)
        result = scanner._detect_tape_head_level_dips(audio)
        assert result.defect_type == DefectType.TAPE_HEAD_LEVEL_DIP
        assert result.severity < 0.05

    def test_11_synthetic_dips_detected(self):
        """Synthetic 15 dB dips at known times should be detected."""
        from backend.core.defect_scanner import DefectScanner, MaterialType

        scanner = DefectScanner.__new__(DefectScanner)
        scanner.sample_rate = SR
        scanner.material_type = MaterialType.TAPE
        audio = _sine(duration=10.0, amp=0.3)
        dipped = _inject_level_dips(audio, dip_times=[1.0, 3.0, 5.0, 7.0], dip_depth_db=15.0)
        result = scanner._detect_tape_head_level_dips(dipped)
        assert result.severity > 0.10, f"Expected severity > 0.10, got {result.severity}"
        assert result.metadata["dip_count"] >= 3

    def test_12_severity_scales_with_dip_count(self):
        """More dips should yield higher severity."""
        from backend.core.defect_scanner import DefectScanner, MaterialType

        scanner = DefectScanner.__new__(DefectScanner)
        scanner.sample_rate = SR
        scanner.material_type = MaterialType.TAPE
        audio = _sine(duration=15.0, amp=0.3)
        few_dips = _inject_level_dips(audio, dip_times=[2.0, 6.0], dip_depth_db=12.0)
        many_dips = _inject_level_dips(
            audio, dip_times=[1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0, 11.5, 13.0], dip_depth_db=12.0
        )
        sev_few = scanner._detect_tape_head_level_dips(few_dips).severity
        sev_many = scanner._detect_tape_head_level_dips(many_dips).severity
        assert sev_many > sev_few

    def test_13_very_short_audio(self):
        """Short audio (< 1 s) should return zero severity without error."""
        from backend.core.defect_scanner import DefectScanner, MaterialType

        scanner = DefectScanner.__new__(DefectScanner)
        scanner.sample_rate = SR
        scanner.material_type = MaterialType.TAPE
        audio = _sine(duration=0.5)
        result = scanner._detect_tape_head_level_dips(audio)
        assert result.severity == 0.0

    def test_14_silence_not_detected_as_dip(self):
        """Near-silence regions (< -55 dBFS) should not trigger detection."""
        from backend.core.defect_scanner import DefectScanner, MaterialType

        scanner = DefectScanner.__new__(DefectScanner)
        scanner.sample_rate = SR
        scanner.material_type = MaterialType.TAPE
        # Very quiet signal — all below -55 dBFS
        audio = _sine(duration=5.0, amp=0.001)
        result = scanner._detect_tape_head_level_dips(audio)
        assert result.severity < 0.05

    def test_15_locations_within_audio_range(self):
        """Detected locations must be within [0, duration]."""
        from backend.core.defect_scanner import DefectScanner, MaterialType

        scanner = DefectScanner.__new__(DefectScanner)
        scanner.sample_rate = SR
        scanner.material_type = MaterialType.TAPE
        duration = 10.0
        audio = _sine(duration=duration, amp=0.3)
        dipped = _inject_level_dips(audio, dip_times=[2.0, 5.0, 8.0], dip_depth_db=18.0)
        result = scanner._detect_tape_head_level_dips(dipped)
        for start_s, end_s in result.locations:
            assert start_s >= 0.0
            assert end_s <= duration + 0.1  # small tolerance for framing

    def test_16_dense_dips_over_1000_events_are_not_capped(self):
        """Dense tape dip streams must preserve full core location lists (>1000)."""
        from backend.core.defect_scanner import DefectScanner, MaterialType

        scanner = DefectScanner.__new__(DefectScanner)
        scanner.sample_rate = SR
        scanner.material_type = MaterialType.TAPE

        duration = 180.0
        audio = _sine(duration=duration, amp=0.35)

        # 1100 dips, evenly spaced, non-overlapping.
        n_events = 1100
        dip_times = [0.5 + i * 0.16 for i in range(n_events)]
        dip_times = [t for t in dip_times if (t + 0.06) < duration]
        dipped = _inject_level_dips(
            audio,
            dip_times=dip_times,
            dip_duration_s=0.12,
            dip_depth_db=14.0,
        )

        result = scanner._detect_tape_head_level_dips(dipped)
        assert result.metadata.get("dip_count", 0) > 1000
        assert len(result.locations) > 1000
        assert result.metadata.get("locations_returned", 0) == len(result.locations)


# ===========================================================================
# Class 3: Phase 12 Tape Level Stabilizer
# ===========================================================================


class TestTapeLevelStabilizer:
    def _get_phase(self):
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        return WowFlutterFix()

    def test_20_method_exists(self):
        phase = self._get_phase()
        assert hasattr(phase, "_stabilize_tape_level")

    def test_21_clean_signal_passthrough(self):
        """Clean signal should pass through unchanged."""
        phase = self._get_phase()
        audio = _sine(duration=5.0)
        result, n_dips = phase._stabilize_tape_level(audio, SR, 1.0)
        assert result.shape == audio.shape
        # Should detect 0 or very few dips in clean signal
        assert n_dips <= 1
        # Output should be close to input (max gain ~0 dB)
        rms_orig = float(np.sqrt(np.mean(audio**2)))
        rms_result = float(np.sqrt(np.mean(result**2)))
        assert abs(rms_result - rms_orig) / (rms_orig + 1e-10) < 0.1

    def test_22_dips_are_repaired(self):
        """Injected dips should be leveled out."""
        phase = self._get_phase()
        audio = _sine(duration=10.0, amp=0.3)
        dipped = _inject_level_dips(audio, dip_times=[2.0, 5.0, 8.0], dip_depth_db=12.0)
        result, n_dips = phase._stabilize_tape_level(dipped, SR, 1.0)
        assert n_dips >= 2, f"Expected >= 2 repaired dips, got {n_dips}"
        # Check that dip regions are now closer to original level
        for t in [2.0, 5.0, 8.0]:
            s = int(t * SR)
            e = min(s + int(0.2 * SR), result.shape[0])
            rms_orig = float(np.sqrt(np.mean(audio[s:e] ** 2)))
            rms_dipped = float(np.sqrt(np.mean(dipped[s:e] ** 2)))
            rms_repaired = float(np.sqrt(np.mean(result[s:e] ** 2)))
            # Repaired should be closer to original than dipped is
            if e <= result.shape[0]:
                assert rms_repaired > rms_dipped, f"At t={t}s: repaired RMS should exceed dipped RMS"

    def test_22b_repaired_dips_are_not_audibly_residual(self):
        """Counted tape-dip repairs must reduce residual event loss below a clearly audible gap."""
        phase = self._get_phase()
        audio = _sine(duration=10.0, amp=0.3)
        dipped = _inject_level_dips(audio, dip_times=[2.0, 5.0, 8.0], dip_depth_db=15.0)

        result, n_dips = phase._stabilize_tape_level(dipped, SR, 0.35, is_primary_tape=True)

        assert n_dips >= 3
        for t in [2.0, 5.0, 8.0]:
            s = int((t + 0.09) * SR)
            e = s + int(0.05 * SR)
            rms_orig = float(np.sqrt(np.mean(audio[s:e] ** 2)))
            rms_repaired = float(np.sqrt(np.mean(result[s:e] ** 2)))
            residual_loss_db = 20.0 * np.log10((rms_repaired + 1e-15) / (rms_orig + 1e-15))
            assert residual_loss_db >= -2.0, f"At t={t}s residual dip is still {residual_loss_db:.2f} dB"

    def test_22c_confirmed_chain_tape_dips_are_not_left_barely_touched(self):
        """Confirmed tape-stage dips in transferred material need stronger repair than generic dynamics."""
        phase = self._get_phase()
        audio = _sine(duration=8.0, amp=0.3)
        dipped = _inject_level_dips(audio, dip_times=[2.0, 5.0], dip_depth_db=15.0)

        result, n_dips = phase._stabilize_tape_level(
            dipped,
            SR,
            0.35,
            is_primary_tape=False,
            confirmed_tape_dip=True,
        )

        assert n_dips >= 2
        for t in [2.0, 5.0]:
            s = int((t + 0.09) * SR)
            e = s + int(0.05 * SR)
            rms_orig = float(np.sqrt(np.mean(audio[s:e] ** 2)))
            rms_repaired = float(np.sqrt(np.mean(result[s:e] ** 2)))
            residual_loss_db = 20.0 * np.log10((rms_repaired + 1e-15) / (rms_orig + 1e-15))
            assert residual_loss_db >= -3.0, f"At t={t}s confirmed chain dip remains {residual_loss_db:.2f} dB"

    def test_22d_per_event_oracle_repairs_deeper_dips_more_strongly(self):
        """Deep and shallow tape-level defects must not receive a uniform correction."""
        phase = self._get_phase()
        audio = _sine(duration=8.0, amp=0.3)
        shallow = _inject_level_dips(audio, dip_times=[2.0], dip_depth_db=7.0)
        mixed = _inject_level_dips(shallow, dip_times=[5.0], dip_depth_db=17.0)

        result, n_dips = phase._stabilize_tape_level(mixed, SR, 0.35, is_primary_tape=True)

        assert n_dips >= 2
        shallow_s = int((2.0 + 0.09) * SR)
        deep_s = int((5.0 + 0.09) * SR)
        window = int(0.05 * SR)
        shallow_gain_db = 20.0 * np.log10(
            (float(np.sqrt(np.mean(result[shallow_s : shallow_s + window] ** 2))) + 1e-15)
            / (float(np.sqrt(np.mean(mixed[shallow_s : shallow_s + window] ** 2))) + 1e-15)
        )
        deep_gain_db = 20.0 * np.log10(
            (float(np.sqrt(np.mean(result[deep_s : deep_s + window] ** 2))) + 1e-15)
            / (float(np.sqrt(np.mean(mixed[deep_s : deep_s + window] ** 2))) + 1e-15)
        )
        assert deep_gain_db > shallow_gain_db + 4.0

    def test_22e_vibrato_zone_caps_level_dip_repair_strength(self):
        """Vocal-protected zones must cap tape-level repair instead of forcing a uniform heavy boost."""
        phase = self._get_phase()
        audio = _sine(duration=5.0, amp=0.3)
        dipped = _inject_level_dips(audio, dip_times=[2.0], dip_depth_db=15.0)

        free_result, _ = phase._stabilize_tape_level(dipped, SR, 0.35, is_primary_tape=True)
        capped_result, n_dips = phase._stabilize_tape_level(
            dipped,
            SR,
            0.35,
            is_primary_tape=True,
            protected_zones=[(1.8, 2.4, 0.20)],
        )

        assert n_dips >= 1
        s = int((2.0 + 0.09) * SR)
        e = s + int(0.05 * SR)
        free_gain_db = 20.0 * np.log10(
            (float(np.sqrt(np.mean(free_result[s:e] ** 2))) + 1e-15)
            / (float(np.sqrt(np.mean(dipped[s:e] ** 2))) + 1e-15)
        )
        capped_gain_db = 20.0 * np.log10(
            (float(np.sqrt(np.mean(capped_result[s:e] ** 2))) + 1e-15)
            / (float(np.sqrt(np.mean(dipped[s:e] ** 2))) + 1e-15)
        )
        assert capped_gain_db < free_gain_db - 6.0

    def test_23_output_shape_mono(self):
        """Output shape must match input for mono."""
        phase = self._get_phase()
        audio = _sine(duration=3.0)
        result, _ = phase._stabilize_tape_level(audio, SR, 0.8)
        assert result.shape == audio.shape
        assert result.dtype == np.float32

    def test_24_output_shape_stereo(self):
        """Output shape must match input for stereo."""
        phase = self._get_phase()
        audio = _sine_stereo(duration=3.0)
        dipped = _inject_level_dips(audio, dip_times=[1.0], dip_depth_db=10.0)
        result, _ = phase._stabilize_tape_level(dipped, SR, 0.8)
        assert result.shape == audio.shape

    def test_24b_channels_first_stereo_is_repaired_without_axis_collapse(self):
        """UV3 liefert Stereo als (2, N); der Stabilizer darf daraus kein 2-Sample-STFT machen."""
        phase = self._get_phase()
        audio = _sine_stereo(duration=4.0, amp=0.3)
        dipped = _inject_level_dips(audio, dip_times=[1.2], dip_depth_db=14.0)
        dipped_cf = dipped.T.copy()

        result, n_dips = phase._stabilize_tape_level(dipped_cf, SR, 0.8)

        assert n_dips >= 1
        assert result.shape == dipped_cf.shape
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
        s = int(1.29 * SR)
        e = s + int(0.04 * SR)
        rms_dipped = float(np.sqrt(np.mean(dipped_cf[0, s:e] ** 2)))
        rms_repaired = float(np.sqrt(np.mean(result[0, s:e] ** 2)))
        assert rms_repaired > rms_dipped

    def test_25_nan_inf_guard(self):
        """NaN/Inf in input must not propagate to output."""
        phase = self._get_phase()
        audio = _sine(duration=3.0)
        audio[1000] = np.nan
        audio[2000] = np.inf
        audio[3000] = -np.inf
        result, _ = phase._stabilize_tape_level(audio, SR, 1.0)
        assert np.isfinite(result).all()

    def test_26_output_clipped(self):
        """Output must be clipped to [-1, 1]."""
        phase = self._get_phase()
        audio = _sine(duration=5.0, amp=0.9)
        dipped = _inject_level_dips(audio, dip_times=[1.0], dip_depth_db=20.0)
        result, _ = phase._stabilize_tape_level(dipped, SR, 1.0)
        assert float(np.max(np.abs(result))) <= 1.0

    def test_27_zero_strength_passthrough(self):
        """strength=0 should return audio unchanged."""
        phase = self._get_phase()
        audio = _sine(duration=3.0)
        dipped = _inject_level_dips(audio, dip_times=[1.0], dip_depth_db=15.0)
        result, n_dips = phase._stabilize_tape_level(dipped, SR, 0.0)
        assert n_dips == 0
        np.testing.assert_array_equal(result, dipped)

    def test_28_max_gain_limited(self):
        """Gain should not exceed max_gain_db (15 dB) even for deep dips."""
        phase = self._get_phase()
        audio = _sine(duration=5.0, amp=0.4)
        # Inject a very deep dip (25 dB) — should be capped at 15 dB gain
        dipped = _inject_level_dips(audio, dip_times=[2.0], dip_depth_db=25.0)
        result, _ = phase._stabilize_tape_level(dipped, SR, 1.0)
        # The repaired region should be louder than dipped but not exceed max gain
        s = int(2.1 * SR)
        e = s + int(0.05 * SR)
        rms_dipped = float(np.sqrt(np.mean(dipped[s:e] ** 2)))
        rms_result = float(np.sqrt(np.mean(result[s:e] ** 2)))
        gain_applied_db = 20.0 * np.log10((rms_result + 1e-15) / (rms_dipped + 1e-15))
        assert gain_applied_db <= 16.0  # allow 1 dB margin above 15 dB max

    def test_29_very_short_audio(self):
        """Very short audio should return unchanged without error."""
        phase = self._get_phase()
        audio = np.zeros(100, dtype=np.float32)
        result, n_dips = phase._stabilize_tape_level(audio, SR, 1.0)
        assert n_dips == 0
        assert result.shape == audio.shape

    def test_30_strength_scaling(self):
        """Higher strength should produce more gain correction."""
        phase = self._get_phase()
        audio = _sine(duration=10.0, amp=0.3)
        dipped = _inject_level_dips(audio, dip_times=[2.0, 5.0], dip_depth_db=12.0)
        result_low, _ = phase._stabilize_tape_level(dipped.copy(), SR, 0.3)
        result_high, _ = phase._stabilize_tape_level(dipped.copy(), SR, 1.0)
        # At dip location, high strength should yield louder signal
        s = int(2.1 * SR)
        e = s + int(0.05 * SR)
        rms_low = float(np.sqrt(np.mean(result_low[s:e] ** 2)))
        rms_high = float(np.sqrt(np.mean(result_high[s:e] ** 2)))
        assert rms_high >= rms_low

    def test_31_primary_tape_start_dip_is_repaired(self):
        """Primary tape/cassette dips near song start must not be hidden by intro protection."""
        phase = self._get_phase()
        audio = _sine(duration=6.0, amp=0.3)
        dipped = _inject_level_dips(audio, dip_times=[1.2], dip_duration_s=0.18, dip_depth_db=14.0)

        result, n_dips = phase._stabilize_tape_level(dipped, SR, 0.35, is_primary_tape=True)

        assert n_dips >= 1
        s = int(1.29 * SR)
        e = s + int(0.04 * SR)
        rms_dipped = float(np.sqrt(np.mean(dipped[s:e] ** 2)))
        rms_repaired = float(np.sqrt(np.mean(result[s:e] ** 2)))
        gain_applied_db = 20.0 * np.log10((rms_repaired + 1e-15) / (rms_dipped + 1e-15))
        assert gain_applied_db >= 2.0


# ===========================================================================
# Class 4: CausalDefectReasoner Routing
# ===========================================================================


class TestCausalReasonerRouting:
    def test_40_cause_exists(self):
        from backend.core.causal_defect_reasoner import CAUSES

        assert "tape_head_contact_instability" in CAUSES

    def test_41_cause_to_phases_exists(self):
        from backend.core.causal_defect_reasoner import CAUSE_TO_PHASES

        assert "tape_head_contact_instability" in CAUSE_TO_PHASES
        phases = CAUSE_TO_PHASES["tape_head_contact_instability"]
        assert "phase_12_wow_flutter_fix" in phases

    def test_42_likelihood_fn_exists(self):
        from backend.core.causal_defect_reasoner import LIKELIHOOD_FNS

        assert "tape_head_contact_instability" in LIKELIHOOD_FNS

    def test_43_tape_material_prior_is_strong(self):
        from backend.core.causal_defect_reasoner import MATERIAL_PRIORS

        # Tape should have a notably lower prior (= higher probability)
        assert MATERIAL_PRIORS["tape"]["tape_head_contact_instability"] < 0.5

    def test_44_digital_material_prior_neutral(self):
        from backend.core.causal_defect_reasoner import MATERIAL_PRIORS

        # Digital material has no tape head → near-zero prior (N/A)
        assert MATERIAL_PRIORS["digital"]["tape_head_contact_instability"] <= 0.01


# ===========================================================================
# Class 5: Integration — Phase 12 process() with tape material
# ===========================================================================


class TestPhase12Integration:
    def test_50_process_tape_with_dips(self):
        """Phase 12 process() on tape material with dips should report tape_level_dips_repaired."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        phase = WowFlutterFix()
        audio = _sine(duration=10.0, amp=0.3)
        dipped = _inject_level_dips(audio, dip_times=[2.0, 5.0, 8.0], dip_depth_db=15.0)
        result = phase.process(dipped, SR, material=MaterialType.TAPE)
        assert result.success
        assert "tape_level_dips_repaired" in result.metrics
        assert result.metrics["tape_level_dips_repaired"] >= 0

    def test_50b_process_cassette_channels_first_preserves_shape_and_repairs_dips(self):
        """Full process() must handle UV3 stereo layout (2, N) for cassette tape defects."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        phase = WowFlutterFix()
        audio = _sine_stereo(duration=4.0, amp=0.3)
        dipped = _inject_level_dips(audio, dip_times=[1.2], dip_depth_db=14.0)
        dipped_cf = dipped.T.copy()

        result = phase.process(dipped_cf, SR, material=MaterialType.CASSETTE, strength=0.35, use_ml_hybrid=False)

        assert result.success
        assert result.audio.shape == dipped_cf.shape
        assert np.isfinite(result.audio).all()
        assert result.metrics.get("tape_level_dips_repaired", 0) >= 1

    def test_51_process_cd_digital_no_stabilizer(self):
        """Phase 12 on CD_DIGITAL should NOT run tape level stabilizer."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        phase = WowFlutterFix()
        audio = _sine(duration=5.0, amp=0.3)
        dipped = _inject_level_dips(audio, dip_times=[2.0], dip_depth_db=15.0)
        result = phase.process(dipped, SR, material=MaterialType.CD_DIGITAL)
        assert result.success
        assert result.metrics.get("tape_level_dips_repaired", 0) == 0

    def test_52_output_finite_and_clipped(self):
        """Output from full process must be finite and clipped."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        phase = WowFlutterFix()
        audio = _sine(duration=5.0, amp=0.3)
        dipped = _inject_level_dips(audio, dip_times=[1.5, 3.0], dip_depth_db=12.0)
        result = phase.process(dipped, SR, material=MaterialType.TAPE)
        assert np.isfinite(result.audio).all()
        assert float(np.max(np.abs(result.audio))) <= 1.0


# ===========================================================================
# Class 6: Cross-Material Gate — _should_keep_cross_material_tape_head_level_dip
# ===========================================================================


class TestCrossMaterialGate:
    """§9.1a Cross-Material-Fallback: starke Dip-Morphologie muss auch bei
    Fehlklassifikation (z.B. Vinyl statt Tape) durchgelassen werden."""

    @pytest.fixture(autouse=True)
    def _disable_crepe(self, monkeypatch):
        monkeypatch.setenv("AURIK_DISABLE_CREPE", "1")

    # -- Hilfsmethode direkt (kein Scanner-Scan nötig) ---------------------

    def test_60_method_exists_on_scanner(self):
        """Die Methode muss direkt auf DefectScanner verfügbar sein."""
        from backend.core.defect_scanner import DefectScanner

        assert hasattr(DefectScanner, "_should_keep_cross_material_tape_head_level_dip")

    def test_61_strong_morphology_passes(self):
        """Severity ≥ 0.12, count ≥ 2, depth ≥ 6 dB, rate ≥ 0.15 → True."""
        from backend.core.defect_scanner import DefectScanner, DefectScore, DefectType

        score = DefectScore(DefectType.TAPE_HEAD_LEVEL_DIP, 0.45, 0.80)
        score.metadata = {
            "dip_count": 5,
            "mean_depth_db": 12.0,
            "event_rate_per_s": 0.62,
        }
        assert DefectScanner._should_keep_cross_material_tape_head_level_dip(score) is True

    def test_62_weak_severity_blocked(self):
        """Severity < 0.12 → immer False, auch bei starker Morphologie."""
        from backend.core.defect_scanner import DefectScanner, DefectScore, DefectType

        score = DefectScore(DefectType.TAPE_HEAD_LEVEL_DIP, 0.08, 0.70)
        score.metadata = {
            "dip_count": 6,
            "mean_depth_db": 14.0,
            "event_rate_per_s": 0.70,
        }
        assert DefectScanner._should_keep_cross_material_tape_head_level_dip(score) is False

    def test_63_single_event_blocked(self):
        """dip_count < 2 → False (einzelner Lautstärke-Einbruch, kein periodisches Muster)."""
        from backend.core.defect_scanner import DefectScanner, DefectScore, DefectType

        score = DefectScore(DefectType.TAPE_HEAD_LEVEL_DIP, 0.30, 0.80)
        score.metadata = {
            "dip_count": 1,
            "mean_depth_db": 15.0,
            "event_rate_per_s": 0.50,
        }
        assert DefectScanner._should_keep_cross_material_tape_head_level_dip(score) is False

    def test_64_shallow_dips_blocked(self):
        """mean_depth_db < 6.0 → False (kein hörbarer Kopfkontakt-Einbruch)."""
        from backend.core.defect_scanner import DefectScanner, DefectScore, DefectType

        score = DefectScore(DefectType.TAPE_HEAD_LEVEL_DIP, 0.25, 0.75)
        score.metadata = {
            "dip_count": 4,
            "mean_depth_db": 3.0,
            "event_rate_per_s": 0.40,
        }
        assert DefectScanner._should_keep_cross_material_tape_head_level_dip(score) is False

    def test_65_low_event_rate_blocked(self):
        """event_rate_per_s < 0.15 → False (zu selten für periodischen Kopf-Kontakt)."""
        from backend.core.defect_scanner import DefectScanner, DefectScore, DefectType

        score = DefectScore(DefectType.TAPE_HEAD_LEVEL_DIP, 0.20, 0.75)
        score.metadata = {
            "dip_count": 3,
            "mean_depth_db": 10.0,
            "event_rate_per_s": 0.08,
        }
        assert DefectScanner._should_keep_cross_material_tape_head_level_dip(score) is False

    # -- Integration mit vollständigem Scanner-Scan ------------------------

    def test_66_vinyl_with_strong_dips_uses_fallback(self):
        """Starke periodische Dips auf Vinyl → cross_material_fallback=True, severity > 0."""
        from backend.core.defect_scanner import DefectScanner, MaterialType

        sr = SR
        dur = 10.0
        n = int(sr * dur)
        t = np.linspace(0, dur, n, endpoint=False, dtype=np.float32)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        # 7 starke periodische Dips (~15 dB)
        for start_s in np.arange(1.0, dur - 0.5, 1.2):
            audio = _inject_level_dips(audio, dip_times=[float(start_s)], dip_depth_db=15.0)

        scanner = DefectScanner.__new__(DefectScanner)
        scanner.sample_rate = sr
        scanner.material_type = MaterialType.TAPE  # für _detect_tape_head_level_dips
        raw = scanner._detect_tape_head_level_dips(audio)

        # Prüfe, ob der Fallback-Guard greifen würde
        should_keep = DefectScanner._should_keep_cross_material_tape_head_level_dip(raw)
        assert should_keep is True, (
            f"Erwartet: cross-material-Fallback greift, aber: "
            f"sev={raw.severity:.3f} dip_count={raw.metadata.get('dip_count')} "
            f"depth={raw.metadata.get('mean_depth_db'):.1f} "
            f"rate={raw.metadata.get('event_rate_per_s'):.3f}"
        )

    def test_67_clean_vinyl_not_affected(self):
        """Sauberes Vinyl-Signal → Severity bleibt 0, kein Fallback."""
        from backend.core.defect_scanner import DefectScanner, MaterialType

        sr = SR
        n = int(sr * 8.0)
        t = np.linspace(0, 8.0, n, endpoint=False, dtype=np.float32)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        scanner = DefectScanner.__new__(DefectScanner)
        scanner.sample_rate = sr
        scanner.material_type = MaterialType.VINYL
        raw = scanner._detect_tape_head_level_dips(audio)

        assert raw.severity < 0.05
        assert DefectScanner._should_keep_cross_material_tape_head_level_dip(raw) is False

    def test_68_full_scan_vinyl_cross_material_severity_nonzero(self):
        """Vollständiger scan() auf Vinyl mit starken Dips → TAPE_HEAD_LEVEL_DIP > 0."""
        from backend.core.defect_scanner import DefectType, MaterialType, get_defect_scanner

        sr = SR
        dur = 10.0
        n = int(sr * dur)
        t = np.linspace(0, dur, n, endpoint=False, dtype=np.float32)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        for start_s in np.arange(1.0, dur - 0.5, 1.2):
            audio = _inject_level_dips(audio, dip_times=[float(start_s)], dip_depth_db=15.0)

        scanner = get_defect_scanner(sr)
        result = scanner.scan(audio, sr, MaterialType.VINYL)
        sev = result.scores[DefectType.TAPE_HEAD_LEVEL_DIP].severity
        meta = result.scores[DefectType.TAPE_HEAD_LEVEL_DIP].metadata

        assert sev > 0.0, f"Erwartet: cross-material-Fallback setzt severity > 0, got {sev}"
        assert meta.get("cross_material_fallback") is True

    def test_69_full_scan_clean_vinyl_severity_zero(self):
        """Vollständiger scan() auf sauberem Vinyl → TAPE_HEAD_LEVEL_DIP bleibt 0."""
        from backend.core.defect_scanner import DefectType, MaterialType, get_defect_scanner

        sr = SR
        n = int(sr * 6.0)
        t = np.linspace(0, 6.0, n, endpoint=False, dtype=np.float32)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        scanner = get_defect_scanner(sr)
        result = scanner.scan(audio, sr, MaterialType.VINYL)
        sev = result.scores[DefectType.TAPE_HEAD_LEVEL_DIP].severity
        assert sev == 0.0, f"False Positive: sauberes Vinyl → severity={sev:.4f}"
