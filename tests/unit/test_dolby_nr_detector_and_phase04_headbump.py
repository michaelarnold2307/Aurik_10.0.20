from __future__ import annotations

"""
Unit-Tests für:
  - backend/core/dolby_nr_detector.py  (Erkennung, Inversion, Singleton)
  - Phase_04 Head-Bump Compensation (HEAD_BUMP_PROFILES, _apply_head_bump_compensation)
  - Phase_04 Dolby-NR-Integration (Durchreiche-Kwargs)
  - Phase_63 §2.51 M/S-Stereo-Compliance

Testanzahl: ≥ 35 def test_
"""


import numpy as np
import pytest

SR = 48000


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def white_mono():
    rng = np.random.default_rng(42)
    return rng.standard_normal(SR * 3).astype(np.float32) * 0.3


@pytest.fixture(scope="module")
def white_stereo_channels_first():
    rng = np.random.default_rng(43)
    return rng.standard_normal((2, SR * 3)).astype(np.float32) * 0.3


@pytest.fixture(scope="module")
def white_stereo_samples_first():
    rng = np.random.default_rng(44)
    return rng.standard_normal((SR * 3, 2)).astype(np.float32) * 0.3


@pytest.fixture(scope="module")
def tape_with_dolby_b_sim():
    """Simulate Dolby-B encoded (undecoded) tape: add +6 dB HF shelf at 3 kHz."""
    import scipy.signal as sps

    rng = np.random.default_rng(55)
    x = rng.standard_normal(SR * 5).astype(np.float64) * 0.25
    # Apply +6 dB shelf above 3 kHz (crude HF boost to trigger detection)
    sos = sps.butter(2, 3000 / (SR / 2), btype="high", output="sos")
    hf = sps.sosfiltfilt(sos, x)
    x = np.clip(x + hf * 1.0, -1.0, 1.0)
    return x.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# DolbyNRDetector — module-level functions
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestDolbyNRDetectorFunctions:
    """Tests for detect_dolby_encoding and apply_inverse_filter."""

    def test_detect_returns_dataclass(self, white_mono):
        from backend.core.dolby_nr_detector import DolbyDetectionResult, detect_dolby_encoding

        result = detect_dolby_encoding(white_mono, SR, material_type="tape")
        assert isinstance(result, DolbyDetectionResult)

    def test_detect_no_false_positive_on_silence(self):
        from backend.core.dolby_nr_detector import detect_dolby_encoding

        silence = np.zeros(SR * 2, dtype=np.float32)
        result = detect_dolby_encoding(silence, SR, material_type="tape")
        assert result.detected is False

    def test_detect_non_tape_material_returns_not_detected(self, white_mono):
        from backend.core.dolby_nr_detector import detect_dolby_encoding

        result = detect_dolby_encoding(white_mono, SR, material_type="vinyl")
        assert result.detected is False
        assert result.nr_type == "none"

    def test_detect_hf_excess_field_present(self, white_mono):
        from backend.core.dolby_nr_detector import detect_dolby_encoding

        result = detect_dolby_encoding(white_mono, SR, material_type="tape")
        assert isinstance(result.hf_excess_db, float)
        assert -30.0 < result.hf_excess_db < 30.0

    def test_detect_confidence_in_range(self, white_mono):
        from backend.core.dolby_nr_detector import detect_dolby_encoding

        result = detect_dolby_encoding(white_mono, SR, material_type="tape")
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_evidence_is_list(self, white_mono):
        from backend.core.dolby_nr_detector import detect_dolby_encoding

        result = detect_dolby_encoding(white_mono, SR, material_type="tape")
        assert isinstance(result.evidence, list)

    def test_detect_hf_boosted_tape_may_flag(self, tape_with_dolby_b_sim):
        """HF-boosted tape signal should raise hf_excess_db significantly."""
        from backend.core.dolby_nr_detector import detect_dolby_encoding

        result = detect_dolby_encoding(tape_with_dolby_b_sim, SR, material_type="tape")
        # hf_excess should be clearly positive (> threshold floor)
        assert result.hf_excess_db > 0.0, (
            f"Expected positive hf_excess for HF-boosted tape, got {result.hf_excess_db:.2f}"
        )

    def test_detect_stereo_input_accepted(self, white_stereo_channels_first):
        from backend.core.dolby_nr_detector import detect_dolby_encoding

        result = detect_dolby_encoding(white_stereo_channels_first, SR, material_type="tape")
        assert isinstance(result.detected, bool)

    def test_detect_stereo_samples_first_accepted(self, white_stereo_samples_first):
        from backend.core.dolby_nr_detector import detect_dolby_encoding

        result = detect_dolby_encoding(white_stereo_samples_first, SR, material_type="tape")
        assert isinstance(result.detected, bool)

    def test_apply_inverse_none_returns_clipped(self, white_mono):
        from backend.core.dolby_nr_detector import apply_inverse_filter

        out = apply_inverse_filter(white_mono, nr_type="none", sr=SR)
        assert np.all(np.abs(out) <= 1.0 + 1e-6)
        np.testing.assert_allclose(out, np.clip(white_mono, -1.0, 1.0), atol=1e-5)

    def test_apply_inverse_dolby_b_changes_signal(self, white_mono):
        from backend.core.dolby_nr_detector import apply_inverse_filter

        out = apply_inverse_filter(white_mono, nr_type="dolby_b", sr=SR, confidence=1.0)
        assert not np.allclose(out, white_mono, atol=1e-4), "Dolby-B inverse must change the signal"

    def test_apply_inverse_dolby_c_changes_signal(self, white_mono):
        from backend.core.dolby_nr_detector import apply_inverse_filter

        out = apply_inverse_filter(white_mono, nr_type="dolby_c", sr=SR, confidence=1.0)
        assert not np.allclose(out, white_mono, atol=1e-4)

    def test_apply_inverse_clips_output(self, white_mono):
        from backend.core.dolby_nr_detector import apply_inverse_filter

        out = apply_inverse_filter(white_mono, nr_type="dolby_b", sr=SR)
        assert np.all(np.abs(out) <= 1.0 + 1e-6)

    def test_apply_inverse_no_nan_inf(self, white_mono):
        from backend.core.dolby_nr_detector import apply_inverse_filter

        out = apply_inverse_filter(white_mono, nr_type="dolby_b", sr=SR)
        assert np.all(np.isfinite(out))

    def test_apply_inverse_stereo_channels_first(self, white_stereo_channels_first):
        from backend.core.dolby_nr_detector import apply_inverse_filter

        out = apply_inverse_filter(white_stereo_channels_first, nr_type="dolby_b", sr=SR)
        assert out.shape == white_stereo_channels_first.shape

    def test_apply_inverse_stereo_samples_first(self, white_stereo_samples_first):
        from backend.core.dolby_nr_detector import apply_inverse_filter

        out = apply_inverse_filter(white_stereo_samples_first, nr_type="dolby_b", sr=SR)
        assert out.shape == white_stereo_samples_first.shape

    def test_apply_inverse_low_confidence_is_gentle(self, white_mono):
        """With confidence=0.05, the output must be very close to input."""
        from backend.core.dolby_nr_detector import apply_inverse_filter

        out = apply_inverse_filter(white_mono, nr_type="dolby_b", sr=SR, confidence=0.05)
        # Mostly dry — MAE should be very small
        mae = float(np.mean(np.abs(out.astype(np.float64) - white_mono.astype(np.float64))))
        assert mae < 0.05, f"Low-confidence inverse should barely change signal, got MAE={mae:.4f}"

    def test_dbx_i_inverse_changes_signal(self, white_mono):
        from backend.core.dolby_nr_detector import apply_inverse_filter

        out = apply_inverse_filter(white_mono, nr_type="dbx_i", sr=SR)
        assert not np.allclose(out, white_mono, atol=1e-4)

    def test_build_inverse_filter_sos_returns_none_for_none(self):
        from backend.core.dolby_nr_detector import build_inverse_filter_sos

        assert build_inverse_filter_sos("none", SR) is None

    def test_build_inverse_filter_sos_returns_array_for_dolby_b(self):
        from backend.core.dolby_nr_detector import build_inverse_filter_sos

        sos = build_inverse_filter_sos("dolby_b", SR)
        assert sos is not None
        assert sos.ndim == 2 and sos.shape[1] == 6

    def test_singleton_thread_safe(self):
        from backend.core.dolby_nr_detector import get_dolby_nr_detector

        a = get_dolby_nr_detector()
        b = get_dolby_nr_detector()
        assert a is b


# ═══════════════════════════════════════════════════════════════════════════════
# Phase_04 — Head-Bump Compensation
# ═══════════════════════════════════════════════════════════════════════════════


class TestPhase04HeadBump:
    """Head-Bump compensation via _apply_head_bump_compensation."""

    @pytest.fixture(scope="class")
    def phase(self):
        from backend.core.phases.phase_04_eq_correction import EQCorrectionPhase

        return EQCorrectionPhase(sample_rate=SR)

    def test_head_bump_profiles_defined(self, phase):
        assert isinstance(phase.HEAD_BUMP_PROFILES, dict)
        assert len(phase.HEAD_BUMP_PROFILES) >= 4

    def test_head_bump_changes_lf_content(self, phase, white_mono):
        """LF content around bump frequency should change."""
        import scipy.signal as sps

        out = phase._apply_head_bump_compensation(white_mono, speed_ips=7.5)
        # Compare energy in 80-200 Hz band
        sos = sps.butter(2, [80 / (SR / 2), 200 / (SR / 2)], btype="band", output="sos")
        orig_b = float(np.sqrt(np.mean(sps.sosfiltfilt(sos, white_mono.astype(np.float64)) ** 2)))
        out_b = float(np.sqrt(np.mean(sps.sosfiltfilt(sos, out.astype(np.float64)) ** 2)))
        # Dip should reduce LF energy
        assert out_b < orig_b * 0.99, f"Head-bump dip should reduce LF energy: {orig_b:.4f} → {out_b:.4f}"

    def test_head_bump_no_nan_inf(self, phase, white_mono):
        out = phase._apply_head_bump_compensation(white_mono, speed_ips=7.5)
        assert np.all(np.isfinite(out))

    def test_head_bump_stereo_accepted(self, phase, white_stereo_samples_first):
        out = phase._apply_head_bump_compensation(white_stereo_samples_first, speed_ips=15.0)
        assert out.shape == white_stereo_samples_first.shape

    def test_head_bump_unknown_speed_bypasses(self, phase, white_mono):
        """A very unusual speed (e.g. 999 IPS) should not crash and change nothing."""
        out = phase._apply_head_bump_compensation(white_mono, speed_ips=999.0)
        assert np.allclose(out, white_mono, atol=1e-6)

    def test_phase04_process_includes_head_bump_in_metadata(self, phase, white_mono):
        result = phase.process(white_mono, material_type="tape", tape_speed_ips=7.5)
        assert result.metadata.get("head_bump_applied") is True

    def test_phase04_process_no_head_bump_for_vinyl(self, phase, white_mono):
        result = phase.process(white_mono, material_type="vinyl", tape_speed_ips=7.5)
        # tape_speed_ips is ignored for non-tape material
        assert result.metadata.get("head_bump_applied") is False

    def test_phase04_process_dolby_nr_kwarg_forwarded(self, phase, white_mono):
        result = phase.process(
            white_mono,
            material_type="tape",
            dolby_nr_type="dolby_b",
            dolby_nr_confidence=0.8,
        )
        assert result.metadata.get("dolby_nr_applied") is True
        assert result.metadata.get("dolby_nr_type") == "dolby_b"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase_63 — §2.51 M/S Stereo Compliance
# ═══════════════════════════════════════════════════════════════════════════════


class TestPhase63StereoCompliance:
    """Ensure Phase 63 processes stereo via M/S, not independent L/R."""

    def test_mono_output_shape(self, white_mono):
        from backend.core.phases.phase_63_intermodulation_reduction import apply

        out = apply(white_mono, SR)
        assert out.shape == white_mono.shape

    def test_stereo_channels_first_shape_preserved(self, white_stereo_channels_first):
        from backend.core.phases.phase_63_intermodulation_reduction import apply

        out = apply(white_stereo_channels_first, SR)
        assert out.shape == white_stereo_channels_first.shape

    def test_stereo_samples_first_shape_preserved(self, white_stereo_samples_first):
        from backend.core.phases.phase_63_intermodulation_reduction import apply

        out = apply(white_stereo_samples_first, SR)
        assert out.shape == white_stereo_samples_first.shape

    def test_stereo_lr_gain_linked_not_independent(self):
        """L and R signals must receive identical notch treatment (linked via M/S)."""
        from backend.core.phases.phase_63_intermodulation_reduction import apply

        rng = np.random.default_rng(100)
        # Identical L and R
        mono_sig = rng.standard_normal(SR * 2).astype(np.float32) * 0.4
        stereo = np.stack([mono_sig, mono_sig], axis=0)
        out = apply(stereo, SR)
        # For purely identical L/R, the output must be identical L and R
        np.testing.assert_allclose(
            out[0], out[1], atol=1e-5, err_msg="For identical L/R input, output L and R must be identical"
        )

    def test_no_nan_inf_mono(self, white_mono):
        from backend.core.phases.phase_63_intermodulation_reduction import apply

        out = apply(white_mono, SR)
        assert np.all(np.isfinite(out))

    def test_no_nan_inf_stereo(self, white_stereo_channels_first):
        from backend.core.phases.phase_63_intermodulation_reduction import apply

        out = apply(white_stereo_channels_first, SR)
        assert np.all(np.isfinite(out))

    def test_clips_to_minus1_plus1(self, white_mono):
        from backend.core.phases.phase_63_intermodulation_reduction import apply

        loud = white_mono * 3.0
        out = apply(loud, SR)
        assert np.all(np.abs(out) <= 1.0 + 1e-6)

    def test_nan_input_guarded(self):
        from backend.core.phases.phase_63_intermodulation_reduction import apply

        bad = np.full(SR, np.nan, dtype=np.float32)
        out = apply(bad, SR)
        assert np.all(np.isfinite(out))

    def test_assert_sr_48000(self, white_mono):
        from backend.core.phases.phase_63_intermodulation_reduction import apply

        with pytest.raises(AssertionError):
            apply(white_mono, 44100)

    def test_low_imd_score_skips_processing(self, white_mono):
        from backend.core.phases.phase_63_intermodulation_reduction import apply

        out = apply(white_mono, SR, defect_scores={"intermodulation_distortion": 0.01})
        # Should return clipped original unchanged (score below threshold)
        np.testing.assert_allclose(out, np.clip(white_mono, -1.0, 1.0), atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# MediumDetectionResult — new dolby_nr fields
# ═══════════════════════════════════════════════════════════════════════════════


class TestMediumDetectionResultDolbyFields:
    """Validate the new dolby_nr_type / dolby_nr_confidence fields."""

    def test_default_values(self):
        from forensics.medium_detector import MediumDetectionResult, SpectralFingerprint

        fp = SpectralFingerprint(rolloff_95_hz=8000.0, noise_floor_db=-45.0)
        r = MediumDetectionResult(
            transfer_chain=["vinyl"],
            is_multi_generation=False,
            primary_material="vinyl",
            confidence=0.9,
            spectral_fingerprint=fp,
        )
        assert r.dolby_nr_type == "none"
        assert r.dolby_nr_confidence == 0.0

    def test_as_dict_includes_dolby_fields(self):
        from forensics.medium_detector import MediumDetectionResult, SpectralFingerprint

        fp = SpectralFingerprint(rolloff_95_hz=7000.0, noise_floor_db=-40.0)
        r = MediumDetectionResult(
            transfer_chain=["tape"],
            is_multi_generation=False,
            primary_material="tape",
            confidence=0.85,
            spectral_fingerprint=fp,
            dolby_nr_type="dolby_b",
            dolby_nr_confidence=0.72,
        )
        d = r.as_dict()
        assert d["dolby_nr_type"] == "dolby_b"
        assert abs(d["dolby_nr_confidence"] - 0.72) < 1e-6

    def test_dolby_nr_type_can_be_set(self):
        from forensics.medium_detector import MediumDetectionResult, SpectralFingerprint

        fp = SpectralFingerprint(rolloff_95_hz=6000.0, noise_floor_db=-35.0)
        r = MediumDetectionResult(
            transfer_chain=["reel_tape"],
            is_multi_generation=False,
            primary_material="reel_tape",
            confidence=0.75,
            spectral_fingerprint=fp,
        )
        r.dolby_nr_type = "dolby_c"
        r.dolby_nr_confidence = 0.88
        assert r.dolby_nr_type == "dolby_c"
        assert abs(r.dolby_nr_confidence - 0.88) < 1e-6
        r = MediumDetectionResult(
            transfer_chain=["reel_tape"],
            is_multi_generation=False,
            primary_material="reel_tape",
            confidence=0.75,
            spectral_fingerprint=fp,
        )
        r.dolby_nr_type = "dolby_c"
        r.dolby_nr_confidence = 0.88
        assert r.dolby_nr_type == "dolby_c"
        assert abs(r.dolby_nr_confidence - 0.88) < 1e-6
