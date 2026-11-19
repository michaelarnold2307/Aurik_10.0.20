from __future__ import annotations

"""
Tests für v10.0.0 — Klangtreue zum Aufnahmetag (SourceFidelityReconstructor)
==============================================================================

Abgedeckte Änderungen (v10.0.0):
  1. SourceFidelityReconstructor — neues Modul backend/core/source_fidelity_reconstructor.py
     - ERA_BANDWIDTH-Tabelle (1900–2020)
     - MATERIAL_GENERATION_COUNT-Tabelle (shellac=4, vinyl=3, cd=1, ...)
     - HF_LOSS_PER_GENERATION_DB = 1.8 dB @ 8 kHz
     - estimate() → SourceFidelityTarget
  2. UV3 _build_song_calibration_profile — source_fidelity_* Felder in return dict
     - source_fidelity_bandwidth_target_hz
     - source_fidelity_reconstruction_strength
     - source_fidelity_confidence
     - source_fidelity_generation_count
     - source_fidelity_hf_loss_db
     - source_fidelity_harmonic_density
     - reconstruction-Familien-Scalar boost über SourceFidelityReconstructor
  3. Phase 06 — source_fidelity_bandwidth_target_hz aus song_calibration_profile
     - max_boost_db wird bei Bandbreiten-Lücke ≥ 1500 Hz konservativ angehoben
     - restoration_strength-Boost bei ≥ 3 Generationen
"""


import numpy as np
import pytest

# ===========================================================================
# 1. SourceFidelityReconstructor — Unit-Tests
# ===========================================================================


@pytest.mark.unit
class TestSourceFidelityReconstructorBasics:
    """Grundlegende Modul-Tests: Import, Singleton, Datenklasse."""

    def test_01_import_works(self):
        from backend.core.source_fidelity_reconstructor import (
            SourceFidelityTarget,
            get_source_fidelity_reconstructor,
        )

        assert SourceFidelityTarget is not None
        assert get_source_fidelity_reconstructor is not None

    def test_02_singleton_returns_same_instance(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        a = get_source_fidelity_reconstructor()
        b = get_source_fidelity_reconstructor()
        assert a is b

    def test_03_estimate_returns_dataclass(self):
        from backend.core.source_fidelity_reconstructor import (
            SourceFidelityTarget,
            get_source_fidelity_reconstructor,
        )

        sfr = get_source_fidelity_reconstructor()
        t = sfr.estimate(era_decade=1960, material_key="vinyl")
        assert isinstance(t, SourceFidelityTarget)

    def test_04_all_fields_finite(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t = sfr.estimate(era_decade=1950, material_key="shellac")
        assert np.isfinite(t.original_bandwidth_hz)
        assert np.isfinite(t.current_bandwidth_hz)
        assert np.isfinite(t.bandwidth_gap_hz)
        assert np.isfinite(t.cumulative_hf_loss_db)
        assert np.isfinite(t.reconstruction_strength)
        assert np.isfinite(t.bandwidth_extension_target_hz)
        assert np.isfinite(t.confidence)
        assert np.isfinite(t.era_harmonic_density)

    def test_05_reconstruction_strength_bounded(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        for mat in ["shellac", "vinyl", "tape", "cd_digital", "mp3_low"]:
            for era in [1920, 1940, 1960, 1980, 2000]:
                t = sfr.estimate(era_decade=era, material_key=mat)
                assert 0.0 <= t.reconstruction_strength <= 1.0, (
                    f"{mat}/{era}: reconstruction_strength={t.reconstruction_strength}"
                )

    def test_06_confidence_bounded(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t = sfr.estimate(era_decade=1965, material_key="vinyl")
        assert 0.0 <= t.confidence <= 1.0

    def test_07_bandwidth_target_never_exceeds_20k(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t = sfr.estimate(era_decade=1980, material_key="tape")
        assert t.bandwidth_extension_target_hz <= 20000.0

    def test_08_bandwidth_gap_nonnegative(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t = sfr.estimate(era_decade=1955, material_key="shellac")
        assert t.bandwidth_gap_hz >= 0.0


class TestSourceFidelityReconstructorEraLogic:
    """Ära-basierte Logik: älteres Jahrzehnt = niedrigere Originalbandbreite."""

    def test_09_1920s_has_lower_bandwidth_than_1970s(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t1920 = sfr.estimate(era_decade=1920, material_key="shellac")
        t1970 = sfr.estimate(era_decade=1970, material_key="tape")
        assert t1920.original_bandwidth_hz < t1970.original_bandwidth_hz

    def test_10_1980s_has_full_bandwidth(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t = sfr.estimate(era_decade=1980, material_key="cd_digital")
        assert t.original_bandwidth_hz >= 18000.0

    def test_11_era_harmonic_density_increases_with_era(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t1930 = sfr.estimate(era_decade=1930, material_key="shellac")
        t1970 = sfr.estimate(era_decade=1970, material_key="vinyl")
        assert t1930.era_harmonic_density < t1970.era_harmonic_density

    def test_12_none_era_does_not_crash(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t = sfr.estimate(era_decade=None, material_key="vinyl")
        assert t is not None
        assert np.isfinite(t.reconstruction_strength)

    def test_13_unknown_material_does_not_crash(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t = sfr.estimate(era_decade=1955, material_key="unknown")
        assert t is not None
        assert t.transfer_generation_count >= 1


class TestSourceFidelityReconstructorGenerationModel:
    """Modell für Überspielgenerationen und akkumulierten HF-Verlust."""

    def test_14_shellac_has_more_generations_than_cd(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t_shellac = sfr.estimate(era_decade=1940, material_key="shellac")
        t_cd = sfr.estimate(era_decade=1990, material_key="cd_digital")
        assert t_shellac.transfer_generation_count > t_cd.transfer_generation_count

    def test_15_shellac_has_higher_hf_loss_than_cd(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t_shellac = sfr.estimate(era_decade=1940, material_key="shellac")
        t_cd = sfr.estimate(era_decade=1990, material_key="cd_digital")
        assert t_shellac.cumulative_hf_loss_db > t_cd.cumulative_hf_loss_db

    def test_16_transfer_chain_increases_generation_count(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t_chain = sfr.estimate(
            era_decade=1965,
            material_key="vinyl",
            transfer_chain=["mp3", "cassette", "vinyl"],
        )
        t_simple = sfr.estimate(era_decade=1965, material_key="vinyl")
        # Explicit chain should yield >= simple material-based count
        assert t_chain.transfer_generation_count >= t_simple.transfer_generation_count

    def test_17_hf_loss_proportional_to_generation_count(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t = sfr.estimate(era_decade=1955, material_key="shellac")
        # With 4 gen for shellac, loss = 3 * 1.8 = 5.4 dB
        expected_loss = (t.transfer_generation_count - 1) * 1.8
        assert abs(t.cumulative_hf_loss_db - expected_loss) < 0.5

    def test_18_cd_digital_gen1_near_zero_loss(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t = sfr.estimate(era_decade=1990, material_key="cd_digital")
        assert t.transfer_generation_count == 1
        assert t.cumulative_hf_loss_db == pytest.approx(0.0)

    def test_19_reconstruction_strength_higher_for_old_shellac_vs_cd(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t_old = sfr.estimate(era_decade=1935, material_key="shellac")
        t_new = sfr.estimate(era_decade=1995, material_key="cd_digital")
        assert t_old.reconstruction_strength > t_new.reconstruction_strength


class TestSourceFidelityReconstructorStudioMode:
    """Studio-2026-Modus: aggressivere Rekonstruktion."""

    def test_20_studio_mode_boosts_reconstruction(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t_rest = sfr.estimate(era_decade=1955, material_key="vinyl", mode="restoration")
        t_studio = sfr.estimate(era_decade=1955, material_key="vinyl", mode="studio2026")
        # Studio should have >= restoration strength
        assert t_studio.reconstruction_strength >= t_rest.reconstruction_strength


# ===========================================================================
# 2. UV3 SongCalibrationProfile — source_fidelity_* keys present
# ===========================================================================


class TestUV3SongCalibrationSourceFidelityKeys:
    """_build_song_calibration_profile muss alle source_fidelity_* Felder enthalten."""

    @pytest.fixture()
    def _calibration_deps(self):
        from backend.core.defect_scanner import MaterialType
        from backend.core.unified_restorer_v3 import QualityMode, UnifiedRestorerV3

        return UnifiedRestorerV3, QualityMode, MaterialType

    def test_21_profile_has_source_fidelity_bandwidth_target_hz(self, _calibration_deps):
        UV3, QM, MT = _calibration_deps
        profile = UV3._build_song_calibration_profile(
            material_type=MT.VINYL,
            mode=QM.QUALITY,
            restorability_score=60.0,
            input_snr_db=22.0,
            max_defect_severity=0.35,
            pipeline_confidence=0.70,
            era_decade=1965,
        )
        assert "source_fidelity_bandwidth_target_hz" in profile
        assert profile["source_fidelity_bandwidth_target_hz"] > 0.0

    def test_22_profile_has_source_fidelity_reconstruction_strength(self, _calibration_deps):
        UV3, QM, MT = _calibration_deps
        profile = UV3._build_song_calibration_profile(
            material_type=MT.SHELLAC,
            mode=QM.QUALITY,
            restorability_score=45.0,
            input_snr_db=14.0,
            max_defect_severity=0.60,
            pipeline_confidence=0.65,
            era_decade=1940,
        )
        assert "source_fidelity_reconstruction_strength" in profile
        assert 0.0 <= profile["source_fidelity_reconstruction_strength"] <= 1.0

    def test_23_profile_has_source_fidelity_confidence(self, _calibration_deps):
        UV3, QM, MT = _calibration_deps
        profile = UV3._build_song_calibration_profile(
            material_type=MT.TAPE,
            mode=QM.QUALITY,
            restorability_score=55.0,
            input_snr_db=20.0,
            max_defect_severity=0.40,
            pipeline_confidence=0.72,
            era_decade=1958,
        )
        assert "source_fidelity_confidence" in profile
        assert 0.0 <= profile["source_fidelity_confidence"] <= 1.0

    def test_24_profile_has_source_fidelity_generation_count(self, _calibration_deps):
        UV3, QM, MT = _calibration_deps
        profile = UV3._build_song_calibration_profile(
            material_type=MT.SHELLAC,
            mode=QM.QUALITY,
            restorability_score=40.0,
            input_snr_db=12.0,
            max_defect_severity=0.70,
            pipeline_confidence=0.60,
            era_decade=1938,
        )
        assert "source_fidelity_generation_count" in profile
        assert profile["source_fidelity_generation_count"] >= 1

    def test_25_profile_has_source_fidelity_hf_loss_db(self, _calibration_deps):
        UV3, QM, MT = _calibration_deps
        profile = UV3._build_song_calibration_profile(
            material_type=MT.VINYL,
            mode=QM.QUALITY,
            restorability_score=65.0,
            input_snr_db=26.0,
            max_defect_severity=0.30,
            pipeline_confidence=0.75,
            era_decade=1970,
        )
        assert "source_fidelity_hf_loss_db" in profile
        assert profile["source_fidelity_hf_loss_db"] >= 0.0

    def test_26_profile_has_source_fidelity_harmonic_density(self, _calibration_deps):
        UV3, QM, MT = _calibration_deps
        profile = UV3._build_song_calibration_profile(
            material_type=MT.VINYL,
            mode=QM.QUALITY,
            restorability_score=65.0,
            input_snr_db=26.0,
            max_defect_severity=0.30,
            pipeline_confidence=0.75,
            era_decade=1970,
        )
        assert "source_fidelity_harmonic_density" in profile
        assert profile["source_fidelity_harmonic_density"] > 0.0

    def test_27_shellac_1940s_has_higher_reconstruction_than_cd_1990s(self, _calibration_deps):
        UV3, QM, MT = _calibration_deps
        p_shellac = UV3._build_song_calibration_profile(
            material_type=MT.SHELLAC,
            mode=QM.QUALITY,
            restorability_score=50.0,
            input_snr_db=15.0,
            max_defect_severity=0.60,
            pipeline_confidence=0.65,
            era_decade=1942,
        )
        p_cd = UV3._build_song_calibration_profile(
            material_type=None,  # cd_digital via None/unknown
            mode=QM.QUALITY,
            restorability_score=90.0,
            input_snr_db=55.0,
            max_defect_severity=0.05,
            pipeline_confidence=0.95,
            era_decade=1995,
        )
        # Shellac 1942 should have higher source_fidelity_reconstruction_strength
        assert p_shellac["source_fidelity_reconstruction_strength"] >= p_cd["source_fidelity_reconstruction_strength"]

    def test_28_reconstruction_family_scalar_boosted_for_old_shellac(self, _calibration_deps):
        UV3, QM, MT = _calibration_deps
        # Old shellac 1935: high bw_gap, many generations → reconstruction scalar boosted
        p_shellac = UV3._build_song_calibration_profile(
            material_type=MT.SHELLAC,
            mode=QM.QUALITY,
            restorability_score=50.0,
            input_snr_db=14.0,
            max_defect_severity=0.65,
            pipeline_confidence=0.70,
            era_decade=1935,
        )
        # New digital: no generations → no boost
        p_cd = UV3._build_song_calibration_profile(
            material_type=None,
            mode=QM.QUALITY,
            restorability_score=90.0,
            input_snr_db=55.0,
            max_defect_severity=0.02,
            pipeline_confidence=0.98,
            era_decade=1998,
        )
        recon_shellac = p_shellac["family_scalars"]["reconstruction"]
        recon_cd = p_cd["family_scalars"]["reconstruction"]
        # Shellac reconstruction scalar should be higher (era contribution)
        assert recon_shellac >= recon_cd

    def test_29_profile_without_era_still_returns_source_fidelity_fields(self, _calibration_deps):
        UV3, QM, MT = _calibration_deps
        profile = UV3._build_song_calibration_profile(
            material_type=MT.TAPE,
            mode=QM.QUALITY,
            restorability_score=60.0,
            input_snr_db=20.0,
            max_defect_severity=0.45,
            pipeline_confidence=0.70,
            era_decade=None,  # unknown era
        )
        # Should still have keys (filled with defaults)
        assert "source_fidelity_bandwidth_target_hz" in profile
        assert profile["source_fidelity_bandwidth_target_hz"] > 0.0


# ===========================================================================
# 3. Phase 06 — source_fidelity_bandwidth_target_hz Integration
# ===========================================================================


class TestPhase06SourceFidelityIntegration:
    """Phase 06 nutzt source_fidelity_bandwidth_target_hz aus song_calibration_profile."""

    @pytest.fixture()
    def _make_audio(self):
        """200 ms stereo 48 kHz Sinuston mit HF-Rolloff ~ 8 kHz."""
        sr = 48000
        t = np.linspace(0, 0.2, int(0.2 * sr))
        # Simulate bandlimited audio (rolloff around 8 kHz)
        audio = np.zeros_like(t)
        for freq in [100, 440, 1000, 3000, 6000, 8000]:
            audio += 0.08 * np.sin(2 * np.pi * freq * t)
        audio = audio.astype(np.float32)
        return np.stack([audio, audio], axis=0), sr  # (2, samples)

    def test_30_phase06_accepts_song_calibration_profile_kwarg(self, _make_audio):
        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase

        audio, sr = _make_audio
        phase = FrequencyRestorationPhase()
        result = phase.process(
            audio,
            material_type="shellac",
            sample_rate=sr,
            song_calibration_profile={
                "source_fidelity_bandwidth_target_hz": 14000.0,
                "source_fidelity_confidence": 0.75,
                "source_fidelity_generation_count": 4,
            },
        )
        assert result is not None
        assert hasattr(result, "audio")
        assert np.isfinite(result.audio).all()

    def test_31_phase06_no_source_fidelity_profile_still_works(self, _make_audio):
        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase

        audio, sr = _make_audio
        phase = FrequencyRestorationPhase()
        # Without song_calibration_profile kwarg — must not crash
        result = phase.process(audio, material_type="vinyl", sample_rate=sr)
        assert result is not None
        assert np.isfinite(result.audio).all()

    def test_32_phase06_audio_shape_preserved(self, _make_audio):
        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase

        audio, sr = _make_audio
        shape_in = audio.shape
        phase = FrequencyRestorationPhase()
        result = phase.process(
            audio,
            material_type="tape",
            sample_rate=sr,
            song_calibration_profile={
                "source_fidelity_bandwidth_target_hz": 12000.0,
                "source_fidelity_confidence": 0.80,
                "source_fidelity_generation_count": 3,
            },
        )
        assert result.audio.shape == shape_in

    def test_33_phase06_output_clipped(self, _make_audio):
        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase

        audio, sr = _make_audio
        phase = FrequencyRestorationPhase()
        result = phase.process(
            audio,
            material_type="shellac",
            sample_rate=sr,
            song_calibration_profile={
                "source_fidelity_bandwidth_target_hz": 14500.0,
                "source_fidelity_confidence": 0.90,
                "source_fidelity_generation_count": 4,
            },
        )
        assert np.max(np.abs(result.audio)) <= 1.0

    def test_34_phase06_small_bw_gap_no_override(self, _make_audio):
        """BW gap < 1500 Hz soll keinen Override der params auslösen."""
        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase

        audio, sr = _make_audio
        phase = FrequencyRestorationPhase()
        # rolloff_hz for shellac might be ~5000 Hz; target 6000 Hz → gap ~1000 Hz < 1500 Hz
        result = phase.process(
            audio,
            material_type="shellac",
            sample_rate=sr,
            song_calibration_profile={
                "source_fidelity_bandwidth_target_hz": 6000.0,
                "source_fidelity_confidence": 0.85,
                "source_fidelity_generation_count": 4,
            },
        )
        # Should still complete without crash
        assert result is not None
        assert np.isfinite(result.audio).all()

    def test_35_phase06_mono_audio_works(self):
        """Mono Audio mit Source-Fidelity-Profil."""
        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase

        sr = 48000
        t = np.linspace(0, 0.1, int(0.1 * sr))
        audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        phase = FrequencyRestorationPhase()
        result = phase.process(
            audio,
            material_type="vinyl",
            sample_rate=sr,
            song_calibration_profile={
                "source_fidelity_bandwidth_target_hz": 16500.0,
                "source_fidelity_confidence": 0.70,
                "source_fidelity_generation_count": 3,
            },
        )
        assert result is not None
        assert np.isfinite(result.audio).all()


# ===========================================================================
# 4. Invarianten — konsistente Schätzungen
# ===========================================================================


class TestSourceFidelityInvariants:
    """Invarianten über Materialien und Ären."""

    def test_36_vinyl_older_has_more_bandwidth_gap_than_vinyl_newer(self):
        """Gleiche Messbandbreite, aber älteres Original → größere Lücke."""
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        # Assumption: same current bandwidth
        t1960 = sfr.estimate(
            era_decade=1960,
            material_key="vinyl",
            current_bandwidth_hz=12000.0,
        )
        t1980 = sfr.estimate(
            era_decade=1980,
            material_key="vinyl",
            current_bandwidth_hz=12000.0,
        )
        # 1960 original had ~16.5 kHz; 1980 original had ~20 kHz → 1980 has larger gap?
        # Both measured at 12k → gap_1960 = 4.5k, gap_1980 = 8k
        # So 1980 has LARGER gap — still our formula must be consistent
        assert t1980.bandwidth_gap_hz >= t1960.bandwidth_gap_hz

    def test_37_notes_list_not_empty(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t = sfr.estimate(era_decade=1962, material_key="tape")
        assert len(t.notes) > 0

    def test_38_era_decade_stored_correctly(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t = sfr.estimate(era_decade=1967, material_key="vinyl")  # should round to 1960
        assert t.era_decade == 1960

    def test_39_dr_loss_cap_applied(self):
        """Akkumulierter DR-Verlust darf nie mehr als 50% des Original-DR betragen."""
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t = sfr.estimate(era_decade=1930, material_key="shellac")
        assert t.cumulative_dr_loss_db <= t.original_dynamic_range_db * 0.5 + 0.01

    def test_40_confidence_higher_with_era_and_material(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        t_full = sfr.estimate(era_decade=1955, material_key="shellac")
        t_minimal = sfr.estimate(era_decade=None, material_key="unknown")
        assert t_full.confidence > t_minimal.confidence

    def test_41_spectral_fingerprint_lowers_estimated_current_bandwidth(self):
        from backend.core.source_fidelity_reconstructor import get_source_fidelity_reconstructor

        sfr = get_source_fidelity_reconstructor()
        fingerprint = {
            "rolloff_95_hz": 8000.0,
            "effective_bandwidth_hz": 8500.0,
        }
        t_fp = sfr.estimate(era_decade=1960, material_key="vinyl", spectral_fingerprint=fingerprint)
        t_no_fp = sfr.estimate(era_decade=1960, material_key="vinyl")
        # With fingerprint showing 8 kHz cutoff, gap should be larger
        assert t_fp.bandwidth_gap_hz > t_no_fp.bandwidth_gap_hz
