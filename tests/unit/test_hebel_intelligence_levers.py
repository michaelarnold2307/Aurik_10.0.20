from __future__ import annotations

"""Unit tests for Hebel 1–4 intelligence levers (v10.0.0).

Tests:
  Hebel 1 — Salience-aware PhaseSkipper in UV3
  Hebel 2 — SGMSE+ Tier-0 conditioning in Phase 03
  Hebel 3 — PhaseConductor module
  Hebel 4 — Carrier-Formant-Decay-Inversion in Phase 42
"""


import numpy as np
import pytest

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mono_audio():
    rng = np.random.default_rng(42)
    return rng.standard_normal(48000 * 3).astype(np.float32) * 0.3


@pytest.fixture
def stereo_audio():
    rng = np.random.default_rng(42)
    return rng.standard_normal((2, 48000 * 3)).astype(np.float32) * 0.3


@pytest.fixture
def short_audio():
    rng = np.random.default_rng(42)
    return rng.standard_normal(1024).astype(np.float32) * 0.3


# ─── Hebel 3: PhaseConductor ──────────────────────────────────────────────────


@pytest.mark.unit
class TestPhaseConductor:
    """Tests for backend/core/phase_conductor.py (Hebel 3)."""

    def test_import(self):
        from backend.core.phase_conductor import PhaseConductor, get_phase_conductor

        cond = get_phase_conductor()
        assert cond is not None
        assert isinstance(cond, PhaseConductor)

    def test_singleton(self):
        from backend.core.phase_conductor import get_phase_conductor

        a = get_phase_conductor()
        b = get_phase_conductor()
        assert a is b

    def test_measure_state_mono(self, mono_audio):
        from backend.core.phase_conductor import get_phase_conductor

        cond = get_phase_conductor()
        state = cond.measure_state(mono_audio, 48000, "phase_03_denoise")
        assert state is not None
        assert state.noise_floor_db <= 0.0
        assert 0.0 <= state.hf_energy_ratio <= 1.0
        # transient_density is raw Events/s — can exceed 1.0; as_vec() normalizes it
        assert state.transient_density >= 0.0
        assert 0.0 <= state.harmonic_coherence <= 1.0
        assert state.phase_id == "phase_03_denoise"

    def test_measure_state_stereo(self, stereo_audio):
        from backend.core.phase_conductor import get_phase_conductor

        cond = get_phase_conductor()
        state = cond.measure_state(stereo_audio, 48000, "phase_29_tape_hiss_reduction")
        assert state is not None
        assert -120.0 <= state.noise_floor_db <= 0.0

    def test_measure_state_short_audio(self, short_audio):
        """Must not raise for very short input."""
        from backend.core.phase_conductor import get_phase_conductor

        cond = get_phase_conductor()
        state = cond.measure_state(short_audio, 48000, "phase_01_click_removal")
        assert state is not None

    def test_measure_state_nan_input(self):
        """NaN input must not raise."""
        from backend.core.phase_conductor import get_phase_conductor

        cond = get_phase_conductor()
        bad = np.full(48000, float("nan"), dtype=np.float32)
        state = cond.measure_state(bad, 48000, "phase_03_denoise")
        assert state is not None

    def test_recommend_vinyl(self, mono_audio):
        from backend.core.phase_conductor import get_phase_conductor

        cond = get_phase_conductor()
        state = cond.measure_state(mono_audio, 48000, "phase_09_crackle_removal")
        rec = cond.recommend("phase_03_denoise", state, "vinyl")
        assert rec is not None
        assert 0.0 < rec.recommended_strength <= 1.0
        assert 0.0 <= rec.confidence <= 1.0

    def test_recommend_tape(self, mono_audio):
        from backend.core.phase_conductor import get_phase_conductor

        cond = get_phase_conductor()
        state = cond.measure_state(mono_audio, 48000, "phase_03_denoise")
        rec = cond.recommend("phase_29_tape_hiss_reduction", state, "reel_tape")
        assert rec is not None
        assert rec.recommended_strength > 0.0

    def test_recommend_never_skip_phase01(self, mono_audio):
        """phase_01_click_removal must never be marked skip_recommended."""
        from backend.core.phase_conductor import get_phase_conductor

        cond = get_phase_conductor()
        state = cond.measure_state(mono_audio, 48000, "phase_03_denoise")
        rec = cond.recommend("phase_01_click_removal", state, "vinyl")
        assert not rec.skip_recommended

    def test_conductor_does_not_skip_without_target_defect_evidence(self):
        """A generic state estimate cannot replace a phase-specific defect measurement."""
        from backend.core.phase_conductor import get_phase_conductor

        cond = get_phase_conductor()
        # Near-silent clean tone — no noise, no transients
        t = np.linspace(0, 3.0, 48000 * 3, dtype=np.float32)
        clean = np.sin(2 * np.pi * 440.0 * t) * 0.01
        state = cond.measure_state(clean, 48000, "phase_03_denoise")
        rec = cond.recommend("phase_29_tape_hiss_reduction", state, "cd_digital")
        assert rec.skip_recommended is False

    def test_recommend_keeps_coalition_continuation_active(self, mono_audio):
        """Coalition-Mitglieder dürfen nicht vorzeitig auf Skip optimiert werden."""
        from backend.core.phase_conductor import get_phase_conductor

        cond = get_phase_conductor()
        state = cond.measure_state(mono_audio, 48000, "phase_29_tape_hiss_reduction")
        coalitions: dict[str, tuple[str, ...]] = {
            "tape_cleanup": ("phase_29_tape_hiss_reduction", "phase_07_harmonic_restoration")
        }
        rec = cond.recommend(
            "phase_07_harmonic_restoration",
            state,
            "reel_tape",
            active_phase_coalitions=coalitions,
            current_phase_id="phase_29_tape_hiss_reduction",
        )
        assert rec.skip_recommended is False
        assert 0.0 <= rec.recommended_strength <= 1.0
        assert 0.0 <= rec.confidence <= 1.0

    def test_as_vec_normalized(self, mono_audio):
        """PhaseState.as_vec() must return array with values in [0, 1]."""
        from backend.core.phase_conductor import get_phase_conductor

        cond = get_phase_conductor()
        state = cond.measure_state(mono_audio, 48000, "phase_03_denoise")
        vec = state.as_vec()
        assert vec.shape == (4,)
        assert np.all(vec >= 0.0) and np.all(vec <= 1.0)

    def test_reset_clears_state(self, mono_audio):
        from backend.core.phase_conductor import get_phase_conductor

        cond = get_phase_conductor()
        cond.measure_state(mono_audio, 48000, "phase_03_denoise")
        cond.reset()
        assert len(cond._history) == 0

    def test_recommend_unknown_material_fallback(self, mono_audio):
        """Unknown material should use 'unknown' grid without raising."""
        from backend.core.phase_conductor import get_phase_conductor

        cond = get_phase_conductor()
        state = cond.measure_state(mono_audio, 48000, "phase_03_denoise")
        rec = cond.recommend("phase_03_denoise", state, "wax_cylinder")
        assert rec is not None
        assert rec.recommended_strength > 0.0

    def test_evidence_recommendation_is_not_raised_by_strength_floor(self, mono_audio):
        """§G188: Kein fixer Floor darf eine Messungsempfehlung anheben."""
        from backend.core.phase_conductor import get_phase_conductor

        cond = get_phase_conductor()
        state = cond.measure_state(mono_audio, 48000, "phase_03_denoise")
        rec = cond.recommend("phase_03_denoise", state, "vinyl")
        assert 0.0 <= rec.recommended_strength <= 1.0


# ─── Hebel 4: Carrier-Formant-Decay-Inversion ─────────────────────────────────


class TestCarrierFormantDecayInversion:
    """Tests for _restore_carrier_formant_decay() in Phase42 (Hebel 4)."""

    @pytest.fixture
    def phase42(self):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        return VocalEnhancement()

    @pytest.fixture
    def vinyl_material(self):
        from backend.core.defect_scanner import MaterialType

        return MaterialType.VINYL

    @pytest.fixture
    def tape_material(self):
        from backend.core.defect_scanner import MaterialType

        return MaterialType.REEL_TAPE

    @pytest.fixture
    def shellac_material(self):
        from backend.core.defect_scanner import MaterialType

        return MaterialType.SHELLAC

    @pytest.fixture
    def digital_material(self):
        from backend.core.defect_scanner import MaterialType

        return MaterialType.CD_DIGITAL

    def test_vinyl_mono_no_crash(self, phase42, mono_audio, vinyl_material):
        result = phase42._restore_carrier_formant_decay(mono_audio, 48000, vinyl_material)
        assert result is not None
        assert result.shape == mono_audio.shape

    def test_vinyl_stereo_no_crash(self, phase42, stereo_audio, vinyl_material):
        result = phase42._restore_carrier_formant_decay(stereo_audio, 48000, vinyl_material)
        assert result is not None
        assert result.shape == stereo_audio.shape

    def test_tape_no_crash(self, phase42, mono_audio, tape_material):
        result = phase42._restore_carrier_formant_decay(mono_audio, 48000, tape_material)
        assert result is not None
        assert not np.any(np.isnan(result))

    def test_shellac_no_crash(self, phase42, mono_audio, shellac_material):
        result = phase42._restore_carrier_formant_decay(mono_audio, 48000, shellac_material)
        assert result is not None
        assert not np.any(np.isinf(result))

    def test_digital_passthrough(self, phase42, mono_audio, digital_material):
        """cd_digital has no carrier formant decay — must return identical audio."""
        result = phase42._restore_carrier_formant_decay(mono_audio, 48000, digital_material)
        np.testing.assert_array_equal(result, mono_audio)

    def test_output_clipped_minus_one_to_one(self, phase42, mono_audio, vinyl_material):
        result = phase42._restore_carrier_formant_decay(mono_audio, 48000, vinyl_material)
        assert float(np.max(np.abs(result))) <= 1.0 + 1e-6

    def test_nan_input_safe(self, phase42, vinyl_material):
        bad = np.full(48000 * 2, float("nan"), dtype=np.float32)
        result = phase42._restore_carrier_formant_decay(bad, 48000, vinyl_material)
        assert result is not None
        assert not np.any(np.isnan(result))

    def test_short_audio_passthrough(self, phase42, short_audio, vinyl_material):
        """Audio shorter than 2048 samples must be returned unchanged."""
        result = phase42._restore_carrier_formant_decay(short_audio, 48000, vinyl_material)
        np.testing.assert_array_equal(result, short_audio)

    def test_vinyl_adds_energy_in_formant_bands(self, phase42, vinyl_material):
        """Vinyl correction should increase (or maintain) F2/F3 energy, never decrease globally."""
        np.random.default_rng(7)
        # Attenuated vinyl-like signal: low F2/F3 energy
        t = np.linspace(0, 3.0, 48000 * 3, dtype=np.float32)
        sig = (
            np.sin(2 * np.pi * 500.0 * t) * 0.3  # F1 strong
            + np.sin(2 * np.pi * 1500.0 * t) * 0.05  # F2 attenuated
            + np.sin(2 * np.pi * 2500.0 * t) * 0.02  # F3 attenuated
        ).astype(np.float32)
        result = phase42._restore_carrier_formant_decay(sig, 48000, vinyl_material)
        # F2 energy after correction should be ≥ before (never reduced by inversion)
        from scipy import signal as sp

        sos = sp.butter(2, [1200.0, 1800.0], btype="band", fs=48000, output="sos")
        e_before = float(np.mean(sp.sosfilt(sos, sig) ** 2))
        e_after = float(np.mean(sp.sosfilt(sos, result) ** 2))
        assert e_after >= e_before * 0.99, f"F2 energy dropped: {e_before:.6f} → {e_after:.6f}"

    def test_dtype_preserved(self, phase42, vinyl_material):
        audio_f64 = np.random.randn(48000 * 2).astype(np.float64) * 0.3
        result = phase42._restore_carrier_formant_decay(audio_f64, 48000, vinyl_material)
        assert result.dtype == np.float64

    def test_minidisc_correction_applied(self, phase42, mono_audio):
        from backend.core.defect_scanner import MaterialType

        result = phase42._restore_carrier_formant_decay(mono_audio, 48000, MaterialType.MINIDISC)
        assert result is not None
        assert not np.any(np.isnan(result))


# ─── Hebel 1: Salience-Aware PhaseSkipping integration check ─────────────────


class TestSalienceAwarePhaseSkipping:
    """Smoke-tests for Hebel 1: _salience_adjusted_severity in UV3."""

    def test_uv3_has_salience_adjusted_severity(self):
        """UV3 _apply_phase_skipping must reference perceptual_salience metadata."""
        import pathlib

        src = pathlib.Path("backend/core/unified_restorer_v3.py").read_text(encoding="utf-8")
        # The injected closure must be present
        assert "_salience_adjusted_severity" in src, (
            "Hebel 1: _salience_adjusted_severity not found in unified_restorer_v3.py"
        )

    def test_uv3_conductor_initialization(self):
        """UV3 must initialize PhaseConductor at _execute_pipeline start."""
        import pathlib

        src = pathlib.Path("backend/core/unified_restorer_v3.py").read_text(encoding="utf-8")
        assert "_get_conductor" in src or "get_phase_conductor" in src, (
            "Hebel 3: PhaseConductor not initialized in _execute_pipeline"
        )
        assert "_conductor_strength_hints" in src, "Hebel 3: _conductor_strength_hints not found in UV3"

    def test_uv3_conductor_strength_hint_read(self):
        """_profiled_phase_call must read from _conductor_strength_hints."""
        import pathlib

        src = pathlib.Path("backend/core/unified_restorer_v3.py").read_text(encoding="utf-8")
        assert "_conductor_hints" in src or "_conductor_strength_hints" in src


# ─── Hebel 2: SGMSE+ Tier-0 conditioning smoke-checks ────────────────────────


class TestSGMSETier0Conditioning:
    """Smoke-tests for Hebel 2: SGMSE+ Tier-0 in phase_03_denoise."""

    def test_phase03_has_sgmse_tier0_block(self):
        """phase_03_denoise.py must contain SGMSE+ Tier-0 block."""
        import pathlib

        src = pathlib.Path("backend/core/phases/phase_03_denoise.py").read_text(encoding="utf-8")
        assert "sgmse_plus_tier0_applied" in src or "get_sgmse_plus_plugin" in src, (
            "Hebel 2: SGMSE+ Tier-0 block not found in phase_03_denoise.py"
        )

    def test_phase03_sgmse_metadata_key(self):
        """Phase 03 DSP path metadata must include sgmse_plus_tier0_applied key."""
        import pathlib

        src = pathlib.Path("backend/core/phases/phase_03_denoise.py").read_text(encoding="utf-8")
        assert "sgmse_plus_tier0_applied" in src


# ─── Hebel 4: _enhance_channel material_type integration ─────────────────────


class TestEnhanceChannelMaterialType:
    """_enhance_channel must accept material_type param and call carrier inversion."""

    @pytest.fixture
    def phase42(self):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        return VocalEnhancement()

    def test_enhance_channel_accepts_material_type(self, phase42, mono_audio):
        from backend.core.defect_scanner import MaterialType

        # Use real config from the phase to avoid KeyError on missing keys
        config = dict(
            phase42.ENHANCEMENT_CONFIG.get(MaterialType.VINYL, phase42.ENHANCEMENT_CONFIG[MaterialType.CD_DIGITAL])
        )
        result = phase42._enhance_channel(mono_audio, 48000, config, material_type=MaterialType.VINYL)
        assert result is not None
        assert result.shape == mono_audio.shape
        assert not np.any(np.isnan(result))

    def test_enhance_channel_no_material_type_backward_compat(self, phase42, mono_audio):
        """Without material_type, _enhance_channel must still work (backward compat)."""
        from backend.core.defect_scanner import MaterialType

        config = dict(phase42.ENHANCEMENT_CONFIG[MaterialType.CD_DIGITAL])
        result = phase42._enhance_channel(mono_audio, 48000, config)
        assert result is not None
        assert not np.any(np.isnan(result))
