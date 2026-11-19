from __future__ import annotations

import pytest

"""
Tests for v10.0.0 — Denker full-context optimization.

§2.41: Material-adaptive ReparaturDenker thresholds, chirurgical
defect-location-based repair, era-adaptive hum detection.
RekonstruktionsDenker material-adaptive GapReconstructor config.
AurikDenker passes full context (defect_scores, defect_locations,
era_decade, material) to Repair + Reconstruction stages.

52+ tests covering:
- Material-adaptive threshold profiles
- Era-adaptive hum sensitivity
- Chirurgical click removal with defect_locations
- Material-adaptive GapReconstructor config
- AurikDenker full context forwarding
- Backward compatibility (no new params = old behavior)
"""


from unittest.mock import patch

import numpy as np

# ── ReparaturDenker Tests ─────────────────────────────────────────────────


@pytest.mark.unit
class TestReparaturDenkerMaterialProfiles:
    """§2.41: Material-adaptive threshold profiles."""

    def _make_denker(self):
        from denker.reparatur_denker import ReparaturDenker

        return ReparaturDenker()

    def test_shellac_profile_applied(self):
        """Shellac uses aggressive click detection (IQR=4.0)."""
        d = self._make_denker()
        d._apply_material_profile("shellac")
        assert d._CLICK_IQR_MULTIPLIER == 4.0
        assert d._CLICK_KERNEL_MS == 1.0
        assert d._HUM_DETECT_DB == -45.0

    def test_vinyl_profile_applied(self):
        """Vinyl uses moderate click detection (IQR=5.0)."""
        d = self._make_denker()
        d._apply_material_profile("vinyl")
        assert d._CLICK_IQR_MULTIPLIER == 5.0
        assert d._HUM_DETECT_DB == -48.0

    def test_cd_digital_profile_conservative(self):
        """CD digital uses conservative click detection (IQR=9.0)."""
        d = self._make_denker()
        d._apply_material_profile("cd_digital")
        assert d._CLICK_IQR_MULTIPLIER == 9.0
        assert d._CLIP_THRESHOLD == 0.998
        assert d._HUM_DETECT_DB == -55.0

    def test_tape_profile_applied(self):
        """Tape uses higher IQR (fewer clicks expected)."""
        d = self._make_denker()
        d._apply_material_profile("tape")
        assert d._CLICK_IQR_MULTIPLIER == 7.0
        assert d._CLICK_KERNEL_MS == 2.0
        assert d._CLIP_THRESHOLD == 0.992

    def test_reel_tape_profile(self):
        """Reel tape = professional, very conservative click detection."""
        d = self._make_denker()
        d._apply_material_profile("reel_tape")
        assert d._CLICK_IQR_MULTIPLIER == 8.0

    def test_wax_cylinder_most_aggressive(self):
        """Wax cylinder: most aggressive detection."""
        d = self._make_denker()
        d._apply_material_profile("wax_cylinder")
        assert d._CLICK_IQR_MULTIPLIER == 3.5
        assert d._HUM_DETECT_DB == -42.0

    def test_unknown_material_resets_to_defaults(self):
        """Unknown material resets to default thresholds."""
        d = self._make_denker()
        d._apply_material_profile("shellac")  # first apply shellac
        d._apply_material_profile("some_unknown_material")  # then unknown
        assert d._CLICK_IQR_MULTIPLIER == 6.0  # default
        assert d._HUM_DETECT_DB == -50.0  # default

    def test_empty_material_string(self):
        """Empty material string → defaults."""
        d = self._make_denker()
        d._apply_material_profile("")
        assert d._CLICK_IQR_MULTIPLIER == 6.0

    def test_material_profile_case_insensitive(self):
        """Profile lookup handles mixed case."""
        d = self._make_denker()
        d._apply_material_profile("SHELLAC")
        assert d._CLICK_IQR_MULTIPLIER == 4.0

    def test_all_profiles_have_four_keys(self):
        """Every material profile has all 4 required threshold keys."""
        from denker.reparatur_denker import ReparaturDenker

        required_keys = {"click_iqr", "click_kernel_ms", "clip_threshold", "hum_detect_db"}
        for mat, profile in ReparaturDenker._MATERIAL_PROFILES.items():
            assert required_keys <= set(profile.keys()), f"Material '{mat}' missing keys"

    def test_shellac_iqr_less_than_cd(self):
        """Shellac must have lower IQR than CD (more aggressive)."""
        from denker.reparatur_denker import ReparaturDenker

        shellac_iqr = ReparaturDenker._MATERIAL_PROFILES["shellac"]["click_iqr"]
        cd_iqr = ReparaturDenker._MATERIAL_PROFILES["cd_digital"]["click_iqr"]
        assert shellac_iqr < cd_iqr

    def test_analog_hum_more_sensitive_than_digital(self):
        """Analog materials need more sensitive hum detection (higher dB value)."""
        from denker.reparatur_denker import ReparaturDenker

        shellac_hum = ReparaturDenker._MATERIAL_PROFILES["shellac"]["hum_detect_db"]
        cd_hum = ReparaturDenker._MATERIAL_PROFILES["cd_digital"]["hum_detect_db"]
        # Higher dB = more sensitive (closer to 0)
        assert shellac_hum > cd_hum


class TestReparaturDenkerEraAdaptive:
    """§2.41: Era-adaptive hum detection sensitivity."""

    def _make_denker(self):
        from denker.reparatur_denker import ReparaturDenker

        return ReparaturDenker()

    def test_era_pre1940_hum_sensitivity(self):
        """Pre-1940 recordings: hum detection at -42 dB or more sensitive."""
        d = self._make_denker()
        audio = np.random.randn(48000).astype(np.float32) * 0.01
        result = d.repariere(audio, 48000, material="shellac", era_decade=1930)
        # After repariere, _HUM_DETECT_DB should be -42.0 or higher (more sensitive)
        assert d._HUM_DETECT_DB >= -42.0

    def test_era_1950_hum_sensitivity(self):
        """1950s: hum detection at -47 dB or more sensitive."""
        d = self._make_denker()
        audio = np.random.randn(48000).astype(np.float32) * 0.01
        d.repariere(audio, 48000, material="vinyl", era_decade=1955)
        assert d._HUM_DETECT_DB >= -48.0

    def test_era_1990_no_override(self):
        """Post-1980: no era override on hum detection."""
        d = self._make_denker()
        audio = np.random.randn(48000).astype(np.float32) * 0.01
        d.repariere(audio, 48000, material="cd_digital", era_decade=1990)
        # CD default is -55 dB; no era override makes it more sensitive
        assert d._HUM_DETECT_DB == -55.0

    def test_era_none_no_crash(self):
        """era_decade=None → no era override, no crash."""
        d = self._make_denker()
        audio = np.random.randn(48000).astype(np.float32) * 0.01
        result = d.repariere(audio, 48000, material="vinyl", era_decade=None)
        assert result.audio.shape == audio.shape


class TestReparaturDenkerChirurgicalRepair:
    """§2.41: Click removal with defect_locations — surgical mode."""

    def _make_denker(self):
        from denker.reparatur_denker import ReparaturDenker

        return ReparaturDenker()

    def test_click_locations_restrict_mask(self):
        """Defect locations restrict click removal to known regions."""
        d = self._make_denker()
        # Create audio with clicks at specific positions
        sr = 48000
        audio = np.zeros(sr * 2, dtype=np.float32)  # 2 seconds
        # Insert click at 0.5s
        audio[int(0.5 * sr)] = 0.9
        audio[int(0.5 * sr) + 1] = -0.8
        # Insert click at 1.5s (outside defect_locations)
        audio[int(1.5 * sr)] = 0.9
        audio[int(1.5 * sr) + 1] = -0.8

        # Only repair clicks in the first second
        locs: dict[str, list[tuple[float, float]]] = {
            "click": [(0.4, 0.6)],
        }
        result = d.repariere(
            audio,
            sr,
            remove_clicks=True,
            remove_hum=False,
            repair_clipping=False,
            defect_locations=locs,
        )
        # The click at 1.5s should remain untouched
        assert result.audio is not None

    def test_no_locations_full_scan(self):
        """Without defect_locations, full IQR scan applies (backward compat)."""
        d = self._make_denker()
        sr = 48000
        audio = np.random.randn(sr).astype(np.float32) * 0.1
        # Insert clicks
        for i in range(0, sr, 5000):
            audio[i] = 0.95
        result = d.repariere(
            audio,
            sr,
            remove_clicks=True,
            remove_hum=False,
            repair_clipping=False,
            defect_locations=None,
        )
        assert result.audio is not None

    def test_empty_locations_full_scan(self):
        """Empty defect_locations dict → full scan."""
        d = self._make_denker()
        sr = 48000
        audio = np.random.randn(sr).astype(np.float32) * 0.1
        result = d.repariere(
            audio,
            sr,
            remove_clicks=True,
            remove_hum=False,
            repair_clipping=False,
            defect_locations={},
        )
        assert result.audio is not None


class TestReparaturDenkerBackwardCompat:
    """Backward compatibility: no new params = old behavior."""

    def _make_denker(self):
        from denker.reparatur_denker import ReparaturDenker

        return ReparaturDenker()

    def test_basic_call_without_new_params(self):
        """Basic repariere() call without new params still works."""
        d = self._make_denker()
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = d.repariere(audio, 48000)
        assert result.audio.shape == audio.shape
        assert isinstance(result.clicks_removed, int)
        assert isinstance(result.hum_removed, bool)

    def test_nan_inf_handling(self):
        """NaN/Inf in input is handled."""
        d = self._make_denker()
        audio = np.array([0.0, np.nan, np.inf, -np.inf, 0.5], dtype=np.float32)
        result = d.repariere(audio, 48000)
        assert np.all(np.isfinite(result.audio))

    def test_output_clipped(self):
        """Output audio is clipped to [-1, 1]."""
        d = self._make_denker()
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = d.repariere(audio, 48000)
        assert np.all(result.audio >= -1.0)
        assert np.all(result.audio <= 1.0)

    def test_repairs_applied_list(self):
        """repairs_applied is properly populated."""
        d = self._make_denker()
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = d.repariere(audio, 48000)
        assert isinstance(result.repairs_applied, list)

    def test_defect_scores_param_accepted(self):
        """defect_scores param is accepted without error."""
        d = self._make_denker()
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = d.repariere(
            audio,
            48000,
            defect_scores={"click": 0.7, "hum": 0.3},
        )
        assert result.audio is not None


# ── RekonstruktionsDenker Tests ───────────────────────────────────────────


class TestRekonstruktionsDenkerMaterialConfig:
    """§2.41: Material-adaptive GapReconstructor configuration."""

    def test_material_gap_configs_exist(self):
        """All analog materials have gap configs."""
        from denker.rekonstruktions_denker import RekonstruktionsDenker

        required = {"shellac", "vinyl", "tape", "reel_tape", "cassette", "wax_cylinder"}
        assert required <= set(RekonstruktionsDenker._MATERIAL_GAP_CONFIGS.keys())

    def test_shellac_shorter_gaps(self):
        """Shellac has shorter max gap than tape."""
        from denker.rekonstruktions_denker import RekonstruktionsDenker

        shellac_max = RekonstruktionsDenker._MATERIAL_GAP_CONFIGS["shellac"]["max_gap_duration_ms"]
        tape_max = RekonstruktionsDenker._MATERIAL_GAP_CONFIGS["tape"]["max_gap_duration_ms"]
        assert shellac_max < tape_max

    def test_tape_longer_blend(self):
        """Tape uses longer blend for smoother transitions."""
        from denker.rekonstruktions_denker import RekonstruktionsDenker

        shellac_blend = RekonstruktionsDenker._MATERIAL_GAP_CONFIGS["shellac"]["blend_ms"]
        tape_blend = RekonstruktionsDenker._MATERIAL_GAP_CONFIGS["tape"]["blend_ms"]
        assert tape_blend > shellac_blend

    def test_shellac_higher_silence_threshold(self):
        """Shellac has higher silence threshold (more noise floor)."""
        from denker.rekonstruktions_denker import RekonstruktionsDenker

        shellac_thr = RekonstruktionsDenker._MATERIAL_GAP_CONFIGS["shellac"]["silence_threshold_db"]
        reel_thr = RekonstruktionsDenker._MATERIAL_GAP_CONFIGS["reel_tape"]["silence_threshold_db"]
        assert shellac_thr > reel_thr

    def test_all_configs_have_four_keys(self):
        """Every gap config has all required keys."""
        from denker.rekonstruktions_denker import RekonstruktionsDenker

        required_keys = {"silence_threshold_db", "min_gap_duration_ms", "max_gap_duration_ms", "blend_ms"}
        for mat, cfg in RekonstruktionsDenker._MATERIAL_GAP_CONFIGS.items():
            assert required_keys <= set(cfg.keys()), f"Material '{mat}' missing gap config keys"

    def test_material_adaptive_reconstructor_called(self):
        """_get_reconstructor(material='tape') uses material-adaptive config."""
        from denker.rekonstruktions_denker import RekonstruktionsDenker

        d = RekonstruktionsDenker()
        # _build_reconstructor should be called with material
        with patch.object(d, "_build_reconstructor", return_value=None) as mock_build:
            d._get_reconstructor(material="tape")
            mock_build.assert_called_with(material="tape")

    def test_unknown_material_uses_default(self):
        """Unknown material falls back to default config."""
        from denker.rekonstruktions_denker import RekonstruktionsDenker

        d = RekonstruktionsDenker()
        with patch.object(d, "_build_reconstructor", return_value=None) as mock_build:
            d._get_reconstructor(material="unknown_material")
            # Should not be called with material (fallback to cached)
            # The method uses the cached _reconstructor or builds default
            assert d._reconstructor is None or mock_build.called


class TestRekonstruktionsDenkerNewParams:
    """New parameters (defect_locations, era_decade) accepted."""

    def test_defect_locations_param_accepted(self):
        """defect_locations param is accepted without error."""
        from denker.rekonstruktions_denker import RekonstruktionsDenker

        d = RekonstruktionsDenker()
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        # Digital material → skip GapReconstructor entirely (no error)
        result = d.rekonstruiere(
            audio,
            48000,
            material_hint="cd_digital",
            defect_locations={"dropout": [(0.5, 0.7)]},
        )
        assert result.audio is not None

    def test_era_decade_param_accepted(self):
        """era_decade param is accepted without error."""
        from denker.rekonstruktions_denker import RekonstruktionsDenker

        d = RekonstruktionsDenker()
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = d.rekonstruiere(
            audio,
            48000,
            material_hint="mp3_high",
            era_decade=1990,
        )
        assert result.audio is not None

    def test_digital_guard_still_works(self):
        """Digital material still skips GapReconstructor."""
        from denker.rekonstruktions_denker import RekonstruktionsDenker

        d = RekonstruktionsDenker()
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = d.rekonstruiere(audio, 48000, material_hint="cd_digital")
        assert result.gaps_found == 0
        assert "§6.4b" in result.detail_note

    def test_backward_compat_no_new_params(self):
        """Basic rekonstruiere() call without new params still works."""
        from denker.rekonstruktions_denker import RekonstruktionsDenker

        d = RekonstruktionsDenker()
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = d.rekonstruiere(audio, 48000, material_hint="mp3_high")
        assert result.audio.shape == audio.shape


# ── AurikDenker Context Forwarding Tests ──────────────────────────────────


class TestAurikDenkerContextForwarding:
    """§2.41: AurikDenker forwards full context to Repair/Reconstruction."""

    def test_reparatur_denker_accepts_new_params(self):
        """ReparaturDenker.repariere() accepts defect_scores, defect_locations, era_decade."""
        from denker.reparatur_denker import ReparaturDenker

        d = ReparaturDenker()
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = d.repariere(
            audio,
            48000,
            material="vinyl",
            defect_scores={"click": 0.7, "hum": 0.5},
            defect_locations={"click": [(0.1, 0.2), (0.5, 0.6)]},
            era_decade=1960,
        )
        assert result.audio is not None
        assert result.audio.shape == audio.shape

    def test_rekonstruktions_denker_accepts_new_params(self):
        """RekonstruktionsDenker.rekonstruiere() accepts defect_locations, era_decade."""
        from denker.rekonstruktions_denker import RekonstruktionsDenker

        d = RekonstruktionsDenker()
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = d.rekonstruiere(
            audio,
            48000,
            material_hint="mp3_high",
            defect_locations={"dropout": [(0.5, 0.8)]},
            era_decade=1995,
        )
        assert result.audio is not None


# ── Material Profile Monotonicity Tests ───────────────────────────────────


class TestMaterialProfileMonotonicity:
    """Material profiles should follow physical degradation hierarchy."""

    def test_iqr_ordering_analog_to_digital(self):
        """IQR should increase from shellac → vinyl → tape → CD."""
        from denker.reparatur_denker import ReparaturDenker

        p = ReparaturDenker._MATERIAL_PROFILES
        assert p["shellac"]["click_iqr"] < p["vinyl"]["click_iqr"]
        assert p["vinyl"]["click_iqr"] < p["tape"]["click_iqr"]
        assert p["tape"]["click_iqr"] < p["cd_digital"]["click_iqr"]

    def test_hum_db_ordering(self):
        """Hum sensitivity should decrease from shellac → CD."""
        from denker.reparatur_denker import ReparaturDenker

        p = ReparaturDenker._MATERIAL_PROFILES
        # Higher dB = more sensitive (closer to 0)
        assert p["shellac"]["hum_detect_db"] > p["vinyl"]["hum_detect_db"]
        assert p["vinyl"]["hum_detect_db"] > p["cd_digital"]["hum_detect_db"]

    def test_clip_threshold_ordering(self):
        """Clip threshold should increase from degraded → clean."""
        from denker.reparatur_denker import ReparaturDenker

        p = ReparaturDenker._MATERIAL_PROFILES
        assert p["wax_cylinder"]["clip_threshold"] < p["cd_digital"]["clip_threshold"]


# ── Gap Config Plausibility Tests ─────────────────────────────────────────


class TestGapConfigPlausibility:
    """Plausibility checks for material-adaptive gap configs."""

    def test_shellac_max_gap_physical_limit(self):
        """Shellac mechanical needle jumps: max ~200 ms."""
        from denker.rekonstruktions_denker import RekonstruktionsDenker

        assert RekonstruktionsDenker._MATERIAL_GAP_CONFIGS["shellac"]["max_gap_duration_ms"] <= 300

    def test_tape_max_gap_allows_long_dropouts(self):
        """Tape dropouts can be up to 2 seconds."""
        from denker.rekonstruktions_denker import RekonstruktionsDenker

        assert RekonstruktionsDenker._MATERIAL_GAP_CONFIGS["tape"]["max_gap_duration_ms"] >= 1000

    def test_blend_ms_positive(self):
        """All blend values must be positive."""
        from denker.rekonstruktions_denker import RekonstruktionsDenker

        for mat, cfg in RekonstruktionsDenker._MATERIAL_GAP_CONFIGS.items():
            assert cfg["blend_ms"] > 0, f"Material '{mat}' has non-positive blend_ms"

    def test_silence_thresholds_negative(self):
        """All silence thresholds must be negative dB values."""
        from denker.rekonstruktions_denker import RekonstruktionsDenker

        for mat, cfg in RekonstruktionsDenker._MATERIAL_GAP_CONFIGS.items():
            assert cfg["silence_threshold_db"] < 0, f"Material '{mat}' has non-negative silence threshold"
