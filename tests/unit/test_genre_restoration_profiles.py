import pytest

"""Unit-Tests für Genre-Restaurierungsprofile aus core.genre_classifier.

Testet:
- SCHLAGER_RESTORATION_PROFILE
- JAZZ_RESTORATION_PROFILE
- KLASSIK_RESTORATION_PROFILE
- OPER_RESTORATION_PROFILE
- ROCK_RESTORATION_PROFILE
- get_restoration_profile() Convenience-Funktion
- Alle Profile haben "gp_memory_key"
- Korrekte Parameter-Werte
"""

from backend.core.genre_classifier import (
    GENRE_RESTORATION_PROFILES,
    JAZZ_RESTORATION_PROFILE,
    KLASSIK_RESTORATION_PROFILE,
    OPER_RESTORATION_PROFILE,
    ROCK_RESTORATION_PROFILE,
    SCHLAGER_RESTORATION_PROFILE,
    get_restoration_profile,
)


@pytest.mark.unit
class TestGenreRestorationProfiles:
    """Tests für Genre-Restaurierungsprofile."""

    def test_01_schlager_profile_has_gp_memory_key_schlager(self):
        """SCHLAGER_RESTORATION_PROFILE enthält 'gp_memory_key' == 'schlager'."""
        assert "gp_memory_key" in SCHLAGER_RESTORATION_PROFILE
        assert SCHLAGER_RESTORATION_PROFILE["gp_memory_key"] == "schlager"

    def test_02_jazz_profile_has_groove_dtw_max_ms(self):
        """JAZZ_RESTORATION_PROFILE enthält 'groove_dtw_max_ms' == 4.0."""
        assert "groove_dtw_max_ms" in JAZZ_RESTORATION_PROFILE
        assert JAZZ_RESTORATION_PROFILE["groove_dtw_max_ms"] == 4.0

    def test_03_klassik_profile_dereverb_disabled(self):
        """KLASSIK_RESTORATION_PROFILE enthält 'phase_20_dereverb_enabled' == False."""
        assert "phase_20_dereverb_enabled" in KLASSIK_RESTORATION_PROFILE
        assert KLASSIK_RESTORATION_PROFILE["phase_20_dereverb_enabled"] is False

    def test_04_oper_profile_formant_pearson_threshold(self):
        """OPER_RESTORATION_PROFILE enthält 'formant_pearson_threshold' == 0.97."""
        assert "formant_pearson_threshold" in OPER_RESTORATION_PROFILE
        assert OPER_RESTORATION_PROFILE["formant_pearson_threshold"] == 0.97

    def test_05_rock_profile_brillanz_target(self):
        """ROCK_RESTORATION_PROFILE enthält 'brillanz_target' == 0.90."""
        assert "brillanz_target" in ROCK_RESTORATION_PROFILE
        assert ROCK_RESTORATION_PROFILE["brillanz_target"] == 0.90

    def test_06_get_restoration_profile_jazz_returns_jazz_profile(self):
        """get_restoration_profile('Jazz') == JAZZ_RESTORATION_PROFILE."""
        profile = get_restoration_profile("Jazz")
        assert profile == JAZZ_RESTORATION_PROFILE

    def test_07_get_restoration_profile_schlager_returns_schlager_profile(self):
        """get_restoration_profile('Schlager') == SCHLAGER_RESTORATION_PROFILE."""
        profile = get_restoration_profile("Schlager")
        assert profile == SCHLAGER_RESTORATION_PROFILE

    def test_08_get_restoration_profile_oper_returns_oper_profile(self):
        """get_restoration_profile('Oper') == OPER_RESTORATION_PROFILE."""
        profile = get_restoration_profile("Oper")
        assert profile == OPER_RESTORATION_PROFILE

    def test_09_get_restoration_profile_unknown_returns_empty_dict(self):
        """get_restoration_profile('Unbekannt') == {} (leeres Dict als Fallback)."""
        profile = get_restoration_profile("Unbekannt")
        assert profile == {}

    def test_10_all_profiles_have_gp_memory_key(self):
        """Alle Profile haben 'gp_memory_key' Schlüssel."""
        for profile in [
            SCHLAGER_RESTORATION_PROFILE,
            JAZZ_RESTORATION_PROFILE,
            KLASSIK_RESTORATION_PROFILE,
            OPER_RESTORATION_PROFILE,
            ROCK_RESTORATION_PROFILE,
        ]:
            assert "gp_memory_key" in profile

    def test_11_all_profiles_are_dicts(self):
        """Alle Profile sind Dict-Instanzen."""
        for profile in [
            SCHLAGER_RESTORATION_PROFILE,
            JAZZ_RESTORATION_PROFILE,
            KLASSIK_RESTORATION_PROFILE,
            OPER_RESTORATION_PROFILE,
            ROCK_RESTORATION_PROFILE,
        ]:
            assert isinstance(profile, dict)

    def test_12_genre_restoration_profiles_has_multiple_entries(self):
        """GENRE_RESTORATION_PROFILES hat ≥ 5 Einträge."""
        assert len(GENRE_RESTORATION_PROFILES) >= 5

    def test_13_get_restoration_profile_rock_returns_rock_profile(self):
        """get_restoration_profile('Rock') gibt ROCK_RESTORATION_PROFILE."""
        profile = get_restoration_profile("Rock")
        assert profile == ROCK_RESTORATION_PROFILE

    def test_14_get_restoration_profile_klassik_returns_klassik_profile(self):
        """get_restoration_profile('Klassik') gibt KLASSIK_RESTORATION_PROFILE."""
        profile = get_restoration_profile("Klassik")
        assert profile == KLASSIK_RESTORATION_PROFILE

    def test_15_jazz_profile_gp_memory_key_jazz(self):
        """JAZZ_RESTORATION_PROFILE: gp_memory_key = 'jazz'."""
        assert JAZZ_RESTORATION_PROFILE["gp_memory_key"] == "jazz"

    def test_16_klassik_profile_gp_memory_key_orchestral(self):
        """KLASSIK_RESTORATION_PROFILE: gp_memory_key = 'orchestral'."""
        assert KLASSIK_RESTORATION_PROFILE["gp_memory_key"] == "orchestral"

    def test_17_oper_profile_gp_memory_key_opera(self):
        """OPER_RESTORATION_PROFILE: gp_memory_key = 'opera'."""
        assert OPER_RESTORATION_PROFILE["gp_memory_key"] == "opera"

    def test_18_rock_profile_gp_memory_key_rock(self):
        """ROCK_RESTORATION_PROFILE: gp_memory_key = 'rock'."""
        assert ROCK_RESTORATION_PROFILE["gp_memory_key"] == "rock"

    def test_19_schlager_profile_tonal_center_threshold(self):
        """SCHLAGER_RESTORATION_PROFILE: tonal_center_threshold = 0.97."""
        assert SCHLAGER_RESTORATION_PROFILE["tonal_center_threshold"] == 0.97

    def test_20_jazz_profile_compression_ratio_cap(self):
        """JAZZ_RESTORATION_PROFILE: compression_ratio_cap = 1.8."""
        assert JAZZ_RESTORATION_PROFILE["compression_ratio_cap"] == 1.8

    def test_21_klassik_profile_transient_preservation_strength(self):
        """KLASSIK_RESTORATION_PROFILE: transient_preservation_strength = 1.0."""
        assert KLASSIK_RESTORATION_PROFILE["transient_preservation_strength"] == 1.0

    def test_22_oper_profile_deessing_strength_cap(self):
        """OPER_RESTORATION_PROFILE: deessing_strength_cap = 0.35."""
        assert OPER_RESTORATION_PROFILE["deessing_strength_cap"] == 0.35

    def test_23_rock_profile_transient_preservation_strength(self):
        """ROCK_RESTORATION_PROFILE: transient_preservation_strength = 1.0."""
        assert ROCK_RESTORATION_PROFILE["transient_preservation_strength"] == 1.0

    def test_24_klassik_profile_groove_dtw_max_ms(self):
        """KLASSIK_RESTORATION_PROFILE: groove_dtw_max_ms = 10.0."""
        assert KLASSIK_RESTORATION_PROFILE["groove_dtw_max_ms"] == 10.0

    def test_25_schlager_profile_waerme_target(self):
        """SCHLAGER_RESTORATION_PROFILE: waerme_target = 0.88."""
        assert SCHLAGER_RESTORATION_PROFILE["waerme_target"] == 0.88
