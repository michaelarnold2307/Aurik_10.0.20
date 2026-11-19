"""Unit-Tests für EraVocalProfile (§EraVocalProfile, v10.0.0)."""

import pytest

from backend.core.musical_goals.era_vocal_profile import (
    ERA_VOCAL_PROFILES,
    EraVocalProfile,
    get_era_vocal_profile,
    resolve_formant_tolerance_db,
)


@pytest.mark.unit
class TestEraVocalProfileConstants:
    """Prüft die Konstanten in ERA_VOCAL_PROFILES."""

    def test_all_five_eras_present(self):
        expected = {"1900_1925", "1925_1945", "1945_1960", "1960_1975", "1975_plus"}
        assert set(ERA_VOCAL_PROFILES.keys()) == expected

    def test_profiles_are_frozen_dataclass(self):
        for key, profile in ERA_VOCAL_PROFILES.items():
            assert isinstance(profile, EraVocalProfile), f"Profil {key} ist keine EraVocalProfile-Instanz"
            with pytest.raises((AttributeError, TypeError)):
                profile.f1_tolerance_db = 99.0  # type: ignore[misc]

    def test_f1_tolerance_decreases_with_era(self):
        """Historische Ären sollen höhere F1-Toleranz haben als modernes Material."""
        early = ERA_VOCAL_PROFILES["1900_1925"].f1_tolerance_db
        modern = ERA_VOCAL_PROFILES["1975_plus"].f1_tolerance_db
        assert early > modern, f"Early-Ära f1_tol={early} soll > Modern f1_tol={modern} sein"

    def test_vibrato_hz_range_is_tuple_of_two(self):
        for key, profile in ERA_VOCAL_PROFILES.items():
            assert len(profile.vibrato_hz_range) == 2, f"Profil {key}: vibrato_hz_range muss 2-Tupel sein"
            lo, hi = profile.vibrato_hz_range
            assert lo < hi, f"Profil {key}: vibrato_hz_range-Untergrenze {lo} muss < Obergrenze {hi} sein"

    def test_nasality_expected_only_in_old_eras(self):
        assert ERA_VOCAL_PROFILES["1900_1925"].nasality_expected is True
        assert ERA_VOCAL_PROFILES["1925_1945"].nasality_expected is True
        assert ERA_VOCAL_PROFILES["1975_plus"].nasality_expected is False

    def test_dynamic_range_increases_over_eras(self):
        earliest = ERA_VOCAL_PROFILES["1900_1925"].dynamic_range_typical_lu
        latest = ERA_VOCAL_PROFILES["1975_plus"].dynamic_range_typical_lu
        assert latest > earliest


class TestGetEraVocalProfile:
    """Prüft get_era_vocal_profile() Boundary-Mapping."""

    def test_none_returns_modern_profile(self):
        result = get_era_vocal_profile(None)
        assert result is ERA_VOCAL_PROFILES["1975_plus"]

    def test_1900_maps_to_earliest(self):
        result = get_era_vocal_profile(1900)
        assert result is ERA_VOCAL_PROFILES["1900_1925"]

    def test_1924_maps_to_earliest(self):
        result = get_era_vocal_profile(1924)
        assert result is ERA_VOCAL_PROFILES["1900_1925"]

    def test_1925_maps_to_second(self):
        result = get_era_vocal_profile(1925)
        assert result is ERA_VOCAL_PROFILES["1925_1945"]

    def test_1944_maps_to_second(self):
        result = get_era_vocal_profile(1944)
        assert result is ERA_VOCAL_PROFILES["1925_1945"]

    def test_1945_maps_to_third(self):
        result = get_era_vocal_profile(1945)
        assert result is ERA_VOCAL_PROFILES["1945_1960"]

    def test_1960_maps_to_fourth(self):
        result = get_era_vocal_profile(1960)
        assert result is ERA_VOCAL_PROFILES["1960_1975"]

    def test_1975_maps_to_modern(self):
        result = get_era_vocal_profile(1975)
        assert result is ERA_VOCAL_PROFILES["1975_plus"]

    def test_2020_maps_to_modern(self):
        result = get_era_vocal_profile(2020)
        assert result is ERA_VOCAL_PROFILES["1975_plus"]

    def test_return_type_is_era_vocal_profile(self):
        for decade in [1910, 1930, 1950, 1965, 1980, 2000]:
            result = get_era_vocal_profile(decade)
            assert isinstance(result, EraVocalProfile), f"get_era_vocal_profile({decade}) → kein EraVocalProfile"


class TestResolveFormantToleranceDb:
    """Prüft zentrale era-adaptive Formant-Guard-Toleranz."""

    def test_historical_tolerance_is_more_permissive_than_modern(self):
        assert resolve_formant_tolerance_db(1910) > resolve_formant_tolerance_db(1980)

    def test_modern_tolerance_keeps_legacy_two_db_guard(self):
        assert resolve_formant_tolerance_db(2000) == pytest.approx(2.0)

    def test_profile_argument_takes_precedence(self):
        profile = EraVocalProfile(
            vibrato_hz_range=(4.0, 6.0),
            f1_tolerance_db=3.25,
            f2_f4_tolerance_db=2.0,
            nasality_expected=False,
            dynamic_range_typical_lu=10.0,
        )
        assert resolve_formant_tolerance_db(2000, era_profile=profile) == pytest.approx(3.25)
