"""
Tests für Adaptive Thresholds (Genre/Medium Calibration).

Component 0.9.4: Genre & Medium-Specific Calibration
"""

import json
import tempfile
from pathlib import Path

import pytest

from backend.core.musical_goals.adaptive_thresholds import AdaptiveThresholdsManager, ThresholdProfile


@pytest.fixture
def manager():
    """Create AdaptiveThresholdsManager instance."""
    return AdaptiveThresholdsManager()


@pytest.fixture
def manager_with_profiles():
    """Create manager with custom profiles file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        profiles_path = Path(tmpdir) / "profiles.json"
        manager = AdaptiveThresholdsManager(profiles_path=profiles_path)
        yield manager


@pytest.mark.unit
class TestBaseThresholds:
    """Test base threshold functionality."""

    def test_base_thresholds_exist(self, manager):
        """Test dass alle base thresholds definiert sind."""
        assert len(manager.BASE_THRESHOLDS) == 7

        expected_goals = {
            "bass-kraft",
            "brillanz",
            "waerme",
            "natuerlichkeit",
            "authentizitaet",
            "emotionalitaet",
            "transparenz",
        }
        assert set(manager.BASE_THRESHOLDS.keys()) == expected_goals

    def test_base_thresholds_in_range(self, manager):
        """Test dass alle base thresholds im valid range sind."""
        for goal, threshold in manager.BASE_THRESHOLDS.items():
            assert 0.0 <= threshold <= 1.0

    def test_default_thresholds_returned(self, manager):
        """Test dass ohne context base thresholds returned werden."""
        thresholds = manager.get_thresholds()

        assert thresholds == manager.BASE_THRESHOLDS


class TestMediumAdjustments:
    """Test medium-specific adjustments."""

    def test_vinyl_adjustments(self, manager):
        """Test vinyl-specific adjustments."""
        thresholds = manager.get_thresholds(medium_type="vinyl")

        # Vinyl sollte Wärme erhöhen
        assert thresholds["waerme"] > manager.BASE_THRESHOLDS["waerme"]

        # Vinyl sollte Authentizität erhöhen
        assert thresholds["authentizitaet"] > manager.BASE_THRESHOLDS["authentizitaet"]

    def test_tape_adjustments(self, manager):
        """Test tape-specific adjustments."""
        thresholds = manager.get_thresholds(medium_type="tape")

        # Tape sollte maximale Wärme haben
        assert thresholds["waerme"] > manager.BASE_THRESHOLDS["waerme"]
        assert thresholds["waerme"] > manager.get_thresholds(medium_type="vinyl")["waerme"]

    def test_shellac_adjustments(self, manager):
        """Test shellac-specific adjustments."""
        thresholds = manager.get_thresholds(medium_type="shellac")

        # Shellac sollte Authentizität maximieren
        assert thresholds["authentizitaet"] > manager.BASE_THRESHOLDS["authentizitaet"]

        # Shellac sollte Brillanz reduzieren (limited HF)
        assert thresholds["brillanz"] < manager.BASE_THRESHOLDS["brillanz"]

    def test_digital_adjustments(self, manager):
        """Test digital-specific adjustments."""
        thresholds = manager.get_thresholds(medium_type="digital")

        # Digital sollte Transparenz erhöhen
        assert thresholds["transparenz"] > manager.BASE_THRESHOLDS["transparenz"]


class TestGenreAdjustments:
    """Test genre-specific adjustments."""

    def test_classical_adjustments(self, manager):
        """Test classical genre adjustments."""
        thresholds = manager.get_thresholds(genre="classical")

        # Classical: Natürlichkeit und Authentizität höchste Priorität
        assert thresholds["natuerlichkeit"] > manager.BASE_THRESHOLDS["natuerlichkeit"]
        assert thresholds["authentizitaet"] > manager.BASE_THRESHOLDS["authentizitaet"]
        assert thresholds["transparenz"] > manager.BASE_THRESHOLDS["transparenz"]

    def test_rock_adjustments(self, manager):
        """Test rock genre adjustments."""
        thresholds = manager.get_thresholds(genre="rock")

        # Rock: Bass-Kraft und Emotionalität wichtig
        assert thresholds["bass-kraft"] > manager.BASE_THRESHOLDS["bass-kraft"]
        assert thresholds["emotionalitaet"] > manager.BASE_THRESHOLDS["emotionalitaet"]

    def test_jazz_adjustments(self, manager):
        """Test jazz genre adjustments."""
        thresholds = manager.get_thresholds(genre="jazz")

        # Jazz: Wärme und Natürlichkeit
        assert thresholds["waerme"] > manager.BASE_THRESHOLDS["waerme"]
        assert thresholds["natuerlichkeit"] > manager.BASE_THRESHOLDS["natuerlichkeit"]

    def test_electronic_adjustments(self, manager):
        """Test electronic genre adjustments."""
        thresholds = manager.get_thresholds(genre="electronic")

        # Electronic: Bass-Kraft hoch, Natürlichkeit niedrig
        assert thresholds["bass-kraft"] > manager.BASE_THRESHOLDS["bass-kraft"]
        assert thresholds["natuerlichkeit"] < manager.BASE_THRESHOLDS["natuerlichkeit"]


class TestInstrumentAdjustments:
    """Test instrument focus adjustments."""

    def test_vocals_adjustments(self, manager):
        """Test vocals-specific adjustments."""
        thresholds = manager.get_thresholds(instrument_focus="vocals")

        # Vocals: Authentizität und Natürlichkeit wichtig
        assert thresholds["authentizitaet"] > manager.BASE_THRESHOLDS["authentizitaet"]
        assert thresholds["natuerlichkeit"] > manager.BASE_THRESHOLDS["natuerlichkeit"]

    def test_drums_adjustments(self, manager):
        """Test drums-specific adjustments."""
        thresholds = manager.get_thresholds(instrument_focus="drums")

        # Drums: Bass-Kraft und Transparenz
        assert thresholds["bass-kraft"] > manager.BASE_THRESHOLDS["bass-kraft"]
        assert thresholds["transparenz"] > manager.BASE_THRESHOLDS["transparenz"]

    def test_strings_adjustments(self, manager):
        """Test strings-specific adjustments."""
        thresholds = manager.get_thresholds(instrument_focus="strings")

        # Strings: Wärme und Emotionalität
        assert thresholds["waerme"] > manager.BASE_THRESHOLDS["waerme"]
        assert thresholds["emotionalitaet"] > manager.BASE_THRESHOLDS["emotionalitaet"]


class TestCombinedAdjustments:
    """Test combining multiple adjustment types."""

    def test_vinyl_jazz_vocals(self, manager):
        """Test vinyl + jazz + vocals combination."""
        thresholds = manager.get_thresholds(medium_type="vinyl", genre="jazz", instrument_focus="vocals")

        # Should combine all adjustments
        # Wärme from vinyl + jazz
        assert thresholds["waerme"] > manager.BASE_THRESHOLDS["waerme"] + 0.15

        # Natürlichkeit from jazz + vocals
        assert thresholds["natuerlichkeit"] > manager.BASE_THRESHOLDS["natuerlichkeit"]

    def test_digital_rock_drums(self, manager):
        """Test digital + rock + drums combination."""
        thresholds = manager.get_thresholds(medium_type="digital", genre="rock", instrument_focus="drums")

        # Bass-Kraft from rock (+0.10) + drums (+0.10) = +0.20
        # Base is 0.85, so should be high but might be clamped at 1.0
        base_bass = manager.BASE_THRESHOLDS["bass-kraft"]
        assert thresholds["bass-kraft"] >= base_bass + 0.15  # At least +0.15
        assert thresholds["bass-kraft"] <= 1.0  # But clamped at max

        # Transparenz from digital + drums
        assert thresholds["transparenz"] > manager.BASE_THRESHOLDS["transparenz"]

    def test_custom_adjustments(self, manager):
        """Test with additional custom adjustments."""
        custom = {"bass-kraft": +0.05, "brillanz": -0.03}

        thresholds = manager.get_thresholds(genre="jazz", custom_adjustments=custom)

        # Should include custom adjustments
        assert thresholds["bass-kraft"] > manager.get_thresholds(genre="jazz")["bass-kraft"]


class TestThresholdClamping:
    """Test that thresholds are clamped to valid range."""

    def test_thresholds_never_above_one(self, manager):
        """Test dass thresholds nie > 1.0 werden."""
        # Extreme combination
        thresholds = manager.get_thresholds(
            medium_type="vinyl",
            genre="classical",
            instrument_focus="strings",
            custom_adjustments={"natuerlichkeit": +0.50},
        )

        for goal, value in thresholds.items():
            assert value <= 1.0

    def test_thresholds_never_below_zero(self, manager):
        """Test dass thresholds nie < 0.0 werden."""
        # Extreme negative adjustments
        thresholds = manager.get_thresholds(genre="electronic", custom_adjustments={"natuerlichkeit": -0.50})

        for goal, value in thresholds.items():
            assert value >= 0.0


class TestThresholdProfiles:
    """Test threshold profile creation."""

    def test_get_profile(self, manager):
        """Test getting complete profile."""
        profile = manager.get_profile(genre="jazz", medium_type="vinyl", instrument_focus="vocals")

        assert isinstance(profile, ThresholdProfile)
        assert len(profile.thresholds) == 7
        assert profile.applies_to["genre"] == "jazz"
        assert profile.applies_to["medium_type"] == "vinyl"
        assert profile.applies_to["instrument_focus"] == "vocals"

    def test_profile_name_construction(self, manager):
        """Test profile name building."""
        profile = manager.get_profile(genre="rock", medium_type="vinyl")

        assert "Rock" in profile.name
        assert "Vinyl" in profile.name

    def test_create_custom_profile(self, manager):
        """Test creating custom profile."""
        custom_thresholds = {
            "bass-kraft": 0.92,
            "brillanz": 0.88,
            "waerme": 0.85,
            "natuerlichkeit": 0.95,
            "authentizitaet": 0.93,
            "emotionalitaet": 0.90,
            "transparenz": 0.91,
        }

        profile = manager.create_custom_profile(
            name="My Custom Profile", thresholds=custom_thresholds, description="Test profile"
        )

        assert profile.name == "My Custom Profile"
        assert profile.thresholds == custom_thresholds
        assert len(manager.custom_profiles) == 1

    def test_custom_profile_validation(self, manager):
        """Test dass custom profiles validiert werden."""
        invalid_thresholds = {
            "bass-kraft": 1.5,  # > 1.0
            "brillanz": 0.88,
            "waerme": 0.85,
            "natuerlichkeit": 0.95,
            "authentizitaet": 0.93,
            "emotionalitaet": 0.90,
            "transparenz": 0.91,
        }

        with pytest.raises(ValueError):
            manager.create_custom_profile(name="Invalid Profile", thresholds=invalid_thresholds)


class TestProfilePersistence:
    """Test saving and loading custom profiles."""

    def test_save_custom_profiles(self, manager_with_profiles):
        """Test saving custom profiles to file."""
        manager = manager_with_profiles

        # Create custom profile
        manager.create_custom_profile(
            name="Test Profile", thresholds=manager.BASE_THRESHOLDS.copy(), description="Test"
        )

        # Save
        manager.save_custom_profiles()

        # File should exist
        assert manager.profiles_path.exists()

        # Load and verify
        with open(manager.profiles_path) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["name"] == "Test Profile"

    def test_load_custom_profiles(self):
        """Test loading custom profiles from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiles_path = Path(tmpdir) / "profiles.json"

            # Create profile file
            profiles_data = [
                {
                    "name": "Loaded Profile",
                    "thresholds": {
                        "bass-kraft": 0.90,
                        "brillanz": 0.88,
                        "waerme": 0.85,
                        "natuerlichkeit": 0.95,
                        "authentizitaet": 0.93,
                        "emotionalitaet": 0.90,
                        "transparenz": 0.91,
                    },
                    "description": "Test loaded profile",
                    "applies_to": {"genre": "jazz"},
                }
            ]

            with open(profiles_path, "w") as f:
                json.dump(profiles_data, f)

            # Load
            manager = AdaptiveThresholdsManager(profiles_path=profiles_path)

            assert len(manager.custom_profiles) == 1
            assert manager.custom_profiles[0].name == "Loaded Profile"


class TestSupportedContexts:
    """Test getting supported contexts."""

    def test_get_all_genres(self, manager):
        """Test getting all supported genres."""
        genres = manager.get_all_genres()

        assert len(genres) >= 10
        assert "classical" in genres
        assert "jazz" in genres
        assert "rock" in genres
        assert "electronic" in genres

    def test_get_all_medium_types(self, manager):
        """Test getting all supported medium types."""
        mediums = manager.get_all_medium_types()

        assert len(mediums) >= 4
        assert "vinyl" in mediums
        assert "tape" in mediums
        assert "shellac" in mediums
        assert "digital" in mediums

    def test_get_all_instruments(self, manager):
        """Test getting all supported instruments."""
        instruments = manager.get_all_instruments()

        assert len(instruments) >= 7
        assert "vocals" in instruments
        assert "drums" in instruments
        assert "strings" in instruments


class TestProfileComparison:
    """Test comparing threshold profiles."""

    def test_compare_profiles(self, manager):
        """Test comparing two profiles."""
        profile1 = {"genre": "jazz"}
        profile2 = {"genre": "rock"}

        differences = manager.compare_profiles(profile1, profile2)

        assert len(differences) == 7

        # Rock has higher bass-kraft than jazz
        assert differences["bass-kraft"] > 0

        # Jazz has higher warmth than rock
        assert differences["waerme"] < 0


class TestStatistics:
    """Test statistics tracking."""

    def test_calibration_count(self, manager):
        """Test that calibration count increases."""
        initial = manager.calibration_count

        manager.get_thresholds(genre="jazz")
        manager.get_thresholds(medium_type="vinyl")

        assert manager.calibration_count == initial + 2

    def test_get_statistics(self, manager):
        """Test getting statistics."""
        # Create custom profile
        manager.create_custom_profile(name="Test", thresholds=manager.BASE_THRESHOLDS.copy())

        stats = manager.get_statistics()

        assert "calibration_count" in stats
        assert "custom_profiles" in stats
        assert "supported_genres" in stats
        assert "supported_mediums" in stats
        assert "supported_instruments" in stats
        assert "base_thresholds" in stats

        assert stats["custom_profiles"] == 1
        assert stats["supported_genres"] >= 10
        assert stats["supported_mediums"] >= 4
