"""
Tests for Processing Modes

Component 4.2: Processing Modes Tests
Tests all 5 processing modes, mode switching, goal validation, and recommendations.
"""

import sys
from pathlib import Path

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.musical_goals.processing_modes import (
    PROCESSING_MODE_CONFIGS,
    ProcessingMode,
    ProcessingModeManager,
    get_mode_from_string,
)

# =============================================================================
# Test Processing Mode Configs
# =============================================================================


@pytest.mark.unit
class TestProcessingModeConfigs:
    """Test that all processing mode configurations are valid."""

    def test_all_modes_have_configs(self):
        """All ProcessingMode enum values have configurations."""
        for mode in ProcessingMode:
            assert mode in PROCESSING_MODE_CONFIGS, f"Missing config for {mode}"

    def test_configs_have_all_musical_goals(self):
        """All configs define all 7 Musical Goals."""
        expected_goals = {
            "brillanz",
            "waerme",
            "natuerlichkeit",
            "authentizitaet",
            "emotionalitaet",
            "transparenz",
            "bass-kraft",
        }

        for mode, config in PROCESSING_MODE_CONFIGS.items():
            assert set(config.musical_goals.keys()) == expected_goals, f"{mode.value} missing goals"

    def test_musical_goals_in_valid_range(self):
        """All Musical Goals values are in [0.7, 1.0] range."""
        for mode, config in PROCESSING_MODE_CONFIGS.items():
            for goal, value in config.musical_goals.items():
                assert 0.7 <= value <= 1.0, f"{mode.value}: {goal} = {value} out of range"

    def test_goal_weights_exist(self):
        """All configs have goal weights."""
        for mode, config in PROCESSING_MODE_CONFIGS.items():
            assert len(config.goal_weights) == 7, f"{mode.value} has {len(config.goal_weights)} weights (expected 7)"

    def test_goal_weights_in_valid_range(self):
        """Goal weights are in [0.0, 1.0] range."""
        for mode, config in PROCESSING_MODE_CONFIGS.items():
            for goal, weight in config.goal_weights.items():
                assert 0.0 <= weight <= 1.0, f"{mode.value}: {goal} weight = {weight} out of range"

    def test_quality_thresholds_exist(self):
        """All configs have quality thresholds."""
        for mode, config in PROCESSING_MODE_CONFIGS.items():
            assert len(config.quality_thresholds) == 7, f"{mode.value} has {len(config.quality_thresholds)} thresholds"

    def test_thresholds_less_than_targets(self):
        """Quality thresholds are less than or equal to target goals."""
        for mode, config in PROCESSING_MODE_CONFIGS.items():
            for goal in config.musical_goals.keys():
                threshold = config.quality_thresholds[goal]
                target = config.musical_goals[goal]
                assert threshold <= target, f"{mode.value}: {goal} threshold {threshold} > target {target}"

    def test_processing_params_exist(self):
        """All configs have processing parameters."""
        for mode, config in PROCESSING_MODE_CONFIGS.items():
            assert len(config.processing_params) > 0, f"{mode.value} has no processing params"


# =============================================================================
# Test Mode-Specific Characteristics
# =============================================================================


class TestModeCharacteristics:
    """Test that each mode has expected priority characteristics."""

    def test_studio_2026_prioritizes_brilliance(self):
        """STUDIO_2026 has highest brilliance and transparency."""
        config = PROCESSING_MODE_CONFIGS[ProcessingMode.STUDIO_2026]

        assert config.musical_goals["brillanz"] >= 0.93
        assert config.musical_goals["transparenz"] >= 0.93
        assert config.goal_weights["brillanz"] >= 0.95
        assert config.goal_weights["transparenz"] >= 0.95

    def test_restoration_is_balanced(self):
        """RESTORATION has balanced goals (all close to each other)."""
        config = PROCESSING_MODE_CONFIGS[ProcessingMode.RESTORATION]

        goals = list(config.musical_goals.values())
        max_goal = max(goals)
        min_goal = min(goals)

        # Should have small range (balanced)
        assert max_goal - min_goal <= 0.10, f"RESTORATION not balanced: range = {max_goal - min_goal}"


# =============================================================================
# Test ProcessingModeManager
# =============================================================================


class TestProcessingModeManager:
    """Test ProcessingModeManager functionality."""

    def test_initialization_default(self):
        """Manager initializes with RESTORATION by default."""
        manager = ProcessingModeManager()
        assert manager.current_mode == ProcessingMode.RESTORATION
        assert manager.config.mode == ProcessingMode.RESTORATION

    def test_initialization_custom_mode(self):
        """Manager can initialize with custom mode."""
        manager = ProcessingModeManager(mode=ProcessingMode.STUDIO_2026)
        assert manager.current_mode == ProcessingMode.STUDIO_2026

    def test_set_mode(self):
        """set_mode() wechselt korrekt zwischen RESTORATION und STUDIO_2026."""
        manager = ProcessingModeManager()
        manager.set_mode(ProcessingMode.STUDIO_2026)
        assert manager.current_mode == ProcessingMode.STUDIO_2026
        assert manager.config.mode == ProcessingMode.STUDIO_2026

    def test_get_musical_goals_for_mode(self):
        """get_musical_goals_for_mode() returns correct goals."""
        manager = ProcessingModeManager()

        goals = manager.get_musical_goals_for_mode(ProcessingMode.STUDIO_2026)
        assert "brillanz" in goals
        assert goals["brillanz"] >= 0.93

    def test_get_processing_params_for_mode(self):
        """get_processing_params_for_mode() gibt korrekte Parameter zurück."""
        manager = ProcessingModeManager()
        params = manager.get_processing_params_for_mode(ProcessingMode.RESTORATION)
        assert isinstance(params, dict)

    def test_validate_goals_all_pass(self):
        """validate_goals_against_mode() passes when all goals meet thresholds."""
        manager = ProcessingModeManager(ProcessingMode.RESTORATION)

        # Perfect scores
        achieved_goals = {
            "brillanz": 0.90,
            "waerme": 0.90,
            "natuerlichkeit": 0.90,
            "authentizitaet": 0.90,
            "emotionalitaet": 0.90,
            "transparenz": 0.90,
            "bass-kraft": 0.90,
        }

        result = manager.validate_goals_against_mode(achieved_goals)

        assert result["passed"] is True
        assert len(result["violations"]) == 0
        assert result["overall_score"] > 0.95

    def test_validate_goals_some_fail(self):
        """validate_goals_against_mode() fails when goals below thresholds."""
        manager = ProcessingModeManager(ProcessingMode.STUDIO_2026)

        # Low brilliance (STUDIO_2026 requires high brilliance)
        achieved_goals = {
            "brillanz": 0.70,  # Too low
            "waerme": 0.85,
            "natuerlichkeit": 0.85,
            "authentizitaet": 0.85,
            "emotionalitaet": 0.85,
            "transparenz": 0.85,
            "bass-kraft": 0.85,
        }

        result = manager.validate_goals_against_mode(achieved_goals)

        assert result["passed"] is False
        assert len(result["violations"]) > 0
        assert "brillanz" in result["violations"][0]

    def test_validate_goals_scores_structure(self):
        """validate_goals_against_mode() returns correct scores structure."""
        manager = ProcessingModeManager()

        achieved_goals = {
            "brillanz": 0.85,
            "waerme": 0.85,
            "natuerlichkeit": 0.88,
            "authentizitaet": 0.88,
            "emotionalitaet": 0.85,
            "transparenz": 0.88,
            "bass-kraft": 0.85,
        }

        result = manager.validate_goals_against_mode(achieved_goals)

        assert "scores" in result
        assert "brillanz" in result["scores"]

        brillanz_score = result["scores"]["brillanz"]
        assert "target" in brillanz_score
        assert "achieved" in brillanz_score
        assert "threshold" in brillanz_score
        assert "passed" in brillanz_score
        assert "weight" in brillanz_score

    def test_compare_modes(self):
        """compare_modes() returns sorted results for all modes."""
        manager = ProcessingModeManager()

        # High warmth, low brilliance
        achieved_goals = {
            "brillanz": 0.70,
            "waerme": 0.95,
            "natuerlichkeit": 0.92,
            "authentizitaet": 0.88,
            "emotionalitaet": 0.95,
            "transparenz": 0.75,
            "bass-kraft": 0.82,
        }

        results = manager.compare_modes(achieved_goals)

        # Es gibt nur 2 gültige Modi: RESTORATION und STUDIO_2026
        assert len(results) == 2  # Alle gültigen Modi

        # Check all modes present
        modes_returned = {r["mode"] for r in results}
        expected_modes = {m.value for m in ProcessingMode}
        assert modes_returned == expected_modes

        # Results sorted by overall_score (descending)
        for i in range(len(results) - 1):
            assert results[i]["overall_score"] >= results[i + 1]["overall_score"]

        # Mit den aktuellen Modi sollte restoration oder studio_2026 oben stehen
        top_mode = results[0]["mode"]
        assert top_mode in ["restoration", "studio_2026"]


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_mode_from_string_valid(self):
        """get_mode_from_string() wandelt gültige Strings um."""
        assert get_mode_from_string("restoration") == ProcessingMode.RESTORATION
        assert get_mode_from_string("STUDIO_2026") == ProcessingMode.STUDIO_2026

    def test_get_mode_from_string_invalid(self):
        """get_mode_from_string() wirft ValueError für ungültige Strings."""
        with pytest.raises(ValueError):
            get_mode_from_string("invalid_mode")


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for realistic scenarios."""

    def test_studio_2026_workflow(self):
        """Kompletter Workflow: STUDIO_2026 auswählen und modernen Mix validieren."""
        manager = ProcessingModeManager(ProcessingMode.STUDIO_2026)
        modern_mix = {
            "brillanz": 0.93,
            "waerme": 0.78,
            "natuerlichkeit": 0.80,
            "authentizitaet": 0.78,
            "emotionalitaet": 0.88,
            "transparenz": 0.93,
            "bass-kraft": 0.90,
        }
        result = manager.validate_goals_against_mode(modern_mix)
        assert result["passed"] is True
        assert result["mode"] == ProcessingMode.STUDIO_2026.value


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
