"""Tests für neue §2.59-Module: ContractValidator, SafeDict, QualityMode, DefectManifest."""

import pytest


@pytest.mark.unit
class TestSafeDict:
    """SafeDict: .get()-Masking-Detektor."""

    def test_known_key_no_warning(self):
        from backend.core.safe_dict import SafeDict

        d = SafeDict({"clicks": 0.8}, name="test", known_keys={"clicks", "hum"})
        assert d.get("clicks", 0.0) == 0.8

    def test_unknown_key_returns_default(self):
        from backend.core.safe_dict import SafeDict

        d = SafeDict({"clicks": 0.8}, name="test", known_keys={"clicks"})
        # Should return default, not raise
        assert d.get("hiss", 0.0) == 0.0

    def test_missing_key_still_returns_default(self):
        from backend.core.safe_dict import SafeDict

        d = SafeDict({}, name="test", known_keys=set())
        assert d.get("anything", 42) == 42

    def test_warn_disabled(self):
        from backend.core.safe_dict import SafeDict

        d = SafeDict(
            {"clicks": 0.8},
            name="test",
            known_keys={"clicks"},
            warn_on_missing=False,
        )
        assert d.get("hiss", 0.0) == 0.0  # No warning when disabled

    def test_direct_access_still_raises(self):
        from backend.core.safe_dict import SafeDict

        d = SafeDict({"clicks": 0.8}, name="test", known_keys={"clicks"})
        with pytest.raises(KeyError):
            d["hiss"]


class TestQualityMode:
    """QualityMode: Validierung von Mode-Strings."""

    def test_valid_modes(self):
        from backend.core.quality_mode import validate_mode

        assert validate_mode("restoration") == "quality"
        assert validate_mode("studio_2026") == "maximum"
        assert validate_mode("balanced") == "balanced"
        assert validate_mode("fast") == "fast"
        assert validate_mode("quality") == "quality"

    def test_aliases(self):
        from backend.core.quality_mode import validate_mode

        assert validate_mode("Restoration") == "quality"
        assert validate_mode(" RESTORATION ") == "quality"

    def test_invalid_falls_back(self):
        from backend.core.quality_mode import validate_mode

        assert validate_mode("garbage") == "quality"
        assert validate_mode("") == "quality"

    def test_none_falls_back(self):
        from backend.core.quality_mode import validate_mode

        assert validate_mode(None) == "quality"

    def test_known_typo_gets_fallback(self):
        from backend.core.quality_mode import validate_mode

        # "restoraton" is a common typo
        result = validate_mode("restoraton")
        assert result == "quality"  # Falls back


class TestContractValidator:
    """ContractValidator: Cross-Module-Konsistenz."""

    def test_runs_without_error(self):
        from backend.core.defect_contract_validator import run_contract_validation

        result = run_contract_validation()
        assert isinstance(result, dict)
        assert "ok" in result
        assert "violations" in result

    def test_current_state_is_clean(self):
        from backend.core.defect_contract_validator import run_contract_validation

        result = run_contract_validation()
        assert result["ok"], f"Expected 0 violations, got: {result['details']}"
        assert result["violations"] == 0

    def test_contract_violation_repr(self):
        from backend.core.defect_contract_validator import ContractViolation

        cv = ContractViolation("A", "B", "detail")
        assert "A" in str(cv)
        assert "B" in str(cv)
        assert "detail" in str(cv)


class TestDefectManifest:
    """DefectManifest: Kanonische Defekt-Registry."""

    def test_manifest_initializes(self):
        from backend.core.defect_manifest import get_defect_manifest

        dm = get_defect_manifest()
        assert dm is not None
        assert len(dm._entries) > 50  # 54+ DefectTypes

    def test_known_defect_has_phases(self):
        from backend.core.defect_manifest import get_defect_manifest

        dm = get_defect_manifest()
        phases = dm.get_phases_for_defect("clicks")
        assert len(phases) > 0
        assert "phase_01_click_removal" in phases

    def test_known_defect_has_goals(self):
        from backend.core.defect_manifest import get_defect_manifest

        dm = get_defect_manifest()
        goals = dm.get_goals_for_defect("bandwidth_loss")
        assert len(goals) > 0
        assert "brillanz" in goals

    def test_known_defect_has_strength(self):
        from backend.core.defect_manifest import get_defect_manifest

        dm = get_defect_manifest()
        strength = dm.get_strength_category("clicks")
        assert strength.category == "gentle"

    def test_soft_saturation_is_preserved(self):
        from backend.core.defect_manifest import get_defect_manifest

        dm = get_defect_manifest()
        strength = dm.get_strength_category("soft_saturation")
        assert strength.category == "preserve"
        assert strength.max_strength == 0.0  # Never repair saturation

    def test_unknown_defect_gets_neutral(self):
        from backend.core.defect_manifest import get_defect_manifest

        dm = get_defect_manifest()
        strength = dm.get_strength_category("nonexistent_defect")
        assert strength.category == "neutral"

    def test_get_all_phases_deduplicates(self):
        from backend.core.defect_manifest import get_defect_manifest

        dm = get_defect_manifest()
        phases = dm.get_all_phases_for_defects(["clicks", "crackle"])
        assert len(phases) == len(set(phases))  # No duplicates
        assert "phase_01_click_removal" in phases

    def test_as_dict_exports(self):
        from backend.core.defect_manifest import get_defect_manifest

        dm = get_defect_manifest()
        exported = dm.as_dict()
        assert isinstance(exported, dict)
        assert "clicks" in exported
        assert "phases" in exported["clicks"]
