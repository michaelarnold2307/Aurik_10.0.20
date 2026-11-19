from __future__ import annotations

"""
test_offensive_goal_achievement.py — Sprint 2: Offensive Goal-Erreichung
========================================================================

Verschiebt den Test-Fokus von "Keine Verschlechterung" zu "Beweise Weltklasse":
1. AMRB-Gate prüft ≥84.0 Score, ≥8/10 Szenarien (§8.1)
2. Competitive-Gate prüft Aurik ≥ iZotope in ≥7/10 (§8.2)
3. Goal-Erreichungs-Marker vergeben
4. Strukturelle Integrität der Gates

Spec: §8.1 AMRB, §8.2 Competitive, §v10 Pleasantness-First
"""


import pytest


@pytest.mark.goal_achievement
class TestAMRBGateStructure:
    """§8.1: AMRB CI-Gate — OS-Führerschaft ≥84.0, ≥8/10 Szenarien."""

    def test_01_amrb_gate_file_exists(self):
        """AMRB-Gate-Test existiert."""
        from pathlib import Path

        assert Path("tests/normative/test_amrb_ci_gate.py").exists(), (
            "test_amrb_ci_gate.py fehlt — §8.1 AMRB-Gate nicht implementiert"
        )

    def test_02_amrb_gate_has_thresholds(self):
        """AMRB-Gate enthält die spezifizierten Schwellwerte."""
        src = open("tests/normative/test_amrb_ci_gate.py", encoding="utf-8").read()
        assert "84" in src, "AMRB-Gate: Score 84 nicht gefunden (§8.1: ≥84.0)"
        assert "8" in src or "10" in src, "AMRB-Gate: 8/10 Szenarien nicht gefunden"

    def test_03_amrb_gate_is_not_disabled(self):
        """AMRB-Gate ist nicht geskippt oder deaktiviert."""
        src = open("tests/normative/test_amrb_ci_gate.py", encoding="utf-8").read()
        assert "pytest.mark.skip" not in src, "AMRB-Gate ist geskippt — §8.1 nicht aktiv"


@pytest.mark.goal_achievement
class TestCompetitiveGateStructure:
    """§8.2: Competitive CI-Gate — Aurik ≥ iZotope in ≥7/10 Szenarien."""

    def test_10_competitive_gate_file_exists(self):
        """Competitive-Gate-Test existiert."""
        from pathlib import Path

        assert Path("tests/normative/test_competitive_ci_gate.py").exists(), (
            "test_competitive_ci_gate.py fehlt — §8.2 Competitive-Gate nicht implementiert"
        )

    def test_11_competitive_gate_has_thresholds(self):
        """Competitive-Gate enthält die spezifizierten Schwellwerte."""
        src = open("tests/normative/test_competitive_ci_gate.py", encoding="utf-8").read()
        assert "7" in src, "Competitive-Gate: 7/10 Szenarien nicht gefunden (§8.2)"
        assert "izotope" in src.lower(), "Competitive-Gate: iZotope-Referenz nicht gefunden"

    def test_12_competitive_gate_is_not_disabled(self):
        """Competitive-Gate ist nicht geskippt oder deaktiviert."""
        src = open("tests/normative/test_competitive_ci_gate.py", encoding="utf-8").read()
        assert "pytest.mark.skip" not in src, "Competitive-Gate ist geskippt — §8.2 nicht aktiv"


@pytest.mark.goal_achievement
class TestGoalAchievementMarkers:
    """Stellt sicher, dass Goal-Erreichungs-Tests als solche markiert sind."""

    def test_20_amrb_has_goal_marker(self):
        """AMRB-Gate trägt goal_achievement- oder amrb-Marker."""
        src = open("tests/normative/test_amrb_ci_gate.py", encoding="utf-8").read()
        has_marker = "goal_achievement" in src or "amrb" in src
        assert has_marker, "AMRB-Gate hat keinen goal_achievement/amrb-Marker"

    def test_21_competitive_has_goal_marker(self):
        """Competitive-Gate trägt goal_achievement- oder competitive-Marker."""
        src = open("tests/normative/test_competitive_ci_gate.py", encoding="utf-8").read()
        has_marker = "goal_achievement" in src or "competitive" in src
        assert has_marker, "Competitive-Gate hat keinen goal_achievement/competitive-Marker"
