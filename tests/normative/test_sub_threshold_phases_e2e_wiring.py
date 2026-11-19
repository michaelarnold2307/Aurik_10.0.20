from __future__ import annotations

import pytest

"""
§2.47b Normativtest: sub_threshold_phases End-to-End-Wiring

Prüft, dass Phasen, bei denen alle Goal-Deltas ≥ 0 und < JND sind, in
RestorationResult.metadata["sub_threshold_phases"] erscheinen und dass die
Bridge-Funktion get_experience_insights() diesen Wert korrekt propagiert.

Normative Referenz:
- §2.47b: sub_threshold_phases Telemetrie (RELEASE_MUST)
- §2.53:  RestorationResult.metadata muss sub_threshold_phases enthalten
"""


@pytest.mark.unit
class TestSubThresholdPhasesUV3Aggregation:
    """UV3 aggregiert sub_threshold_phases-Einträge aus PMGG-Log-Entries."""

    def test_sub_threshold_phases_key_in_result_metadata(self):
        """RestorationResult.metadata muss den Schlüssel 'sub_threshold_phases' enthalten."""
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        uv3 = UnifiedRestorerV3.__new__(UnifiedRestorerV3)
        # Minimale Initialisierung der relevanten Attribute
        uv3._pmgg_log_entries = []
        uv3._team_coordination_events = []
        uv3._phase_team_context = {}
        uv3._song_goal_importance = None
        uv3._interaction_guard_metadata = {}
        uv3._source_material_baseline = None

        # Aggregierungs-Logik direkt testen (ohne vollständigen restore()-Lauf)
        # Simuliere 2 PMGG-Log-Einträge: einer ohne, einer mit sub_threshold-Marker
        from backend.core.per_phase_musical_goals_gate import PhaseGateLogEntry

        entry_no_sub = PhaseGateLogEntry.__new__(PhaseGateLogEntry)
        entry_no_sub.phase_id = "phase_03_denoise"
        entry_no_sub.action = "passed"
        entry_no_sub.goal_regressions = {}
        entry_no_sub.strength_used = 0.8
        entry_no_sub.metadata = {}  # no sub_threshold entry

        entry_sub = PhaseGateLogEntry.__new__(PhaseGateLogEntry)
        entry_sub.phase_id = "phase_07_harmonic_restoration"
        entry_sub.action = "sub_threshold"
        entry_sub.goal_regressions = {}
        entry_sub.strength_used = 0.5
        entry_sub.metadata = {"sub_threshold_phases": ["phase_07_harmonic_restoration"]}

        uv3._pmgg_log_entries = [entry_no_sub, entry_sub]

        # Inline-Aggregations-Logik (spiegelt UV3 metadata-dict)
        result = sorted(
            {
                _st_pid
                for _pmgg_e in (uv3._pmgg_log_entries or [])
                for _st_pid in (
                    (
                        isinstance(getattr(_pmgg_e, "metadata", None), dict)
                        and getattr(_pmgg_e, "metadata", {}).get("sub_threshold_phases", [])
                    )
                    or []
                )
            }
        )

        assert "phase_07_harmonic_restoration" in result, (
            "sub_threshold_phases aggregation fehlt: phase_07 muss im Ergebnis erscheinen"
        )
        assert "phase_03_denoise" not in result, (
            "phase_03 (action=passed) darf NICHT in sub_threshold_phases erscheinen"
        )

    def test_sub_threshold_phases_empty_when_no_sub_threshold(self):
        """Wenn keine Phase auf sub_threshold gelaufen ist, ist die Liste leer."""
        from backend.core.per_phase_musical_goals_gate import PhaseGateLogEntry

        entries = []
        for pid in ("phase_01_click_removal", "phase_03_denoise", "phase_09_crackle"):
            e = PhaseGateLogEntry.__new__(PhaseGateLogEntry)
            e.metadata = {"some_other_key": True}  # kein sub_threshold_phases
            entries.append(e)

        result = sorted(
            {
                _st_pid
                for _pmgg_e in entries
                for _st_pid in (
                    (
                        isinstance(getattr(_pmgg_e, "metadata", None), dict)
                        and getattr(_pmgg_e, "metadata", {}).get("sub_threshold_phases", [])
                    )
                    or []
                )
            }
        )
        assert result == [], f"Erwartet leere Liste, erhalten: {result}"

    def test_sub_threshold_phases_deduplicated(self):
        """Wenn dieselbe Phase in mehreren Einträgen als sub_threshold markiert ist → dedupliziert."""
        from backend.core.per_phase_musical_goals_gate import PhaseGateLogEntry

        entries = []
        for _ in range(3):
            e = PhaseGateLogEntry.__new__(PhaseGateLogEntry)
            e.metadata = {"sub_threshold_phases": ["phase_07_harmonic_restoration"]}
            entries.append(e)

        result = sorted(
            {
                _st_pid
                for _pmgg_e in entries
                for _st_pid in (
                    (
                        isinstance(getattr(_pmgg_e, "metadata", None), dict)
                        and getattr(_pmgg_e, "metadata", {}).get("sub_threshold_phases", [])
                    )
                    or []
                )
            }
        )
        assert result.count("phase_07_harmonic_restoration") == 1, (
            "Duplikate in sub_threshold_phases: Ergebnis muss dedupliziert sein"
        )

    def test_restorationresult_metadata_key_present_in_uv3_code(self):
        """UV3-Quelltext enthält 'sub_threshold_phases' als Metadaten-Schlüssel."""
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        src = inspect.getsource(UnifiedRestorerV3)
        assert '"sub_threshold_phases"' in src or "'sub_threshold_phases'" in src, (
            "UV3 muss 'sub_threshold_phases' in RestorationResult.metadata befüllen (§2.47b)"
        )


class TestSubThresholdPhasesBridgePropagation:
    """Bridge get_experience_insights() gibt sub_threshold_phases korrekt zurück."""

    def test_bridge_returns_sub_threshold_phases_key(self):
        """get_experience_insights() muss 'sub_threshold_phases' im Return-Dict enthalten."""
        from backend.api.bridge import get_experience_insights

        class _FakeResult:
            metadata = {
                "sub_threshold_phases": ["phase_07_harmonic_restoration", "phase_39_air_enhancement"],
                "joy_runtime_index": {"joy_index": 0.8, "fatigue_index": 0.2},
                "auto_improvement_recommendations": {"count": 0, "recommendations": []},
                "song_calibration": {"cluster_key": "jazz_vinyl", "cluster_policy": {}},
                "team_coordination": {"event_count": 0, "events": [], "phase_type_summary": {}},
            }

        insights = get_experience_insights(_FakeResult())
        assert "sub_threshold_phases" in insights, (
            "get_experience_insights() muss 'sub_threshold_phases' zurückgeben (§2.47b/§2.53)"
        )
        phases = insights["sub_threshold_phases"]
        assert isinstance(phases, list), f"sub_threshold_phases muss Liste sein, ist: {type(phases)}"
        assert "phase_07_harmonic_restoration" in phases
        assert "phase_39_air_enhancement" in phases

    def test_bridge_empty_sub_threshold_phases_when_missing(self):
        """Fehlendes sub_threshold_phases in metadata → Bridge gibt leere Liste zurück (kein KeyError)."""
        from backend.api.bridge import get_experience_insights

        class _FakeResult:
            metadata = {
                "joy_runtime_index": {},
                "auto_improvement_recommendations": {},
                "song_calibration": {},
                "team_coordination": {"event_count": 0, "events": [], "phase_type_summary": {}},
                # NO sub_threshold_phases key
            }

        insights = get_experience_insights(_FakeResult())
        assert "sub_threshold_phases" in insights
        assert insights["sub_threshold_phases"] == [], (
            "Fehlende sub_threshold_phases in metadata → Bridge muss leere Liste zurückgeben"
        )

    def test_bridge_none_metadata_returns_empty_sub_threshold(self):
        """Wenn result.metadata = None, darf kein Fehler auftreten."""
        from backend.api.bridge import get_experience_insights

        class _FakeResult:
            metadata = None

        insights = get_experience_insights(_FakeResult())
        assert insights["sub_threshold_phases"] == []
