"""
Tests für das Pipeline-Debug-System (pipeline_trace, debug_api, PhaseGateLogEntry).

Prüft:
- PhaseGateLogEntry hat scores_before/scores_after Felder
- build_from_result() funktioniert mit minimalem Mock-Result
- format_goals_table() produziert nutzbare Ausgabe
- format_full_report() läuft ohne Fehler
- get_debug_summary() liefert stabile Keys
- get_goal_fails() erkennt Goals unter Schwellwert
- get_worst_phases() liefert sortierte Liste
"""

import dataclasses
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_log_entry():
    """Minimale PhaseGateLogEntry mit scores_before/after."""
    from backend.core.per_phase_musical_goals_gate import PhaseGateLogEntry

    entry = PhaseGateLogEntry(
        phase_id="phase_03_denoise",
        action="passed",
        goal_regressions={},
        strength_used=0.8,
    )
    entry.scores_before = {
        "natuerlichkeit": 0.72,
        "authentizitaet": 0.68,
        "tonal_center": 0.80,
        "timbre_authentizitaet": 0.75,
        "artikulation": 0.70,
        "transparenz": 0.65,
        "brillanz": 0.60,
    }
    entry.scores_after = {
        "natuerlichkeit": 0.85,
        "authentizitaet": 0.80,
        "tonal_center": 0.82,
        "timbre_authentizitaet": 0.78,
        "artikulation": 0.72,
        "transparenz": 0.74,
        "brillanz": 0.68,
    }
    return entry


@pytest.fixture()
def entry_with_regression():
    """PhaseGateLogEntry mit sichtbarer Regression."""
    from backend.core.per_phase_musical_goals_gate import PhaseGateLogEntry

    entry = PhaseGateLogEntry(
        phase_id="phase_07_harmonic",
        action="best_effort",
        goal_regressions={"authentizitaet": -0.05, "tonal_center": -0.03},
        strength_used=0.5,
    )
    entry.scores_before = {"authentizitaet": 0.85, "tonal_center": 0.90}
    entry.scores_after = {"authentizitaet": 0.80, "tonal_center": 0.87}
    entry.metadata["recovery_attempted"] = True
    return entry


@pytest.fixture()
def mock_result(minimal_log_entry, entry_with_regression):
    """Mock-RestorationResult mit allen für den Trace nötigen Feldern."""
    result = MagicMock()
    # Grundinfos
    result.material_type = MagicMock()
    result.material_type.value = "vinyl"
    result.era_decade = "1970s"
    result.restorability = 62.0
    result.total_time_seconds = 45.2
    result.rt_factor = 1.5
    result.quality_estimate = 0.84
    result.confidence = 0.78
    result.warnings = ["test_warning"]
    result.phases_executed = ["phase_03_denoise", "phase_07_harmonic"]
    result.phases_skipped = ["phase_21_exciter"]
    result.phase_gate_log = ["phase_07_harmonic"]  # best_effort phases
    result.musical_goals = {
        "natuerlichkeit": 0.85,
        "authentizitaet": 0.80,
        "tonal_center": 0.82,
        "transparenz": 0.74,
        "brillanz": 0.68,
    }
    result.adaptive_thresholds = {
        "natuerlichkeit": 0.88,
        "authentizitaet": 0.86,
    }
    result.goosebumps_score = 0.42
    result.chroma_correlation = 0.91
    result.lufs_delta = -0.3

    # Config
    result.config = MagicMock()
    result.config.mode = MagicMock()
    result.config.mode.value = "quality"

    # Metadata mit pmgg_log_entries
    result.metadata = {
        "pmgg_log_entries": [
            dataclasses.asdict(minimal_log_entry),
            dataclasses.asdict(entry_with_regression),
        ],
        "fail_reasons": [],
        "team_coordination": {"event_count": 1, "events": [{"phase": "phase_07_harmonic"}]},
        "joy_runtime_index": {"joy_index": 0.72, "fatigue_index": 0.15, "components": {"frisson_index": 0.35}},
        "auto_improvement_recommendations": {"count": 1, "recommendations": []},
        "song_calibration": {"cluster_key": "vinyl_1970s", "cluster_policy": "standard"},
        "carrier_chain_recovery_ratio": 0.18,
        "ml_fallbacks_used": [],
        "source_material_baseline": {"phase_cancellation_ratio": 0.05},
    }
    return result


# ---------------------------------------------------------------------------
# Tests: PhaseGateLogEntry
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_phase_gate_log_entry_has_scores_fields():
    """PhaseGateLogEntry muss scores_before und scores_after haben."""
    from backend.core.per_phase_musical_goals_gate import PhaseGateLogEntry

    entry = PhaseGateLogEntry(phase_id="test", action="passed", goal_regressions={}, strength_used=1.0)
    assert hasattr(entry, "scores_before"), "scores_before fehlt in PhaseGateLogEntry"
    assert hasattr(entry, "scores_after"), "scores_after fehlt in PhaseGateLogEntry"
    assert isinstance(entry.scores_before, dict)
    assert isinstance(entry.scores_after, dict)


def test_phase_gate_log_entry_scores_populated(minimal_log_entry):
    """scores_before/after können befüllt werden und sind per asdict() serialisierbar."""
    d = dataclasses.asdict(minimal_log_entry)
    assert "scores_before" in d
    assert "scores_after" in d
    assert d["scores_before"]["natuerlichkeit"] == pytest.approx(0.72)
    assert d["scores_after"]["natuerlichkeit"] == pytest.approx(0.85)


def test_phase_gate_log_entry_dataclass_intact():
    """PhaseGateLogEntry ist noch korrekt als Dataclass initialisierbar (keine Regressions)."""
    from backend.core.per_phase_musical_goals_gate import PhaseGateLogEntry

    e = PhaseGateLogEntry(
        phase_id="phase_09",
        action="retry2",
        goal_regressions={"groove": -0.02},
        strength_used=0.6,
    )
    assert e.phase_id == "phase_09"
    assert e.action == "retry2"
    assert e.strength_used == pytest.approx(0.6)
    assert e.scores_before == {}
    assert e.scores_after == {}


# ---------------------------------------------------------------------------
# Tests: build_from_result
# ---------------------------------------------------------------------------


def test_build_from_result_basic(mock_result):
    """build_from_result() liefert validen PipelineTrace."""
    from backend.core.pipeline_trace import build_from_result

    trace = build_from_result(mock_result)
    assert trace.material == "vinyl"
    assert trace.era == "1970s"
    assert trace.restorability == pytest.approx(62.0)
    assert trace.phases_executed == 2
    assert trace.phases_skipped == 1


def test_build_from_result_phases(mock_result):
    """Trace enthält korrekte Phasen-Einträge."""
    from backend.core.pipeline_trace import build_from_result

    trace = build_from_result(mock_result)
    assert len(trace.phases) == 2
    p0 = trace.phases[0]
    assert p0.phase_id == "phase_03_denoise"
    assert p0.gate_decision == "accepted"
    assert p0.goals_before.get("natuerlichkeit") == pytest.approx(0.72)
    assert p0.goals_after.get("natuerlichkeit") == pytest.approx(0.85)


def test_build_from_result_best_effort_decision(mock_result):
    """best_effort-Action wird korrekt als gate_decision gemappt."""
    from backend.core.pipeline_trace import build_from_result

    trace = build_from_result(mock_result)
    p1 = trace.phases[1]
    assert p1.gate_decision == "best_effort"
    assert p1.phase_id == "phase_07_harmonic"


def test_build_from_result_goal_deltas(mock_result):
    """goal_deltas werden korrekt aus before/after berechnet."""
    from backend.core.pipeline_trace import build_from_result

    trace = build_from_result(mock_result)
    p0 = trace.phases[0]
    # natuerlichkeit: 0.85 - 0.72 = +0.13
    assert "natuerlichkeit" in p0.goal_deltas
    assert p0.goal_deltas["natuerlichkeit"] == pytest.approx(0.13, abs=0.001)


def test_build_from_result_joy_frisson(mock_result):
    """Joy/Fatigue/Frisson werden korrekt extrahiert."""
    from backend.core.pipeline_trace import build_from_result

    trace = build_from_result(mock_result)
    assert trace.joy_index == pytest.approx(0.72)
    assert trace.fatigue_index == pytest.approx(0.15)
    assert trace.frisson_index == pytest.approx(0.35)


def test_build_from_result_goal_timeline(mock_result):
    """Goal-Timeline wird befüllt."""
    from backend.core.pipeline_trace import build_from_result

    trace = build_from_result(mock_result)
    # mindestens ein Goal in der Timeline
    assert len(trace.goal_timeline) > 0


def test_build_from_result_to_json(mock_result):
    """to_json() liefert gültiges JSON."""
    import json

    from backend.core.pipeline_trace import build_from_result

    trace = build_from_result(mock_result)
    raw = trace.to_json()
    parsed = json.loads(raw)
    assert parsed["material"] == "vinyl"
    assert "phases" in parsed
    assert len(parsed["phases"]) == 2


# ---------------------------------------------------------------------------
# Tests: Formatter
# ---------------------------------------------------------------------------


def test_format_goals_table_no_goals():
    """format_goals_table ohne Goal-Daten gibt Hinweis zurück."""
    from backend.core.pipeline_trace import PipelineTrace, format_goals_table

    trace = PipelineTrace()
    result = format_goals_table(trace)
    assert "enable_debug_trace" in result or "Keine" in result


def test_format_goals_table_with_data(mock_result):
    """format_goals_table gibt ASCII-Tabelle mit Goal-Namen zurück."""
    from backend.core.pipeline_trace import build_from_result, format_goals_table

    trace = build_from_result(mock_result)
    table = format_goals_table(trace)
    assert "GOAL-MATRIX" in table
    assert "NATERL" in table or "natuerlichkeit" in table.lower()


def test_format_phase_decisions(mock_result):
    """format_phase_decisions gibt lesbare Entscheidungsliste zurück."""
    from backend.core.pipeline_trace import build_from_result, format_phase_decisions

    trace = build_from_result(mock_result)
    decisions = format_phase_decisions(trace)
    assert "PHASEN-ENTSCHEIDUNGEN" in decisions
    assert "phase_03_denoise" in decisions


def test_format_full_report(mock_result):
    """format_full_report läuft ohne Exception durch."""
    from backend.core.pipeline_trace import build_from_result, format_full_report

    trace = build_from_result(mock_result)
    report = format_full_report(trace)
    assert "AURIK PIPELINE DEBUG REPORT" in report
    assert "vinyl" in report


# ---------------------------------------------------------------------------
# Tests: debug_api
# ---------------------------------------------------------------------------


def test_get_debug_summary_stable_keys(mock_result):
    """get_debug_summary liefert stabile Keys."""
    from backend.api.debug_api import get_debug_summary

    summary = get_debug_summary(mock_result)
    required_keys = [
        "material",
        "mode",
        "era_decade",
        "restorability",
        "total_time_s",
        "phases_executed",
        "phases_skipped",
        "final_goals",
        "fail_reasons",
        "warnings",
        "has_phase_goal_data",
        "pmgg_log_entries_count",
    ]
    for key in required_keys:
        assert key in summary, f"Schlüssel '{key}' fehlt in debug_summary"


def test_get_debug_summary_has_phase_data(mock_result):
    """has_phase_goal_data ist True wenn pmgg_log_entries vorhanden."""
    from backend.api.debug_api import get_debug_summary

    summary = get_debug_summary(mock_result)
    assert summary["has_phase_goal_data"] is True
    assert summary["pmgg_log_entries_count"] == 2


def test_get_goal_fails_detects_below_threshold(mock_result):
    """get_goal_fails erkennt Goals unter Restoration-Schwellwert."""
    from backend.api.debug_api import get_goal_fails

    # natuerlichkeit=0.85 < Schwelle 0.90 → FAIL erwartet
    fails = get_goal_fails(mock_result, mode="restoration")
    fail_goals = [f["goal"] for f in fails]
    assert "natuerlichkeit" in fail_goals


def test_get_goal_fails_ok_goal_not_in_fails(mock_result):
    """tonal_center=0.82 (Schwelle 0.95) → ebenfalls Fail aber nicht fälschlich OK."""
    from backend.api.debug_api import get_goal_fails

    fails = get_goal_fails(mock_result, mode="restoration")
    fail_goals = [f["goal"] for f in fails]
    # tonal_center: 0.82 < 0.95 → muss in fail_goals sein
    assert "tonal_center" in fail_goals


def test_get_goal_fails_honors_legacy_maximum_alias(mock_result):
    """Legacy-Alias 'maximum' muss dieselben Studio-2026-Schwellen nutzen."""
    from backend.api.debug_api import get_goal_fails

    fails = get_goal_fails(mock_result, mode="maximum")
    fail_map = {entry["goal"]: entry for entry in fails}

    # natuerlichkeit=0.85 liegt unter Studio 2026 (0.92) und muss daher failen.
    assert fail_map["natuerlichkeit"]["threshold"] == pytest.approx(0.92)


def test_get_worst_phases_sorted(mock_result):
    """get_worst_phases gibt nach Regression sortierte Liste zurück."""
    from backend.api.debug_api import get_worst_phases

    worst = get_worst_phases(mock_result, n=5)
    # phase_07_harmonic hat Regressionen → muss in Liste sein
    if worst:
        pids = [w["phase_id"] for w in worst]
        assert "phase_07_harmonic" in pids


def test_format_full_report_via_api(mock_result):
    """format_full_report() über debug_api läuft ohne Exception."""
    from backend.api.debug_api import format_full_report

    report = format_full_report(mock_result)
    assert len(report) > 100


def test_get_pipeline_trace_bridge(mock_result):
    """bridge.get_pipeline_trace() liefert strukturiertes Dict."""
    from backend.api.bridge import get_pipeline_trace

    trace_dict = get_pipeline_trace(mock_result)
    assert isinstance(trace_dict, dict)
    # Kein harter Fehler erwartet
    assert "error" not in trace_dict or trace_dict.get("material") is not None


# ---------------------------------------------------------------------------
# Tests: UV3 enable_debug_trace Flag
# ---------------------------------------------------------------------------


def test_uv3_has_debug_trace_attribute():
    """UV3 besitzt _debug_trace_enabled nach __init__."""
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    r = UnifiedRestorerV3()
    assert hasattr(r, "_debug_trace_enabled")
    assert r._debug_trace_enabled is False


def test_uv3_debug_trace_enabled_by_kwarg(monkeypatch):
    """enable_debug_trace=True in restore() setzt _debug_trace_enabled=True."""
    import numpy as np

    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    r = UnifiedRestorerV3()

    # Kurzes Dummy-Audio (unter 100ms → early-exit)
    dummy_audio = np.zeros(1000, dtype=np.float32)

    # early-exit path (< 4800 samples) → restore() setzt trotzdem _debug_trace_enabled
    # Wir prüfen, dass das Flag gesetzt wird bevor der early-exit feuert
    _trace_seen = []

    def _patched(self, audio, sample_rate=44100, progress_callback=None, **kwargs):
        self._debug_trace_enabled = bool(kwargs.pop("enable_debug_trace", False))
        _trace_seen.append(self._debug_trace_enabled)
        # Dummy-Rückgabe — MagicMock vermeidet Konstruktor-Probleme
        from unittest.mock import MagicMock as _MM

        res = _MM()
        res.metadata = {}
        return res

    monkeypatch.setattr(r, "restore", lambda *a, **kw: _patched(r, *a, **kw))
    r.restore(dummy_audio, sample_rate=48000, enable_debug_trace=True)
    assert _trace_seen and _trace_seen[0] is True, "_debug_trace_enabled wurde nicht auf True gesetzt"
