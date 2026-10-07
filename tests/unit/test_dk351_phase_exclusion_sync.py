"""§D-K3-51: Exclude-konsistente §v10.709-Degradations-Prüfung + PMGG/CIG-Sync.

Produktionsbefund 2026-10-07: Der §v10.709-Guard prüfte die UNGEFILTERTEN
PMGG-Scores (alle 15 Goals) — bewusste Phasen-Ausschlüsse (`tonal_center` für
HPF-/Synthese-Phasen) erzeugten FALSE-Degradationen (phase_23/30/37) und
bauten den 3er-Zähler zum EMERGENCY-STOP. Dazu zwei Map-Lücken:

  - `phase_30` (DC-Offset = HPF) fehlte `tonal_center` in PMGG UND CIG —
    identische Physik wie `phase_05` (Rumble, Δ=0.6583 belegt 2026-05-06);
    heutiger Befund Δ=0.8925.
  - `phase_23` (FlashSR) fehlte KOMPLETT in der PMGG — der CIG hatte den
    Block seit 2026-04-24 vollständig (Δ=0.7893) — §2.55-Sync-Lücke.

Fix: `check_iteration_abort_excluding()` (+ `degraded_deltas` als Zeuge, §G8 (copilot-instructions.md)),
UV3-§v10.709 nutzt `PhaseGateLogEntry.metadata["goal_exclusions"]`, Map-Lücken
geschlossen. Gepinnt: Filter-Semantik (verdeckt nichts anderes), Klassen-Kontrakte
HPF {phase_05, phase_30} und Synthese {phase_23, phase_24, phase_55}.
"""

from __future__ import annotations

from backend.core.cumulative_interaction_guard import _PHASE_SPECIFIC_DRIFT_EXCLUSIONS as CIG_EXCLUSIONS
from backend.core.goal_priority_protocol import check_iteration_abort, check_iteration_abort_excluding
from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS as PMGG_EXCLUSIONS


def test_exclude_filters_tonal_center_regression() -> None:
    """Mit Exclude: tonal_center-Delta 0,4 löst KEINEN Abbruch aus."""
    result = check_iteration_abort_excluding(
        {"tonal_center": 0.9, "natuerlichkeit": 0.8},
        {"tonal_center": 0.5, "natuerlichkeit": 0.8},
        {"tonal_center"},
    )
    assert result.should_abort is False
    assert result.degraded_goals == []


def test_without_exclude_regression_aborts_with_deltas() -> None:
    """Ohne Exclude bricht die Regression ab — und die Deltas sind bezeugt (§G8 (copilot-instructions.md))."""
    result = check_iteration_abort_excluding({"tonal_center": 0.9}, {"tonal_center": 0.5})
    assert result.should_abort is True
    assert result.degraded_goals == ["tonal_center"]
    assert abs(result.degraded_deltas["tonal_center"] - 0.4) < 1e-9


def test_excluding_does_not_weaken_other_goals() -> None:
    """Der Exclude darf andere P1/P2-Regressionen nicht verdecken."""
    result = check_iteration_abort_excluding(
        {"tonal_center": 0.9, "natuerlichkeit": 0.8},
        {"tonal_center": 0.1, "natuerlichkeit": 0.6},
        {"tonal_center"},
    )
    assert result.should_abort is True
    assert result.degraded_goals == ["natuerlichkeit"]


def test_check_iteration_abort_unchanged_for_plain_callers() -> None:
    """Die Bestands-API bleibt unverändert verfügbar (kein Bruch für Alt-Aufrufer)."""
    result = check_iteration_abort({"tonal_center": 0.9}, {"tonal_center": 0.5})
    assert result.should_abort is True


def test_phase_30_hpf_class_has_tonal_center_in_both_guards() -> None:
    """HPF-Klasse (phase_05 Rumble + phase_30 DC-Offset): tonal_center in BEIDEN Guards."""
    for phase in ("phase_05", "phase_30"):
        assert "tonal_center" in PMGG_EXCLUSIONS[phase], f"PMGG {phase}: tonal_center fehlt"
        assert "tonal_center" in CIG_EXCLUSIONS[phase], f"CIG {phase}: tonal_center fehlt"


def test_phase_23_synthesis_block_exists_and_synced() -> None:
    """Synthese-Klasse: PMGG phase_23 muss existieren und tonal_center tragen (CIG-Sync)."""
    assert "phase_23" in PMGG_EXCLUSIONS, "PMGG phase_23-Block fehlt (CIG hatte ihn seit 2026-04-24)"
    assert "tonal_center" in PMGG_EXCLUSIONS["phase_23"]
    assert "tonal_center" in CIG_EXCLUSIONS["phase_23"]
    # Die PMGG-Menge ist eine Teilmenge der CIG-Ausschlüsse (gleiche Mechanik, §2.55).
    assert PMGG_EXCLUSIONS["phase_23"] <= CIG_EXCLUSIONS["phase_23"]


def test_synthesis_siblings_have_tonal_center_in_both_guards() -> None:
    """Synthese-Geschwister phase_24/phase_55 tragen tonal_center beidseitig."""
    for phase in ("phase_24", "phase_55"):
        assert "tonal_center" in PMGG_EXCLUSIONS[phase]
        assert "tonal_center" in CIG_EXCLUSIONS[phase]
