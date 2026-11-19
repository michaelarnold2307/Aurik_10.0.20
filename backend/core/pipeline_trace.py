"""
pipeline_trace.py — Strukturierte Debug-Telemetrie für die Aurik-Pipeline.

Vollständige Sicht auf jeden Phasen-Durchlauf ohne Raten:
- 15 Musical-Goals vor/nach jeder Phase (Zeitreihe)
- Gate-Entscheidungen (accepted / rolled_back / skipped / best_effort)
- Phase-Timing, Stärke, Wet/Dry-Faktoren
- CIG-Rollbacks, ML-Fallbacks, Fail-Reasons

Zwei Modi:
  1. post_hoc  — build_from_result(result) nutzt pmgg_log_entries aus metadata
  2. live      — UV3 schreibt metadata["pmgg_log_entries"] wenn enable_debug_trace=True

Usage:
    # Im Code
    trace = build_from_result(restoration_result)
    print(format_full_report(trace))

    # Als JSON speichern
    with open("trace.json", "w") as f:
        f.write(trace.to_json())
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Kanonische Reihenfolge der 15 Musical Goals (Prio P1 → P5)
CANONICAL_GOALS: list[str] = [
    "natuerlichkeit",  # P1
    "authentizitaet",  # P1
    "tonal_center",  # P2
    "timbre_authentizitaet",  # P2
    "artikulation",  # P2
    "transient_energie",  # P2/P3
    "emotionalitaet",  # P3
    "micro_dynamics",  # P3 (alias: mikrodynamik)
    "groove",  # P3
    "transparenz",  # P4
    "waerme",  # P4
    "bass_kraft",  # P4 (alias: basskraft)
    "separation_fidelity",  # P4
    "brillanz",  # P5
    "spatial_depth",  # P5 (alias: raumtiefe)
]

# Kurzbezeichnungen für die ASCII-Tabelle (max 8 Zeichen)
GOAL_ABBREV: dict[str, str] = {
    "natuerlichkeit": "NATERL",
    "authentizitaet": "AUTHEN",
    "tonal_center": "TONAL",
    "timbre_authentizitaet": "TIMBRE",
    "artikulation": "ARTIK",
    "transient_energie": "TRANSENG",
    "emotionalitaet": "EMOTION",
    "micro_dynamics": "MIKRODYN",
    "mikrodynamik": "MIKRODYN",  # backward-compat alias
    "groove": "GROOVE",
    "transparenz": "TRANSP",
    "waerme": "WAERME",
    "bass_kraft": "BASSKR",
    "basskraft": "BASSKR",  # backward-compat alias
    "separation_fidelity": "SEPFID",
    "sep_fidelity": "SEPFID",
    "brillanz": "BRILLANZ",
    "spatial_depth": "RAUMTF",
    "raumtiefe": "RAUMTF",  # backward-compat alias
}

# Schwellwerte Restoration (P1-P5 Böden aus Spec 09)
RESTORATION_THRESHOLDS: dict[str, float] = {
    "natuerlichkeit": 0.90,
    "authentizitaet": 0.88,
    "tonal_center": 0.95,
    "timbre_authentizitaet": 0.87,
    "artikulation": 0.85,
    "emotionalitaet": 0.82,
    "micro_dynamics": 0.88,
    "mikrodynamik": 0.88,  # backward-compat alias
    "groove": 0.83,
    "transparenz": 0.82,
    "waerme": 0.75,
    "bass_kraft": 0.78,
    "basskraft": 0.78,  # backward-compat alias
    "sep_fidelity": 0.78,
    "brillanz": 0.78,
    "spatial_depth": 0.70,
    "raumtiefe": 0.70,  # backward-compat alias
}

STUDIO_THRESHOLDS: dict[str, float] = {
    "natuerlichkeit": 0.92,
    "authentizitaet": 0.90,
    "tonal_center": 0.96,
    "timbre_authentizitaet": 0.89,
    "artikulation": 0.87,
    "emotionalitaet": 0.84,
    "micro_dynamics": 0.90,
    "mikrodynamik": 0.90,  # backward-compat alias
    "groove": 0.85,
    "transparenz": 0.85,
    "waerme": 0.78,
    "bass_kraft": 0.80,
    "basskraft": 0.80,  # backward-compat alias
    "sep_fidelity": 0.80,
    "brillanz": 0.82,
    "spatial_depth": 0.74,
    "raumtiefe": 0.74,  # backward-compat alias
}


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class PhaseTrace:
    """Vollständiger Snapshot eines einzelnen Phasen-Durchlaufs."""

    phase_id: str
    phase_index: int = 0
    duration_s: float = 0.0
    strength_used: float = 1.0
    wet_dry_factor: float = 1.0
    gate_decision: str = "unknown"
    # "accepted" | "rolled_back" | "skipped" | "best_effort" | "skipped_pre_pipeline"
    gate_reason: str = ""
    pmgg_retries: int = 0
    rolled_back: bool = False
    goal_regressions: dict[str, float] = field(default_factory=dict)
    goals_before: dict[str, float] = field(default_factory=dict)
    goals_after: dict[str, float] = field(default_factory=dict)
    goal_deltas: dict[str, float] = field(default_factory=dict)  # computed from before/after
    team_policy_reason: str = ""
    team_excluded_goals: list[str] = field(default_factory=list)
    recovery_attempted: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert den Phasen-Snapshot als JSON-kompatibles Dict."""
        return {
            "phase_id": self.phase_id,
            "phase_index": self.phase_index,
            "duration_s": round(self.duration_s, 3),
            "strength_used": round(self.strength_used, 4),
            "wet_dry_factor": round(self.wet_dry_factor, 4),
            "gate_decision": self.gate_decision,
            "gate_reason": self.gate_reason,
            "pmgg_retries": self.pmgg_retries,
            "rolled_back": self.rolled_back,
            "goal_regressions": {k: round(v, 4) for k, v in self.goal_regressions.items()},
            "goals_before": {k: round(v, 4) for k, v in self.goals_before.items()},
            "goals_after": {k: round(v, 4) for k, v in self.goals_after.items()},
            "goal_deltas": {k: round(v, 4) for k, v in self.goal_deltas.items()},
            "team_policy_reason": self.team_policy_reason,
            "team_excluded_goals": self.team_excluded_goals,
            "recovery_attempted": self.recovery_attempted,
            "notes": self.notes,
        }


@dataclass
class PipelineTrace:
    """Vollständige Debug-Telemetrie eines Pipeline-Laufs."""

    run_id: str = ""
    timestamp: str = ""
    mode: str = ""
    material: str = ""
    era: str = ""
    restorability: float = 0.0
    phases: list[PhaseTrace] = field(default_factory=list)
    skipped_phases: list[str] = field(default_factory=list)
    fail_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    final_goals: dict[str, float] = field(default_factory=dict)
    initial_goals: dict[str, float] = field(default_factory=dict)
    adaptive_thresholds: dict[str, float] = field(default_factory=dict)
    hpi_score: float = 0.0
    total_duration_s: float = 0.0
    phases_executed: int = 0
    phases_skipped: int = 0
    phases_rolled_back: int = 0
    goal_timeline: dict[str, list[float]] = field(default_factory=dict)
    # Parallel timeline: list of phase_ids (x-Axis)
    timeline_phase_ids: list[str] = field(default_factory=list)
    cig_rollbacks: list[dict[str, Any]] = field(default_factory=list)
    ml_fallbacks: list[str] = field(default_factory=list)
    team_coordination_events: list[dict[str, Any]] = field(default_factory=list)
    joy_index: float = 0.0
    fatigue_index: float = 0.0
    frisson_index: float = 0.0
    data_source: str = "post_hoc"  # "post_hoc" | "live"

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert den Pipeline-Trace als JSON-kompatibles Dict."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "material": self.material,
            "era": self.era,
            "restorability": round(self.restorability, 2),
            "phases_executed": self.phases_executed,
            "phases_skipped": self.phases_skipped,
            "phases_rolled_back": self.phases_rolled_back,
            "hpi_score": round(self.hpi_score, 4),
            "total_duration_s": round(self.total_duration_s, 1),
            "initial_goals": {k: round(v, 4) for k, v in self.initial_goals.items()},
            "final_goals": {k: round(v, 4) for k, v in self.final_goals.items()},
            "adaptive_thresholds": {k: round(v, 4) for k, v in self.adaptive_thresholds.items()},
            "goal_timeline": {k: [round(v, 4) for v in vs] for k, vs in self.goal_timeline.items()},
            "timeline_phase_ids": self.timeline_phase_ids,
            "skipped_phases": self.skipped_phases,
            "fail_reasons": self.fail_reasons,
            "warnings": self.warnings,
            "cig_rollbacks": self.cig_rollbacks,
            "ml_fallbacks": self.ml_fallbacks,
            "team_coordination_events": self.team_coordination_events,
            "joy_index": round(self.joy_index, 3),
            "fatigue_index": round(self.fatigue_index, 3),
            "frisson_index": round(self.frisson_index, 3),
            "data_source": self.data_source,
            "phases": [p.to_dict() for p in self.phases],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialisiert den Pipeline-Trace als formatierter JSON-String."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------------------
# Extraktion aus RestorationResult
# ---------------------------------------------------------------------------


def build_from_result(result: Any) -> PipelineTrace:
    """
    Baut einen PipelineTrace aus einem RestorationResult.

    Nutzt:
    - result.phases_executed / result.phases_skipped
    - result.phase_gate_log (best_effort phases)
    - result.musical_goals (Ziele am Pipeline-Ende)
    - result.metadata (pmgg_log_entries, fail_reasons, team_coordination, ...)
    - result.total_time_seconds
    - result.restorability
    - result.era_decade
    - result.material_type
    - result.adaptive_thresholds
    """
    trace = PipelineTrace(
        run_id=_safe_str(getattr(result, "run_id", None)) or f"run_{int(time.time())}",
        timestamp=_safe_str(getattr(result, "timestamp", None)) or _now_iso(),
        data_source="post_hoc",
    )

    # --- Grundinfos ---
    _material = getattr(result, "material_type", None)
    if _material is None:
        trace.material = ""
    else:
        _mat_val = getattr(_material, "value", None)
        trace.material = str(_mat_val) if _mat_val is not None else str(_material)
    trace.era = str(getattr(result, "era_decade", "") or "")
    trace.restorability = float(getattr(result, "restorability", 0) or 0)
    trace.total_duration_s = float(getattr(result, "total_time_seconds", 0) or 0)
    trace.phases_executed = len(getattr(result, "phases_executed", []) or [])
    trace.skipped_phases = list(getattr(result, "phases_skipped", []) or [])
    trace.phases_skipped = len(trace.skipped_phases)
    trace.warnings = list(getattr(result, "warnings", []) or [])

    # --- Adaptive Thresholds ---
    _at = getattr(result, "adaptive_thresholds", None)
    if isinstance(_at, dict):
        trace.adaptive_thresholds = {str(k): float(v) for k, v in _at.items()}

    # --- Finales Musical Goals ---
    _mg = getattr(result, "musical_goals", None)
    if isinstance(_mg, dict):
        trace.final_goals = {str(k): float(v) for k, v in _mg.items() if v is not None}

    # --- Mode ---
    _config = getattr(result, "config", None)
    if _config:
        _m = getattr(_config, "mode", None)
        if _m is None:
            trace.mode = ""
        else:
            _mval = getattr(_m, "value", None)
            trace.mode = str(_mval) if _mval is not None else str(_m)

    # --- Metadata ---
    meta = getattr(result, "metadata", {}) or {}

    # Fail-Reasons
    _fr = meta.get("fail_reasons", [])
    if isinstance(_fr, list):
        trace.fail_reasons = [str(x) for x in _fr]
    elif isinstance(_fr, dict):
        trace.fail_reasons = [f"{k}: {v}" for k, v in _fr.items()]

    # ML-Fallbacks
    _mlf = meta.get("ml_fallbacks_used", [])
    if isinstance(_mlf, list):
        trace.ml_fallbacks = [str(x) for x in _mlf]

    # Team-Koordination
    _tc = meta.get("team_coordination", {})
    if isinstance(_tc, dict):
        _events = _tc.get("events", [])
        if isinstance(_events, list):
            trace.team_coordination_events = _events

    # CIG-Rollbacks
    _ig = meta.get("interaction_guard", {})
    if isinstance(_ig, dict):
        _rb = _ig.get("rollback_log", [])
        if isinstance(_rb, list):
            trace.cig_rollbacks = _rb

    # Joy/Fatigue/Frisson
    _joy = meta.get("joy_runtime_index", {})
    if isinstance(_joy, dict):
        trace.joy_index = float(_joy.get("joy_index", 0) or 0)
        trace.fatigue_index = float(_joy.get("fatigue_index", 0) or 0)
        _comps = _joy.get("components", {}) or {}
        trace.frisson_index = float(_comps.get("frisson_index", 0) or 0)

    # --- PMGG Log Entries (Herzstück) ---
    _entries = meta.get("pmgg_log_entries", [])
    if not _entries:
        # Fallback: phase_gate_log enthält nur best_effort IDs
        _bef = list(getattr(result, "phase_gate_log", []) or [])
        _executed = list(getattr(result, "phases_executed", []) or [])
        for idx, pid in enumerate(_executed):
            decision = "best_effort" if pid in _bef else "accepted"
            pt = PhaseTrace(
                phase_id=pid,
                phase_index=idx,
                gate_decision=decision,
            )
            trace.phases.append(pt)
    else:
        set(getattr(result, "phases_executed", []) or [])
        for idx, entry in enumerate(_entries):
            if isinstance(entry, dict):
                _e = entry
            else:
                # PhaseGateLogEntry dataclass
                try:
                    import dataclasses as _dc

                    _e = _dc.asdict(entry)
                except Exception:
                    _e = entry.__dict__ if hasattr(entry, "__dict__") else {}

            _pid = str(_e.get("phase_id", f"phase_{idx}"))
            _action = str(_e.get("action", "unknown"))
            _gate = _map_action_to_decision(_action)
            _sb = dict(_e.get("scores_before", {}) or {})
            _sa = dict(_e.get("scores_after", {}) or {})
            _deltas = {
                g: round(float(_sa.get(g, 0)) - float(_sb.get(g, 0)), 4)
                for g in set(list(_sb.keys()) + list(_sa.keys()))
                if _sb.get(g) is not None and _sa.get(g) is not None
            }
            _meta_e = dict(_e.get("metadata", {}) or {})

            # Retry-Zählung aus Action-String ableiten
            _retries = 0
            if _action.startswith("retry"):
                try:
                    _retries = int(_action.replace("retry", "").replace("r", "") or "1")
                except ValueError:
                    _retries = 1
            elif "best_effort_r" in _action:
                try:
                    _retries = int(_action.rsplit("_r", maxsplit=1)[-1])
                except ValueError:
                    _retries = 0

            pt = PhaseTrace(
                phase_id=_pid,
                phase_index=idx,
                gate_decision=_gate,
                pmgg_retries=_retries,
                rolled_back=bool(_e.get("rolled_back", False)),
                goal_regressions={k: float(v) for k, v in (_e.get("goal_regressions", {}) or {}).items()},
                goals_before={k: float(v) for k, v in _sb.items()},
                goals_after={k: float(v) for k, v in _sa.items()},
                goal_deltas=_deltas,
                team_policy_reason=str(_meta_e.get("team_policy_reason", "") or ""),
                team_excluded_goals=list(_meta_e.get("team_excluded_goals", []) or []),
                recovery_attempted=bool(_meta_e.get("recovery_attempted", False)),
                strength_used=float(_e.get("strength_used", 1.0) or 1.0),
            )
            trace.phases.append(pt)
            if _gate == "rolled_back":
                trace.phases_rolled_back += 1

    # --- Goal-Timeline aufbauen ---
    _build_goal_timeline(trace)

    # --- Initiale Goals: erste Phase scores_before (wenn vorhanden) ---
    if trace.phases and trace.phases[0].goals_before:
        trace.initial_goals = dict(trace.phases[0].goals_before)

    return trace


def _map_action_to_decision(action: str) -> str:
    """Mappt PMGG-Action auf vereinfachte Gate-Decision für Traces."""
    if action == "passed":
        return "accepted"
    if action.startswith("best_effort"):
        return "best_effort"
    if action.startswith("retry"):
        return "accepted"  # letzte retry = accepted
    if action == "rolled_back" or action.startswith("rollback"):
        return "rolled_back"
    if action == "skipped":
        return "skipped"
    return action


def _build_goal_timeline(trace: PipelineTrace) -> None:
    """Befüllt trace.goal_timeline und trace.timeline_phase_ids."""
    timeline: dict[str, list[float]] = {g: [] for g in CANONICAL_GOALS}
    phase_ids: list[str] = []

    for pt in trace.phases:
        if not pt.goals_after:
            continue
        phase_ids.append(pt.phase_id)
        for g in CANONICAL_GOALS:
            val = pt.goals_after.get(g)
            if val is None and pt.goals_before.get(g) is not None:
                val = pt.goals_before[g]  # kein After → kein Delta
            if val is not None:
                timeline[g].append(round(float(val), 4))
            elif timeline[g]:
                # Letzten bekannten Wert fortschreiben
                timeline[g].append(timeline[g][-1])
            else:
                timeline[g].append(0.0)

    # Nur Goals mit mind. einem Messwert übernehmen
    trace.goal_timeline = {g: vs for g, vs in timeline.items() if any(v > 0 for v in vs)}
    trace.timeline_phase_ids = phase_ids


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Formatter: ASCII-Tabellen und Berichte
# ---------------------------------------------------------------------------


def format_goals_table(trace: PipelineTrace, mode: str | None = None) -> str:
    """
    Erstellt eine ASCII-Matrix: 15 Musical Goals × Phasen.

    Zeigt für jede Phase den Goal-Score nach der Phase.
    Regressionen (<Schwellwert) werden mit [!] markiert.
    Regressionen gegenüber Vorphasen mit [-] markiert.
    """
    phases_with_goals = [p for p in trace.phases if p.goals_after]
    if not phases_with_goals:
        return "(Keine Goal-Daten verfügbar — enable_debug_trace=True beim restore()-Aufruf setzen)"

    goals_to_show = [g for g in CANONICAL_GOALS if any(p.goals_after.get(g) is not None for p in phases_with_goals)]
    if not goals_to_show:
        return "(Keine Goal-Messungen in Trace enthalten)"

    thresholds = STUDIO_THRESHOLDS if (mode or trace.mode or "").startswith("studio") else RESTORATION_THRESHOLDS

    # Spaltenbreite: phase_id abkürzen auf max 12 Zeichen
    col_w = 8
    lbl_w = 10  # Goal-Kürzel max 10 Zeichen

    # Header
    header_ids = [p.phase_id.replace("phase_", "p").replace("_", ".")[:12] for p in phases_with_goals]
    lines = []
    lines.append("=" * (lbl_w + 2 + len(header_ids) * (col_w + 1)))
    lines.append("GOAL-MATRIX: Musical Goals × Pipeline-Phasen")
    lines.append(
        f"Material: {trace.material or '?'}  Mode: {trace.mode or '?'}  "
        f"Restorability: {trace.restorability:.0f}  Phasen: {len(phases_with_goals)}"
    )
    lines.append("-" * (lbl_w + 2 + len(header_ids) * (col_w + 1)))

    # Phase-IDs Zeile
    hdr = f"{'GOAL':<{lbl_w}}  " + "  ".join(f"{p_id:>{col_w - 1}}" for p_id in header_ids)
    lines.append(hdr)
    lines.append("-" * (lbl_w + 2 + len(header_ids) * (col_w + 1)))

    for goal in goals_to_show:
        abbr = GOAL_ABBREV.get(goal, goal[:lbl_w])
        thr = thresholds.get(goal, 0.0)
        row_parts = []
        prev_val: float | None = None
        for p in phases_with_goals:
            val = p.goals_after.get(goal)
            if val is None:
                row_parts.append(f"{'  ---':>{col_w}}")
            else:
                flag = ""
                if val < thr:
                    flag = "!"
                elif prev_val is not None and val < prev_val - 0.01:
                    flag = "-"
                row_parts.append(f"{val:>{col_w - 1}.3f}{flag}")
            prev_val = val if val is not None else prev_val

        row = f"{abbr:<{lbl_w}}  " + "  ".join(row_parts)
        lines.append(row)

    lines.append("-" * (lbl_w + 2 + len(header_ids) * (col_w + 1)))

    # Final-Goals Zeile
    if trace.final_goals:
        final_parts = []
        for p in phases_with_goals:
            final_parts.append(" " * (col_w))
        lines.append(
            f"{'FINAL':<{lbl_w}}  "
            + "  ".join(f"{trace.final_goals.get(g, 0.0):>{col_w - 1}.3f} " for g in goals_to_show[:1])
        )
        lines.append("Finale Goals (vollständig):")
        for g in goals_to_show:
            fv = trace.final_goals.get(g)
            thr = thresholds.get(g, 0.0)
            flag = " [FAIL]" if fv is not None and fv < thr else ""
            lines.append(
                f"  {GOAL_ABBREV.get(g, g):<12}  {fv:>6.3f}  (Schwelle: {thr:.2f}){flag}"
                if fv is not None
                else f"  {GOAL_ABBREV.get(g, g):<12}  ---"
            )

    lines.append("=" * (lbl_w + 2 + len(header_ids) * (col_w + 1)))
    lines.append("Legende: [!] unter Schwellwert  [-] Regression gegenüber Vorphasen  --- kein Messwert")
    return "\n".join(lines)


def format_phase_decisions(trace: PipelineTrace) -> str:
    """Zeigt alle Phasen-Entscheidungen in Kurzform."""
    lines = []
    lines.append("=" * 80)
    lines.append("PHASEN-ENTSCHEIDUNGEN")
    lines.append(
        f"Ausgeführt: {trace.phases_executed}  Übersprungen: {trace.phases_skipped}  "
        f"Rollbacks: {trace.phases_rolled_back}"
    )
    lines.append("-" * 80)
    lines.append(f"{'#':<4} {'Phase-ID':<35} {'Decision':<15} {'Strength':>8}  {'Retries':>7}  Hinweis")
    lines.append("-" * 80)

    for i, pt in enumerate(trace.phases):
        decision_sym = {
            "accepted": "✓ OK",
            "best_effort": "⚡ EFFORT",
            "rolled_back": "✗ ROLLBK",
            "skipped": "— SKIP",
        }.get(pt.gate_decision, pt.gate_decision[:8])
        hint = pt.team_policy_reason[:30] if pt.team_policy_reason else ""
        if pt.goal_regressions:
            worst = min(pt.goal_regressions.values())
            # select goal with minimal regression value
            worst_goal = min(pt.goal_regressions.items(), key=lambda kv: kv[1])[0]
            hint = hint or f"{worst_goal}: {worst:+.3f}"
        lines.append(
            f"{i + 1:<4} {pt.phase_id:<35} {decision_sym:<15} {pt.strength_used:>8.3f}  {pt.pmgg_retries:>7}  {hint}"
        )

    if trace.skipped_phases:
        lines.append("-" * 80)
        lines.append("Übersprungene Phasen:")
        for pid in trace.skipped_phases:
            lines.append(f"  — {pid}")

    lines.append("=" * 80)
    return "\n".join(lines)


def format_goal_deltas(trace: PipelineTrace) -> str:
    """Zeigt die kumulativen Goal-Deltas (Anfang → Ende) pro Goal."""
    lines = []
    lines.append("=" * 60)
    lines.append("GOAL-DELTAS: Anfang → Ende der Pipeline")
    lines.append("-" * 60)
    lines.append(f"{'Goal':<22}  {'Start':>6}  {'Ende':>6}  {'Δ':>7}  {'Status'}")
    lines.append("-" * 60)

    thresholds = STUDIO_THRESHOLDS if trace.mode and "studio" in trace.mode else RESTORATION_THRESHOLDS

    for g in CANONICAL_GOALS:
        init_val = trace.initial_goals.get(g)
        final_val = trace.final_goals.get(g)
        thr = thresholds.get(g, 0.0)
        if init_val is None and final_val is None:
            continue
        delta = (final_val - init_val) if (final_val is not None and init_val is not None) else None
        delta_str = f"{delta:+.3f}" if delta is not None else "  n/a"
        final_str = f"{final_val:.3f}" if final_val is not None else "  ---"
        init_str = f"{init_val:.3f}" if init_val is not None else "  ---"
        status = ""
        if final_val is not None:
            status = "✓ OK" if final_val >= thr else f"✗ FAIL ({thr:.2f})"
        lines.append(f"{GOAL_ABBREV.get(g, g):<22}  {init_str:>6}  {final_str:>6}  {delta_str:>7}  {status}")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_full_report(trace: PipelineTrace) -> str:
    """Vollständiger Debug-Bericht: Überblick + Decisions + Goals-Matrix + Deltas."""
    lines = []
    lines.append("")
    lines.append("╔" + "═" * 78 + "╗")
    lines.append("║  AURIK PIPELINE DEBUG REPORT" + " " * 49 + "║")
    lines.append("╚" + "═" * 78 + "╝")
    lines.append("")
    lines.append(f"  Zeitstempel : {trace.timestamp}")
    lines.append(f"  Run-ID      : {trace.run_id}")
    lines.append(f"  Modus       : {trace.mode or '?'}")
    lines.append(f"  Material    : {trace.material or '?'}")
    lines.append(f"  Ära         : {trace.era or '?'}")
    lines.append(f"  Restorability: {trace.restorability:.1f}")
    lines.append(f"  Gesamtzeit  : {trace.total_duration_s:.1f}s")
    lines.append(f"  HPI-Score   : {trace.hpi_score:.4f}")
    lines.append(
        f"  Phasen      : {trace.phases_executed} ausgeführt, "
        f"{trace.phases_skipped} übersprungen, {trace.phases_rolled_back} Rollbacks"
    )
    lines.append(
        f"  Joy/Müdigkeit/Gänsehaut: {trace.joy_index:.2f} / {trace.fatigue_index:.2f} / {trace.frisson_index:.2f}"
    )
    if trace.data_source == "post_hoc":
        lines.append("  ⚠ DATEN: post_hoc — für vollständige Goal-Daten enable_debug_trace=True setzen")

    if trace.fail_reasons:
        lines.append("")
        lines.append("  FAIL-REASONS:")
        for fr in trace.fail_reasons:
            lines.append(f"    ✗ {fr}")

    if trace.warnings:
        lines.append("")
        lines.append("  WARNUNGEN:")
        for w in trace.warnings[:10]:
            lines.append(f"    ⚠ {w}")

    if trace.ml_fallbacks:
        lines.append("")
        lines.append("  ML-FALLBACKS:")
        for f_ in trace.ml_fallbacks:
            lines.append(f"    ↩ {f_}")

    if trace.cig_rollbacks:
        lines.append("")
        lines.append(f"  CIG-ROLLBACKS ({len(trace.cig_rollbacks)}):")
        for rb in trace.cig_rollbacks[:5]:
            lines.append(f"    ✗ {rb}")

    lines.append("")
    lines.append(format_phase_decisions(trace))
    lines.append("")
    lines.append(format_goal_deltas(trace))
    lines.append("")
    lines.append(format_goals_table(trace))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Singleton-Accessor (optional — für Laufzeit-Trace-Sampling)
# ---------------------------------------------------------------------------

_last_trace: PipelineTrace | None = None
_trace_lock = __import__("threading").Lock()


def get_last_trace() -> PipelineTrace | None:
    """Gibt den letzten gespeicherten Trace zurück (thread-sicher)."""
    with _trace_lock:
        return _last_trace


def store_trace(trace: PipelineTrace) -> None:
    """Speichert Trace als letzten bekannten Trace (thread-sicher)."""
    global _last_trace  # pylint: disable=global-statement
    with _trace_lock:
        _last_trace = trace
