"""goal_monitor.py — Aurik 10.0.0 Musical-Goals-Diagnose-Monitor.

Liest `logs/aurik_backend.log` und zeigt für jeden abgeschlossenen Lauf:
    • Alle 15 Ziel-Scores mit effektivem Schwellwert, Gap und Pass/Fail
  • Welche Phasen den größten Rückgang pro Ziel verursacht haben (PMGG-Trajektorie)
  • Zusammenfassung: Welche Ziele scheitern systematisch und warum

Verwendung:
    python scripts/goal_monitor.py                   # letzten Lauf analysieren
    python scripts/goal_monitor.py --all             # alle Läufe anzeigen
    python scripts/goal_monitor.py --runs 3          # letzte N Läufe
    python scripts/goal_monitor.py --goal tonal_center   # nur dieses Ziel
    python scripts/goal_monitor.py --log path/to/other.log
    python scripts/goal_monitor.py --since "2026-04-26 02:00"
    python scripts/goal_monitor.py --json            # maschinenlesbare JSON-Ausgabe
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LOG = Path(__file__).parent.parent / "logs" / "aurik_backend.log"

# Priority groups per spec §15 Musical Goals
_P1 = {"natuerlichkeit", "authentizitaet"}
_P2 = {"tonal_center", "timbre_authentizitaet", "artikulation", "transient_energie"}
_P3 = {"emotionalitaet", "mikro_dynamik", "groove"}
_P4 = {"transparenz", "waerme", "bass_kraft", "separation_fidelity"}
_P5 = {"brillanz", "raumtiefe"}

_PRIORITY: dict[str, str] = {}
for _g in _P1:
    _PRIORITY[_g] = "P1"
for _g in _P2:
    _PRIORITY[_g] = "P2"
for _g in _P3:
    _PRIORITY[_g] = "P3"
for _g in _P4:
    _PRIORITY[_g] = "P4"
for _g in _P5:
    _PRIORITY[_g] = "P5"

# ANSI colours (disabled if stdout not a tty)
_COLOUR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _COLOUR:
        return text
    return f"\033[{code}m{text}\033[0m"


# pylint: disable=unnecessary-lambda-assignment
RED = lambda t: _c("31", t)
GRN = lambda t: _c("32", t)
YLW = lambda t: _c("33", t)
BLD = lambda t: _c("1", t)
DIM = lambda t: _c("2", t)
CYN = lambda t: _c("36", t)
MGN = lambda t: _c("35", t)
# pylint: enable=unnecessary-lambda-assignment

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GoalEntry:
    """Score-Eintrag eines Musical Goals für einen Restaurierungs-Lauf."""

    score: float = 0.0
    threshold: float = 0.85
    gap: float = 0.0
    passed: bool = False
    applicable: bool = True


@dataclass
class PhaseRegression:
    """Größter Einzelphasen-Rückgang eines Musical Goals."""

    phase_id: str = ""
    goal: str = ""
    delta: float = 0.0


@dataclass
class CigRollback:
    """CIG-Rollback-Ereignis aus der Pipeline."""

    phase_id: str = ""
    trigger_goal: str = ""
    drift: float = 0.0
    tolerance: float = 0.0
    rollback_to: str = ""


@dataclass
class MlFallback:
    """ML-Modell-Fallback-Ereignis aus der Pipeline."""

    phase: str = ""
    model: str = ""
    reason: str = ""
    fallback: str = ""


@dataclass
class PhaseExec:
    """Ausführungsparameter einer Phase aus der Pipeline."""

    phase_id: str = ""
    strength: str = "None"
    explicit: bool = False
    vcap: str = "None"
    conductor: str = "None"
    songcal: float = 1.0


@dataclass
class HpiComponents:
    """HPI-Komponentenwerte aus dem Holistic Perceptual Gate."""

    mode: str = ""
    hpi: float = 0.0
    passed: bool = False
    timbral: float = 1.0
    mert: float = 1.0
    artifact: float = 1.0
    emotional: float = 1.0


@dataclass
class RunReport:
    """Vollständiger Bericht eines Restaurierungs-Laufs."""

    timestamp: str
    material: str
    mode: str
    excellence: float
    n_violations: int
    goals: dict[str, GoalEntry] = field(default_factory=dict)
    # goal → list of (phase_id, before, after, delta, action) sorted by phase order
    phase_trajectories: dict[str, list[tuple[str, float, float, float, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # biggest single-phase regression per goal
    worst_regression: dict[str, PhaseRegression] = field(default_factory=dict)
    # new diagnostic data
    cig_rollbacks: list[CigRollback] = field(default_factory=list)
    ml_fallbacks: list[MlFallback] = field(default_factory=list)
    phase_execs: list[PhaseExec] = field(default_factory=list)
    hpi: HpiComponents | None = None

    def violated_goals(self) -> list[str]:
        """Gibt alle nicht erfüllten und anwendbaren Ziel-Namen zurück."""
        return [g for g, e in self.goals.items() if not e.passed and e.applicable]

    def passed_goals(self) -> list[str]:
        """Gibt alle erfüllten und anwendbaren Ziel-Namen zurück."""
        return [g for g, e in self.goals.items() if e.passed and e.applicable]


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

# GOAL_SCORECARD line produced by UV3 after §GOAL_MONITOR patch:
# Backend log:  [2026-04-26 07:09:12,123] INFO ...: 🎯 GOAL_SCORECARD mat=vinyl ...
# CLI log:      INFO: 🎯 GOAL_SCORECARD mat=vinyl ...
# Timestamp group is optional to support both formats.
_TS_PAT = r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)"
_TS_OPT = r"(?:" + _TS_PAT + r".*?)?"  # optional timestamp prefix

_RE_SCORECARD = re.compile(
    _TS_OPT + r"GOAL_SCORECARD mat=(?P<mat>\S+) mode=(?P<mode>\S+)"
    r" excellence=(?P<exc>[\d.]+) violations=(?P<viol>\d+)\|(?P<goals>.+)"
)
_RE_GOAL_FIELD = re.compile(r"(\w+):score=([\d.]+),thr=([\d.]+),gap=([+-][\d.]+),pass=([01]),app=([01])")

# PMGG per-phase line (already in log since project start):
# PMGG waerme §9.7.14  phase=phase_05_rumble_filter  before=1.0000  after=1.0000
#   delta=+0.0000  action=passed  strength=1.00
_RE_PMGG = re.compile(
    _TS_OPT + r"PMGG (\w+) §[\d.]+\s+phase=(\S+)\s+before=([\d.]+)\s+after=([\d.]+)"
    r"\s+delta=([+-][\d.]+)\s+action=(\S+)\s+strength=([\d.]+)"
)

# Pipeline start marker (used to group phases by run)
_RE_PIPELINE_START = re.compile(_TS_OPT + r"Step 3/4: Executing Restoration Pipeline")

# ML_FALLBACK line (UV3 §MONITOR):
# 🔌 ML_FALLBACK phase=phase_06 model=AudioSR reason=OOM fallback=dsp_1
_RE_ML_FALLBACK = re.compile(_TS_OPT + r"🔌 ML_FALLBACK phase=(\S+) model=(\S+) reason=(\S+) fallback=(\S+)")

# CIG_ROLLBACK line (cumulative_interaction_guard.py §MONITOR):
# ⚠️ CIG_ROLLBACK phase=phase_23 trigger_goal=tonal_center drift=-0.0821 tolerance=-0.0500 rollback_to=...
_RE_CIG_ROLLBACK = re.compile(
    _TS_OPT
    + r"⚠️ CIG_ROLLBACK phase=(\S+) trigger_goal=(\S+) drift=([+-][\d.]+) tolerance=([+-][\d.]+) rollback_to=(\S+)"
)

# PHASE_EXEC line (UV3 §MONITOR):
# 📊 PHASE_EXEC phase=phase_06 strength=0.300 explicit=0 vcap=0.300 conductor=None songcal=0.650
_RE_PHASE_EXEC = re.compile(
    _TS_OPT + r"📊 PHASE_EXEC phase=(\S+) strength=(\S+) explicit=([01]) vcap=(\S+) conductor=(\S+) songcal=([\d.]+)"
)

# HPI_COMP line (UV3 extended):
# 🎯 HPI_COMP mode=Restoration hpi=0.3421 passed=0 timbral=0.512 mert=0.891 artifact=0.987 emotional=0.743
_RE_HPI_COMP = re.compile(
    _TS_OPT + r"🎯 HPI_COMP mode=(\S+) hpi=([\d.]+) passed=([01])"
    r" timbral=([\d.]+) mert=([\d.]+) artifact=([\d.]+) emotional=([\d.]+)"
)


def _parse_ts(ts_str: str | None) -> datetime | None:
    if ts_str is None:
        return None
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")


def parse_log(log_path: Path, since: datetime | None = None) -> list[RunReport]:
    """Parse log file into a list of RunReport objects (one per pipeline run)."""
    runs: list[RunReport] = []
    # Working state for the current in-progress run
    current_pmgg: dict[str, list[tuple[str, float, float, float, str]]] = defaultdict(list)
    current_cig: list[CigRollback] = []
    current_ml: list[MlFallback] = []
    current_execs: list[PhaseExec] = []
    current_hpi: HpiComponents | None = None

    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip()

            # Detect pipeline start → reset per-run buffers
            m_start = _RE_PIPELINE_START.search(line)
            if m_start:
                ts = _parse_ts(m_start.group("ts"))
                if since and ts is not None and ts < since:
                    current_pmgg = defaultdict(list)
                    current_cig = []
                    current_ml = []
                    current_execs = []
                    current_hpi = None
                    continue
                current_pmgg = defaultdict(list)
                current_cig = []
                current_ml = []
                current_execs = []
                current_hpi = None
                continue

            # ML_FALLBACK
            m_ml = _RE_ML_FALLBACK.search(line)
            if m_ml:
                ts = _parse_ts(m_ml.group("ts"))
                if not since or ts is None or ts >= since:
                    current_ml.append(
                        MlFallback(
                            phase=m_ml.group(2),
                            model=m_ml.group(3),
                            reason=m_ml.group(4),
                            fallback=m_ml.group(5),
                        )
                    )
                continue

            # CIG_ROLLBACK
            m_cig = _RE_CIG_ROLLBACK.search(line)
            if m_cig:
                ts = _parse_ts(m_cig.group("ts"))
                if not since or ts is None or ts >= since:
                    current_cig.append(
                        CigRollback(
                            phase_id=m_cig.group(2),
                            trigger_goal=m_cig.group(3),
                            drift=float(m_cig.group(4)),
                            tolerance=float(m_cig.group(5)),
                            rollback_to=m_cig.group(6),
                        )
                    )
                continue

            # PHASE_EXEC
            m_pe = _RE_PHASE_EXEC.search(line)
            if m_pe:
                ts = _parse_ts(m_pe.group("ts"))
                if not since or ts is None or ts >= since:
                    current_execs.append(
                        PhaseExec(
                            phase_id=m_pe.group(2),
                            strength=m_pe.group(3),
                            explicit=m_pe.group(4) == "1",
                            vcap=m_pe.group(5),
                            conductor=m_pe.group(6),
                            songcal=float(m_pe.group(7)),
                        )
                    )
                continue

            # HPI_COMP
            m_hpi = _RE_HPI_COMP.search(line)
            if m_hpi:
                ts = _parse_ts(m_hpi.group("ts"))
                if not since or ts is None or ts >= since:
                    current_hpi = HpiComponents(
                        mode=m_hpi.group(2),
                        hpi=float(m_hpi.group(3)),
                        passed=m_hpi.group(4) == "1",
                        timbral=float(m_hpi.group(5)),
                        mert=float(m_hpi.group(6)),
                        artifact=float(m_hpi.group(7)),
                        emotional=float(m_hpi.group(8)),
                    )
                continue

            # PMGG per-phase entry → accumulate into current run buffer
            m_pmgg = _RE_PMGG.search(line)
            if m_pmgg:
                ts = _parse_ts(m_pmgg.group("ts"))
                if since and ts is not None and ts < since:
                    continue
                goal = m_pmgg.group(2)
                phase_id = m_pmgg.group(3)
                before = float(m_pmgg.group(4))
                after = float(m_pmgg.group(5))
                delta = float(m_pmgg.group(6))
                action = m_pmgg.group(7)
                current_pmgg[goal].append((phase_id, before, after, delta, action))
                continue

            # GOAL_SCORECARD → finalize run report
            m_sc = _RE_SCORECARD.search(line)
            if m_sc:
                ts = _parse_ts(m_sc.group("ts"))
                if since and ts is not None and ts < since:
                    continue
                material = m_sc.group("mat")
                mode = m_sc.group("mode")
                excellence = float(m_sc.group("exc"))
                n_violations = int(m_sc.group("viol"))
                goal_fields_raw = m_sc.group("goals")

                goals: dict[str, GoalEntry] = {}
                for gm in _RE_GOAL_FIELD.finditer(goal_fields_raw):
                    g_name = gm.group(1)
                    goals[g_name] = GoalEntry(
                        score=float(gm.group(2)),
                        threshold=float(gm.group(3)),
                        gap=float(gm.group(4)),
                        passed=gm.group(5) == "1",
                        applicable=gm.group(6) == "1",
                    )

                # Build worst-regression map from accumulated PMGG entries
                worst: dict[str, PhaseRegression] = {}
                for goal, entries in current_pmgg.items():
                    worst_delta = 0.0
                    worst_phase = ""
                    for phase_id, _bef, _aft, delta, _act in entries:
                        if delta < worst_delta:
                            worst_delta = delta
                            worst_phase = phase_id
                    if worst_phase:
                        worst[goal] = PhaseRegression(phase_id=worst_phase, goal=goal, delta=worst_delta)

                report = RunReport(
                    timestamp=m_sc.group("ts") or "unknown",
                    material=material,
                    mode=mode,
                    excellence=excellence,
                    n_violations=n_violations,
                    goals=goals,
                    phase_trajectories=dict(current_pmgg),
                    worst_regression=worst,
                    cig_rollbacks=list(current_cig),
                    ml_fallbacks=list(current_ml),
                    phase_execs=list(current_execs),
                    hpi=current_hpi,
                )
                runs.append(report)
                # Keep PMGG buffer alive (FeedbackChain may produce another SCORECARD)
                # but reset for clean separation of pipeline runs
                current_pmgg = defaultdict(list)
                current_cig = []
                current_ml = []
                current_execs = []
                current_hpi = None
                continue

    return runs


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_GOAL_ORDER = [
    "natuerlichkeit",
    "authentizitaet",
    "tonal_center",
    "timbre_authentizitaet",
    "artikulation",
    "transient_energie",
    "emotionalitaet",
    "mikro_dynamik",
    "groove",
    "transparenz",
    "waerme",
    "bass_kraft",
    "separation_fidelity",
    "brillanz",
    "raumtiefe",
]


def _goal_row(goal: str, entry: GoalEntry) -> str:
    prio = _PRIORITY.get(goal, "  ")
    mark = GRN("✓") if entry.passed else RED("✗")
    if not entry.applicable:
        mark = DIM("–")
    gap_str = f"{entry.gap:+.4f}"
    gap_col = GRN(gap_str) if entry.gap >= 0 else RED(gap_str)
    score_col = GRN(f"{entry.score:.4f}") if entry.passed else RED(f"{entry.score:.4f}")
    return f"  {mark} {prio:2s}  {goal:<25s}  score={score_col}  thr={entry.threshold:.4f}  gap={gap_col}" + (
        "" if entry.applicable else DIM("  [N/A]")
    )


def _phase_trajectory_table(goal: str, entries: list[tuple]) -> list[str]:
    lines: list[str] = []
    if not entries:
        return lines
    lines.append(f"    {DIM('Phase trajectory for')} {CYN(goal)}:")
    for phase_id, before, after, delta, action in entries:
        delta_col = GRN(f"{delta:+.4f}") if delta >= 0 else RED(f"{delta:+.4f}")
        action_col = GRN(action) if "passed" in action else YLW(action)
        lines.append(f"      {phase_id:<42s}  {before:.4f} → {after:.4f}  Δ={delta_col}  [{action_col}]")
    return lines


def print_run(report: RunReport, show_trajectories: bool = False, filter_goal: str | None = None) -> None:
    """Gibt einen formatierten Bericht eines Restaurierungs-Laufs auf der Konsole aus."""
    n_passed = len(report.passed_goals())
    n_total = sum(1 for e in report.goals.values() if e.applicable)
    header = (
        f"\n{'=' * 72}\n"
        f"{BLD('RUN')} {report.timestamp}  "
        f"mat={MGN(report.material)}  mode={CYN(report.mode)}\n"
        f"Excellence={BLD(f'{report.excellence:.4f}')}  "
        f"Goals: {GRN(str(n_passed))}/{n_total} passed"
        + (f"  {RED(f'{report.n_violations} violations')}" if report.n_violations else f"  {GRN('all passed')}")
        + f"\n{'=' * 72}"
    )
    print(header)

    # Group by priority
    for prio_label, prio_set in [("P1", _P1), ("P2", _P2), ("P3", _P3), ("P4", _P4), ("P5", _P5)]:
        group_goals = [g for g in _GOAL_ORDER if g in prio_set and g in report.goals]
        if filter_goal and not any(g == filter_goal for g in group_goals):
            continue
        if not group_goals:
            continue
        print(f"\n  {BLD(prio_label)}")
        for goal in group_goals:
            if filter_goal and goal != filter_goal:
                continue
            entry = report.goals[goal]
            print(_goal_row(goal, entry))
            # Show worst regression phase
            wr = report.worst_regression.get(goal)
            if wr and wr.delta < -0.001:
                print(f"         {DIM('▼ worst phase:')} {YLW(wr.phase_id)}  Δ={RED(f'{wr.delta:+.4f}')}")
            # Show full trajectory if requested
            if show_trajectories:
                traj = report.phase_trajectories.get(goal, [])
                for traj_line in _phase_trajectory_table(goal, traj):
                    print(traj_line)

    # Summary: failed goals ranked by gap (worst first)
    failed = sorted(
        [(g, e) for g, e in report.goals.items() if not e.passed and e.applicable],
        key=lambda x: x[1].gap,
    )
    if failed:
        print(f"\n  {RED(BLD('FAILED GOALS (sorted by gap):'))}")
        for goal, entry in failed:
            prio = _PRIORITY.get(goal, "  ")
            gap_str = RED(f"{entry.gap:+.4f}")
            wr = report.worst_regression.get(goal)
            worst_str = f"  ← worst phase: {YLW(wr.phase_id)} Δ={RED(f'{wr.delta:+.4f}')}" if wr else ""
            print(
                f"    {prio:2s}  {goal:<25s}  gap={gap_str}"
                f"  score={entry.score:.4f}  thr={entry.threshold:.4f}{worst_str}"
            )

    # Phase impact ranking: which phases caused the most cumulative regression?
    all_regressions: dict[str, float] = defaultdict(float)
    for goal, entries in report.phase_trajectories.items():
        for phase_id, _bef, _aft, delta, _act in entries:
            if delta < 0:
                all_regressions[phase_id] += delta  # sum of negative deltas

    if all_regressions:
        top_phases = sorted(all_regressions.items(), key=lambda x: x[1])[:8]
        print(f"\n  {BLD('TOP REGRESSION PHASES (cumulative Δ across goals):')}")
        for phase_id, cum_delta in top_phases:
            print(f"    {phase_id:<42s}  cumΔ={RED(f'{cum_delta:+.4f}')}")

    # HPI component breakdown
    if report.hpi is not None:
        h = report.hpi
        hpi_col = GRN(f"{h.hpi:.4f}") if h.passed else RED(f"{h.hpi:.4f}")
        print(f"\n  {BLD('HPI COMPONENTS:')}  hpi={hpi_col}  passed={GRN('✓') if h.passed else RED('✗')}")
        components = [
            ("timbral_fidelity", h.timbral),
            ("mert_similarity", h.mert),
            ("artifact_freedom", h.artifact),
            ("emotional_arc", h.emotional),
        ]
        for label, val in components:
            col = GRN(f"{val:.3f}") if val >= 0.80 else (YLW(f"{val:.3f}") if val >= 0.60 else RED(f"{val:.3f}"))
            bottleneck = "  ← BOTTLENECK" if val < 0.60 else ""
            print(f"    {label:<25s}  {col}{RED(bottleneck) if bottleneck else ''}")

    # CIG rollback summary
    if report.cig_rollbacks:
        print(f"\n  {BLD('CIG ROLLBACKS')} ({len(report.cig_rollbacks)} total):")
        for rb in report.cig_rollbacks:
            drift_col = RED(f"{rb.drift:+.4f}")
            print(
                f"    {rb.phase_id:<35s}  trigger={YLW(rb.trigger_goal):<20s}  "
                f"drift={drift_col}  tol={rb.tolerance:+.4f}  → {DIM(rb.rollback_to)}"
            )

    # ML fallback summary
    if report.ml_fallbacks:
        print(f"\n  {BLD('ML FALLBACKS')} ({len(report.ml_fallbacks)} total):")
        for fb in report.ml_fallbacks:
            print(f"    {fb.phase:<35s}  model={YLW(fb.model):<20s}  reason={RED(fb.reason):<8s}  → {fb.fallback}")

    # Phase strength bottleneck table (phases with very low strength)
    if report.phase_execs:
        capped = [
            (pe, float(pe.strength))
            for pe in report.phase_execs
            if pe.strength not in ("None", "") and float(pe.strength) < 0.35
        ]
        if capped:
            capped.sort(key=lambda x: x[1])
            print(f"\n  {BLD('STRENGTH BOTTLENECKS')} (phases with strength < 0.35):")
            for pe, s in capped[:10]:
                vcap_str = f"vcap={pe.vcap}" if pe.vcap != "None" else ""
                cond_str = f"cond={pe.conductor}" if pe.conductor != "None" else ""
                flags = "  ".join(x for x in [vcap_str, cond_str] if x)
                print(
                    f"    {pe.phase_id:<40s}  str={RED(f'{s:.3f}')}  explicit={'Y' if pe.explicit else 'N'}  "
                    f"songcal={pe.songcal:.3f}  {DIM(flags)}"
                )

    print()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def runs_to_json(runs: list[RunReport]) -> dict:
    """Serialisiert eine Liste von Lauf-Berichten als JSON-kompatibles Dict."""
    out = []
    for r in runs:
        goals_dict = {}
        for g, e in r.goals.items():
            goals_dict[g] = {
                "score": round(e.score, 4),
                "threshold": round(e.threshold, 4),
                "gap": round(e.gap, 4),
                "passed": e.passed,
                "applicable": e.applicable,
                "priority": _PRIORITY.get(g, ""),
                "worst_regression_phase": r.worst_regression[g].phase_id if g in r.worst_regression else None,
                "worst_regression_delta": round(r.worst_regression[g].delta, 4) if g in r.worst_regression else None,
            }
        out.append(
            {
                "timestamp": r.timestamp,
                "material": r.material,
                "mode": r.mode,
                "excellence": round(r.excellence, 4),
                "violations": r.n_violations,
                "goals": goals_dict,
            }
        )
    return {"runs": out}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Einstiegspunkt des CLI-Diagnose-Monitors."""
    parser = argparse.ArgumentParser(
        description="Aurik 10.0.0 Musical-Goals-Diagnose-Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log", default=str(DEFAULT_LOG), help="Pfad zur Log-Datei")
    parser.add_argument("--all", action="store_true", help="Alle Läufe anzeigen (Standard: nur letzter)")
    parser.add_argument("--runs", type=int, default=1, help="Letzte N Läufe anzeigen")
    parser.add_argument("--goal", default=None, help="Nur dieses Ziel anzeigen")
    parser.add_argument(
        "--trajectory",
        action="store_true",
        help="Phase-Trajektorie pro Ziel anzeigen (zeigt jeden PMGG-Schritt)",
    )
    parser.add_argument(
        "--since",
        default=None,
        help='Nur Läufe ab diesem Zeitpunkt (z.B. "2026-04-26 02:00")',
    )
    parser.add_argument("--json", action="store_true", help="Maschinenlesbare JSON-Ausgabe")
    parser.add_argument("--watch", action="store_true", help="Log live beobachten (Ctrl+C zum Beenden)")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log-Datei nicht gefunden: {log_path}", file=sys.stderr)
        sys.exit(1)

    since_dt: datetime | None = None
    if args.since:
        try:
            since_dt = datetime.strptime(args.since, "%Y-%m-%d %H:%M")
        except ValueError:
            print(f"Ungültiges --since Format (erwartet: 'YYYY-MM-DD HH:MM'): {args.since}", file=sys.stderr)
            sys.exit(1)

    if args.watch:
        _watch_mode(log_path, args)
        return

    runs = parse_log(log_path, since=since_dt)

    if not runs:
        if since_dt:
            print(
                f"Keine GOAL_SCORECARD-Einträge im Log seit {args.since} gefunden.\n"
                f"Hinweis: GOAL_SCORECARD wird erst ab dem nächsten Lauf produziert\n"
                f"(nach dem UV3 §GOAL_MONITOR-Patch). Ältere Läufe haben keine Scorecard.",
                file=sys.stderr,
            )
        else:
            print(
                "Keine GOAL_SCORECARD-Einträge im Log gefunden.\n"
                "Hinweis: Das Scorecard-Log wird erst nach dem nächsten Aurik-Lauf produziert.",
                file=sys.stderr,
            )
        sys.exit(0)

    n_show = len(runs) if args.all else min(args.runs, len(runs))
    selected = runs[-n_show:]

    if args.json:
        print(json.dumps(runs_to_json(selected), ensure_ascii=False, indent=2))
        return

    print(f"\n{BLD('Aurik 10.0.0 — Musical Goals Monitor')}  {DIM(f'[{n_show} Lauf/Läufe aus {log_path.name}]')}")

    for report in selected:
        print_run(report, show_trajectories=args.trajectory, filter_goal=args.goal)

    # Cross-run summary if showing multiple runs
    if len(selected) > 1:
        _print_cross_run_summary(selected, args.goal)


def _print_cross_run_summary(runs: list[RunReport], filter_goal: str | None) -> None:
    """Show which goals fail most consistently across runs."""
    fail_count: dict[str, int] = defaultdict(int)
    gap_sum: dict[str, float] = defaultdict(float)
    for r in runs:
        for g, e in r.goals.items():
            if filter_goal and g != filter_goal:
                continue
            if not e.applicable:
                continue
            if not e.passed:
                fail_count[g] += 1
            gap_sum[g] += e.gap

    if not fail_count:
        print(f"\n{GRN('✓ Alle beobachteten Ziele in allen Läufen erfüllt.')}")
        return

    print(f"\n{BLD('CROSS-RUN FAILURE SUMMARY')} ({len(runs)} Läufe):")
    ranked = sorted(fail_count.items(), key=lambda x: (-x[1], gap_sum[x[0]]))
    for goal, count in ranked:
        prio = _PRIORITY.get(goal, "  ")
        avg_gap = gap_sum[goal] / len(runs)
        progress_bar = RED("█" * count) + DIM("░" * (len(runs) - count))
        print(
            f"  {prio:2s}  {goal:<25s}  fail={RED(str(count))}/{len(runs)}  "
            f"avgGap={RED(f'{avg_gap:+.4f}') if avg_gap < 0 else GRN(f'{avg_gap:+.4f}')}  [{progress_bar}]"
        )


def _watch_mode(log_path: Path, args: argparse.Namespace) -> None:
    """Tail the log file and print new GOAL_SCORECARD entries as they appear."""
    import time  # pylint: disable=import-outside-toplevel

    print(f"{BLD('Aurik 10.0.0 — Goal Monitor LIVE')}  {DIM(f'(watching {log_path.name})')}")
    print(DIM("Warte auf neuen Lauf... (Ctrl+C zum Beenden)"))

    seen_scorecards: set[str] = set()
    # Populate already-seen runs so we only show new ones
    try:
        for r in parse_log(log_path):
            seen_scorecards.add(r.timestamp)
    except Exception:
        logger.warning("goal_monitor.py::_watch_mode fallback", exc_info=True)

    try:
        while True:
            time.sleep(3)
            try:
                new_runs = [r for r in parse_log(log_path) if r.timestamp not in seen_scorecards]
                for r in new_runs:
                    print_run(r, show_trajectories=args.trajectory, filter_goal=args.goal)
                    seen_scorecards.add(r.timestamp)
            except Exception as exc:
                print(f"{YLW('Parse-Fehler:')} {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print(f"\n{DIM('Monitor beendet.')}")


if __name__ == "__main__":
    main()
