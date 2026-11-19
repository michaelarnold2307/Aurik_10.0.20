"""Audit psychoacoustic harmonization coverage across all phase modules.

Non-blocking audit: identifies modules that still miss central strength/locality
harmonization conventions and writes a prioritized report for production fixes.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PhaseAuditRow:
    phase_file: str
    has_process: bool
    has_kwargs: bool
    uses_strength: bool
    uses_phase_locality_factor: bool
    computes_effective_strength: bool
    has_zero_strength_skip: bool
    has_effective_strength_telemetry: bool
    harmonized: bool


@dataclass(frozen=True)
class AuditSummary:
    total_phases: int
    harmonized_phases: int
    missing_harmonization: int
    coverage_ratio: float


_PROCESS_RE = re.compile(r"def\s+process\s*\((.*?)\)\s*->", re.DOTALL)


def _scan_file(path: Path, root: Path) -> PhaseAuditRow:
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = _PROCESS_RE.search(text)
    has_process = m is not None
    sig = m.group(1) if m else ""

    has_kwargs = "**kwargs" in sig
    uses_strength = bool(re.search(r"kwargs\.get\(\s*['\"]strength['\"]", text)) or " effective_strength" in text
    uses_phase_locality_factor = "phase_locality_factor" in text
    computes_effective_strength = "effective_strength" in text or "_effective_strength" in text
    has_zero_strength_skip = bool(
        re.search(r"effective_strength\s*<=\s*(?:0\.0|1e-6)", text)
        or re.search(r"_effective_strength\s*<=\s*(?:0\.0|1e-6)", text)
    )
    has_effective_strength_telemetry = bool(
        re.search(r"['\"]effective_strength['\"]\s*:", text) or re.search(r"['\"]phase_locality_factor['\"]\s*:", text)
    )

    harmonized = all(
        [
            has_process,
            has_kwargs,
            uses_strength,
            uses_phase_locality_factor,
            computes_effective_strength,
            has_zero_strength_skip,
            has_effective_strength_telemetry,
        ]
    )

    return PhaseAuditRow(
        phase_file=str(path.relative_to(root)),
        has_process=has_process,
        has_kwargs=has_kwargs,
        uses_strength=uses_strength,
        uses_phase_locality_factor=uses_phase_locality_factor,
        computes_effective_strength=computes_effective_strength,
        has_zero_strength_skip=has_zero_strength_skip,
        has_effective_strength_telemetry=has_effective_strength_telemetry,
        harmonized=harmonized,
    )


def run_audit(root: Path) -> tuple[AuditSummary, list[PhaseAuditRow], list[str]]:
    phase_files = sorted(
        p for p in (root / "backend" / "core" / "phases").glob("phase_*.py") if p.name != "phase_interface.py"
    )
    rows = [_scan_file(p, root) for p in phase_files]
    harmonized = [r for r in rows if r.harmonized]
    missing = [r for r in rows if not r.harmonized]

    summary = AuditSummary(
        total_phases=len(rows),
        harmonized_phases=len(harmonized),
        missing_harmonization=len(missing),
        coverage_ratio=float(len(harmonized) / max(1, len(rows))),
    )

    recommendations: list[str] = []
    if missing:
        recommendations.append(
            "Priorisiere Phasen ohne effective_strength + zero-strength-skip, um Ueberverarbeitung zu vermeiden."
        )
        recommendations.append(
            "Fuehre phase_locality_factor in verbleibenden Modulen ein, damit Defekt-Lokalitaet global respektiert wird."
        )
        recommendations.append(
            "Ergaenze Telemetrie (effective_strength, phase_locality_factor) fuer Auditierbarkeit in allen Phasen."
        )
    else:
        recommendations.append("Alle Phasen entsprechen den Harmonisierungskriterien.")

    return summary, rows, recommendations


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase Harmonization Audit (non-blocking)")
    parser.add_argument(
        "--output",
        default="reports/phase_harmonization_audit.json",
        help="Output report path",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    summary, rows, recommendations = run_audit(root)

    report = {
        "audit": "phase_harmonization_v1",
        "summary": asdict(summary),
        "rows": [asdict(r) for r in rows],
        "recommendations": recommendations,
    }

    out_path = (root / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print(
        f"Phase Harmonization Audit: {summary.harmonized_phases}/{summary.total_phases} harmonized "
        f"({summary.coverage_ratio * 100:.1f}%)"
    )
    print(f"Report: {out_path}")
    # Non-blocking by design: always exit 0 to support iterative production hardening.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
