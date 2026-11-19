"""Runtime spec compliance check for Aurik live frontend/backend runs.

Evaluates the latest processing run in backend/frontend logs against key
RELEASE_MUST invariants from copilot-instructions/spec sections.

Output:
- JSON report at audit/runtime_spec_report.json
- Exit code 0 only when all required checks pass
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_ALLOWED_MODE_PATTERNS = (
    re.compile(r"\bmode\s*=\s*\"?restoration\"?\b", re.IGNORECASE),
    re.compile(r"\bmode\s*=\s*\"?studio2026\"?\b", re.IGNORECASE),
    re.compile(r"\bmode\s*=\s*\"?studio_2026\"?\b", re.IGNORECASE),
    re.compile(r'"mode"\s*:\s*"restoration"', re.IGNORECASE),
    re.compile(r'"mode"\s*:\s*"studio2026"', re.IGNORECASE),
    re.compile(r'"mode"\s*:\s*"studio_2026"', re.IGNORECASE),
)


@dataclass
class CheckResult:
    id: str
    title: str
    passed: bool
    required: bool
    evidence: str


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _find_latest_run_window(backend_lines: list[str]) -> tuple[list[str], int, int]:
    start_idx = -1
    for i, line in enumerate(backend_lines):
        if "AurikDenker.denke() gestartet" in line:
            start_idx = i
    if start_idx < 0:
        return [], -1, -1

    end_idx = len(backend_lines)
    for i in range(start_idx + 1, len(backend_lines)):
        if "AurikDenker.denke() abgeschlossen" in backend_lines[i]:
            end_idx = i + 1
            break

    return backend_lines[start_idx:end_idx], start_idx, end_idx


def _contains(lines: list[str], needle: str) -> bool:
    return any(needle in l for l in lines)


def _count(lines: list[str], needle: str) -> int:
    return sum(1 for l in lines if needle in l)


def _count_medium_detector_activations(lines: list[str]) -> int:
    """Counts effective MediumDetector activations, excluding cached handovers.

    Only UV3 summary lines with an explicit source marker are considered.
    Cached reuse (Quelle=cached) must not be treated as a fresh detector run.
    """

    count = 0
    for line in lines:
        if "🔍 MediumDetector:" not in line:
            continue
        if "Quelle=" not in line:
            continue
        if "Quelle=cached" in line:
            continue
        count += 1
    return count


def _has_allowed_mode(lines: list[str]) -> bool:
    """True if run logs contain a canonical allowed mode marker.

    Accepts both internal and UI spellings:
    - restoration
    - studio2026
    - studio_2026
    """

    return any(any(rx.search(line) for rx in _ALLOWED_MODE_PATTERNS) for line in lines)


def _extract_float(lines: list[str], pattern: str) -> float | None:
    rx = re.compile(pattern)
    for line in lines:
        m = rx.search(line)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                return None
    return None


def _has_material_downgrade_cassette_to_tape(lines: list[str]) -> bool:
    """Erkennt, ob ein erkannter Kassette-Typ im selben Lauf auf generisches Tape fällt."""
    saw_mc_cassette = any("🔍 MediumDetector: Material=cassette" in line for line in lines)
    saw_songcal_tape = any("🎛️ SongCalibration: material=tape" in line for line in lines)
    return bool(saw_mc_cassette and saw_songcal_tape)


def run_check(backend_log: Path, frontend_log: Path, output: Path) -> dict[str, Any]:
    backend_lines = _read_lines(backend_log)
    frontend_lines = _read_lines(frontend_log)

    run_lines, start_idx, end_idx = _find_latest_run_window(backend_lines)
    checks: list[CheckResult] = []

    checks.append(
        CheckResult(
            id="entry_denker",
            title="AurikDenker Einstieg genutzt",
            passed=start_idx >= 0,
            required=True,
            evidence=(
                "AurikDenker.denke() gestartet gefunden" if start_idx >= 0 else "Kein AurikDenker-Start im Backend-Log"
            ),
        )
    )

    if run_lines:
        run_completed = _contains(run_lines, "AurikDenker.denke() abgeschlossen")
        checks.append(
            CheckResult(
                id="run_completed",
                title="AurikDenker Lauf vollständig abgeschlossen",
                passed=run_completed,
                required=True,
                evidence=(
                    "AurikDenker.denke() abgeschlossen gefunden"
                    if run_completed
                    else "Kein AurikDenker-Abschluss im Run-Window"
                ),
            )
        )

        mode_ok = _has_allowed_mode(run_lines)
        checks.append(
            CheckResult(
                id="mode_contract",
                title="Nur erlaubter Betriebsmodus",
                passed=mode_ok,
                required=True,
                evidence="mode=restoration/studio_2026 vorhanden" if mode_ok else "Kein erlaubter mode=... Eintrag",
            )
        )

        preanalysis_ok = _contains(run_lines, "Verwende gecachten DefectScan") or _contains(
            run_lines, "cached_defect_result übernommen"
        )
        checks.append(
            CheckResult(
                id="preanalysis_handover",
                title="§2.47a PreAnalysis-Handover sichtbar",
                passed=preanalysis_ok,
                required=True,
                evidence=(
                    "Cached Defect/PreAnalysis im Lauf genutzt" if preanalysis_ok else "Kein Handover-Hinweis im Lauf"
                ),
            )
        )

        medium_detect_count = _count_medium_detector_activations(run_lines)
        checks.append(
            CheckResult(
                id="medium_detect_single",
                title="MediumDetector im Lauf nicht mehrfach aktiv",
                passed=medium_detect_count <= 1,
                required=True,
                evidence=f"🔍 MediumDetector fresh_count={medium_detect_count} (cached ignoriert)",
            )
        )

        artifact_freedom = _extract_float(run_lines, r"§2\.49 Final artifact_freedom=([0-9]+\.[0-9]+)")
        checks.append(
            CheckResult(
                id="artifact_gate",
                title="§2.49 Artifact-Freedom Gate erfüllt",
                passed=artifact_freedom is not None and artifact_freedom >= 0.95,
                required=True,
                evidence=(
                    f"artifact_freedom={artifact_freedom:.3f}"
                    if artifact_freedom is not None
                    else "Kein Final-AFG-Eintrag"
                ),
            )
        )

        hpi_present = _contains(run_lines, "§2.44 HPI(")
        checks.append(
            CheckResult(
                id="hpi_gate",
                title="§2.44 HPI-Gate protokolliert",
                passed=hpi_present,
                required=True,
                evidence="HPI-Eintrag vorhanden" if hpi_present else "Kein HPI-Eintrag im Lauf",
            )
        )

        # §2.53 runtime telemetry propagation checks (strictly required by spec)
        joy_present = _contains(run_lines, "joy_runtime_index") or _contains(frontend_lines[-400:], "Freude")
        rec_present = _contains(run_lines, "auto_improvement_recommendations")
        cluster_present = _contains(run_lines, "cluster_policy")

        checks.append(
            CheckResult(
                id="exp_joy_runtime",
                title="§2.53 joy_runtime_index propagiert",
                passed=joy_present,
                required=True,
                evidence="Joy-Index-Signal gefunden" if joy_present else "Kein joy_runtime_index/Freude-Signal im Log",
            )
        )
        checks.append(
            CheckResult(
                id="exp_auto_improve",
                title="§2.53 auto_improvement_recommendations propagiert",
                passed=rec_present,
                required=True,
                evidence=(
                    "Auto-Improve-Signal gefunden"
                    if rec_present
                    else "Kein auto_improvement_recommendations-Signal im Log"
                ),
            )
        )
        checks.append(
            CheckResult(
                id="exp_cluster_policy",
                title="§2.53 cluster_policy propagiert",
                passed=cluster_present,
                required=True,
                evidence="cluster_policy gefunden" if cluster_present else "Kein cluster_policy-Signal im Log",
            )
        )

        dag_final_logged = _contains(run_lines, "§7.5a Phase-DAG FINAL:")
        dag_final_has_violation = any(
            "§7.5a Phase-DAG FINAL:" in line and "HARD_BEFORE-Verletzung" in line for line in run_lines
        )
        checks.append(
            CheckResult(
                id="dag_final_clean",
                title="§7.5a Finale DAG-Validierung ohne Verletzung",
                passed=dag_final_logged and not dag_final_has_violation,
                required=True,
                evidence=(
                    "Finale DAG-Validierung vorhanden, keine HARD_BEFORE-Verletzung"
                    if dag_final_logged and not dag_final_has_violation
                    else (
                        "Finale DAG-Validierung fehlt"
                        if not dag_final_logged
                        else "Finale DAG-Validierung meldet HARD_BEFORE-Verletzung"
                    )
                ),
            )
        )

        phase12_started = _contains(run_lines, "phase_12_wow_flutter_fix startet")
        phase12_low_conf_skip = _contains(run_lines, "Phase 12: Pitch-Konfidenz zu niedrig")
        checks.append(
            CheckResult(
                id="phase12_effective_when_started",
                title="Phase 12 wirkt bei Ausführung (kein low-confidence Komplett-Skip)",
                passed=(not phase12_started) or (not phase12_low_conf_skip),
                required=True,
                evidence=(
                    "Phase 12 nicht im Laufplan"
                    if not phase12_started
                    else (
                        "Phase 12 ohne low-confidence Skip ausgeführt"
                        if not phase12_low_conf_skip
                        else "Phase 12 gestartet, aber low-confidence Skip erkannt"
                    )
                ),
            )
        )

        material_downgrade = _has_material_downgrade_cassette_to_tape(run_lines)
        checks.append(
            CheckResult(
                id="material_specificity_guard",
                title="Material-Spezifität erhalten (kein cassette→tape Downcast)",
                passed=not material_downgrade,
                required=True,
                evidence=(
                    "Kein cassette→tape Downcast im Lauf"
                    if not material_downgrade
                    else "MediumDetector meldet cassette, SongCalibration nutzt tape"
                ),
            )
        )

        exzellenz_path_ok = (
            _contains(run_lines, "Legacy-Goal-Messpfad")
            or _contains(run_lines, "messe_und_repariere")
            or _contains(run_lines, "Zeit-Domain-Repair für P3-P5")
        )
        checks.append(
            CheckResult(
                id="exzellenz_api_contract",
                title="§2.53a Exzellenz-API-Pfad erkennbar",
                passed=exzellenz_path_ok,
                required=False,
                evidence=(
                    "Exzellenz-API-Hinweis gefunden"
                    if exzellenz_path_ok
                    else "Kein expliziter Exzellenz-API-Hinweis im Log (inconclusive)"
                ),
            )
        )

    required_checks = [c for c in checks if c.required]
    passed_required = sum(1 for c in required_checks if c.passed)
    compliance_ok = passed_required == len(required_checks) and len(required_checks) > 0

    report = {
        "timestamp": datetime.now().isoformat(),
        "backend_log": str(backend_log),
        "frontend_log": str(frontend_log),
        "latest_run_window": {"start_line": start_idx + 1 if start_idx >= 0 else -1, "end_line": end_idx},
        "compliance_ok": compliance_ok,
        "required_passed": passed_required,
        "required_total": len(required_checks),
        "checks": [asdict(c) for c in checks],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Runtime Spec Compliance Check (Aurik)")
    parser.add_argument("--backend-log", default="logs/aurik_backend.log")
    parser.add_argument("--frontend-log", default="logs/aurik_frontend.out")
    parser.add_argument("--output", default="audit/runtime_spec_report.json")
    args = parser.parse_args()

    report = run_check(Path(args.backend_log), Path(args.frontend_log), Path(args.output))
    print(f"Runtime-Spec-Report: {args.output}")
    print(
        f"Required checks: {report['required_passed']}/{report['required_total']} | "
        f"compliance_ok={report['compliance_ok']}"
    )

    for chk in report["checks"]:
        state = "PASS" if chk["passed"] else "FAIL"
        req = "REQ" if chk["required"] else "OPT"
        print(f"[{req}][{state}] {chk['id']}: {chk['evidence']}")

    return 0 if report["compliance_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
