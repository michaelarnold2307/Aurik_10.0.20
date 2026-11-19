"""Release readiness check with deterministic 0-10 scoring.

The script validates documented quality gates against audit trail entries,
calculates a release readiness score (0..10), writes a JSON report and returns
an exit code suitable for CI usage.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

SCORE_MAX: float = 10.0
RELEASE_READY_THRESHOLD: float = 9.5
MIN_REQUIRED_GATES_STRICT: int = 5
DOC_EXEMPT_PREFIXES: tuple[str, ...] = (
    "quality_gate_passed::",
    "vocal_quality::",
    "scores.quality_gates::",
    "features.quality_gates::",
    "release_result::",
)

_VOICE_FIRST_BLOCKER_ALIASES: dict[str, tuple[str, ...]] = {
    "vqi": ("vqi", "vqi_gate", "vocal_quality_index"),
    "formant": ("formant", "formant_integrity", "vocal_formant_stability"),
    "vibrato": ("vibrato", "vibrato_depth", "vibrato_depth_preserved"),
    "micro_dynamics": (
        "micro_dynamics",
        "mikrodynamik",
        "micro_dynamic_correlation",
        "mikrodynamik_korrelation",
    ),
}


def load_audit_log(audit_path: str = "audit/audit_trail.json") -> list[dict[str, Any]]:
    """Load audit entries from JSON, returning an empty list on missing/invalid files."""
    path = Path(audit_path)
    if not path.exists():
        print("Audit-Log nicht gefunden.")
        return []
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (json.JSONDecodeError, OSError):
        print("Audit-Log konnte nicht gelesen werden.")
        return []

    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    return []


def check_compliance(
    audit_data: list[dict[str, Any]],
    doc_gates_path: str = "docs/audit/QUALITY_GATES.md",
    doc_policy_path: str = "policy/policy_engine.py",
    include_diagnostic_gates: bool = True,
) -> tuple[bool, list[str]]:
    """Verify gate documentation and failing gates against audit entries."""
    compliance_ok = True
    change_set: set[str] = set()

    doc_gates = ""
    gates_path = Path(doc_gates_path)
    if gates_path.exists():
        with gates_path.open("r", encoding="utf-8") as file:
            doc_gates = file.read()

    # Keep a strict policy-file existence check as baseline safety signal.
    if not Path(doc_policy_path).exists():
        compliance_ok = False
        change_set.add("Policy-Datei fehlt: policy/policy_engine.py")

    for gate, value in _iter_gates(audit_data, include_diagnostic_gates=include_diagnostic_gates):
        needs_doc = not gate.startswith(DOC_EXEMPT_PREFIXES)
        if needs_doc and gate not in doc_gates:
            compliance_ok = False
            change_set.add(f"Quality-Gate '{gate}' nicht in Dokumentation.")
        if value is False:
            compliance_ok = False
            change_set.add(f"Quality-Gate '{gate}' nicht bestanden.")

    if include_diagnostic_gates:
        voice_first_issues = _check_voice_first_blockers(audit_data)
        if voice_first_issues:
            compliance_ok = False
            change_set.update(voice_first_issues)

    total_gates, _ = _gate_stats(audit_data, include_diagnostic_gates=include_diagnostic_gates)
    if include_diagnostic_gates and total_gates < MIN_REQUIRED_GATES_STRICT:
        compliance_ok = False
        change_set.add(
            f"Audit-Abdeckung zu gering: {total_gates} Gates < Mindestabdeckung {MIN_REQUIRED_GATES_STRICT} "
            "(diagnostic_gates aktiv)."
        )

    return compliance_ok, sorted(change_set)


def _gate_stats(audit_data: list[dict[str, Any]], include_diagnostic_gates: bool = False) -> tuple[int, int]:
    """Return total gate count and number of passed gates from audit entries.

    Supports all known gate formats from audit_trail.json.
    """
    total = 0
    passed = 0
    for _, value in _iter_gates(audit_data, include_diagnostic_gates=include_diagnostic_gates):
        total += 1
        if value is not False:
            passed += 1
    return total, passed


def _iter_gates(audit_data: list[dict[str, Any]], include_diagnostic_gates: bool = False) -> Iterator[tuple[str, bool]]:
    """Yield normalized gate entries across heterogeneous audit payloads."""
    negative_release_states = {
        "release_check_not_available",
        "failed",
        "error",
        "blocked",
        "not_ready",
    }

    for entry in audit_data:
        # 1) Canonical explicit results
        results = entry.get("results", {})
        if isinstance(results, dict):
            for gate, value in results.items():
                if isinstance(gate, str) and isinstance(value, bool):
                    yield gate, value

        # 2) Legacy phase-level boolean
        qgp = entry.get("quality_gate_passed")
        if isinstance(qgp, bool):
            step = entry.get("step", "phase")
            yield f"quality_gate_passed::{step}", qgp

        if include_diagnostic_gates:
            # 3) Vocal checks at root level
            vocal_check = entry.get("vocal_quality_check", {})
            if isinstance(vocal_check, dict):
                for gate, value in vocal_check.items():
                    if isinstance(gate, str) and isinstance(value, bool):
                        yield f"vocal_quality::{gate}", value

            # 4) Nested quality_gates inside scores/features
            for parent_name in ("scores", "features"):
                parent = entry.get(parent_name, {})
                if isinstance(parent, dict):
                    nested = parent.get("quality_gates", {})
                    if isinstance(nested, dict):
                        for gate, value in nested.items():
                            if isinstance(gate, str) and isinstance(value, bool):
                                yield f"{parent_name}.quality_gates::{gate}", value

            # 5) Release status at root level
            release_result = entry.get("release_result", {})
            if isinstance(release_result, dict):
                status = release_result.get("status")
                if isinstance(status, str):
                    normalized = status.strip().lower()
                    if normalized in negative_release_states:
                        yield "release_result::status", False


def _is_vocal_entry(entry: dict[str, Any]) -> bool:
    """Erkennt, ob ein Audit-Eintrag als vokalrelevant behandelt werden muss."""
    scores = entry.get("scores", {})
    if isinstance(scores, dict):
        media = scores.get("media_characteristics", {})
        if isinstance(media, dict) and bool(media.get("vocal", False)):
            return True
        for key in ("Singing voice", "Vocals", "panns_singing"):
            value = scores.get(key)
            if isinstance(value, (int, float)) and float(value) >= 0.25:
                return True

    vocal_check = entry.get("vocal_quality_check", {})
    return isinstance(vocal_check, dict) and bool(vocal_check)


def _resolve_blocker_status(vocal_check: dict[str, Any], aliases: tuple[str, ...]) -> bool | None:
    """Liefert den ersten booleschen Wert für einen Blocker oder None wenn nicht vorhanden."""
    for alias in aliases:
        value = vocal_check.get(alias)
        if isinstance(value, bool):
            return value
    return None


def _check_voice_first_blockers(audit_data: list[dict[str, Any]]) -> set[str]:
    """Prüft [RELEASE_MUST] Voice-First-Blocker in vokalrelevanten Audit-Einträgen."""
    issues: set[str] = set()
    missing: set[str] = set()
    failed: set[str] = set()
    vocal_entries = 0

    for entry in audit_data:
        if not _is_vocal_entry(entry):
            continue
        vocal_entries += 1
        vocal_check = entry.get("vocal_quality_check", {})
        if not isinstance(vocal_check, dict):
            vocal_check = {}

        for blocker, aliases in _VOICE_FIRST_BLOCKER_ALIASES.items():
            status = _resolve_blocker_status(vocal_check, aliases)
            if status is None:
                missing.add(blocker)
            elif status is False:
                failed.add(blocker)

    if vocal_entries == 0:
        return issues
    if missing:
        issues.add(
            "Voice-First-Blocker fehlen fuer vokalrelevante Runs: "
            + ", ".join(sorted(missing))
            + f" (runs={vocal_entries})."
        )
    if failed:
        issues.add("Voice-First-Blocker nicht bestanden: " + ", ".join(sorted(failed)) + f" (runs={vocal_entries}).")
    return issues


def calculate_release_score(
    compliance_ok: bool,
    changes: list[str],
    audit_data: list[dict[str, Any]],
    include_diagnostic_gates: bool = False,
) -> float:
    """Compute release score in [0, 10].

    Rule set:
    - Base score is gate pass-rate scaled to 0..10.
    - Missing documentation and failed gates apply additional penalties.
    - Empty audit log is penalized to avoid false 10/10 reports.
    """
    total_gates, passed_gates = _gate_stats(audit_data, include_diagnostic_gates=include_diagnostic_gates)
    if total_gates == 0:
        score = 6.0
    else:
        score = SCORE_MAX * (passed_gates / total_gates)

    undocumented_count = sum(1 for c in changes if "nicht in Dokumentation" in c)
    failed_gate_count = sum(1 for c in changes if "nicht bestanden" in c)
    score -= undocumented_count * 0.2
    score -= failed_gate_count * 0.5

    if not compliance_ok:
        score -= 0.5

    return round(max(0.0, min(SCORE_MAX, score)), 2)


def generate_release_report(
    compliance_ok: bool,
    changes: list[str],
    audit_data: list[dict[str, Any]],
    output_path: str = "audit/release_report.json",
    include_diagnostic_gates: bool = False,
) -> dict[str, Any]:
    """Generate and persist release report JSON."""
    score = calculate_release_score(
        compliance_ok,
        changes,
        audit_data,
        include_diagnostic_gates=include_diagnostic_gates,
    )
    release_ready = compliance_ok and score >= RELEASE_READY_THRESHOLD

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "compliance_ok": compliance_ok,
        "release_ready": release_ready,
        "score": score,
        "score_max": SCORE_MAX,
        "changes": changes,
        "audit_summary": audit_data[-5:] if audit_data else [],
        "gate_stats": {
            "total": _gate_stats(audit_data, include_diagnostic_gates=include_diagnostic_gates)[0],
            "passed": _gate_stats(audit_data, include_diagnostic_gates=include_diagnostic_gates)[1],
            "diagnostic_gates_included": include_diagnostic_gates,
        },
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"Release-Report generiert: {output_path}")
    print(f"Release-Score: {score:.2f}/{SCORE_MAX:.0f}")
    if not release_ready:
        print("WARNUNG: Release nicht freigegeben. Bitte Änderungen prüfen.")
        for change in changes:
            print(f"- {change}")

    return report


def check_release(
    audit_path: str = "audit/audit_trail.json",
    gates_doc: str = "docs/audit/QUALITY_GATES.md",
    policy_path: str = "policy/policy_engine.py",
    include_diagnostic_gates: bool = True,
    output_path: str = "audit/release_report.json",
) -> dict[str, Any]:
    """Return release status dict for callers inside the production pipeline."""
    audit_data = load_audit_log(audit_path)
    compliance_ok, changes = check_compliance(
        audit_data,
        gates_doc,
        policy_path,
        include_diagnostic_gates=include_diagnostic_gates,
    )
    report = generate_release_report(
        compliance_ok,
        changes,
        audit_data,
        output_path=output_path,
        include_diagnostic_gates=include_diagnostic_gates,
    )
    return {
        "status": "release_ready" if report.get("release_ready") else "blocked",
        "release_ready": bool(report.get("release_ready")),
        "score": float(report.get("score", 0.0)),
        "score_max": float(report.get("score_max", SCORE_MAX)),
        "changes": report.get("changes", []),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns 0 for release-ready reports, else 1."""
    parser = argparse.ArgumentParser(description="Aurik Release-Check und Score-Berechnung")
    parser.add_argument("--audit-path", default="audit/audit_trail.json")
    parser.add_argument("--gates-doc", default="docs/audit/QUALITY_GATES.md")
    parser.add_argument("--policy-path", default="policy/policy_engine.py")
    parser.add_argument("--output", default="audit/release_report.json")
    parser.add_argument(
        "--exclude-diagnostic-gates",
        action="store_true",
        help="Nur kanonische Results-Gates auswerten (weniger strikt).",
    )
    args = parser.parse_args(argv)

    report = check_release(
        audit_path=args.audit_path,
        gates_doc=args.gates_doc,
        policy_path=args.policy_path,
        include_diagnostic_gates=not args.exclude_diagnostic_gates,
        output_path=args.output,
    )
    return 0 if report.get("release_ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
