"""Consolidate release and runtime audit reports into one final status.

This module resolves contradictory top-level signals by enforcing a strict policy:
- final_ready is True only if BOTH release and runtime required checks are green
- contradictions are explicit in output metadata
- newest-run policy based on report timestamps
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ConsolidatedStatus:
    timestamp: str
    release_report_path: str
    runtime_report_path: str
    release_timestamp: str | None
    runtime_timestamp: str | None
    latest_source: str
    release_ready: bool
    runtime_compliance_ok: bool
    required_passed: int
    required_total: int
    bridge_import_status_present: bool
    bridge_import_status_passed: bool | None
    bridge_import_status_evidence: str
    coverage_manifest_path: str | None
    coverage_baseline_path: str | None
    coverage_manifest_ok: bool
    coverage_baseline_ok: bool
    coverage_drift_ok: bool
    coverage_drift_findings: list[str]
    contradiction: bool
    final_ready: bool
    reasons: list[str]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _parse_iso(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts.strip():
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _normalize_manifest_checks(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for item in checks:
        if not isinstance(item, dict):
            continue
        check_id = item.get("id")
        if not isinstance(check_id, str) or not check_id.strip():
            continue

        raw_field_coverage = item.get("field_coverage")
        field_coverage: list[str] = []
        if isinstance(raw_field_coverage, list):
            field_coverage = sorted(
                {str(path) for path in raw_field_coverage if isinstance(path, str) and path.strip()}
            )

        raw_effective_policy = item.get("effective_policy")
        effective_policy: dict[str, Any] = {}
        if isinstance(raw_effective_policy, dict):
            for key in sorted(raw_effective_policy.keys()):
                value = raw_effective_policy.get(key)
                if isinstance(key, str) and isinstance(value, (int, float, bool, str)):
                    effective_policy[key] = value

        normalized[check_id] = {
            "required": bool(item.get("required", False)),
            "guard_class": str(item.get("guard_class", "")),
            "field_coverage": field_coverage,
            "effective_policy": effective_policy,
        }

    return normalized


def _compare_coverage_manifests(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    current_checks = _normalize_manifest_checks(current)
    baseline_checks = _normalize_manifest_checks(baseline)

    findings: list[str] = []

    current_ids = set(current_checks.keys())
    baseline_ids = set(baseline_checks.keys())

    for check_id in sorted(baseline_ids - current_ids):
        findings.append(f"missing_check:{check_id}")
    for check_id in sorted(current_ids - baseline_ids):
        findings.append(f"unexpected_check:{check_id}")

    for check_id in sorted(current_ids & baseline_ids):
        current_item = current_checks[check_id]
        baseline_item = baseline_checks[check_id]

        if bool(current_item.get("required")) != bool(baseline_item.get("required")):
            findings.append(f"required_flag_changed:{check_id}")

        if str(current_item.get("guard_class", "")) != str(baseline_item.get("guard_class", "")):
            findings.append(f"guard_class_changed:{check_id}")

        if list(current_item.get("field_coverage", [])) != list(baseline_item.get("field_coverage", [])):
            findings.append(f"field_coverage_changed:{check_id}")

        current_policy = current_item.get("effective_policy", {})
        baseline_policy = baseline_item.get("effective_policy", {})
        if current_policy != baseline_policy:
            findings.append(f"effective_policy_changed:{check_id}")

    return findings


def consolidate(
    release_report_path: str = "audit/release_report.json",
    runtime_report_path: str = "audit/runtime_spec_report.json",
    output_path: str = "audit/consolidated_release_status.json",
    coverage_manifest_path: str | None = None,
    coverage_baseline_path: str | None = None,
) -> dict[str, Any]:
    release_path = Path(release_report_path)
    runtime_path = Path(runtime_report_path)

    release = _load_json(release_path)
    runtime = _load_json(runtime_path)

    manifest_requested = bool(coverage_manifest_path)
    baseline_requested = bool(coverage_baseline_path)

    coverage_manifest = _load_json(Path(coverage_manifest_path or "")) if manifest_requested else {}
    coverage_baseline = _load_json(Path(coverage_baseline_path or "")) if baseline_requested else {}

    coverage_manifest_ok = not manifest_requested or (
        bool(coverage_manifest) and isinstance(coverage_manifest.get("checks"), list)
    )
    coverage_baseline_ok = not baseline_requested or (
        bool(coverage_baseline) and isinstance(coverage_baseline.get("checks"), list)
    )

    coverage_drift_findings: list[str] = []
    coverage_drift_ok = True
    if manifest_requested and not coverage_manifest_ok:
        coverage_drift_ok = False
    if baseline_requested and not coverage_baseline_ok:
        coverage_drift_ok = False
    if coverage_manifest_ok and coverage_baseline_ok and manifest_requested and baseline_requested:
        coverage_drift_findings = _compare_coverage_manifests(coverage_manifest, coverage_baseline)
        if coverage_drift_findings:
            coverage_drift_ok = False

    release_ready = bool(release.get("release_ready", False))
    runtime_ok = bool(runtime.get("compliance_ok", False))

    required_passed = int(runtime.get("required_passed", 0) or 0)
    required_total = int(runtime.get("required_total", 0) or 0)

    bridge_import_status_present = False
    bridge_import_status_passed: bool | None = None
    bridge_import_status_evidence = ""
    runtime_checks = runtime.get("checks")
    if isinstance(runtime_checks, list):
        bridge_check = next(
            (
                item
                for item in runtime_checks
                if isinstance(item, dict) and str(item.get("id", "")) == "bridge_import_status_runtime"
            ),
            None,
        )
        if isinstance(bridge_check, dict):
            bridge_import_status_present = True
            if isinstance(bridge_check.get("passed"), bool):
                bridge_import_status_passed = bool(bridge_check.get("passed"))
            bridge_import_status_evidence = str(bridge_check.get("evidence", "") or "")

    release_ts_raw = release.get("timestamp")
    runtime_ts_raw = runtime.get("timestamp")
    release_ts = _parse_iso(release_ts_raw)
    runtime_ts = _parse_iso(runtime_ts_raw)

    if release_ts and runtime_ts:
        latest_source = "runtime" if runtime_ts >= release_ts else "release"
    elif runtime_ts:
        latest_source = "runtime"
    elif release_ts:
        latest_source = "release"
    else:
        latest_source = "unknown"

    contradiction = release_ready != runtime_ok
    reasons: list[str] = []

    if not release:
        reasons.append("release_report_missing_or_invalid")
    if not runtime:
        reasons.append("runtime_report_missing_or_invalid")
    if release and not release_ready:
        reasons.append("release_not_ready")
    if runtime and not runtime_ok:
        reasons.append("runtime_compliance_failed")
    if runtime and required_total > 0 and required_passed < required_total:
        reasons.append(f"runtime_required_failed:{required_passed}/{required_total}")
    if bridge_import_status_present and bridge_import_status_passed is False:
        reasons.append("bridge_import_status_runtime_failed_non_blocking")
    if manifest_requested and not coverage_manifest_ok:
        reasons.append("coverage_manifest_missing_or_invalid")
    if baseline_requested and not coverage_baseline_ok:
        reasons.append("coverage_baseline_missing_or_invalid")
    if coverage_drift_findings:
        reasons.append(f"guard_coverage_drift_detected:{len(coverage_drift_findings)}")
    if contradiction:
        reasons.append("release_runtime_contradiction")

    final_ready = release_ready and runtime_ok and bool(release) and bool(runtime) and coverage_drift_ok

    payload = ConsolidatedStatus(
        timestamp=datetime.now().isoformat(),
        release_report_path=str(release_path),
        runtime_report_path=str(runtime_path),
        release_timestamp=release_ts_raw if isinstance(release_ts_raw, str) else None,
        runtime_timestamp=runtime_ts_raw if isinstance(runtime_ts_raw, str) else None,
        latest_source=latest_source,
        release_ready=release_ready,
        runtime_compliance_ok=runtime_ok,
        required_passed=required_passed,
        required_total=required_total,
        bridge_import_status_present=bridge_import_status_present,
        bridge_import_status_passed=bridge_import_status_passed,
        bridge_import_status_evidence=bridge_import_status_evidence,
        coverage_manifest_path=coverage_manifest_path,
        coverage_baseline_path=coverage_baseline_path,
        coverage_manifest_ok=coverage_manifest_ok,
        coverage_baseline_ok=coverage_baseline_ok,
        coverage_drift_ok=coverage_drift_ok,
        coverage_drift_findings=coverage_drift_findings,
        contradiction=contradiction,
        final_ready=final_ready,
        reasons=reasons,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return asdict(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consolidate release + runtime audit status")
    parser.add_argument("--release-report", default="audit/release_report.json")
    parser.add_argument("--runtime-report", default="audit/runtime_spec_report.json")
    parser.add_argument("--coverage-manifest", default="")
    parser.add_argument("--coverage-baseline", default="")
    parser.add_argument("--output", default="audit/consolidated_release_status.json")
    args = parser.parse_args(argv)

    report = consolidate(
        release_report_path=args.release_report,
        runtime_report_path=args.runtime_report,
        coverage_manifest_path=args.coverage_manifest or None,
        coverage_baseline_path=args.coverage_baseline or None,
        output_path=args.output,
    )

    print(
        "Consolidated status: final_ready={} | release_ready={} | runtime_compliance_ok={} | coverage_drift_ok={}".format(
            report.get("final_ready"),
            report.get("release_ready"),
            report.get("runtime_compliance_ok"),
            report.get("coverage_drift_ok"),
        )
    )
    if report.get("reasons"):
        print("Reasons:")
        for r in report["reasons"]:
            print(f"- {r}")

    return 0 if report.get("final_ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
