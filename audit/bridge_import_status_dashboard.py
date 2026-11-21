"""Build a compact bridge-import status dashboard summary from consolidated gate status."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OSError(f"{path}: could not read consolidated report ({exc})") from exc
    except UnicodeError as exc:
        raise ValueError(f"{path}: consolidated report is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path}: consolidated report must be a non-empty JSON object")
    return payload


def build_bridge_import_status_summary(consolidated: dict[str, Any]) -> dict[str, Any]:
    reasons = consolidated.get("reasons")
    reason_items = [x for x in reasons if isinstance(x, str)] if isinstance(reasons, list) else []
    reason_flagged = "bridge_import_status_runtime_failed_non_blocking" in reason_items

    bridge_present = consolidated.get("bridge_import_status_present") is True
    bridge_passed_raw = consolidated.get("bridge_import_status_passed")
    bridge_passed = bridge_passed_raw if isinstance(bridge_passed_raw, bool) else None
    evidence_raw = consolidated.get("bridge_import_status_evidence")
    evidence = evidence_raw if isinstance(evidence_raw, str) else ""

    if not bridge_present:
        severity = "unknown"
    elif bridge_passed is True:
        severity = "ok"
    else:
        severity = "warning"

    return {
        "bridge_import_status_present": bridge_present,
        "bridge_import_status_passed": bridge_passed,
        "bridge_import_status_evidence": evidence,
        "reason_flagged": reason_flagged,
        "final_ready": consolidated.get("final_ready") is True,
        "severity": severity,
    }


def format_one_line(summary: dict[str, Any]) -> str:
    status = summary.get("bridge_import_status_passed")
    if status is True:
        status_text = "passed"
    elif status is False:
        status_text = "failed"
    else:
        status_text = "unknown"

    evidence_raw = summary.get("bridge_import_status_evidence")
    evidence = evidence_raw.strip() if isinstance(evidence_raw, str) else ""
    evidence_text = evidence if evidence else "-"
    severity = summary.get("severity")
    severity_text = severity if severity in ("unknown", "ok", "warning") else "unknown"

    return (
        "BRIDGE_IMPORT_STATUS "
        f"severity={severity_text} "
        f"present={summary.get('bridge_import_status_present') is True} "
        f"status={status_text} "
        f"reason_flagged={summary.get('reason_flagged') is True} "
        f"final_ready={summary.get('final_ready') is True} "
        f"evidence={json.dumps(evidence_text, ensure_ascii=False)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build bridge import status summary from consolidated gate report")
    parser.add_argument(
        "--consolidated",
        default="audit/consolidated_release_status_voice_first_runtime.json",
        help="Path to consolidated_release_status JSON",
    )
    parser.add_argument(
        "--output",
        default="audit/bridge_import_status_summary_voice_first_runtime.json",
        help="Path to write compact bridge-import status summary JSON",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return exit code 1 unless bridge import status is explicitly passed.",
    )
    args = parser.parse_args(argv)

    consolidated_path = Path(args.consolidated)
    out = Path(args.output)
    try:
        if consolidated_path.resolve() == out.resolve():
            raise ValueError("input and output resolve to the same file")
        consolidated = _load_json(consolidated_path)
    except (OSError, RuntimeError, ValueError) as exc:
        evidence = json.dumps(str(exc))
        print(
            "BRIDGE_IMPORT_STATUS severity=unknown present=False status=unknown "
            f"reason_flagged=False final_ready=False evidence={evidence}",
            file=sys.stderr,
        )
        return 1

    summary = build_bridge_import_status_summary(consolidated)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        print(f"BRIDGE_IMPORT_STATUS output_error={json.dumps(str(exc))}", file=sys.stderr)
        return 1

    print(format_one_line(summary))

    if args.fail_on_warning and summary["severity"] != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
