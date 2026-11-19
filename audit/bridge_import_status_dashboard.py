"""Build a compact bridge-import status dashboard summary from consolidated gate status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_bridge_import_status_summary(consolidated: dict[str, Any]) -> dict[str, Any]:
    reasons = consolidated.get("reasons")
    reason_items = [str(x) for x in reasons] if isinstance(reasons, list) else []
    reason_flagged = "bridge_import_status_runtime_failed_non_blocking" in reason_items

    bridge_present = bool(consolidated.get("bridge_import_status_present", False))
    bridge_passed_raw = consolidated.get("bridge_import_status_passed")
    bridge_passed = bridge_passed_raw if isinstance(bridge_passed_raw, bool) else None
    evidence = str(consolidated.get("bridge_import_status_evidence", "") or "")

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
        "final_ready": bool(consolidated.get("final_ready", False)),
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

    evidence = str(summary.get("bridge_import_status_evidence", "") or "").strip()
    evidence_text = evidence if evidence else "-"

    return (
        "BRIDGE_IMPORT_STATUS "
        f"severity={summary.get('severity', 'unknown')} "
        f"present={bool(summary.get('bridge_import_status_present', False))} "
        f"status={status_text} "
        f"reason_flagged={bool(summary.get('reason_flagged', False))} "
        f"final_ready={bool(summary.get('final_ready', False))} "
        f'evidence="{evidence_text}"'
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
        help="Return exit code 1 when bridge import status is present and failed.",
    )
    args = parser.parse_args(argv)

    consolidated = _load_json(Path(args.consolidated))
    if not consolidated:
        print(
            'BRIDGE_IMPORT_STATUS severity=unknown present=False status=unknown reason_flagged=False final_ready=False evidence="consolidated_missing_or_invalid"'
        )
        return 1

    summary = build_bridge_import_status_summary(consolidated)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(format_one_line(summary))

    if (
        args.fail_on_warning
        and summary.get("bridge_import_status_present")
        and summary.get("bridge_import_status_passed") is False
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
