"""Run the real-audio Golden-Set gate for DefectScanner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core.real_audio_defect_golden_gate import (
    run_real_audio_defect_golden_gate,  # pylint: disable=wrong-import-position
)


def main() -> int:
    """CLI entry point for the real-audio DefectScanner Golden-Set gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="audit/real_audio_defect_golden_manifest.json")
    parser.add_argument("--output", default="audit/real_audio_defect_golden_report.json")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--no-fail", action="store_true", help="Write report but always return exit code 0")
    args = parser.parse_args()

    report = run_real_audio_defect_golden_gate(
        manifest_path=REPO_ROOT / args.manifest,
        repo_root=REPO_ROOT,
        allow_missing=bool(args.allow_missing),
        allow_empty=bool(args.allow_empty),
    )
    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    gate = report.gate
    print(
        "real_audio_defect_golden_gate "
        f"passed={gate.passed} recall={gate.recall:.3f} precision={gate.precision:.3f} "
        f"locality={gate.locality_recall:.3f} scanned={report.scanned_cases} "
        f"skipped={len(report.skipped_cases)} output={args.output}"
    )
    if gate.fail_reasons:
        print("fail_reasons=" + ",".join(gate.fail_reasons))
    return 0 if args.no_fail or gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
