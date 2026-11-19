"""Run the real-audio Golden-Set gate for executed strategy and export safety."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core.real_audio_execution_golden_gate import (
    run_real_audio_execution_golden_gate,  # pylint: disable=wrong-import-position
)


def main() -> int:
    """CLI entry point for the real-audio Execution/Export Golden-Set gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="audit/real_audio_strategy_golden_manifest.json")
    parser.add_argument("--output", default="audit/real_audio_execution_golden_report.json")
    parser.add_argument("--export-dir", default="test_output/real_audio_execution_golden_gate")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--no-fail", action="store_true", help="Write report but always return exit code 0")
    args = parser.parse_args()

    report = run_real_audio_execution_golden_gate(
        manifest_path=REPO_ROOT / args.manifest,
        repo_root=REPO_ROOT,
        output_dir=REPO_ROOT / args.export_dir,
        allow_missing=bool(args.allow_missing),
        allow_empty=bool(args.allow_empty),
        max_cases=args.max_cases,
    )
    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    gate = report.gate
    print(
        "real_audio_execution_golden_gate "
        f"passed={gate.passed} phase_recall={gate.phase_execution_recall:.3f} "
        f"phase_delta={gate.phase_delta_coverage:.3f} artifact={gate.artifact_contract_rate:.3f} "
        f"hpi={gate.hpi_contract_rate:.3f} vocal={gate.vocal_contract_rate:.3f} "
        f"export={gate.export_contract_rate:.3f} forbidden={gate.forbidden_phase_executions} "
        f"runtime_factor={gate.runtime_factor:.3f} scanned={report.scanned_cases} "
        f"skipped={len(report.skipped_cases)} output={args.output}"
    )
    if gate.fail_reasons:
        print("fail_reasons=" + ",".join(gate.fail_reasons))
    return 0 if args.no_fail or gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
