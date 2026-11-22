"""Run the real-audio Golden-Set gate for final restoration quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core.real_audio_restoration_quality_gate import (  # pylint: disable=wrong-import-position
    run_real_audio_restoration_quality_gate,
)


def main() -> int:
    """CLI entry point for the real-audio final restoration quality gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-report", default="audit/real_audio_execution_golden_report.json")
    parser.add_argument("--manifest", default="audit/real_audio_strategy_golden_manifest.json")
    parser.add_argument("--output", default="audit/real_audio_restoration_quality_report.json")
    parser.add_argument("--external-benchmark-cases", type=int, default=0)
    parser.add_argument("--no-fail", action="store_true", help="Write report but always return exit code 0")
    args = parser.parse_args()

    report = run_real_audio_restoration_quality_gate(
        execution_report_path=REPO_ROOT / args.execution_report,
        manifest_path=REPO_ROOT / args.manifest if args.manifest else None,
        external_benchmark_cases=int(args.external_benchmark_cases),
    )
    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    gate = report.gate
    print(
        "real_audio_restoration_quality_gate "
        f"passed={gate.passed} non_degraded={gate.non_degraded_export_rate:.3f} "
        f"unblocked={gate.unblocked_export_rate:.3f} artifact_contract={gate.artifact_contract_rate:.3f} "
        f"hpi_contract={gate.hpi_contract_rate:.3f} vocal_contract={gate.vocal_contract_rate:.3f} "
        f"export_contract={gate.export_contract_rate:.3f} evidence_complete={gate.case_evidence_complete_rate:.3f} "
        f"case_coverage={gate.case_id_coverage_rate:.3f}/{gate.expected_case_count} "
        f"musical={gate.musical_goal_case_pass_rate:.3f} "
        f"noise_texture={gate.noise_texture_case_pass_rate:.3f} goosebumps={gate.goosebumps_case_pass_rate:.3f} "
        f"vocal={gate.vocal_floor_pass_rate:.3f} hpi_avg={(gate.hpi_average or 0.0):.3f} "
        f"quality_avg_report_only={(gate.quality_estimate_average or 0.0):.3f} "
        f"runtime_factor={gate.runtime_factor:.3f} "
        f"cases={gate.real_audio_cases} vocal_cases={gate.vocal_cases} "
        f"external_benchmark_cases={gate.external_benchmark_cases} output={args.output}"
    )
    if gate.fail_reasons:
        print("fail_reasons=" + ";".join(gate.fail_reasons))
    if gate.prioritized_actions:
        print("prioritized_actions=" + ",".join(gate.prioritized_actions))
    return 0 if args.no_fail or gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
