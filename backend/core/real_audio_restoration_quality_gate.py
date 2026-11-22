"""Real-audio Golden-Set gate for final restoration quality.

This gate consumes the execution/export Golden-Set report and evaluates the
layer that matters for a world-class claim: not whether unsafe exports are
blocked, but whether Aurik produces non-degraded, musically valid, vocal-safe,
artifact-free final exports on real audio. Every case must carry explicit
contract results and complete evidence; missing evidence is never a pass. Case
IDs must match the manifest exactly, and any execution failure reason blocks
the case, including failure codes not yet classified by this gate.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend.core.real_audio_defect_golden_gate import _load_manifest

_MUSICAL_GOALS = "MUSICAL_GOALS_VIOLATION"
_NOISE_TEXTURE = "NOISE_TEXTURE_INCOHERENT"
_GOOSEBUMPS = "GOOSEBUMPS_LOW"
_VQI = "VQI_BELOW_THRESHOLD"
_HPI_FAILURE_CODES = {"HPI_EXCEPTION", "HPI_FAIL_FINAL"}
_VOCAL_FAILURE_CODES = {_VQI, "VQI_BELOW_MAXIMUM_ALIGNMENT", "SINGER_ID_BELOW_THRESHOLD", "SINGMOS_BELOW_2_5"}
_KNOWN_FAILURE_CODES = {
    _MUSICAL_GOALS,
    _NOISE_TEXTURE,
    _GOOSEBUMPS,
    *_HPI_FAILURE_CODES,
    *_VOCAL_FAILURE_CODES,
    "FINAL_EXPORT_AUDIO_GATE_FAIL",
    "STEREO_SAFETY_HARD_FAIL",
}


@dataclass(frozen=True)
class RestorationQualityThresholds:
    """Thresholds for the final real-audio restoration quality gate."""

    min_non_degraded_export_rate: float = 0.85
    min_unblocked_export_rate: float = 0.90
    min_musical_goal_case_pass_rate: float = 0.90
    min_noise_texture_case_pass_rate: float = 0.94
    min_goosebumps_case_pass_rate: float = 0.90
    min_final_quality_case_pass_rate: float = 0.85
    min_vocal_floor_pass_rate: float = 1.0
    min_hpi_average: float = 0.78
    min_quality_estimate_average: float | None = None  # Legacy manifest field; report-only.
    max_runtime_factor: float = 8.0
    min_real_audio_cases: int = 80
    min_vocal_cases: int = 30
    min_external_benchmark_cases: int = 20


@dataclass(frozen=True)
class RestorationQualityCaseResult:
    """Serialisierbares Endqualitäts-Bewertungsergebnis pro Fall."""

    case_id: str
    non_degraded_export: bool
    unblocked_export: bool
    artifact_contract_passed: bool
    hpi_contract_passed: bool
    vocal_contract_passed: bool
    export_contract_passed: bool
    evidence_complete: bool
    missing_evidence: tuple[str, ...]
    musical_goals_passed: bool
    noise_texture_passed: bool
    goosebumps_passed: bool
    vocal_floor_passed: bool
    final_quality_passed: bool
    vocal_required: bool
    hpi: float | None
    quality_estimate: float | None
    vqi: float | None
    vqi_floor: float | None
    fail_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RestorationQualityGateResult:
    """Aggregate final-quality gate result."""

    passed: bool
    non_degraded_export_rate: float
    unblocked_export_rate: float
    artifact_contract_rate: float
    hpi_contract_rate: float
    vocal_contract_rate: float
    export_contract_rate: float
    case_evidence_complete_rate: float
    case_id_coverage_complete: bool
    case_id_coverage_rate: float
    expected_case_count: int
    missing_case_ids: tuple[str, ...]
    unexpected_case_ids: tuple[str, ...]
    duplicate_case_ids: tuple[str, ...]
    musical_goal_case_pass_rate: float
    noise_texture_case_pass_rate: float
    goosebumps_case_pass_rate: float
    final_quality_case_pass_rate: float
    vocal_floor_pass_rate: float
    hpi_average: float | None
    quality_estimate_average: float | None
    runtime_factor: float
    real_audio_cases: int
    vocal_cases: int
    external_benchmark_cases: int
    fail_reasons: tuple[str, ...]
    prioritized_actions: tuple[str, ...]


@dataclass(frozen=True)
class RealAudioRestorationQualityGateReport:
    """Serialisierbares Ergebnis des Real-Audio-Endqualitäts-Gates."""

    gate: RestorationQualityGateResult
    cases: list[RestorationQualityCaseResult]
    execution_report_path: str
    manifest_path: str | None
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Gibt a JSON-serializable representation zurück."""
        return {
            "gate": asdict(self.gate),
            "cases": [asdict(case) for case in self.cases],
            "execution_report_path": self.execution_report_path,
            "manifest_path": self.manifest_path,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _thresholds_from_manifest(payload: dict[str, Any] | None) -> RestorationQualityThresholds:
    raw = payload.get("restoration_quality_thresholds") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        raw = {}
    return RestorationQualityThresholds(
        min_non_degraded_export_rate=float(raw.get("min_non_degraded_export_rate", 0.85)),
        min_unblocked_export_rate=float(raw.get("min_unblocked_export_rate", 0.90)),
        min_musical_goal_case_pass_rate=float(raw.get("min_musical_goal_case_pass_rate", 0.90)),
        min_noise_texture_case_pass_rate=float(raw.get("min_noise_texture_case_pass_rate", 0.94)),
        min_goosebumps_case_pass_rate=float(raw.get("min_goosebumps_case_pass_rate", 0.90)),
        min_final_quality_case_pass_rate=float(raw.get("min_final_quality_case_pass_rate", 0.85)),
        min_vocal_floor_pass_rate=float(raw.get("min_vocal_floor_pass_rate", 1.0)),
        min_hpi_average=float(raw.get("min_hpi_average", 0.78)),
        min_quality_estimate_average=(
            float(raw["min_quality_estimate_average"]) if raw.get("min_quality_estimate_average") is not None else None
        ),
        max_runtime_factor=float(raw.get("max_runtime_factor", 8.0)),
        min_real_audio_cases=int(raw.get("min_real_audio_cases", 80)),
        min_vocal_cases=int(raw.get("min_vocal_cases", 30)),
        min_external_benchmark_cases=int(raw.get("min_external_benchmark_cases", 20)),
    )


def _expected_case_ids(payload: dict[str, Any] | None) -> tuple[str, ...] | None:
    raw_cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(raw_cases, list) or not raw_cases:
        return None
    case_ids: list[str] = []
    for case in raw_cases:
        case_id = case.get("case_id") if isinstance(case, dict) else None
        if not isinstance(case_id, str) or not case_id.strip():
            return None
        case_ids.append(case_id.strip())
    return tuple(case_ids) if len(case_ids) == len(set(case_ids)) else None


def _as_fail_codes(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(item.strip() for item in raw if isinstance(item, str) and item.strip())


def _optional_float(raw: object) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _required_actions(
    case: dict[str, Any],
    fail_codes: tuple[str, ...],
    missing_evidence: tuple[str, ...],
) -> tuple[str, ...]:
    actions: list[str] = []
    degradation = str(case.get("degradation_status", "") or "").strip().lower()
    export_blocked = bool(case.get("export_blocked", False)) or str(case.get("export_strategy", "") or "") == "blocked"
    if missing_evidence:
        actions.append("complete_quality_evidence")
    if degradation not in {"", "ok"} or export_blocked:
        actions.append("productive_non_degraded_export_recovery")
    if case.get("artifact_contract_passed") is not True:
        actions.append("artifact_freedom_recovery")
    if "FINAL_EXPORT_AUDIO_GATE_FAIL" in fail_codes:
        actions.append("productive_non_degraded_export_recovery")
    if _HPI_FAILURE_CODES.intersection(fail_codes):
        actions.append("hpi_quality_candidate_ranking")
    if _VOCAL_FAILURE_CODES.intersection(fail_codes):
        actions.append("vocal_vqi_recovery")
    if "STEREO_SAFETY_HARD_FAIL" in fail_codes:
        actions.append("stereo_safety_recovery")
    if set(fail_codes) - _KNOWN_FAILURE_CODES:
        actions.append("triage_quality_failure_codes")
    if _MUSICAL_GOALS in fail_codes:
        actions.append("goal_directed_candidate_recovery")
    if _NOISE_TEXTURE in fail_codes:
        actions.append("noise_texture_repair")
    if _GOOSEBUMPS in fail_codes:
        actions.append("frisson_goosebumps_protection")
    if _VQI in fail_codes:
        actions.append("vocal_vqi_recovery")
    if _optional_float(case.get("hpi")) is not None and float(case.get("hpi")) < 0.72:  # type: ignore[arg-type]
        actions.append("hpi_quality_candidate_ranking")
    return tuple(dict.fromkeys(actions))


def _case_from_execution(case: dict[str, Any]) -> RestorationQualityCaseResult:
    missing: list[str] = []

    def read_bool(key: str) -> bool:
        value = case.get(key)
        if not isinstance(value, bool):
            missing.append(key)
            return False
        return value

    case_id_value = case.get("case_id")
    case_id = case_id_value.strip() if isinstance(case_id_value, str) else ""
    if not case_id:
        missing.append("case_id")

    raw_fail_reasons = case.get("fail_reasons")
    fail_reasons_complete = isinstance(raw_fail_reasons, (list, tuple)) and all(
        isinstance(item, str) and item.strip() for item in raw_fail_reasons
    )
    if not fail_reasons_complete:
        missing.append("fail_reasons")
    fail_codes = _as_fail_codes(raw_fail_reasons)

    degradation_value = case.get("degradation_status")
    degradation = degradation_value.strip().lower() if isinstance(degradation_value, str) else ""
    if not degradation:
        missing.append("degradation_status")
    export_strategy_value = case.get("export_strategy")
    export_strategy = export_strategy_value.strip() if isinstance(export_strategy_value, str) else ""
    if not export_strategy:
        missing.append("export_strategy")

    export_blocked = read_bool("export_blocked")
    artifact_contract_passed = read_bool("artifact_contract_passed")
    hpi_contract_passed = read_bool("hpi_contract_passed")
    vocal_required = read_bool("vocal_required")
    vocal_contract_passed = read_bool("vocal_contract_passed")
    export_contract_passed = read_bool("export_contract_passed")

    artifact_freedom = _optional_float(case.get("artifact_freedom"))
    if artifact_freedom is None:
        missing.append("artifact_freedom")
    hpi = _optional_float(case.get("hpi"))
    if hpi is None:
        missing.append("hpi")
    metadata_value = case.get("metadata")
    if not isinstance(metadata_value, dict):
        missing.append("metadata")
    metadata = metadata_value if isinstance(metadata_value, dict) else {}
    quality_estimate = _optional_float(metadata.get("quality_estimate"))
    vqi = _optional_float(case.get("vqi"))
    vqi_floor = _optional_float(metadata.get("vqi_floor"))
    if vocal_required and vqi is None:
        missing.append("vqi")
    if vocal_required and vqi_floor is None:
        missing.append("metadata.vqi_floor")

    missing_evidence = tuple(dict.fromkeys(missing))
    evidence_complete = not missing_evidence
    non_degraded = degradation == "ok" and export_contract_passed
    unblocked = not export_blocked and export_strategy != "blocked" and export_contract_passed
    musical_ok = fail_reasons_complete and _MUSICAL_GOALS not in fail_codes
    noise_ok = fail_reasons_complete and _NOISE_TEXTURE not in fail_codes
    goosebumps_ok = fail_reasons_complete and _GOOSEBUMPS not in fail_codes
    vocal_ok = vocal_contract_passed and not _VOCAL_FAILURE_CODES.intersection(fail_codes)
    if vocal_required:
        vocal_ok = vocal_ok and vqi is not None and vqi_floor is not None and vqi >= vqi_floor
    final_quality_passed = all(
        (
            evidence_complete,
            non_degraded,
            unblocked,
            artifact_contract_passed,
            hpi_contract_passed,
            not fail_codes,
            musical_ok,
            noise_ok,
            goosebumps_ok,
            vocal_ok,
        )
    )

    return RestorationQualityCaseResult(
        case_id=case_id or "unknown",
        non_degraded_export=non_degraded,
        unblocked_export=unblocked,
        artifact_contract_passed=artifact_contract_passed,
        hpi_contract_passed=hpi_contract_passed,
        vocal_contract_passed=vocal_contract_passed,
        export_contract_passed=export_contract_passed,
        evidence_complete=evidence_complete,
        missing_evidence=missing_evidence,
        musical_goals_passed=musical_ok,
        noise_texture_passed=noise_ok,
        goosebumps_passed=goosebumps_ok,
        vocal_floor_passed=vocal_ok,
        final_quality_passed=final_quality_passed,
        vocal_required=vocal_required,
        hpi=hpi,
        quality_estimate=quality_estimate,
        vqi=vqi,
        vqi_floor=vqi_floor,
        fail_reasons=fail_codes,
        required_actions=_required_actions(case, fail_codes, missing_evidence),
        metadata={
            "path": metadata.get("path"),  # type: ignore[union-attr]
            "material": metadata.get("material"),  # type: ignore[union-attr]
            "degradation_status": degradation,
            "export_strategy": export_strategy,
            "vocal_required": vocal_required,
            "artifact_freedom": artifact_freedom,
        },
    )


def _rate(cases: list[RestorationQualityCaseResult], predicate: str) -> float:
    total = max(len(cases), 1)
    return sum(1 for case in cases if bool(getattr(case, predicate))) / total


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _ordered_action_summary(
    cases: list[RestorationQualityCaseResult],
    gate_level_actions: list[str],
) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for action in gate_level_actions:
        counts[action] = counts.get(action, 0) + max(len(cases), 1)
    for case in cases:
        for action in case.required_actions:
            counts[action] = counts.get(action, 0) + 1
    priority = [
        "create_real_audio_restoration_quality_gate",
        "complete_quality_evidence",
        "goal_directed_candidate_recovery",
        "artifact_freedom_recovery",
        "stereo_safety_recovery",
        "noise_texture_repair",
        "frisson_goosebumps_protection",
        "vocal_vqi_recovery",
        "triage_quality_failure_codes",
        "phase_minimalism_runtime_budget",
        "hpi_quality_candidate_ranking",
        "expand_real_audio_golden_set",
        "add_external_rx_top_tool_benchmark",
        "productive_non_degraded_export_recovery",
    ]
    ordered = [action for action in priority if counts.get(action, 0) > 0]
    ordered.extend(sorted(action for action in counts if action not in set(ordered)))
    return tuple(ordered)


def evaluate_restoration_quality_gate(
    execution_report: dict[str, Any],
    thresholds: RestorationQualityThresholds,
    *,
    external_benchmark_cases: int = 0,
    expected_case_ids: tuple[str, ...] | None = None,
) -> tuple[RestorationQualityGateResult, list[RestorationQualityCaseResult]]:
    """Bewertet final restoration quality from an execution Golden-Set report."""
    _cases_raw = execution_report.get("cases")
    raw_cases: list[Any] = _cases_raw if isinstance(_cases_raw, list) else []
    cases = [_case_from_execution(case) for case in raw_cases if isinstance(case, dict)]
    actual_case_ids = [case.case_id for case in cases]
    actual_case_id_set = set(actual_case_ids)
    duplicates = tuple(sorted(case_id for case_id, count in Counter(actual_case_ids).items() if count > 1))
    expected_case_id_set = set(expected_case_ids or ())
    missing_case_ids = tuple(sorted(expected_case_id_set - actual_case_id_set))
    unexpected_case_ids = tuple(sorted(actual_case_id_set - expected_case_id_set))
    case_id_coverage_rate = (
        len(expected_case_id_set - set(missing_case_ids)) / len(expected_case_id_set) if expected_case_id_set else 0.0
    )
    case_id_coverage_complete = (
        expected_case_ids is not None
        and len(expected_case_ids) == len(expected_case_id_set)
        and bool(expected_case_id_set)
        and not missing_case_ids
        and not unexpected_case_ids
        and not duplicates
    )
    vocal_cases = [case for case in cases if case.vocal_required]
    hpi_values = [case.hpi for case in cases if isinstance(case.hpi, (int, float))]
    quality_values = [case.quality_estimate for case in cases if isinstance(case.quality_estimate, (int, float))]
    raw_execution_gate = execution_report.get("gate")
    execution_gate = raw_execution_gate if isinstance(raw_execution_gate, dict) else {}
    runtime_value = _optional_float(execution_gate.get("runtime_factor"))
    runtime_evidence_complete = (
        isinstance(raw_execution_gate, dict) and runtime_value is not None and runtime_value > 0.0
    )
    runtime_factor = runtime_value if runtime_value is not None else 0.0

    non_degraded_rate = _rate(cases, "non_degraded_export")
    unblocked_rate = _rate(cases, "unblocked_export")
    artifact_contract_rate = _rate(cases, "artifact_contract_passed")
    hpi_contract_rate = _rate(cases, "hpi_contract_passed")
    export_contract_rate = _rate(cases, "export_contract_passed")
    case_evidence_complete_rate = _rate(cases, "evidence_complete")
    musical_rate = _rate(cases, "musical_goals_passed")
    noise_rate = _rate(cases, "noise_texture_passed")
    goosebumps_rate = _rate(cases, "goosebumps_passed")
    final_quality_rate = _rate(cases, "final_quality_passed")
    vocal_contract_rate = 1.0 if not vocal_cases else _rate(vocal_cases, "vocal_contract_passed")
    vocal_rate = 1.0 if not vocal_cases else _rate(vocal_cases, "vocal_floor_passed")
    hpi_avg = _mean([float(value) for value in hpi_values])
    quality_avg = _mean([float(value) for value in quality_values])

    fail_reasons: list[str] = []
    gate_actions = ["create_real_audio_restoration_quality_gate"]
    if not case_id_coverage_complete:
        fail_reasons.append(
            "case_id_coverage incomplete "
            f"(expected={len(expected_case_id_set)}, missing={len(missing_case_ids)}, "
            f"unexpected={len(unexpected_case_ids)}, duplicates={len(duplicates)})"
        )
        gate_actions.append("verify_golden_set_case_coverage")
    if not runtime_evidence_complete:
        fail_reasons.append("execution_gate runtime_factor missing, non-finite, or non-positive")
        gate_actions.append("complete_quality_evidence")
    if non_degraded_rate < thresholds.min_non_degraded_export_rate:
        fail_reasons.append(
            f"non_degraded_export_rate {non_degraded_rate:.3f} < {thresholds.min_non_degraded_export_rate:.3f}"
        )
    if unblocked_rate < thresholds.min_unblocked_export_rate:
        fail_reasons.append(f"unblocked_export_rate {unblocked_rate:.3f} < {thresholds.min_unblocked_export_rate:.3f}")
    if artifact_contract_rate < 1.0:
        fail_reasons.append(f"artifact_contract_rate {artifact_contract_rate:.3f} < 1.000")
        gate_actions.append("artifact_freedom_recovery")
    if hpi_contract_rate < 1.0:
        fail_reasons.append(f"hpi_contract_rate {hpi_contract_rate:.3f} < 1.000")
    if vocal_contract_rate < 1.0:
        fail_reasons.append(f"vocal_contract_rate {vocal_contract_rate:.3f} < 1.000")
    if export_contract_rate < 1.0:
        fail_reasons.append(f"export_contract_rate {export_contract_rate:.3f} < 1.000")
    if case_evidence_complete_rate < 1.0:
        fail_reasons.append(f"case_evidence_complete_rate {case_evidence_complete_rate:.3f} < 1.000")
        gate_actions.append("complete_quality_evidence")
    if musical_rate < thresholds.min_musical_goal_case_pass_rate:
        fail_reasons.append(
            f"musical_goal_case_pass_rate {musical_rate:.3f} < {thresholds.min_musical_goal_case_pass_rate:.3f}"
        )
    if noise_rate < thresholds.min_noise_texture_case_pass_rate:
        fail_reasons.append(
            f"noise_texture_case_pass_rate {noise_rate:.3f} < {thresholds.min_noise_texture_case_pass_rate:.3f}"
        )
    if goosebumps_rate < thresholds.min_goosebumps_case_pass_rate:
        fail_reasons.append(
            f"goosebumps_case_pass_rate {goosebumps_rate:.3f} < {thresholds.min_goosebumps_case_pass_rate:.3f}"
        )
    if final_quality_rate < thresholds.min_final_quality_case_pass_rate:
        fail_reasons.append(
            f"final_quality_case_pass_rate {final_quality_rate:.3f} < {thresholds.min_final_quality_case_pass_rate:.3f}"
        )
    if vocal_rate < thresholds.min_vocal_floor_pass_rate:
        fail_reasons.append(f"vocal_floor_pass_rate {vocal_rate:.3f} < {thresholds.min_vocal_floor_pass_rate:.3f}")
    if hpi_avg is None or hpi_avg < thresholds.min_hpi_average:
        fail_reasons.append(f"hpi_average {(hpi_avg or 0.0):.3f} < {thresholds.min_hpi_average:.3f}")
    if runtime_factor > thresholds.max_runtime_factor:
        fail_reasons.append(f"runtime_factor {runtime_factor:.3f} > {thresholds.max_runtime_factor:.3f}")
        gate_actions.append("phase_minimalism_runtime_budget")
    if len(cases) < thresholds.min_real_audio_cases:
        fail_reasons.append(f"real_audio_cases {len(cases)} < {thresholds.min_real_audio_cases}")
        gate_actions.append("expand_real_audio_golden_set")
    if len(vocal_cases) < thresholds.min_vocal_cases:
        fail_reasons.append(f"vocal_cases {len(vocal_cases)} < {thresholds.min_vocal_cases}")
        gate_actions.append("expand_real_audio_golden_set")
    if external_benchmark_cases < thresholds.min_external_benchmark_cases:
        fail_reasons.append(
            f"external_benchmark_cases {external_benchmark_cases} < {thresholds.min_external_benchmark_cases}"
        )
        gate_actions.append("add_external_rx_top_tool_benchmark")

    gate = RestorationQualityGateResult(
        passed=not fail_reasons,
        non_degraded_export_rate=float(non_degraded_rate),
        unblocked_export_rate=float(unblocked_rate),
        artifact_contract_rate=float(artifact_contract_rate),
        hpi_contract_rate=float(hpi_contract_rate),
        vocal_contract_rate=float(vocal_contract_rate),
        export_contract_rate=float(export_contract_rate),
        case_evidence_complete_rate=float(case_evidence_complete_rate),
        case_id_coverage_complete=case_id_coverage_complete,
        case_id_coverage_rate=float(case_id_coverage_rate),
        expected_case_count=len(expected_case_id_set),
        missing_case_ids=missing_case_ids,
        unexpected_case_ids=unexpected_case_ids,
        duplicate_case_ids=duplicates,
        musical_goal_case_pass_rate=float(musical_rate),
        noise_texture_case_pass_rate=float(noise_rate),
        goosebumps_case_pass_rate=float(goosebumps_rate),
        final_quality_case_pass_rate=float(final_quality_rate),
        vocal_floor_pass_rate=float(vocal_rate),
        hpi_average=hpi_avg,
        quality_estimate_average=quality_avg,
        runtime_factor=float(runtime_factor),
        real_audio_cases=int(len(cases)),
        vocal_cases=int(len(vocal_cases)),
        external_benchmark_cases=int(external_benchmark_cases),
        fail_reasons=tuple(fail_reasons),
        prioritized_actions=_ordered_action_summary(cases, gate_actions),
    )
    return gate, cases


def run_real_audio_restoration_quality_gate(
    *,
    execution_report_path: Path,
    manifest_path: Path | None = None,
    external_benchmark_cases: int = 0,
) -> RealAudioRestorationQualityGateReport:
    """Führt aus: the final restoration quality gate from a saved execution report."""
    start_time = time.time()
    execution_report_path = execution_report_path.resolve()
    execution_report = json.loads(execution_report_path.read_text(encoding="utf-8"))
    manifest_payload = _load_manifest(manifest_path.resolve()) if manifest_path is not None else None
    thresholds = _thresholds_from_manifest(manifest_payload)
    gate, cases = evaluate_restoration_quality_gate(
        execution_report,
        thresholds,
        external_benchmark_cases=external_benchmark_cases,
        expected_case_ids=_expected_case_ids(manifest_payload),
    )
    return RealAudioRestorationQualityGateReport(
        gate=gate,
        cases=cases,
        execution_report_path=str(execution_report_path),
        manifest_path=str(manifest_path.resolve()) if manifest_path is not None else None,
        elapsed_seconds=float(time.time() - start_time),
    )


__all__ = [
    "RealAudioRestorationQualityGateReport",
    "RestorationQualityCaseResult",
    "RestorationQualityGateResult",
    "RestorationQualityThresholds",
    "evaluate_restoration_quality_gate",
    "run_real_audio_restoration_quality_gate",
]
