"""Real-audio Golden-Set gate for final restoration quality.

This gate consumes the execution/export Golden-Set report and evaluates the
layer that matters for a world-class claim: not whether unsafe exports are
blocked, but whether Aurik produces non-degraded, musically valid, vocal-safe,
artifact-free final exports on real audio.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend.core.real_audio_defect_golden_gate import _load_manifest

_MUSICAL_GOALS = "MUSICAL_GOALS_VIOLATION"
_NOISE_TEXTURE = "NOISE_TEXTURE_INCOHERENT"
_GOOSEBUMPS = "GOOSEBUMPS_LOW"
_VQI = "VQI_BELOW_THRESHOLD"


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
    min_quality_estimate_average: float = 0.84
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
    musical_goals_passed: bool
    noise_texture_passed: bool
    goosebumps_passed: bool
    vocal_floor_passed: bool
    final_quality_passed: bool
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
        min_quality_estimate_average=float(raw.get("min_quality_estimate_average", 0.84)),
        max_runtime_factor=float(raw.get("max_runtime_factor", 8.0)),
        min_real_audio_cases=int(raw.get("min_real_audio_cases", 80)),
        min_vocal_cases=int(raw.get("min_vocal_cases", 30)),
        min_external_benchmark_cases=int(raw.get("min_external_benchmark_cases", 20)),
    )


def _as_fail_codes(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _optional_float(raw: object) -> float | None:
    return float(raw) if isinstance(raw, (int, float)) else None


def _required_actions(case: dict[str, Any], fail_codes: tuple[str, ...]) -> tuple[str, ...]:
    actions: list[str] = []
    degradation = str(case.get("degradation_status", "") or "").strip().lower()
    export_blocked = bool(case.get("export_blocked", False)) or str(case.get("export_strategy", "") or "") == "blocked"
    if degradation not in {"", "ok"} or export_blocked:
        actions.append("productive_non_degraded_export_recovery")
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
    fail_codes = _as_fail_codes(case.get("fail_reasons"))
    degradation = str(case.get("degradation_status", "") or "").strip().lower()
    export_strategy = str(case.get("export_strategy", "") or "")
    non_degraded = degradation in {"", "ok"}
    unblocked = not bool(case.get("export_blocked", False)) and export_strategy != "blocked"
    musical_ok = _MUSICAL_GOALS not in fail_codes
    noise_ok = _NOISE_TEXTURE not in fail_codes
    goosebumps_ok = _GOOSEBUMPS not in fail_codes
    vocal_required = bool(case.get("vocal_required", False))
    vocal_ok = (not vocal_required) or _VQI not in fail_codes
    hpi = _optional_float(case.get("hpi"))
    metadata = case.get("metadata") if isinstance(case.get("metadata"), dict) else {}
    quality_estimate = _optional_float(metadata.get("quality_estimate"))  # type: ignore[union-attr]
    vqi = _optional_float(case.get("vqi"))
    vqi_floor = _optional_float(metadata.get("vqi_floor"))  # type: ignore[union-attr]
    final_quality_passed = bool(non_degraded and unblocked and musical_ok and noise_ok and goosebumps_ok and vocal_ok)

    return RestorationQualityCaseResult(
        case_id=str(case.get("case_id", "unknown") or "unknown"),
        non_degraded_export=non_degraded,
        unblocked_export=unblocked,
        musical_goals_passed=musical_ok,
        noise_texture_passed=noise_ok,
        goosebumps_passed=goosebumps_ok,
        vocal_floor_passed=vocal_ok,
        final_quality_passed=final_quality_passed,
        hpi=hpi,
        quality_estimate=quality_estimate,
        vqi=vqi,
        vqi_floor=vqi_floor,
        fail_reasons=fail_codes,
        required_actions=_required_actions(case, fail_codes),
        metadata={
            "path": metadata.get("path"),  # type: ignore[union-attr]
            "material": metadata.get("material"),  # type: ignore[union-attr]
            "degradation_status": degradation,
            "export_strategy": export_strategy,
            "vocal_required": vocal_required,
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
        "goal_directed_candidate_recovery",
        "noise_texture_repair",
        "frisson_goosebumps_protection",
        "vocal_vqi_recovery",
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
) -> tuple[RestorationQualityGateResult, list[RestorationQualityCaseResult]]:
    """Bewertet final restoration quality from an execution Golden-Set report."""
    _cases_raw = execution_report.get("cases")
    raw_cases: list[Any] = _cases_raw if isinstance(_cases_raw, list) else []
    cases = [_case_from_execution(case) for case in raw_cases if isinstance(case, dict)]
    vocal_cases = [case for case in cases if bool(case.metadata.get("vocal_required", False))]
    hpi_values = [case.hpi for case in cases if isinstance(case.hpi, (int, float))]
    quality_values = [case.quality_estimate for case in cases if isinstance(case.quality_estimate, (int, float))]
    execution_gate = execution_report.get("gate") if isinstance(execution_report.get("gate"), dict) else {}
    runtime_factor = float(execution_gate.get("runtime_factor", 0.0) or 0.0)  # type: ignore[union-attr]

    non_degraded_rate = _rate(cases, "non_degraded_export")
    unblocked_rate = _rate(cases, "unblocked_export")
    musical_rate = _rate(cases, "musical_goals_passed")
    noise_rate = _rate(cases, "noise_texture_passed")
    goosebumps_rate = _rate(cases, "goosebumps_passed")
    final_quality_rate = _rate(cases, "final_quality_passed")
    vocal_rate = (
        1.0 if not vocal_cases else sum(1 for case in vocal_cases if case.vocal_floor_passed) / len(vocal_cases)
    )
    hpi_avg = _mean([float(value) for value in hpi_values])
    quality_avg = _mean([float(value) for value in quality_values])

    fail_reasons: list[str] = []
    gate_actions = ["create_real_audio_restoration_quality_gate"]
    if non_degraded_rate < thresholds.min_non_degraded_export_rate:
        fail_reasons.append(
            f"non_degraded_export_rate {non_degraded_rate:.3f} < {thresholds.min_non_degraded_export_rate:.3f}"
        )
    if unblocked_rate < thresholds.min_unblocked_export_rate:
        fail_reasons.append(f"unblocked_export_rate {unblocked_rate:.3f} < {thresholds.min_unblocked_export_rate:.3f}")
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
    if quality_avg is None or quality_avg < thresholds.min_quality_estimate_average:
        fail_reasons.append(
            f"quality_estimate_average {(quality_avg or 0.0):.3f} < {thresholds.min_quality_estimate_average:.3f}"
        )
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
