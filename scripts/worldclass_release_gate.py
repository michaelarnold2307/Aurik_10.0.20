"""Harter Worldclass-Release-Gate auf Basis des KPI-Dashboards.

Failt mit Exit-Code 2, wenn einer der konfigurierten KPI-Targets verfehlt wird.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _check_min(value: float | None, minimum: float, name: str) -> str | None:
    if value is None:
        return f"{name}: fehlt"
    if value < minimum:
        return f"{name}: {value:.6f} < {minimum:.6f}"
    return None


def _check_max(value: float | None, maximum: float, name: str) -> str | None:
    if value is None:
        return f"{name}: fehlt"
    if value > maximum:
        return f"{name}: {value:.6f} > {maximum:.6f}"
    return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueft Worldclass KPI Gate")
    parser.add_argument(
        "--dashboard-json",
        default="reports/worldclass/worldclass_kpi_dashboard.json",
        help="Pfad zum KPI-Dashboard JSON",
    )
    parser.add_argument(
        "--trusted-report-json",
        default="",
        help="Pfad zum Trusted Vocal Restoration Report JSON; default: neben Dashboard.",
    )
    args = parser.parse_args()

    dashboard_path = Path(args.dashboard_json).resolve()
    data = json.loads(dashboard_path.read_text(encoding="utf-8"))

    kpis = data.get("kpis", {})
    targets = data.get("targets", {})
    real_audio_corpus = data.get("real_audio_corpus", {})
    trusted_vocal_restoration = data.get("trusted_vocal_restoration", {})

    violations: list[str] = []

    artifact_samples = int(kpis.get("artifact_freedom_sample_count", 0) or 0)
    vqi_samples = int(kpis.get("vqi_sample_count", 0) or 0)
    runtime_samples = int(kpis.get("runtime_sample_count", 0) or 0)
    wcs_samples = int(kpis.get("wcs_sample_count", 0) or 0)
    defect_detection_samples = int(kpis.get("defect_detection_sample_count", 0) or 0)
    defect_confidence_samples = int(kpis.get("defect_confidence_sample_count", 0) or 0)
    defect_inaudible_samples = int(kpis.get("defect_inaudible_sample_count", 0) or 0)

    min_artifact_samples = int(targets.get("artifact_freedom_min_samples", 1) or 1)
    min_vqi_samples = int(targets.get("vqi_min_samples", 1) or 1)
    min_runtime_samples = int(targets.get("runtime_min_samples", 1) or 1)
    min_wcs_samples = int(targets.get("wcs_min_samples", 1) or 1)
    min_defect_detection_samples = int(targets.get("defect_detection_min_samples", 1) or 1)
    min_defect_confidence_samples = int(targets.get("defect_confidence_min_samples", 1) or 1)
    min_defect_inaudible_samples = int(targets.get("defect_inaudible_min_samples", 1) or 1)

    if artifact_samples < min_artifact_samples:
        violations.append(f"artifact_freedom_sample_count: {artifact_samples} < {min_artifact_samples}")
    if vqi_samples < min_vqi_samples:
        violations.append(f"vqi_sample_count: {vqi_samples} < {min_vqi_samples}")
    if runtime_samples < min_runtime_samples:
        violations.append(f"runtime_sample_count: {runtime_samples} < {min_runtime_samples}")
    if wcs_samples < min_wcs_samples:
        violations.append(f"wcs_sample_count: {wcs_samples} < {min_wcs_samples}")
    if defect_detection_samples < min_defect_detection_samples:
        violations.append(f"defect_detection_sample_count: {defect_detection_samples} < {min_defect_detection_samples}")
    if defect_confidence_samples < min_defect_confidence_samples:
        violations.append(
            f"defect_confidence_sample_count: {defect_confidence_samples} < {min_defect_confidence_samples}"
        )
    if defect_inaudible_samples < min_defect_inaudible_samples:
        violations.append(f"defect_inaudible_sample_count: {defect_inaudible_samples} < {min_defect_inaudible_samples}")

    corpus_diversity = kpis.get("corpus_diversity", {})
    if isinstance(real_audio_corpus, dict) and real_audio_corpus:
        if not isinstance(corpus_diversity, dict):
            violations.append("corpus_diversity: fehlt")
        else:
            min_cases = int(real_audio_corpus.get("min_cases", 0) or 0)
            total_samples = int(corpus_diversity.get("total_samples", 0) or 0)
            if total_samples < min_cases:
                violations.append(f"real_audio_total_samples: {total_samples} < {min_cases}")
            for key in (
                "missing_required_materials",
                "missing_required_case_ids",
                "non_vocal_required_case_ids",
            ):
                missing = corpus_diversity.get(key, [])
                if isinstance(missing, list) and missing:
                    violations.append(f"{key}: {missing}")

    if isinstance(trusted_vocal_restoration, dict) and trusted_vocal_restoration:
        trusted_path = (
            Path(args.trusted_report_json).resolve()
            if args.trusted_report_json
            else dashboard_path.parent / "trusted_vocal_restoration_report.json"
        )
        if not trusted_path.exists():
            violations.append(f"trusted_vocal_restoration_report: fehlt ({trusted_path})")
        else:
            trusted_report = json.loads(trusted_path.read_text(encoding="utf-8"))
            best_possible = trusted_report.get("best_possible_restoration", {})
            if not isinstance(best_possible, dict) or best_possible.get("best_possible_reached") is not True:
                violations.append("trusted_vocal_restoration_report: best_possible_restoration fehlt")
            user_confidence = trusted_report.get("user_confidence_summary", {})
            if not isinstance(user_confidence, dict) or user_confidence.get("manual_action_required") is not False:
                violations.append("trusted_vocal_restoration_report: user_confidence_summary fehlt")

    v = _check_min(
        _to_float(kpis.get("artifact_freedom_pass_rate")),
        float(targets.get("artifact_freedom_pass_rate_min", 0.99)),
        "artifact_freedom_pass_rate",
    )
    if v:
        violations.append(v)

    v = _check_min(
        _to_float(kpis.get("defect_confidence_pass_rate")),
        float(targets.get("defect_confidence_pass_rate_min", 0.90)),
        "defect_confidence_pass_rate",
    )
    if v:
        violations.append(v)

    v = _check_min(
        _to_float(kpis.get("defect_confidence_coverage_rate")),
        float(targets.get("defect_confidence_coverage_rate_min", 0.95)),
        "defect_confidence_coverage_rate",
    )
    if v:
        violations.append(v)

    v = _check_min(
        _to_float(kpis.get("era_confidence_pass_rate")),
        float(targets.get("era_confidence_pass_rate_min", 0.80)),
        "era_confidence_pass_rate",
    )
    if v:
        violations.append(v)

    v = _check_min(
        _to_float(kpis.get("genre_confidence_pass_rate")),
        float(targets.get("genre_confidence_pass_rate_min", 0.80)),
        "genre_confidence_pass_rate",
    )
    if v:
        violations.append(v)

    v = _check_min(
        _to_float(kpis.get("material_confidence_pass_rate")),
        float(targets.get("material_confidence_pass_rate_min", 0.80)),
        "material_confidence_pass_rate",
    )
    if v:
        violations.append(v)

    v = _check_min(
        _to_float(kpis.get("pipeline_confidence_pass_rate")),
        float(targets.get("pipeline_confidence_pass_rate_min", 0.90)),
        "pipeline_confidence_pass_rate",
    )
    if v:
        violations.append(v)

    v = _check_min(
        _to_float(kpis.get("uncertainty_coverage_rate")),
        float(targets.get("uncertainty_coverage_rate_min", 0.70)),
        "uncertainty_coverage_rate",
    )
    if v:
        violations.append(v)

    v = _check_min(
        _to_float(kpis.get("defect_detection_pass_rate")),
        float(targets.get("defect_detection_pass_rate_min", 0.95)),
        "defect_detection_pass_rate",
    )
    if v:
        violations.append(v)

    v = _check_min(
        _to_float(kpis.get("defect_inaudible_or_max_reduction_pass_rate")),
        float(targets.get("defect_inaudible_or_max_reduction_pass_rate_min", 0.95)),
        "defect_inaudible_or_max_reduction_pass_rate",
    )
    if v:
        violations.append(v)

    v = _check_min(
        _to_float(kpis.get("vqi_margin_pass_rate")),
        float(targets.get("vqi_margin_pass_rate_min", 0.95)),
        "vqi_margin_pass_rate",
    )
    if v:
        violations.append(v)

    v = _check_min(
        _to_float(kpis.get("wcs_pass_rate")),
        float(targets.get("wcs_pass_rate_min", 0.95)),
        "wcs_pass_rate",
    )
    if v:
        violations.append(v)

    v = _check_max(
        _to_float(kpis.get("false_reject_rate")),
        float(targets.get("false_reject_rate_max", 0.08)),
        "false_reject_rate",
    )
    if v:
        violations.append(v)

    v = _check_max(
        _to_float(kpis.get("runtime_p95_seconds")),
        float(targets.get("runtime_p95_seconds_max", 600.0)),
        "runtime_p95_seconds",
    )
    if v:
        violations.append(v)

    if violations:
        print("WORLDCLASS RELEASE GATE: FAIL")
        for violation in violations:
            print(f" - {violation}")
        raise SystemExit(2)

    print("WORLDCLASS RELEASE GATE: PASS")


if __name__ == "__main__":
    main()
