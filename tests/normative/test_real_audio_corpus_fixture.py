from __future__ import annotations

"""Normative Real-Audio-Korpus-Fixture fuer R5-R12-Folgegates."""


import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.test_uat_acceptance_criteria import (
    _integrated_lufs_or_fallback,
    _noise_floor_dbfs,
    _run_real_audio_restore_with_timeout,
    _safe_corr,
    _to_mono,
    _to_samples_first,
)

_TRACKED_PERCEPTUAL_GOALS = (
    "tonal_center",
    "micro_dynamics",
    "transparenz",
    "waerme",
    "brillanz",
    "vocal_quality",
)

_MATERIAL_BUCKET_ORDER = ("codec", "tape", "vinyl", "digital", "vocals")


def _comfort_delta_score(
    *,
    floor_before: float,
    floor_after_cmp: float,
    lufs_before: float,
    lufs_after: float,
    corr_before: float,
    corr_after: float,
) -> float:
    noise_delta = float(np.clip((floor_before - floor_after_cmp) / 6.0, -1.0, 1.0))
    loudness_stability = float(np.clip((0.75 - abs(lufs_after - lufs_before)) / 3.0, -1.0, 1.0))
    stereo_delta = float(np.clip((corr_before - corr_after) / 2.0, -1.0, 1.0))
    return float(np.clip(0.55 * noise_delta + 0.30 * loudness_stability + 0.15 * stereo_delta, -1.0, 1.0))


def _write_worldclass_corpus_report(report: dict[str, Any]) -> None:
    out_path = Path(os.environ.get("AURIK_REAL_AUDIO_CORPUS_REPORT", "reports/worldclass/real_audio_corpus_gate.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(out_path)


def _case_material_bucket(case: dict[str, Any]) -> str:
    path = Path(str(case.get("path", "")))
    parent = path.parent.name.lower()
    if parent in {"tape", "vinyl", "digital", "vocals"}:
        return parent
    suffix = path.suffix.lower()
    if suffix in {".mp3", ".aac", ".m4a", ".ogg"}:
        return "codec"
    return "other"


def _select_material_balanced_cases(cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in _MATERIAL_BUCKET_ORDER}
    buckets["other"] = []
    for case in cases:
        buckets.setdefault(_case_material_bucket(case), []).append(case)

    selected: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    while len(selected) < min(limit, len(cases)):
        progressed = False
        for bucket in (*_MATERIAL_BUCKET_ORDER, "other"):
            while buckets.get(bucket):
                candidate = buckets[bucket].pop(0)
                path_key = str(candidate.get("path", ""))
                if path_key in seen_paths:
                    continue
                selected.append(candidate)
                seen_paths.add(path_key)
                progressed = True
                break
            if len(selected) >= min(limit, len(cases)):
                break
        if not progressed:
            break
    return selected


def _diagnose_comfort_delta(
    *,
    floor_before: float,
    floor_after_cmp: float,
    lufs_before: float,
    lufs_after: float,
    corr_before: float,
    corr_after: float,
    best_goal_delta: float,
    worst_priority_delta: float,
) -> dict[str, float | str]:
    noise_delta_db = float(floor_before - floor_after_cmp)
    lufs_delta_db = float(lufs_after - lufs_before)
    stereo_delta = float(corr_before - corr_after)
    driver_values = {
        "noise_floor": abs(noise_delta_db),
        "loudness": abs(lufs_delta_db),
        "stereo": abs(stereo_delta),
        "musical_goals": max(abs(best_goal_delta), abs(worst_priority_delta)),
    }
    primary_driver = max(driver_values.items(), key=lambda item: item[1])[0]
    return {
        "noise_floor_delta_db": round(noise_delta_db, 6),
        "lufs_delta_db": round(lufs_delta_db, 6),
        "stereo_corr_delta": round(stereo_delta, 6),
        "primary_driver": primary_driver,
    }


def _summarize_report_groups(report_cases: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in report_cases:
        grouped.setdefault(str(case.get("material_bucket", "unknown")), []).append(case)

    summary: dict[str, dict[str, float | int]] = {}
    for bucket, cases in sorted(grouped.items()):
        comfort = [float(case.get("human_hearing_comfort_delta", 0.0)) for case in cases]
        noise = [float(case.get("diagnostics", {}).get("noise_floor_delta_db", 0.0)) for case in cases]
        summary[bucket] = {
            "case_count": len(cases),
            "mean_human_hearing_comfort_delta": round(float(np.mean(comfort)) if comfort else 0.0, 6),
            "min_human_hearing_comfort_delta": round(float(min(comfort, default=0.0)), 6),
            "mean_noise_floor_delta_db": round(float(np.mean(noise)) if noise else 0.0, 6),
        }
    return summary


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_real_audio_corpus_fixture_covers_multiple_local_sources(
    real_audio_corpus_cases: list[dict[str, Any]],
) -> None:
    paths = [Path(str(case["path"])) for case in real_audio_corpus_cases]
    categories = {path.parent.name for path in paths}

    assert len(paths) >= 4
    assert {"tape", "vinyl", "digital", "vocals"} & categories

    for case in real_audio_corpus_cases:
        audio = np.asarray(case["audio"], dtype=np.float32)
        sr = int(case["sr"])
        assert sr == 48_000
        assert audio.ndim == 2
        assert audio.shape[1] == 2
        assert audio.shape[0] >= sr * 2
        assert np.isfinite(audio).all()
        assert float(np.max(np.abs(audio))) <= 1.0 + 1e-6


@pytest.mark.normative
@pytest.mark.heavy
@pytest.mark.timeout(600)
def test_real_audio_corpus_restore_no_harm_gate(
    request: pytest.FixtureRequest,
    real_audio_corpus_cases: list[dict[str, Any]],
) -> None:
    if not bool(request.config.getoption("--run-heavy-tests")):
        pytest.skip("Real-Audio-Korpus-Restore-Gate nur mit --run-heavy-tests")

    limit = max(1, int(float(os.environ.get("AURIK_REAL_AUDIO_CORPUS_RESTORE_LIMIT", "4") or 4)))
    max_seconds = max(2.0, float(os.environ.get("AURIK_REAL_AUDIO_CORPUS_RESTORE_SECONDS", "3.0") or 3.0))
    ml_budget_s = float(os.environ.get("AURIK_REAL_AUDIO_CORPUS_ML_RUNTIME_BUDGET_S", "2.0") or 2.0)
    timeout_s = float(os.environ.get("AURIK_REAL_AUDIO_CORPUS_RESTORE_TIMEOUT_S", "300") or 300.0)

    from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

    checker = MusicalGoalsChecker(mode="restoration")
    checked: list[str] = []
    report_cases: list[dict[str, Any]] = []
    comfort_delta_scores: list[float] = []
    best_goal_deltas: list[float] = []
    selected_cases = _select_material_balanced_cases(real_audio_corpus_cases, limit)
    selected_buckets = {_case_material_bucket(case) for case in selected_cases}
    assert len(selected_buckets) >= min(4, len(selected_cases)), (
        f"Korpus-Auswahl nicht materialbreit: {selected_buckets}"
    )

    for case in selected_cases:
        material_bucket = _case_material_bucket(case)
        sr = int(case["sr"])
        original = _to_samples_first(np.asarray(case["audio"], dtype=np.float32))
        original = original[: min(original.shape[0], int(sr * max_seconds))]
        restorer_input = original.T if original.ndim == 2 else original

        payload = _run_real_audio_restore_with_timeout(restorer_input, sr, ml_budget_s, timeout_s)
        restored = _to_samples_first(np.asarray(payload["audio"], dtype=np.float32))
        n = min(original.shape[0], restored.shape[0])
        original = original[:n]
        restored = restored[:n]

        assert n >= sr * 2, f"Restore-Ausgabe zu kurz: {case['path']}"
        assert restored.ndim == 2 and restored.shape[1] == 2, f"Stereo-Layout verloren: {case['path']}"
        assert np.isfinite(restored).all(), f"NaN/Inf im Restore: {case['path']}"
        assert float(np.max(np.abs(restored))) <= 1.0 + 1e-6, f"Restore außerhalb [-1,1]: {case['path']}"

        corr_before = _safe_corr(original[:, 0], original[:, 1])
        corr_after = _safe_corr(restored[:, 0], restored[:, 1])
        corr_limit = 1.0 if corr_before >= 0.995 else min(0.995, corr_before + 0.12)
        assert corr_after <= corr_limit + 1e-9, (
            f"Stereo-Kollaps im Korpusfall: {case['path']} corr {corr_before:.3f}->{corr_after:.3f}"
        )

        floor_before = _noise_floor_dbfs(original)
        floor_after = _noise_floor_dbfs(restored)
        lufs_before = _integrated_lufs_or_fallback(original, sr)
        lufs_after = _integrated_lufs_or_fallback(restored, sr)
        floor_after_cmp = floor_after - abs(lufs_after - lufs_before)
        material_key = str(payload.get("material_type", "unknown") or "unknown").lower()
        allowance_db = {
            "vinyl": 2.5,
            "shellac": 2.5,
            "lacquer_disc": 2.5,
            "acetate": 2.5,
            "reel_tape": 2.0,
            "tape": 2.0,
            "cassette": 2.0,
            "mp3_low": 1.5,
            "mp3_high": 1.0,
            "aac": 1.0,
            "streaming": 1.0,
            "cd_digital": 0.5,
            "dat": 0.5,
        }.get(material_key, 1.5)
        assert floor_after_cmp <= floor_before + allowance_db, (
            f"Rauschboden-Korpusregression: {case['path']} {floor_before:.2f}->{floor_after_cmp:.2f} "
            f"material={material_key} limit=+{allowance_db:.1f}dB"
        )

        original_mono = _to_mono(original)
        restored_mono = _to_mono(restored)
        goals_before = checker.measure_all(original_mono, sr)
        goals_after = checker.measure_all(restored_mono, sr, reference=original_mono)
        goal_deltas = {
            goal: float(goals_after.get(goal, 0.0) - goals_before.get(goal, 0.0))
            for goal in _TRACKED_PERCEPTUAL_GOALS
            if goal in goals_before or goal in goals_after
        }
        priority_deltas = {
            goal: delta
            for goal, delta in goal_deltas.items()
            if goal in {"tonal_center", "micro_dynamics", "vocal_quality"}
        }
        worst_priority_delta = min(priority_deltas.values(), default=0.0)
        best_delta = max(goal_deltas.values(), default=0.0)
        comfort_delta = _comfort_delta_score(
            floor_before=floor_before,
            floor_after_cmp=floor_after_cmp,
            lufs_before=lufs_before,
            lufs_after=lufs_after,
            corr_before=corr_before,
            corr_after=corr_after,
        )
        assert worst_priority_delta >= -0.18, (
            f"Perzeptuelle P0/P2-Regression im Korpusfall: {case['path']} "
            f"worst={worst_priority_delta:.3f} deltas={goal_deltas}"
        )
        assert best_delta >= -0.02, (
            f"Kein stabiler perzeptueller Zielvektor im Korpusfall: {case['path']} deltas={goal_deltas}"
        )
        assert comfort_delta > 0.0, (
            f"Human-Hearing-Comfort-Regression im Korpusfall: {case['path']} "
            f"comfort_delta={comfort_delta:.3f} floor={floor_before:.2f}->{floor_after_cmp:.2f} "
            f"lufs={lufs_before:.2f}->{lufs_after:.2f} corr={corr_before:.3f}->{corr_after:.3f}"
        )

        comfort_delta_scores.append(comfort_delta)
        best_goal_deltas.append(best_delta)

        report_cases.append(
            {
                "path": str(case["path"]),
                "material_bucket": material_bucket,
                "material_type": material_key,
                "samples": int(n),
                "sr": int(sr),
                "stereo_corr_before": round(float(corr_before), 6),
                "stereo_corr_after": round(float(corr_after), 6),
                "noise_floor_before_dbfs": round(float(floor_before), 3),
                "noise_floor_after_cmp_dbfs": round(float(floor_after_cmp), 3),
                "lufs_before": round(float(lufs_before), 3),
                "lufs_after": round(float(lufs_after), 3),
                "goal_deltas": {key: round(float(value), 6) for key, value in sorted(goal_deltas.items())},
                "best_goal_delta": round(float(best_delta), 6),
                "human_hearing_comfort_delta": round(float(comfort_delta), 6),
                "diagnostics": _diagnose_comfort_delta(
                    floor_before=floor_before,
                    floor_after_cmp=floor_after_cmp,
                    lufs_before=lufs_before,
                    lufs_after=lufs_after,
                    corr_before=corr_before,
                    corr_after=corr_after,
                    best_goal_delta=best_delta,
                    worst_priority_delta=worst_priority_delta,
                ),
                "worst_priority_delta": round(float(worst_priority_delta), 6),
                "verdict": "PASS",
            }
        )
        checked.append(str(case["path"]))

    assert len(checked) == len(selected_cases)
    mean_comfort_delta = float(np.mean(comfort_delta_scores)) if comfort_delta_scores else 0.0
    mean_best_goal_delta = float(np.mean(best_goal_deltas)) if best_goal_deltas else 0.0
    assert mean_comfort_delta > 0.0, f"Korpus-Comfort-Gesamtdelta nicht positiv: {mean_comfort_delta:.3f}"
    group_summary = _summarize_report_groups(report_cases)
    assert all(float(group.get("mean_human_hearing_comfort_delta", 0.0)) > 0.0 for group in group_summary.values()), (
        f"Materialgruppen-Comfort nicht durchgehend positiv: {group_summary}"
    )
    _write_worldclass_corpus_report(
        {
            "schema": "aurik.worldclass_real_audio_evidence.v2",
            "gate": "real_audio_corpus_material_balanced_positive_comfort_delta",
            "case_count": len(report_cases),
            "cases": report_cases,
            "material_group_summary": group_summary,
            "selection": {
                "requested_limit": int(limit),
                "selected_buckets": sorted(selected_buckets),
            },
            "summary": {
                "mean_best_goal_delta": round(mean_best_goal_delta, 6),
                "mean_human_hearing_comfort_delta": round(mean_comfort_delta, 6),
                "min_human_hearing_comfort_delta": round(float(min(comfort_delta_scores, default=0.0)), 6),
            },
            "verdict": "PASS",
        }
    )
