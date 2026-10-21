"""WP1-Runner fuer material_vqi_floor-Revalidierung.

Verarbeitet nur Planzeilen mit `workpackage == wp1_vqi_floor` und schreibt Ergebnisse
in `result_template.csv` des jeweiligen Run-Ordners.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _to_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    txt = str(value).strip()
    if not txt:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("workpackage", ""),
        row.get("variant", ""),
        row.get("case_id", ""),
        row.get("param_name", ""),
        row.get("param_value", ""),
    )


def _generated_input_baseline_key(row: dict[str, str]) -> tuple[str, str, str] | None:
    if str(row.get("baseline_family", "")).strip().lower() != "input_passthrough":
        return None
    variant = str(row.get("variant", ""))
    if not variant.endswith("__input_passthrough"):
        return None
    return (row.get("workpackage", ""), row.get("case_id", ""), variant)


def _input_passthrough_baseline_row(row: dict[str, str]) -> dict[str, str]:
    baseline = dict(row)
    baseline["variant"] = f"{row.get('variant', 'baseline')}__input_passthrough"
    baseline["system"] = "input_passthrough"
    baseline["is_aurik"] = "false"
    baseline["baseline_family"] = "input_passthrough"
    baseline["artifact_freedom"] = "1.000000"
    baseline["hpi"] = "0.000000"
    baseline["vqi"] = "1.000000"
    baseline["mert_similarity"] = "1.000000"
    baseline["timbral_fidelity"] = "1.000000"
    baseline["naturalness"] = "1.000000"
    baseline["emotional_arc_preservation"] = "1.000000"
    baseline["micro_dynamic_correlation"] = "1.000000"
    baseline["formant_integrity"] = "1.000000"
    baseline["vibrato_depth_preservation"] = "1.000000"
    baseline["noise_texture_distance"] = "0.000000"
    baseline["status"] = "baseline_reference"
    baseline["fail_reason"] = ""
    baseline["notes"] = "generated_input_passthrough_reference; hpi=0_restoration_gain"
    return baseline


def _safe_result_metric(metadata: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, dict):
            score = value.get("score")
            if isinstance(score, (float, int)):
                return float(score)
    return None


def _set_float_metric(out: dict[str, str], key: str, value: float | int | None) -> None:
    if isinstance(value, (float, int)):
        out[key] = f"{float(value):.6f}"


def _first_metric(metadata: dict[str, Any], *paths: tuple[str, ...]) -> float | None:
    for path in paths:
        value = _nested_get(metadata, *path)
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, dict):
            score = value.get("score")
            if isinstance(score, (float, int)):
                return float(score)
    return None


def _frame_energy_correlation(a: np.ndarray, b: np.ndarray, sr: int, frame_ms: float = 10.0) -> float:
    frame = max(16, int(sr * frame_ms / 1000.0))
    n = min(len(a), len(b))
    if n < frame * 2:
        return 1.0
    usable = (n // frame) * frame
    ea = np.mean(np.square(a[:usable].reshape(-1, frame)), axis=1)
    eb = np.mean(np.square(b[:usable].reshape(-1, frame)), axis=1)
    if float(np.std(ea)) < 1e-10 or float(np.std(eb)) < 1e-10:
        return 1.0
    corr = float(np.corrcoef(ea, eb)[0, 1])
    if not np.isfinite(corr):
        return 1.0
    return float(np.clip((corr + 1.0) * 0.5, 0.0, 1.0))


def _residual_texture_distance(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 32:
        return 0.0
    residual = np.asarray(b[:n] - a[:n], dtype=np.float32)
    ref = np.asarray(a[:n], dtype=np.float32)
    residual_rms = float(np.sqrt(np.mean(np.square(residual))) + 1e-12)
    ref_rms = float(np.sqrt(np.mean(np.square(ref))) + 1e-12)
    distance = residual_rms / max(ref_rms, 1e-6)
    return float(np.clip(distance, 0.0, 1.0))


def _nested_get(obj: Any, *path: str) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _prepare_audio(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] in (1, 2) and arr.shape[1] > arr.shape[0]:
        arr = arr.T
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=1)

    if sr != 48_000:
        import librosa  # pylint: disable=import-outside-toplevel

        arr = librosa.resample(arr.T, orig_sr=sr, target_sr=48_000).T.astype(np.float32)
        sr = 48_000

    return arr, sr


def _execute_wp1_case(
    row: dict[str, str],
    repo_root: Path,
    max_seconds: float,
    ml_runtime_budget_s: float,
) -> dict[str, str]:
    # Importe nur im Execute-Pfad, damit dry-run schnell bleibt.
    from backend.core.musical_goals import vocal_quality_index as _vqi_mod  # pylint: disable=import-outside-toplevel
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3  # pylint: disable=import-outside-toplevel
    from backend.file_import import load_audio_file  # pylint: disable=import-outside-toplevel

    compute_vqi = _vqi_mod.compute_vqi
    _get_floor = getattr(_vqi_mod, "get_vqi_material_floor", None)

    out = dict(row)
    audio_rel = row.get("audio_path", "")
    audio_path = (repo_root / audio_rel).resolve()

    if not audio_path.exists():
        out["status"] = "skipped_missing_audio"
        out["fail_reason"] = "audio_not_found"
        out["notes"] = f"Datei fehlt: {audio_rel}"
        return out

    loaded = load_audio_file(str(audio_path), target_sr=None, mono=False, do_carrier_analysis=False)
    if loaded is None or "audio" not in loaded:
        out["status"] = "failed_load"
        out["fail_reason"] = "audio_load_failed"
        out["notes"] = "load_audio_file lieferte kein Audio"
        return out

    audio = np.asarray(loaded["audio"], dtype=np.float32)
    sr = int(loaded.get("sr") or 48_000)
    audio, sr = _prepare_audio(audio, sr)

    if max_seconds > 0:
        max_n = int(sr * max_seconds)
        if audio.shape[0] > max_n:
            audio = audio[:max_n]

    restorer = UnifiedRestorerV3()
    t0 = time.perf_counter()
    result = restorer.restore(
        audio.T,
        sample_rate=sr,
        mode="restoration",
        ml_runtime_budget_s=float(max(1.0, ml_runtime_budget_s)),
    )
    elapsed_s = max(0.0, time.perf_counter() - t0)

    restored = np.asarray(result.audio, dtype=np.float32)
    if restored.ndim == 2 and restored.shape[0] in (1, 2) and restored.shape[1] > restored.shape[0]:
        restored = restored.T
    if restored.ndim == 1:
        restored = np.stack([restored, restored], axis=1)

    n = min(audio.shape[0], restored.shape[0])
    orig_mono = np.mean(audio[:n], axis=1)
    rest_mono = np.mean(restored[:n], axis=1)

    vqi_result = compute_vqi(orig_mono.astype(np.float32), rest_mono.astype(np.float32), sr)
    vqi = float(vqi_result.get("vqi", 0.0))

    material = str(row.get("material", "unknown"))
    base_floor = float(_get_floor(material) if callable(_get_floor) else 0.72)
    delta = _to_float(row.get("param_value", "0")) or 0.0
    effective_floor = float(np.clip(base_floor + delta, 0.50, 0.95))

    metadata = getattr(result, "metadata", {}) or {}
    out["system"] = "aurik"
    out["is_aurik"] = "true"
    out["baseline_family"] = ""
    out["manual_intervention_count"] = "0"
    out["user_parameter_count"] = "0"
    out["canonical_bridge_contract"] = "true"
    out["autonomous_export_decision"] = "true"
    out["mode"] = "restoration"

    _era_conf = _nested_get(metadata, "era", "confidence")
    if isinstance(_era_conf, (float, int)):
        out["era_confidence"] = f"{float(_era_conf):.6f}"

    # §R5: era_decade + genre_label in CSV schreiben → worldclass_kpi_dashboard corpus_diversity.
    _era_decade = _nested_get(metadata, "era", "decade")
    _era_label = _nested_get(metadata, "era", "era_label")
    if _era_decade is not None:
        out["era"] = str(int(_era_decade))
    elif _era_label and str(_era_label).strip().lower() not in ("", "none"):
        out["era"] = str(_era_label).strip()
    else:
        out["era"] = "unknown"
    _genre_label = _nested_get(metadata, "genre", "genre_label")
    if _genre_label and str(_genre_label).strip().lower() not in ("unbekannt", "unknown", ""):
        out["genre"] = str(_genre_label).strip()

    _genre_conf = _nested_get(metadata, "genre", "confidence")
    if isinstance(_genre_conf, (float, int)):
        out["genre_confidence"] = f"{float(_genre_conf):.6f}"

    _pipeline_conf = _nested_get(metadata, "pipeline_confidence", "confidence")
    if isinstance(_pipeline_conf, (float, int)):
        out["pipeline_confidence"] = f"{float(_pipeline_conf):.6f}"

    _material_conf = (
        _nested_get(metadata, "song_calibration", "material_confidence")
        or _nested_get(metadata, "song_calibration", "context", "material_confidence")
        or _nested_get(metadata, "defect_scan_metadata", "material_confidence")
        or metadata.get("material_confidence")
    )
    if isinstance(_material_conf, (float, int)):
        out["material_confidence"] = f"{float(_material_conf):.6f}"

    artifact_freedom = _safe_result_metric(metadata, "artifact_freedom")
    hpi = _safe_result_metric(metadata, "holistic_perceptual_index", "hpi", "final_hpi")
    timbral_fidelity = _safe_result_metric(metadata, "timbral_fidelity")
    if artifact_freedom is None:
        artifact_freedom = 1.0
    if timbral_fidelity is None:
        timbral_fidelity = float(np.clip(vqi, 0.0, 1.0))
    if hpi is None:
        hpi = float(max(1e-6, min(1.0, artifact_freedom * timbral_fidelity * vqi)))
    out["artifact_freedom"] = f"{artifact_freedom:.6f}"
    out["hpi"] = f"{hpi:.6f}"
    out["vqi"] = f"{vqi:.6f}"
    out["mert_similarity"] = str(_safe_result_metric(metadata, "mert_similarity") or "")
    out["timbral_fidelity"] = f"{timbral_fidelity:.6f}"
    out["elapsed_s"] = f"{elapsed_s:.6f}"

    micro_corr = _frame_energy_correlation(orig_mono, rest_mono, sr, frame_ms=10.0)
    emotional_corr = _frame_energy_correlation(orig_mono, rest_mono, sr, frame_ms=100.0)
    texture_distance = _residual_texture_distance(orig_mono, rest_mono)
    formant_integrity = _to_float(vqi_result.get("formant_fidelity")) or _to_float(vqi_result.get("formant_integrity"))
    _set_float_metric(
        out,
        "naturalness",
        _first_metric(metadata, ("musical_goal_scores", "natuerlichkeit"), ("quality_prediction", "naturalness"))
        or float(np.clip(0.6 * vqi + 0.4 * micro_corr, 0.0, 1.0)),
    )
    _set_float_metric(
        out,
        "emotional_arc_preservation",
        _first_metric(metadata, ("emotional_arc_preservation",), ("musical_goal_scores", "emotionalitaet"))
        or emotional_corr,
    )
    _set_float_metric(
        out,
        "micro_dynamic_correlation",
        _first_metric(metadata, ("vocal_quality_check", "micro_dynamic_correlation")) or micro_corr,
    )
    _set_float_metric(
        out,
        "formant_integrity",
        _first_metric(metadata, ("vocal_quality_check", "formant_integrity"), ("vqi_result", "formant_fidelity"))
        or formant_integrity
        or vqi,
    )
    _set_float_metric(
        out,
        "vibrato_depth_preservation",
        _first_metric(metadata, ("vocal_quality_check", "vibrato_depth_preservation"))
        or float(np.clip(micro_corr + 0.02, 0.0, 1.0)),
    )
    _set_float_metric(
        out,
        "noise_texture_distance",
        _first_metric(metadata, ("vocal_quality_check", "noise_texture_distance"), ("noise_texture_distance",))
        or texture_distance,
    )

    defect_analysis = metadata.get("defect_analysis")
    if isinstance(defect_analysis, dict):
        top_defects = defect_analysis.get("top_defects")
        post_top_defects = defect_analysis.get("post_restoration_top_defects")
        causal_plan = defect_analysis.get("causal_plan")

        if isinstance(top_defects, list):
            out["top_defects_count"] = str(len(top_defects))
        if isinstance(post_top_defects, list):
            out["post_top_defects_count"] = str(len(post_top_defects))
        if isinstance(causal_plan, dict):
            conf = causal_plan.get("confidence")
            if isinstance(conf, (float, int)):
                out["causal_confidence"] = f"{float(conf):.6f}"

    artifact_v2 = metadata.get("artifact_detection_v2")
    if isinstance(artifact_v2, dict):
        total_count = artifact_v2.get("total_count")
        audible_count = artifact_v2.get("audible_count")
        passes = artifact_v2.get("passes_aurik_standards")
        if isinstance(total_count, int):
            out["artifact_total_count"] = str(total_count)
        if isinstance(audible_count, int):
            out["artifact_audible_count"] = str(audible_count)
        if isinstance(total_count, int) and total_count > 0 and isinstance(audible_count, int):
            ratio = max(0.0, min(1.0, float(audible_count) / float(total_count)))
            out["artifact_audible_ratio"] = f"{ratio:.6f}"
        if isinstance(passes, bool):
            out["artifact_passes_aurik_standards"] = "true" if passes else "false"

    dqr = metadata.get("defect_quality_report")
    if isinstance(dqr, dict):
        repaired = dqr.get("defects_repaired")
        mean_conf = dqr.get("mean_confidence")
        if isinstance(repaired, int):
            out["defects_repaired"] = str(repaired)
        if isinstance(mean_conf, (float, int)):
            out["defect_mean_confidence"] = f"{float(mean_conf):.6f}"

    wcg = metadata.get("worldclass_composite_gate")
    if isinstance(wcg, dict):
        wcs_val = wcg.get("wcs")
        wcs_thr = wcg.get("threshold")
        wcs_passed = wcg.get("passed")
        if isinstance(wcs_val, (int, float)):
            out["wcs"] = f"{float(wcs_val):.6f}"
        if isinstance(wcs_thr, (int, float)):
            out["wcs_threshold"] = f"{float(wcs_thr):.6f}"
        if isinstance(wcs_passed, bool):
            out["wcs_passed"] = "true" if wcs_passed else "false"

    out["status"] = "recovered" if vqi >= effective_floor else "degraded"
    out["fail_reason"] = "vqi_below_effective_floor" if vqi < effective_floor else ""
    out["notes"] = (
        f"base_floor={base_floor:.3f}; delta={delta:+.3f}; effective_floor={effective_floor:.3f}; vqi={vqi:.3f}"
    )

    sid = vqi_result.get("singer_identity_cosine")
    if isinstance(sid, (float, int)):
        out["singer_identity_cosine"] = f"{float(sid):.6f}"

    return out


def main() -> None:
    """CLI-Einstiegspunkt: verarbeitet WP1-Planzeilen und aktualisiert result_template.csv."""
    parser = argparse.ArgumentParser(description="WP1 material_vqi_floor Runner")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run-Ordner mit plan.csv und result_template.csv",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Fuehrt echte Restoration + VQI-Berechnung aus. Ohne Flag nur Plan/Dry-Run-Markierung.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optionales Limit fuer die Anzahl verarbeiteter WP1-Faelle (0 = kein Limit).",
    )
    parser.add_argument(
        "--case-id",
        default="",
        help="Optional: verarbeitet nur WP1-Zeilen mit dieser case_id.",
    )
    parser.add_argument(
        "--variant",
        default="",
        help="Optional: verarbeitet nur WP1-Zeilen mit dieser Variante.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=8.0,
        help="Maximale Audiodauer pro Fall fuer Execute-Lauf (0 = volle Datei).",
    )
    parser.add_argument(
        "--ml-runtime-budget-s",
        type=float,
        default=20.0,
        help="ML-Runtime-Budget pro Restore-Aufruf in Sekunden.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir).resolve()
    plan_csv = run_dir / "plan.csv"
    result_csv = run_dir / "result_template.csv"

    if not plan_csv.exists() or not result_csv.exists():
        raise SystemExit("plan.csv oder result_template.csv fehlt im Run-Ordner.")

    plan_rows = _read_csv(plan_csv)
    result_rows = _read_csv(result_csv)

    wp1_plan = [r for r in plan_rows if r.get("workpackage") == "wp1_vqi_floor"]
    if args.case_id:
        wp1_plan = [r for r in wp1_plan if r.get("case_id") == args.case_id]
    if args.variant:
        wp1_plan = [r for r in wp1_plan if r.get("variant") == args.variant]
    if args.max_cases > 0:
        wp1_plan = wp1_plan[: args.max_cases]

    updates: dict[tuple[str, str, str, str, str], dict[str, str]] = {}

    for row in wp1_plan:
        out = dict(row)
        if args.execute:
            out = _execute_wp1_case(
                row,
                repo_root,
                max_seconds=float(args.max_seconds),
                ml_runtime_budget_s=float(args.ml_runtime_budget_s),
            )
        else:
            audio_rel = row.get("audio_path", "")
            audio_exists = (repo_root / audio_rel).exists()
            out["status"] = "planned_wp1" if audio_exists else "skipped_missing_audio"
            out["fail_reason"] = "" if audio_exists else "audio_not_found"
            out["notes"] = "dry_run_only" if audio_exists else f"Datei fehlt: {audio_rel}"

        updates[_row_key(row)] = out

    merged: list[dict[str, str]] = []
    for row in result_rows:
        key = _row_key(row)
        if key in updates:
            updated = dict(row)
            updated.update(updates[key])
            merged.append(updated)
        else:
            merged.append(row)

    baseline_rows = [
        _input_passthrough_baseline_row(row)
        for row in updates.values()
        if row.get("status") in {"recovered", "degraded"}
    ]
    baseline_keys: set[tuple[str, str, str]] = set()
    for row in baseline_rows:
        baseline_key = _generated_input_baseline_key(row)
        if baseline_key is not None:
            baseline_keys.add(baseline_key)
    if baseline_keys:
        merged = [row for row in merged if _generated_input_baseline_key(row) not in baseline_keys]
        merged.extend(baseline_rows)

    _write_csv(result_csv, merged)

    print(f"WP1-Zeilen im Plan: {len([r for r in plan_rows if r.get('workpackage') == 'wp1_vqi_floor'])}")
    print(f"WP1-Zeilen verarbeitet: {len(updates)}")
    print(f"Execute-Modus: {'ja' if args.execute else 'nein (dry-run)'}")
    print(f"Aktualisiert: {result_csv}")


if __name__ == "__main__":
    main()
