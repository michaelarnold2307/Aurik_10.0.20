"""Erzeugt den reproduzierbaren Experimentplan fuer Klasse-C-Revalidierung (PR-A).

Ausgabe:
- reports/revalidation/<run_id>/plan.csv
- reports/revalidation/<run_id>/result_template.csv
- reports/revalidation/<run_id>/run_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaseItem:
    """Ein einzelner Revalidierungsfall aus dem Manifest."""

    case_id: str
    audio_path: str
    material: str
    mode: str
    restorability_bin: str
    vocal_focus: bool


def _load_manifest(path: Path) -> list[CaseItem]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items: list[CaseItem] = []
    for raw in data.get("cases", []):
        items.append(
            CaseItem(
                case_id=str(raw["case_id"]),
                audio_path=str(raw["audio_path"]),
                material=str(raw["material"]),
                mode=str(raw.get("mode", "restoration")),
                restorability_bin=str(raw.get("restorability_bin", "unknown")),
                vocal_focus=bool(raw.get("vocal_focus", True)),
            )
        )
    return items


def _expand_plan(cases: list[CaseItem]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # WP1: material_vqi_floor Sensitivitaet
    wp1_variants = [
        ("wp1_vqi_floor", "baseline_minus_0p03", -0.03),
        ("wp1_vqi_floor", "baseline", 0.0),
        ("wp1_vqi_floor", "baseline_plus_0p03", 0.03),
    ]

    # WP2: MERT-Floor Grid
    wp2_variants = [
        ("wp2_mert_floor", "mert_0p45", 0.45),
        ("wp2_mert_floor", "mert_0p50", 0.50),
        ("wp2_mert_floor", "mert_0p55", 0.55),
    ]

    # WP3: timbral floor curve candidates
    wp3_variants = [
        ("wp3_timbral_floor", "curve_baseline", "baseline"),
        ("wp3_timbral_floor", "curve_flatter", "flat_minus"),
        ("wp3_timbral_floor", "curve_steeper", "steep_plus"),
    ]

    for case in cases:
        if case.mode != "restoration":
            continue

        for wp, variant, delta in wp1_variants:
            rows.append(
                {
                    "workpackage": wp,
                    "variant": variant,
                    "case_id": case.case_id,
                    "audio_path": case.audio_path,
                    "material": case.material,
                    "restorability_bin": case.restorability_bin,
                    "vocal_focus": case.vocal_focus,
                    "param_name": "material_vqi_floor_delta",
                    "param_value": delta,
                }
            )

        for wp, variant, floor in wp2_variants:
            rows.append(
                {
                    "workpackage": wp,
                    "variant": variant,
                    "case_id": case.case_id,
                    "audio_path": case.audio_path,
                    "material": case.material,
                    "restorability_bin": case.restorability_bin,
                    "vocal_focus": case.vocal_focus,
                    "param_name": "mert_floor",
                    "param_value": floor,
                }
            )

        for wp, variant, curve in wp3_variants:
            rows.append(
                {
                    "workpackage": wp,
                    "variant": variant,
                    "case_id": case.case_id,
                    "audio_path": case.audio_path,
                    "material": case.material,
                    "restorability_bin": case.restorability_bin,
                    "vocal_focus": case.vocal_focus,
                    "param_name": "timbral_curve",
                    "param_value": curve,
                }
            )

    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        headers = [
            "workpackage",
            "variant",
            "case_id",
            "audio_path",
            "material",
            "restorability_bin",
            "vocal_focus",
            "param_name",
            "param_value",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_result_template(path: Path, plan_rows: list[dict[str, Any]]) -> None:
    result_rows: list[dict[str, Any]] = []
    for row in plan_rows:
        result_rows.append(
            {
                **row,
                "system": "aurik",
                "is_aurik": "true",
                "baseline_family": "",
                "era": "",
                "genre": "",
                "artifact_freedom": "",
                "vqi": "",
                "mert_similarity": "",
                "timbral_fidelity": "",
                "hpi": "",
                "naturalness": "",
                "emotional_arc_preservation": "",
                "micro_dynamic_correlation": "",
                "formant_integrity": "",
                "vibrato_depth_preservation": "",
                "noise_texture_distance": "",
                "manual_intervention_count": "0",
                "user_parameter_count": "0",
                "canonical_bridge_contract": "true",
                "autonomous_export_decision": "true",
                "mode": "restoration",
                "status": "",
                "fail_reason": "",
                "mushra_light_vocal": "",
                "singer_identity_cosine": "",
                "notes": "",
            }
        )
    _write_csv(path, result_rows)


def main() -> None:
    """CLI-Einstiegspunkt: erzeugt Plan-, Ergebnis-Template- und Metadaten-Dateien."""
    parser = argparse.ArgumentParser(description="Erzeuge Klasse-C Revalidierungsplan")
    parser.add_argument(
        "--manifest",
        default="config/class_c_revalidation_manifest.example.json",
        help="Pfad zum Case-Manifest (JSON).",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/revalidation",
        help="Basisordner fuer Plan-Outputs.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest_path = (root / args.manifest).resolve()
    out_base = (root / args.out_dir).resolve()

    cases = _load_manifest(manifest_path)
    plan_rows = _expand_plan(cases)

    run_id = f"class_c_reval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    plan_csv = run_dir / "plan.csv"
    result_template_csv = run_dir / "result_template.csv"
    metadata_json = run_dir / "run_metadata.json"

    _write_csv(plan_csv, plan_rows)
    _write_result_template(result_template_csv, plan_rows)

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(manifest_path),
        "num_cases": len(cases),
        "num_plan_rows": len(plan_rows),
        "workpackages": ["wp1_vqi_floor", "wp2_mert_floor", "wp3_timbral_floor"],
    }
    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Plan erstellt: {plan_csv}")
    print(f"Result-Template: {result_template_csv}")
    print(f"Run-Metadaten: {metadata_json}")
    print(f"Planzeilen: {len(plan_rows)}")


if __name__ == "__main__":
    main()
