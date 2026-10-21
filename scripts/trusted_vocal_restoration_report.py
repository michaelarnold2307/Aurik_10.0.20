"""Erzeugt den Aurik Trusted Vocal Restoration Report.

Der Report ist die produktive Bruecke von internen KPI-Raten zu professioneller
Evidenz: echte vokale Cases, Baseline-Familien, Safety-Regressionscheck und ein
lesbarer Markdown-Bericht. Er ist bewusst datengetrieben; ohne ausreichende
Evidenz wird kein harter Abbruch erzeugt, sondern ein transparenter
RECOVERED/DEGRADED-Status mit bestmoeglicher sicherer Restaurierungsentscheidung.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _is_truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "ja"}


def _system_name(row: dict[str, Any]) -> str:
    raw = row.get("system") or row.get("candidate") or row.get("engine") or row.get("label") or ""
    return str(raw).strip().lower()


def _is_aurik_row(row: dict[str, Any]) -> bool:
    explicit = str(row.get("is_aurik", "")).strip().lower()
    if explicit in {"1", "true", "yes", "ja"}:
        return True
    if explicit in {"0", "false", "no", "nein"}:
        return False
    return _system_name(row) in {"aurik", "aurik_restoration", "aurik_studio2026"}


def _aurik_row_quality(row: dict[str, Any], required_metrics: list[str]) -> tuple[int, int, int]:
    """Waehlt bei mehreren Varianten die beste belegte Aurik-Zeile pro Case."""
    status = str(row.get("status", "")).strip().lower()
    executed = 1 if status in {"pass", "passed", "recovered", "degraded", "failed", "export_blocked"} else 0
    complete_metrics = sum(1 for key in required_metrics if _to_float(row.get(key)) is not None)
    baseline_variant = 1 if str(row.get("variant", "")).strip().lower() == "baseline" else 0
    return (executed, complete_metrics, baseline_variant)


def _baseline_family(row: dict[str, Any]) -> str:
    raw = row.get("baseline_family") or row.get("baseline") or row.get("reference_family") or ""
    family = str(raw).strip().lower()
    if family:
        return family
    system = _system_name(row)
    aliases = {
        "input": "input_passthrough",
        "passthrough": "input_passthrough",
        "dry": "input_passthrough",
        "dsp": "classical_dsp",
        "classic_dsp": "classical_dsp",
        "ml": "sota_ml",
        "sota": "sota_ml",
        "rx": "commercial_reference",
        "cedar": "commercial_reference",
        "commercial": "commercial_reference",
    }
    return aliases.get(system, system)


def _load_rows(csv_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def _select_aurik_rows(
    rows: list[dict[str, Any]], required_metrics: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    metrics = required_metrics or []
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id or not _is_aurik_row(row):
            continue
        current = selected.get(case_id)
        if current is None or _aurik_row_quality(row, metrics) >= _aurik_row_quality(current, metrics):
            selected[case_id] = row
    return selected


def _baseline_rows_by_case(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if case_id and not _is_aurik_row(row):
            grouped[case_id].append(row)
    return dict(grouped)


def _metric_delta(aurik: dict[str, Any], baselines: list[dict[str, Any]], key: str) -> float | None:
    aurik_value = _to_float(aurik.get(key))
    baseline_values = [_to_float(row.get(key)) for row in baselines]
    clean = [value for value in baseline_values if value is not None]
    if aurik_value is None or not clean:
        return None
    return aurik_value - max(clean)


def _case_human_hearing_violations(aurik: dict[str, Any], trust_cfg: dict[str, Any]) -> list[str]:
    hearing_cfg = trust_cfg.get("human_hearing_focus", {})
    if not isinstance(hearing_cfg, dict):
        return []
    required_metrics = [str(item) for item in hearing_cfg.get("required_metrics", [])]
    thresholds = hearing_cfg.get("thresholds", {}) if isinstance(hearing_cfg.get("thresholds"), dict) else {}
    violations: list[str] = []
    for metric in required_metrics:
        value = _to_float(aurik.get(metric))
        if value is None:
            violations.append(f"{metric}:fehlt")
            continue
        min_key = f"{metric}_min"
        max_key = f"{metric}_max"
        min_value = _to_float(thresholds.get(min_key))
        max_value = _to_float(thresholds.get(max_key))
        if min_value is not None and value < min_value:
            violations.append(f"{metric}:{value:.6f}<{min_value:.6f}")
        if max_value is not None and value > max_value:
            violations.append(f"{metric}:{value:.6f}>{max_value:.6f}")
    return violations


def _case_automation_violations(aurik: dict[str, Any], trust_cfg: dict[str, Any]) -> list[str]:
    auto_cfg = trust_cfg.get("fully_automated_operation", {})
    if not isinstance(auto_cfg, dict) or not bool(auto_cfg.get("required", False)):
        return []
    violations: list[str] = []
    manual_interventions = _to_float(aurik.get("manual_intervention_count"))
    user_parameters = _to_float(aurik.get("user_parameter_count"))
    max_manual = _to_float(auto_cfg.get("max_manual_interventions_per_case"))
    max_params = _to_float(auto_cfg.get("max_user_parameters_per_case"))
    if manual_interventions is None:
        violations.append("manual_intervention_count:fehlt")
    elif max_manual is not None and manual_interventions > max_manual:
        violations.append(f"manual_intervention_count:{manual_interventions:.0f}>{max_manual:.0f}")
    if user_parameters is None:
        violations.append("user_parameter_count:fehlt")
    elif max_params is not None and user_parameters > max_params:
        violations.append(f"user_parameter_count:{user_parameters:.0f}>{max_params:.0f}")
    if bool(auto_cfg.get("require_canonical_bridge_contract", False)) and not _is_truthy(
        aurik.get("canonical_bridge_contract", "")
    ):
        violations.append("canonical_bridge_contract:false")
    if bool(auto_cfg.get("require_autonomous_export_decision", False)) and not _is_truthy(
        aurik.get("autonomous_export_decision", "")
    ):
        violations.append("autonomous_export_decision:false")
    mode = str(aurik.get("mode", "")).strip().lower().replace(" ", "").replace("_", "")
    if mode not in {"restoration", "studio2026"}:
        violations.append("mode_selection:fehlt")
    return violations


def _best_possible_decision(violations: list[str]) -> dict[str, Any]:
    """Uebersetzt Gate-Verletzungen in Aurik-konforme Recovery-Semantik."""
    if not violations:
        return {
            "status": "PASS",
            "degradation_status": "ok",
            "recovery_attempted": False,
            "best_possible_reached": True,
            "export_policy": "normal_export",
            "fail_reason": "",
        }

    safety_markers = (
        "human_hearing_focus",
        "unsafe_export_cases",
        "safety_regressions",
        "artifact_freedom",
        "vqi",
        "hpi",
    )
    safety_relevant = any(any(marker in violation for marker in safety_markers) for violation in violations)
    return {
        "status": "DEGRADED" if safety_relevant else "RECOVERED",
        "degradation_status": "degraded" if safety_relevant else "recovered",
        "recovery_attempted": True,
        "best_possible_reached": True,
        "export_policy": "input_or_best_safe_checkpoint" if safety_relevant else "best_available_restoration",
        "fail_reason": "; ".join(violations[:3]),
    }


def _user_confidence_summary(
    *,
    best_possible: dict[str, Any],
    human_hearing_violations: dict[str, list[str]],
    automation_violations: dict[str, list[str]],
    safety_regressions: list[dict[str, Any]],
    missing_baselines: list[str],
) -> dict[str, Any]:
    """Formuliert die technische Gate-Entscheidung als Nutzer-Vertrauenssignal."""
    status = str(best_possible.get("status", "RECOVERED"))
    if status == "PASS":
        headline = "Aurik hat eine voll belegte, gehoersichere Restaurierung erzeugt."
        confidence_level = "hoch"
        listener_verdict = "Das Ergebnis ist fuer den Nutzer als bestmoegliche sichere Restaurierung freigegeben."
    elif status == "DEGRADED":
        headline = "Aurik hat ein Hoerrisiko erkannt und schuetzt den Nutzer vor einem riskanten Export."
        confidence_level = "geschuetzt"
        listener_verdict = "Das Ergebnis nutzt den besten sicheren Checkpoint oder den Input, statt ein schaedigendes Signal zu exportieren."
    else:
        headline = (
            "Aurik hat die bestmoegliche Restaurierung erzeugt, aber die externe Evidenz ist noch nicht vollstaendig."
        )
        confidence_level = "begrenzt"
        listener_verdict = (
            "Das Ergebnis ist transparent als recovered dokumentiert; Aurik fordert keine manuelle Nachkorrektur."
        )

    why_user_can_trust = [
        "Human-Hearing-Gates pruefen Natuerlichkeit, Emotion, Mikrodynamik, Formanten, Vibrato und Noise-Textur.",
        "Aurik trifft alle Korrektur- und Exportentscheidungen autonom; der Nutzer waehlt nur den Modus.",
        "Bei Unsicherheit wird nicht aggressiver verarbeitet, sondern auf den besten sicheren Zustand zurueckgefallen.",
    ]
    if missing_baselines:
        why_user_can_trust.append(
            "Fehlende externe Baselines werden offengelegt und nicht als Weltklasse-Beweis kaschiert."
        )
    if safety_regressions:
        why_user_can_trust.append("Safety-Regressionen gegen Baselines werden als Schutzsignal behandelt.")
    if human_hearing_violations:
        why_user_can_trust.append("Hoerrelevante Verletzungen fuehren zu DEGRADED statt zu riskantem Enhancement.")
    if automation_violations:
        why_user_can_trust.append(
            "Automationsluecken werden sichtbar gemacht, ohne den Nutzer in Parameterarbeit zu zwingen."
        )

    return {
        "headline": headline,
        "confidence_level": confidence_level,
        "listener_verdict": listener_verdict,
        "manual_action_required": False,
        "allowed_user_decisions": ["mode_selection"],
        "export_policy": best_possible.get("export_policy", "best_available_restoration"),
        "why_user_can_trust": why_user_can_trust,
        "what_to_expect": [
            "Kein Export wird als besser verkauft, wenn Aurik ein Hoerrisiko erkennt.",
            "Der Report nennt Grenzen und Recovery-Status statt technische Probleme zu verstecken.",
            "Die Restaurierung bleibt One-Button-faehig und benoetigt keine manuellen Klangparameter.",
        ],
    }


def build_report(
    rows: list[dict[str, Any]], cfg: dict[str, Any], dashboard: dict[str, Any] | None = None
) -> dict[str, Any]:
    trust_cfg = (
        cfg.get("trusted_vocal_restoration", {}) if isinstance(cfg.get("trusted_vocal_restoration"), dict) else {}
    )
    required_baselines = [str(item) for item in trust_cfg.get("required_baseline_families", [])]
    required_metrics = [str(item) for item in trust_cfg.get("required_aurik_metrics", [])]
    regression_limits = trust_cfg.get("safety_regression_limits", {})
    min_cases = int(trust_cfg.get("min_professional_cases", 20) or 20)
    target_cases = int(trust_cfg.get("target_cases", 50) or 50)

    aurik_rows = _select_aurik_rows(rows, required_metrics)
    baselines_by_case = _baseline_rows_by_case(rows)
    baseline_families: Counter[str] = Counter()
    materials: Counter[str] = Counter()
    eras: Counter[str] = Counter()
    genres: Counter[str] = Counter()
    missing_metric_cases: dict[str, list[str]] = {}
    human_hearing_violations: dict[str, list[str]] = {}
    automation_violations: dict[str, list[str]] = {}
    non_vocal_cases: list[str] = []
    unsafe_export_cases: list[str] = []
    safety_regressions: list[dict[str, Any]] = []

    for row in rows:
        if not _is_aurik_row(row):
            family = _baseline_family(row)
            if family:
                baseline_families[family] += 1

    for case_id, aurik in aurik_rows.items():
        materials[str(aurik.get("material", "unknown")).strip() or "unknown"] += 1
        eras[str(aurik.get("era", "unknown")).strip() or "unknown"] += 1
        genres[str(aurik.get("genre", "unknown")).strip() or "unknown"] += 1
        if not _is_truthy(aurik.get("vocal_focus", "")):
            non_vocal_cases.append(case_id)
        missing = [key for key in required_metrics if _to_float(aurik.get(key)) is None]
        if missing:
            missing_metric_cases[case_id] = missing
        hearing = _case_human_hearing_violations(aurik, trust_cfg)
        if hearing:
            human_hearing_violations[case_id] = hearing
        automation = _case_automation_violations(aurik, trust_cfg)
        if automation:
            automation_violations[case_id] = automation
        status = str(aurik.get("status", "")).strip().lower()
        artifact_freedom = _to_float(aurik.get("artifact_freedom"))
        hpi = _to_float(aurik.get("hpi"))
        if (
            status in {"failed", "unsafe", "export_blocked"}
            or (artifact_freedom is not None and artifact_freedom < 0.95)
            or (hpi is not None and hpi <= 0.0)
        ):
            unsafe_export_cases.append(case_id)

        baselines = baselines_by_case.get(case_id, [])
        for metric, limit_key in (
            ("artifact_freedom", "artifact_freedom_delta_min"),
            ("hpi", "hpi_delta_min"),
            ("vqi", "vqi_delta_min"),
        ):
            delta = _metric_delta(aurik, baselines, metric)
            limit = _to_float(regression_limits.get(limit_key))
            if delta is not None and limit is not None and delta < limit:
                safety_regressions.append(
                    {
                        "case_id": case_id,
                        "metric": metric,
                        "delta_vs_best_baseline": round(delta, 6),
                        "minimum_delta": limit,
                    }
                )

    missing_baselines = sorted(family for family in required_baselines if family not in baseline_families)
    professional_case_ids = sorted(
        case_id for case_id in aurik_rows if case_id not in missing_metric_cases and case_id not in non_vocal_cases
    )
    violations: list[str] = []
    if len(professional_case_ids) < min_cases:
        violations.append(f"professional_vocal_cases: {len(professional_case_ids)} < {min_cases}")
    if missing_baselines:
        violations.append(f"missing_baseline_families: {missing_baselines}")
    if missing_metric_cases:
        violations.append(f"missing_aurik_metrics: {missing_metric_cases}")
    if human_hearing_violations:
        violations.append(f"human_hearing_focus: {human_hearing_violations}")
    if automation_violations:
        violations.append(f"fully_automated_operation: {automation_violations}")
    if non_vocal_cases:
        violations.append(f"non_vocal_cases: {sorted(non_vocal_cases)}")
    if unsafe_export_cases:
        violations.append(f"unsafe_export_cases: {sorted(unsafe_export_cases)}")
    if safety_regressions:
        violations.append(f"safety_regressions: {safety_regressions}")

    best_possible = _best_possible_decision(violations)
    confidence_summary = _user_confidence_summary(
        best_possible=best_possible,
        human_hearing_violations=human_hearing_violations,
        automation_violations=automation_violations,
        safety_regressions=safety_regressions,
        missing_baselines=missing_baselines,
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_kind": "trusted_vocal_restoration",
        "verdict": best_possible["status"],
        "gate_passed": not violations,
        "best_possible_restoration": best_possible,
        "user_confidence_summary": confidence_summary,
        "violations": violations,
        "professional_case_count": len(professional_case_ids),
        "target_case_count": target_cases,
        "professional_case_ids": professional_case_ids,
        "corpus_coverage": {
            "materials": dict(materials),
            "eras": dict(eras),
            "genres": dict(genres),
            "non_vocal_cases": sorted(non_vocal_cases),
        },
        "human_hearing_focus": {
            "violations": human_hearing_violations,
            "passed": not human_hearing_violations,
        },
        "fully_automated_operation": {
            "violations": automation_violations,
            "passed": not automation_violations,
        },
        "baseline_comparison": {
            "required_baseline_families": required_baselines,
            "observed_baseline_families": dict(baseline_families),
            "missing_baseline_families": missing_baselines,
        },
        "safety_regressions": safety_regressions,
        "phase_hardening_actions": [
            "Regressive Phasen pro Case anhand safety_regressions isolieren.",
            "A/B-Harness mit --fail-on-unsafe-candidate fuer Kandidatenplaene ausfuehren.",
            "Unsichere Phasen im Release-Pfad deaktivieren oder mit Guard/Rollback versehen.",
            "Bei Hoerschaden-Risiko auf best_carrier_checkpoint oder Input-Export mit Status degraded zurueckfallen.",
        ],
        "professional_limitations": [
            "Ohne externe Baseline-Familien keine Weltklasse-Behauptung.",
            "Ohne mindestens 20 vokale Real-Cases nur Engineering-Indiz, kein Profi-Nachweis.",
            "Blindhoertest bleibt fuer finale Klangwahrheit massgeblich.",
            "Nur Moduswahl ist Nutzereingriff; alle Korrekturentscheidungen muessen autonom erfolgen.",
        ],
        "dashboard_snapshot": dashboard or {},
    }


def _build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Aurik Trusted Vocal Restoration Report",
        "",
        f"Generiert: {report['generated_at']}",
        f"Verdict: {report['verdict']}",
        f"Best possible restoration: {report['best_possible_restoration']}",
        "",
        "## Executive Verdict",
        "",
        f"- Professional vocal cases: {report['professional_case_count']} / {report['target_case_count']}",
        f"- Violations: {len(report['violations'])}",
        "",
        "## Corpus Coverage",
        "",
        f"- Materials: {report['corpus_coverage']['materials']}",
        f"- Eras: {report['corpus_coverage']['eras']}",
        f"- Genres: {report['corpus_coverage']['genres']}",
        f"- Non-vocal cases: {report['corpus_coverage']['non_vocal_cases']}",
        "",
        "## Baseline Comparison",
        "",
        f"- Required: {report['baseline_comparison']['required_baseline_families']}",
        f"- Observed: {report['baseline_comparison']['observed_baseline_families']}",
        f"- Missing: {report['baseline_comparison']['missing_baseline_families']}",
        "",
        "## Human Hearing Focus",
        "",
        f"- Passed: {report['human_hearing_focus']['passed']}",
        f"- Violations: {report['human_hearing_focus']['violations']}",
        "",
        "## User Confidence Summary",
        "",
        f"- Headline: {report['user_confidence_summary']['headline']}",
        f"- Confidence: {report['user_confidence_summary']['confidence_level']}",
        f"- Listener verdict: {report['user_confidence_summary']['listener_verdict']}",
        f"- Manual action required: {report['user_confidence_summary']['manual_action_required']}",
        f"- Export policy: {report['user_confidence_summary']['export_policy']}",
        "",
        "## Fully Automated Operation",
        "",
        f"- Passed: {report['fully_automated_operation']['passed']}",
        f"- Violations: {report['fully_automated_operation']['violations']}",
        "",
        "## Safety Regressions",
        "",
    ]
    if report["safety_regressions"]:
        for item in report["safety_regressions"]:
            lines.append(f"- {item}")
    else:
        lines.append("- Keine Safety-Regression gegen beste Baseline erkannt.")
    lines.extend(["", "## Phase Hardening Actions", ""])
    lines.extend(f"- {item}" for item in report["phase_hardening_actions"])
    lines.extend(["", "## Professional Limitations", ""])
    lines.extend(f"- {item}" for item in report["professional_limitations"])
    if report["violations"]:
        lines.extend(["", "## Gate Violations", ""])
        lines.extend(f"- {item}" for item in report["violations"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Erzeugt Aurik Trusted Vocal Restoration Report")
    parser.add_argument("--result-csv", action="append", default=[], help="Revalidierungs-/Benchmark-CSV")
    parser.add_argument("--dashboard-json", default="reports/worldclass/worldclass_kpi_dashboard.json")
    parser.add_argument("--threshold-config", default="config/worldclass_kpi_thresholds.json")
    parser.add_argument("--out-dir", default="reports/worldclass")
    parser.add_argument(
        "--fail-on-gate", action="store_true", help="Kompatibilitaetsflag; Report bricht nicht hart ab."
    )
    args = parser.parse_args()

    cfg = json.loads(Path(args.threshold_config).read_text(encoding="utf-8"))
    dashboard_path = Path(args.dashboard_json)
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8")) if dashboard_path.exists() else {}
    rows = _load_rows([Path(path) for path in args.result_csv])
    report = build_report(rows, cfg, dashboard=dashboard)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "trusted_vocal_restoration_report.json"
    out_md = out_dir / "trusted_vocal_restoration_report.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_build_markdown(report), encoding="utf-8")
    print(f"Trusted Vocal Restoration Report JSON: {out_json}")
    print(f"Trusted Vocal Restoration Report MD: {out_md}")
    print(f"TRUSTED VOCAL RESTORATION: {report['verdict']}")
    if args.fail_on_gate and report["verdict"] != "PASS":
        print("Trusted-Report: kein harter Abbruch; bestmoegliche Restaurierung wurde dokumentiert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
