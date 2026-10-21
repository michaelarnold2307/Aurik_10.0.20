"""Aggregiert Klasse-C-Revalidierungsresultate zu einem auditfaehigen Bericht.

Eingang:
- result_template.csv (mit ausgefuellten Metriken)

Ausgang:
- summary.json
- summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median


def _to_float(value: str) -> float | None:
    if value is None:
        return None
    txt = str(value).strip()
    if not txt:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _to_bool(value: str) -> bool:
    txt = str(value).strip().lower()
    return txt in {"1", "true", "yes", "ja"}


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _calc_group_stats(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = f"{row.get('workpackage', '?')}::{row.get('variant', '?')}"
        groups[key].append(row)

    result: dict[str, dict[str, object]] = {}
    for key, grows in groups.items():
        hpi_vals: list[float] = []
        for r in grows:
            val = _to_float(r.get("hpi", ""))
            if val is not None:
                hpi_vals.append(val)

        afg_vals: list[float] = []
        for r in grows:
            val = _to_float(r.get("artifact_freedom", ""))
            if val is not None:
                afg_vals.append(val)

        vqi_vals: list[float] = []
        for r in grows:
            val = _to_float(r.get("vqi", ""))
            if val is not None:
                vqi_vals.append(val)

        statuses = Counter((r.get("status", "") or "").strip().lower() for r in grows)
        fail_reasons = Counter(
            (r.get("fail_reason", "") or "").strip() for r in grows if (r.get("fail_reason", "") or "").strip()
        )

        result[key] = {
            "num_rows": len(grows),
            "num_with_hpi": len(hpi_vals),
            "median_hpi": median(hpi_vals) if hpi_vals else None,
            "median_artifact_freedom": median(afg_vals) if afg_vals else None,
            "median_vqi": median(vqi_vals) if vqi_vals else None,
            "status_counts": dict(statuses),
            "top_fail_reasons": fail_reasons.most_common(5),
        }
    return result


def _recommend_variant(group_stats: dict[str, dict[str, object]]) -> dict[str, str]:
    """Waehlt pro Workpackage den robustesten Kandidaten anhand Median-HPI + AFG-Sicherheit."""
    by_wp: dict[str, list[tuple[str, dict[str, object]]]] = defaultdict(list)
    for key, stats in group_stats.items():
        wp, variant = key.split("::", 1)
        by_wp[wp].append((variant, stats))

    recs: dict[str, str] = {}
    for wp, items in by_wp.items():
        best_variant = "keine_bewertbare_variante"
        best_score = -1e9
        for variant, stats in items:
            med_hpi = stats.get("median_hpi")
            med_afg = stats.get("median_artifact_freedom")
            num_hpi_raw = stats.get("num_with_hpi")
            num_hpi = int(num_hpi_raw) if isinstance(num_hpi_raw, int) else 0

            if not isinstance(med_hpi, (int, float)) or num_hpi == 0:
                continue

            # AFG-Veto orientiert an Projektregel: < 0.95 ist kritisch
            afg_penalty = 0.0
            if isinstance(med_afg, (int, float)) and med_afg < 0.95:
                afg_penalty = 0.2

            score = float(med_hpi) - afg_penalty
            if score > best_score:
                best_score = score
                best_variant = variant

        recs[wp] = best_variant
    return recs


def _build_markdown(
    run_name: str,
    rows: list[dict[str, str]],
    group_stats: dict[str, dict[str, object]],
    recs: dict[str, str],
) -> str:
    fail_counter = Counter(
        (r.get("fail_reason", "") or "").strip() for r in rows if (r.get("fail_reason", "") or "").strip()
    )
    statuses = Counter((r.get("status", "") or "").strip().lower() for r in rows)

    lines: list[str] = []
    lines.append("# Klasse-C Revalidierung Summary")
    lines.append("")
    lines.append(f"Run: {run_name}")
    lines.append(f"Generiert: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Gesamt")
    lines.append("")
    lines.append(f"- Zeilen gesamt: {len(rows)}")
    lines.append(f"- Statusverteilung: {dict(statuses)}")
    lines.append(f"- Top Fail-Reasons: {fail_counter.most_common(8)}")
    lines.append("")
    lines.append("## Variantenstatistik")
    lines.append("")
    lines.append("| Workpackage/Variante | Zeilen | HPI-Median | AFG-Median | VQI-Median |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")

    for key in sorted(group_stats.keys()):
        st = group_stats[key]
        lines.append(
            "| "
            + key
            + f" | {st.get('num_rows', 0)}"
            + f" | {st.get('median_hpi')}"
            + f" | {st.get('median_artifact_freedom')}"
            + f" | {st.get('median_vqi')} |"
        )

    lines.append("")
    lines.append("## WP-Empfehlung")
    lines.append("")
    if recs:
        for wp in sorted(recs.keys()):
            lines.append(f"- {wp}: {recs[wp]}")
    else:
        lines.append("- Keine bewertbaren Varianten (fehlende HPI-Daten).")

    lines.append("")
    lines.append("## Hinweise")
    lines.append("")
    lines.append(
        "- Empfehlungen sind nur gueltig, wenn die zugrunde liegenden CSV-Metriken vollstaendig "
        "und korrekt eingetragen wurden."
    )
    lines.append("- Varianten mit Median-AFG < 0.95 werden in der Empfehlung abgestraft (Sicherheitsveto).")

    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI-Einstiegspunkt: liest result_template.csv und schreibt summary.json/summary.md."""
    parser = argparse.ArgumentParser(description="Aggregiere Klasse-C-Revalidierungsresultate")
    parser.add_argument("--input-csv", required=True, help="Pfad zu result_template.csv")
    parser.add_argument("--out-dir", default="", help="Ausgabeordner (Default: Ordner der CSV)")
    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else input_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(input_csv)
    # Filter: nur echte Messzeilen mit mindestens einem eingetragenen Ergebnisfeld
    measured_rows = [
        r
        for r in rows
        if any(
            _to_float(r.get(k, "")) is not None
            for k in ["artifact_freedom", "vqi", "mert_similarity", "timbral_fidelity", "hpi"]
        )
        or _to_bool(r.get("status", ""))
        or (r.get("status", "") or "").strip() != ""
    ]

    group_stats = _calc_group_stats(measured_rows)
    recs = _recommend_variant(group_stats)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_csv": str(input_csv),
        "num_rows_total": len(rows),
        "num_rows_measured": len(measured_rows),
        "group_stats": group_stats,
        "recommendations": recs,
    }

    summary_json = out_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    run_name = input_csv.parent.name
    summary_md = out_dir / "summary.md"
    summary_md.write_text(_build_markdown(run_name, measured_rows, group_stats, recs), encoding="utf-8")

    print(f"Summary JSON: {summary_json}")
    print(f"Summary MD: {summary_md}")
    print(f"Messzeilen: {len(measured_rows)} / {len(rows)}")


if __name__ == "__main__":
    main()
