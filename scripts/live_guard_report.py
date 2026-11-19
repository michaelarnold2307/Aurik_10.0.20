"""Aurik Live-Guard Incident-Report-Aggregator
==============================================
Liest alle Incident-Verzeichnisse unter reports/live_guard/incidents/
und erstellt eine kompakte Tagesübersicht in reports/live_guard/daily_report.txt.

Verwendung:
    python scripts/live_guard_report.py
    python scripts/live_guard_report.py --incident-dir /pfad/zu/incidents
    python scripts/live_guard_report.py --since 2026-04-12
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


def _read_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


def _detect_root_cause(incident: Path) -> str:
    causes: list[str] = []

    def _check(log: Path, marker: str, label: str) -> None:
        if log.exists() and marker in log.read_text(encoding="utf-8", errors="ignore"):
            causes.append(label)

    _check(incident / "uat_r5_r12.log", "mert_plugin.py", "MERT-Timeout (CPU/RAM)")
    _check(incident / "uat_r5_r12.log", "Timeout (>180.0s)", "Pytest-Timeout 180s")
    _check(incident / "uat_r5_r12.log", "Timeout (>600.0s)", "Pytest-Timeout 600s")
    _check(incident / "uat_r5_r12.log", "phase_24_dropout_repair", "Phase-24-MRSA-Laufzeit")
    _check(incident / "backend_tail.log", "PLM: Pipeline aktiv, aber RAM kritisch", "RAM-Druck (PLM)")
    _check(incident / "backend_tail.log", "§2.48 Rollback", "CIG-Rollback §2.48")
    _check(incident / "backend_tail.log", "OOM-Guard", "OOM-Guard aktiv")
    _check(incident / "runtime_spec_check.log", "FAIL", "Runtime-Spec-Verstoß")
    _check(incident / "compliance_check.log", "VIOLATION", "Code-Compliance-Verletzung")

    # Repair-Attempts
    for attempt_dir in sorted(incident.glob("attempt_*")):
        diag = attempt_dir / "repair_diag.log"
        if diag.exists():
            txt = diag.read_text(encoding="utf-8", errors="ignore")
            if "detected_mert_timeout=1" in txt and "MERT-Timeout (CPU/RAM)" not in causes:
                causes.append("MERT-Timeout (auto-detected im Hook)")

    return ", ".join(causes) if causes else "unbekannt"


def aggregate(incident_dir: Path, since: datetime | None = None) -> list[dict]:
    rows: list[dict] = []
    if not incident_dir.exists():
        return rows

    for d in sorted(incident_dir.iterdir()):
        if not d.is_dir():
            continue

        try:
            ts = datetime.strptime(d.name[:15], "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            ts = datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc)

        if since and ts < since:
            continue

        meta = _read_env(d / "meta.txt")
        status_env = _read_env(d / "status.env")
        status = status_env.get("status", "no_repair_run")

        # Collate attempt summaries
        attempts: list[dict] = []
        for attempt_dir in sorted(d.glob("attempt_*")):
            s = _read_env(attempt_dir / "summary.env")
            attempts.append(
                {
                    "name": attempt_dir.name,
                    "runtime_rc": s.get("runtime_spec_rc", "?"),
                    "compliance_rc": s.get("compliance_rc", "?"),
                    "r10_rc": s.get("uat_r10_rc", "?"),
                    "full_rc": s.get("uat_r5_r12_rc", "?"),
                }
            )

        root_cause = _detect_root_cause(d)

        rows.append(
            {
                "timestamp": ts,
                "incident_id": d.name,
                "status": status,
                "root_cause": root_cause,
                "attempts": attempts,
                "meta_reason": meta.get("reason", ""),
            }
        )

    return rows


def format_report(rows: list[dict], today: str) -> str:
    lines: list[str] = [
        f"Aurik Live-Guard — Tagesübersicht {today}",
        "=" * 60,
        f"Gesamt Incidents: {len(rows)}",
    ]

    recovered = sum(1 for r in rows if r["status"] == "recovered")
    degraded = sum(1 for r in rows if r["status"] == "degraded")
    no_repair = sum(1 for r in rows if r["status"] not in ("recovered", "degraded"))

    lines += [
        f"  recovered : {recovered}",
        f"  degraded  : {degraded}",
        f"  kein Repair-Run: {no_repair}",
        "",
    ]

    if not rows:
        lines.append("Keine Incidents im gewählten Zeitraum.")
    else:
        for r in rows:
            status_sym = {"recovered": "✓", "degraded": "✗"}.get(r["status"], "?")
            lines.append(f"[{status_sym}] {r['timestamp'].strftime('%H:%M:%S')}  {r['incident_id']}")
            lines.append(f"     Status     : {r['status']}")
            lines.append(f"     Root-Cause : {r['root_cause']}")
            if r["meta_reason"]:
                lines.append(f"     Auslöser   : {r['meta_reason']}")
            for a in r["attempts"]:
                ok = all(rc == "0" for rc in (a["runtime_rc"], a["compliance_rc"], a["r10_rc"]))
                sym = "✓" if ok else "✗"
                lines.append(
                    f"     {sym} {a['name']}: runtime={a['runtime_rc']} "
                    f"compliance={a['compliance_rc']} r10={a['r10_rc']} r5-r12={a['full_rc']}"
                )
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aurik Live-Guard Incident-Report-Aggregator")
    parser.add_argument(
        "--incident-dir",
        default=str(Path(__file__).parent.parent / "reports" / "live_guard" / "incidents"),
    )
    parser.add_argument("--since", default=None, help="Nur Incidents ab diesem Datum (YYYY-MM-DD)")
    parser.add_argument("--output", default=None, help="Ausgabedatei (Standard: reports/live_guard/daily_report.txt)")
    args = parser.parse_args()

    incident_dir = Path(args.incident_dir)
    since_dt: datetime | None = None
    if args.since:
        since_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    today = datetime.now().strftime("%Y-%m-%d")
    rows = aggregate(incident_dir, since=since_dt)
    report = format_report(rows, today)

    print(report)

    out_path = Path(args.output) if args.output else incident_dir.parent / "daily_report.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\n→ Bericht gespeichert: {out_path}")


if __name__ == "__main__":
    main()
