#!/usr/bin/env python3
"""Release-Report je Version — „keine Zeile ohne Beleg“ (Pfeiler 8 der SOTA-Roadmap).

Der Report manifestiert die Qualität jedes Releases aus messbaren Quellen:
Version & Provenienz, Artefakt-Fingerabdrücke, Release-Kandidaten-Metriken
(``scripts/release_candidate_pipeline.py``), Hör-Verdikte (``scripts/ab_harness.py``),
Fallback-Quote (B1 des Wohlklang-Vertrags) — und listet jede **Provenienz-Lücke**
offen auf. Jede Aussagezeile trägt ihre Quelle; eine Zeile ohne Quelle ist ein
Defekt (Gate-Test ``test_release_report.py``).

Aufruf (deterministisch, §G5 (copilot-instructions.md)):
    python3 -B scripts/release_report.py --version 10.13.0 \\
        [--candidates-report docs/reports/current/<datei>.json] \\
        [--hoerverdikte docs/reports/current/<datei>_ab.json] \\
        [--out docs/reports/current/<datei>_release_report.md]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_QU = "  (Quelle: {q})"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quelle_line(text: str, quelle: str) -> str:
    return text + _QU.format(q=quelle)


def _git_provenienz() -> tuple[list[str], list[str]]:
    """(zeilen, luecken) — Git-Stände mit Quelle; jede fehlende Angabe = Lücke."""
    zeilen: list[str] = []
    luecken: list[str] = []
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=30,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=30,
            check=True,
        ).stdout.strip()
        zeilen.append(
            _quelle_line(f"- Git-Revision: `{sha}`" + (" (unsauber)" if dirty else " (sauber)"), "git rev-parse/status")
        )
    except Exception as exc:
        luecken.append(f"Git-Provenienz nicht bestimmbar: {exc}")
    return zeilen, luecken


def _version_zeile(version: str) -> tuple[list[str], list[str]]:
    zeilen: list[str] = []
    luecken: list[str] = []
    version_py = PROJECT_ROOT / "backend" / "core" / "version.py"
    try:
        quelle = version_py.read_text(encoding="utf-8")
        if f'"{version}"' in quelle or f"'{version}'" in quelle:
            zeilen.append(_quelle_line(f"- Version: **{version}**", "backend/core/version.py"))
        else:
            luecken.append(f"Version {version} steht nicht in backend/core/version.py (Single Source verletzt)")
    except Exception as exc:
        luecken.append(f"version.py nicht lesbar: {exc}")
    return zeilen, luecken


def _fingerabdruck_zeilen(_arg: object = None) -> tuple[list[str], list[str]]:
    registry = PROJECT_ROOT / ".github" / "ML_ARTIFACT_FINGERPRINTS.md"
    if not registry.is_file():
        return [], [".github/ML_ARTIFACT_FINGERPRINTS.md fehlt (Artefakt-Herkunft §III.13 (copilot-instructions.md))"]
    n = sum(
        1
        for line in registry.read_text(encoding="utf-8").splitlines()
        if line.startswith("| ") and line.count("|") >= 4
    )
    zeilen = [
        _quelle_line(
            f"- Artefakt-Registry: {n} Zeilen, sha256 `{_sha256(registry)[:16]}…`",
            ".github/ML_ARTIFACT_FINGERPRINTS.md",
        )
    ]
    return zeilen, []


def _kandidaten_zeilen(report: Path | None) -> tuple[list[str], list[str]]:
    if report is None:
        return [], ["Release-Kandidaten-Metrik nicht belegt (scripts/release_candidate_pipeline.py nicht ausgeführt)"]
    if not report.is_file():
        return [], [f"Kandidaten-Report fehlt: {report}"]
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [f"Kandidaten-Report unlesbar: {exc}"]
    zeilen: list[str] = []
    luecken: list[str] = []
    quelle = str(report.relative_to(PROJECT_ROOT)) if report.is_relative_to(PROJECT_ROOT) else str(report)
    kands = data.get("kandidaten") or ([{"flag": "unbekannt", "summary": data}] if "rows" in data else [])
    if not kands:
        return [], [f"unerwartetes Schema im Kandidaten-Report: {quelle}"]
    zeilen.append(_quelle_line(f"- Kandidaten: {len(kands)} Flag(s) gemessen", quelle))
    for kand in kands:
        flag = kand.get("flag", "?")
        summ = kand.get("summary", {}) or {}
        fb = summ.get("fallback_count")
        gate = summ.get("flag_freigabe")
        zeilen.append(
            _quelle_line(
                f"- `{flag}`: fallback_count={fb if fb is not None else 'nicht im Report'}, "
                f"flag_freigabe={gate if gate else 'nicht im Report'}",
                quelle,
            )
        )
        if fb is None:
            luecken.append(f"fallback_count fehlt für `{flag}` (Report ohne Pfeiler-5-Zählung)")
    return zeilen, luecken


def _hoer_zeilen(verdikte: Path | None) -> tuple[list[str], list[str]]:
    if verdikte is None:
        return [], ["Hör-Verdikte nicht belegt (scripts/ab_harness.py collect nicht ausgeführt)"]
    if not verdikte.is_file():
        return [], [f"Hör-Verdikt-Report fehlt: {verdikte}"]
    try:
        data = json.loads(verdikte.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [f"Hör-Verdikt-Report unlesbar: {exc}"]
    quelle = str(verdikte.relative_to(PROJECT_ROOT)) if verdikte.is_relative_to(PROJECT_ROOT) else str(verdikte)
    stat = data.get("statistik", {})
    zeilen = [
        _quelle_line(
            f"- Hör-Bild: {stat.get('n_entscheidend', '?')} gültige Urteile, "
            f"Siegquote A {stat.get('siegquote_a', '?')}, p = {stat.get('p_wert_vorzeichen_test', '?')}",
            quelle,
        ),
        _quelle_line(f"- Statistisch signifikant (p < 0.05): {stat.get('signifikant', '?')}", quelle),
    ]
    return zeilen, []


def build_report(
    version: str,
    *,
    candidates_report: Path | None = None,
    hoerverdikte: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """(Markdown, JSON) — jede Aussagezeile trägt ihre Quelle; Lücken explizit."""
    zeilen: list[str] = [
        f"# Release-Report — Aurik {version}",
        "",
        "> „Keine Zeile ohne Beleg“ (Pfeiler 8): Jede Aussagezeile trägt ihre Quelle;",
        "> fehlende Quellen stehen unter **Provenienz-Lücken**.",
        "",
    ]
    luecken: list[str] = []

    for title, block in [
        ("Version & Provenienz", _version_zeile(version)),
        ("Git", _git_provenienz()),
        ("Modelle & Fingerabdrücke", _fingerabdruck_zeilen()),
        ("Release-Kandidaten & Fallback-Quote", _kandidaten_zeilen(candidates_report)),
        ("Hör-Verdikte (Hörordnung entscheidet)", _hoer_zeilen(hoerverdikte)),
    ]:
        zz, ll = block
        luecken.extend(ll)
        zeilen.append(f"## {title}")
        zeilen.extend(zz or ["- nicht belegt"])
        zeilen.append("")

    zeilen.append("## Provenienz-Lücken")
    zeilen.extend(f"- {g}" for g in luecken or ["- keine"])
    zeilen.append("")

    md = "\n".join(zeilen)
    payload: dict[str, Any] = {
        "version": version,
        "zeilen": [z for z in zeilen if z.startswith("- ") or z.startswith("| ")],
        "provenienz_luecken": luecken,
    }
    return md, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Release-Report je Version (keine Zeile ohne Beleg)")
    parser.add_argument("--version", required=True, help="Release-Version (muß in backend/core/version.py stehen)")
    parser.add_argument("--candidates-report", type=Path, default=None)
    parser.add_argument("--hoerverdikte", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Markdown-Ziel; JSON-Seitenkarren daneben")
    args = parser.parse_args(argv)

    md, payload = build_report(
        args.version,
        candidates_report=args.candidates_report,
        hoerverdikte=args.hoerverdikte,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Release-Report geschrieben: {args.out}", flush=True)
        print(f"Provenienz-Lücken: {len(payload['provenienz_luecken'])}", flush=True)
    else:
        print(md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
