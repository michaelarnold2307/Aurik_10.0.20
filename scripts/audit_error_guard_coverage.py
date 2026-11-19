"""Audit ErrorGuard-Coverage über alle 68 Phasen.

§15.8: Scannt ``backend/core/phases/phase_*.py`` und identifiziert:
  1. Phasen MIT ErrorGuard/guard_error (geschützt)
  2. Phasen MIT @phase_error_guard (neuer Decorator)
  3. Phasen OHNE jeglichen Schutz (ungeschützt)

Ausgabe:
    - Konsolen-Zusammenfassung (Markdown-Tabelle)
    - ``error_guard_gaps.json`` mit detaillierter Liste

Nutzung:
    python scripts/audit_error_guard_coverage.py
    python scripts/audit_error_guard_coverage.py --json-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_PHASES_DIR = _PROJECT_ROOT / "backend" / "core" / "phases"

# ── Muster für ErrorGuard-Erkennung ─────────────────────────────────────────
_GUARD_PATTERNS: list[str] = [
    "guard_error",
    "ErrorGuard",
    "error_guard",
    "phase_error_guard",
    "@phase_error_guard",
    "errors.guard",
    "_safe_process",
    "PhaseInterface",  # Basisklasse mit integriertem _safe_process
]

_NAN_PATTERNS: list[str] = [
    "nan_to_num",
    "isfinite",
    "isnan",
    "isinf",
    "np.isfinite",
    "np.isnan",
    "np.isinf",
    "np.nan_to_num",
]


def scan_phase(filepath: Path) -> dict:
    """Scannt eine einzelne Phase-Datei.

    Returns:
        Dict mit: file, phase_name, has_error_guard, has_nan_check,
                  error_guard_matches, nan_check_matches, line_count.
    """
    content = filepath.read_text(encoding="utf-8", errors="replace")

    guard_matches: list[str] = []
    for pattern in _GUARD_PATTERNS:
        if pattern in content:
            guard_matches.append(pattern)

    nan_matches: list[str] = []
    for pattern in _NAN_PATTERNS:
        if pattern in content:
            nan_matches.append(pattern)

    # Phasen-Name aus Docstring oder Dateiname
    phase_name = filepath.stem.replace("phase", "Phase ").replace("_", " ").title()

    return {
        "file": str(filepath.relative_to(_PROJECT_ROOT)),
        "phase_name": phase_name,
        "has_error_guard": len(guard_matches) > 0,
        "has_nan_check": len(nan_matches) > 0,
        "error_guard_matches": guard_matches,
        "nan_check_matches": nan_matches[:5],  # max 5
        "line_count": len(content.splitlines()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit ErrorGuard-Coverage")
    parser.add_argument("--json-only", action="store_true", help="Nur JSON ausgeben")
    parser.add_argument("--output", default="error_guard_gaps.json", help="JSON-Ausgabedatei")
    args = parser.parse_args()

    if not _PHASES_DIR.is_dir():
        print(f"❌ Phasen-Verzeichnis nicht gefunden: {_PHASES_DIR}", file=sys.stderr)
        return 1

    # ── Alle Phasen scannen ───────────────────────────────────────────────
    phase_files = sorted(_PHASES_DIR.glob("phase_*.py"))
    results: list[dict] = []

    for fp in phase_files:
        results.append(scan_phase(fp))

    # ── Statistiken ───────────────────────────────────────────────────────
    total = len(results)
    guarded = [r for r in results if r["has_error_guard"]]
    nan_checked = [r for r in results if r["has_nan_check"]]
    unprotected = [r for r in results if not r["has_error_guard"]]

    # ── JSON speichern ────────────────────────────────────────────────────
    output_path = _PROJECT_ROOT / args.output
    output_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "total_phases": total,
                    "guarded_phases": len(guarded),
                    "nan_checked_phases": len(nan_checked),
                    "unprotected_phases": len(unprotected),
                    "coverage_pct": round(len(guarded) / max(total, 1) * 100, 1),
                    "audit_date": __import__("datetime").datetime.now().isoformat(),
                },
                "guarded": [r["file"] for r in guarded],
                "nan_checked": [r["file"] for r in nan_checked],
                "unprotected": [
                    {
                        "file": r["file"],
                        "line_count": r["line_count"],
                        "has_nan_check": r["has_nan_check"],
                    }
                    for r in unprotected
                ],
                "details": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if args.json_only:
        print(json.dumps({"ok": True, "total": total, "unprotected": len(unprotected)}))
        return 0

    # ── Konsolen-Ausgabe ───────────────────────────────────────────────────
    sep = "=" * 70
    print(f"\n{sep}")
    print("  Aurik ErrorGuard Coverage Audit")
    print(f"{sep}")
    print(f"  Phasen gesamt:           {total}")
    print(f"  Mit ErrorGuard:          {len(guarded)} ({len(guarded) / max(total, 1) * 100:.1f}%)")
    print(f"  Mit NaN/Inf-Check:       {len(nan_checked)} ({len(nan_checked) / max(total, 1) * 100:.1f}%)")
    print(f"  ❌ UNGESCHÜTZT:           {len(unprotected)} ({len(unprotected) / max(total, 1) * 100:.1f}%)")
    print(f"{sep}\n")

    if unprotected:
        print("## Ungeschützte Phasen (ErrorGuard fehlt)\n")
        print(f"| {'#':>3} | {'Phase':<45} | {'Zeilen':>6} | {'NaN-Check':>9} |")
        print(f"|{'-' * 5}|{'-' * 47}|{'-' * 8}|{'-' * 11}|")
        for i, phase in enumerate(unprotected, 1):
            nan_ok = "✅" if phase["has_nan_check"] else "❌"
            print(f"| {i:>3} | {phase['file']:<45} | {phase['line_count']:>6} | {nan_ok:>9} |")
        print()

    print(f"📄 JSON-Report: {output_path}")
    print("✅ Audit abgeschlossen.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
