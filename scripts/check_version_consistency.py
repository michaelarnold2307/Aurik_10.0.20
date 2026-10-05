#!/usr/bin/env python3
"""Version-Consistency-Check — Sprint D. Spec v10.700 F1.

Prüft, dass die Versionsangaben in pyproject.toml, README.md und CHANGELOG.md mit
der Single Source of Truth `backend/core/version.py` übereinstimmen.
(2026-10-05: vorher war pyproject.toml kanonisch — dadurch blieb eine Drift von
zwei Patch-Ständen unbemerkt: version.py 10.3.4, pyproject 10.3.0, README 10.2.0.)

CI-Gate: Exit 0 = konsistent, Exit 1 = Inkonsistenz.

Usage:
    python scripts/check_version_consistency.py [--fix]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CORE_FILES = ["pyproject.toml", "README.md", "CHANGELOG.md"]


def extract_version(filepath: Path) -> str | None:
    """Extrahiert die Version aus einer Datei."""
    with open(filepath) as f:
        content = f.read()
    # version.py: __version__ = "10.0.18" | pyproject.toml: version = "10.0.18"
    m = re.search(r'(?:__)?version(?:__)?\s*=\s*"(\d+\.\d+\.\d+)"', content)
    if m:
        return m.group(1)
    # README.md/CHANGELOG.md: **Version:** 10.0.18 or ## 10.0.18 (...)
    m = re.search(r"(?:\*\*Version:?\*\*|##)\s*(\d+\.\d+\.\d+)", content)
    if m:
        return m.group(1)
    return None


def main():
    fix = "--fix" in sys.argv[1:]
    # Kanonische Quelle (2026-10-05): backend/core/version.py — dokumentierte
    # "Single source of truth". pyproject.toml war zuvor kanonisch und driftete
    # unbemerkt zwei Patch-Stände hinterher.
    canonical = extract_version(PROJECT_ROOT / "backend" / "core" / "version.py")
    if not canonical:
        print("❌ Kanonische Version nicht in backend/core/version.py gefunden")
        sys.exit(1)

    print(f"Kanonische Version: {canonical}\n")

    ok = True
    for fname in CORE_FILES:
        fpath = PROJECT_ROOT / fname
        if not fpath.exists():
            continue
        old = extract_version(fpath)
        if old is None:
            print(f"  ❌ {fname}: Keine Version gefunden")
            ok = False
            continue
        if old == canonical:
            print(f"  ✅ {fname}: {old}")
            continue
        if not fix:
            print(f"  ❌ {fname}: {old} (erwartet {canonical})")
            ok = False
            continue
        text = fpath.read_text(encoding="utf-8")
        if fname == "CHANGELOG.md":
            # Der Check liest die erste ##-Abschnittsüberschrift — genau die wird
            # gesetzt; historische Abschnitte bleiben unverändert.
            new_text, n = re.subn(rf"^## {re.escape(old)}", f"## {canonical}", text, count=1, flags=re.MULTILINE)
        else:
            # Nur das erste Vorkommen (z. B. das **Version:**-Feld), Historie bleibt.
            new_text, n = re.subn(re.escape(old), canonical, text, count=1)
        if n == 0:
            print(f"  ❌ {fname}: {old} konnte nicht ersetzt werden")
            ok = False
            continue
        fpath.write_text(new_text, encoding="utf-8")
        print(f"  🔧 {fname}: {old} -> {canonical}")

    if ok:
        print(f"\n✅ Alle {len(CORE_FILES)} Dateien konsistent: {canonical}")
        sys.exit(0)
    print(f"\n❌ Inkonsistenz gefunden. Kanonisch: backend/core/version.py = {canonical}")
    if not fix:
        print("   Führe 'python scripts/check_version_consistency.py --fix' aus")
    sys.exit(1)


if __name__ == "__main__":
    main()
