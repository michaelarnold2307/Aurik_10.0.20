#!/usr/bin/env python3
"""Plattform-Kompatibilitäts-Check für Aurik.


§15.4: Prüft plattformübergreifende Kompatibilität vor Merge.
- Echte Windows-Hartkodierte Pfade (C:\\...)
- CRLF-Zeilenenden in Projektdateien (nicht models/)
- Case-sensitivity bei Imports
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent

# Verzeichnisse die vom Scan ausgeschlossen sind
_SKIP_PREFIXES = (".venv", "__pycache__", "models/", "temp_repro/", ".git/")


def _should_skip(rel_path: str) -> bool:
    """True wenn Datei/Verzeichnis übersprungen werden soll."""
    return any(prefix in rel_path for prefix in _SKIP_PREFIXES)


def check_path_separators() -> tuple[bool, list[str]]:
    """Prüft auf hartkodierte Windows-Pfade (C:\\Users\\...)."""
    issues: list[str] = []
    for py_file in _PROJECT_ROOT.rglob("*.py"):
        rel = str(py_file.relative_to(_PROJECT_ROOT))
        if _should_skip(rel):
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            logger.debug("Stiller Ersatzpfad dokumentiert (Bug 9/V74)", exc_info=True)
            continue
        for i, line in enumerate(content.splitlines(), 1):
            if re.search(r'["\x27][A-Za-z]:\\\\', line):
                issues.append(f"{rel}:{i}: Hardcoded Windows path: {line.strip()[:80]}")
    return len(issues) == 0, issues


def check_line_endings() -> tuple[bool, list[str]]:
    """Prüft ob Projekt-.py-Dateien LF verwenden (nicht CRLF)."""
    issues: list[str] = []
    for py_file in _PROJECT_ROOT.rglob("*.py"):
        rel = str(py_file.relative_to(_PROJECT_ROOT))
        if _should_skip(rel):
            continue
        try:
            content = py_file.read_bytes()
        except Exception:
            logger.debug("read_bytes() für %s fehlgeschlagen", rel, exc_info=True)
            continue
        if b"\r\n" in content:
            issues.append(f"{rel}: CRLF line endings detected")
    return len(issues) == 0, issues


def check_case_conflicts() -> tuple[bool, list[str]]:
    """Prüft auf case-sensitivity-Konflikte bei Imports."""
    issues: list[str] = []
    py_files: dict[str, str] = {}
    for py_file in _PROJECT_ROOT.rglob("*.py"):
        rel = str(py_file.relative_to(_PROJECT_ROOT))
        if _should_skip(rel):
            continue
        key = rel.lower()
        if key in py_files:
            existing = py_files[key]
            issues.append(f"Case conflict: {existing} vs {rel}")
        py_files[key] = rel
    return len(issues) == 0, issues


def main() -> int:
    all_ok = True

    for name, checker in [
        ("Path Separators (no C:\\...)", check_path_separators),
        ("Line Endings (LF only)", check_line_endings),
        ("Case Conflicts", check_case_conflicts),
    ]:
        ok, issues = checker()
        status = "OK" if ok else "ISSUES"
        print(f"[{status}] {name}: {len(issues)} issue(s)")
        for issue in issues[:10]:
            print(f"     {issue}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll platform compatibility checks passed.")
        return 0
    else:
        print("\nPlatform compatibility issues found.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
