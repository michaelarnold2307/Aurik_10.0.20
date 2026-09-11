#!/usr/bin/env python3
"""

Compliance Check: @staticmethod + self.X-Zugriffe (§2.59, Bugfix 2026-07-09)

Scannt alle Python-Dateien unter backend/, denker/, forensics/ auf:
- Methoden mit @staticmethod-Dekorator, die trotzdem self.<attr> verwenden.
- Dies führt zu NameError zur Laufzeit, weil @staticmethod kein self
  als ersten Parameter übergibt.

Der Check ist fokussiert: nur Methoden mit >= 5 Parametern (die
wahrscheinlichsten Kandidaten für versehentliches @staticmethod).

Usage:
    python scripts/compliance/check_staticmethod_self.py [--fix]

    --fix  Zeigt die betroffenen Zeilen an (kein automatischer Fix)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import ast
import os
import sys
from pathlib import Path

# Verzeichnisse, die gescannt werden
SCAN_DIRS = ["backend", "denker", "forensics"]

# Methode: Prüfe, ob ein Funktionsknoten @staticmethod ist
# und trotzdem self in seinem Body verwendet.


class StaticMethodSelfChecker(ast.NodeVisitor):
    """AST-Visitor: findet @staticmethod-Methoden mit self-Zugriffen."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.violations: list[tuple[int, str, str]] = []  # (line, method_name, self_access)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self._check_method(node.name, item)
        # Nicht in nested classes rekursiv — nur top-level
        for item in node.body:
            if isinstance(item, ast.ClassDef):
                self.visit_ClassDef(item)

    def _check_method(self, class_name: str, func: ast.FunctionDef) -> None:
        # Prüfe: ist @staticmethod?
        is_static = any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in func.decorator_list)
        if not is_static:
            return

        # Prüfe: verwendet die Methode self?
        self_accesses: list[str] = []
        for node in ast.walk(func):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "self":
                    self_accesses.append(f"self.{node.attr}")
            elif isinstance(node, ast.Name) and node.id == "self":
                self_accesses.append("self")

        if self_accesses:
            # Deduplizieren
            unique = sorted(set(self_accesses))
            self.violations.append(
                (
                    func.lineno,
                    f"{class_name}.{func.name}",
                    ", ".join(unique[:5]),  # max 5 self-Zugriffe anzeigen
                )
            )


def scan_directory(root: Path, scan_dir: str) -> list[tuple[str, int, str, str]]:
    """Scannt ein Verzeichnis rekursiv nach Python-Dateien.

    Returns:
        List of (filepath, line, method, self_accesses)
    """
    results: list[tuple[str, int, str, str]] = []
    target = root / scan_dir
    if not target.is_dir():
        return results

    for py_file in target.rglob("*.py"):
        try:
            source = py_file.read_text(encoding="utf-8")
        except Exception:
            logger.debug("Stiller Ersatzpfad dokumentiert (Bug 9/V74)", exc_info=True)
            continue

        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        checker = StaticMethodSelfChecker(str(py_file))
        checker.visit(tree)

        for line, method, accesses in checker.violations:
            results.append((str(py_file), line, method, accesses))

    return results


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent  # Repository-Root
    show_fix = "--fix" in sys.argv

    all_violations: list[tuple[str, int, str, str]] = []
    for scan_dir in SCAN_DIRS:
        all_violations.extend(scan_directory(root, scan_dir))

    if not all_violations:
        print("✅ Keine @staticmethod + self.X-Verletzungen gefunden.")
        return

    print(f"❌ {len(all_violations)} @staticmethod + self.X-Verletzung(en) gefunden:\n")
    for filepath, line, method, accesses in sorted(all_violations):
        rel = os.path.relpath(filepath, root)
        print(f"  {rel}:{line}  {method}")
        print(f"    → Zugriffe auf: {accesses}")
        if show_fix:
            print("    → FIX: Entferne '@staticmethod' und füge 'self' als ersten Parameter ein.")
        print()

    print(
        "HINWEIS: @staticmethod auf Methoden mit self.X-Zugriffen führt zu "
        "NameError zur Laufzeit. Entweder @staticmethod entfernen und self als "
        "ersten Parameter hinzufügen, ODER self.X durch andere Referenzen ersetzen."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
