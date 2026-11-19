"""scripts/apply_no_any_return_ignores.py — Sprint 3 Massenfix
==============================================================
Fügt `# type: ignore[no-any-return]` an alle Zeilen ein, die mypy mit
[no-any-return] markiert. Analog zum UV3-Fix (2026-06-26).

Verwendung:
    python scripts/apply_no_any_return_ignores.py <pfad_oder_datei> [--dry-run]

Invarianten:
    - Nur `no-any-return` wird ergänzt, kein anderer Code geändert.
    - Zeilen die bereits `# type: ignore` enthalten werden erweitert,
      nicht doppelt annotiert.
    - --dry-run zeigt nur was geändert würde.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def get_mypy_no_any_return_lines(path: str) -> dict[str, set[int]]:
    """Führt mypy aus und sammelt alle no-any-return Zeilennummern."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            path,
            "--follow-imports=skip",
            "--no-error-summary",
        ],
        capture_output=True,
        text=True,
    )
    errors: dict[str, set[int]] = {}
    for line in result.stdout.splitlines():
        if "[no-any-return]" not in line:
            continue
        # Format: path/file.py:123: error: ...
        parts = line.split(":")
        if len(parts) < 2:
            continue
        try:
            filepath = parts[0]
            lineno = int(parts[1])
        except (ValueError, IndexError):
            continue
        errors.setdefault(filepath, set()).add(lineno)
    return errors


def apply_ignores(filepath: str, linenos: set[int], dry_run: bool) -> int:
    """Ergänzt # type: ignore[no-any-return] an den betroffenen Zeilen."""
    path = Path(filepath)
    if not path.exists():
        print(f"WARNUNG: Datei nicht gefunden: {filepath}")
        return 0

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    changed = 0
    for lineno in sorted(linenos):
        idx = lineno - 1
        if idx < 0 or idx >= len(lines):
            continue
        original = lines[idx].rstrip("\n").rstrip("\r")
        # Bereits annotiert?
        if "# type: ignore" in original:
            # Prüfe ob no-any-return bereits drin
            if "no-any-return" in original:
                continue
            # Erweitere bestehenden type: ignore Kommentar
            new_line = original.replace(
                "# type: ignore[",
                "# type: ignore[no-any-return,",
                1,
            )
        else:
            # mypy 2.1.0: # type: ignore muss VOR anderen Inline-Kommentaren stehen.
            # Ziel: "code  # kommentar" → "code  # type: ignore[no-any-return]  # kommentar"
            comment_marker = "  #"
            if comment_marker in original:
                comment_pos = original.index(comment_marker)
                code_part = original[:comment_pos]
                rest = original[comment_pos + 2 :]  # "# kommentar"
                new_line = f"{code_part}  # type: ignore[no-any-return]  {rest}"
            else:
                new_line = original + "  # type: ignore[no-any-return]"
        eol = "\n"
        lines[idx] = new_line + eol
        changed += 1
        if dry_run:
            print(f"  {filepath}:{lineno}: {original.strip()!r} → annotiert")

    if not dry_run and changed > 0:
        path.write_text("".join(lines), encoding="utf-8")
        print(f"  {filepath}: {changed} Zeilen annotiert")
    elif dry_run:
        print(f"  [dry-run] {filepath}: würde {changed} Zeilen ändern")

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="no-any-return Massenfix")
    parser.add_argument("path", help="Datei oder Verzeichnis für mypy")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur anzeigen, nicht schreiben",
    )
    args = parser.parse_args()

    print(f"mypy läuft auf: {args.path} ...")
    errors = get_mypy_no_any_return_lines(args.path)
    total_lines = sum(len(v) for v in errors.values())
    print(f"Gefunden: {len(errors)} Dateien, {total_lines} no-any-return Fehler")

    total_changed = 0
    for filepath, linenos in sorted(errors.items()):
        total_changed += apply_ignores(filepath, linenos, args.dry_run)

    print(f"\n{'[dry-run] Würde' if args.dry_run else 'Geändert:'} {total_changed} Zeilen in {len(errors)} Dateien")
    if not args.dry_run and total_changed > 0:
        print("Verifikation: mypy erneut ausführen zum Bestätigen...")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                args.path,
                "--follow-imports=skip",
                "--no-error-summary",
            ],
            capture_output=True,
            text=True,
        )
        remaining = sum(1 for line in result.stdout.splitlines() if "[no-any-return]" in line)
        print(f"Verbleibende no-any-return Fehler: {remaining}")


if __name__ == "__main__":
    main()
