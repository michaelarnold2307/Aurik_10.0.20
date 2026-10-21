"""Annotiert bestehende mypy Real-Bugs mit zielgenauen `# type: ignore[...]`.

Dieses Skript ist fuer Sprint-2-Altlagen gedacht. Es ignoriert `var-annotated`,
weil diese Klasse in Sprint 4 durch echte Annotationen bereinigt wird. Es setzt
`# type: ignore[...]` immer vor andere Inline-Kommentare, weil mypy 2.1.0 die
umgekehrte Reihenfolge nicht zuverlaessig akzeptiert.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SKIP_CODES = {"var-annotated"}
ERROR_CODE_RE = re.compile(r"\[([a-z0-9-]+)\]$")
IGNORE_RE = re.compile(r"# type: ignore\[(.*?)\]")


def collect_errors(targets: list[str]) -> dict[str, dict[int, list[str]]]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            *targets,
            "--follow-imports=skip",
            "--no-error-summary",
            "--show-error-codes",
        ],
        capture_output=True,
        text=True,
    )
    errors: dict[str, dict[int, list[str]]] = {}
    for line in result.stdout.splitlines() + result.stderr.splitlines():
        if ": error:" not in line:
            continue
        match = ERROR_CODE_RE.search(line)
        if not match:
            continue
        code = match.group(1)
        if code in SKIP_CODES:
            continue
        parts = line.split(":")
        if len(parts) < 2:
            continue
        try:
            filepath = parts[0]
            lineno = int(parts[1])
        except (ValueError, IndexError):
            continue
        errors.setdefault(filepath, {}).setdefault(lineno, [])
        if code not in errors[filepath][lineno]:
            errors[filepath][lineno].append(code)
    return errors


def add_ignore_to_line(original: str, codes: list[str]) -> str:
    match = IGNORE_RE.search(original)
    if match:
        existing = [code.strip() for code in match.group(1).split(",") if code.strip()]
        missing = [code for code in codes if code not in existing]
        if not missing:
            return original
        merged = existing + missing
        return original[: match.start()] + f"# type: ignore[{','.join(merged)}]" + original[match.end() :]

    codes_str = ",".join(codes)
    marker = "  #"
    if marker in original:
        pos = original.index(marker)
        code_part = original[:pos]
        comment_part = original[pos + 2 :]
        return f"{code_part}  # type: ignore[{codes_str}]  {comment_part}"
    return f"{original}  # type: ignore[{codes_str}]"


def apply_errors(errors: dict[str, dict[int, list[str]]], dry_run: bool) -> int:
    changed = 0
    for filepath, lineno_codes in sorted(errors.items()):
        path = Path(filepath)
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        file_changed = 0
        for lineno, codes in sorted(lineno_codes.items()):
            idx = lineno - 1
            if idx < 0 or idx >= len(lines):
                continue
            old = lines[idx].rstrip("\n").rstrip("\r")
            new = add_ignore_to_line(old, codes)
            if new == old:
                continue
            lines[idx] = f"{new}\n"
            file_changed += 1
        if file_changed:
            changed += file_changed
            if dry_run:
                print(f"[dry-run] {filepath}: {file_changed} Zeilen")
            else:
                path.write_text("".join(lines), encoding="utf-8")
                print(f"{filepath}: {file_changed} Zeilen")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="mypy Real-Bug-Ignores fuer Altlagen setzen")
    parser.add_argument("targets", nargs="+", help="Dateien oder Verzeichnisse fuer mypy")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    errors = collect_errors(args.targets)
    total = sum(len(lines) for lines in errors.values())
    print(f"Gefunden: {len(errors)} Dateien, {total} Fehlerzeilen")
    changed = apply_errors(errors, dry_run=args.dry_run)
    print(f"{'Wuerde aendern' if args.dry_run else 'Geaendert'}: {changed} Zeilen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
