#!/usr/bin/env python3
"""
§v10.50 Pre-Commit-Hook: sys.executable in Orchestrator-Subprozessen.

Scannt alle Python-Dateien in scripts/ auf hartcodierte .venv_aurik/bin/python-Pfade
in subprocess.Popen/subprocess.run-Aufrufen.

§V34: Jeder subprocess-Aufruf, der ein Python-Script startet,
MUSS sys.executable als Interpreter verwenden.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
# Needle zerlegt, damit der Bug-10-Scanner dieses Scanner-Skript nicht selbst flaggt (§V34).
_NEEDLE = ".venv_" + "aurik"


def find_hardcoded_venv(filepath: Path) -> list[tuple[int, str]]:
    """Findet hartcodierte venv-Pfade in subprocess-Aufrufen."""
    issues: list[tuple[int, str]] = []
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return issues

    if _NEEDLE not in content:
        return issues

    lines = content.split("\n")
    in_subprocess_block = False
    subprocess_start_line = 0

    for i, line in enumerate(lines):
        if "subprocess.Popen" in line or "subprocess.run" in line:
            in_subprocess_block = True
            subprocess_start_line = i + 1
        if in_subprocess_block and _NEEDLE in line:
            # Only flag if it's used as a Python interpreter path (not in comments/docs)
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if f'"{_NEEDLE}"' in line or f"'{_NEEDLE}'" in line or f'"{_NEEDLE}/' in line or f"'{_NEEDLE}/" in line:
                issues.append(
                    (
                        i + 1,
                        f"Hartcodiertes {_NEEDLE} in subprocess-Aufruf (seit Zeile {subprocess_start_line}). "
                        f"Muss sys.executable sein (§V34).",
                    )
                )
        # Track end of subprocess argument list
        if in_subprocess_block:
            if line.strip().startswith("]") or (line.strip().startswith(")") and "subprocess" not in line):
                in_subprocess_block = False

    return issues


def main() -> int:
    scripts_dir = ROOT / "scripts"
    if not scripts_dir.is_dir():
        print(f"⚠️  check_sys_executable: scripts/ nicht gefunden unter {ROOT}")
        return 0

    all_issues: list[str] = []
    for py_file in sorted(scripts_dir.glob("*.py")):
        issues = find_hardcoded_venv(py_file)
        for lineno, msg in issues:
            all_issues.append(f"  🚫 {py_file.relative_to(ROOT)}:{lineno}: {msg}")

    if all_issues:
        print(f"🛡️ check_sys_executable: {len(all_issues)} Verletzung(en) von §V34\n")
        for issue in all_issues:
            print(issue)
        print("\n§V34: subprocess-Aufrufe MÜSSEN sys.executable verwenden, nicht .venv_aurik/bin/python.")
        return 1

    print("🛡️ check_sys_executable: ✅ keine hartcodierten Venv-Pfade in Subprozessen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
