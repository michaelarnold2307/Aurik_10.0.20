"""Fix two systematic spec violations (copilot-instructions.md §10.1, §6.6):

1. print() → logger.debug() in all production code (backend/core, plugins, denker)
2. Add 'assert sample_rate == 48000' to all phase process() methods

Usage:
    python scripts/fix_spec_conformance.py [--dry-run]
"""

import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DRY_RUN = "--dry-run" in sys.argv


# ---------------------------------------------------------------------------
# Helper: ensure logger is defined in file
# ---------------------------------------------------------------------------
def ensure_logger(content: str) -> str:
    """Insert 'import logging' and 'logger = ...' if not already present.

    Only considers TOP-LEVEL imports (indentation == 0) to avoid inserting
    the logger_statement inside try/except blocks or function bodies.
    """
    if "getLogger(__name__)" in content:
        return content  # already correct

    lines = content.splitlines(keepends=True)

    # Find insertion point: after the last TOP-LEVEL import block only.
    # A top-level import has no leading whitespace (indentation == 0).
    last_import_idx = 0
    for i, line in enumerate(lines):
        # Only match unindented imports (top-level module scope)
        if (line.startswith("import ") or line.startswith("from ")) and not line[0].isspace():
            last_import_idx = i

    # Safety: if last_import_idx is still 0 and line 0 is not an import,
    # insert at the very beginning (after any module docstring).
    if last_import_idx == 0 and lines and not (lines[0].startswith("import ") or lines[0].startswith("from ")):
        # Find end of module docstring (if any)
        insert_at = 0
        stripped0 = lines[0].strip()
        for quote in ('"""', "'''"):
            if stripped0.startswith(quote):
                rest = stripped0[3:]
                if quote in rest:
                    insert_at = 1
                else:
                    k = 1
                    while k < len(lines) and quote not in lines[k]:
                        k += 1
                    insert_at = k + 1
                break
    else:
        insert_at = last_import_idx + 1

    to_insert: list[str] = []

    if "import logging" not in content:
        to_insert.append("import logging\n")

    to_insert.append("logger = logging.getLogger(__name__)\n")

    lines[insert_at:insert_at] = to_insert
    return "".join(lines)


# ---------------------------------------------------------------------------
# Fix 1: print() → logger.debug()
# ---------------------------------------------------------------------------
def fix_prints_in_file(path: Path) -> int:
    """Replace every bare print() call (outside comments) with logger.debug().
    Returns the number of lines changed.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        logger.warning("fix_spec_conformance.py::fix_prints_in_file fallback", exc_info=True)
        return 0

    if "print(" not in content:
        return 0

    lines = content.splitlines(keepends=True)
    new_lines: list[str] = []
    count = 0

    for line in lines:
        stripped = line.lstrip()

        # Do NOT replace inside comment lines
        if stripped.startswith("#"):
            new_lines.append(line)
            continue

        # Replace empty print() → logger.debug("") first (avoids logger.debug() with no args)
        new_line = re.sub(r"\bprint\(\s*\)", 'logger.debug("")', line)
        # Replace all remaining print( → logger.debug(
        new_line = re.sub(r"\bprint\(", "logger.debug(", new_line)

        if new_line != line:
            count += 1

        new_lines.append(new_line)

    if count == 0:
        return 0

    new_content = ensure_logger("".join(new_lines))

    if not DRY_RUN:
        path.write_text(new_content, encoding="utf-8")

    return count


# ---------------------------------------------------------------------------
# Fix 2: SR-Assert in phase process() methods
# ---------------------------------------------------------------------------
def _find_sr_param(sig_text: str) -> str:
    """Return the parameter name used for sample rate in a method signature."""
    if "sample_rate" in sig_text:
        return "sample_rate"
    if re.search(r"\bsr\b", sig_text):
        return "sr"
    return "sample_rate"  # safe fallback


def _find_body_start(lines: list[str], sig_end: int) -> int:
    """Return the line index of the first actual body statement
    (skipping any docstring that immediately follows the signature).
    """
    body = sig_end + 1
    if body >= len(lines):
        return body

    stripped = lines[body].strip()

    # Detect triple-quoted docstring opening
    for q in ('"""', "'''"):
        if stripped.startswith(q):
            rest = stripped[3:]
            # Single-line docstring: opening and closing on same line
            if q in rest:
                return body + 1
            # Multi-line: scan forward for closing quotes
            k = body + 1
            while k < len(lines) and q not in lines[k]:
                k += 1
            return k + 1

    return body


def add_sr_assert_to_phase(path: Path) -> bool:
    """Insert 'assert sample_rate == 48000 ...' into process() body.
    Returns True if the file was modified.
    """
    content = path.read_text(encoding="utf-8")

    # Skip if already asserted
    if "== 48000" in content or ("48_000" in content and "assert" in content):
        return False

    lines = content.splitlines(keepends=True)

    # Locate the process() method definition
    proc_start: int | None = None
    for i, line in enumerate(lines):
        if re.match(r"\s+def process\s*\(", line):
            proc_start = i
            break

    if proc_start is None:
        return False

    # Collect full signature text (up to the closing ')' of the parameter list)
    sig_text = ""
    sig_end = proc_start
    for k in range(proc_start, min(proc_start + 20, len(lines))):
        sig_text += lines[k]
        # Signature line ends when we see ') :' or ')-> ... :'
        if re.search(r"\)\s*(?:->.*?)?\s*:\s*$", lines[k].rstrip()):
            sig_end = k
            break

    sr_param = _find_sr_param(sig_text)
    body_start = _find_body_start(lines, sig_end)

    if body_start >= len(lines):
        return False

    # Determine indentation (match the first body line)
    body_line = lines[body_start]
    indent = len(body_line) - len(body_line.lstrip())

    # Guard: avoid inserting twice
    if "== 48000" in (lines[body_start] if body_start < len(lines) else ""):
        return False

    assert_line = f'{" " * indent}assert {sr_param} == 48000, f"SR muss 48000 Hz sein, erhalten: {{{sr_param}}}"\n'

    lines.insert(body_start, assert_line)

    if not DRY_RUN:
        path.write_text("".join(lines), encoding="utf-8")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    mode = "DRY-RUN" if DRY_RUN else "LIVE"
    print(f"\n=== Aurik 10.0.0 Spec-Konformität-Fix ({mode}) ===\n")

    # ------------------------------------------------------------------
    # Fix 1: print() → logger.debug()
    # ------------------------------------------------------------------
    print("--- Fix 1: print() → logger.debug() ---")
    dirs_to_scan = [
        ROOT / "backend" / "core",
        ROOT / "plugins",
        ROOT / "denker",
    ]

    total_lines = 0
    total_files = 0

    for scan_dir in dirs_to_scan:
        for py_file in sorted(scan_dir.rglob("*.py")):
            if "__pycache__" in str(py_file):
                continue
            n = fix_prints_in_file(py_file)
            if n:
                total_files += 1
                total_lines += n
                rel = py_file.relative_to(ROOT)
                print(f"  ✅ {rel}  ({n} Zeilen)")

    print(f"\n  Gesamt: {total_lines} print()-Zeilen in {total_files} Dateien konvertiert\n")

    # ------------------------------------------------------------------
    # Fix 2: SR-Assert in all phase files
    # ------------------------------------------------------------------
    print("--- Fix 2: assert sample_rate == 48000 in Phasen ---")
    phases_dir = ROOT / "backend" / "core" / "phases"
    sr_fixed = 0

    for phase_file in sorted(phases_dir.glob("phase_[0-9]*.py")):
        if add_sr_assert_to_phase(phase_file):
            sr_fixed += 1
            print(f"  ✅ {phase_file.name}")
        else:
            print(f"  ⏭  {phase_file.name}  (bereits vorhanden oder kein process())")

    print(f"\n  Gesamt: {sr_fixed} Phasen mit SR-Assert versehen\n")

    print("=== Fertig ===\n")


if __name__ == "__main__":
    main()
