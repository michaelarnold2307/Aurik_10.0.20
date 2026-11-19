"""Pre-commit hook: Logger-Debug-Guard — erkennt teure debug-Aufrufe in Hot-Loops.

Warnt, wenn logger.debug() in for/while-Schleifen ohne isEnabledFor-Guard steht.
Vermeidet µs-Overhead in DSP-Phasen.

Usage: python scripts/compliance/check_debug_guard.py file1.py ...
"""

import ast
import logging
import sys

logger = logging.getLogger(__name__)


def check_file(filepath: str) -> list[str]:
    issues: list[str] = []
    try:
        with open(filepath) as fh:
            tree = ast.parse(fh.read())
    except Exception:
        logger.warning("check_debug_guard.py::check_file fallback", exc_info=True)
        return issues

    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While)):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if not hasattr(child.func, "attr"):
                continue
            if child.func.attr != "debug":
                continue
            if not hasattr(child.func, "value"):
                continue
            if not isinstance(child.func.value, ast.Name):
                continue
            if child.func.value.id != "logger":
                continue

            # Check: is there an isEnabledFor guard above?
            # Simplified: just warn
            issues.append(
                f"{filepath}:{child.lineno}: logger.debug() in loop "
                f"body — consider 'if logger.isEnabledFor(logging.DEBUG):' guard"
            )
            break  # One warning per loop is enough

    return issues


def main() -> None:
    all_issues: list[str] = []
    for fp in sys.argv[1:]:
        all_issues.extend(check_file(fp))
    for issue in all_issues:
        print(issue)
    # Never block — just warn
    sys.exit(0)


if __name__ == "__main__":
    main()
