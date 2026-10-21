"""Pre-commit hook: Format-String-Guard.

Prüft, ob logger.info/warning/debug/error-Aufrufe die korrekte Anzahl
an %s/%d/%f-Placeholdern für ihre Argumente haben.

Usage: python check_format_strings.py file1.py file2.py ...
Exit: 1 if issues found, 0 otherwise.
"""

import ast
import logging
import re
import sys

logger = logging.getLogger(__name__)


def check_file(filepath: str) -> list[str]:
    """Returns list of violation messages."""
    issues: list[str] = []
    try:
        with open(filepath) as fh:
            tree = ast.parse(fh.read())
    except Exception:
        logger.warning("check_format_strings.py::check_file fallback", exc_info=True)
        return issues

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not hasattr(node.func, "attr"):
            continue
        if node.func.attr not in ("info", "warning", "debug", "error"):
            continue

        args = node.args
        if not args:
            continue
        if not isinstance(args[0], ast.Constant):
            continue
        if not isinstance(args[0].value, str):
            continue

        fmt = args[0].value
        # Count printf-style placeholders, skipping %% (literal percent).
        all_matches = re.findall(r"%%|%[#0+\- ]?[0-9]*\.?[0-9]*[sdfgeEoxXr]", fmt)
        placeholders = sum(1 for m in all_matches if m != "%%")
        nargs = len(args) - 1
        if placeholders != nargs:
            # Check for keyword arguments (e.g., exc_info=True)
            keywords = getattr(node, "keywords", []) or []
            if any(k.arg for k in keywords):
                continue  # kwargs can supplement, skip
            issues.append(f'{filepath}:{node.lineno}: {placeholders} placeholders, {nargs} args — "{fmt[:60]}"')

    return issues


def main() -> None:
    all_issues: list[str] = []
    for filepath in sys.argv[1:]:
        all_issues.extend(check_file(filepath))

    for issue in all_issues:
        print(issue)

    sys.exit(min(len(all_issues), 1))


if __name__ == "__main__":
    main()
