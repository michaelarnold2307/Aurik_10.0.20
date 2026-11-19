"""Prueft, dass mypy-Ignore-Kommentare vor anderen Inline-Kommentaren stehen.

mypy 2.1.0 akzeptiert ein `# type: ignore[...]` nicht zuverlaessig, wenn davor
bereits ein anderer Inline-Kommentar steht:

    return value  # erklaerung  # type: ignore[no-any-return]

Kanonisch ist:

    return value  # type: ignore[no-any-return]  # erklaerung
"""

from __future__ import annotations

import re
import sys
import tokenize
from pathlib import Path

BAD_INLINE_IGNORE = re.compile(r"^#\s+(?!type:).+?\s+# type: ignore\[")


def iter_python_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in path.rglob("*.py") if p.is_file())
    return sorted(set(files))


def main() -> int:
    roots = sys.argv[1:] or ["backend", "plugins", "Aurik10", "cli", "scripts"]
    violations: list[tuple[Path, int, str]] = []

    for path in iter_python_files(roots):
        try:
            with tokenize.open(path) as handle:
                tokens = tokenize.generate_tokens(handle.readline)
                for token in tokens:
                    if token.type != tokenize.COMMENT:
                        continue
                    if token.start[1] == 0:
                        continue
                    if not token.line[: token.start[1]].strip():
                        continue
                    if BAD_INLINE_IGNORE.search(token.string):
                        violations.append((path, token.start[0], token.line.strip()))
        except (SyntaxError, tokenize.TokenError, UnicodeDecodeError):
            continue

    if violations:
        print("mypy type-ignore Reihenfolge verletzt:")
        for path, lineno, line in violations[:50]:
            print(f"  {path}:{lineno}: {line}")
        if len(violations) > 50:
            print(f"  ... {len(violations) - 50} weitere")
        print("\nKanonisch: code  # type: ignore[code]  # erklaerender Kommentar")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
