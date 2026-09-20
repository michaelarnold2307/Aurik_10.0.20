#!/usr/bin/env python3
"""§v10.802 Version-Guard — Bump-Konsistenz gegen die Git-Historie (Warnstufe).

Prüft seit der letzten Änderung an ``backend/core/version.py``:
  - feat-Commits ohne Minor-Bump  → WARNUNG
  - fix-/perf-Commits ohne Patch-Bump → HINWEIS
Kein Fail: Der Guard informiert; der Bump-Entscheid bleibt beim Autor
(§v10.802 (copilot-instructions.md) Versions-Vertrag).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = "backend/core/version.py"


def _git(*args: str) -> str:
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        return ""
    return r.stdout.strip()


def _read_version() -> str:
    p = ROOT / VERSION_FILE
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip('"')
    return "0.0.0"


def _parse_version(v: str) -> tuple[int, int, int]:
    parts = v.split(".")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def main() -> int:
    last = _git("log", "-1", "--format=%H", "--", VERSION_FILE)
    if not last:
        print("version-guard: keine Git-Historie verfügbar — übersprungen.")
        return 0
    cur = _read_version()
    last_v = _read_version_from_commit(last)
    commits = _git("log", "--format=%s", f"{last}..HEAD").splitlines()
    commits = [c for c in commits if c.strip()]
    n_feat = sum(1 for c in commits if c.startswith("feat"))
    n_patch = sum(1 for c in commits if c.startswith(("fix", "perf")))
    if not commits:
        return 0
    try:
        last_p = _parse_version(last_v)
        cur_p = _parse_version(cur)
    except ValueError:
        print(f"version-guard: Version nicht parsebar ({cur!r} / {last_v!r}).")
        return 0
    minor_bumped = cur_p[1] > last_p[1] or cur_p[0] > last_p[0]
    patch_bumped = cur_p[2] > last_p[2] or minor_bumped
    if n_feat and not minor_bumped:
        print(
            f"⚠ version-guard: {n_feat} feat-Commits seit {last_v} ohne Minor-Bump "
            f"(aktuell {cur}) — §v10.802 (copilot-instructions.md): Minor-Bump im selben Merge."
        )
    if n_patch and not patch_bumped:
        print(
            f"ℹ version-guard: {n_patch} fix/perf-Commits seit {last_v} ohne Patch-Bump "
            f"(aktuell {cur}) — §v10.802 (copilot-instructions.md): Patch-Bump im selben Merge."
        )
    return 0


def _read_version_from_commit(sha: str) -> str:
    content = _git("show", f"{sha}:{VERSION_FILE}")
    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip('"')
    return "0.0.0"


if __name__ == "__main__":
    sys.exit(main())
