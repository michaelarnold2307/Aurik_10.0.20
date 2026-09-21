#!/usr/bin/env python3
"""Generate RELEASE_MUST coverage report for spec-to-test traceability.

This script links RELEASE_MUST clauses in .github/copilot-instructions.md
against test gates in tests/normative/ and tests/unit/.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / ".github" / "copilot-instructions.md"
NORMATIVE_TESTS = ROOT / "tests" / "normative"
UNIT_TESTS = ROOT / "tests" / "unit"
TESTS_ROOT = ROOT / "tests"
REPORT_PATH = ROOT / "reports" / "release_must_coverage.json"

# Hard traceability anchors for headings that are semantically broad and cannot
# be matched reliably by keyword heuristics alone.
# Nur Einträge für AKTUELLE [RELEASE_MUST]-Header in copilot-instructions.md —
# veraltete Header wurden 2026-09-21 entfernt (tote Einträge wären irreführend).
FORCED_TRACEABILITY: dict[str, list[str]] = {
    "## [RELEASE_MUST] Autonomer Magic-Button-Betrieb": [
        "tests/normative/test_magic_button_autopilot_ci_gate.py",
    ],
    "## [RELEASE_MUST] Strength-Envelope-Nichtdegeneration (v10.0.x)": [
        "tests/unit/test_strength_envelope_non_degenerate.py",
        "tests/unit/test_b3_full_song_defect_merge.py",
    ],
}


@dataclass(frozen=True)
class CoverageItem:
    """Traceability result for one RELEASE_MUST instruction line."""

    release_must: str
    matched_tests: list[str]
    covered: bool


def _extract_release_must_lines(text: str) -> list[str]:
    items: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if "[RELEASE_MUST]" not in line:
            continue
        if len(line) < 18:
            continue
        items.append(line)
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _keywords(line: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9_\-]+", line.lower())
    stop = {
        "release_must",
        "der",
        "die",
        "das",
        "und",
        "mit",
        "für",
        "von",
        "auf",
        "in",
        "zu",
        "no",
        "mode",
    }
    return {w for w in words if len(w) >= 5 and w not in stop}


def _iter_test_files() -> Iterable[Path]:
    files: list[Path] = []
    if TESTS_ROOT.exists():
        files.extend(sorted(TESTS_ROOT.rglob("test_*.py")))
    return files


def _extract_explicit_paths(release_line: str) -> list[str]:
    """Extract explicit test/config paths from a release line.

    If copilot instructions reference concrete files (e.g. tests/unit/test_x.py,
    conftest.py, benchmarks/...py), we can directly map those paths and avoid
    keyword-only false negatives.
    """
    return re.findall(r"[A-Za-z0-9_./-]+\.py", release_line)


def _match_tests(release_line: str) -> list[str]:
    forced_paths = FORCED_TRACEABILITY.get(release_line, [])
    forced_matches = [p for p in forced_paths if (ROOT / p).exists()]

    explicit_matches: list[str] = []
    for raw in _extract_explicit_paths(release_line):
        path = ROOT / raw
        if path.exists():
            explicit_matches.append(raw)

    line_keys = _keywords(release_line)
    if not line_keys:
        # Keep explicit path matches even if keyword extraction yields no tokens.
        return sorted(set(forced_matches + explicit_matches))

    matches: list[str] = list(forced_matches + explicit_matches)
    for path in _iter_test_files():
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        hit_count = sum(1 for key in line_keys if key in text)
        if hit_count >= 2:
            matches.append(str(path.relative_to(ROOT)))
    return sorted(set(matches))


def build_report() -> dict:
    """Build the RELEASE_MUST-to-test coverage report payload."""

    spec_text = SPEC_PATH.read_text(encoding="utf-8", errors="replace")
    release_must_items = _extract_release_must_lines(spec_text)

    coverage_items: list[CoverageItem] = []
    for item in release_must_items:
        tests = _match_tests(item)
        coverage_items.append(CoverageItem(release_must=item, matched_tests=tests, covered=bool(tests)))

    total = len(coverage_items)
    covered = sum(1 for item in coverage_items if item.covered)
    pct = (covered / total * 100.0) if total else 0.0

    return {
        "source": str(SPEC_PATH.relative_to(ROOT)),
        "test_dirs": [
            str(NORMATIVE_TESTS.relative_to(ROOT)),
            str(UNIT_TESTS.relative_to(ROOT)),
        ],
        "total_release_must_items": total,
        "covered_items": covered,
        "coverage_percent": round(pct, 2),
        "items": [asdict(item) for item in coverage_items],
    }


def main() -> int:
    """Write the coverage report and return the CI-style exit code."""

    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Hard gate for CI: every RELEASE_MUST line should map to at least one test/config path.
    uncovered = report["total_release_must_items"] - report["covered_items"]
    if uncovered > 0:
        print(
            f"RELEASE_MUST coverage incomplete: {report['covered_items']}/{report['total_release_must_items']} "
            f"({report['coverage_percent']}%)."
        )
        print(f"Report: {REPORT_PATH.relative_to(ROOT)}")
        return 2

    print(
        f"RELEASE_MUST coverage OK: {report['covered_items']}/{report['total_release_must_items']} "
        f"({report['coverage_percent']}%)."
    )
    print(f"Report: {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
