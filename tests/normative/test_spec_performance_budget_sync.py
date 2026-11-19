from __future__ import annotations

"""Normativer Sync-Test fuer Performance-Budget zwischen Slim-Core und Spec 07.

[RELEASE_MUST]
- Gemeinsame Budget-Operationen in .github/copilot-instructions.md und
  .github/specs/07_quality_and_tests.md muessen identische Sekundenwerte haben.
"""


import re
from pathlib import Path

import pytest

_CORE = Path(".github/copilot-instructions.md")
_SPEC07 = Path(".github/specs/07_quality_and_tests.md")

_HEADER = "| Operation | Limit / Minute Audio |"


def _extract_budget_table(text: str) -> dict[str, float]:
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if _HEADER in ln)
    except StopIteration as exc:
        raise AssertionError(f"Budget-Tabelle nicht gefunden: {_HEADER}") from exc

    out: dict[str, float] = {}
    for ln in lines[start + 2 :]:
        if not ln.strip().startswith("|"):
            break
        parts = [p.strip() for p in ln.strip().strip("|").split("|")]
        if len(parts) < 2:
            continue
        op, cell = parts[0], parts[1]
        sec_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*s", cell)
        if not sec_match:
            continue
        out[op] = float(sec_match.group(1))
    return out


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_common_performance_budgets_are_in_sync() -> None:
    assert _CORE.exists(), f"Fehlt: {_CORE}"
    assert _SPEC07.exists(), f"Fehlt: {_SPEC07}"

    core = _extract_budget_table(_CORE.read_text(encoding="utf-8"))
    spec = _extract_budget_table(_SPEC07.read_text(encoding="utf-8"))

    common_ops = set(core) & set(spec)
    assert common_ops, "Keine gemeinsamen Budget-Operationen gefunden."

    mismatches: list[str] = []
    for op in sorted(common_ops):
        if core[op] != spec[op]:
            mismatches.append(f"{op}: core={core[op]}s, spec={spec[op]}s")

    assert not mismatches, "Performance-Budget-Sync verletzt zwischen Slim-Core und Spec 07:\n" + "\n".join(mismatches)


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_defect_scanner_budget_matches_calibrated_norm() -> None:
    """Quality-ERB-Default-Norm (16 s/min, gemessen 14,9 s/min warm nach
    PERF-R19/R19b/R19c + R20, 2026-11-07). Das 4-s-Fast-Ziel bleibt als
    dokumentiertes Ziel hinter der Hör-Freigabe des Batch-Pfads (§2.46h).
    Beide Dateien müssen die erste geparste Zahl (16) synchron tragen."""
    core = _extract_budget_table(_CORE.read_text(encoding="utf-8"))
    spec = _extract_budget_table(_SPEC07.read_text(encoding="utf-8"))

    assert core.get("DefectScanner") == 16.0, (
        "DefectScanner-Budget in copilot-instructions muss 16 s/min sein (Quality-ERB-Standard, gemessen)."
    )
    assert spec.get("DefectScanner") == 16.0, (
        "DefectScanner-Budget in Spec 07 muss 16 s/min sein (Quality-ERB-Standard, gemessen)."
    )
