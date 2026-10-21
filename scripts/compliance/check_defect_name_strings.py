"""Compliance Check: Defekt-Namen in .get() / in()-Patterns (§2.59)

Scannt auf das exakte Muster, das Bugs verursacht hat:
  - _ds.get("hiss", ...)      → "hiss" existiert nicht in DefectType
  - defect_type in ("click",)  → "click" sollte "clicks" sein

Usage:
    python scripts/compliance/check_defect_name_strings.py
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.defect_scanner import DefectType

logger = logging.getLogger(__name__)

CANONICAL: set[str] = {e.value for e in DefectType} | {e.name.lower() for e in DefectType}
SYNTHETIC_OK: set[str] = {"wow_flutter", "tape_hiss"}

# Präzise Patterns: nur .get("X") und in ("X","Y") mit Defekt-Keywords
GET_PATTERN = re.compile(r'\.get\(\s*["\']([a-z_]+)["\']')
IN_PATTERN = re.compile(r'in\s*\(\s*["\']([a-z_]+)["\']')

SKIP_DIRS = {"tests", "docs", "__pycache__", ".git", "models", "venv", "golden_samples", "scripts/compliance"}


def scan_file(filepath: Path) -> list[tuple[int, str, str]]:
    """Returns [(line, found_name, suggested_canonical), ...]."""
    try:
        lines = filepath.read_text(encoding="utf-8").split("\n")
    except Exception:
        logger.warning("check_defect_name_strings.py::scan_file fallback", exc_info=True)
        return []

    findings = []
    for i, line in enumerate(lines, 1):
        for match in GET_PATTERN.finditer(line):
            s = match.group(1)
            if s in CANONICAL or s in SYNTHETIC_OK:
                continue
            # Nur melden, wenn es wie ein Defekt-Name aussieht
            if not _looks_like_defect(s):
                continue
            suggestion = _suggest(s)
            findings.append((i, s, suggestion))

        for match in IN_PATTERN.finditer(line):
            s = match.group(1)
            if s in CANONICAL or s in SYNTHETIC_OK:
                continue
            if not _looks_like_defect(s):
                continue
            suggestion = _suggest(s)
            findings.append((i, s, suggestion))

    return findings


def _looks_like_defect(s: str) -> bool:
    """Nur Strings, die tatsächlich Defekt-Namen sein könnten."""
    keywords = {
        "click",
        "pop",
        "crackle",
        "hum",
        "buzz",
        "hiss",
        "rumble",
        "wow",
        "flutter",
        "dropout",
        "clipping",
        "distortion",
        "noise",
        "bandwidth",
        "azimuth",
        "sibilance",
        "reverb",
        "pitch",
        "speed",
        "phase",
        "transient",
        "surface",
        "subsonic",
        "tape_hiss",
        "wow_flutter",
        "hf_loss",
        "broadband",
    }
    return any(kw in s for kw in keywords) and s not in _NON_DEFECT


_NON_DEFECT = {
    "description",
    "transient_ratio",
    "transient_rich",
    "noise_type",
    "phase_id",
    "phase_type",
    "phase_gate",
    "phases_guarded",
    "pre_phase_audio",
    "post_phase_audio",
    "phase_hotspots",
    "worst_phases",
    "phases_executed",
    "phases_skipped",
    "phase_type_summary",
    "denoise",
    "ml_denoise",
    "sgmse_dereverb",
    "rmvpe_pitch",
    "get_clipping_classifier",
    "get_noise_reducer",
    "sub_threshold_phases",
    "has_reverb",
    "has_clipping",
    "hf_loss_db",
    "get_per_phase_musical_goals_gate",
    "destination",
}


def _suggest(s: str) -> str:
    """Findet den nächsten kanonischen Namen."""
    for c in sorted(CANONICAL):
        if s in c or c in s:
            return c
    for c in sorted(CANONICAL):
        if len(s) >= 3 and s[:3] in c:
            return c
    return "?"


def main() -> None:
    scan_dirs = ["backend", "denker", "forensics", "cli"]
    all_findings: list[tuple[str, int, str, str]] = []

    for sd in scan_dirs:
        target = ROOT / sd
        if not target.is_dir():
            continue
        for py_file in target.rglob("*.py"):
            rel = str(py_file.relative_to(ROOT))
            if any(d in rel for d in SKIP_DIRS):
                continue
            for line_no, found, suggestion in scan_file(py_file):
                all_findings.append((rel, line_no, found, suggestion))

    if not all_findings:
        print("✅ Keine Defekt-Namen-Mismatches in .get() / in()-Patterns.")
        return

    by_file: dict[str, list[tuple[int, str, str]]] = {}
    for rel, line_no, found, suggestion in all_findings:
        by_file.setdefault(rel, []).append((line_no, found, suggestion))

    print(f"⚠️  {len(all_findings)} potenzielle Mismatches in {len(by_file)} Dateien:\n")
    for fname in sorted(by_file):
        items = by_file[fname]
        print(f"  {fname}:")
        for line_no, found, suggestion in sorted(items):
            print(f'    L{line_no}: "{found}" → {suggestion}')
        print()

    print(f"Gesamt: {len(all_findings)} Mismatches.")
    sys.exit(1)


if __name__ == "__main__":
    main()
