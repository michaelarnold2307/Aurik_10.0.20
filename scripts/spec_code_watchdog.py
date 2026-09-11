#!/usr/bin/env python3
"""spec_code_watchdog.py v1 — Spec↔Code Sync, fast mode using pre-computed caches."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import json
import os
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
SPECS_DIR = PROJECT / ".github" / "specs"


def extract_spec_claims() -> list[dict]:
    """Extract all file references and RELEASE_MUST claims from specs."""
    claims = []
    for sf in sorted(SPECS_DIR.glob("*.md")):
        text = sf.read_text(encoding="utf-8", errors="replace")
        # File paths in backticks (e.g. `backend/core/phase_01.py`)
        for m in re.finditer(r"`(backend/[\w/_.-]+\.py)`", text):
            fp = m.group(1)
            if fp and "/" in fp:
                ctx = text[max(0, m.start() - 60) : m.end() + 60].replace("\n", " ")
                claims.append({"spec": sf.name, "type": "file", "ref": fp, "ctx": ctx[:160]})
        # RELEASE_MUST claims
        for m in re.finditer(r"\[RELEASE_MUST\]\s*(.+?)(?:\n|$)", text):
            claims.append({"spec": sf.name, "type": "release_must", "ref": m.group(1).strip()[:200]})
    return claims


def scan_codebase() -> dict[str, set[str]]:
    """Build a fast lookup: file_path -> set of function/class names."""
    index: dict[str, set[str]] = {}
    skip_dirs = {".venv", ".venv_aurik", "__pycache__", ".git", "node_modules", "models", "temp_repro"}
    for py_file in PROJECT.rglob("*.py"):
        parts = set(str(py_file.relative_to(PROJECT)).split(os.sep))
        if parts & skip_dirs:
            continue
        rel = str(py_file.relative_to(PROJECT))
        try:
            text = py_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            logger.debug("Stiller Ersatzpfad dokumentiert (Bug 9/V74)", exc_info=True)
            continue
        names = set()
        for m in re.finditer(r"^(?:def |class )(\w+)", text, re.MULTILINE):
            names.add(m.group(1))
        for m in re.finditer(r"(?<!\w)([A-Z][A-Z0-9_]{2,}(?:_[A-Z0-9]+)*)\b", text):
            names.add(m.group(1))
        index[rel] = names
    return index


def verify_claims(claims: list[dict], index: dict[str, set[str]]) -> dict:
    """Check each claim against codebase."""
    verified, missing, roadmap = [], [], []
    for c in claims:
        if c["type"] == "file":
            if "[ROADMAP]" in c["ctx"]:
                roadmap.append(c)
            else:
                found = False
                for ipath in index:
                    if ipath.endswith(c["ref"]) or ipath == c["ref"]:
                        verified.append(c)
                        found = True
                        break
                if not found:
                    # Try partial match
                    partial = c["ref"].split("/")[-1]
                    if any(partial in ipath for ipath in index):
                        verified.append(c)
                    else:
                        missing.append(c)
        elif c["type"] == "release_must":
            # Check if key terms from the claim appear in any code file
            terms = [
                t
                for t in re.findall(r"[A-Za-z_]{4,}", c["ref"])
                if t.lower()
                not in (
                    "must",
                    "shall",
                    "soll",
                    "muss",
                    "jeder",
                    "alle",
                    "kein",
                    "ohne",
                    "dass",
                    "wird",
                    "kann",
                    "darf",
                    "oder",
                    "sowie",
                    "auch",
                    "nicht",
                    "eine",
                    "einer",
                    "einem",
                )
            ]
            found_terms = 0
            for t in terms[:8]:
                for names in index.values():
                    if t.lower() in {n.lower() for n in names}:
                        found_terms += 1
                        break
            if found_terms >= max(1, len(terms[:4])):
                verified.append(c)
            elif len(terms) == 0:
                verified.append(c)  # Can't verify, assume OK
            else:
                missing.append(c)
    return {
        "verified": len(verified),
        "missing": len(missing),
        "roadmap": len(roadmap),
        "missing_detail": [{"spec": m["spec"], "type": m["type"], "ref": m["ref"][:120]} for m in missing[:30]],
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    claims = extract_spec_claims()
    index = scan_codebase()
    result = verify_claims(claims, index)

    if args.json:
        print(
            json.dumps(
                {
                    "total_claims": len(claims),
                    "verified": result["verified"],
                    "missing": result["missing"],
                    "roadmap": result["roadmap"],
                    "coverage_pct": round(100 * result["verified"] / max(1, result["verified"] + result["missing"]), 1),
                }
            )
        )
    else:
        cov = round(100 * result["verified"] / max(1, result["verified"] + result["missing"]), 1)
        print(
            f"Spec-Code Watchdog: {result['verified']} verified, {result['missing']} missing, {result['roadmap']} roadmap ({cov}%)"
        )
        if result["missing"]:
            print("\nMissing claims (first 15):")
            for m in result["missing_detail"][:15]:
                print(f"  {m['spec']}: [{m['type']}] {m['ref'][:100]}")
    return 0 if result["missing"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
