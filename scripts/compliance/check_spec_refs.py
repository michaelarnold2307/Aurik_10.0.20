#!/usr/bin/env python3
"""Pre-commit hook: §-Referenz-Check.


Prüft, ob neue §-Referenzen im SPEC.md dokumentiert sind.
Exit 1 wenn undokumentierte §-Referenzen gefunden werden.
"""

import logging

logger = logging.getLogger(__name__)

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SPEC_PATH = ROOT / "SPEC.md"

# Extrahiere dokumentierte §-Referenzen aus SPEC.md
DOCUMENTED: set[str] = set()
if SPEC_PATH.exists():
    spec_text = SPEC_PATH.read_text(encoding="utf-8")
    for match in re.finditer(r"`(§[A-Za-z0-9._-]+)`", spec_text):
        DOCUMENTED.add(match.group(1))

# Extrahiere §-Referenzen aus geänderten/neuen Dateien
new_refs: dict[str, set[str]] = {}
for fp in sys.argv[1:]:
    try:
        content = Path(fp).read_text(encoding="utf-8")
    except Exception:
        logger.debug("Stiller Ersatzpfad dokumentiert (Bug 9/V74)", exc_info=True)
        continue
    found = set(re.findall(r"§[A-Za-z0-9._-]+", content))
    undocumented = found - DOCUMENTED
    if undocumented:
        new_refs[fp] = undocumented

if not new_refs:
    print(f"§-Check: ✅ Alle Referenzen dokumentiert ({len(DOCUMENTED)} in SPEC.md)")
    sys.exit(0)

print(
    f"§-Check: ❌ {sum(len(v) for v in new_refs.values())} undokumentierte §-Referenzen in {len(new_refs)} Dateien:\n"
)
for fp, refs in sorted(new_refs.items()):
    print(f"  {fp}:")
    for ref in sorted(refs):
        print(f"    {ref}")
print(f"\nBitte in SPEC.md dokumentieren ({SPEC_PATH}).")
sys.exit(1)
