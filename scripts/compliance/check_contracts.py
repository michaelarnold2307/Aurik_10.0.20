#!/usr/bin/env python3
"""Pre-commit hook: ContractValidator — Cross-Module-Integritätsprüfung.

Läuft in CI/pre-commit. Exit 1 bei Inkonsistenzen — und auch bei fehlgeschlagenem
Import (Befund 2026-10-05: Der Hook lief als stiller No-op, weil `backend` nicht
importierbar war — sys.path[0] ist `scripts/compliance/` — und der ImportError-Zweig
mit Exit 0 endete. §V6 (copilot-instructions.md): Silent-Failure-Klasse).
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from backend.core.defect_contract_validator import run_contract_validation

    result = run_contract_validation()
    if result["ok"]:
        print(f"ContractValidator: ✅ OK ({result['violations']} violations)")
        sys.exit(0)
    print(f"ContractValidator: ❌ {result['violations']} VIOLATIONS")
    for detail in result["details"]:
        print(f"  {detail}")
    sys.exit(1)
except ImportError as e:
    # Kein stilles Überspringen: ein nicht importierbares Modul ist ein Fehler.
    print(f"ContractValidator: ❌ Import failed ({e}) — backend nicht importierbar", file=sys.stderr)
    sys.exit(1)
