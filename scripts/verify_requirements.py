"""scripts/verify_requirements.sh-equivalent in Python.

Prüft alle Einträge in requirements/requirements_aurik.txt darauf,
ob sie auf PyPI verfügbar sind (dry-run). Gibt Exit-Code 0 bei Erfolg,
1 bei Fehler zurück.

Ausführen:
    python scripts/verify_requirements.py
    # oder via Wrapper:
    bash scripts/verify_requirements.sh
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent.parent
REQ_FILE = ROOT / "requirements" / "requirements_aurik.txt"


def main() -> int:
    if not REQ_FILE.exists():
        print(f"[WARN] {REQ_FILE} nicht gefunden — nichts zu prüfen.")
        return 0

    print(f"Prüfe Requirements: {REQ_FILE}")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--quiet",
            "-r",
            str(REQ_FILE),
            "--extra-index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        print("[FEHLER] pip dry-run fehlgeschlagen:", file=sys.stderr)
        print(result.stderr[:2000], file=sys.stderr)
        return 1

    if result.stdout.strip():
        print(result.stdout[:1000])

    print("✓ Alle Requirements auf PyPI verfügbar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
