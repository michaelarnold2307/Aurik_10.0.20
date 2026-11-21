"""Tests-Fixture für train_job_runner-Tests (kein Produktionscode).

Der Job-Runner verlangt seit dem Command-Injection-Schutz (§_validate_cmd)
einen echten Projekt-Skript-Pfad — ``python -c`` ist nicht mehr erlaubt.
Dieser Helfer bildet genau die vier Test-Fälle ab:
  - Standard: Exit 0, druckt 'ok'
  - ``--code N``: Exit N
  - ``--sleep S``: schläft S Sekunden (Timeout-Pfad des Runners)
  - ``--append DATEI:ZEICHEN``: hängt ein Zeichen an eine Datei an (Zähler)
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description="Exitcode-Helfer für Job-Runner-Tests")
    ap.add_argument("--code", type=int, default=0, help="Exit-Code (Default 0)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Schlafdauer in Sekunden vor dem Exit")
    ap.add_argument(
        "--append", metavar="DATEI:ZEICHEN", help="Zeichen an Datei anhängen (Zähl-Helfer der Job-Runner-Tests)"
    )
    args = ap.parse_args()
    if args.append:
        from pathlib import Path as _P

        _pfad, _zeichen = args.append.split(":", 1)
        _p = _P(_pfad)
        _p.write_text((_p.read_text(encoding="utf-8") if _p.exists() else "") + _zeichen, encoding="utf-8")
    if args.sleep > 0:
        time.sleep(args.sleep)
    print("ok")
    return args.code


if __name__ == "__main__":
    sys.exit(main())
