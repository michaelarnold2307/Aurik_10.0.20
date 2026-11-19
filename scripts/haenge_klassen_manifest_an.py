"""scripts/haenge_klassen_manifest_an.py — Zeilen einer Klasse ans Haupt-Manifest anhängen.

Letzter Schritt der Daten-Expansions-Ketten (Crackle, Vocal-Harshness;
2026-11-19): hängt die Zeilen EINER Klasse aus einem generierten
`erstelle_klassen_paare.py`-Manifest an das Haupt-Manifest
`data/klassen_paare/manifest.jsonl` an.

Garantien:
- titel-/rezept-/start-/end-dedupliziert: mehrfacher Aufruf ist idempotent
  (Self-Healing der Ketten — ein Neustart der Kette darf nichts verdoppeln);
- Zeilen ANDERER Klassen bleiben unangetastet (kein Leak über Klassen);
- fehlt das Quell-Manifest, wird das gemeldet und mit Exit 0 beendet
  (die Kette darf nicht an einem fehlenden Vorgänger-Artefakt hängen);
- Report-only, kein Flag-Flip, kein Deploy.

Beleg: docs/reports/current/<datum>_klassen_manifest_anhaengen_<klasse>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SCHEMA = "aurik.klassen_manifest_anhaengen/1"
DEFAULT_ZIEL = _ROOT / "data" / "klassen_paare" / "manifest.jsonl"
REPORT_DIR = _ROOT / "docs" / "reports" / "current"


def _schluessel(zeile: dict) -> tuple[str, str, int, int]:
    return (
        str(zeile.get("titel", "")),
        str(zeile.get("rezept", "")),
        int(zeile.get("start", 0)),
        int(zeile.get("end", 0)),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Manifest-Zeilen einer Klasse dedupliziert anhängen")
    ap.add_argument("--quelle", type=Path, required=True, help="Generiertes erstelle_klassen_paare-Manifest")
    ap.add_argument("--ziel", type=Path, default=DEFAULT_ZIEL, help="Haupt-Manifest (wird nur ANGEHÄNGT)")
    ap.add_argument("--klasse", required=True, help="Nur Zeilen dieser Klasse anhängen")
    args = ap.parse_args(argv)

    if not args.quelle.is_file():
        print(f"HINWEIS: Quell-Manifest fehlt ({args.quelle}) — nichts anzuhängen (Exit 0).")
        return 0
    if not args.ziel.is_file():
        print(f"ABBRUCH: Ziel-Manifest fehlt ({args.ziel}).")
        return 2

    quelle_zeilen = [
        json.loads(l)
        for l in args.quelle.read_text(encoding="utf-8").splitlines()
        if l.strip() and json.loads(l).get("klasse") == args.klasse
    ]
    ziel_text = args.ziel.read_text(encoding="utf-8")
    vorhandene = {_schluessel(json.loads(l)) for l in ziel_text.splitlines() if l.strip()}

    neu = [z for z in quelle_zeilen if _schluessel(z) not in vorhandene]
    with args.ziel.open("a", encoding="utf-8") as fh:
        for z in neu:
            fh.write(json.dumps(z, ensure_ascii=False) + "\n")

    beleg = {
        "schema": SCHEMA,
        "datum": date.today().isoformat(),
        "klasse": args.klasse,
        "quelle": str(args.quelle),
        "ziel": str(args.ziel),
        "quelle_zeilen": len(quelle_zeilen),
        "neu_angehaengt": len(neu),
        "uebersprungen_duplikat": len(quelle_zeilen) - len(neu),
    }
    ziel_pfad = REPORT_DIR / f"{date.today().isoformat()}_klassen_manifest_anhaengen_{args.klasse}.json"
    ziel_pfad.write_text(json.dumps(beleg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Klasse {args.klasse}: {len(neu)} Zeilen angehängt ({len(quelle_zeilen) - len(neu)} Duplikate übersprungen). Beleg: {ziel_pfad}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
