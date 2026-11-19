"""scripts/erstelle_m9_exclude.py — Leak-Schutz-Liste für den M9-Neulauf.

Der M9-Neulauf trainiert auf `data/tfgrid_pairs_erweitert/` (2.000 Paare,
Seed 42). Die 3 Gate-Tracks des M9-Gates (`2026-11-14_m9_gate.json`,
`tracks`-Feld) sind in diesen Paaren enthalten (gemessen 2026-11-14) — ohne
Ausschluss wäre der nächste Gate-Lauf nicht leak-frei. Dieses Skript bildet
Track-Namen über das Paar-Manifest (``quelle`` → ``datei``) auf Paar-Dateien
ab und schreibt die Ausschlussliste (ein Dateiname je Zeile) für
``train_tf_gridnet_denoiser.py --exclude-pairs``.

Deterministisch: sortierte Ausgabe, keine Zufallsquelle (§G5 (copilot-instructions.md)).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_MANIFEST = _ROOT / "data" / "tfgrid_pairs_erweitert" / "manifest.json"
DEFAULT_GATE = _ROOT / "docs" / "reports" / "current" / "2026-11-14_m9_gate.json"
DEFAULT_OUT = _ROOT / "data" / "tfgrid_pairs_erweitert" / "exclude_pairs.txt"


def erstelle_exclude_liste(manifest: Path, gate: Path) -> list[str]:
    if not manifest.is_file():
        raise FileNotFoundError(f"Paar-Manifest fehlt: {manifest}")
    if not gate.is_file():
        raise FileNotFoundError(f"Gate-Beleg fehlt: {gate}")
    pairs = json.loads(manifest.read_text(encoding="utf-8"))
    gate_daten = json.loads(gate.read_text(encoding="utf-8"))
    tracks = [str(t) for t in gate_daten.get("tracks", [])]
    if not tracks:
        raise ValueError(f"Gate-Beleg ohne tracks-Feld: {gate}")
    # quelle im Manifest: "train/<Track>/mixture.wav" — Track-Name = zweites Segment
    treffer: set[str] = set()
    for e in pairs.get("paare", []):
        quelle = str(e.get("quelle", ""))
        segmente = quelle.split("/")
        track_name = segmente[1] if len(segmente) > 1 else quelle
        if any(t.lower() in track_name.lower() for t in tracks):
            treffer.add(str(e["datei"]))
    if not treffer:
        raise ValueError("kein Paar passt zu den Gate-Tracks — Ausschlussliste wäre leer")
    return sorted(treffer)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    try:
        liste = erstelle_exclude_liste(args.manifest, args.gate)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ABBRUCH: {exc}")
        return 2
    args.out.write_text("\n".join(liste) + "\n", encoding="utf-8")
    print(f"{len(liste)} Paare ausgeschlossen → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
