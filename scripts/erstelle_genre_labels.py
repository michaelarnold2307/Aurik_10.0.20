"""scripts/erstelle_genre_labels.py — Genre-Labels eigenständig erzeugen (Datenlücke „Genre-Labels").

Hintergrund (2026-11-20, User-Vorgabe „externes Material eigenständig beschaffen"):
Die Produktreife-Liste nennt „Genre-Labels" als offene Datenlücke. Musik-fremde,
lizenzpflichtige Labelsätze (FMA/MedleyDB sind CC BY-NC) dürfen nicht eingekauft
werden (§9/7); stattdessen werden die eigenen Quellen (MUSDB18-HQ-Train + die
Seed-42-Auswahl data/schadens_quellen) **maschinell selbst gelabelt** — mit dem
Produktions-Klassifikator (`backend.core.genre_classifier`, PANNs-Genre-Prior +
Schlager-Speziallogik + Open-Set-Erkennung), 20-s-Mittelausschnitt je Titel,
deterministisch (Seed 42). Die Labels sind maschinell (confidence dokumentiert)
und ersetzen kein menschliches Urteil — sie schließen die Datenlücke „Label-
Existenz" für Genre-konditionierte Messungen.

Beleg: `docs/reports/current/<datum>_genre_labels.json` + `data/genre_labels.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SCHEMA = "aurik.genre_labels/1"
REPORT_DIR = _ROOT / "docs" / "reports" / "current"
DEFAULT_MUSDB = _ROOT / "data" / "musdb18hq" / "train"
DEFAULT_QUELLEN = _ROOT / "data" / "schadens_quellen"
SEKUNDEN = 20.0
SR = 48000


def _mittel_ausschnitt(x: np.ndarray, sekunden: float, sr: int) -> np.ndarray:
    n = int(sekunden * sr)
    if x.shape[-1] <= n:
        return x
    start = (x.shape[-1] - n) // 2
    return x[..., start : start + n]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Genre-Labels maschinell erzeugen")
    ap.add_argument("--musdb", type=Path, default=DEFAULT_MUSDB)
    ap.add_argument("--quellen", type=Path, default=DEFAULT_QUELLEN)
    ap.add_argument("--sekunden", type=float, default=SEKUNDEN)
    args = ap.parse_args(argv)

    import soundfile as sf

    from backend.core.genre_classifier import get_genre_classifier  # type: ignore[import-untyped]

    klassifizierer = get_genre_classifier()
    eintraege = []

    quellen: list[tuple[str, Path]] = []
    if args.musdb.is_dir():
        quellen += [(p.parent.name, p) for p in sorted(args.musdb.glob("*/mixture.wav"))]
    if args.quellen.is_dir():
        quellen += [(p.stem, p) for p in sorted(args.quellen.glob("*.wav"))]

    for titel, pfad in quellen:
        try:
            x, sr = sf.read(str(pfad), dtype="float32", always_2d=False)
            if x.ndim == 2:
                x = x.mean(axis=1)
            ausschnitt = _mittel_ausschnitt(np.asarray(x, np.float32), args.sekunden, sr)
            erg = klassifizierer.classify(ausschnitt, sr)
            eintraege.append(
                {
                    "titel": titel,
                    "pfad": str(pfad.relative_to(_ROOT)),
                    "genre": str(getattr(erg, "primary_genre_label", "unknown")),
                    "confidence": round(float(getattr(erg, "primary_genre_confidence", 0.0)), 4),
                    "familie": str(getattr(erg, "genre_family", "")),
                }
            )
        except Exception as exc:
            eintraege.append(
                {"titel": titel, "pfad": str(pfad), "genre": None, "confidence": None, "fehler": str(exc)[:200]}
            )

    ziel_jsonl = _ROOT / "data" / "genre_labels.jsonl"
    with ziel_jsonl.open("w", encoding="utf-8") as fh:
        for e in eintraege:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    beleg = {
        "schema": SCHEMA,
        "datum": date.today().isoformat(),
        "n_titel": len(eintraege),
        "sekunden_ausschnitt": args.sekunden,
        "methode": "maschinell (genre_classifier, PANNs-Prior + Schlager-Speziallogik, Seed-deterministisch)",
        "eintraege": eintraege,
    }
    ziel = REPORT_DIR / f"{date.today().isoformat()}_genre_labels.json"
    ziel.write_text(json.dumps(beleg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(eintraege)} Titel gelabelt. Beleg: {ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
