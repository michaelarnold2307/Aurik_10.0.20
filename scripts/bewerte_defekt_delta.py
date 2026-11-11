#!/usr/bin/env python3
"""Defekt-Delta-Bewertung — CLI für den Report-Only-Zeugen (§G191 (GEBOTE.md)).

Misst Metriken nur in den annotierten Defektregionen, vorher (Schaden) und
nachher (restauriert), pegel-gematcht, mit Reparaturquote Q je Region und
Defektklasse. Schreibziel: Beleg ``aurik.defektdelta/1`` (kein Gate, kein
Flag, keine Stärke-Änderung — Hörordnung §7, Report-Only-Zeuge).

Eingaben:
  --schaden / --restauriert / --clean   WAV-Dateien des Schadens-Tripels
  --regionen  JSON-Datei: Liste von {"start": s, "end": e, "typ": "...",
              "staerke": w} in Samples (Vertrag mit DefectScanner-Annotationen)
  --sr        Sample-Rate (Default 44100)
  --metrik    defektlast_lang (Default) | musik_mos
  --out       Beleg-Zielpfad (Default: docs/reports/current/b3_defektdelta.json)

Exit-Codes: 0 = Beleg geschrieben; 2 = Eingabe fehlt/unlesbar (fail-closed);
3 = ungültige Regionen-/Schema-Daten. §G5 (copilot-instructions.md):
deterministisch, sha256 der Eingänge, kein time.time().
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _wav_lesen(path: Path) -> tuple[object, int]:
    """Liest WAV (soundfile) und liefert (audio, sample_rate) — fail-closed."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return audio, int(sr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Defekt-Delta-Witness (§G191 (GEBOTE.md))")
    parser.add_argument("--schaden", required=True)
    parser.add_argument("--restauriert", required=True)
    parser.add_argument("--clean", default=None)
    parser.add_argument("--regionen", required=True)
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--metrik", choices=["defektlast_lang", "musik_mos"], default="defektlast_lang")
    parser.add_argument(
        "--out",
        default=str(ROOT / "docs" / "reports" / "current" / "b3_defektdelta.json"),
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    # §G191 (GEBOTE.md) — Report-Only-Zeuge, kein Gate.
    from backend.core.dsp.defekt_delta import beleg_schreiben, berechne_defekt_delta

    schaden_pfad = Path(args.schaden)
    rest_pfad = Path(args.restauriert)
    clean_pfad = Path(args.clean) if args.clean else None
    regionen_pfad = Path(args.regionen)

    for pfad in (schaden_pfad, rest_pfad, regionen_pfad) + ((clean_pfad,) if clean_pfad else ()):
        if not pfad.exists():
            print(f"FEHLER: Eingabe fehlt: {pfad}", file=sys.stderr)
            return 2

    try:
        audio_schaden, sr_ist = _wav_lesen(schaden_pfad)
        audio_rest, _ = _wav_lesen(rest_pfad)
        audio_clean = _wav_lesen(clean_pfad)[0] if clean_pfad else None
    except Exception as exc:  # §V6 (copilot-instructions.md): kein stiller Ausfall
        print(f"FEHLER: WAV nicht lesbar: {exc}", file=sys.stderr)
        return 2

    if sr_ist != args.sr:
        print(
            f"WARNUNG: WAV-Sample-Rate {sr_ist} != --sr {args.sr} — Metrik läuft mit {sr_ist}.",
            file=sys.stderr,
        )
        args.sr = sr_ist

    try:
        regionen = json.loads(regionen_pfad.read_text(encoding="utf-8"))
        if not isinstance(regionen, list):
            raise ValueError("Regionen-Datei muss eine JSON-Liste sein")
    except Exception as exc:
        print(f"FEHLER: Regionen nicht lesbar: {exc}", file=sys.stderr)
        return 3

    try:
        beleg = berechne_defekt_delta(
            regionen=regionen,
            schaden=audio_schaden,
            restauriert=audio_rest,
            clean=audio_clean,
            sample_rate=args.sr,
            metrik_name=args.metrik,
        )
    except ValueError as exc:
        print(f"FEHLER: Bewertung nicht möglich: {exc}", file=sys.stderr)
        return 3

    quellen: dict[str, str] = {
        "schaden": str(schaden_pfad),
        "restauriert": str(rest_pfad),
        "regionen": str(regionen_pfad),
    }
    if clean_pfad is not None:
        quellen["clean"] = str(clean_pfad)

    out_pfad = Path(args.out)
    out_pfad.parent.mkdir(parents=True, exist_ok=True)
    beleg_schreiben(beleg, str(out_pfad), quellen)
    print(f"Defekt-Delta-Beleg geschrieben: {out_pfad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
