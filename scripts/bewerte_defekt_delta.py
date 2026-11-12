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
from typing import Any

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
    regionen_gruppe = parser.add_mutually_exclusive_group(required=True)
    regionen_gruppe.add_argument("--regionen", default=None, help="Regionen-JSON (Vertrag Spec 07 §8.6)")
    regionen_gruppe.add_argument(
        "--scanne-regionen",
        default=None,
        help="WAV, dessen DefectScanner-Scores die Regionen liefern (Kontrollregionen automatisch, §G191 (GEBOTE.md))",
    )
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--metrik", choices=["defektlast_lang", "musik_mos"], default="defektlast_lang")
    parser.add_argument(
        "--out",
        default=str(ROOT / "docs" / "reports" / "current" / "b3_defektdelta.json"),
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    # §G191 (GEBOTE.md) — Report-Only-Zeuge, kein Gate.
    from backend.core.dsp.defekt_delta import (
        beleg_schreiben,
        berechne_defekt_delta,
        kontrollregionen_aus_abdeckung,
        regionen_aus_defekt_scores,
    )

    schaden_pfad = Path(args.schaden)
    rest_pfad = Path(args.restauriert)
    clean_pfad = Path(args.clean) if args.clean else None

    eingabe_pfade = [schaden_pfad, rest_pfad]
    if clean_pfad is not None:
        eingabe_pfade.append(clean_pfad)
    if args.regionen:
        eingabe_pfade.append(Path(args.regionen))
    if args.scanne_regionen:
        eingabe_pfade.append(Path(args.scanne_regionen))
    for pfad in eingabe_pfade:
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

    regionen_quelle: dict[str, Any] = {}
    if args.regionen:
        regionen_pfad = Path(args.regionen)
        try:
            regionen = json.loads(regionen_pfad.read_text(encoding="utf-8"))
            if not isinstance(regionen, list):
                raise ValueError("Regionen-Datei muss eine JSON-Liste sein")
        except Exception as exc:
            print(f"FEHLER: Regionen nicht lesbar: {exc}", file=sys.stderr)
            return 3
        regionen_quelle = {"quelle": "regionen_datei", "pfad": str(regionen_pfad)}
    else:
        scan_pfad = Path(args.scanne_regionen)
        try:
            audio_scan, sr_scan = _wav_lesen(scan_pfad)
            from backend.core.defect_scanner import DefectScanner

            scan_ergebnis = DefectScanner().scan(audio_scan, sample_rate=int(sr_scan))
            regionen_defekt = regionen_aus_defekt_scores(scan_ergebnis.scores, int(sr_scan))
            regionen = regionen_defekt + kontrollregionen_aus_abdeckung(
                regionen_defekt, int(audio_scan.shape[-1]), int(sr_scan)
            )
        except Exception as exc:  # §V6 (copilot-instructions.md): kein stiller Ausfall
            print(f"FEHLER: Scan/Regionen-Extraktion nicht möglich: {exc}", file=sys.stderr)
            return 3
        regionen_quelle = {
            "quelle": "defect_scanner",
            "pfad": str(scan_pfad),
            "anzahl_defektregionen": len(regionen_defekt),
            "anzahl_kontrollregionen": len(regionen) - len(regionen_defekt),
        }

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

    beleg["regionen_quelle"] = regionen_quelle

    quellen: dict[str, str] = {
        "schaden": str(schaden_pfad),
        "restauriert": str(rest_pfad),
    }
    if clean_pfad is not None:
        quellen["clean"] = str(clean_pfad)
    if args.regionen:
        quellen["regionen"] = str(regionen_pfad)
    else:
        quellen["scan_wav"] = str(Path(args.scanne_regionen))

    out_pfad = Path(args.out)
    out_pfad.parent.mkdir(parents=True, exist_ok=True)
    beleg_schreiben(beleg, str(out_pfad), quellen)
    print(f"Defekt-Delta-Beleg geschrieben: {out_pfad}")
    # §G13 (copilot-instructions.md): Q-Ziel je Defektklasse sichtbar machen.
    from backend.core.dsp.defekt_delta import REPARATURQUOTE_ZIEL

    print(f"Q-Ziel je Defektklasse (>= {REPARATURQUOTE_ZIEL}):")
    for typ, agg in sorted((beleg.get("aggregate") or {}).items()):
        q = agg.get("reparaturquote_median")
        erfuellt = agg.get("q_ziel_erfuellt")
        status = "erfuellt" if erfuellt else ("verfehlt" if erfuellt is False else "nicht bewertbar")
        print(f"  {typ}: Q={q} -> {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
