#!/usr/bin/env python
"""Gepaarte DRT-12-Nachmessung (D-K3-GoNoGo-Kontrolle, 2026-11-24/25).

Pro Paar laufen BESTAND- und KANALSATZ-Pass im selben Prozess; zusätzlich ein
zweiter BESTAND-Pass (B1/K/B2) als Stabilitätskontrolle — weichen die
Bestand-Pässe ab, wird das Paar als instabil geflaggt statt gewertet.
Cache-Busting: eindeutiger input_path je Pass (Bridge-Voranalyse-Cache).
Stromausfall-Resilienz (25.11.): Ergebnis-JSONL liegt persistent unter data/
und fertige Paare werden beim Neustart übersprungen (Resume).

Verwendung:
  python3 -B scripts/gepaarte_messung.py --seed 42 --paare 45 \
      --rezepte dropout,kassette,hum,wowflutter,kompression,clicks \
      --weizen-flags <17 Flags, Komma-getrennt>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("AURIK_FORCE_CPU", "1")
os.environ.setdefault("AURIK_NO_RT_LIMIT", "1")

import numpy as np
from scipy.io import wavfile

from scripts.erstelle_schadens_proben import restauriere_schnipsel
from scripts.messe_kanalsatz_ab import (
    KANALSAETZ_FLAGS,
    _paare_auswaehlen,
    _parse_flags,
    _zeugen,
)

_ROOT = Path(__file__).resolve().parent.parent


def _setze_flags(aktiv: list[str]) -> None:
    import backend.core.music_model_flags as flags

    for name in set(KANALSAETZ_FLAGS) | set(aktiv):
        setattr(flags, name, name in aktiv)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--paare", type=int, default=45)
    ap.add_argument("--rezepte", default=None, help="Komma-Liste; nur Paare mit diesen Rezept-Labels")
    ap.add_argument("--weizen-flags", required=True, help="Komma-Liste der aktivierten Weizen-Flags")
    ap.add_argument("--ausgang", default=str(_ROOT / "data" / "gepaarte_messung.jsonl"))
    ap.add_argument("--verzeichnis", default="data/schadens_proben")
    args = ap.parse_args(argv)

    weizen = _parse_flags(args.weizen_flags)
    rezepte = [r.strip() for r in (args.rezepte or "").split(",") if r.strip()]
    paare = _paare_auswaehlen(Path(args.verzeichnis), args.paare, args.seed)
    auswahl = []
    for stamm, pfad in paare:
        if not rezepte or any(f"({r})" in stamm for r in rezepte):
            auswahl.append((stamm, pfad))
    print(f"GEPAART: {len(auswahl)} Paare ausgewählt (Rezepte: {rezepte or 'alle'})", flush=True)

    ausgang = Path(args.ausgang)
    fertig: set[str] = set()
    _kaputte_zeilen = 0
    if ausgang.exists():
        for line in ausgang.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                fertig.add(json.loads(line)["stamm"])
            except Exception as _exc:
                _kaputte_zeilen += 1
                print(f"WARNUNG: unlesbare Resume-Zeile übersprungen ({_exc}) — Zeile {line[:60]!r}", flush=True)
    if _kaputte_zeilen:
        print(f"RESUME: {_kaputte_zeilen} unlesbare Zeile(n) ignoriert", flush=True)
    if fertig:
        print(f"RESUME: {len(fertig)} Paar(e) bereits fertig — übersprungen", flush=True)

    for i, (stamm, pfad) in enumerate(auswahl):
        if stamm in fertig:
            continue
        roh = wavfile.read(pfad)
        sr = int(roh[0])
        d = (
            np.asarray(roh[1], np.float32) / 32768.0
            if np.asarray(roh[1]).dtype != np.float32
            else np.asarray(roh[1], np.float32)
        )
        schaden = d.mean(axis=1) if d.ndim == 2 else d

        # Cache-Busting (D-K3-GoNoGo-Folge): eindeutiger input_path je Pass.
        _setze_flags([])  # B1
        bestand_out = restauriere_schnipsel(schaden, sr, f"{pfad}#b1")
        bz = _zeugen(schaden, bestand_out, sr)

        _setze_flags(weizen)  # K
        kanal_out = restauriere_schnipsel(schaden, sr, f"{pfad}#k")
        kz = _zeugen(schaden, kanal_out, sr)

        _setze_flags([])  # B2 (Stabilitätskontrolle)
        bestand_out2 = restauriere_schnipsel(schaden, sr, f"{pfad}#b2")
        bz2 = _zeugen(schaden, bestand_out2, sr)
        flip = (
            abs(bz["delta_musik_mos"] - bz2["delta_musik_mos"]) > 0.05
            or abs(bz["delta_defektlast"] - bz2["delta_defektlast"]) > 0.05
        )

        zeile = {
            "stamm": stamm,
            "status": "instabil_d_k3_gonogo" if flip else "ok",
            "bestand_stabilitaet": {
                "mos_b1": bz["delta_musik_mos"],
                "mos_b2": bz2["delta_musik_mos"],
                "dl_b1": bz["delta_defektlast"],
                "dl_b2": bz2["delta_defektlast"],
            },
            "bestand_delta_musik_mos": round(bz["delta_musik_mos"], 4),
            "bestand_delta_defektlast": round(bz["delta_defektlast"], 4),
            "bestand_ueberrestauration": bz["ueberrestauration"],
            "kanalsatz_delta_musik_mos": round(kz["delta_musik_mos"], 4),
            "kanalsatz_delta_defektlast": round(kz["delta_defektlast"], 4),
            "kanalsatz_ueberrestauration": kz["ueberrestauration"],
            "delta_zu_bestand_mos": round(kz["delta_musik_mos"] - bz["delta_musik_mos"], 4),
            "delta_zu_bestand_defektlast": round(kz["delta_defektlast"] - bz["delta_defektlast"], 4),
            "weizen_flags": weizen,
        }
        with ausgang.open("a", encoding="utf-8") as f:
            f.write(json.dumps(zeile, ensure_ascii=False) + "\n")
        print(
            f"[{i + 1}/{len(auswahl)}] {stamm} — B: MOS {bz['delta_musik_mos']:+.3f} DL {bz['delta_defektlast']:+.3f}"
            f" | K: MOS {kz['delta_musik_mos']:+.3f} DL {kz['delta_defektlast']:+.3f}"
            f" | ΔBestand: MOS {zeile['delta_zu_bestand_mos']:+.3f} DL {zeile['delta_zu_bestand_defektlast']:+.3f}"
            f" | {'INSTABIL' if flip else 'stabil'}",
            flush=True,
        )
    print("GEPAART_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
