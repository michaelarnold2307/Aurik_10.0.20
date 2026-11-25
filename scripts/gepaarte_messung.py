#!/usr/bin/env python
"""Gepaarte DRT-12-Nachmessung (D-K3-GoNoGo-Kontrolle, 2026-11-24/25).

Pro Paar laufen BESTAND- und KANALSATZ-Pass sowie ein zweiter BESTAND-Pass
(B1/K/B2) — JEDER Pass in einem EIGENEN Child-Prozess (Befund 25.11.: der
In-Prozess-Wiederverwendungs-Drift ist ~100× größer als die Rest-Varianz
frischer Prozesse). Weichen die Bestand-Pässe ab, wird das Paar als instabil
geflaggt statt gewertet. Cache-Busting: eindeutiger input_path je Pass.
Stromausfall-Resilienz: Ergebnis-JSONL persistent unter data/, fertige Paare
werden beim Neustart übersprungen.

Verwendung:
  python3 -B scripts/gepaarte_messung.py --seed 42 --paare 45 \
      --rezepte dropout,kassette,hum,wowflutter,kompression,clicks \
      --weizen-flags <17 Flags, Komma-getrennt>
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("AURIK_FORCE_CPU", "1")
os.environ.setdefault("AURIK_NO_RT_LIMIT", "1")

from scripts.messe_kanalsatz_ab import (  # noqa: E402
    KANALSAETZ_FLAGS,
    _paare_auswaehlen,
    _parse_flags,
)

_PASS_KIND = r"""
import json, os, sys
import numpy as np
from scipy.io import wavfile

os.environ.setdefault("AURIK_FORCE_CPU", "1")
os.environ.setdefault("AURIK_NO_RT_LIMIT", "1")

from scripts.messe_kanalsatz_ab import KANALSAETZ_FLAGS, _parse_flags, _zeugen
from scripts.erstelle_schadens_proben import restauriere_schnipsel


def _setze_flags(aktiv: list[str]) -> None:
    import backend.core.music_model_flags as flags

    for name in set(KANALSAETZ_FLAGS) | set(aktiv):
        setattr(flags, name, name in aktiv)


def main() -> int:
    pfad = sys.argv[1]
    ausgang = sys.argv[2]
    flags_roh = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else ""
    roh = wavfile.read(pfad)
    sr = int(roh[0])
    d = np.asarray(roh[1], np.float32) / 32768.0 if np.asarray(roh[1]).dtype != np.float32 else np.asarray(roh[1], np.float32)
    schaden = d.mean(axis=1) if d.ndim == 2 else d
    weizen = _parse_flags(flags_roh) if flags_roh else []
    _setze_flags(weizen)
    out = restauriere_schnipsel(schaden, sr, pfad + "#" + ausgang)
    z = _zeugen(schaden, out, sr)
    with open(ausgang, "w", encoding="utf-8") as f:
        json.dump(
            {
                "delta_musik_mos": float(z["delta_musik_mos"]),
                "delta_defektlast": float(z["delta_defektlast"]),
                "ueberrestauration": bool(z["ueberrestauration"]),
            },
            f,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _pass_kind_skript() -> Path:
    pfad = _ROOT / "data" / "_gepaarte_pass_kind.py"
    pfad.write_text(_PASS_KIND, encoding="utf-8")
    return pfad


def _ein_pass(pfad: Path, kind: str, weizen: list[str], kind_skript: Path) -> dict:
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=f".{kind}.zeuge.json", delete=False) as tmp:
        zeuge_pfad = Path(tmp.name)
    try:
        flags_roh = ",".join(weizen) if kind == "k" else "-"
        umgebung = dict(os.environ)
        umgebung["PYTHONPATH"] = str(_ROOT)
        umgebung["AURIK_FORCE_CPU"] = "1"
        umgebung["AURIK_NO_RT_LIMIT"] = "1"
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(kind_skript),
                str(pfad),
                str(zeuge_pfad),
                flags_roh,
            ],
            cwd=str(_ROOT),
            env=umgebung,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return json.loads(zeuge_pfad.read_text(encoding="utf-8"))
    finally:
        zeuge_pfad.unlink(missing_ok=True)


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
    if ausgang.exists():
        for line in ausgang.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                fertig.add(json.loads(line)["stamm"])
            except Exception as exc:  # noqa: BLE001 - Resumme-Zeilen sind Belegdaten
                logging.getLogger(__name__).debug("Überspringe korrupte Resume-Zeile: %s", exc)
    if fertig:
        print(f"RESUME: {len(fertig)} Paar(e) bereits fertig — übersprungen", flush=True)

    kind_skript = _pass_kind_skript()
    for i, (stamm, pfad) in enumerate(auswahl):
        if stamm in fertig:
            continue
        try:
            bz = _ein_pass(pfad, "b1", [], kind_skript)
            kz = _ein_pass(pfad, "k", weizen, kind_skript)
            bz2 = _ein_pass(pfad, "b2", [], kind_skript)
        except Exception as exc:  # noqa: BLE001 - Paar überspringen statt Kampagne abbrechen
            logging.getLogger(__name__).warning("Paar %s fehlgeschlagen (%s) — übersprungen", stamm, exc)
            continue
        flip = (
            abs(bz["delta_musik_mos"] - bz2["delta_musik_mos"]) > 0.05
            or abs(bz["delta_defektlast"] - bz2["delta_defektlast"]) > 0.05
        )
        zeile = {
            "stamm": stamm,
            "status": "instabil_d_k3_gonogo" if flip else "ok",
            "bestand_stabilitaet": {
                "mos_b1": bz["delta_musik_mos"], "mos_b2": bz2["delta_musik_mos"],
                "dl_b1": bz["delta_defektlast"], "dl_b2": bz2["delta_defektlast"],
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
