#!/usr/bin/env python
"""Amtsträger-Hörpaar-Generator (Hörsitzung „Hörbarkeit zuerst", Blöcke B/C).

Erzeugt blinde A/B-Paare für die Hör-Instanz: A-Seite = Produktionspfad mit
dem Amtsträger AKTIV (Status quo), B-Seite = identischer Pfad mit der Phase
per `AURIK_PHASE_AUSLASSEN` DEAKTIVIERT (Pass-Through) — der Ablations-Hebel
aus D-K3-110 Modus B wird damit zum Hörpaar-Werkzeug. Pegelmatching p25
(Hörordnung §7), Seiten per Seed verlost, Schlüssel getrennt in
`schluessel.json`, fail-closed gegen Überschreiben. Konventionen wie
`scripts/erstelle_aspade_hoerpaare.py` (p25/pegel-gematcht WIRD von dort
importiert — eine Kopie, D-K3-7-Lehre).

WICHTIG: NICHT parallel zur DRT-12-Kampagne laufen lassen (Last stört die
Kampagnen-Kinder über die Live-Notbremse).

Verwendung:
  AURIK_FORCE_CPU=1 AURIK_NO_RT_LIMIT=1 .venv_aurik/bin/python -B \
      scripts/erstelle_hoerpaare_amtstraeger.py \
      --kanal phase_12_wow_flutter_fix --rezept wowflutter --paare 6
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("AURIK_FORCE_CPU", "1")
os.environ.setdefault("AURIK_NO_RT_LIMIT", "1")

import numpy as np  # noqa: E402
from scipy.io import wavfile  # noqa: E402

logger = logging.getLogger(__name__)

# Kanonische Amtsträger der Hörsitzung (Blöcke B/C des Plans
# `docs/reports/current/2026-11-25_hoersitzung_hoerbarkeit_zuerst_plan.md`).
AMTSTRAEGER: dict[str, str] = {
    "phase_12_wow_flutter_fix": "wowflutter",
    "phase_02_hum_removal": "hum",
}


def _seiten_zuordnung(rng: np.random.Generator, slug: str) -> str:
    """Deterministische A/B-Seiten-Verlösung (§G5): 'A=amt' oder 'A=pass'."""
    return "A=amt" if bool(rng.integers(0, 2)) else "A=pass"


def _rezept_aus_namen(name: str) -> str:
    stem = name[: -len(".wav")] if name.lower().endswith(".wav") else name
    if "(" in stem and stem.rstrip().endswith(")"):
        return stem.rsplit("(", 1)[1].rstrip(")").strip()
    return "unbekannt"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kanal", default="phase_12_wow_flutter_fix", choices=sorted(AMTSTRAEGER))
    ap.add_argument("--rezept", default="", help="Rezept-Label der Schadens-Proben (Default: aus Kanal)")
    ap.add_argument("--paare", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verzeichnis", default=str(_ROOT / "data" / "schadens_proben"))
    ap.add_argument("--ausgang", default=None, help="Zielordner (Default: study_stimuli/hoerproben/amtstraeger_<kanal>)")
    args = ap.parse_args(argv)

    from scripts.erstelle_aspade_hoerpaare import _pegel_gematcht, _p25  # eine Kopie (D-K3-7)

    rezept = args.rezept or AMTSTRAEGER[args.kanal]
    if not args.ausgang:
        args.ausgang = str(_ROOT / "study_stimuli" / "hoerproben" / f"amtstraeger_{args.kanal}")
    out = Path(args.ausgang)
    out.mkdir(parents=True, exist_ok=True)

    wavs = sorted(Path(args.verzeichnis).glob("*.wav"))
    wavs = [w for w in wavs if _rezept_aus_namen(w.name) == rezept]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(wavs), size=min(args.paare, len(wavs)), replace=False)
    auswahl = [wavs[int(i)] for i in sorted(idx)]
    print(f"HOERPAARE: {len(auswahl)} {rezept}-Titel für Kanal {args.kanal}", flush=True)

    from scripts.erstelle_schadens_proben import restauriere_schnipsel

    schluessel: dict[str, str] = {}
    for n, pfad in enumerate(auswahl):
        slug = pfad.stem
        ziel_a = out / f"{slug}_A.wav"
        ziel_b = out / f"{slug}_B.wav"
        if ziel_a.exists() or ziel_b.exists():
            raise FileExistsError(f"Überschreiben verboten: {ziel_a.name}/{ziel_b.name} existieren — fail-closed")
        roh = wavfile.read(str(pfad))
        sr = int(roh[0])
        d = np.asarray(roh[1], np.float32)
        if d.dtype != np.float32:
            d = d / 32768.0
        if d.ndim == 2:
            d = d.mean(axis=1)
        amt = restauriere_schnipsel(d, sr, f"{pfad}#amt")
        alt_env = os.environ.get("AURIK_PHASE_AUSLASSEN")
        os.environ["AURIK_PHASE_AUSLASSEN"] = args.kanal
        try:
            pass_lauf = restauriere_schnipsel(d, sr, f"{pfad}#pass")
        finally:
            if alt_env is None:
                os.environ.pop("AURIK_PHASE_AUSLASSEN", None)
            else:
                os.environ["AURIK_PHASE_AUSLASSEN"] = alt_env
        amt = _pegel_gematcht(d, amt)
        pass_lauf = _pegel_gematcht(d, pass_lauf)
        a_ist_amt = _seiten_zuordnung(rng, slug) == "A=amt"
        wavfile.write(str(ziel_a), sr, ((amt if a_ist_amt else pass_lauf) * 32768.0).astype(np.int16))
        wavfile.write(str(ziel_b), sr, ((pass_lauf if a_ist_amt else amt) * 32768.0).astype(np.int16))
        schluessel[slug] = "A=amtstraeger_aktiv" if a_ist_amt else "A=pass_through"
        print(
            f"[{n + 1}/{len(auswahl)}] {slug} — p25_in={_p25(d):.3f} p25_amt={_p25(amt):.3f} "
            f"p25_pass={_p25(pass_lauf):.3f} ({schluessel[slug]})",
            flush=True,
        )
    (out / "schluessel.json").write_text(json.dumps(schluessel, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"HOERPAARE_DONE: {len(auswahl)} Paare nach {out} (Schlüssel getrennt in schluessel.json)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
