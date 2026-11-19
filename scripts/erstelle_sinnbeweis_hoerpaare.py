#!/usr/bin/env python3
"""Sinn-Beweis-§4-Hörpaket-Erzeuger (2026-11-18).

Baut aus einem fertigen Schadens-Proben-Ordner (schadens_manifest.json +
bewertung.json) ein blindes A/B-Hörpaket im Muster des Declip-Pakets:
  - A = beschädigter 10-s-Ausschnitt, B = Aurik-Restauration (Reihenfolge
    je Paar per Seed vertauscht, blind),
  - p25-Pegelmatching der Restauration auf den Schaden (Hörordnung §7:
    Lautheit entscheidet nicht),
  - Schlüssel getrennt (schluessel.json, öffnen erst NACH dem Urteil),
  - Manifest aurik.hoerpaare/1; deterministisch (§G5 (GEBOTE.md)).
Verwendung z.B. für die Register-Zeilen Dereverb (raum) und Stereo-Imaging
(stereo_imbalance) — beide haben GESICHERT-/Beleg-Status und brauchen noch
den menschlichen §4-Hörvergleich.

Usage:
    python3 -B scripts/erstelle_sinnbeweis_hoerpaare.py \\
        --verzeichnis data/schadens_proben_raum10 --rezepte raum \\
        --paare 6 --ausgang study_stimuli/hoerproben/dereverb_sinnbeweis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent

SEED = 42


def _p25(x: np.ndarray) -> float:
    """Body-Perzentil-Pegelreferenz (p25, kanonisches Pegelmatching der Hörordnung §7)."""
    return float(np.percentile(np.abs(x), 25))


def _match_p25(quelle: np.ndarray, ziel: np.ndarray) -> np.ndarray:
    """Skaliert quelle auf den p25-Pegel von ziel (verhindert Lautheits-Bias)."""
    p_q = _p25(quelle)
    p_z = _p25(ziel)
    if p_q <= 1e-12 or p_z <= 1e-12:
        return np.asarray(quelle, dtype=np.float32)
    return np.asarray(quelle * (p_z / p_q), dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verzeichnis", required=True, help="Schadens-Proben-Ordner mit schadens_manifest.json")
    ap.add_argument("--rezepte", default="raum", help="Komma-Liste der Rezepte für das Paket")
    ap.add_argument("--paare", type=int, default=6, help="A/B-Paare insgesamt")
    ap.add_argument("--ausgang", required=True, help="Ziel-Ordner (Panel-Layout)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    verzeichnis = Path(args.verzeichnis)
    ausgang = Path(args.ausgang)
    manifest_path = verzeichnis / "schadens_manifest.json"
    if not manifest_path.is_file():
        print(f"FEHLT: {manifest_path} — erst erstelle_schadens_proben.py laufen lassen")
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    songs = manifest.get("songs") or []
    rezepte = {r.strip() for r in args.rezepte.split(",") if r.strip()}
    kandidaten = [s for s in songs if s.get("rezept") in rezepte]
    if len(kandidaten) < args.paare:
        print(f"Zu wenige Paare für {sorted(rezepte)}: {len(kandidaten)} < {args.paare}")
        return 2

    rng = np.random.default_rng(args.seed)
    gewaehlt = [kandidaten[i] for i in rng.choice(len(kandidaten), size=args.paare, replace=False)]

    ausgang.mkdir(parents=True, exist_ok=True)
    # Fail-closed: vorhandene A/B-Dateien nie überschreiben (PCHIP-Schutz-Muster).
    for s in gewaehlt:
        stamm = s["stamm"]
        if (ausgang / f"{stamm}_A.wav").exists() or (ausgang / f"{stamm}_B.wav").exists():
            print(f"ABBRUCH: {stamm} existiert bereits in {ausgang} — kein Überschreiben")
            return 2

    schluessel: dict[str, str] = {}
    for s in gewaehlt:
        stamm = s["stamm"]
        schaden_path = Path(s["schaden"])
        restauriert_path = Path(s["restauriert"])
        schaden, sr = sf.read(str(schaden_path), always_2d=False, dtype="float32")
        restauriert, _ = sf.read(str(restauriert_path), always_2d=False, dtype="float32")
        restauriert = _match_p25(restauriert, schaden)
        # Blind: Reihenfolge je Paar würfeln (deterministisch, Seed)
        if rng.random() < 0.5:
            a_name, b_name = f"{stamm}_A.wav", f"{stamm}_B.wav"
            a_sig, b_sig = schaden, restauriert
            schluessel[f"{stamm}_A.wav"] = "schaden"
            schluessel[f"{stamm}_B.wav"] = "restauriert"
        else:
            a_name, b_name = f"{stamm}_A.wav", f"{stamm}_B.wav"
            a_sig, b_sig = restauriert, schaden
            schluessel[f"{stamm}_A.wav"] = "restauriert"
            schluessel[f"{stamm}_B.wav"] = "schaden"
        sf.write(str(ausgang / a_name), a_sig, sr, subtype="FLOAT")
        sf.write(str(ausgang / b_name), b_sig, sr, subtype="FLOAT")
        p25a = _p25(a_sig)
        p25b = _p25(b_sig)
        print(f"{stamm}: geschrieben, p25-rel A/B {((p25b - p25a) / max(p25a, 1e-12)) * 100:+.2f} %")

    paket_manifest = {
        "schema": "aurik.hoerpaare/1",
        "zweck": "Sinn-Beweis-§4-Hörpaket: Schaden (A) vs. Aurik-Restauration (B), blind",
        "seed": args.seed,
        "paare": len(gewaehlt),
        "rezepte": sorted(rezepte),
        "pegel_matching": "p25 (Hörordnung §7)",
        "hinweis": "Hör-Entscheidung menschlich — Schlüssel erst nach dem Urteil öffnen.",
    }
    (ausgang / "manifest.json").write_text(
        json.dumps(paket_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (ausgang / "schluessel.json").write_text(
        json.dumps(schluessel, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Manifest + Schlüssel: {ausgang}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
