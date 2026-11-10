#!/usr/bin/env python3
"""bewerte_schadens_proben — objektiver Sinn-Beweis der Restauration (2026-11-11).

Die Schadens-Proben (`scripts/erstelle_schadens_proben.py`) liefern je Titel
drei Punkte: **clean** (Original), **Schaden** (deterministisch zugefügt) und
**restauriert** (Aurik-Produktionspfad). Dieses Werkzeug beantwortet die
einzige Frage, die zählt: *Wird der zugefügte Schaden objektiv kleiner?*

Messgrößen: `defektlast` (niedriger = besser) und `musik_mos` (höher = besser) —
beide richtungszertifiziert (Spearman 0,95). Beide sind **pegelabhängig**;
deshalb wird der restaurierte Pegel standardmäßig auf den Schadenspegel
gematcht (`--kein-pegel-matching` schaltet das ab). Ohne diese Kontrolle misst
man die Ziel-Lautheits-Regelung der Pipeline statt ihrer Reparaturgüte
(Lehre aus D-K3-94: erst Mess-Artefakte ausschließen, dann urteilen).

Ehrliche Grenze: Die Proben sind 10-s-Mittelausschnitte. Das Ergebnis ist ein
Befund **über diese Clips**, kein Qualitätsurteil über ganze Titel.

CPU-only (§G5 (copilot-instructions.md)-freundlich): stört eine laufende GPU-Kette nicht. Kein
Flag-Flip, kein Deploy — reine Messung.

Beleg-Schema `aurik.schadensbewertung/1`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SR = 48000
JND_DEFEKTLAST = 0.05  # identisch zu wohlklang_zeuge.JND_TOLERANZ
REZEPTE = ("vinyl", "kassette", "clipping", "mp3_128")


def _metriken(audio: np.ndarray) -> dict[str, float]:
    """defektlast + musik_mos eines Signals (mono, 48 kHz)."""
    from backend.core.dsp.wohlklang_zeuge import defektlast
    from plugins.musik_mos_plugin import get_musik_mos

    x = np.asarray(audio, dtype=np.float32)
    return {
        "defektlast": round(float(defektlast(x, SR)), 5),
        "musik_mos": round(float(get_musik_mos().estimate_mos(x, SR).mos), 4),
    }


def _pegel_matching(audio: np.ndarray, ziel_rms: float) -> np.ndarray:
    """Skaliert `audio` auf `ziel_rms` — entfernt die Ziel-Lautheit als Störgröße."""
    x = np.asarray(audio, dtype=np.float64)
    ist = float(np.sqrt((x**2).mean()))
    if ist <= 1e-12:
        return np.asarray(audio, dtype=np.float32)
    return np.asarray(x * (ziel_rms / ist), dtype=np.float32)


def bewerte_paar(
    clean: np.ndarray,
    schaden: np.ndarray,
    restauriert: np.ndarray,
    *,
    pegel_matching: bool = True,
) -> dict[str, object]:
    """Ein Paar bewerten — Richtung: defektlast runter, musik_mos rauf."""
    s = np.asarray(schaden, dtype=np.float32)
    r = np.asarray(restauriert, dtype=np.float32)
    n = min(s.size, r.size)
    s, r = s[:n], r[:n]
    rms_schaden = float(np.sqrt((s.astype(np.float64) ** 2).mean()))
    k = _metriken(clean)
    d = _metriken(s)
    # Pegel-Matching: die Pipeline regelt auf Ziel-Lautheit; ungematcht würde
    # man diesen Regelkreis statt der Reparatur messen.
    r_matched = _pegel_matching(r, rms_schaden) if pegel_matching else r
    e = _metriken(r_matched)
    d_defektlast = e["defektlast"] - d["defektlast"]
    d_musik_mos = e["musik_mos"] - d["musik_mos"]
    # "besser" verlangt, dass KEINE der beiden Metriken über die JND kippt.
    besser = (d_defektlast <= -JND_DEFEKTLAST and d_musik_mos >= 0.0) or (d_musik_mos > 0.0 and d_defektlast <= 0.0)
    return {
        "clean": k,
        "schaden": d,
        "restauriert": e,
        "delta_defektlast": round(float(d_defektlast), 5),
        "delta_musik_mos": round(float(d_musik_mos), 4),
        "besser": bool(besser),
        "pegel_gain_ist": round(float(np.sqrt((r.astype(np.float64) ** 2).mean())) / max(1e-12, rms_schaden), 4),
    }


def aggregat(zeilen: list[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    """Mittelwerte je Rezept + Trefferzahl."""
    aus: dict[str, dict[str, float | int]] = {}
    for rz in REZEPTE:
        g = [z for z in zeilen if z["rezept"] == rz]
        if not g:
            continue
        aus[rz] = {
            "n": len(g),
            "delta_defektlast": round(float(np.mean([z["delta_defektlast"] for z in g])), 5),
            "delta_musik_mos": round(float(np.mean([z["delta_musik_mos"] for z in g])), 4),
            "besser": int(sum(1 for z in g if z["besser"])),
        }
    return aus


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Schadens-Proben objektiv bewerten (CPU)")
    ap.add_argument(
        "--verzeichnis",
        default=str(Path.home() / "Musik" / "schadens_proben"),
        help="Ordner mit schadens_manifest.json",
    )
    ap.add_argument("--out", default="", help="JSON-Beleg-Ziel (Default: <verzeichnis>/bewertung.json)")
    ap.add_argument("--kein-pegel-matching", action="store_true", help="Rohpegel vergleichen (nicht empfohlen)")
    args = ap.parse_args(argv)

    from scripts.erstelle_schadens_proben import _decode_48k, _mittel_ausschnitt

    verzeichnis = Path(args.verzeichnis)
    manifest_pfad = verzeichnis / "schadens_manifest.json"
    if not manifest_pfad.is_file():
        print(f"FEHLT: {manifest_pfad} — erst scripts/erstelle_schadens_proben.py laufen lassen")
        return 2
    manifest = json.loads(manifest_pfad.read_text(encoding="utf-8"))
    songs = manifest.get("songs") or []
    if not songs:
        print("LEER: keine Paare im Manifest")
        return 2

    import soundfile as sf

    zeilen: list[dict[str, object]] = []
    for eintrag in songs:
        if not eintrag.get("restauriert"):
            continue  # --nur-schaden-Lauf: kein Aurik-Ergebnis vorhanden
        clean = _mittel_ausschnitt(_decode_48k(Path(eintrag["quelle"])), float(manifest.get("sekunden", 10.0)))
        schaden = np.asarray(sf.read(eintrag["schaden"], always_2d=False)[0], dtype=np.float32)
        rest = np.asarray(sf.read(eintrag["restauriert"], always_2d=False)[0], dtype=np.float32)
        n = min(clean.size, schaden.size)
        ergebnis = bewerte_paar(clean[:n], schaden, rest, pegel_matching=not args.kein_pegel_matching)
        zeile = {"song": eintrag.get("stamm", ""), "rezept": eintrag.get("rezept", ""), **ergebnis}
        zeilen.append(zeile)
        print(
            f"{str(zeile['song'])[:34]:34s} {str(zeile['rezept']):9s} "
            f"Δdefektlast {ergebnis['delta_defektlast']:+.3f}  Δmusik_mos {ergebnis['delta_musik_mos']:+.2f}  "
            f"{'BESSER' if ergebnis['besser'] else 'schlechter/neutral'}"
        )

    if not zeilen:
        print("KEINE bewertbaren Paare (alle ohne Restauration)")
        return 2
    agg = aggregat(zeilen)
    beleg = {
        "schema": "aurik.schadensbewertung/1",
        "pegel_matching": not args.kein_pegel_matching,
        "jnd_defektlast": JND_DEFEKTLAST,
        "n_paare": len(zeilen),
        "n_besser": int(sum(1 for z in zeilen if z["besser"])),
        "aggregat": agg,
        "zeilen": zeilen,
    }
    ziel = Path(args.out) if args.out else verzeichnis / "bewertung.json"
    ziel.write_text(json.dumps(beleg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nBeleg: {ziel}")
    print(f"Besser: {beleg['n_besser']}/{beleg['n_paare']} Paare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
