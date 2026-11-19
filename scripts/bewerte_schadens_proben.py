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

Stereo (Paritäts-Register „Stereo-Imaging/Azimuth“): Für das Rezept
`stereo_imbalance` misst zusätzlich `stereo_balance_ratio` =
RMS(links)/RMS(rechts) in dB (JND 0,5 dB). Richtung: der Schaden entfernt sich
vom Quell-Wert, die Restauration nähert sich an; diese Balance-Richtung zählt
als **zusätzliches** besser-Kriterium. defektlast/musik_mos bleiben für alle
Rezepte unverändert auf dem Mono-Mixdown (§III (copilot-instructions.md)
Stereo-Layout-Invariante, intern channels-first (C, N)).

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
JND_MOS = 0.05  # DRT-10 (Spec 26, 2026-11-15): musik_mos-JND für die Überrestaurations-Erkennung
JND_BALANCE_DB = 0.5  # Paritäts-Register „Stereo-Imaging/Azimuth“: Inter-Kanal-Pegeldifferenz-JND der Balance-Richtung
REZEPTE = (
    "vinyl",
    "kassette",
    "clipping",
    "mp3_128",
    "hum",
    "wowflutter",
    "kompression",
    "kammfilter",
    "verzerrung",
    "clicks",
    "pitch_drift",
    "dropout",
    "raum",
    "lautheit",
    "amp_drift",
    "stereo_imbalance",
)


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


def _stereo_balance_ratio(audio: np.ndarray) -> float:
    """RMS(links)/RMS(rechts) in dB — Stereo-Layout-Invariante (§III
    (copilot-instructions.md)): channels-first (C, N).

    Mono ⇒ 0,0 dB (per Definition ausgeglichen). Gegen Stille-Kanäle
    deterministisch geschützt (kein inf/NaN im Beleg). Gain-invariant
    (beide Kanäle gleich skaliert ⇒ Balance unverändert) — das Pegel-Matching
    verfälscht die Balance-Richtung also nicht.
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim < 2:
        return 0.0
    links = float(np.sqrt((x[0] ** 2).mean()))
    rechts = float(np.sqrt((x[1] ** 2).mean()))
    if links <= 1e-12 and rechts <= 1e-12:
        return 0.0
    # Einseitige Stille bleibt endlich (Beleg-JSON verträgt kein inf/NaN).
    return float(20.0 * np.log10(max(links, 1e-12) / max(rechts, 1e-12)))


def bewerte_paar(
    clean: np.ndarray,
    schaden: np.ndarray,
    restauriert: np.ndarray,
    *,
    pegel_matching: bool = True,
    rezept: str = "",
) -> dict[str, object]:
    """Ein Paar bewerten — Richtung: defektlast runter, musik_mos rauf.

    Für `rezept == "stereo_imbalance"` zählt zusätzlich die Balance-Richtung
    (stereo_balance_ratio, JND 0,5 dB): der Schaden entfernt sich vom
    Quell-Wert, die Restauration nähert sich an (Paritäts-Register
    „Stereo-Imaging/Azimuth“).
    """
    s = np.asarray(schaden, dtype=np.float32)
    r = np.asarray(restauriert, dtype=np.float32)
    n = min(s.shape[-1], r.shape[-1])  # Zeitachse — gilt für Mono (N,) wie Stereo (C, N)
    s, r = s[..., :n], r[..., :n]
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
    # DRT-10 (Spec 26, 2026-11-15): ÜBERRESTAURATION — die Restauration gewinnt defektlast
    # deutlich, verliert aber Wohlklang deutlich (Jets-vinyl-Befund: −0,203 / −0,551).
    # Ein solcher Kanal ist nicht aktivierungsfähig — Ursachen-Analyse Pflicht (§V7 (copilot-instructions.md)).
    ueberrestauration = bool(d_defektlast <= -JND_DEFEKTLAST and d_musik_mos <= -JND_MOS)
    ergebnis: dict[str, object] = {
        "clean": k,
        "schaden": d,
        "restauriert": e,
        "delta_defektlast": round(float(d_defektlast), 5),
        "delta_musik_mos": round(float(d_musik_mos), 4),
        "besser": bool(besser),
        "ueberrestauration": ueberrestauration,
        "pegel_gain_ist": round(float(np.sqrt((r.astype(np.float64) ** 2).mean())) / max(1e-12, rms_schaden), 4),
    }
    if rezept == "stereo_imbalance":
        # Balance-Richtung als ZUSÄTZLICHES besser-Kriterium: der Schaden muss
        # sich vom Quell-Wert entfernt haben (abstand_schaden), die Restauration
        # muss sich um mehr als die JND nähern. defektlast/musik_mos bleiben
        # unverändert wirksam — die Balance kommt ODER-verknüpft dazu.
        bal_c = _stereo_balance_ratio(clean)
        bal_d = _stereo_balance_ratio(s)
        bal_r = _stereo_balance_ratio(r_matched)
        abstand_schaden = abs(bal_d - bal_c)
        abstand_restauriert = abs(bal_r - bal_c)
        balance_besser = bool(abstand_restauriert <= abstand_schaden - JND_BALANCE_DB)
        ergebnis["stereo_balance"] = {
            "clean_db": round(bal_c, 4),
            "schaden_db": round(bal_d, 4),
            "restauriert_db": round(bal_r, 4),
            "abstand_schaden_db": round(abstand_schaden, 4),
            "abstand_restauriert_db": round(abstand_restauriert, 4),
            "besser": balance_besser,
        }
        ergebnis["besser"] = bool(besser) or balance_besser
    return ergebnis


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
        if rz == "stereo_imbalance":
            aus[rz]["n_balance_besser"] = int(sum(1 for z in g if z.get("stereo_balance", {}).get("besser")))
    return aus


def main(argv: list[str] | None = None) -> int:
    # Crash-Diagnose (2026-11-19): nativer Crash landet im faulthandler-Log.
    import faulthandler as _fh_mod

    _fh_log = ROOT / "logs" / "python_faulthandler_ketten.log"
    _fh_log.parent.mkdir(parents=True, exist_ok=True)
    _fh_handle = open(_fh_log, "a", encoding="utf-8")
    _fh_mod.enable(file=_fh_handle, all_threads=True)

    ap = argparse.ArgumentParser(description="Schadens-Proben objektiv bewerten (CPU)")
    ap.add_argument(
        "--verzeichnis",
        default=str(Path.home() / "Musik" / "schadens_proben"),
        help="Ordner mit schadens_manifest.json",
    )
    ap.add_argument("--out", default="", help="JSON-Beleg-Ziel (Default: <verzeichnis>/bewertung.json)")
    ap.add_argument("--kein-pegel-matching", action="store_true", help="Rohpegel vergleichen (nicht empfohlen)")
    args = ap.parse_args(argv)

    from scripts.erstelle_schadens_proben import _decode_48k, _decode_48k_stereo, _mittel_ausschnitt

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
        rezept = str(eintrag.get("rezept", ""))
        if rezept == "stereo_imbalance":
            # Paritäts-Register „Stereo-Imaging/Azimuth“: die Quelle des
            # Stereo-Rezepts wird stereo dekodiert (2, N) — sonst gäbe es
            # keinen Quell-Wert für die Balance-Richtung.
            clean = _mittel_ausschnitt(
                _decode_48k_stereo(Path(eintrag["quelle"])), float(manifest.get("sekunden", 10.0))
            )
        else:
            clean = _mittel_ausschnitt(_decode_48k(Path(eintrag["quelle"])), float(manifest.get("sekunden", 10.0)))
        schaden = np.asarray(sf.read(eintrag["schaden"], always_2d=False)[0], dtype=np.float32)
        rest = np.asarray(sf.read(eintrag["restauriert"], always_2d=False)[0], dtype=np.float32)
        # Stereo-Layout-Invariante (§III (copilot-instructions.md)): intern (C, N).
        if schaden.ndim == 2:
            schaden = schaden.T  # (Frames, C) → (C, N)
        if rest.ndim == 2:
            rest = rest.T
        n = min(clean.shape[-1], schaden.shape[-1])
        ergebnis = bewerte_paar(
            clean[..., :n], schaden, rest, pegel_matching=not args.kein_pegel_matching, rezept=rezept
        )
        zeile = {"song": eintrag.get("stamm", ""), "rezept": rezept, **ergebnis}
        zeilen.append(zeile)
        zusatz = ""
        if "stereo_balance" in ergebnis:
            b = ergebnis["stereo_balance"]
            zusatz = f"  Balance {b['schaden_db']:+.2f}→{b['restauriert_db']:+.2f} dB (Quelle {b['clean_db']:+.2f})"
        print(
            f"{str(zeile['song'])[:34]:34s} {str(zeile['rezept']):9s} "
            f"Δdefektlast {ergebnis['delta_defektlast']:+.3f}  Δmusik_mos {ergebnis['delta_musik_mos']:+.2f}  "
            f"{'BESSER' if ergebnis['besser'] else 'schlechter/neutral'}{zusatz}"
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
        "n_ueberrestauration": int(sum(1 for z in zeilen if z.get("ueberrestauration"))),
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
