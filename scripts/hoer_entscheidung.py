#!/usr/bin/env python3
"""hoer_entscheidung — komfortable B3/C4-Hör-Entscheidung ohne Wartezeit (2026-11-09).

Spielt die von `erstelle_hoerproben.py` erzeugten 10-s-Proben VERBLINDET ab
(ffplay/aplay/paplay, links → rechts), sammelt die Urteile und schreibt einen
`aurik.ab_stats/1`-Beleg, den `scripts/flag_freigabe.py` direkt als
Hör-Sign-off konsumiert (signifikant + siege_a ≠ siege_b).

Subkommandos:
  --status                     zeigt offene B3/C4-Entscheidungen + vorhandene Proben
  --hoeren <set>               verblindetes A/B über ein Proben-Set, Urteil → Beleg
                               (z. B. c4_vocos, m5_rekonstruktion); bereits
                               beurteilte Proben werden übersprungen, neue Urteile
                               akkumulieren additiv im Beleg
  --wiederholen                mit --hoeren: auch bereits beurteilte Proben erneut
                               anhören (neues Urteil ersetzt das alte je Probe)

Die B3-W-Stufen-Entscheidungen (0/13) werden erst NACH den objektiven
Messungen im trainingsfreien GPU-Fenster anhörbar — das Panel zeigt deren
Status transparent an; es erfindet keine Proben (§G8 (copilot-instructions.md)).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROBEN = ROOT / "study_stimuli" / "hoerproben"
VERDIKTE = ROOT / "docs" / "reports" / "current"

OFFENE_ENTSCHEIDUNGEN = [
    {
        "id": "c4_vocos",
        "name": "C4 Studio-Finisher (Teil): vocos-Base vs. Musik-Finetune",
        "bezug": "D-K2-3",
        "verfuegbar": "jetzt (CPU-ONNX-Proben)",
        "folge": "flag_freigabe.py --verdikt docs/reports/current/hoer_verdikt_c4_vocos.json --signoff …",
    },
    {
        "id": "m5_rekonstruktion",
        "name": "M5 Lücken-Synthese: linear vs. vocos-Finetune",
        "bezug": "D-K2-3/M5",
        "verfuegbar": "jetzt (CPU-ONNX-Proben)",
        "folge": "Hörurteil → gated Rekonstruktions-Konsum vorbereiten (kein Flag-Flip ohne Sign-off)",
    },
    {
        "id": "b3_stufen",
        "name": "B3 Wohlklang-Stufen W-2/W-3/W-8/W-9/W-13 (0/13 aktiv)",
        "bezug": "Hebel B/A — b3_stufenmessung.py",
        "verfuegbar": "nach objektiver Messung im trainingsfreien GPU-Fenster",
        "folge": "flag_freigabe.py --objektivbeschluss … nach Hör-Sign-off",
    },
    {
        "id": "c4_bigvgan",
        "name": "C4 Studio-Finisher (Gesamt): vocos-Finetune vs. BigVGAN-F3",
        "bezug": "D-K2-3/D3-2",
        "verfuegbar": "nach BigVGAN-Training (Kette 2, läuft)",
        "folge": "Hörstichprobe C4 nach D3-2-Abschluss",
    },
]


def _player() -> str | None:
    import shutil

    for kandidat in ("ffplay", "aplay", "paplay"):
        if shutil.which(kandidat):
            return kandidat
    return None


def _spiele(pfad: Path, player: str) -> bool:
    try:
        if player == "paplay":
            subprocess.run(["paplay", str(pfad)], check=True, capture_output=True)
        elif player == "aplay":
            subprocess.run(["aplay", "-q", str(pfad)], check=True, capture_output=True)
        else:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(pfad)],
                check=True,
                capture_output=True,
            )
        return True
    except Exception:
        print(f"  (Abspielen fehlgeschlagen: {pfad})")
        return False


def status() -> int:
    manifest_pfad = PROBEN / "manifest.json"
    print("── Offene B3/C4-Entscheidungen ──────────────────")
    for e in OFFENE_ENTSCHEIDUNGEN:
        print(f"  [{e['id']}] {e['name']} ({e['bezug']})")
        print(f"       verfügbar: {e['verfuegbar']}")
    if manifest_pfad.is_file():
        m = json.loads(manifest_pfad.read_text(encoding="utf-8"))
        print("\n── Vorhandene Proben ───────────────────────────")
        for set_name, s in m.get("sets", {}).items():
            print(f"  {set_name}: {len(s['proben'])} Paare ({m.get('sekunden')} s)")
    else:
        print("\nKeine Proben erzeugt — starte:")
        print("  .venv_aurik/bin/python -B scripts/erstelle_hoerproben.py")
    return 0


def _sign_test(wins: int, n: int) -> float:
    """Zweiseitiger Vorzeichentest gegen H0 = 1/2 (exakt, §G5 (copilot-instructions.md))."""
    if n <= 0:
        return 1.0
    wins = max(0, min(wins, n))
    tail = sum(math.comb(n, k) for k in range(0, min(wins, n - wins) + 1))
    return min(1.0, 2.0 * tail / float(2**n))


def _trial_id(stamm: str, idx: int) -> str:
    """Stabiler Trial-Schlüssel aus Stamm-Slug + Position im Set.

    Die Position ist stabil, weil der Erzeuger Proben nur anhängt und nie
    umsortiert — Urteile überleben so spätere Set-Erweiterungen.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", stamm.lower()).strip("_")[:48] or "stueck"
    return f"{slug}__{idx}"


def _aggregiere(trials: dict[str, dict]) -> dict:
    """ab_stats-Kennzahlen über alle gespeicherten Trials (Sitzungs-übergreifend)."""
    wins_a = sum(1 for v in trials.values() if v.get("gewinner") == "A")
    wins_b = sum(1 for v in trials.values() if v.get("gewinner") == "B")
    ties = sum(1 for v in trials.values() if v.get("gewinner") is None)
    n_dec = wins_a + wins_b
    p = _sign_test(wins_a, n_dec)
    return {
        "n_verdikte": len(trials),
        "n_entscheidend": n_dec,
        "gleichstaende": ties,
        "siege_a": wins_a,
        "siege_b": wins_b,
        "siegquote_a": round(wins_a / n_dec, 4) if n_dec else None,
        "p_wert_vorzeichen_test": round(p, 6),
        "signifikant": bool(n_dec > 0 and p < 0.05),
    }


def hoeren(set_name: str, seed: int, wiederholen: bool = False) -> int:
    manifest_pfad = PROBEN / "manifest.json"
    if not manifest_pfad.is_file():
        print("Keine Proben vorhanden — erst erstelle_hoerproben.py ausführen.")
        return 2
    m = json.loads(manifest_pfad.read_text(encoding="utf-8"))
    s = m.get("sets", {}).get(set_name)
    if not s:
        print(f"Set '{set_name}' nicht im Manifest. Verfügbar: {list(m.get('sets', {}))}")
        return 2
    verdikt_pfad = VERDIKTE / f"hoer_verdikt_{set_name}.json"
    gespeichert: dict[str, dict] = {}
    if verdikt_pfad.is_file():
        try:
            alt = json.loads(verdikt_pfad.read_text(encoding="utf-8"))
            gespeichert = dict(alt.get("trials") or {})
        except Exception:
            pass  # defekter Alt-Beleg: neue Urteile schreiben ihn neu
    player = _player()
    if not player:
        print("Kein CLI-Player (ffplay/aplay/paplay) — bitte die Dateien im eigenen Player öffnen:")
        for p in s["proben"]:
            print(f"  A: {p['a']}")
            print(f"  B: {p['b']}")
        return 1
    rng = random.Random(seed)
    neue: dict[str, dict] = {}
    abgebrochen = False
    print(f"\nSet: {set_name} — {s['entscheidung']}")
    if gespeichert:
        print(
            f"Bereits beurteilt: {len(gespeichert)} Trial(s) — neue Urteile werden addiert"
            f"{' (Wiederholung: alte Urteile werden ersetzt)' if wiederholen else ''}."
        )
    print(f"Player: {player} — automatisch links → rechts; Antwort l/r/=, Enter = nochmal, q = Abbruch.\n")
    for idx, probe in enumerate(s["proben"], start=1):
        tid = _trial_id(probe["stamm"], idx)
        if tid in gespeichert and not wiederholen:
            vor = gespeichert[tid]
            print(
                f"── Probe {idx}/{len(s['proben'])}: {probe['stamm'][:50]} — bereits beurteilt"
                f" ({vor.get('gewinner') or '='}) — übersprungen."
            )
            continue
        left_is_a = rng.random() < 0.5
        pa = Path(probe["a"])
        pb = Path(probe["b"])
        left, right = (pa, pb) if left_is_a else (pb, pa)
        while True:
            print(f"── Probe {idx}/{len(s['proben'])}: {probe['stamm'][:50]}")
            print("  ▶ links …")
            _spiele(left, player)
            print("  ▶ rechts …")
            _spiele(right, player)
            antwort = input("  l / r / = ? ").strip().lower()
            if antwort == "q":
                print("Abbruch — bisherige Urteile werden verwendet.")
                abgebrochen = True
                break
            if antwort == "":
                print("  (nochmal hören …)")
                continue
            if antwort in ("l", "r"):
                gewinner = "A" if (left is pa) == (antwort == "l") else "B"
                neue[tid] = {"stamm": probe["stamm"], "preferred": antwort, "gewinner": gewinner}
                break
            if antwort == "=":
                neue[tid] = {"stamm": probe["stamm"], "preferred": "tie", "gewinner": None}
                break
            print("  Bitte l, r oder = eingeben.")
        if abgebrochen:
            break
    if not neue:
        if gespeichert:
            print("Keine neuen Urteile — Beleg unverändert.")
            return 0
        print("Keine Urteile — kein Beleg geschrieben.")
        return 1

    gespeichert.update(neue)
    stats = _aggregiere(gespeichert)
    stats.update(
        {
            "schema": "aurik.ab_stats/1",
            "label_a": "Bestand/Referenz (A)",
            "label_b": "Challenger (B)",
            # Richtung explizit (flag_freigabe: ab_stats/1 ohne Seite ist nicht buchbar).
            "kandidat_seite": "B",
            "entscheidung": s["entscheidung"],
            "quelle": "scripts/hoer_entscheidung.py",
            "n_trials_gesamt": len(gespeichert),
            "trials": gespeichert,
        }
    )
    verdikt_pfad.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nBeleg geschrieben: {verdikt_pfad} ({len(gespeichert)} Trial(s) gesamt)")
    print(
        f"  Siege A/B: {stats['siege_a']}/{stats['siege_b']} (n={stats['n_entscheidend']}), "
        f"p={stats['p_wert_vorzeichen_test']:.4f}, signifikant={stats['signifikant']}"
    )
    if stats["signifikant"] and stats["siege_a"] != stats["siege_b"]:
        if stats["siege_a"] > stats["siege_b"]:
            print("  → Kandidat A (Referenz/Bestand) signifikant bevorzugt — KEIN Flag-Flip nötig.")
            print(f"    Der Bestandspfad ist der bestätigte SOTA-Zustand (Beleg: {verdikt_pfad.name}).")
        else:
            print(f"  → Kandidat B (Challenger) signifikant bevorzugt — flag_freigabe: --verdikt {verdikt_pfad}")
    else:
        print("  → noch nicht signifikant — nächster Lauf spielt nur die noch unbeurteilten Proben.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="B3/C4-Hör-Entscheidung ohne Wartezeit")
    ap.add_argument("--status", action="store_true", help="Status der Entscheidungen + Proben")
    ap.add_argument("--hoeren", default=None, help="Proben-Set verblindet anhören (c4_vocos, m5_rekonstruktion)")
    ap.add_argument("--wiederholen", action="store_true", help="mit --hoeren: bereits beurteilte Proben erneut anhören")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if args.hoeren:
        return hoeren(args.hoeren, args.seed, args.wiederholen)
    return status()


if __name__ == "__main__":
    sys.exit(main())
