#!/usr/bin/env python3
"""B3-Mess-Harness: Never-worsen-Beleg transparenz alt vs. neu (2026-11-04).

Auftrag: Scorecard-Vergleich vor/nach dem Messkontrakt-Fix auf den
Corpus-Fenstern (B3-Muster) inkl. transparenz-Kluft 0.5226 → Zielkorridor 0.55
(backend/core/calibration_matrix.py:1132) — ohne Verschlechterung anderer
Goals. Muster: scripts/b3_transient_energie_neverworsen.py (dort wird NICHTS
geändert) — Fensterplan, Produktionspfad, Bewertung und Baseline werden von
dort IMPORTIERT, damit der Beleg exakt dieselben Fenster und denselben
Restore-Pfad misst (b3_transient_energie_neverworsen.py: _restore_fenster,
_scorecard_messen, _auf_messrate_48k, BASELINE_2026_11_03).

Der Fix ist METRIK-seitig (TransparenzMetric-Messkontrakt) — die Pipeline-
Ausgabe bleibt bit-identisch (§G5 (copilot-instructions.md)). Der Harness
misst deshalb transparenz DOPPELT auf demselben Audio:

1. ``run``: je Fenster restaurieren, dann messen:
   * musik_mos + defektlast (richtungszertifiziert),
   * die 15-Goal-Scorecard (Vertrag NEU),
   * transparenz DOPPELT: Vertrag NEU (Default) vs. Vertrag ALT
     (``short_form_prior=True`` = wall-clock Blend vor dem Fix),
     blind verlost, Schlüssel getrennt (B3-Muster).
2. ``compare``: blind Paarvergleich + exakter Vorzeichen-Test + Never-worsen
   (JND 0,05, musik_mos/defektlast) gegen einen optionalen Alt-Lauf, dazu der
   Scorecard-Diff je Goal (Alt-Lauf und BASELINE_2026_11_03) und der
   Kontrakt-Beweis: ALT = 0.05·NEU + 0.475 auf 2,5-s-Fenstern (Blend-
   Fingerabdruck), NEU ohne Dauer-Deckel.

Aufruf (AUSFÜHRUNG durch den Manager — Worker-Runtime ohne Shell):
    nice -n 19 python3 -B scripts/b3_transparenz_neverworsen.py run --label nachher
    nice -n 19 python3 -B scripts/b3_transparenz_neverworsen.py compare \
        --alt docs/reports/current/b3_transient_energie_neu.json \
        --neu docs/reports/current/b3_transparenz_nachher.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

# Gemeinsame Konstanten/Helfer aus dem Muster-Harness (unverändert dort).
from scripts.b3_transient_energie_neverworsen import (
    BASELINE_2026_11_03,
    JND_TOLERANZ,
    _auf_messrate_48k,
    _bewerte,
    _restore_fenster,
    _scorecard_messen,
    _sign_test,
    _zu_mono,
)

# Zielkorridor transparenz: _CHAIN_END_GOAL_CEILINGS["shellac"]
# (backend/core/calibration_matrix.py:1132) — gilt als Material-Ceiling auch
# als Scorecard-Schwelle thr=0.5500 (GOAL_SCORECARD-Baseline 2026-11-03).
KORRIDOR_TRANSPARENZ = 0.55


def _transparenz_doppelmessung(audio_out: np.ndarray, sr: int) -> tuple[float, float]:
    """(neu, alt) — transparenz unter beiden Messverträgen auf selbem Audio."""
    from backend.core.musical_goals.musical_goals_metrics import TransparenzMetric

    met = TransparenzMetric()
    neu = float(met.measure(audio_out, sr))
    alt = float(met.measure(audio_out, sr, short_form_prior=True))
    return neu, alt


def run_messen(label: str, fenster: list[tuple[str, np.ndarray, int]], seed: int) -> dict:
    """Ein Lauf über den Fensterplan — inkl. blindem Kontrakt-A/B je Fenster."""
    rng = random.Random(f"{seed}:kontrakt:{label}")
    schluessel: dict[str, str] = {}
    zeilen: list[dict] = []

    for i, (clip, audio, sr) in enumerate(fenster):
        tid = f"{clip}__w{i}"
        eingabe = _zu_mono(audio)
        ausgabe, snap, material = _restore_fenster(audio, sr)
        del snap  # B1-Ledger interessiert hier nicht (transient-Harness führt ihn)

        mos, last = _bewerte(ausgabe, sr)
        scorecard = _scorecard_messen(eingabe, ausgabe, sr, material)
        neu, alt = _transparenz_doppelmessung(ausgabe, sr)

        # Blindheit (B3-Muster): Seiten verlosen, Bewertung sieht nur
        # "links"/"rechts" — erst nach der Lose-Ziehung wird entblindet.
        links_ist_neu = rng.random() < 0.5
        schluessel[tid] = "neu" if links_ist_neu else "alt"
        blindwerte = {
            "links": neu if links_ist_neu else alt,
            "rechts": alt if links_ist_neu else neu,
        }
        zeilen.append(
            {
                "trial_id": tid,
                "material": material,
                "musik_mos": mos,
                "defektlast": last,
                "scorecard": scorecard,
                "kontrakt_blind": dict(blindwerte),
            }
        )

    # Entblindung
    for z in zeilen:
        neu_seite = "links" if schluessel[z["trial_id"]] == "neu" else "rechts"
        alt_seite = "rechts" if neu_seite == "links" else "links"
        z["transparenz_neu"] = z["kontrakt_blind"][neu_seite]
        z["transparenz_alt"] = z["kontrakt_blind"][alt_seite]
        del z["kontrakt_blind"]

    return {
        "schema": "aurik.transparenz.messung/1",
        "label": label,
        "seed": seed,
        "blind_schluessel": schluessel,
        "zeilen": zeilen,
        "quelle": "scripts/b3_transparenz_neverworsen.py",
    }


def compare_bewerten(alt_lauf: dict, neu_lauf: dict, seed: int) -> dict:
    """Blind Paarvergleich + Scorecard-Diff + Kontrakt-Beweis (B3-Muster)."""
    from scripts.ab_harness import sign_test_two_sided

    rng = random.Random(f"{seed}:laufvergleich")
    alt_zeilen = {z["trial_id"]: z for z in alt_lauf.get("zeilen", [])}
    neu_zeilen = {z["trial_id"]: z for z in neu_lauf.get("zeilen", [])}
    gemeinsam = sorted(set(alt_zeilen) & set(neu_zeilen))
    if not gemeinsam:
        raise ValueError("keine gemeinsamen trial_ids zwischen alt- und neu-Lauf")

    paare: list[dict] = []
    for tid in gemeinsam:
        links_ist_neu = rng.random() < 0.5
        links = neu_zeilen[tid] if links_ist_neu else alt_zeilen[tid]
        rechts = alt_zeilen[tid] if links_ist_neu else neu_zeilen[tid]
        paare.append({"trial_id": tid, "links": links, "rechts": rechts, "links_ist_neu": links_ist_neu})

    zeilen = []
    for p in paare:
        neu = p["links"] if p["links_ist_neu"] else p["rechts"]
        alt = p["rechts"] if p["links_ist_neu"] else p["links"]
        eintrag = {
            "trial_id": p["trial_id"],
            "mos_alt": alt["musik_mos"],
            "mos_neu": neu["musik_mos"],
            "last_alt": alt["defektlast"],
            "last_neu": neu["defektlast"],
            "transparenz_alt_scorecard": alt["scorecard"].get("transparenz"),
            "transparenz_neu_scorecard": neu["scorecard"].get("transparenz"),
        }
        # Kontrakt-Beweis nur aus dem NEU-Lauf (dort ist doppelt gemessen).
        if "transparenz_neu" in neu:
            eintrag["kontrakt_neu"] = neu["transparenz_neu"]
            eintrag["kontrakt_alt"] = neu["transparenz_alt"]
        zeilen.append(eintrag)

    mos_deltas = [z["mos_neu"] - z["mos_alt"] for z in zeilen]
    last_deltas = [z["last_neu"] - z["last_alt"] for z in zeilen]
    never_worsen = all(d >= -JND_TOLERANZ for d in mos_deltas) and all(d <= JND_TOLERANZ for d in last_deltas)
    vorteile = sum(1 for d in mos_deltas if d > 0)
    p_wert = _sign_test(sign_test_two_sided, vorteile, len(mos_deltas))
    signifikant = bool(p_wert < 0.05 and sum(mos_deltas) > 0)

    # Scorecard-Diff je Goal (neu − alt) und gegen die Vergleichsbasis 2026-11-03.
    ziele = sorted({k for z in neu_lauf["zeilen"] for k in z["scorecard"]})
    scorecard_diff = {}
    for g in ziele:
        werte_alt = [z["scorecard"].get(g) for z in alt_lauf["zeilen"] if z["scorecard"].get(g) is not None]
        werte_neu = [z["scorecard"].get(g) for z in neu_lauf["zeilen"] if z["scorecard"].get(g) is not None]
        m_alt = float(np.mean(werte_alt)) if werte_alt else None
        m_neu = float(np.mean(werte_neu)) if werte_neu else None
        scorecard_diff[g] = {
            "mittel_alt": round(m_alt, 4) if m_alt is not None else None,
            "mittel_neu": round(m_neu, 4) if m_neu is not None else None,
            "delta": round(m_neu - m_alt, 4) if (m_alt is not None and m_neu is not None) else None,
            "baseline_2026_11_03": BASELINE_2026_11_03.get(g),
        }

    # Kontrakt-A/B auf demselben Audio: ALT muss dem Blend-Fingerabdruck
    # 0.05·NEU + 0.475 auf 2,5-s-Fenstern folgen; NEU darf den Korridor 0.55
    # erreichen (vor dem Fix mathematisch gedeckelt auf 0.525).
    kontrakt = [
        {
            "trial_id": z["trial_id"],
            "neu": round(z["kontrakt_neu"], 4),
            "alt": round(z["kontrakt_alt"], 4),
            "blend_erwartet": round(0.05 * z["kontrakt_neu"] + 0.475, 4),
        }
        for z in zeilen
        if "kontrakt_neu" in z
    ]
    blend_treue = all(abs(k["alt"] - k["blend_erwartet"]) <= 1e-3 for k in kontrakt)
    korridor_erreichbar = any(k["neu"] >= KORRIDOR_TRANSPARENZ for k in kontrakt)

    trans_neu = scorecard_diff.get("transparenz", {}).get("mittel_neu")
    trans_alt = scorecard_diff.get("transparenz", {}).get("mittel_alt")
    transparenz_verbessert = trans_neu is not None and trans_alt is not None and (trans_neu - trans_alt) >= 0.02

    erfuellt = bool(never_worsen and korridor_erreichbar and transparenz_verbessert)
    return {
        "schema": "aurik.transparenz.beschluss/1",
        "seed": seed,
        "n_fenster": len(zeilen),
        "zeilen": zeilen,
        "never_worsen": bool(never_worsen),
        "vorteile": f"{vorteile}/{len(mos_deltas)}",
        "p_wert": round(p_wert, 6),
        "signifikant": signifikant,
        "scorecard_diff": scorecard_diff,
        "baseline_2026_11_03": BASELINE_2026_11_03,
        "kontrakt_ab": kontrakt,
        "kontrakt_blend_treue": bool(blend_treue),
        "korridor_0_55_erreichbar": bool(korridor_erreichbar),
        "transparenz_verbessert": bool(transparenz_verbessert),
        "ergebnis": "ERFUELLT" if erfuellt else "ABGELEHNT",
        "quelle": "scripts/b3_transparenz_neverworsen.py",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="B3-Mess-Harness transparenz alt vs. neu (Never-worsen)")
    ap.add_argument("modus", choices=["run", "compare"])
    ap.add_argument("--label", default="nachher")
    ap.add_argument("--alt", default="")
    ap.add_argument("--neu", default="")
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out_dir = PROJECT_ROOT / "docs" / "reports" / "current"

    if args.modus == "compare":
        if not args.alt or not args.neu:
            log.error("compare braucht --alt und --neu (Run-JSONs)")
            return 2
        alt_lauf = json.loads(Path(args.alt).read_text(encoding="utf-8"))
        neu_lauf = json.loads(Path(args.neu).read_text(encoding="utf-8"))
        beschluss = compare_bewerten(alt_lauf, neu_lauf, args.seed)
        ziel = out_dir / "b3_transparenz_beschluss.json"
        ziel.write_text(json.dumps(beschluss, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log.info("Beschluss geschrieben: %s (Ergebnis=%s)", ziel.name, beschluss["ergebnis"])
        return 0

    # run — Fensterplan IDENTISCH zu scripts/b3_transient_energie_neverworsen.py
    # (Starts alle 2,5 s, --seconds 2.5) — Copy der 4-Plan-Zeilen, weil deren
    # Plan in main() des Muster-Harness gekapselt ist und dessen Datei
    # unverändert bleibt (Auftragsvorgabe).
    from scripts.release_candidate_pipeline import _existing_corpus, _load_clip

    fenster: list[tuple[str, np.ndarray, int]] = []
    for eintrag in _existing_corpus()[: args.clips]:
        pfad = Path(eintrag["path"])
        for w_start in (0.0, 2.5, 5.0, 7.5)[: args.windows]:
            try:
                audio, sr = _load_clip(pfad, start_s=w_start, seconds=args.seconds)
            except ValueError:
                continue
            audio, sr = _auf_messrate_48k(np.asarray(audio, dtype=np.float32), int(sr))
            fenster.append((pfad.stem, audio, sr))
    if not fenster:
        log.error("kein Referenzmaterial")
        return 2

    log.info("Lauf '%s' messen (%d Fenster)...", args.label, len(fenster))
    lauf = run_messen(args.label, fenster, args.seed)
    ziel = out_dir / f"b3_transparenz_{args.label}.json"
    ziel.write_text(json.dumps(lauf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("Lauf '%s' geschrieben: %s", args.label, ziel.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
