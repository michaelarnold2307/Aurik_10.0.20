#!/usr/bin/env python3
"""Muster B: Deterministische Kompositions-Suche ueber W-Stufen-Flags.

Befund (B3-Protokolle 2026-11): Einzel-Flag-Flips waren nie signifikant
(W-1 widerspruechlich, W-2 p=0,69, W-3 Δ≈0) — Kombinationen wurden nie
systematisch gemessen. Dieses Skript sucht deterministisch (Forward-Selection)
die beste Flag-Komposition gegen die Baseline (alle AUS) mit den
richtungszertifizierten Zeugen musik_mos/defektlast (Spearman 0,95) und
never-worsen je Schritt.

KEINE Entscheidung, KEIN dauerhafter Flag-Flip: Flags werden nur
prozess-lokal gesetzt und in ``try/finally`` auf die Ausgangswerte
zurueckgesetzt; das Ergebnis ist ein KANDIDAT fuer EINE Hoer-Session
(Hoer-Instanz entscheidet, dann §4-Verfahren — flag_freigabe.py).

Aufruf (trainingsfreies Fenster, host-sicher):
    nice -n 15 python3 -B scripts/b3_komposition_suche.py \
        [--clips 3] [--windows 2] [--seed 42] [--stufen W-2,W-3,W-8,W-9,W-13]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

JND_TOLERANZ = 0.05  # wie b3_stufenmessung (Audibility-Stufe)

# Audio-wirksame Kandidaten-Flags (W-10/W-11/W-12 sind Zeugen/Release-Flags
# ohne Audio-Wirkung — bewusst NICHT Teil der Suche, §V7 (copilot-instructions.md)).
KANDIDATEN: dict[str, str] = {
    "W-2": "use_scnet_music",
    "W-3": "use_sgmse_musik",
    "W-8": "use_bw_v5",
    "W-9": "use_whisper_denoiser",
    "W-13": "use_bigvgan_f3_studio",
}


def _flag_lesen(stufe: str) -> bool:
    import backend.core.music_model_flags as mmf

    return bool(getattr(mmf, KANDIDATEN[stufe]))


def _flag_setzen(stufe: str, wert: bool) -> None:
    import backend.core.music_model_flags as mmf

    setattr(mmf, KANDIDATEN[stufe], wert)


# ── Reine Suchlogik (modell-frei testbar) ────────────────────────────────────
def zielwert(messung: dict) -> float:
    """Primaere Zielgroesse = musik_mos (G-2); defektlast ist Never-worsen-Guard."""
    return float(messung.get("musik_mos", 0.0))


def never_worsen_ok(messung: dict, baseline: dict) -> bool:
    mos_ok = float(messung.get("musik_mos", 0.0)) >= float(baseline.get("musik_mos", 0.0)) - JND_TOLERANZ
    last_ok = float(messung.get("defektlast", 1e9)) <= float(baseline.get("defektlast", 0.0)) + JND_TOLERANZ
    return bool(mos_ok and last_ok)


def forward_auswahl(scores: dict, kandidaten: list[str], reihenfolge: list[str]) -> tuple[list[str], list[str]]:
    """Greedy-Forward-Selection mit Never-worsen-Schranke (deterministisch).

    ``scores``: {frozenset(komp): messung} — Baseline liegt als frozenset() vor.
    Reihenfolge der Kandidaten-Auswertung ist seed-fest (kein Raten).
    Rückgabe: (beste_komposition, besuchte_kompositionen als Str-Listen).
    """
    baseline = scores.get(frozenset(), {})
    beste: set[str] = set()
    besucht: list[str] = []
    for kandidat in reihenfolge:
        if kandidat not in kandidaten:
            continue
        kandidat_set = frozenset(beste | {kandidat})
        messung = scores.get(kandidat_set)
        if messung is None:
            continue
        besucht.append(",".join(sorted(kandidat_set)) or "baseline")
        if never_worsen_ok(messung, baseline) and zielwert(messung) > zielwert(scores.get(frozenset(beste), baseline)):
            beste.add(kandidat)
    return sorted(beste), besucht


def suche(measure_fn, kandidaten: list[str], reihenfolge: list[str]) -> dict:
    """Lazy Forward-Selection: misst jede Kandidat-Erweiterung nur bei Bedarf.

    ``measure_fn(frozenset(stufen)) -> messung-dict`` — Baseline wird vom
    Aufrufer zuerst als frozenset() gemessen; hier wird sie mitgemessen,
    falls nicht übergeben (Fail-Closed: ohne Baseline keine Never-worsen-Schranke).
    Rückgabe: {"beste": [...], "protokoll": [...], "baseline": {...}}.
    """
    baseline = measure_fn(frozenset())
    beste: set[str] = set()
    beste_messung = dict(baseline)
    protokoll: list[dict] = []
    for kandidat in reihenfolge:
        if kandidat not in kandidaten:
            continue
        kandidat_set = frozenset(beste | {kandidat})
        messung = measure_fn(kandidat_set)
        protokoll.append({"komposition": sorted(kandidat_set), "messung": messung})
        if never_worsen_ok(messung, baseline) and zielwert(messung) > zielwert(beste_messung):
            beste.add(kandidat)
            beste_messung = dict(messung)
    return {"beste": sorted(beste), "protokoll": protokoll, "baseline": baseline}


def _echtes_measure_fn():
    """Mess-Funktion über den Produktionspfad (FAST-Profil, wie b3_stufenmessung)."""
    from scripts.b3_stufenmessung import _bewerte, _restore_fenster

    def mess(komposition: frozenset, audio: np.ndarray, sr: int) -> dict:
        aktive = {s for s in KANDIDATEN if _flag_lesen(s)}
        for s in KANDIDATEN:
            _flag_setzen(s, s in komposition)
        try:
            an_audio, _snap, _meta = _restore_fenster(audio, sr)
            mos, last = _bewerte(an_audio, sr)
        finally:
            for s in KANDIDATEN:
                _flag_setzen(s, s in aktive)
        return {"musik_mos": float(mos), "defektlast": float(last)}

    return mess


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Muster B: Kompositions-Suche über W-Stufen-Flags")
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stufen", default="W-2,W-3,W-8,W-9,W-13")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    kandidaten = [s.strip() for s in args.stufen.split(",") if s.strip() in KANDIDATEN]
    if not kandidaten:
        log.error("keine gültigen Kandidaten (KANDIDATEN: %s)", ",".join(KANDIDATEN))
        return 2

    from scripts.release_candidate_pipeline import _existing_corpus, _load_clip

    fenster: list[tuple[str, np.ndarray, int]] = []
    for eintrag in _existing_corpus()[: args.clips]:
        pfad = Path(eintrag["path"])
        for w_start in (0.0, 2.5, 5.0, 7.5):
            try:
                audio, sr = _load_clip(pfad, start_s=w_start, seconds=args.seconds)
            except ValueError:
                continue
            fenster.append((pfad.stem, np.asarray(audio, dtype=np.float32), int(sr)))
    if not fenster:
        log.error("kein Referenzmaterial")
        return 2

    rng = np.random.default_rng(args.seed)
    reihenfolge = [kandidaten[i] for i in rng.permutation(len(kandidaten))]
    log.info("Suche über %d Kandidaten, Reihenfolge (Seed %d): %s", len(kandidaten), args.seed, ",".join(reihenfolge))

    aktive_urspruenglich = {s: _flag_lesen(s) for s in KANDIDATEN}
    echtes_mess = _echtes_measure_fn()

    def mess(komposition: frozenset) -> dict:
        mos_werte, last_werte = [], []
        for _clip, audio, sr in fenster:
            m = echtes_mess(komposition, audio, sr)
            mos_werte.append(m["musik_mos"])
            last_werte.append(m["defektlast"])
        return {
            "musik_mos": float(np.mean(mos_werte)),
            "defektlast": float(np.mean(last_werte)),
            "fenster": len(fenster),
        }

    try:
        ergebnis = suche(mess, kandidaten, reihenfolge)
    finally:
        for s, wert in aktive_urspruenglich.items():
            _flag_setzen(s, wert)

    ergebnis["seed"] = args.seed
    ergebnis["kandidaten"] = kandidaten
    ergebnis["hinweis"] = (
        "KANDIDAT für EINE Hör-Session — keine Entscheidung, kein dauerhafter "
        "Flag-Flip; Freigabe nur über §4-Verfahren (flag_freigabe.py)."
    )
    out = PROJECT_ROOT / "docs" / "reports" / "current" / "b3_komposition_suche.json"
    out.write_text(json.dumps(ergebnis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("Beste Komposition: %s -> %s", ",".join(ergebnis["beste"]) or "(leer)", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
