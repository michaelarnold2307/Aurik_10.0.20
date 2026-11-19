#!/usr/bin/env python3
"""B3-Mess-Harness: Never-worsen-Beleg PQS-Messvertrag alt vs. neu (2026-11-05).

Messvertrag-Korrektur (D-K2-2/D-K2-3, §V7 (copilot-instructions.md) — Metrik
korrigieren statt Pipeline verbiegen): ``PerceptualQualityScorer.align_signals``
dokumentierte „Signale werden vor Bewertung zeitlich ausgerichtet", implementiert
war nur Trunkierung (Timing-Konflation: ein Latenz-Offset wurde als
Qualitätsverlust gemessen und konnte den §1.4-Studio-Finisher — K2-Sprach-
Vocoder auf Musik — unnötig auslösen). Vertrag NEU richtet in einem begrenzten
Lag-Fenster (±20 ms) aus und wählt ``max(MOS(lag0), MOS(+k), MOS(-k))``:
Never-worsen ist damit **konstruktiv je Fenster garantiert** (Lag 0 ist immer
Kandidat; Gleichstand gewinnt Lag 0).

Muster: ``scripts/b3_transient_energie_neverworsen.py`` (unverändert, nur
Vorbild). Blind-Entscheidung entfällt bewusst: beide Verträge sind
deterministische Objektivmetriken auf identischen Paaren (kein Hör-Bias).
Zwei getrennte Läufe (Vertrag ALT = ``align_signals=False``, Vertrag NEU =
``align_signals=True``) auf demselben Fenster-/Probenplan (Seed 42),
``compare`` rechnet den gepaarten Never-worsen-Beschluss.

Aufruf (die AUSFÜHRUNG macht der Manager — kein Shell in der Worker-Runtime):
    nice -n 19 python3 -B scripts/b3_pqs_mos_neverworsen.py run --label alt
    nice -n 19 python3 -B scripts/b3_pqs_mos_neverworsen.py run --label neu
    nice -n 19 python3 -B scripts/b3_pqs_mos_neverworsen.py compare \\
        --alt docs/reports/current/b3_pqs_mos_alt.json \\
        --neu docs/reports/current/b3_pqs_mos_neu.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

JND_TOLERANZ = 0.05  # MOS-Punkte — Never-worsen-Toleranz (wie b3_transient_energie_neverworsen.py)
PQS_NORM_SCHWELLE = (4.3 - 1.0) / 4.0  # §1.4-Studio-Finisher: PQS-MOS < 4.3
VERTRAG_FUER_LABEL = {"alt": False, "neu": True}  # align_signals


def _vorzeichentest(vorteile: int, n: int) -> float:
    """Zweiseitiger exakter Vorzeichen-Test (Binomial, p = 0.5)."""
    if n <= 0:
        return 1.0
    k = min(vorteile, n - vorteile)
    eine_seite = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return float(min(1.0, 2.0 * eine_seite))


def _proben_paare(name: str, audio: np.ndarray, sr: int, seed: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Deterministische (Referenz, Degradation)-Proben je Fenster (§G5 (copilot-instructions.md)).

    Proben: identisch · rauschen (20 dB SNR) · bandlimit12k (48k→12k→48k) ·
    versatz512 (Timing-Konflation-Auslöser).
    """
    from scipy.signal import resample_poly

    rng = np.random.default_rng(seed)
    ref = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    g = int(np.gcd(48_000, 12_000))
    hin = resample_poly(ref, 12_000 // g, 48_000 // g)
    band = resample_poly(hin, 48_000 // g, 12_000 // g)
    if len(band) < len(ref):
        band = np.pad(band, (0, len(ref) - len(band)))
    band = band[: len(ref)].astype(np.float32)
    rausch_peg = float(np.sqrt(np.mean(ref**2)) + 1e-9)
    rauschen = ref + (rng.standard_normal(len(ref)) * (rausch_peg / 10.0)).astype(np.float32)
    versatz = np.concatenate([np.zeros(512, dtype=np.float32), ref])[: len(ref)]
    return [
        (f"{name}#identisch", ref, ref.copy()),
        (f"{name}#rauschen", ref, rauschen),
        (f"{name}#bandlimit12k", ref, band),
        (f"{name}#versatz512", ref, versatz),
    ]


def run_messen(label: str, fenster: list[tuple[str, np.ndarray, int]], seed: int) -> dict:
    """Misst alle Proben mit EINEM Vertrag (label 'alt' = Vertrag ALT, 'neu' = Vertrag NEU)."""
    from backend.core.perceptual_quality_scorer import PerceptualQualityScorer

    if label not in VERTRAG_FUER_LABEL:
        raise ValueError(f"label muss 'alt' oder 'neu' sein, erhalten: {label}")
    scorer = PerceptualQualityScorer(align_signals=VERTRAG_FUER_LABEL[label])
    zeilen: list[dict] = []
    for fenster_name, audio, sr in fenster:
        for trial_id, ref, deg in _proben_paare(fenster_name, audio, sr, seed):
            ergebnis = scorer.score(ref, deg, int(sr))
            mos = float(ergebnis.pqs_mos)
            pqs_norm = (mos - 1.0) / 4.0
            zeilen.append(
                {
                    "trial_id": trial_id,
                    "mos": round(mos, 6),
                    "pqs_norm": round(pqs_norm, 6),
                    "trigger": bool(pqs_norm < PQS_NORM_SCHWELLE),
                }
            )
    return {
        "schema": "aurik.pqs_mos.messung/1",
        "vertrag": f"align_signals={VERTRAG_FUER_LABEL[label]}",
        "label": label,
        "seed": seed,
        "n_trials": len(zeilen),
        "zeilen": zeilen,
        "quelle": "scripts/b3_pqs_mos_neverworsen.py",
    }


def vergleich_bewerten(alt_lauf: dict, neu_lauf: dict) -> dict:
    """Gepaarter Never-worsen-Beschluss Vertrag ALT vs. NEU (JND 0,05)."""
    alt_zeilen = {z["trial_id"]: z for z in alt_lauf["zeilen"]}
    neu_zeilen = {z["trial_id"]: z for z in neu_lauf["zeilen"]}
    if set(alt_zeilen) != set(neu_zeilen):
        raise ValueError("Trial-Mengen alt/neu unterscheiden sich — Läufe nicht gepaart")
    trial_ids = sorted(alt_zeilen)
    zeilen = []
    for tid in trial_ids:
        a, n = alt_zeilen[tid], neu_zeilen[tid]
        zeilen.append(
            {
                "trial_id": tid,
                "mos_alt": a["mos"],
                "mos_neu": n["mos"],
                "delta": round(n["mos"] - a["mos"], 6),
                "trigger_alt": a["trigger"],
                "trigger_neu": n["trigger"],
            }
        )
    deltas = [z["delta"] for z in zeilen]
    never_worsen = all(d >= -JND_TOLERANZ for d in deltas)
    vorteile = sum(1 for d in deltas if d > 0)
    p_wert = _vorzeichentest(vorteile, len(deltas))
    flips = [z for z in zeilen if z["trigger_alt"] != z["trigger_neu"]]
    return {
        "schema": "aurik.pqs_mos.beschluss/1",
        "seed": alt_lauf.get("seed"),
        "n_trials": len(zeilen),
        "zeilen": zeilen,
        "never_worsen": bool(never_worsen),
        "never_worsen_konstruktiv": "Lag 0 ist im Kandidatenset (max-MOS-Auswahl)",
        "vorteile": f"{vorteile}/{len(deltas)}",
        "p_wert": round(p_wert, 6),
        "trigger_flips": len(flips),
        "ergebnis": "ERFUELLT" if never_worsen else "ABGELEHNT",
        "quelle": "scripts/b3_pqs_mos_neverworsen.py",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="B3-Mess-Harness PQS-Messvertrag alt vs. neu (Never-worsen)")
    ap.add_argument("modus", choices=["run", "compare"])
    ap.add_argument("--label", default="neu", choices=["alt", "neu"])
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
        alt_lauf = json.loads((PROJECT_ROOT / args.alt).read_text(encoding="utf-8"))
        neu_lauf = json.loads((PROJECT_ROOT / args.neu).read_text(encoding="utf-8"))
        beschluss = vergleich_bewerten(alt_lauf, neu_lauf)
        ziel = out_dir / "b3_pqs_mos_beschluss.json"
        ziel.write_text(json.dumps(beschluss, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log.info(
            "Vergleich Vertrag alt/neu: %s (never_worsen=%s, p=%s, Vorteile=%s, Trigger-Flips=%d) -> %s",
            beschluss["ergebnis"],
            beschluss["never_worsen"],
            beschluss["p_wert"],
            beschluss["vorteile"],
            beschluss["trigger_flips"],
            ziel.name,
        )
        return 0

    from scripts.release_candidate_pipeline import _existing_corpus, _load_clip

    fenster: list[tuple[str, np.ndarray, int]] = []
    for eintrag in _existing_corpus()[: args.clips]:
        pfad = Path(eintrag["path"])
        # Fensterplan IDENTISCH zu scripts/b3_transient_energie_neverworsen.py (Starts alle 2,5 s).
        for w_start in (0.0, 2.5, 5.0, 7.5)[: args.windows]:
            try:
                audio, sr = _load_clip(pfad, start_s=w_start, seconds=args.seconds)
            except ValueError:
                continue
            fenster.append((pfad.stem, np.asarray(audio, dtype=np.float32), int(sr)))
    if not fenster:
        log.error("kein Referenzmaterial")
        return 2

    log.info("Lauf '%s' messen (%d Fenster, Vertrag %s)...", args.label, len(fenster), VERTRAG_FUER_LABEL[args.label])
    lauf = run_messen(args.label, fenster, args.seed)
    ziel = out_dir / f"b3_pqs_mos_{args.label}.json"
    ziel.write_text(json.dumps(lauf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("Lauf '%s' geschrieben: %s (Datum %s)", args.label, ziel.name, date.today().isoformat())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
