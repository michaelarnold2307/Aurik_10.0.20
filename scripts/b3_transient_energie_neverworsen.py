#!/usr/bin/env python3
"""B3-Mess-Harness: Never-worsen-Beleg transient_energie alt vs. neu (2026-11-04).

Auftrag: Blind-Vergleich alt vs. neu auf denselben Corpus-Fenstern nach Muster
``scripts/b3_stufenmessung.py`` (Seed 42, Fensterplan, richtungszertifizierte
Metriken musik_mos + defektlast, Sign-Test, Never-worsen/JND 0,05) PLUS die
15-Goal-Scorecard beider Läufe. Vergleichsbasis der Scorecard sind die Werte
des Laufs 2026-11-03 (GOAL_SCORECARD; Log ``2026-11-03_b3_wiederholung_W6_W7.log``,
Zitate siehe ``BASELINE_2026_11_03`` — die Log-Datei selbst liegt nicht im
Worktree, die bekannten Werte sind als Konstante mit Quelle hinterlegt).

Der Fix 2026-11-04 ist METRIK-seitig (Messkontrakt transient_energie) — die
Pipeline-Ausgabe bleibt dadurch bit-identisch (§G5 (copilot-instructions.md)). Der Harness unterstützt
trotzdem beide Vergleichsebenen:

1. ``run``: stellt je Fenster einmal her und misst
   * musik_mos + defektlast (richtungszertifiziert),
   * die 15-Goal-Scorecard (Vertrag neu: ``pre_pipeline_reference`` = Eingabe),
   * transient_energie DOPPELT auf demselben Audio: Vertrag NEU vs. Vertrag ALT
     (``alignment_tolerance_ms=0.0, exclude_impulse_onsets=False`` = Messpfad
     vor dem Fix) — blind verlost, Schlüssel getrennt (B3-Muster),
   * B1: ungeplante ML→DSP-Rückfälle (fallback_ledger).
2. ``compare``: blind Paarvergleich zweier ``run``-JSONs (alt vs. neu) je
   Fenster auf musik_mos/defektlast inkl. Sign-Test + Never-worsen, dazu der
   Scorecard-Diff je Goal gegen den Lauf 2026-11-03.

Aufruf (die AUSFÜHRUNG macht der Manager — kein Shell in der Worker-Runtime):
    nice -n 19 python3 -B scripts/b3_transient_energie_neverworsen.py run --label neu
    nice -n 19 python3 -B scripts/b3_transient_energie_neverworsen.py compare \
        --alt docs/reports/current/b3_transient_energie_alt.json \
        --neu docs/reports/current/b3_transient_energie_neu.json
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

JND_TOLERANZ = 0.05  # MOS-Punkte — Never-worsen-Toleranz (wie b3_stufenmessung.py)

# Vergleichsbasis: GOAL_SCORECARD des Laufs 2026-11-03 (Log-Zitat im Auftrag
# „b3_wiederholung_W6_W7"; Log-Datei nicht im Worktree — Werte laut Auftrag):
#   transient_energie score=0.1609, thr=0.7000, gap=-0.5391, pass=0
#   transparenz  score=0.5226, thr=0.5500 (pass=0)
BASELINE_2026_11_03: dict[str, float] = {
    "transient_energie": 0.1609,
    "transparenz": 0.5226,
}


def _restore_fenster(audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict, str]:
    """Produktionspfad wie b3_stufenmessung._restore_fenster (FAST-Profil).

    Rückgabe: (Mono-Audio, fallback_ledger-Snapshot, material_type-String).
    """
    from backend.core.fallback_ledger import install_log_counter, snapshot
    from backend.core.fallback_ledger import reset as ledger_reset
    from backend.core.performance_guard import QualityMode
    from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3

    install_log_counter()
    # Delta-Semantik (§G1 (copilot-instructions.md): Song-Isolation) — Snapshot nur für DIESEN Lauf.
    ledger_reset()
    cfg = RestorationConfig(
        mode=QualityMode.FAST,
        enable_performance_guard=True,
        enable_phase_gate=True,
        enable_phase_skipping=True,
    )
    restorer = UnifiedRestorerV3(config=cfg)
    ergebnis = restorer.restore(audio, sample_rate=sr, mode="fast", ml_runtime_budget_s=2.0)
    # Stereo-Layout-Invariante (AGENTS.md §3): pipeline-intern (C, N).
    a = np.asarray(ergebnis.audio, dtype=np.float32)
    if a.ndim == 2 and a.shape[0] <= 8 and a.shape[-1] > 8:
        a = a.T
    if a.ndim == 2:
        a = a.mean(axis=1)
    mat = str(getattr(getattr(ergebnis, "material_type", None), "value", "unknown"))
    return a, snapshot(), mat


def _zu_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2 and a.shape[0] <= 2 and a.shape[1] > a.shape[0]:
        return np.asarray(a.mean(axis=0), dtype=np.float32)
    if a.ndim == 2:
        return np.asarray(a.mean(axis=1), dtype=np.float32)
    return a


def _bewerte(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """(musik_mos, defektlast) — beide Metriken sind richtungszertifiziert."""
    from plugins.musik_mos_plugin import get_musik_mos
    from scripts.quality_metric_direction_gate import _defektlast

    return float(get_musik_mos().estimate_mos(audio, sr).mos), _defektlast(audio, sr)


def _scorecard_messen(audio_in: np.ndarray, audio_out: np.ndarray, sr: int, material: str) -> dict[str, float]:
    """15-Goal-Scorecard im Vertrag NEU (pre_pipeline_reference = Eingabe)."""
    from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

    checker = MusicalGoalsChecker(mode="restoration")
    return dict(
        checker.measure_all(
            np.asarray(audio_out, dtype=np.float32),
            sr,
            reference=np.asarray(audio_in, dtype=np.float32),
            material_type=material,
            pre_pipeline_reference=np.asarray(audio_in, dtype=np.float32),
        )
    )


def _auf_messrate_48k(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Deterministische Resampelung auf die Mess-Rate der Metrik
    (§G5 (copilot-instructions.md); 01_musical_goals.md §1.4.6: 48 kHz).

    Corpus-Fenster liegen teils mit 44,1 kHz vor; ohne Normalisierung misst
    die Metrik gegen eine falsche Rate und der Never-worsen-Beleg ist wertlos.
    Die Sample-Achse wird layout-sicher bestimmt (Stereo-Layout-Invariante:
    ``_load_clip`` liefert (N, C), pipeline-intern gilt (C, N) — AGENTS.md §3;
    ein Resample entlang der Kanal-Achse erzeugt Schein-Kanäle).
    """
    from fractions import Fraction

    from scipy.signal import resample_poly

    audio = np.asarray(audio, dtype=np.float32)
    if sr == 48000:
        return audio, 48000
    if audio.ndim == 2 and audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]:
        sample_achse = 1  # (C, N)
    else:
        sample_achse = 0  # (N,) oder (N, C)
    _g = Fraction(48000, sr)
    audio = resample_poly(audio, _g.numerator, _g.denominator, axis=sample_achse).astype(np.float32)
    return audio, 48000


def _kontrakt_doppelmessung(audio_in: np.ndarray, audio_out: np.ndarray, sr: int, material: str) -> tuple[dict, dict]:
    """(neu, alt) — transient_energie unter beiden Messverträgen auf selbem Audio."""
    from backend.core.musical_goals.transient_energy_metric import get_transient_energy_metric

    met = get_transient_energy_metric()
    kwargs = {"audio_input": audio_in, "audio_restored": audio_out, "sr": sr, "material_type": material}
    neu = met.measure_transient_energy(**kwargs)
    alt = met.measure_transient_energy(
        **kwargs,
        alignment_tolerance_ms=0.0,
        exclude_impulse_onsets=False,
    )
    return neu, alt


def run_messen(label: str, fenster: list[tuple[str, np.ndarray, int]], seed: int) -> dict:
    """Ein Lauf über den Fensterplan — inkl. blindem Kontrakt-A/B je Fenster."""
    rng = random.Random(f"{seed}:kontrakt:{label}")
    schluessel: dict[str, str] = {}
    zeilen: list[dict] = []
    defekt_gesamt = 0

    for i, (clip, audio, sr) in enumerate(fenster):
        tid = f"{clip}__w{i}"
        eingabe = _zu_mono(audio)
        ausgabe, snap, material = _restore_fenster(audio, sr)
        defekt_gesamt += int(snap["by_art"].get("defekt", 0))

        mos, last = _bewerte(ausgabe, sr)
        scorecard = _scorecard_messen(eingabe, ausgabe, sr, material)
        neu, alt = _kontrakt_doppelmessung(eingabe, ausgabe, sr, material)

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
                "kontrakt_blind": {
                    k: {
                        "score": v.get("transient_energy_score"),
                        "score_unfiltered": v.get("transient_energy_score_unfiltered"),
                        "n_onsets": v.get("n_onsets_detected"),
                        "n_excluded_impulse": v.get("n_onsets_excluded_impulse"),
                    }
                    for k, v in blindwerte.items()
                },
            }
        )

    # Entblindung
    for z in zeilen:
        neu_seite = "links" if schluessel[z["trial_id"]] == "neu" else "rechts"
        alt_seite = "rechts" if neu_seite == "links" else "links"
        z["transient_neu"] = z["kontrakt_blind"][neu_seite]
        z["transient_alt"] = z["kontrakt_blind"][alt_seite]
        del z["kontrakt_blind"]

    return {
        "schema": "aurik.transient_energie.messung/1",
        "label": label,
        "seed": seed,
        "blind_schluessel": schluessel,
        "defekt_fallbacks": defekt_gesamt,
        "zeilen": zeilen,
        "quelle": "scripts/b3_transient_energie_neverworsen.py",
    }


def _sign_test(p_wert_fn, vorteile: int, gesamt: int) -> float:
    return float(p_wert_fn(vorteile, gesamt))


def compare_bewerten(alt_lauf: dict, neu_lauf: dict, seed: int) -> dict:
    """Blind Paarvergleich alt vs. neu je Fenster + Scorecard-Diff je Goal."""
    from scripts.ab_harness import sign_test_two_sided

    rng = random.Random(f"{seed}:laufvergleich")
    alt_zeilen = {z["trial_id"]: z for z in alt_lauf["zeilen"]}
    neu_zeilen = {z["trial_id"]: z for z in neu_lauf["zeilen"]}
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
        zeilen.append(
            {
                "trial_id": p["trial_id"],
                "mos_alt": alt["musik_mos"],
                "mos_neu": neu["musik_mos"],
                "last_alt": alt["defektlast"],
                "last_neu": neu["defektlast"],
                "transient_alt": alt["transient_alt"]["score"],
                "transient_neu": neu["transient_neu"]["score"],
            }
        )

    mos_deltas = [z["mos_neu"] - z["mos_alt"] for z in zeilen]
    last_deltas = [z["last_neu"] - z["last_alt"] for z in zeilen]
    never_worsen = all(d >= -JND_TOLERANZ for d in mos_deltas) and all(d <= JND_TOLERANZ for d in last_deltas)
    vorteile = sum(1 for d in mos_deltas if d > 0)
    p_wert = _sign_test(sign_test_two_sided, vorteile, len(mos_deltas))
    signifikant = bool(p_wert < 0.05 and sum(mos_deltas) > 0)

    # Scorecard-Diff je Goal (neu − alt) und gegen die Vergleichsbasis 2026-11-03
    ziele = sorted({k for z in neu_lauf["zeilen"] for k in z["scorecard"]})
    scorecard_diff = {}
    for g in ziele:
        werte_alt = [z["scorecard"].get(g) for z in alt_lauf["zeilen"] if z["scorecard"].get(g) is not None]
        werte_neu = [z["scorecard"].get(g) for z in neu_lauf["zeilen"] if z["scorecard"].get(g) is not None]
        m_alt = float(np.mean(werte_alt)) if werte_alt else None
        m_neu = float(np.mean(werte_neu)) if werte_neu else None
        eintrag = {
            "mittel_alt": round(m_alt, 4) if m_alt is not None else None,
            "mittel_neu": round(m_neu, 4) if m_neu is not None else None,
            "delta": round(m_neu - m_alt, 4) if (m_alt is not None and m_neu is not None) else None,
            "baseline_2026_11_03": BASELINE_2026_11_03.get(g),
        }
        scorecard_diff[g] = eintrag

    return {
        "schema": "aurik.transient_energie.beschluss/1",
        "seed": seed,
        "n_fenster": len(zeilen),
        "zeilen": zeilen,
        "never_worsen": bool(never_worsen),
        "vorteile": f"{vorteile}/{len(mos_deltas)}",
        "p_wert": round(p_wert, 6),
        "signifikant": signifikant,
        "scorecard_diff": scorecard_diff,
        "baseline_2026_11_03": BASELINE_2026_11_03,
        "ergebnis": "ERFUELLT" if never_worsen else "ABGELEHNT",
        "quelle": "scripts/b3_transient_energie_neverworsen.py",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="B3-Mess-Harness transient_energie alt vs. neu (Never-worsen)")
    ap.add_argument("modus", choices=["run", "compare"])
    ap.add_argument("--label", default="neu")
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
        beschluss = compare_bewerten(alt_lauf, neu_lauf, args.seed)
        ziel = out_dir / "b3_transient_energie_beschluss.json"
        ziel.write_text(json.dumps(beschluss, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log.info(
            "Vergleich alt/neu: %s (never_worsen=%s, p=%s, Vorteile=%s) -> %s",
            beschluss["ergebnis"],
            beschluss["never_worsen"],
            beschluss["p_wert"],
            beschluss["vorteile"],
            ziel.name,
        )
        return 0

    from scripts.release_candidate_pipeline import _existing_corpus, _load_clip

    fenster: list[tuple[str, np.ndarray, int]] = []
    for eintrag in _existing_corpus()[: args.clips]:
        pfad = Path(eintrag["path"])
        # Fensterplan IDENTISCH zu scripts/b3_stufenmessung.py (Starts alle 2,5 s).
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
    ziel = out_dir / f"b3_transient_energie_{args.label}.json"
    ziel.write_text(json.dumps(lauf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("Lauf '%s' geschrieben: %s (B1 defekt_fallbacks=%d)", args.label, ziel.name, lauf["defekt_fallbacks"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
