#!/usr/bin/env python3
"""B3-Stufenmessung — Objektivbeschluss-Vorlage für die 5 Ausnahme-Stufen.

Die Stufen W-1 (HR-V1), W-4 (MP-SENet), W-5 (DFN), W-6 (MIIPHER-DiT),
W-7 (Harmonic-Inpainting) laufen als „Ausnahme“ ohne den Beleg „blindes A/B“
(Wohlklang-Vertrag). Dieses Skript erbringt ihn über den **formalen
Objektivmetrik-Beschluss** (Vertragsabschnitt B3):

  * Paarvergleich: Restore MIT Schalter vs. OHNE Schalter, je Fenster;
  * Bewertung nur mit **richtungszertifizierten** Metriken (MusikMOS,
    Defektlast — `docs/reports/current/qualitaetsmetrik_richtungszertifikat.json`);
  * Blindheit: die Seiten werden per Seed verlost (Schlüssel getrennt),
    die Bewertung sieht keine Labels;
  * Never-worsen (G-4): keine Metrik je Fenster signifikant schlechter;
  * Signifikanz: Vorzeichen-Test (exakt) über die Fenster, p < 0,05;
  * B1: ungeplante ML→DSP-Rückfälle = 0 (`backend/core/fallback_ledger.py`).

Ergebnis: je Stufe ein Beschluss-JSON (Schema `aurik.objektivbeschluss/1`)
für `scripts/flag_freigabe.py --objektivbeschluss` — **nur wenn alle
Bedingungen erfüllt sind**, sonst ein begründeter Ablehnungs-Report.

Aufruf (host-sicher, parallel zur GPU-Kette):
    nice -n 15 python3 -B scripts/b3_stufenmessung.py [--clips 3] [--windows 2] [--seed 42]
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

JND_TOLERANZ = 0.05  # MOS-Punkte — Never-worsen-Toleranz (Audibility-Stufe)


def _schalter_lesen(name: str) -> tuple[object, str, object]:
    """Liefert (Modul, Attribut, aktueller Wert) je Stufen-Schalter."""
    import backend.core.music_model_flags as mmf
    import plugins.bigvgan_v2_plugin as bv

    mapping = {
        "W-1": (bv, "BIGVGAN_V2_HR_ACTIVATED"),
        "W-4": (mmf, "use_mp_senet_musik"),
        "W-5": (mmf, "use_df_musik"),
        "W-6": (mmf, "use_miipher_dit"),
        "W-7": (mmf, "use_harmonic_inpainting"),
    }
    modul, attr = mapping[name]
    return modul, attr, getattr(modul, attr)


def _schalter_setzen(name: str, wert: object) -> None:
    modul, attr, _ = _schalter_lesen(name)
    setattr(modul, attr, wert)


def _restore_fenster(audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
    """Produktionspfad wie im Kind der Release-Pipeline (FAST-Profil)."""
    from backend.core.fallback_ledger import install_log_counter, snapshot
    from backend.core.fallback_ledger import reset as ledger_reset
    from backend.core.performance_guard import QualityMode
    from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3

    install_log_counter()
    # Delta-Semantik (§G1 (copilot-instructions.md): Song-Isolation, fallback_ledger.reset): der Snapshot
    # am Ende darf nur DIESEN Lauf zählen — ohne Reset summieren sich kumulative
    # Stände je Fenster und verfälschen die B1-Kennzahl (WOHLKLANG_CLAIMS.md).
    ledger_reset()
    cfg = RestorationConfig(
        mode=QualityMode.FAST,
        enable_performance_guard=True,
        enable_phase_gate=True,
        enable_phase_skipping=True,
    )
    restorer = UnifiedRestorerV3(config=cfg)
    ergebnis = restorer.restore(audio, sample_rate=sr, mode="fast", ml_runtime_budget_s=2.0)
    # Stereo-Layout-Invariante (AGENTS.md §3): pipeline-intern (C, N) — ohne
    # Normierung kollabiert mean(axis=1) auf C Samples (Produktionsbefund).
    a = np.asarray(ergebnis.audio, dtype=np.float32)
    if a.ndim == 2 and a.shape[0] <= 8 and a.shape[-1] > 8:
        a = a.T
    if a.ndim == 2:
        a = a.mean(axis=1)
    return a, snapshot()


def _bewerte(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """(musik_mos, defektlast) — beide Metriken sind richtungszertifiziert."""
    from plugins.musik_mos_plugin import get_musik_mos
    from scripts.quality_metric_direction_gate import _defektlast

    return float(get_musik_mos().estimate_mos(audio, sr).mos), _defektlast(audio, sr)


def stufe_messen(name: str, fenster: list[tuple[str, np.ndarray, int]], seed: int) -> dict:
    """Misst eine Stufe blind: Paare je Fenster, Auswertung erst nach Entblindung."""
    from scripts.ab_harness import sign_test_two_sided

    rng = random.Random(f"{seed}:{name}")
    schluessel: dict[str, str] = {}
    paare: list[dict] = []
    defekt_gesamt = 0

    for i, (clip, audio, sr) in enumerate(fenster):
        _schalter_setzen(name, False)
        aus_audio, snap_aus = _restore_fenster(audio, sr)
        _schalter_setzen(name, True)
        an_audio, snap_an = _restore_fenster(audio, sr)
        defekt_gesamt += int(snap_aus["by_art"].get("defekt", 0)) + int(snap_an["by_art"].get("defekt", 0))

        # Blindheit: Seiten verlosen, Bewertung sieht nur "links"/"rechts"
        links_ist_an = rng.random() < 0.5
        tid = f"{clip}__w{i}"
        schluessel[tid] = "an" if links_ist_an else "aus"
        werte = {}
        for seite, sig in (
            ("links", an_audio if links_ist_an else aus_audio),
            ("rechts", aus_audio if links_ist_an else an_audio),
        ):
            mos, last = _bewerte(sig, sr)
            werte[seite] = {"musik_mos": mos, "defektlast": last}
        paare.append({"trial_id": tid, "blindwerte": werte})

    # Entblindung + Auswertung
    zeilen = []
    for p in paare:
        an_seite = "links" if schluessel[p["trial_id"]] == "an" else "rechts"
        aus_seite = "rechts" if an_seite == "links" else "links"
        an, aus = p["blindwerte"][an_seite], p["blindwerte"][aus_seite]
        zeilen.append(
            {
                "trial_id": p["trial_id"],
                "mos_an": an["musik_mos"],
                "mos_aus": aus["musik_mos"],
                "last_an": an["defektlast"],
                "last_aus": aus["defektlast"],
            }
        )

    mos_deltas = [z["mos_an"] - z["mos_aus"] for z in zeilen]
    last_deltas = [z["last_an"] - z["last_aus"] for z in zeilen]
    never_worsen = all(d >= -JND_TOLERANZ for d in mos_deltas) and all(d <= JND_TOLERANZ for d in last_deltas)
    vorteile = sum(1 for d in mos_deltas if d > 0)
    p_wert = sign_test_two_sided(vorteile, len(mos_deltas))
    signifikant = bool(p_wert < 0.05 and sum(mos_deltas) > 0)
    b1_sauber = defekt_gesamt == 0

    erfuellt = bool(never_worsen and signifikant and b1_sauber)
    return {
        "stufe": name,
        "zeilen": zeilen,
        "never_worsen": never_worsen,
        "vorteile": f"{vorteile}/{len(mos_deltas)}",
        "p_wert": round(p_wert, 6),
        "signifikant": signifikant,
        "defekt_fallbacks": defekt_gesamt,
        "b1_sauber": b1_sauber,
        "erfuellt": erfuellt,
    }


def beschluss_schreiben(name: str, messung: dict, out_dir: Path, seed: int) -> Path:
    """Objektivbeschluss-JSON (Schema aurik.objektivbeschluss/1) — oder Ablehnung."""
    erfuellt = messung["erfuellt"]
    daten = {
        "schema": "aurik.objektivbeschluss/1",
        "flag": {
            "W-1": "BIGVGAN_V2_HR_ACTIVATED",
            "W-4": "use_mp_senet_musik",
            "W-5": "use_df_musik",
            "W-6": "use_miipher_dit",
            "W-7": "use_harmonic_inpainting",
        }[name],
        "stufe": name,
        "referenz_songs": len({z["trial_id"].split("__")[0] for z in messung["zeilen"]}),
        "seed": seed,
        "never_worsen": bool(messung["never_worsen"]),
        "signifikant": bool(messung["signifikant"]),
        "defekt_fallbacks": messung["defekt_fallbacks"],
        "blind_ausgewertet": True,
        "messung": "",
        "zielmetrik": {"name": "musik_mos (G-2)", "vorteile": messung["vorteile"], "p_wert": messung["p_wert"]},
        "ergebnis": "ERFUELLT" if erfuellt else "ABGELEHNT",
        "quelle": "scripts/b3_stufenmessung.py",
    }
    ziel = out_dir / f"b3_objektivbeschluss_{name}.json"
    ziel.write_text(json.dumps(daten, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    roh = out_dir / f"b3_stufenmessung_{name}.json"
    roh.write_text(json.dumps(messung, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    daten["messung"] = str(roh.relative_to(PROJECT_ROOT))
    ziel.write_text(json.dumps(daten, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ziel


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="B3-Stufenmessung (Objektivbeschluss fuer Ausnahme-Stufen)")
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stages", default="W-1,W-4,W-5,W-6,W-7")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from scripts.release_candidate_pipeline import _existing_corpus, _load_clip

    fenster: list[tuple[str, np.ndarray, int]] = []
    for eintrag in _existing_corpus()[: args.clips]:
        pfad = Path(eintrag["path"])
        # Adaptive Fenster: Starts alle 2,5 s (Clips 7–10 s => 3–4 Fenster je
        # Clip => ~10–12 Paare). Statistik-Pflicht: ab 6 Paaren ist p<0,05
        # ueberhaupt erreichbar (5 Paare: selbst 5/5 nur p=0,0625).
        for w_start in (0.0, 2.5, 5.0, 7.5):
            try:
                audio, sr = _load_clip(pfad, start_s=w_start, seconds=args.seconds)
            except ValueError:
                continue
            fenster.append((pfad.stem, np.asarray(audio, dtype=np.float32), int(sr)))
    if not fenster:
        log.error("kein Referenzmaterial")
        return 2

    out_dir = PROJECT_ROOT / "docs" / "reports" / "current"
    for name in args.stages.split(","):
        name = name.strip()
        log.info("Stufe %s messen (%d Fenster)...", name, len(fenster))
        try:
            messung = stufe_messen(name, fenster, args.seed)
        except Exception as exc:
            log.error("Stufe %s fehlgeschlagen (%s)", name, exc)
            (out_dir / f"b3_stufenmessung_{name}.json").write_text(
                json.dumps({"stufe": name, "fehler": str(exc)[:200], "erfuellt": False}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            continue
        ziel = beschluss_schreiben(name, messung, out_dir, args.seed)
        log.info(
            "Stufe %s: %s (never_worsen=%s, signifikant=%s, B1=%s) -> %s",
            name,
            messung["erfuellt"],
            messung["never_worsen"],
            messung["signifikant"],
            messung["b1_sauber"],
            ziel.name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
