"""Spreu/Weizen-Trennung der Witness-Kanal-Gate-Belege (Durchbruch-Prüfung).

Trennt die Gate-Belege (`docs/reports/current/*_witness_kopf_gate_<klasse>.json`)
nach den harten Durchbruch-Schwellen — Parity-Register „Harte Regeln" 5(b)
(DRT-Schwellen) + Spec 27 ANT-9 (ehrliches Scheitern):

- **WEIZEN** (echter Durchbruch-Kandidat für den §4-Kanalsatz):
  `q_median >= 0.95` UND `q_anteil_ge_0_5 >= 0.96` UND
  `never_worsen_anteil >= 0.96` UND `musik_mos_delta_anteil_ge_minus_jnd == 1.0`
  (keine Holdout-Region verschlechtert MOS über JND) UND `q_median < 2.0`
  (kein Q-Clip-Symptom = Überrestauration) UND
  `gate_bestanden`/`gate_qualitaet_bestanden` UND mindestens 6 bewertbare
  Schaden/Clean- und Never-worsen-Paare.
- **SPREU**: mindestens ein Kriterium verfehlt — mit präzisem Grund.
  SPREU-Kanäle bleiben report-only und kommen NICHT in den §4-Satz (ANT-9).

Kein Flag-Flip, keine Klangwirkung — reine Klassifikation auf existierenden
Belegen. Fail-closed: ohne Gate-Belege Exit 1.

Usage:
    python3 scripts/trenne_spreu_weizen.py
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_REPORT_DIR = _ROOT / "docs" / "reports" / "current"
_SCHEMA = "aurik.spreu_weizen/1"

# Harte Schwellen (Parity-Register Regel 5(b) + Spec 27 ANT-9; DRT-5: unantastbar)
Q_MEDIAN_MIN = 0.95
Q_ANTEIL_MIN = 0.96
NW_ANTEIL_MIN = 0.96
MOS_ANTEIL_MIN = 0.9995  # == 1.0 bei den auf 4 Stellen gerundeten Belegwerten
Q_CLIP_OBERGRENZE = 2.0  # q == 2.0 = dokumentiertes Clip-Symptom (Überrestauration)
MIN_BEWERTBAR = 6  # mindestens 6 bewertbare Schaden-/Clean-Paare (Regel 5(b))


def _lade_letzte_belege(report_dir: Path) -> dict[str, dict]:
    """Neuesten Gate-Beleg je Klasse (nach mtime, deterministisch je Klasse)."""
    belege: dict[str, dict] = {}
    for pfad in sorted(report_dir.glob("*_witness_kopf_gate_*.json")):
        try:
            daten = json.loads(pfad.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        klasse = daten.get("klasse")
        if not isinstance(klasse, str) or not klasse:
            continue
        alt = belege.get(klasse)
        if alt is None or pfad.stat().st_mtime > alt["_pfad"].stat().st_mtime:
            belege[klasse] = {**daten, "_pfad": pfad}
    return belege


def _pruefe(daten: dict) -> tuple[bool, list[str]]:
    gruende: list[str] = []
    if not daten.get("gate_bestanden"):
        gruende.append("gate_bestanden=false")
    if not daten.get("gate_qualitaet_bestanden"):
        gruende.append("gate_qualitaet_bestanden=false")
    q = daten.get("q_median")
    if not isinstance(q, (int, float)):
        gruende.append("q_median fehlt")
    else:
        if q >= Q_CLIP_OBERGRENZE:
            gruende.append(f"q_clip: q_median {q} >= 2.0 (Überrestauration)")
        if q < Q_MEDIAN_MIN:
            gruende.append(f"q_median {q} < 0.95")
    qa = daten.get("q_anteil_ge_0_5")
    if not isinstance(qa, (int, float)):
        gruende.append("q_anteil fehlt")
    elif qa < Q_ANTEIL_MIN:
        gruende.append(f"q_anteil {qa} < 0.96")
    nw = daten.get("never_worsen_anteil")
    if not isinstance(nw, (int, float)):
        gruende.append("never_worsen_anteil fehlt")
    elif nw < NW_ANTEIL_MIN:
        gruende.append(f"never_worsen {nw} < 0.96")
    mos = daten.get("musik_mos_delta_anteil_ge_minus_jnd")
    if not isinstance(mos, (int, float)):
        gruende.append("mos_anteil fehlt (kein --mos-Beleg)")
    elif mos < MOS_ANTEIL_MIN:
        gruende.append(f"mos_anteil {mos} < 1.0 (MOS-Verschlechterung über JND)")
    nqb = daten.get("n_q_bewertbar")
    if not isinstance(nqb, (int, float)) or nqb < MIN_BEWERTBAR:
        gruende.append(f"n_q_bewertbar {nqb} < 6")
    nwb = daten.get("n_never_worsen_bewertbar")
    if not isinstance(nwb, (int, float)) or nwb < MIN_BEWERTBAR:
        gruende.append(f"n_never_worsen_bewertbar {nwb} < 6")
    return (not gruende), gruende


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-dir", type=Path, default=_REPORT_DIR)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    belege = _lade_letzte_belege(args.report_dir)
    if not belege:
        print("FEHLER: keine Witness-Gate-Belege gefunden (fail-closed).")
        return 1

    klassen: dict[str, dict] = {}
    weizen: list[str] = []
    spreu: dict[str, list[str]] = {}
    for klasse in sorted(belege):
        daten = belege[klasse]
        bestanden, gruende = _pruefe(daten)
        eintrag = {
            "klasse": klasse,
            "urteil": "WEIZEN" if bestanden else "SPREU",
            "q_median": daten.get("q_median"),
            "q_anteil_ge_0_5": daten.get("q_anteil_ge_0_5"),
            "never_worsen_anteil": daten.get("never_worsen_anteil"),
            "musik_mos_delta_anteil_ge_minus_jnd": daten.get("musik_mos_delta_anteil_ge_minus_jnd"),
            "n_q_bewertbar": daten.get("n_q_bewertbar"),
            "n_never_worsen_bewertbar": daten.get("n_never_worsen_bewertbar"),
            "checkpoint": daten.get("checkpoint"),
            "beleg": daten["_pfad"].name,
        }
        if bestanden:
            eintrag["gruende"] = []
            weizen.append(klasse)
        else:
            eintrag["gruende"] = gruende
            spreu[klasse] = gruende
        klassen[klasse] = eintrag

    beleg = {
        "schema": _SCHEMA,
        "datum": date.today().isoformat(),
        "schwellen": {
            "q_median_min": Q_MEDIAN_MIN,
            "q_anteil_min": Q_ANTEIL_MIN,
            "never_worsen_min": NW_ANTEIL_MIN,
            "mos_anteil_min": MOS_ANTEIL_MIN,
            "q_clip_obergrenze": Q_CLIP_OBERGRENZE,
            "min_bewertbar": MIN_BEWERTBAR,
        },
        "weizen": weizen,
        "spreu": spreu,
        "klassen": klassen,
    }
    ziel = args.out or (_REPORT_DIR / f"{date.today().isoformat()}_spreu_weizen_trennung.json")
    ziel.write_text(json.dumps(beleg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"WEIZEN ({len(weizen)}): {', '.join(weizen) or '—'}")
    print(f"SPREU ({len(spreu)}):")
    for klasse, gruende in spreu.items():
        print(f"  {klasse}: {'; '.join(gruende)}")
    print(f"Beleg: {ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
