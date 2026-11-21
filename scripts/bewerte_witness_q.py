"""scripts/bewerte_witness_q.py — §G13-Q-Erstbelege über das gelernte Schadens-Witness.

Die Q≈0-Klassen der Erstbelege (2026-11-12) waren für defektlast blind — die
Q-Erstbelege konnten die Reparatur weder belegen noch widerlegen. Das
supervidierte Witness (``backend/core/dsp/witness_detektor.py``, 7/7 Klassen
Gate-bestanden, Holdout-AUC 0,90–1,00) macht sie messbar: dieses Skript bucht
je Klasse die Witness-Q-Quote über die k398-Regionenpaare
(clean/schaden/restauriert aus data/klassen_paare/manifest.jsonl).

Definition: Q_witness = (P_schaden − P_restauriert) / (P_schaden − P_clean),
gebucht nur bei Clean-Kontrast > 0,05 (sonst nicht bewertbar — dieselbe
Fail-closed-Logik wie defekt_delta /4). Report-only — kein Gate, kein Flag
(§G191 (GEBOTE.md)); der Beleg macht die Sinn-Beweis-Lücke messbar.

CPU-only, deterministisch (Seed 42, §G5 (copilot-instructions.md)).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.core.dsp.witness_detektor import (
    WITNESS_CHECKPOINT_PFADE,
    witness_kanal_verfuegbar,
    witness_schaden_kanal,
)

MANIFEST = _ROOT / "data" / "klassen_paare" / "manifest.jsonl"
REPORT_DIR = _ROOT / "docs" / "reports" / "current"
SCHEMA = "aurik.witness_q/1"
STICHPROBE = 300
KONTRAST_MIN = 0.05


def _lade(pfad: str) -> np.ndarray | None:
    from scipy.io import wavfile

    p = Path(pfad)
    if not p.is_file():
        return None
    roh = wavfile.read(p)
    if not isinstance(roh, tuple) or len(roh) != 2:
        return None
    sr = int(roh[0])
    d = roh[1]
    x = np.asarray(d, np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x, sr


def buche(klasse: str, eintraege: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    gewaehlt = rng.sample(eintraege, min(STICHPROBE, len(eintraege)))
    quoten: list[float] = []
    never_worsen: list[bool] = []
    uebersprungen = 0
    n_bewertbar = 0
    for e in gewaehlt:
        s = _lade(e["schaden"])
        c = _lade(e["clean"])
        r = _lade(e["restauriert"])
        if s is None or c is None or r is None:
            uebersprungen += 1
            continue
        p_s = witness_schaden_kanal(s[0], s[1], klasse)
        p_c = witness_schaden_kanal(c[0], c[1], klasse)
        p_r = witness_schaden_kanal(r[0], r[1], klasse)
        kontrast = p_s - p_c
        if kontrast > KONTRAST_MIN:
            quoten.append((p_s - p_r) / kontrast)
            n_bewertbar += 1
        never_worsen.append(p_r <= p_s + 0.02)
    return {
        "klasse": klasse,
        "regionen_stichprobe": len(gewaehlt),
        "uebersprungen": uebersprungen,
        "n_q_bewertbar": n_bewertbar,
        "q_witness_median": round(float(np.median(quoten)), 4) if quoten else None,
        "anteil_q_ge_0_5": round(float(np.mean([q >= 0.5 for q in quoten])), 4) if quoten else None,
        "never_worsen_anteil": round(float(np.mean(never_worsen)), 4) if never_worsen else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="§G13-Q-Erstbelege über das Schadens-Witness")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stichprobe", type=int, default=STICHPROBE)
    ap.add_argument("--klassen", default=None)
    ap.add_argument(
        "--manifest", type=Path, default=MANIFEST, help="Manifest-JSONL (Default: data/klassen_paare/manifest.jsonl)"
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Beleg-Pfad (Default: docs/reports/current/<datum>_witness_q_erstbelege.json)",
    )
    args = ap.parse_args(argv)

    klassen = args.klassen.split(",") if args.klassen else sorted(WITNESS_CHECKPOINT_PFADE.keys())
    for k in klassen:
        if not witness_kanal_verfuegbar(k):
            print(f"ABBRUCH: Witness-Checkpoint fehlt für {k} — erst scripts/train_witness_detektor.py ausführen.")
            return 2

    eintraege: dict[str, list[dict[str, Any]]] = {}
    with args.manifest.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            eintraege.setdefault(str(d["klasse"]), []).append(d)

    rng = random.Random(args.seed)
    ergebnisse = [buche(k, eintraege.get(k, []), rng) for k in klassen]
    beleg = {"schema": SCHEMA, "datum": date.today().isoformat(), "seed": args.seed, "ergebnisse": ergebnisse}
    ziel = args.report or (REPORT_DIR / f"{date.today().isoformat()}_witness_q_erstbelege.json")
    ziel.write_text(json.dumps(beleg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(ergebnisse, indent=1, ensure_ascii=False))
    print(f"Beleg: {ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
