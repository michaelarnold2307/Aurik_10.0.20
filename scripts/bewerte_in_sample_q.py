#!/usr/bin/env python3
"""bewerte_in_sample_q — Pre-Flight-Gate des Aktivierungs-Fahrplans (B-0(3)).

`docs/reports/current/2026-11-13_aktivierungs_fahrplan.md` verlangt: Vor
jeder Holdout-Bewertung eines Per-Klassen-Kopfes muss die In-Sample-Q auf
dem Train-Split (300 deterministische Regionen) ≥ `REPARATURQUOTE_ZIEL`
(0,5) sein — sonst ist der Ansatz zu verwerfen, NICHT mehr Epochen zu
trainieren. Grund: Der In-Sample-Q-Orkel (median 0,0101, Report
`2026-11-13_aktivierungs_fahrplan.md`) bewies, dass das L1-Wellenform-
Objektiv blind für die §G13 (copilot-instructions.md)-Gate-Metrik ist;
ein Modell, das nicht einmal auf den Trainingsdaten Q erzeugt, kann den
Holdout nicht bestehen.

Q-Definition identisch zu `bewerte_klassen_kopf.py` (eine Wahrheit):
Q = (defektlast_schaden − defektlast_repariert) / (defektlast_schaden −
defektlast_clean), nur bei Clean-Kontrast > JND 0,05. Report-Only-Zeuge,
kein Flag-Flip — Freigaben ausschließlich über §4.

Aufruf:
    python3 -B scripts/bewerte_in_sample_q.py \
        --klasse groove_echo --manifest data/klassen_paare_kombi_gefiltert/manifest.jsonl \
        --checkpoint models/klassen_koepfe/groove_echo_proxy/checkpoint_latest.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.dsp.defekt_delta import REPARATURQUOTE_ZIEL, defektlast_lang
from backend.core.dsp.wohlklang_zeuge import JND_TOLERANZ

_REGIONEN_PREFLIGHT = 300


def _gate_entscheidung(q_median: float | None, n_q: int) -> bool:
    """Pre-Flight-Gate: Q >= Ziel UND mindestens 6 Q-fähige Paare (§G13 (copilot-instructions.md))."""
    return bool(n_q >= 6 and q_median is not None and q_median >= REPARATURQUOTE_ZIEL)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="In-Sample-Q-Pre-Flight (Fahrplan B-0(3))")
    ap.add_argument("--klasse", default="groove_echo")
    ap.add_argument("--manifest", default="data/klassen_paare/manifest.jsonl")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    from scripts.bewerte_klassen_kopf import _fenster, _q_berechnen, _titel_hash, _wav_lesen
    from scripts.train_klassen_kopf import _MODEL_SCHEMA, WINDOW, build_model

    manifest_pfad = ROOT / args.manifest
    if not manifest_pfad.exists():
        print(f"FEHLER: Manifest fehlt: {manifest_pfad}", file=sys.stderr)
        return 2
    manifest_bytes = manifest_pfad.read_bytes()
    zeilen = [json.loads(l) for l in manifest_bytes.decode("utf-8").splitlines() if l.strip()]
    eintraege = [z for z in zeilen if z.get("klasse") == args.klasse and z.get("clean") and z.get("titel")]
    if not eintraege:
        print(f"FEHLER: keine clean-Einträge für {args.klasse}", file=sys.stderr)
        return 2

    ckpt_pfad = Path(args.checkpoint)
    if not ckpt_pfad.exists():
        print(f"FEHLER: Checkpoint fehlt: {ckpt_pfad}", file=sys.stderr)
        return 2
    device = torch.device("cpu")
    try:
        res = torch.load(ckpt_pfad, map_location=device, weights_only=False)
        if res.get("model_schema") != _MODEL_SCHEMA:
            raise ValueError("Checkpoint-Modellvertrag fehlt oder ist inkompatibel")
        provenance = res.get("split_provenance", {})
        if provenance.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
            raise ValueError("Manifest-Hash stimmt nicht mit dem Checkpoint überein")
        train_titles = set(provenance.get("train_titel_sha256", []))
        holdout_title = provenance.get("holdout_titel_sha256")
        if not train_titles or not holdout_title or holdout_title in train_titles:
            raise ValueError("Checkpoint-Titel-Split ist unvollständig oder überlappt")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FEHLER: Checkpoint-Provenienz ungültig: {exc}", file=sys.stderr)
        return 2

    # Nur Train-Split-Regionen, deterministisch sortiert, Obergrenze 300 (§G5 (copilot-instructions.md)).
    train_entries = [e for e in eintraege if _titel_hash(str(e.get("titel", "")).strip()) in train_titles]
    if not train_entries:
        print("FEHLER: Manifest enthält keine Train-Split-Regionen", file=sys.stderr)
        return 2
    train_entries.sort(key=lambda z: z.get("sha256_pcm_schaden", ""))
    train_entries = train_entries[:_REGIONEN_PREFLIGHT]

    model = build_model(device)
    try:
        model.load_state_dict(res["modell"])
    except (KeyError, RuntimeError, TypeError) as exc:
        print(f"FEHLER: Checkpoint-Modellzustand ungültig: {exc}", file=sys.stderr)
        return 2
    model.eval()

    quoten: list[float] = []
    n_q_unter_jnd = 0
    zeilen: list[dict] = []
    for z in train_entries:
        sr, schaden = _wav_lesen(ROOT / z["schaden"])
        sr_clean, clean = _wav_lesen(ROOT / z["clean"])
        if sr != sr_clean:
            print(f"FEHLER: Sample-Raten unterscheiden sich: {z['schaden']} / {z['clean']}", file=sys.stderr)
            return 2
        schaden_w = _fenster(schaden, WINDOW)
        clean_w = _fenster(clean, WINDOW)
        with torch.no_grad():
            repariert = model(torch.from_numpy(schaden_w).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
        last_s = float(defektlast_lang(schaden_w, sr))
        last_r = float(defektlast_lang(repariert, sr))
        last_c = float(defektlast_lang(clean_w, sr))
        q = _q_berechnen(last_s, last_r, last_c)
        if q is None:
            n_q_unter_jnd += 1
        else:
            quoten.append(q)
        zeilen.append(
            {
                "sha256": z.get("sha256_pcm_schaden", "")[:8],
                "defektlast_schaden": round(last_s, 5),
                "defektlast_repariert": round(last_r, 5),
                "defektlast_clean": round(last_c, 5),
                "q": round(q, 4) if q is not None else None,
            }
        )

    q_median = float(np.median(quoten)) if quoten else None
    bestanden = _gate_entscheidung(q_median, len(quoten))
    befund = {
        "schema": "aurik.in_sample_q/1",
        "klasse": args.klasse,
        "checkpoint": str(ckpt_pfad),
        "checkpoint_sha256": hashlib.sha256(ckpt_pfad.read_bytes()).hexdigest(),
        "checkpoint_epoch": int(res.get("epoch", -1)) + 1,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "regionen_ausgewaehlt": len(train_entries),
        "regionen_ziel": _REGIONEN_PREFLIGHT,
        "n_q": len(quoten),
        "n_q_unter_jnd_ausgeschlossen": n_q_unter_jnd,
        "q_jnd": JND_TOLERANZ,
        "q_median": round(q_median, 4) if q_median is not None else None,
        "q_p25": round(float(np.percentile(quoten, 25)), 4) if quoten else None,
        "q_p75": round(float(np.percentile(quoten, 75)), 4) if quoten else None,
        "q_ziel": REPARATURQUOTE_ZIEL,
        "bestanden": bestanden,
        "hinweis": (
            "Pre-Flight-Gate (Fahrplan B-0(3)): ohne bestanden=true keine Holdout-"
            "Bewertung; Report-Only-Zeuge, kein Flag-Flip, Freigaben nur §4."
        ),
        "zeilen": zeilen[:60],
    }
    out_pfad = (
        Path(args.out) if args.out else (ROOT / "docs" / "reports" / "current" / f"in_sample_q_{args.klasse}.json")
    )
    out_pfad.parent.mkdir(parents=True, exist_ok=True)
    out_pfad.write_text(json.dumps(befund, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Beleg: {out_pfad}")
    print(
        f"In-Sample-Q median={befund['q_median']} (Ziel >= {REPARATURQUOTE_ZIEL}) -> "
        f"{'BESTANDEN' if bestanden else 'VERFEHLT'} | "
        f"n={len(zeilen)}, Q-fähig={len(quoten)}, unter JND={n_q_unter_jnd}"
    )
    return 0 if bestanden else 1


if __name__ == "__main__":
    raise SystemExit(main())
