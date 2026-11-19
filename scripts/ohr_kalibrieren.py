"""ohr_kalibrieren — „Mein Ohr“: persönliche A/B-Urteile in den Ohr-Prior fitten.

Liest eine Urteils-Session (JSON, Schema `aurik.ohr_urteile/1`), berechnet die
Feature-Differenzen über `backend.core.dsp.ohr_prior`, fittet den
Bradley-Terry-Head und schreibt das Artefakt `ohr_prior.json` (Schema
`aurik.ohr_prior/1`) — deterministisch (§G5 (copilot-instructions.md)),
CPU-only, kein Flag-Flip, kein Deploy.

Session-Schema (aurik.ohr_urteile/1):
{
  "schema": "aurik.ohr_urteile/1",
  "sr_hint": 48000,
  "pairs": [
    {"a": "pfad/zu/a.wav", "b": "pfad/zu/b.wav", "choice": "A"},
    {"a": "...", "b": "...", "choice": "B"}
  ]
}
`choice` = "A" bedeutet: A klang für DICH besser (Wohlklang/Defektbehebung).

Verwendung:
  python3 -B scripts/ohr_kalibrieren.py --urteile urteile.json \
      --out models/ohr/ohr_prior.json [--seed 42] [--pseudo-count 10]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def _load_audio(path: str, sr: int) -> np.ndarray:
    """Lädt Mono@`sr` — abweichende Datei-Raten werden deterministisch
    resampelt (§G5 (copilot-instructions.md)); kein Hard-Fail auf 44,1-kHz-
    Quellen (Produktionsbefund: restaurierte Exporte sind 44,1 kHz).
    """
    import math

    import soundfile as sf
    from scipy.signal import resample_poly

    data, file_sr = sf.read(path, always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    x = np.asarray(data, dtype=np.float64)
    if file_sr != sr:
        gcd = math.gcd(file_sr, sr)
        x = resample_poly(x, sr // gcd, file_sr // gcd)
        logger.info("ohr_kalibrieren: %s von %d Hz auf %d Hz resampelt", path, file_sr, sr)
    return x.astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ohr-Prior aus persönlichen A/B-Urteilen fitten")
    ap.add_argument("--urteile", required=True, help="Urteils-JSON (aurik.ohr_urteile/1)")
    ap.add_argument("--out", required=True, help="Ausgabe-Artefakt ohr_prior.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pseudo-count", type=float, default=10.0)
    args = ap.parse_args(argv)

    session = json.loads(Path(args.urteile).read_text(encoding="utf-8"))
    if session.get("schema") != "aurik.ohr_urteile/1":
        logger.error("Unbekanntes Sitzungs-Schema: %s", session.get("schema"))
        return 2
    sr = int(session.get("sr_hint", 48000))
    pairs = []
    for p in session.get("pairs", []):
        a = _load_audio(p["a"], sr)
        b = _load_audio(p["b"], sr)
        label = 1.0 if str(p["choice"]).upper() == "A" else 0.0
        pairs.append((a, b, label))

    if not pairs:
        logger.error("Keine Urteile in der Sitzung (§V6 (copilot-instructions.md))")
        return 2

    from backend.core.dsp.ohr_prior import fit_ohr_prior

    fit = fit_ohr_prior(pairs, sr, pseudo_count=args.pseudo_count, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(fit.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info(
        "Ohr-Prior geschrieben: %s (n=%d, log_loss=%.4f, getragen=%.2f)",
        out,
        fit.n_labels,
        fit.log_loss,
        fit.n_labels / (fit.n_labels + args.pseudo_count),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
