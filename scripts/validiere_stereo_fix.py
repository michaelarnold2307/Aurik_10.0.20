"""scripts/validiere_stereo_fix.py — Stereo-Balance-End-to-End-Validierung.

Hintergrund (2026-11-19): Der Detektor erkennt STEREO_IMBALANCE (sev 1,0/conf
0,99), aber der Produktionslauf veränderte das Snowmine-Paar NICHT — Ursache
war die Causal-Reasoner-Planung (die Cause-Posterior lag außerhalb der Top-3,
phase_15 wurde nie geplant; Fix: Harte-Evidenz-Rettung im Reasoner).

Dieses Skript misst den End-to-End-Beweis: ein Stereo-Paar durch den
Produktionspfad (`restauriere_schnipsel`, CPU-only) → Balance vor/nach + MOS/
Defektlast-Witness. Beleg: `docs/reports/current/<datum>_snowmine_stereo_fix_v2.json`,
Schema `aurik.snowmine_stereo_fix/1`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SCHEMA = "aurik.snowmine_stereo_fix/1"
REPORT_DIR = _ROOT / "docs" / "reports" / "current"


def _balance_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        return 0.0
    l = float(np.sqrt(np.mean(x[:, 0] ** 2)))
    r = float(np.sqrt(np.mean(x[:, 1] ** 2)))
    return float(20.0 * np.log10((l + 1e-12) / (r + 1e-12)))


def main(argv: list[str] | None = None) -> int:
    import faulthandler

    fh = open(_ROOT / "logs" / "python_faulthandler_ketten.log", "a", encoding="utf-8")
    faulthandler.enable(file=fh, all_threads=True)

    ap = argparse.ArgumentParser(description="Stereo-Balance-End-to-End-Validierung")
    ap.add_argument("--paar", type=Path, required=True, help="Stereo-Schadenspaar (.wav)")
    ap.add_argument("--ausgang", type=Path, default=None, help="Optional: restauriertes WAV hier ablegen")
    args = ap.parse_args(argv)

    os.environ.setdefault("AURIK_FORCE_CPU", "1")
    os.environ.setdefault("AURIK_NO_RT_LIMIT", "1")

    from scipy.io import wavfile

    from scripts.erstelle_schadens_proben import restauriere_schnipsel  # pylint: disable=import-outside-toplevel

    roh = wavfile.read(args.paar)
    sr = int(roh[0])
    d = roh[1]
    x = np.asarray(d, np.float32) / 32768.0 if np.asarray(d).dtype != np.float32 else np.asarray(d, np.float32)
    schaden = x.mean(axis=1) if x.ndim != 2 or x.shape[1] != 2 else x
    vorher = _balance_db(schaden)

    t0 = time.time()
    restauriert = restauriere_schnipsel(np.asarray(schaden, dtype=np.float32), sr, str(args.paar))
    dauer = time.time() - t0

    nachher = _balance_db(np.asarray(restauriert, dtype=np.float32))

    if args.ausgang is not None:
        import soundfile as sf

        sf.write(str(args.ausgang), np.asarray(restauriert, dtype=np.float32), sr)

    # MOS/Defektlast-Witness (mono-gemessen wie die übrigen Bewertungen)
    from backend.core.dsp.wohlklang_zeuge import messe_wohlklang_zeuge

    mono_s = schaden.mean(axis=1) if schaden.ndim == 2 else schaden
    mono_r = (
        np.asarray(restauriert, dtype=np.float32).mean(axis=1)
        if np.asarray(restauriert, dtype=np.float32).ndim == 2
        else np.asarray(restauriert, dtype=np.float32)
    )
    zeuge = messe_wohlklang_zeuge(mono_s, mono_r, sr)
    delta_mos = float(zeuge["delta"]["musik_mos"])
    delta_defekt = float(zeuge["delta"]["defektlast"])

    beleg = {
        "schema": SCHEMA,
        "datum": date.today().isoformat(),
        "titel": args.paar.stem,
        "rezept": "stereo_imbalance",
        "dauer_restauration_s": round(dauer, 1),
        "stereo_balance": {
            "vorher_db": round(vorher, 4),
            "nachher_db": round(nachher, 4),
            "delta_db": round(nachher - vorher, 4),
            "besser": abs(nachher) < abs(vorher),
        },
        "delta_musik_mos": round(delta_mos, 4),
        "delta_defektlast": round(delta_defekt, 4),
    }
    ziel = REPORT_DIR / f"{date.today().isoformat()}_snowmine_stereo_fix_v2.json"
    ziel.write_text(json.dumps(beleg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Balance {vorher:+.2f} -> {nachher:+.2f} dB | ΔMOS {delta_mos:+.3f} | Beleg: {ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
