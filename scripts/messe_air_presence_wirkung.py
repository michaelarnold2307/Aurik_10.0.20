#!/usr/bin/env python3
"""messe_air_presence_wirkung — D-K3-98-Wirkungsmessung auf D-K3-95-Rezepten.

Offener Punkt aus `docs/reports/current/2026-11-11_d_k3_98_air_presence_nola.md`
§4: Der NOLA-/Hard-Clip-Fix ist synthetisch verifiziert (440-Hz-/12-kHz-Sinus),
aber die WIRKUNG auf die Schadens-Rezepte clipping/kassette (D-K3-95) war
unbewiesen. Dieses Werkzeug schließt die Lücke — gleiche Proben, zwei Ebenen:

1. **Enhancer-Kern** (default, CPU-leicht): `enhance_air_presence` direkt auf
   den degradierten Proben + Listening-Witness-Gate + Eingangs-Decken-Vertrag.
2. **Stem-Pfad** (`--stem-pfad`, Produktions-Trigger): `StemLevelRestorer.restore`
   mit festem `panns_singing` — zeichnet `air_presence_used` + Zeuge der echten
   Produktionsstufe (§SLR-1e2b) auf.

Never-worsen-Vertrag (D-K3-98): `hnr_drop < 1.0 dB`, `pitch_drift < 8 ct`,
`flat_top_rise < 0.02`; Eingangs-Decke: `max|out| <= min(0,999, max|in|)`
(k=0-Passthrough, wenn der Eingang selbst über der Decke liegt).
Richtungsmetriken musik_mos/defektlast als Delta je Ebene (JND 0,05).

Deterministisch (§G5 (copilot-instructions.md)): kein `time.time()`, kein Zufall.
CPU-only, stört eine laufende GPU-Kette nicht. Kein Flag-Flip, kein Deploy.
Beleg-Schema `aurik.air_presence_wirkung/1`.

Aufruf:
    nice -n 19 python3 -B scripts/messe_air_presence_wirkung.py [--verzeichnis DIR] \
        [--rezepte clipping,kassette] [--stem-pfad] [--panns 0.8]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SR = 48000
JND_TOLERANZ = 0.05  # identisch zu backend/core/dsp/wohlklang_zeuge.JND_TOLERANZ


def _lade_proben(verzeichnis: Path, rezepte: tuple[str, ...]) -> list[dict]:
    """Degradierte Proben je Rezept (KEINE `_restauriert`-Gegenstücke)."""
    from scipy.io import wavfile

    proben: list[dict] = []
    for wav in sorted(verzeichnis.glob("*.wav")):
        name = wav.name
        if "_restauriert" in name:
            continue
        rezept = next((r for r in rezepte if f"({r})" in name), None)
        if rezept is None:
            continue
        _wf = wavfile.read(wav)
        if not isinstance(_wf, tuple) or len(_wf) < 2:  # Bug 12: Index-basiert statt Unpacking
            raise ValueError("wavfile.read() ohne (sr, data)-Tupel")
        sr = int(_wf[0])
        daten = np.asarray(_wf[1])
        if sr != SR:
            raise ValueError(f"{wav}: {sr} Hz erwartet {SR}")
        if np.issubdtype(daten.dtype, np.integer):
            daten = daten.astype(np.float32) / 32768.0
        else:
            daten = daten.astype(np.float32)
        proben.append({"titel": wav.stem, "rezept": rezept, "audio": daten, "pfad": str(wav)})
    return proben


def _metriken(audio: np.ndarray) -> dict[str, float]:
    """musik_mos + defektlast (mono-Downmix, 48 kHz) — richtungszertifiziert."""
    from backend.core.dsp.wohlklang_zeuge import defektlast
    from plugins.musik_mos_plugin import get_musik_mos

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=-1)
    return {
        "musik_mos": round(float(get_musik_mos().estimate_mos(x, SR).mos), 4),
        "defektlast": round(float(defektlast(x, SR)), 5),
    }


def _enhancer_kern(audio: np.ndarray) -> tuple[dict, np.ndarray]:
    """D-K3-98-Kern direkt: enhance_air_presence + Witness-Gate + Decken-Vertrag.

    `audio` channels-last (N,) oder (N, 2); Rückgabe (befund, ausgabe
    channels-last wie Produktion). `applied` spiegelt die Produktions-Semantik
    (Witness-Gate bestanden); `rms_delta` beziffert die tatsächliche Wirkung.
    """
    from backend.core.dsp.air_presence_enhancer import enhance_air_presence
    from backend.core.listening_witness import evaluate_listening_witness

    x = np.asarray(audio, dtype=np.float32)
    x_cf = x.T if x.ndim == 2 else x  # Enhancer erwartet channels-first
    enhanced = enhance_air_presence(x_cf, sr=SR, strength=0.15)
    out_cf = np.asarray(enhanced, dtype=np.float32)
    out_cf = np.nan_to_num(out_cf, nan=0.0, posinf=0.0, neginf=0.0)
    out_cf = np.clip(out_cf, -1.0, 1.0)
    out_cl = out_cf.T if x.ndim == 2 else out_cf  # Produktions-Layout channels-last

    zeuge = evaluate_listening_witness(x, out_cl, SR, "air_presence")
    ok = zeuge.hnr_drop_db < 1.0 and zeuge.pitch_drift_cents < 8.0 and zeuge.flat_top_rise < 0.02

    decke_in = float(min(0.999, np.abs(x).max())) if x.size else 0.0
    decke_out = float(np.abs(out_cl).max()) if out_cl.size else 0.0
    decke_ok = decke_out <= decke_in + 1e-7

    rms_delta = float(np.sqrt(np.mean((out_cl.ravel() - x.ravel()) ** 2))) if x.size else 0.0
    return {
        "applied": bool(ok),
        "witness_ok": bool(ok),
        "hnr_drop_db": round(float(zeuge.hnr_drop_db), 4),
        "pitch_drift_cents": round(float(zeuge.pitch_drift_cents), 4),
        "flat_top_rise": round(float(zeuge.flat_top_rise), 5),
        "decke_in": round(decke_in, 6),
        "decke_out": round(decke_out, 6),
        "decke_ok": bool(decke_ok),
        "rms_delta": round(rms_delta, 7),
    }, out_cl


def _stem_pfad(audio: np.ndarray, panns: float) -> dict:
    """Produktions-Trigger: StemLevelRestorer (air_presence_used + Zeuge)."""
    from backend.core.dsp.stem_level_restorer import get_stem_level_restorer

    slr = get_stem_level_restorer()
    ergebnis = slr.restore(audio=audio, sample_rate=SR, panns_singing=panns, restoration_context={})
    if ergebnis is None:
        return {"gelaufen": False, "grund": "zu kurz / uebersprungen"}
    zeuge = ergebnis.air_presence_witness or {}
    return {
        "gelaufen": True,
        "success": bool(ergebnis.success),
        "air_presence_used": bool(ergebnis.air_presence_used),
        "applied": bool(zeuge.get("applied", False)),
        "witness": zeuge,
    }


def _delta_metriken(vorher: np.ndarray, nachher: np.ndarray) -> dict:
    m_in = _metriken(vorher)
    m_out = _metriken(nachher)
    return {
        "musik_mos_delta": round(m_out["musik_mos"] - m_in["musik_mos"], 4),
        "defektlast_delta": round(m_out["defektlast"] - m_in["defektlast"], 5),
        "nie_schlechter_als_jnd": bool(
            (m_out["musik_mos"] - m_in["musik_mos"]) >= -JND_TOLERANZ
            and (m_out["defektlast"] - m_in["defektlast"]) <= JND_TOLERANZ
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D-K3-98-Wirkungsmessung auf D-K3-95-Rezepten")
    ap.add_argument("--verzeichnis", default="/home/michael/Musik/schadens_proben")
    ap.add_argument("--rezepte", default="clipping,kassette")
    ap.add_argument("--stem-pfad", action="store_true", help="zusätzlich Produktions-Stem-Pfad messen")
    ap.add_argument("--panns", type=float, default=0.8, help="panns_singing für den Stem-Pfad")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    rezepte = tuple(r.strip() for r in args.rezepte.split(",") if r.strip())
    proben = _lade_proben(Path(args.verzeichnis), rezepte)
    if not proben:
        print(f"keine Proben für Rezepte {rezepte} in {args.verzeichnis}", file=sys.stderr)
        return 2

    zeilen: list[dict] = []
    for probe in proben:
        kern, out_cl = _enhancer_kern(probe["audio"])
        kern["delta_metriken"] = _delta_metriken(probe["audio"], out_cl)
        zeile: dict = {
            "titel": probe["titel"],
            "rezept": probe["rezept"],
            "ebene_kern": kern,
        }
        if args.stem_pfad:
            zeile["ebene_stem"] = _stem_pfad(probe["audio"], args.panns)
        zeilen.append(zeile)

    befund = {
        "schema": "aurik.air_presence_wirkung/1",
        "datum": date.today().isoformat(),
        "quelle": "scripts/messe_air_presence_wirkung.py",
        "proben_anzahl": len(zeilen),
        "rezepte": list(rezepte),
        "panns_singing": args.panns if args.stem_pfad else None,
        "zeilen": zeilen,
        "zusammenfassung": {
            "kern_engagiert": sum(1 for z in zeilen if z["ebene_kern"]["applied"]),
            "kern_witness_ok": sum(1 for z in zeilen if z["ebene_kern"]["witness_ok"]),
            "kern_decke_ok": sum(1 for z in zeilen if z["ebene_kern"]["decke_ok"]),
            "kern_nie_schlechter": sum(
                1 for z in zeilen if z["ebene_kern"]["delta_metriken"]["nie_schlechter_als_jnd"]
            ),
        },
    }
    if args.stem_pfad:
        gelaufen = [z for z in zeilen if z.get("ebene_stem", {}).get("gelaufen")]
        befund["zusammenfassung"]["stem_gelaufen"] = len(gelaufen)
        befund["zusammenfassung"]["stem_air_used"] = sum(
            1 for z in gelaufen if z["ebene_stem"].get("air_presence_used")
        )

    ziel = (
        Path(args.out)
        if args.out
        else (ROOT / "docs" / "reports" / "current" / f"{date.today().isoformat()}_air_presence_wirkung.json")
    )
    ziel.write_text(json.dumps(befund, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Beleg: {ziel}")
    print(json.dumps(befund["zusammenfassung"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
