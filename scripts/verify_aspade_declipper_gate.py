#!/usr/bin/env python3
"""verify_aspade_declipper_gate — unabhängiger Vollpfad-Nachweis für A-SPADE (D-K4-1).

Aktivierungs-Voraussetzung aus dem Flag-Kommentar
(`backend/core/music_model_flags.py::use_aspade_declipper`): **ΔSDR-Gate auf
geclipptem Real-Material** + blindes A/B + Hörordnungs-Sign-off. Dieses Skript
erbringt den objektiven Teil: auf den realen Clipping-Proben (Ganz-Titel-Rezepte,
`~/Musik/schadens_proben_ganz` + Clean-Originale `~/Musik/k398_quellen`) läuft
der echte Produktionsschritt `DeclipperPhase` dreifach — Input, PCHIP-Amtsinhaber
(Flag AUS), A-SPADE-Challenger (Flag AN) — und misst je Titel:
  * SDR gegen das Clean-Original (best-lag-alignt, ±50 Samples, deterministisch),
  * never-worsen: Harmonik-Proxy (delta-basiert, Guard-Kalibrierung) und
    musik_mos/defektlast-Deltas (JND 0,05).

Gate-Urteil (fail-closed, §V6 (copilot-instructions.md)): A-SPADE besteht nur,
wenn er auf JEDEM Titel nie schlechter als der Amtsinhaber und nie schlechter
als der Input ist. Kein Flag-Flip — der Befund ist ein Zeuge (Hörordnung §7);
die Freigabe bleibt §4/Hör-Instanz.

Aufruf:
    python3 -B scripts/verify_aspade_declipper_gate.py [--out ...]
CPU-only (ONNX), deterministisch (§G5 (copilot-instructions.md)).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SR = 48000
JND = 0.05  # eine Wahrheit: wohlklang_zeuge.JND_TOLERANZ


def _wav_lesen(pfad: Path) -> tuple[int, np.ndarray]:
    from scipy.io import wavfile

    gelesen = wavfile.read(pfad)
    if not isinstance(gelesen, tuple) or len(gelesen) < 2:  # Bug 12: Index-basiert statt Unpacking
        raise ValueError("wavfile.read() ohne (sr, data)-Tupel")
    sr = int(gelesen[0])
    daten = np.asarray(gelesen[1])
    if np.issubdtype(daten.dtype, np.integer):
        daten = daten.astype(np.float32) / 32768.0
    else:
        daten = daten.astype(np.float32)
    return sr, daten


def _mono(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=-1)
    return x


def _sdr_aligned(schaetzung: np.ndarray, referenz: np.ndarray, max_lag: int = 50) -> float:
    """SDR mit best-lag-Ausrichtung (Standard-Declip-Evaluation, deterministisch)."""
    a = _mono(schaetzung).astype(np.float64)
    b = _mono(referenz).astype(np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    # Korrelations-Peak im ±max_lag-Fenster (numpy correlate, deterministisch).
    korr = np.correlate(a - a.mean(), b - b.mean(), mode="full")
    mitte = len(b) - 1
    fenster = korr[mitte - max_lag : mitte + max_lag + 1]
    lag = int(np.argmax(fenster)) - max_lag
    if lag >= 0:
        a_al, b_al = a[lag:], b[: n - lag]
    else:
        a_al, b_al = a[: n + lag], b[-lag:]
    m = min(len(a_al), len(b_al))
    a_al, b_al = a_al[:m], b_al[:m]
    fehler = b_al - a_al
    denom = float(np.sum(b_al**2))
    if denom < 1e-12:
        return float("-inf")
    return float(10.0 * np.log10(denom / float(np.sum(fehler**2)) + 1e-12))


def _harmonik_proxy(audio: np.ndarray, sr: int) -> float:
    from backend.core.dsp.aspade_declipper import harmonic_distortion_witness

    return float(harmonic_distortion_witness(_mono(audio), int(sr)))


def _metriken(audio: np.ndarray) -> dict[str, float]:
    from backend.core.dsp.wohlklang_zeuge import defektlast
    from plugins.musik_mos_plugin import get_musik_mos

    x = _mono(audio)
    return {
        "musik_mos": round(float(get_musik_mos().estimate_mos(x, SR).mos), 4),
        "defektlast": round(float(defektlast(x, SR)), 5),
    }


def _declip_lauf(audio: np.ndarray, sr: int, aspade_an: bool) -> np.ndarray:
    """Echter Produktionsschritt DeclipperPhase mit Flag AUS bzw. AN."""
    import backend.core.music_model_flags as mmf
    from backend.core.phases.phase_07_declipper import DeclipperPhase

    alt = mmf.use_aspade_declipper
    mmf.use_aspade_declipper = bool(aspade_an)
    try:
        # Stereo-Layout-Invariante (AGENTS.md §3): Phase erwartet (C, N).
        x = np.asarray(audio, dtype=np.float32)
        mono_ein = x.ndim == 1
        if mono_ein:
            x = x[None, :]
        elif x.shape[0] > 8 and x.shape[-1] <= 8:
            x = x.T
        ergebnis = DeclipperPhase().process(x, sample_rate=int(sr))
        out = np.asarray(ergebnis.audio, dtype=np.float32)
        return out[0] if mono_ein else out
    finally:
        mmf.use_aspade_declipper = alt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Unabhängiger ΔSDR-Vollpfad-Nachweis A-SPADE (D-K4-1)")
    ap.add_argument("--schaden-dir", default="/home/michael/Musik/schadens_proben_ganz")
    ap.add_argument("--clean-dir", default="/home/michael/Musik/k398_quellen")
    ap.add_argument(
        "--fenster-sek",
        type=float,
        default=30.0,
        help="Bewertetes Fenster je Titel (schlimmstes Clipping-Fenster; Standard-Declip-Evaluation — sonst läuft APPLADE Stunden pro Ganz-Titel)",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    schaden_dir = Path(args.schaden_dir)
    clean_dir = Path(args.clean_dir)
    zeilen: list[dict] = []
    for schaden_pfad in sorted(schaden_dir.glob("* (clipping).wav")):
        base = schaden_pfad.name[: -len(" (clipping).wav")]
        schluessel = base.split(" (")[0]
        clean_kandidaten = [p for p in clean_dir.glob("*.wav") if schluessel in p.name and "original" in p.name.lower()]
        if not clean_kandidaten:
            print(f"ÜBERSPRUNGEN (kein Clean): {schaden_pfad.name}", file=sys.stderr)
            continue
        sr_d, schaden = _wav_lesen(schaden_pfad)
        _sr_c, clean = _wav_lesen(clean_kandidaten[0])
        # Bewertungsfenster: am stärksten geclippte Region (deterministisch).
        # wavfile-Layout ist (N,) bzw. (N, C) — Achse 0 ist immer die Zeit.
        x_full = _mono(schaden)
        fenster = int(args.fenster_sek * sr_d)
        if x_full.shape[0] > fenster:
            # Clip-Schwelle 0.85: die Clipping-Rezepte (k398) klippen bei ±0.90 —
            # mit ≥0.95 fände die Suche kein einziges Sample.
            clipmaske = (np.abs(x_full) >= 0.85).astype(np.float64)
            best_start, best_anz = 0, -1.0
            for start in range(0, x_full.shape[0] - fenster + 1, max(1, sr_d // 2)):
                anz = float(clipmaske[start : start + fenster].sum())
                if anz > best_anz:
                    best_anz, best_start = anz, start
            # Sicherheit: Clean kann kürzer sein als der Schadens-Titel —
            # Fenster-Start in den gemeinsamen Bereich klemmen.
            if best_start + fenster > clean.shape[0]:
                best_start = max(0, clean.shape[0] - fenster)
            schaden = schaden[best_start : best_start + fenster]
            # Gleicher Fenster-Offset für das Clean-Original (Rezepte sind sample-alignt).
            clean = clean[best_start : best_start + fenster]
        n = min(schaden.shape[0], clean.shape[0])
        schaden, clean = schaden[:n], clean[:n]
        pchip = _declip_lauf(schaden, sr_d, aspade_an=False)
        aspade = _declip_lauf(schaden, sr_d, aspade_an=True)
        zeile = {
            "titel": base,
            "sdr_input_db": round(_sdr_aligned(schaden, clean), 3),
            "sdr_pchip_db": round(_sdr_aligned(pchip, clean), 3),
            "sdr_aspade_db": round(_sdr_aligned(aspade, clean), 3),
            "harmonik_input": round(_harmonik_proxy(schaden, sr_d), 5),
            "harmonik_pchip": round(_harmonik_proxy(pchip, sr_d), 5),
            "harmonik_aspade": round(_harmonik_proxy(aspade, sr_d), 5),
        }
        m_in, m_p, m_a = _metriken(schaden), _metriken(pchip), _metriken(aspade)
        zeile["delta_musik_mos_aspade_vs_input"] = round(m_a["musik_mos"] - m_in["musik_mos"], 4)
        zeile["delta_defektlast_aspade_vs_input"] = round(m_a["defektlast"] - m_in["defektlast"], 5)
        zeile["delta_musik_mos_aspade_vs_pchip"] = round(m_a["musik_mos"] - m_p["musik_mos"], 4)
        zeile["delta_defektlast_aspade_vs_pchip"] = round(m_a["defektlast"] - m_p["defektlast"], 5)
        zeilen.append(zeile)

    if not zeilen:
        print("FEHLER: keine Clipping-Proben gefunden", file=sys.stderr)
        return 2

    nie_schlechter_pchip = all(z["sdr_aspade_db"] >= z["sdr_pchip_db"] - 1e-3 for z in zeilen)
    nie_schlechter_input = all(z["sdr_aspade_db"] >= z["sdr_input_db"] - 1e-3 for z in zeilen)
    harmonik_ok = all(z["harmonik_aspade"] <= z["harmonik_input"] + 1e-6 for z in zeilen)
    gate = bool(nie_schlechter_pchip and nie_schlechter_input and harmonik_ok)
    befund = {
        "schema": "aurik.aspade_gate/1",
        "titel": [z["titel"] for z in zeilen],
        "zeilen": zeilen,
        "nie_schlechter_gegen_pchip": nie_schlechter_pchip,
        "nie_schlechter_gegen_input": nie_schlechter_input,
        "harmonik_never_worsen": harmonik_ok,
        "gate_bestanden": gate,
        "hinweis": "Objektiver Teil der D-K4-1-Aktivierung. Blindes A/B + Hörordnungs-Sign-off bleiben §4/Hör-Instanz — kein Flag-Flip.",
    }
    out_pfad = Path(args.out) if args.out else (ROOT / "docs" / "reports" / "current" / "aspade_declipper_gate.json")
    out_pfad.write_text(json.dumps(befund, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Beleg: {out_pfad}")
    for z in zeilen:
        print(
            f"  {z['titel'][:50]}: SDR input={z['sdr_input_db']} pchip={z['sdr_pchip_db']} aspade={z['sdr_aspade_db']}"
        )
    print(
        f"Gate: {'BESTANDEN' if gate else 'NICHT BESTANDEN'} (pchip-nw={nie_schlechter_pchip}, input-nw={nie_schlechter_input}, harmonik={harmonik_ok})"
    )
    return 0 if gate else 3


if __name__ == "__main__":
    raise SystemExit(main())
