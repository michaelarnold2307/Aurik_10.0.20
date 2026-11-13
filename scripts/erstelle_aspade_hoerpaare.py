#!/usr/bin/env python3
"""erstelle_aspade_hoerpaare — blinde A/B-Hörpaare für A-SPADE (D-K4-1-Hörnachweis).

D-K4-1-Aktivierungs-Kette: ΔSDR-Gate (scripts/verify_aspade_declipper_gate.py)
→ **blindes A/B** → Hörordnungs-Sign-off. Dieses Skript erzeugt die blinden
Paare: je Clipping-Titel das am stärksten geclippte 10-s-Fenster, einmal
PCHIP-declippt (Amtsinhaber) und einmal A-SPADE-declippt (Challenger),
Seiten per Seed verlost (Schlüssel getrennt in `schluessel.json`), 48 kHz
mono, deterministisch (§G5 (copilot-instructions.md)). Die HÖR-ENTSCHEIDUNG
selbst bleibt menschlich (scripts/hoer_entscheidung.py).

Kein Flag-Flip — Flag bleibt AUS bis zum vollständigen §4-Verfahren.

Aufruf:
    python3 -B scripts/erstelle_aspade_hoerpaare.py
        [--schaden-dir ~/Musik/schadens_proben_ganz]
        [--ausgang study_stimuli/hoerproben/aspade] [--seed 42]
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


def _wav_schreiben(pfad: Path, audio: np.ndarray, sr: int) -> None:
    from scipy.io import wavfile

    pfad.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(pfad, sr, (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16))


def _declip_lauf(audio: np.ndarray, sr: int, aspade_an: bool) -> np.ndarray:
    import backend.core.music_model_flags as mmf
    from backend.core.phases.phase_07_declipper import DeclipperPhase

    alt = mmf.use_aspade_declipper
    mmf.use_aspade_declipper = bool(aspade_an)
    try:
        ergebnis = DeclipperPhase().process(audio, sample_rate=int(sr))
        return np.asarray(ergebnis.audio, dtype=np.float32)
    finally:
        mmf.use_aspade_declipper = alt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Blinde A/B-Hörpaare für A-SPADE (D-K4-1)")
    ap.add_argument("--schaden-dir", default="/home/michael/Musik/schadens_proben_ganz")
    ap.add_argument("--ausgang", default=str(ROOT / "study_stimuli" / "hoerproben" / "aspade"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sekunden", type=float, default=10.0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    schaden_dir = Path(args.schaden_dir)
    out = Path(args.ausgang)
    schluessel: dict[str, str] = {}
    anzahl = 0
    for schaden_pfad in sorted(schaden_dir.glob("* (clipping).wav")):
        sr, audio = _wav_lesen(schaden_pfad)
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim == 2:
            x = x.mean(axis=-1)
        fenster = int(args.sekunden * sr)
        if len(x) < fenster:
            continue
        # Am stärksten geclipptes Fenster — Schwelle 0.85, die k398-Rezepte
        # klippen bei ±0.90 (deterministisch: Argmax der Sample-Anzahl).
        clipmaske = (np.abs(x) >= 0.85).astype(np.float64)
        best_start = 0
        best_anz = -1.0
        for start in range(0, len(x) - fenster + 1, sr // 2):
            anz = float(clipmaske[start : start + fenster].sum())
            if anz > best_anz:
                best_anz, best_start = anz, start
        segment = x[best_start : best_start + fenster].astype(np.float32)
        pchip = _declip_lauf(segment, sr, aspade_an=False)
        aspade = _declip_lauf(segment, sr, aspade_an=True)
        a_ist_aspade = bool(rng.random() < 0.5)
        slug = "".join(c if c.isalnum() else "_" for c in schaden_pfad.stem)[:60]
        _wav_schreiben(out / f"{slug}_A.wav", aspade if a_ist_aspade else pchip, sr)
        _wav_schreiben(out / f"{slug}_B.wav", pchip if a_ist_aspade else aspade, sr)
        schluessel[slug] = "A=aspade" if a_ist_aspade else "A=pchip"
        anzahl += 1

    if anzahl == 0:
        print("FEHLER: keine Clipping-Proben gefunden", file=sys.stderr)
        return 2
    manifest = {
        "schema": "aurik.hoerpaare/1",
        "zweck": "D-K4-1-Hörnachweis A-SPADE vs. PCHIP (blindes A/B)",
        "seed": args.seed,
        "paare": anzahl,
        "hinweis": "Hör-Entscheidung menschlich (scripts/hoer_entscheidung.py) — kein Flag-Flip.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "schluessel.json").write_text(json.dumps(schluessel, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Hörpaare: {anzahl} nach {out} (Schlüssel getrennt in schluessel.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
