#!/usr/bin/env python3
"""erstelle_aspade_hoerpaare — blinde A/B-Hörpaare für A-SPADE (D-K4-1-Hörnachweis).

D-K4-1-Aktivierungs-Kette: ΔSDR-Gate (scripts/verify_aspade_declipper_gate.py)
→ **blindes A/B** → Hörordnungs-Sign-off. Dieses Skript erzeugt die blinden
Paare: je Clipping-Titel das am stärksten geclippte 10-s-Fenster, einmal
PCHIP-declippt (Amtsinhaber) und einmal A-SPADE-declippt (Challenger),
Seiten per Seed verlost (Schlüssel getrennt in `schluessel.json`), 48 kHz
mono, deterministisch (§G5 (copilot-instructions.md)). Die HÖR-ENTSCHEIDUNG
selbst bleibt menschlich (scripts/hoer_entscheidung.py).

`--produktionspfad`: A/B über den ECHTEN UV3-Produktionspfad (Fahrplan
A-1, `docs/reports/current/2026-11-13_aktivierungs_fahrplan.md`) —
Flag-AUS vs. Flag-AN (UnifiedRestorerV3), gleiches 10-s-Fenster,
Seed-Seiten, Schlüssel getrennt. Dieser Modus misst die reale Flag-Wirkung
inkl. Musik-Declip-Kopf-Challenger (Schritt A-1, parallel, kein Training).

Kein Flag-Flip — Flag bleibt AUS bis zum vollständigen §4-Verfahren.

Aufruf:
    python3 -B scripts/erstelle_aspade_hoerpaare.py
        [--schaden-dir ~/Musik/schadens_proben_ganz]
        [--ausgang study_stimuli/hoerproben/aspade] [--seed 42]
        [--produktionspfad]
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


def _ausgabe_pfad(out: Path, dateiname: str) -> Path:
    """Resolve generated paths and reject destinations outside the output directory."""
    basis = out.resolve()
    ziel = (basis / dateiname).resolve()
    try:
        ziel.relative_to(basis)
    except ValueError as exc:
        raise ValueError(f"Ausgabepfad liegt außerhalb des Ausgabe-Verzeichnisses: {dateiname}") from exc
    return ziel


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


def _fenster_schlimmstes(x: np.ndarray, fenster: int, schwelle: float, schritt: int) -> np.ndarray:
    """Am stärksten geclippte Region — deterministisches Argmax der Sample-Anzahl (§G5 (copilot-instructions.md))."""
    clipmaske = (np.abs(x) >= schwelle).astype(np.float64)
    best_start = 0
    best_anz = -1.0
    for start in range(0, len(x) - fenster + 1, schritt):
        anz = float(clipmaske[start : start + fenster].sum())
        if anz > best_anz:
            best_anz, best_start = anz, start
    return x[best_start : best_start + fenster].astype(np.float32)


def _p25(x: np.ndarray) -> float:
    """Body-Perzentil-Pegelreferenz (p25, kanonisches Pegelmatching der Hörordnung §7)."""
    return float(np.percentile(np.abs(np.asarray(x, dtype=np.float32)), 25))


def _pegel_gematcht(ziel: np.ndarray, quelle: np.ndarray) -> np.ndarray:
    """Skaliert quelle auf den p25-Pegel von ziel — verhindert Lautheits-Bias im blinden A/B."""
    p_z, p_q = _p25(ziel), _p25(quelle)
    if p_q <= 1e-9:
        return np.asarray(quelle, dtype=np.float32)
    return (np.asarray(quelle, dtype=np.float32) * (p_z / p_q)).astype(np.float32)


def _produktionspfad_lauf(audio: np.ndarray, sr: int, flag_an: bool) -> np.ndarray:
    """Echter UV3-Produktionspfad (Fahrplan A-1) mit Flag-AUS/AN — kein Declipper-Direktaufruf."""
    import backend.core.music_model_flags as mmf
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    alt = mmf.use_aspade_declipper
    mmf.use_aspade_declipper = bool(flag_an)
    try:
        return np.asarray(UnifiedRestorerV3().restore(audio, sample_rate=int(sr)).audio, dtype=np.float32)
    finally:
        mmf.use_aspade_declipper = alt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Blinde A/B-Hörpaare für A-SPADE (D-K4-1)")
    ap.add_argument("--schaden-dir", default="/home/michael/Musik/schadens_proben_ganz")
    ap.add_argument("--ausgang", default=str(ROOT / "study_stimuli" / "hoerproben" / "aspade"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sekunden", type=float, default=10.0)
    ap.add_argument(
        "--produktionspfad",
        action="store_true",
        help="A/B über den echten UV3-Produktionspfad (Flag-AUS vs. Flag-AN, Fahrplan A-1)",
    )
    args = ap.parse_args(argv)

    if args.produktionspfad and "aspade" in str(args.ausgang):
        print(
            "HINWEIS: --produktionspfad in den aspade-Ausgang schreiben? "
            "Proben sind dann keine PCHIP-vs.-A-SPADE-Paare mehr.",
            file=sys.stderr,
        )
        return 2

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
        segment = _fenster_schlimmstes(x, fenster, 0.85, sr // 2)
        if args.produktionspfad:
            aus = _produktionspfad_lauf(segment, sr, flag_an=False)
            an = _produktionspfad_lauf(segment, sr, flag_an=True)
            # Pegelmatching (p25) gegen das Eingangs-Fenster: beide Varianten
            # gleich laut, sonst entscheidet Lautheit statt Qualität (Hörordnung §7).
            aus = _pegel_gematcht(segment, aus)
            an = _pegel_gematcht(segment, an)
            a_ist_an = bool(rng.random() < 0.5)
            slug = "".join(c if c.isalnum() else "_" for c in schaden_pfad.stem)[:60]
            _wav_schreiben(_ausgabe_pfad(out, f"{slug}_A.wav"), an if a_ist_an else aus, sr)
            _wav_schreiben(_ausgabe_pfad(out, f"{slug}_B.wav"), aus if a_ist_an else an, sr)
            schluessel[slug] = "A=flag_an" if a_ist_an else "A=flag_aus"
        else:
            pchip = _declip_lauf(segment, sr, aspade_an=False)
            aspade = _declip_lauf(segment, sr, aspade_an=True)
            a_ist_aspade = bool(rng.random() < 0.5)
            slug = "".join(c if c.isalnum() else "_" for c in schaden_pfad.stem)[:60]
            _wav_schreiben(_ausgabe_pfad(out, f"{slug}_A.wav"), aspade if a_ist_aspade else pchip, sr)
            _wav_schreiben(_ausgabe_pfad(out, f"{slug}_B.wav"), pchip if a_ist_aspade else aspade, sr)
            schluessel[slug] = "A=aspade" if a_ist_aspade else "A=pchip"
        anzahl += 1

    if anzahl == 0:
        print("FEHLER: keine Clipping-Proben gefunden", file=sys.stderr)
        return 2
    manifest = {
        "schema": "aurik.hoerpaare/1",
        "zweck": (
            "D-K4-1-Hörnachweis A-SPADE vs. PCHIP (blindes A/B)"
            if not args.produktionspfad
            else "Fahrplan-A-1-Hörpaket: UV3-Produktionspfad Flag-AUS vs. Flag-AN (blindes A/B)"
        ),
        "seed": args.seed,
        "paare": anzahl,
        "hinweis": "Hör-Entscheidung menschlich (scripts/hoer_entscheidung.py) — kein Flag-Flip.",
    }
    _ausgabe_pfad(out, "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _ausgabe_pfad(out, "schluessel.json").write_text(
        json.dumps(schluessel, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Hörpaare: {anzahl} nach {out} (Schlüssel getrennt in schluessel.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
