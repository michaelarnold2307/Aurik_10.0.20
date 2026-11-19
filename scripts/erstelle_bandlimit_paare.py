#!/usr/bin/env python3
"""erstelle_bandlimit_paare — Rezept-Paare für die Klasse bandwidth_loss.

§G13-Q-Klassenkopf-Kette (Fahrplan `2026-11-13_aktivierungs_fahrplan.md`):
Die Metrik-Sichtbarkeits-Messung (2026-11-14) zeigte — anders als bei
groove_echo (0/24 Regionen in richtiger Richtung) — dass die Gate-Metrik
`defektlast_lang` einen 8-kHz-Bandlimit-Schaden am stärksten von allen
k398-Rezept-Familien sieht: 63 % der Regionen > JND 0,05, median Δ +0,17.
Die Schadens-Abbildung ist ein linearer Tiefpass (dieselbe Produktions-
Variante `bandlimit` aus `scripts/ohr_panel.defekt_variante`, die auch die
kassette-Rezeptur nutzt) — die am einfachsten lernbare Abbildung der
Q≈0-Klassen.

Deterministisch (§G5 (copilot-instructions.md)): MUSDB-Clean-Mixtures als Basis,
8192er-Fenster (Trainer-WINDOW), Seed 42, sha256_pcm. Kontrast-Filter
(Fahrplan B-0(1)): nur Regionen mit defektlast(schaden) − defektlast(clean)
> JND werden ins Manifest übernommen. Kein Flag-Flip.

Aufruf:
    python3 -B scripts/erstelle_bandlimit_paare.py \
        [--clean-dir data/musdb_mixtures] \
        [--out data/klassen_paare_bandlimit] [--fenster-pro-titel 12]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KLASSE = "bandwidth_loss"
WINDOW = 8192
JND = 0.05


def _sha256_pcm(audio: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()


def _wav_lesen(pfad: Path) -> tuple[int, np.ndarray]:
    from scipy.io import wavfile

    gelesen = wavfile.read(pfad)
    if not isinstance(gelesen, tuple) or len(gelesen) < 2:
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


def _pfad_relativ(pfad: Path) -> str:
    """Repo-relativ wenn möglich, sonst absolut (tmp_path-Tests liegen außerhalb)."""
    try:
        return str(pfad.relative_to(ROOT))
    except ValueError:
        return str(pfad)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rezept-Paare bandwidth_loss (Fahrplan B-0(1))")
    ap.add_argument("--clean-dir", default="data/musdb_mixtures")
    ap.add_argument("--out", default="data/klassen_paare_bandlimit")
    ap.add_argument("--fenster-pro-titel", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    from backend.core.dsp.defekt_delta import defektlast_lang
    from scripts.ohr_panel import defekt_variante

    clean_dir = ROOT / args.clean_dir
    titel = sorted(clean_dir.glob("*.wav"))
    if not titel:
        print(f"FEHLER: keine Clean-Titel in {clean_dir}", file=sys.stderr)
        return 2
    out = ROOT / args.out
    (out / KLASSE).mkdir(parents=True, exist_ok=True)
    manifest_pfad = out / "manifest.jsonl"

    rng = np.random.default_rng(args.seed)
    zeilen: list[str] = []
    gesamt = 0
    uebernommen = 0
    for p in titel:
        sr, d = _wav_lesen(p)
        x = np.asarray(d, dtype=np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        x = x / (np.abs(x).max() + 1e-12)
        n_fenster = max(1, len(x) // WINDOW)
        # Deterministische Fensterauswahl: gleichmäßig verteilt, Seed-stabil.
        picks = np.linspace(0, n_fenster - 1, min(args.fenster_pro_titel, n_fenster)).astype(int)
        slug_titel = "".join(c if c.isalnum() else "_" for c in p.stem)
        for idx in picks:
            s0 = int(idx) * WINDOW
            if s0 + WINDOW > len(x):
                continue
            clean = np.ascontiguousarray(x[s0 : s0 + WINDOW], dtype=np.float32)
            # Produktions-Variante bandlimit (kassette-Rezeptur-Anteil), deterministisch.
            schaden = np.asarray(defekt_variante(clean, "bandlimit", rng), dtype=np.float32)
            lc = float(defektlast_lang(clean, sr))
            ls = float(defektlast_lang(schaden, sr))
            gesamt += 1
            if not (ls - lc > JND):
                continue  # Kontrast-Filter (Fahrplan B-0(1))
            slug = f"{slug_titel}_{s0}"
            p_s = out / KLASSE / f"{slug}_schaden.wav"
            p_c = out / KLASSE / f"{slug}_clean.wav"
            _wav_schreiben(p_s, schaden, sr)
            _wav_schreiben(p_c, clean, sr)
            zeilen.append(
                json.dumps(
                    {
                        "klasse": KLASSE,
                        "titel": p.stem,
                        "rezept": "bandlimit",
                        "start": s0,
                        "end": s0 + WINDOW,
                        "schaden": _pfad_relativ(p_s),
                        "clean": _pfad_relativ(p_c),
                        "sha256_pcm_schaden": _sha256_pcm(schaden),
                        "sha256_pcm_clean": _sha256_pcm(clean),
                    },
                    ensure_ascii=False,
                )
            )
            uebernommen += 1

    if not zeilen:
        print("FEHLER: keine Regionen bestanden den Kontrast-Filter", file=sys.stderr)
        return 2
    manifest_pfad.write_text("\n".join(zeilen) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_pfad} ({uebernommen}/{gesamt} Regionen nach Kontrast-Filter)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
