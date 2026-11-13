#!/usr/bin/env python3
"""erstelle_klassen_paare — Per-Klassen-Trainingsdaten aus den Schadens-Rezepten.

Trainingsstrategie Stufe 3 (Report
`docs/reports/current/2026-11-13_hoerzentrierte_trainingsstrategie.md`):
Die deterministischen Schadens-Rezepte (`erstelle_schadens_proben.py`) plus
Clean-Originale sind bereits überwachte Reparatur-Paare. Dieses Skript
extrahiert je Defektklasse (DefectScanner-Typ) die Regionen als
schaden/restauriert/clean-Tripel — Grundlage für kleine Per-Klassen-Köpfe
(Priorität = die Q≈0-Klassen der §G13-Erstbelege: compression_artifacts,
dynamic_compression_excess, groove_echo, hum, jitter_artifacts, sibilance,
sticky_shed_residue, tape_head_clog, tape_head_level_dip, transport_bump).

Deterministisch (§G5 (copilot-instructions.md)), CPU-only, sha256_pcm. Kein Flag-Flip.

Aufruf:
    python3 -B scripts/erstelle_klassen_paare.py \
        --schaden-dir ~/Musik/schadens_proben_ganz --clean-dir ~/Musik/k398_quellen \
        --out data/klassen_paare
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


def _sha256_pcm(audio: np.ndarray) -> str:
    """Deterministischer PCM-Hash (Repo-Standard, D-K3-88-Lehre: kein WAV-Hash)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(audio, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def _pfad_relativ(pfad: Path) -> str:
    """Repo-relativ, wenn möglich — sonst absolut (externe Daten-/tmp-Dirs)."""
    aufgeloest = pfad.resolve()
    try:
        return str(aufgeloest.relative_to(ROOT))
    except ValueError:
        return str(aufgeloest)


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-Klassen-Trainingspaare aus den Schadens-Rezepten")
    ap.add_argument("--schaden-dir", default="/home/michael/Musik/schadens_proben_ganz")
    ap.add_argument("--clean-dir", default="/home/michael/Musik/k398_quellen")
    ap.add_argument("--out", default=str(ROOT / "data" / "klassen_paare"))
    ap.add_argument("--rezepte", default="clipping,kassette")
    args = ap.parse_args(argv)

    from backend.core.defect_scanner import DefectScanner
    from backend.core.dsp.defekt_delta import regionen_aus_defekt_scores

    schaden_dir = Path(args.schaden_dir)
    clean_dir = Path(args.clean_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rezepte = tuple(r.strip() for r in args.rezepte.split(",") if r.strip())
    manifest_pfad = out / "manifest.jsonl"
    zeilen: list[str] = []
    klassen_zaehler: dict[str, int] = {}

    for schaden_pfad in sorted(schaden_dir.glob("*.wav")):
        name = schaden_pfad.name
        if "_restauriert" in name:
            continue
        rezept = next((r for r in rezepte if f"({r})" in name), None)
        if rezept is None:
            continue
        base = name[: -len(f" ({rezept}).wav")]
        rest_pfad = schaden_dir / f"{base} ({rezept})_restauriert.wav"
        if not rest_pfad.exists():
            print(f"ÜBERSPRUNGEN (kein _restauriert): {name}", file=sys.stderr)
            continue
        schluessel = base.split(" (")[0] if " (" in base else base
        clean_kandidaten = [p for p in clean_dir.glob("*.wav") if schluessel in p.name and "original" in p.name.lower()]
        clean_pfad = clean_kandidaten[0] if clean_kandidaten else None

        sr_d, audio_schaden = _wav_lesen(schaden_pfad)
        sr_r, audio_rest = _wav_lesen(rest_pfad)
        audio_clean = None
        if clean_pfad is not None:
            sr_c, audio_clean_roh = _wav_lesen(clean_pfad)
            n = min(audio_schaden.shape[0], audio_clean_roh.shape[0])
            # Offset-0-Annahme der Rezept-Erzeugung (volle Titel ab Start); Längenangleich fail-safe.
            audio_clean = audio_clean_roh[:n]
            audio_schaden = audio_schaden[:n]
            audio_rest = audio_rest[:n]

        scan = DefectScanner().scan(audio_schaden, sample_rate=int(sr_d))
        regionen = regionen_aus_defekt_scores(scan.scores, int(sr_d))
        slug_titel = "".join(c if c.isalnum() else "_" for c in base)

        for region in regionen:
            start, ende = int(region["start"]), int(region["end"])
            start = max(0, min(start, audio_schaden.shape[0] - 1))
            ende = max(start + 1, min(ende, audio_schaden.shape[0]))
            klasse = str(region.get("typ", "unbekannt"))
            klassen_zaehler[klasse] = klassen_zaehler.get(klasse, 0) + 1
            slug = f"{slug_titel}_{start}_{ende}"
            kdir = out / klasse
            p_schaden = kdir / f"{slug}_schaden.wav"
            p_rest = kdir / f"{slug}_restauriert.wav"
            p_clean = kdir / f"{slug}_clean.wav"
            _wav_schreiben(p_schaden, audio_schaden[start:ende], int(sr_d))
            _wav_schreiben(p_rest, audio_rest[start:ende], int(sr_r))
            eintrag: dict = {
                "klasse": klasse,
                "titel": base,
                "rezept": rezept,
                "start": start,
                "end": ende,
                "schaden": _pfad_relativ(p_schaden),
                "restauriert": _pfad_relativ(p_rest),
                "clean": None,
                "sha256_pcm_schaden": _sha256_pcm(audio_schaden[start:ende]),
            }
            if audio_clean is not None:
                _wav_schreiben(p_clean, audio_clean[start:ende], int(sr_d))
                eintrag["clean"] = _pfad_relativ(p_clean)
                eintrag["sha256_pcm_clean"] = _sha256_pcm(audio_clean[start:ende])
            zeilen.append(json.dumps(eintrag, ensure_ascii=False))

    manifest_pfad.write_text("\n".join(zeilen) + ("\n" if zeilen else ""), encoding="utf-8")
    print(f"Manifest: {manifest_pfad} ({len(zeilen)} Regionen)")
    print("Klassen-Bilanz (sortiert):")
    for klasse, anzahl in sorted(klassen_zaehler.items(), key=lambda kv: -kv[1]):
        print(f"  {klasse}: {anzahl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
