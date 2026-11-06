#!/usr/bin/env python3
"""D-K4-8 (SOTA-M12): Echt-Trainingsdaten für TP-V2-Phasenschätzung aus MUSDB18-HQ.

Sammelt **transiente-reiche** Stem-Dateien aus MUSDB18-HQ (``drums.wav`` sowie
``other.wav``/``bass.wav`` als Percussion-artige Komponenten) und schreibt sie als
mono/segmentierte ``*.wav``-Segmente in ein Ausgabeverzeichnis, das
``scripts/train_tp_v2_phase.py --real-dir`` direkt konsumiert.

MUSTER: ``scripts/prepare_tfgrid_pairs.py`` (deterministisch Seed 42 §G5 (GEBOTE.md)
(copilot-instructions.md), Manifest mit sha256 je Datei (Pfeiler 4, Provenienz),
Soundfile-32-bit-float ⇒ bit_depth=32 ⇒ keine Dither-Pflicht (§V5 (copilot-instructions.md)
(copilot-instructions.md))).

Signalpfad-Kern ist MUSIK (§III.11 (copilot-instructions.md)) — die Trainingsdaten
sind deshalb echte Musik-Stems, keine Sprach-/Synth-Daten. Transiente-Target-
Metadaten (Seg-Index, Quell-Stem, Onset-Dichte) werden im Manifest festgehalten;
die eigentliche GT (Phasen-Δφ) leitet ``train_tp_v2_phase.py`` intern aus den
Waveformen ab (TransientPhaseDataset).

Bestimmung:
  1. Jede Stem-Datei wird deterministisch in Segmente zerlegt (``seg_s``, 48 kHz).
  2. Segmente werden nach transienter Aktivität (Transientenergie über
     kurzes Analysefenster) sortiert — die aktivsten Segmente werden
     bevorzugt übernommen (``--transient-quantil``).
  3. Ausgabe + Manifest ``aurik.tp2_transient_pairs/1`` mit sha256 je Segment.

Aufruf (Daten-Synthese, KEIN Training, KEIN Flag-Flip):
    python3 -B scripts/prepare_tp2_transient_pairs.py \
        --musdb-root data/musdb18hq --out-dir data/tp2_transient_pairs --anzahl 200

Exit-Codes: 0 = ok, 2 = Nutzungs-/Datenfehler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCHEMA = "aurik.tp2_transient_pairs/1"
# Synchro zu scripts/train_tp_v2_phase.py — dort SEGMENT_SAMPLES aus TARGET_SR.
TARGET_SR = 48_000
DEFAULT_SEED = 42
STEM_NAMES = ("drums", "other", "bass")  # transient-reich; vocals bewusst ausgenommen


def _sha256(pfad: Path) -> str:
    return hashlib.sha256(pfad.read_bytes()).hexdigest()


def _to_mono(arr: np.ndarray) -> np.ndarray:
    """Normalisiert (C, N) oder (N, C) auf Mono (N,) — Layout-Invariante (AGENTS.md §3)."""
    if arr.ndim == 1:
        return arr
    if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
        return arr.mean(axis=0)  # (C, N) channels-first
    return arr.mean(axis=1)  # (N, C)


def _lade_wav_mono(pfad: Path, target_sr: int) -> np.ndarray:
    """Lädt eine WAV mono auf ``target_sr`` (deterministische Resynthese)."""
    import librosa  # erst bei Bedarf — Hauptschnittstelle bleibt import-leicht

    mono, _ = librosa.load(str(pfad), sr=target_sr, mono=True)
    return mono.astype(np.float32)


def _transientenergie(seg: np.ndarray) -> float:
    """Robuster Transienten-Proxy: Energie der Hüllkurven-Differenzen (Onset-Dichte).

    Deterministisch, ohne externe Onset-Bibliothek (§G5 (copilot-instructions.md)).
    """
    if seg.size < 32:
        return 0.0
    # Kurzzeit-Energie-Hüllkurve (nicht-überlappende Fenster à 512).
    win = 512
    n_win = seg.size // win
    if n_win < 2:
        return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
    frames = seg[: n_win * win].reshape(n_win, win)
    energy = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    diffs = np.abs(np.diff(energy))
    return float(np.mean(diffs) * 10.0 + np.sqrt(np.mean(energy**2)))


def erzeuge_paare(
    musdb_root: Path,
    out_dir: Path,
    *,
    anzahl: int = 200,
    seg_s: float = 2.5,
    seed: int = DEFAULT_SEED,
    transient_quantil: float = 0.6,
) -> dict:
    """Erzeugt transiente-reiche Segmente aus MUSDB18-HQ-Stems + Manifest."""
    if anzahl < 1 or seg_s <= 0:
        raise ValueError("anzahl >= 1 und seg_s > 0 erforderlich")
    if not 0.0 < transient_quantil <= 1.0:
        raise ValueError("transient_quantil muss in (0, 1] liegen")

    stem_pfade: list[tuple[str, Path]] = []
    for split in ("train", "test"):
        for stem in STEM_NAMES:
            stem_pfade.extend((stem, p) for p in sorted(musdb_root.glob(f"{split}/*/{stem}.wav")))
    if not stem_pfade:
        raise FileNotFoundError(f"keine Stem-Dateien ({'/'.join(STEM_NAMES)}) unter {musdb_root} (train/test)")

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 1) Alle Kandidatensegmente transient bewerten (deterministische Reihenfolge).
    kandidaten: list[tuple[float, str, Path, int, int]] = []  # (score, stem, pfad, start, ende)
    for stem_name, pfad in stem_pfade:
        audio = _lade_wav_mono(pfad, TARGET_SR)
        seg_n = int(round(seg_s * TARGET_SR))
        n_segs = audio.size // seg_n
        for k in range(n_segs):
            start = k * seg_n
            ende = start + seg_n
            seg = audio[start:ende]
            score = _transientenergie(seg)
            kandidaten.append((score, stem_name, pfad, start, ende))
        logger.debug("%s: %d Segmente", pfad.name, n_segs)

    if not kandidaten:
        raise ValueError("keine vollständigen Segmente gefunden — seg_s zu groß?")

    # 2) Nur die transient-stärksten Segmente behalten (Quantil), dann deterministisch
    #    mischen (rng), damit die Auswahl nicht rein sortiert ist.
    kandidaten.sort(key=lambda t: t[0], reverse=True)
    grenze = max(1, int(round(len(kandidaten) * transient_quantil)))
    auswahl = kandidaten[:grenze]
    idx = rng.permutation(len(auswahl))
    auswahl = [auswahl[i] for i in idx[: min(anzahl, len(auswahl))]]

    # 3) Segmente schreiben + Manifest.
    eintraege = []
    for i, (score, stem_name, pfad, start, ende) in enumerate(auswahl):
        audio = _lade_wav_mono(pfad, TARGET_SR)
        seg = audio[start:ende]
        name = f"seg_{i:04d}.wav"
        seg_pfad = out_dir / name
        import soundfile as sf

        sf.write(str(seg_pfad), seg, TARGET_SR, subtype="FLOAT")
        eintraege.append(
            {
                "datei": name,
                "quelle": str(pfad),
                "quelle_stem": stem_name,
                "start_sample": int(start),
                "ende_sample": int(ende),
                "transient_score": float(score),
                "sha256": _sha256(seg_pfad),
            }
        )
        if (i + 1) % 50 == 0:
            logger.info("%d/%d Segmente geschrieben", i + 1, len(auswahl))

    manifest = {
        "schema": SCHEMA,
        "seed": seed,
        "target_sr": TARGET_SR,
        "seg_s": seg_s,
        "anzahl": len(eintraege),
        "transient_quantil": transient_quantil,
        "stems": list(STEM_NAMES),
        "quelle": "scripts/prepare_tp2_transient_pairs.py",
        "segmente": eintraege,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info(
        "fertig: %d transiente Segmente nach %s (Manifest: manifest.json)",
        len(eintraege),
        out_dir,
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="D-K4-8: Transient-reiche MUSDB18-HQ-Stem-Segmente für TP-V2 (Seed 42)."
    )
    parser.add_argument("--musdb-root", type=Path, default=Path("data/musdb18hq"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/tp2_transient_pairs"))
    parser.add_argument("--anzahl", type=int, default=200)
    parser.add_argument("--seg-s", type=float, default=2.5)
    parser.add_argument("--transient-quantil", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    try:
        erzeuge_paare(
            args.musdb_root,
            args.out_dir,
            anzahl=args.anzahl,
            seg_s=args.seg_s,
            seed=args.seed,
            transient_quantil=args.transient_quantil,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Datenfehler: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
