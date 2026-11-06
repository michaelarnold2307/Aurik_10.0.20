#!/usr/bin/env python3
"""Echt-Trainingspaare (clean/noisy) für SOTA-M9 TF-GridNet (Defizit D-K4-2).

K1-konform (SOTA_DEFICIT_REGISTER Klasse K1): **clean = echte Musik**
(MUSDB18-HQ ``mixture.wav``), **noisy = clean + additives Rauschen** — das
DNS-Muster (DNS-Challenge): das Zielsignal bleibt echte Musik
(§III.11 (copilot-instructions.md)). Bewusst KEIN ``--synthetic``
(Sinus + Rauschen) im Trainingspfad — das ist laut Docstring von
``scripts/train_tf_gridnet_denoiser.py`` nur „zum Test" und würde ein
K1-Defizit erzeugen (nur Sinus-Harmonische statt Musik gesehen).

Determinismus (§G5 (copilot-instructions.md)): gleicher ``--seed`` ⇒
byte-identische Paare. Track-Wahl, Offsets, Rauschquellen und SNR-Stufen
sind reine Funktionen von ``(seed, pair_index)`` — kein ``time.time()``,
kein ungesäter Zufall.

Ausgabe (von ``scripts/gpu_kette3_k4_rest.sh`` R2 als ``--clean-dir``/
``--noisy-dir`` konsumiert — Paarung über gleiche Dateinamen):
    data/tfgrid_pairs/clean/pair_0000.wav …
    data/tfgrid_pairs/noisy/pair_0000.wav …
    data/tfgrid_pairs/manifest.json   (Schema aurik.tfgrid_pairs/1)

Rauschquellen (bevorzugt echte Samples, §V6 (copilot-instructions.md)):
    1. models/deepfilternet_v3_ii/assets/noise_freesound_*.wav (falls vorhanden)
    2. farbiges Rauschen (pink/brown/white) als zweite Quelle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Synchro zu scripts/train_tf_gridnet_denoiser.py (dessen TARGET_SR-Import aus
# backend/core/dsp/tf_gridnet_denoiser.py); hier als Konstante, damit das
# Datensatz-Werkzeug keine Backend-Abhängigkeit zieht (§G9 (GEBOTE.md) Ein Quelle-Wahrung
# über die dokumentierte Kopplung im Manifest-Feld target_sr).
TARGET_SR = 48_000

SCHEMA = "aurik.tfgrid_pairs/1"
DEFAULT_SNR_DB = (0.0, 5.0, 10.0, 15.0, 20.0)
# Lautstärke-Referenz der echten Rausch-Samples (dokumentierte Asset-Pfade).
DEFAULT_NOISE_WAVS = (
    "models/deepfilternet_v3_ii/assets/noise_freesound_2530.wav",
    "models/deepfilternet_v3_ii/assets/noise_freesound_573577.wav",
)


def _sha256(pfad: Path) -> str:
    return hashlib.sha256(pfad.read_bytes()).hexdigest()


def _zu_mono(audio: np.ndarray) -> np.ndarray:
    """Stereo-Layout-Invariante (AGENTS.md §3): akzeptiert (C, N) ODER (N, C).

    Pipeline-intern ist Stereo channels-first (C, N); WAV-Leser liefern
    (N, C). Die Normalisierung erfolgt an jeder Modulgrenze — hier beim Lesen.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"erwarte 2-D Audio (C, N) oder (N, C), bekam {arr.shape}")
    if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
        # (C, N) — channels-first: über Kanäle mitteln
        return arr.mean(axis=0)
    # (N, C) — soundfile-Standard
    return arr.mean(axis=1)


def _lade_wav_mono(pfad: Path, target_sr: int) -> np.ndarray:
    """Lädt eine WAV mono auf ``target_sr`` (deterministische Resynthese)."""
    import librosa  # erst bei Bedarf — Hauptschnittstelle bleibt import-leicht

    mono, _ = librosa.load(str(pfad), sr=target_sr, mono=True)
    return mono.astype(np.float32)


def _farbrauschen(n: int, art: str, rng: np.random.Generator) -> np.ndarray:
    """Farbiges Rauschen (pink/brown/white), deterministisch aus ``rng``."""
    weiss = rng.standard_normal(n).astype(np.float64)
    if art == "white":
        rauschen = weiss
    elif art == "pink":
        # Voss-McCartney-freier Spektral-Ansatz: 1/f-Amplitudenform
        spektrum = np.fft.rfft(weiss)
        freqs = np.fft.rfftfreq(n, d=1.0 / TARGET_SR)
        freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
        spektrum = spektrum / np.sqrt(freqs)
        rauschen = np.fft.irfft(spektrum, n=n)
    elif art == "brown":
        rauschen = np.cumsum(weiss)
        # Trend entfernen, sonst dominiert der Gleichanteil
        rauschen = rauschen - np.linspace(0.0, rauschen[-1], n, endpoint=False)
    else:
        raise ValueError(f"unbekannte Rauschart: {art}")
    std = float(np.std(rauschen))
    if std < 1e-12:
        raise ValueError("degeneriertes Rauschen (std ≈ 0)")
    return (rauschen / std).astype(np.float32)


def _snr_skalieren(clean: np.ndarray, rauschen: np.ndarray, snr_db: float) -> np.ndarray:
    """Skaliert ``rauschen`` auf die Ziel-SNR gegen ``clean`` (Leistungsverhältnis)."""
    p_clean = float(np.mean(clean.astype(np.float64) ** 2))
    p_r = float(np.mean(rauschen.astype(np.float64) ** 2))
    if p_clean < 1e-12 or p_r < 1e-12:
        raise ValueError("Null-Leistung — Segment/Rauschen nicht skalierbar")
    faktor = np.sqrt(p_clean / (p_r * 10.0 ** (snr_db / 10.0)))
    return (rauschen * faktor).astype(np.float32)


def erzeuge_paare(
    musdb_root: Path,
    out_dir: Path,
    *,
    anzahl: int = 300,
    seg_s: float = 4.0,
    seed: int = 42,
    snr_db_stufen: tuple[float, ...] = DEFAULT_SNR_DB,
    noise_wavs: tuple[str, ...] = DEFAULT_NOISE_WAVS,
) -> dict:
    """Erzeugt ``anzahl`` Paare + Manifest; deterministisch (§G5 (GEBOTE.md))."""
    if anzahl < 1 or seg_s <= 0:
        raise ValueError("anzahl >= 1 und seg_s > 0 erforderlich")

    mix_pfade = sorted(musdb_root.glob("train/*/mixture.wav")) + sorted(musdb_root.glob("test/*/mixture.wav"))
    if not mix_pfade:
        raise FileNotFoundError(f"keine mixture.wav unter {musdb_root} (train/test)")

    echte_noise = [Path(p) for p in noise_wavs if Path(p).is_file()]
    if echte_noise:
        logger.info("Rauschquellen: %d echte Samples (bevorzugt)", len(echte_noise))
    else:
        logger.warning(
            "keine echten Rausch-Samples gefunden (%s) — nur farbiges Rauschen "
            "(§V6 (copilot-instructions.md): Ersatzpfad begründet)",
            ", ".join(noise_wavs),
        )

    seg_n = int(round(seg_s * TARGET_SR))
    clean_dir = out_dir / "clean"
    noisy_dir = out_dir / "noisy"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    noise_cache: dict[Path, np.ndarray] = {}
    eintraege = []
    for i in range(anzahl):
        rng = np.random.default_rng([seed, i])  # rein aus (seed, i), §G5 (GEBOTE.md)
        mix_pfad = mix_pfade[int(rng.integers(len(mix_pfade)))]
        mono = _lade_wav_mono(mix_pfad, TARGET_SR)
        if len(mono) < seg_n:
            raise ValueError(f"Track zu kurz für {seg_s}s-Segment: {mix_pfad}")
        start = int(rng.integers(0, len(mono) - seg_n + 1))
        clean = mono[start : start + seg_n]
        snr_db = float(snr_db_stufen[i % len(snr_db_stufen)])

        if echte_noise and rng.random() < 0.5:
            n_pfad = echte_noise[int(rng.integers(len(echte_noise)))]
            if n_pfad not in noise_cache:
                noise_cache[n_pfad] = _lade_wav_mono(n_pfad, TARGET_SR)
            quelle = n_pfad.name
            quelle_art = "echt"
            n_src = noise_cache[n_pfad]
            if len(n_src) < seg_n:
                n_src = np.tile(n_src, int(np.ceil(seg_n / len(n_src))))[:seg_n]
            n_start = int(rng.integers(0, len(n_src) - seg_n + 1))
            rauschen = n_src[n_start : n_start + seg_n]
        else:
            art = ("pink", "brown", "white")[int(rng.integers(3))]
            quelle, quelle_art = f"farbig:{art}", "farbig"
            rauschen = _farbrauschen(seg_n, art, rng)

        noisy = clean + _snr_skalieren(clean, rauschen, snr_db)
        # §0a (copilot-instructions.md): NaN/Inf-Schutz an der Ausgabegrenze
        noisy = np.nan_to_num(noisy, nan=0.0, posinf=1.0, neginf=-1.0)

        name = f"pair_{i:04d}.wav"
        # 32-bit float ⇒ bit_depth = 32 ⇒ keine Dither-Pflicht (§V5 (copilot-instructions.md)
        # (copilot-instructions.md) greift nur bei bit_depth < 32).
        sf.write(str(clean_dir / name), clean, TARGET_SR, subtype="FLOAT")
        sf.write(str(noisy_dir / name), noisy, TARGET_SR, subtype="FLOAT")
        eintraege.append(
            {
                "datei": name,
                "quelle": mix_pfad.relative_to(musdb_root).as_posix(),
                "offset_s": round(start / TARGET_SR, 4),
                "snr_db": snr_db,
                "rauschquelle": quelle,
                "rauschquelle_art": quelle_art,
                "sha256_clean": _sha256(clean_dir / name),
                "sha256_noisy": _sha256(noisy_dir / name),
            }
        )
        if (i + 1) % 50 == 0:
            logger.info("%d/%d Paare erzeugt", i + 1, anzahl)

    manifest = {
        "schema": SCHEMA,
        "seed": seed,
        "target_sr": TARGET_SR,
        "seg_s": seg_s,
        "anzahl": anzahl,
        "snr_db_stufen": list(snr_db_stufen),
        "snr_verteilung": {str(s): sum(1 for e in eintraege if e["snr_db"] == s) for s in snr_db_stufen},
        "rauschquellen": sorted({e["rauschquelle"] for e in eintraege}),
        "k1_hinweis": (
            "clean = echte Musik (MUSDB18-HQ); noisy = clean + additives Rauschen "
            "(DNS-Muster). KEIN --synthetic (Sinus+Rauschen) — K1-Vermeidung."
        ),
        "paare": eintraege,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("fertig: %d Paare nach %s (Manifest: manifest.json)", anzahl, out_dir)
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--musdb-root", type=Path, default=Path("data/musdb18hq"))
    p.add_argument("--out-dir", type=Path, default=Path("data/tfgrid_pairs"))
    p.add_argument("--pairs", type=int, default=300)
    p.add_argument("--seg-seconds", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42, help="Determinismus §G5 (GEBOTE.md)")
    args = p.parse_args(argv)
    try:
        erzeuge_paare(
            args.musdb_root,
            args.out_dir,
            anzahl=args.pairs,
            seg_s=args.seg_seconds,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Abbruch: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
