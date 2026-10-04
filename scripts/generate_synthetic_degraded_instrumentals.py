#!/usr/bin/env python3
"""Erzeugt deterministische MUSDB18-HQ-Instrumentalpaare für Symphonia.

Die Referenz ist die phasengleiche Summe aus drums, bass und other. Die
Degradationen stammen aus dem kanonischen Cantus-Generator und bleiben damit
bei Dither und Seeds identisch geprüft (§G5, §V5 (VERBOTEN.md)).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_synthetic_degraded_vocals import DEFAULT_SEED, KINDS, _load_vocal, _seed_for, degrade_one

logger = logging.getLogger(__name__)


def generate(musdb_root: Path, out_root: Path, variants_per_track: int = 8, max_tracks: int = 0) -> Path:
    """Erzeugt Paare ohne Vocal-Leakage; Rückgabe ist das Manifest."""
    tracks = sorted(path for split in ("train", "test") for path in (musdb_root / split).glob("*") if path.is_dir())
    if max_tracks:
        tracks = tracks[:max_tracks]
    if not tracks:
        raise FileNotFoundError(f"Keine MUSDB18-HQ-Tracks unter {musdb_root}")
    rows: list[dict] = []
    for track_dir in tracks:
        components = [track_dir / name for name in ("drums.wav", "bass.wav", "other.wav")]
        if not all(path.is_file() for path in components):
            continue
        clean = np.sum([_load_vocal(path) for path in components], axis=0, dtype=np.float32)
        clean = clean / (float(np.max(np.abs(clean))) + 1e-10) * 0.891
        split = track_dir.parent.name
        output = out_root / split
        output.mkdir(parents=True, exist_ok=True)
        clean_path = output / f"{track_dir.name}__instrumental_clean.wav"
        wavfile.write(clean_path, 48000, clean.T.astype(np.float32))
        for variant in range(variants_per_track):
            seed = _seed_for(DEFAULT_SEED, track_dir.name, variant)
            np.random.seed(seed % (2**32))
            degraded, params = degrade_one(clean, KINDS[variant % len(KINDS)], np.random.default_rng(seed))
            degraded_path = output / f"{track_dir.name}__instrumental_v{variant:02d}.wav"
            wavfile.write(degraded_path, 48000, degraded.T.astype(np.float32))
            rows.append({"track": track_dir.name, "split": split, "variant": variant, "clean": str(clean_path.relative_to(out_root)), "degraded": str(degraded_path.relative_to(out_root)), "params": params, "seed": seed, "sr": 48000, "sources": ["drums", "bass", "other"]})
    manifest = out_root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    logger.info("Symphonia: %d Instrumentalpaare → %s", len(rows), manifest)
    return manifest


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musdb-root", type=Path, default=Path("data/musdb18hq"))
    parser.add_argument("--out", type=Path, default=Path("data/symphonia_pairs"))
    parser.add_argument("--variants-per-track", type=int, default=8)
    parser.add_argument("--max-tracks", type=int, default=0)
    args = parser.parse_args()
    generate(args.musdb_root, args.out, args.variants_per_track, args.max_tracks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
