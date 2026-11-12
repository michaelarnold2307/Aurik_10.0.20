#!/usr/bin/env python3
"""R5-Daten (SOTA-M14): ECHTE degradierte Vocal-Paare für Cantus domain_adapt.

Warum: ``gpu_kette1_welle_c.sh`` R5 (Cantus Phase 3) verlangt
``data/cantus_real/*_degraded.wav`` mit zugehörigem ``*_clean.wav``. Die
synthetischen ``cantus_pairs`` (weißes Rauschen, synthetischer Hall, Clip)
sind für domain_adapt die falsche Domäne — Phase 3 braucht **echte**
Degradationen (§V7 (copilot-instructions.md): Ursache statt Symptom).

Echte Quellen (lokal, ohne Lizenz-/Beschaffungsrisiko):
- ``data/vinyl_noise_bank/`` — echte, beschädigte Vinyl-Aufnahmen
  (Leerrillen, echt_beschaedigt) als realer Rausch-/Knistern-Träger;
- ``data/rir_real/`` — echte Raumimpulsantworten (Omni-Erdeworks-Messungen);
- Clean-Seite: MUSDB18-HQ-Vocals (trainiertes, lizenziertes Korpus).

Degradations-Ketten (deterministisch, Seed 42, §G5 (GEBOTE.md)):
- ``vinyl``: echte Vinyl-Rauschbank + deterministische Klicker (echte
  Knistern-Statistik) bei zufälligem, geloggtem SNR;
- ``rir``: echte RIR-Faltung (wet-Mix geloggt);
- ``vinyl_rir``: beides hintereinander.

Ausgabe: ``<stem>_clean.wav`` / ``<stem>_degraded.wav`` (Paarung per
``deg.name.replace("_degraded.wav", "_clean.wav")`` in train_cantus.py
Z. 644/645) + ``manifest.jsonl`` (track, split, kind, params, clean,
degraded, sha256_pcm beider Seiten — float32-PCM-Hash, §G5 (GEBOTE.md)).

Determinismus: feste Seeds, sortierte Track-Reihenfolge, keine
zeitabhängigen Entscheidungen.

Usage:
    python3 -B scripts/erstelle_cantus_real_paare.py \
        --musdb-root data/musdb18hq --out data/cantus_real \
        --tracks 20 --seconds 10 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

logger = logging.getLogger(__name__)

SR = 48000
KINDS = ("vinyl", "rir", "vinyl_rir")


def _pcm_sha256(x: np.ndarray) -> str:
    """Hash der float32-PCM-Daten (verbindlich, §G5 (GEBOTE.md))."""
    return hashlib.sha256(np.asarray(x, dtype=np.float32).tobytes()).hexdigest()


def _resample_48k(x: np.ndarray, sr_in: int) -> np.ndarray:
    from scipy.signal import resample_poly

    if sr_in == SR:
        return np.asarray(x, dtype=np.float32)
    g = int(np.gcd(int(sr_in), SR))
    return resample_poly(np.asarray(x, dtype=np.float64), SR // g, int(sr_in) // g).astype(np.float32)


def _load_real_noise_bank(noise_dir: Path, rng: np.random.Generator) -> np.ndarray:
    """Zieht ein zufälliges Segment aus der echten Vinyl-Rauschbank (deterministisch)."""
    wavs = sorted(p for p in noise_dir.glob("*.wav") if p.is_file())
    if not wavs:
        raise FileNotFoundError(f"keine Rauschbank-WAVs in {noise_dir}")
    from soundfile import SoundFile

    for _attempt in range(8):
        src = wavs[int(rng.integers(0, len(wavs)))]
        with SoundFile(str(src)) as f:
            sr_in = f.samplerate
            total = len(f)
            need = int(12.0 * sr_in)
            if total <= need:
                f.seek(0)
                data = f.read(dtype="float32", always_2d=True)
            else:
                start = int(rng.integers(0, total - need))
                f.seek(start)
                data = f.read(need, dtype="float32", always_2d=True)
        mono = np.asarray(data, dtype=np.float32).mean(axis=1)
        mono = _resample_48k(mono, int(sr_in))
        mono = mono[: int(12.0 * SR)]
        if mono.size >= SR and float(np.sqrt((mono**2).mean())) > 1e-6:
            return np.asarray(mono, dtype=np.float32)
    raise RuntimeError("kein brauchbares Rausch-Segment gefunden")


def _load_real_ir(ir_dir: Path, rng: np.random.Generator) -> np.ndarray:
    """Zieht eine echte Raumimpulsantwort (deterministisch), gekürzt auf 1,5 s."""
    wavs = sorted(p for p in ir_dir.glob("*.wav") if p.is_file())
    if not wavs:
        raise FileNotFoundError(f"keine IR-WAVs in {ir_dir}")
    from soundfile import SoundFile

    for _attempt in range(8):
        src = wavs[int(rng.integers(0, len(wavs)))]
        with SoundFile(str(src)) as f:
            sr_in = f.samplerate
            data = f.read(dtype="float32", always_2d=True)
        mono = np.asarray(data, dtype=np.float32).mean(axis=1)
        mono = _resample_48k(mono, int(sr_in))
        mono = mono[: int(1.5 * SR)]
        if mono.size >= int(0.1 * SR) and float(np.sqrt((mono**2).mean())) > 1e-6:
            return np.asarray(mono, dtype=np.float32)
    raise RuntimeError("keine brauchbare IR gefunden")


def _vinyl_noise(clean: np.ndarray, noise: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    """Echtes Vinyl-Rauschen bei geloggtem SNR + deterministische Klicker."""
    from scripts.erstelle_schadens_proben import klicker

    need = clean.shape[-1]
    if noise.shape[-1] < need:
        noise = np.pad(noise, (0, need - noise.shape[-1]))
    start = int(rng.integers(0, max(1, noise.shape[-1] - need + 1)))
    seg = noise[start : start + need]
    p_clean = float(np.sqrt((clean**2).mean())) + 1e-12
    p_noise = float(np.sqrt((seg**2).mean())) + 1e-12
    gain = p_clean / p_noise * (10 ** (-snr_db / 20.0))
    out = np.asarray(clean + gain * seg, dtype=np.float64)
    out = klicker(np.asarray(out, dtype=np.float32), rng, dichte=2.0)
    peak = float(np.max(np.abs(out))) + 1e-12
    return np.asarray(out * min(1.0, 0.98 / peak), dtype=np.float32)


def _rir_mix(clean: np.ndarray, ir: np.ndarray, wet: float) -> np.ndarray:
    from scipy.signal import fftconvolve

    ir = ir / (float(np.sqrt((ir**2).mean())) + 1e-12)
    tail = fftconvolve(np.asarray(clean, dtype=np.float64), np.asarray(ir, dtype=np.float64), mode="full")[
        : clean.shape[-1]
    ]
    out = (1.0 - wet) * np.asarray(clean, dtype=np.float64) + wet * tail
    peak = float(np.max(np.abs(out))) + 1e-12
    return np.asarray(out * min(1.0, 0.98 / peak), dtype=np.float32)


def _degradiere(
    clean: np.ndarray, kind: str, rng: np.random.Generator, noise_bank: Path, ir_dir: Path
) -> tuple[np.ndarray, dict]:
    params: dict = {"kind": kind}
    out = np.asarray(clean, dtype=np.float32)
    if kind in ("vinyl", "vinyl_rir"):
        snr_db = float(rng.uniform(4.0, 14.0))
        params["snr_db"] = round(snr_db, 4)
        params["noise_quelle"] = "data/vinyl_noise_bank"
        out = _vinyl_noise(out, _load_real_noise_bank(noise_bank, rng), rng, snr_db)
    if kind in ("rir", "vinyl_rir"):
        wet = float(rng.uniform(0.15, 0.45))
        params["wet"] = round(wet, 4)
        params["ir_quelle"] = "data/rir_real"
        out = _rir_mix(out, _load_real_ir(ir_dir, rng), wet)
    return out, params


def _vocal_stems(musdb_root: Path, limit: int) -> list[Path]:
    split = musdb_root / "train"
    stems: list[Path] = []
    for track in sorted(p for p in split.iterdir() if p.is_dir()):
        v = track / "vocals.wav"
        if v.is_file():
            stems.append(v)
        if len(stems) >= limit:
            break
    return stems


def _mittel_ausschnitt(audio: np.ndarray, sekunden: float) -> np.ndarray:
    arr = np.asarray(audio)
    want = int(round(sekunden * SR))
    if arr.shape[-1] <= want or sekunden <= 0:
        return arr
    start = (arr.shape[-1] - want) // 2
    return arr[..., start : start + want]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--musdb-root", type=Path, default=Path("data/musdb18hq"))
    ap.add_argument("--out", type=Path, default=Path("data/cantus_real"))
    ap.add_argument("--noise-bank", type=Path, default=Path("data/vinyl_noise_bank"))
    ap.add_argument("--ir-dir", type=Path, default=Path("data/rir_real"))
    ap.add_argument("--tracks", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import soundfile as sf

    from scripts.erstelle_schadens_proben import _decode_48k

    rng = np.random.default_rng(args.seed)
    stems = _vocal_stems(args.musdb_root, args.tracks)
    if not stems:
        logger.error("keine Vocal-Stems unter %s", args.musdb_root)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    zeilen: list[dict] = []
    for stem in stems:
        audio = _decode_48k(stem)
        clean = np.asarray(_mittel_ausschnitt(audio, args.seconds), dtype=np.float32)
        track_name = stem.parent.name
        for kind in KINDS:
            degraded, params = _degradiere(clean, kind, rng, args.noise_bank, args.ir_dir)
            base = f"{track_name}__{kind}"
            clean_path = args.out / f"{base}_clean.wav"
            deg_path = args.out / f"{base}_degraded.wav"
            sf.write(str(clean_path), clean, SR, subtype="FLOAT")
            sf.write(str(deg_path), degraded, SR, subtype="FLOAT")
            zeilen.append(
                {
                    "track": track_name,
                    "split": "train",
                    "variant": kind,
                    "kind": kind,
                    "params": params,
                    "clean": clean_path.name,
                    "degraded": deg_path.name,
                    "sha256_pcm": _pcm_sha256(degraded),
                    "sha256_pcm_clean": _pcm_sha256(clean),
                }
            )
            logger.info("%s: snr/wet=%s", base, params)

    manifest = args.out / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for row in zeilen:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
    logger.info("%d Paare -> %s", len(zeilen), manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
