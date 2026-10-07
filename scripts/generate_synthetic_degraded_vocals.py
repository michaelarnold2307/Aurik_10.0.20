#!/usr/bin/env python3
"""Generiert synthetisch degradierte/clean-Vocal-Paare für das Cantus-Training.

Problem: Es gibt keinen großen öffentlichen Datensatz mit paarweisen
degradierten/cleanen Gesang. Lösung: synthetische Degradation auf cleanen
MUSDB18-HQ-Vocal-Stems.

Degradationsarten (pro Paar deterministisch parametrisiert, §G5 (GEBOTE.md)):
  - add_noise      : Rauschen bei SNR ∈ [-5, 20] dB (weiß + rosa)
  - add_reverb     : Raumhall, synthetische IR, RT60 ∈ [0.1, 2.0] s
  - clip           : Clipping-Verzerrung (hard/soft, Drive-Serie)
  - bitcrush       : Bit-Tiefen-Reduktion 8/12/16 bit — MIT TPDF-Dither
                     (§V5 (VERBOTEN.md): kein nacktes astype(int16); TPDF als
                     dokumentierter Fallback der POW-r-Type-3-Erstwahl)
  - resample        : Low-Sample-Rate-Artefakte (8/16/22.05 kHz Bandbreitenlimit)
  - hum             : Netzbrummen 50 Hz + Harmonische
  - tape            : Tape-Sättigung + Wow&Flatter + Hiss
  - stack           : deterministische Kombination mehrerer Degradationen

Ausgabe (float32-WAV, 48 kHz, Stereo-Layout beim Schreiben (N, 2), intern
channels-first (C, N) — §Stereo-Layout-Invariante (copilot-instructions.md)):
  <out>/<split>/<track>__clean.wav
  <out>/<split>/<track>__vNN__<kind>.wav
  <out>/manifest.jsonl  — Paare + vollständige Degradations-Parameter + Seeds

Usage:
    python3 -B scripts/generate_synthetic_degraded_vocals.py \
        --musdb-root data/musdb18hq --out data/cantus_pairs --variants-per-track 8
    python3 -B scripts/generate_synthetic_degraded_vocals.py \
        --musdb-root data/musdb18hq --out data/cantus_pairs --variants-per-track 24 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import cast

import numpy as np
from scipy import signal
from scipy.io import wavfile

logger = logging.getLogger(__name__)

SR_OUT = 48000
DEFAULT_SEED = 20261004
KINDS = ("add_noise", "add_reverb", "clip", "bitcrush", "resample", "hum", "tape", "stack")


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────


def _seed_for(seed: int, track: str, variant: int) -> int:
    """Deterministischer Paar-Seed aus (seed, track, variant) — §G5 (GEBOTE.md)."""
    h = 0
    for ch in f"{track}:{variant}".encode():
        h = (h * 131 + ch) % (2**31 - 1)
    return int((seed + h) % (2**31 - 1))


def _to_channels_first(audio: np.ndarray) -> np.ndarray:
    """Normalisiert auf Layout channels-first (C, N) — §Stereo-Layout-Invariante.

    §G9 (copilot-instructions.md): Die Entscheidung kommt aus der einen
    kanonischen Quelle `backend.core.audio_layout` (diese Kopie war eine von
    vier divergenten Heuristiken, Muster D-K3-6).
    """
    from backend.core.audio_layout import normalize_channels_first

    arr_cn, _ = normalize_channels_first(np.asarray(audio, dtype=np.float32))
    return arr_cn


def _finite_guard(audio: np.ndarray, where: str) -> np.ndarray:
    """NaN/Inf-Schutz jeder Phase (§0a (copilot-instructions.md))."""
    if not np.all(np.isfinite(audio)):
        n_bad = int(np.sum(~np.isfinite(audio)))
        logger.warning("§0a NaN/Inf-Schutz: %d nicht-finite Samples in %s → 0.0", n_bad, where)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio.astype(np.float32)


def _load_vocal(path: Path, sr_out: int = SR_OUT) -> np.ndarray:
    """Lädt Vocal-Stem und resampelt nach 48 kHz mono/stereo (C, N) float32."""
    read_result = wavfile.read(path)
    if not isinstance(read_result, tuple) or len(read_result) != 2:
        raise ValueError(f"Ungültiges WAV-Leseergebnis für {path}")
    sr_in = int(read_result[0])
    data = np.asarray(read_result[1])
    audio = data.astype(np.float32)
    if data.dtype == np.int16:
        audio /= 32768.0
    elif data.dtype == np.int32:
        audio /= 2147483648.0
    audio = _to_channels_first(audio)
    if sr_in != sr_out:
        g = np.gcd(sr_in, sr_out)
        audio = np.stack(
            [signal.resample_poly(ch, sr_out // g, sr_in // g).astype(np.float32) for ch in audio],
            axis=0,
        )
    return _finite_guard(audio, f"load {path.name}")


def _snr_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Skaliert `noise` so, dass clean:noise = snr_db — shape (C, N)."""
    n = noise[:, : clean.shape[1]]
    if n.shape[1] < clean.shape[1]:
        reps = int(np.ceil(clean.shape[1] / max(n.shape[1], 1)))
        n = np.tile(n, (1, reps))[:, : clean.shape[1]]
    p_clean = float(np.mean(clean**2)) + 1e-12
    p_noise = float(np.mean(n**2)) + 1e-12
    return np.asarray(n * np.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10.0))), dtype=np.float32)


def _pink_noise(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Rosa Rauschen via 1/f-Filter auf weißem Rauschen (Voss-McCartney-Ersatz)."""
    white = rng.standard_normal(shape).astype(np.float32)
    coefficients = signal.butter(1, 0.05, btype="low", output="ba")
    if coefficients is None:
        raise RuntimeError("Butterworth-Filterkoeffizienten konnten nicht erzeugt werden")
    b, a = cast(tuple[np.ndarray, np.ndarray], coefficients)
    pink = np.asarray(signal.lfilter(b, a, white, axis=-1), dtype=np.float32)
    pink = np.asarray(pink + 0.3 * white, dtype=np.float32)
    return pink


# ── Degradationen (alle deterministisch aus rng + expliziten Parametern) ─────


def add_noise(audio: np.ndarray, rng: np.random.Generator, snr_db: float, kind: str = "white") -> np.ndarray:
    if kind == "pink":
        noise = _pink_noise(rng, audio.shape)
    else:
        noise = rng.standard_normal(audio.shape).astype(np.float32)
    return _finite_guard(audio + _snr_noise(audio, noise, snr_db), "add_noise")


def add_reverb(audio: np.ndarray, rng: np.random.Generator, rt60: float, wet: float = 0.35) -> np.ndarray:
    """Raumhall via exponentiell abklingender Rausch-IR (pro Kanal leicht verschieden)."""
    sr = SR_OUT
    n_ir = max(int(rt60 * sr), 64)
    t = np.arange(n_ir, dtype=np.float32) / sr
    decay = np.exp(-6.9 * t / rt60).astype(np.float32)
    out = np.zeros_like(audio)
    for c in range(audio.shape[0]):
        ir = rng.standard_normal(n_ir).astype(np.float32) * decay
        coefficients = signal.butter(2, 6000.0 / (sr / 2), btype="low", output="ba")  # Hall ohne HF-Glanz
        if coefficients is None:
            raise RuntimeError("Butterworth-Filterkoeffizienten konnten nicht erzeugt werden")
        b, a = cast(tuple[np.ndarray, np.ndarray], coefficients)
        ir = signal.lfilter(b, a, ir).astype(np.float32)
        ir /= float(np.sqrt(np.sum(ir**2)) + 1e-12)
        wet_sig = signal.fftconvolve(audio[c], ir)[: audio.shape[1]].astype(np.float32)
        out[c] = (1.0 - wet) * audio[c] + wet * wet_sig
    return _finite_guard(out, "add_reverb")


def apply_distortion_clip(audio: np.ndarray, drive: float, mode: str = "hard") -> np.ndarray:
    x = audio * drive
    if mode == "soft":
        x = np.tanh(x) / np.tanh(drive)  # sanfte Sättigung
    else:
        x = np.clip(x, -1.0, 1.0)  # bewusstes Clipping als Degradation
    return _finite_guard(x / max(drive * 0.5, 1.0), "clip")


def apply_bitcrush(audio: np.ndarray, bits: int) -> np.ndarray:
    """Bit-Tiefen-Reduktion MIT TPDF-Dither (§V5 (VERBOTEN.md)).

    POW-r Type 3 ist die dokumentierte Erstwahl der Produktions-Exporte;
    für den synthetischen Degradationspfad genügt der zugelassene
    TPDF-Fallback (dreiecksverteiltes Dither ±1 LSB vor der Quantisierung).
    """
    step = 2.0 ** (1 - bits)
    dither = (rng_tpdf(audio.shape) - rng_tpdf(audio.shape)) * step  # TPDF = U1 − U2
    quant = np.round((audio + dither) / step) * step
    return _finite_guard(quant, "bitcrush")


def rng_tpdf(shape: tuple[int, ...]) -> np.ndarray:
    """Hilfs-Uniform (0,1) über globalen numpy-Seed — Determinismus beim Aufrufer."""
    return np.random.random(shape).astype(np.float32)


def apply_resample(audio: np.ndarray, rate: int) -> np.ndarray:
    """Bandbreitenlimit + Resampling-Artefakte (8k/16k/22.05k und zurück)."""
    g = np.gcd(rate, SR_OUT)
    out = np.stack(
        [
            signal.resample_poly(signal.resample_poly(ch, rate // g, SR_OUT // g), SR_OUT // g, rate // g)[
                : audio.shape[1]
            ].astype(np.float32)
            for ch in audio
        ],
        axis=0,
    )
    return _finite_guard(out, "apply_resample")


def add_hum(audio: np.ndarray, freq: float = 50.0, level: float = 0.02) -> np.ndarray:
    n = audio.shape[1]
    t = np.arange(n, dtype=np.float32) / SR_OUT
    hum = sum(np.sin(2 * np.pi * freq * k * t) / k for k in (1, 2, 3, 4)).astype(np.float32)
    return _finite_guard(audio + level * hum[np.newaxis, :], "add_hum")


def apply_tape_saturation(audio: np.ndarray, rng: np.random.Generator, drive: float = 1.6) -> np.ndarray:
    """Tape-Artefakte: Sättigung (tanh) + Wow&Flatter (langsame PM) + Hiss."""
    n = audio.shape[1]
    sat = np.tanh(audio * drive) / np.tanh(drive)
    # Wow & Flutter: langsame Lautstärken-Modulation als PM-Ersatz (±0.3 % Delay-Jitter)
    t = np.arange(n, dtype=np.float32) / SR_OUT
    warp = (
        2.5 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 2 * np.pi))
        + 1.8 * np.sin(2 * np.pi * 6.1 * t + rng.uniform(0, 2 * np.pi))
    ).astype(np.float32)
    idx = np.clip(np.arange(n) + warp, 0, n - 1)
    i0 = np.floor(idx).astype(np.int64)
    frac = (idx - i0).astype(np.float32)
    i1 = np.minimum(i0 + 1, n - 1)
    wow = (
        np.take_along_axis(sat, i0[np.newaxis, :], axis=1) * (1 - frac)
        + np.take_along_axis(sat, i1[np.newaxis, :], axis=1) * frac
    ).astype(np.float32)
    hiss = _pink_noise(rng, audio.shape) * 0.003
    return _finite_guard(0.95 * wow + hiss, "tape")


def _stack_degradations(audio: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Deterministische Kombination 2–3 verschiedener Degradationen."""
    kinds_pool = ["add_noise", "add_reverb", "clip", "bitcrush", "resample", "hum", "tape"]
    n_stack = int(rng.integers(2, 4))
    chosen = [kinds_pool[i] for i in rng.choice(len(kinds_pool), size=n_stack, replace=False)]
    params: dict = {"sub_kinds": chosen}
    out = audio
    for kind in chosen:
        out, sub = degrade_one(out, kind, rng)
        params.update({f"{kind}_{k}": v for k, v in sub.items()})
    return out, params


def degrade_one(audio: np.ndarray, kind: str, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Eine Degradation mit zufällig gezogenen, dokumentierten Parametern."""
    if kind == "add_noise":
        snr = float(rng.uniform(-5.0, 20.0))
        noise_kind = "pink" if rng.random() < 0.5 else "white"
        return add_noise(audio, rng, snr, noise_kind), {"snr_db": snr, "noise_kind": noise_kind}
    if kind == "add_reverb":
        rt60 = float(rng.uniform(0.1, 2.0))
        wet = float(rng.uniform(0.2, 0.5))
        return add_reverb(audio, rng, rt60, wet), {"rt60_s": rt60, "wet": wet}
    if kind == "clip":
        drive = float(rng.uniform(1.5, 6.0))
        mode = "soft" if rng.random() < 0.5 else "hard"
        return apply_distortion_clip(audio, drive, mode), {"drive": drive, "mode": mode}
    if kind == "bitcrush":
        bits = int(rng.choice([8, 12, 16]))
        return apply_bitcrush(audio, bits), {"bits": bits, "dither": "tpdf"}
    if kind == "resample":
        rate = int(rng.choice([8000, 16000, 22050]))
        return apply_resample(audio, rate), {"rate_hz": rate}
    if kind == "hum":
        freq = float(rng.choice([50.0, 60.0]))
        level = float(rng.uniform(0.01, 0.06))
        return add_hum(audio, freq, level), {"freq_hz": freq, "level": level}
    if kind == "tape":
        drive = float(rng.uniform(1.2, 2.5))
        return apply_tape_saturation(audio, rng, drive), {"drive": drive}
    if kind == "stack":
        return _stack_degradations(audio, rng)
    raise ValueError(f"Unbekannte Degradation: {kind}")


# ── Datensatz-Erzeugung ──────────────────────────────────────────────────────


def _collect_vocals(musdb_root: Path) -> list[Path]:
    """MUSDB18-HQ: train/<track>/vocals.wav (analog zu train_miipher_dit.py)."""
    vocals = []
    for split in ("train", "test"):
        split_dir = musdb_root / split
        if split_dir.is_dir():
            for track_dir in sorted(split_dir.iterdir()):
                vf = track_dir / "vocals.wav"
                if track_dir.is_dir() and vf.is_file():
                    vocals.append(vf)
    return vocals


def generate(
    musdb_root: Path,
    out_root: Path,
    variants_per_track: int = 8,
    seed: int = DEFAULT_SEED,
    max_tracks: int = 0,
    resume: bool = False,
) -> Path:
    """Erzeugt deterministische Paare; ``resume`` behält vollständige Tracks."""
    vocals = _collect_vocals(musdb_root)
    if max_tracks > 0:
        vocals = vocals[:max_tracks]
    if not vocals:
        raise FileNotFoundError(f"Keine MUSDB18-HQ-Vocals gefunden unter {musdb_root}/train/<track>/vocals.wav")

    manifest_path = out_root / "manifest.jsonl"
    out_root.mkdir(parents=True, exist_ok=True)
    existing_by_track: dict[tuple[str, str], list[dict]] = {}
    if resume and manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as manifest:
            for line_number, line in enumerate(manifest, 1):
                try:
                    row = json.loads(line)
                    key = (str(row["split"]), str(row["track"]))
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"Ungueltige Manifest-Zeile {line_number}: {exc}") from exc
                existing_by_track.setdefault(key, []).append(row)

    complete_tracks: set[tuple[str, str]] = set()
    for key, rows in existing_by_track.items():
        variants = [int(row["variant"]) for row in rows]
        variant_set = set(variants)
        files_exist = all(
            (out_root / relative).is_file() for row in rows for relative in (row["clean"], row["degraded"])
        )
        requested_variants_exist = set(range(variants_per_track)).issubset(variant_set)
        if len(variants) == len(variant_set) and requested_variants_exist and files_exist:
            complete_tracks.add(key)

    rows_by_track: dict[tuple[str, str], list[dict]] = {}
    n_generated = 0
    for vf in vocals:
        track = vf.parent.name
        split = vf.parent.parent.name
        key = (split, track)
        if key in complete_tracks:
            rows_by_track[key] = sorted(existing_by_track[key], key=lambda row: int(row["variant"]))
            logger.info("Track %s: %d vorhandene Paare beibehalten", track, len(rows_by_track[key]))
            continue

        split_out = out_root / split
        split_out.mkdir(parents=True, exist_ok=True)
        clean = _load_vocal(vf)
        clean = _finite_guard(clean / (float(np.max(np.abs(clean))) + 1e-10) * 0.891, "normalize")
        clean_path = split_out / f"{track}__clean.wav"
        wavfile.write(clean_path, SR_OUT, clean.T.astype(np.float32))  # (N, C)

        track_rows = []
        for vi in range(variants_per_track):
            kind = KINDS[vi % len(KINDS)]
            pair_seed = _seed_for(seed, track, vi)
            rng = np.random.default_rng(pair_seed)
            np.random.seed(pair_seed % (2**32))  # bitcrush-Dither-Saat (§G5 (GEBOTE.md))
            degraded, params = degrade_one(clean, kind, rng)
            degraded_path = split_out / f"{track}__v{vi:02d}__{kind}.wav"
            wavfile.write(degraded_path, SR_OUT, degraded.T.astype(np.float32))
            track_rows.append(
                {
                    "track": track,
                    "split": split,
                    "variant": vi,
                    "kind": kind,
                    "params": params,
                    "clean": str(clean_path.relative_to(out_root)),
                    "degraded": str(degraded_path.relative_to(out_root)),
                    "sr": SR_OUT,
                    "seed": pair_seed,
                    "channels": int(clean.shape[0]),
                }
            )
            n_generated += 1
        rows_by_track[key] = track_rows
        logger.info("Track %s: %d Paare erzeugt", track, variants_per_track)

    ordered_keys = [(vf.parent.parent.name, vf.parent.name) for vf in vocals]
    ordered_keys.extend(key for key in existing_by_track if key not in rows_by_track)
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    total_pairs = 0
    with manifest_tmp.open("w", encoding="utf-8") as manifest:
        for key in ordered_keys:
            for row in rows_by_track.get(key, existing_by_track.get(key, [])):
                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_pairs += 1
    os.replace(manifest_tmp, manifest_path)

    logger.info("Fertig: %d Paare (%d neu) → %s", total_pairs, n_generated, manifest_path)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--musdb-root", type=Path, default=Path("data/musdb18hq"))
    p.add_argument("--out", type=Path, default=Path("data/cantus_pairs"))
    p.add_argument("--variants-per-track", type=int, default=8)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Session-Master-Seed (§G5 (GEBOTE.md))")
    p.add_argument("--max-tracks", type=int, default=0, help="0 = alle Tracks")
    p.add_argument("--resume", action="store_true", help="Vollstaendige deterministische Tracks beibehalten")
    args = p.parse_args(argv)

    try:
        generate(args.musdb_root, args.out, args.variants_per_track, args.seed, args.max_tracks, args.resume)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
