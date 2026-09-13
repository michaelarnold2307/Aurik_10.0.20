#!/usr/bin/env python3
"""SOTA-ML-V4: UTMOS-Musik-MOS-Validierung — Delta-Kalibrierung auf MUSDB-Paaren.

Kein Finetune möglich (UTMOSv2 ist ein fertiges MOS-Prädiktions-Ensemble) —
validiert wird nur, ob UTMOS auf Musik die RICHTUNG erkennt
(MOS(original) > MOS(degradiert)) und wie groß das Delta pro Degradationsart
ausfällt. Daraus ergibt sich die Schwelle für den Einsatz im Aurik-Audio-Modus.

Degradationsstufen je Track (30 s, deterministisch, Seed 42 — gleiche
Track-Auswahl wie scripts/benchmark_bwe_candidates.py):

- ref        : Original (Referenz)
- band12     : Bandlimit 12 kHz (BWE-Baseline)
- band16     : Bandlimit 16 kHz
- noise10    : weißes Rauschen, SNR 10 dB
- noise0     : weißes Rauschen, SNR 0 dB (stark hörbar)

Report: docs/reports/current/<datum>_utmos_music_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

_ROOT = Path(__file__).resolve().parent.parent
_MUSDB = _ROOT / "data" / "musdb18hq" / "test"
_REPORT_DIR = _ROOT / "docs" / "reports" / "current"
_TARGET_SR = 48000


def _load_mixture(track_dir: Path, seconds: int, seed: int) -> np.ndarray:
    _wf = wavfile.read(str(track_dir / "mixture.wav"))
    if not isinstance(_wf, tuple) or len(_wf) < 2:  # Bug 12: Index-basiert statt Unpacking
        raise ValueError("wavfile.read() ohne (sr, data)-Tupel")
    sr = int(_wf[0])
    wav = np.asarray(_wf[1])
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / 32768.0
    else:
        wav = wav.astype(np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != _TARGET_SR:
        g = int(np.gcd(sr, _TARGET_SR))
        wav = resample_poly(wav, _TARGET_SR // g, sr // g).astype(np.float32)
    n = int(seconds * _TARGET_SR)
    rng = np.random.default_rng(seed)
    if len(wav) > n + _TARGET_SR:
        start = int(rng.integers(_TARGET_SR, len(wav) - n - _TARGET_SR))
        wav = wav[start : start + n]
    else:
        wav = wav[:n]
    peak = float(np.max(np.abs(wav))) if wav.size else 1.0
    return (wav / peak).astype(np.float32) if peak > 0 else wav


def _bandlimit(x: np.ndarray, low_sr: int) -> np.ndarray:
    g = int(np.gcd(_TARGET_SR, low_sr))
    down = resample_poly(x, low_sr // g, _TARGET_SR // g)
    up = resample_poly(down, _TARGET_SR // g, low_sr // g)
    if len(up) > len(x):
        up = up[: len(x)]
    elif len(up) < len(x):
        up = np.pad(up, (0, len(x) - len(up)))
    return up.astype(np.float32)


def _add_noise(x: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(x.size).astype(np.float32)
    sig_pow = float(np.mean(x**2))
    noi_pow = float(np.mean(noise**2))
    scale = np.sqrt(sig_pow / (noi_pow * 10 ** (snr_db / 10.0)))
    return (x + scale * noise).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=3)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, str(_ROOT))
    from plugins.utmos_plugin import estimate_mos

    tracks = sorted(p for p in _MUSDB.glob("*") if p.is_dir())
    if not tracks:
        print(f"MUSDB18HQ-Testset fehlt: {_MUSDB}")
        return 2
    rng = np.random.default_rng(args.seed)
    chosen = [tracks[i] for i in rng.choice(len(tracks), size=min(args.tracks, len(tracks)), replace=False)]
    chosen.sort()

    variants = ("ref", "band12", "band16", "noise10", "noise0")
    results: dict = {"seed": args.seed, "tracks": [], "summary": {}}
    deltas: dict[str, list[float]] = {v: [] for v in variants if v != "ref"}

    for track in chosen:
        print(f"Track: {track.name}")
        ref = _load_mixture(track, args.seconds, args.seed)
        audio = {
            "ref": ref,
            "band12": _bandlimit(ref, 12000),
            "band16": _bandlimit(ref, 16000),
            "noise10": _add_noise(ref, 10.0, args.seed),
            "noise0": _add_noise(ref, 0.0, args.seed),
        }
        row: dict = {"track": track.name, "mos": {}, "delta_vs_ref": {}}
        for v in variants:
            t0 = time.perf_counter()
            res = estimate_mos(audio[v], _TARGET_SR)
            dt = time.perf_counter() - t0
            row["mos"][v] = round(float(res.mos), 4)
            if v != "ref":
                d = row["mos"]["ref"] - row["mos"][v]
                row["delta_vs_ref"][v] = round(d, 4)
                deltas[v].append(d)
            print(f"  {v:8s}: MOS={res.mos:.3f} ({res.model_used}) [{dt:.1f}s]")
        results["tracks"].append(row)

    results["summary"] = {
        v: {
            "mean_delta": float(np.mean(deltas[v])),
            "min_delta": float(min(deltas[v])),
            "direction_correct": all(d > 0 for d in deltas[v]),
        }
        for v in deltas
    }
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = _REPORT_DIR / f"{date.today().isoformat()}_utmos_music_calibration.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {out}")
    print(json.dumps(results["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
