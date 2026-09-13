#!/usr/bin/env python3
"""SOTA-ML-V1 VORAB: AERO-vs-FlashSR-Kandidaten-Benchmark auf MUSDB18HQ-Musik.

Bandbreiten-Erweiterer (BWE) für die Rolle „Hochband-Rekonstruktion":

- **AERO** (slp-rl/aero, BLSTM, 12 kHz → 48 kHz, ONNX) — Challenger.
- **FlashSR** (NVSR, 16 kHz → 48 kHz, ONNX) — Incumbent (§V7 (copilot-instructions.md): eine Lösung pro Rolle).

Drei Vergleiche (je 1-s-Segment, Metriken SDR + SegSNR):

1. **AERO nativ**: 12-kHz-Quelle → AERO vs. 12-kHz-Baseline (ΔSDR/ΔSegSNR).
2. **FlashSR nativ**: 16-kHz-Quelle → FlashSR vs. 16-kHz-Baseline.
3. **Gleiche 12-kHz-Quelle**: AERO vs. FlashSR direkt (gemeinsamer Nenner).

Never-worsen-Gate (Akzeptanz wie EAR-VAE): min ΔSDR ≥ 0 je Segment.

Determinismus (§G5 (GEBOTE.md)): feste Track-Auswahl (Seed 42), keine Zeitquellen in
Entscheidungen. Report: docs/reports/current/<datum>_bwe_candidates.json

Usage:
    python3 scripts/benchmark_bwe_candidates.py [--tracks 3] [--seconds 30] [--seed 42]
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
_SEG_S = 1.0
_FRAME_S = 0.020  # SegSNR-Frame


def _sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """SDR in dB gegen die Referenz (float, sonst +inf/NaN wird zu -inf)."""
    err = est - ref
    den = float(np.sum(err**2))
    num = float(np.sum(ref**2))
    if den <= 1e-30:
        return float(np.inf) if num > 1e-30 else 0.0
    return float(10.0 * np.log10(num / den))


def _segsnr(est: np.ndarray, ref: np.ndarray) -> float:
    """Segmental SNR (20-ms-Frames, stille Frames übersprungen)."""
    n = _FRAME_S * _TARGET_SR
    n = int(n)
    snrs: list[float] = []
    for i in range(0, len(ref) - n + 1, n):
        rf = ref[i : i + n]
        ef = est[i : i + n] - rf
        num = float(np.sum(rf**2))
        den = float(np.sum(ef**2))
        if num < 1e-10:  # Stille-Frame überspringen
            continue
        if den <= 1e-30:
            snrs.append(60.0)  # Deckel statt +inf
        else:
            snrs.append(float(10.0 * np.log10(num / den)))
    return float(np.mean(snrs)) if snrs else float("-inf")


def _segmentwise(fn: callable, est: np.ndarray, ref: np.ndarray) -> list[float]:
    seg = int(_SEG_S * _TARGET_SR)
    out: list[float] = []
    for i in range(0, len(ref) - seg + 1, seg):
        out.append(fn(est[i : i + seg], ref[i : i + seg]))
    return out


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
    # deterministische Startposition (Mitte des Tracks)
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
    """48 kHz → low_sr → 48 kHz (Anti-Alias-Lowpass, Länge stabilisiert)."""
    g = int(np.gcd(_TARGET_SR, low_sr))
    down = resample_poly(x, low_sr // g, _TARGET_SR // g)
    up = resample_poly(down, _TARGET_SR // g, low_sr // g)
    if len(up) > len(x):
        up = up[: len(x)]
    elif len(up) < len(x):
        up = np.pad(up, (0, len(x) - len(up)))
    return up.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=3)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, str(_ROOT))
    from plugins.aero_plugin import get_aero_plugin
    from plugins.flashsr_plugin import get_flashsr_plugin

    tracks = sorted(p for p in _MUSDB.glob("*") if p.is_dir())
    if not tracks:
        print(f"MUSDB18HQ-Testset fehlt: {_MUSDB}")
        return 2
    rng = np.random.default_rng(args.seed)
    chosen = [tracks[i] for i in rng.choice(len(tracks), size=min(args.tracks, len(tracks)), replace=False)]
    chosen.sort()

    aero = get_aero_plugin(device="cpu")
    flash = get_flashsr_plugin()
    results: dict = {"seed": args.seed, "tracks": [], "summary": {}}
    agg = {"aero_native": [], "flash_native": [], "shared12": []}

    for track in chosen:
        print(f"Track: {track.name}")
        ref = _load_mixture(track, args.seconds, args.seed)
        base12 = _bandlimit(ref, 12000)
        base16 = _bandlimit(ref, 16000)

        # --- Vergleich 1: AERO nativ (12-kHz-Quelle) -----------------------
        src12 = resample_poly(ref, 12000 // 480, _TARGET_SR // 480)  # 48k → 12k
        t0 = time.perf_counter()
        aero_out = aero.enhance(src12.astype(np.float32), 12000)
        aero_dt = time.perf_counter() - t0
        if aero_out is None or len(aero_out) < len(ref):
            print("  AERO: FEHLGESCHLAGEN/None")
            return 3
        aero_out = np.asarray(aero_out, dtype=np.float32)[: len(ref)]
        d_sdr_1 = [a - b for a, b in zip(_segmentwise(_sdr, aero_out, ref), _segmentwise(_sdr, base12, ref))]
        d_segsnr_1 = [a - b for a, b in zip(_segmentwise(_segsnr, aero_out, ref), _segmentwise(_segsnr, base12, ref))]
        agg["aero_native"].append(min(d_sdr_1))
        print(
            f"  AERO   nativ: meanΔSDR={np.mean(d_sdr_1):+.2f}  minΔSDR={min(d_sdr_1):+.2f}  "
            f"meanΔSegSNR={np.mean(d_segsnr_1):+.2f}  ({aero_dt:.1f}s)"
        )

        # --- Vergleich 2: FlashSR nativ (16-kHz-Quelle) --------------------
        t0 = time.perf_counter()
        flash_out = flash.process(base16, _TARGET_SR, target_sr=_TARGET_SR)
        flash_dt = time.perf_counter() - t0
        flash_out = np.asarray(flash_out, dtype=np.float32)
        if flash_out.ndim == 2:
            flash_out = flash_out[0]
        flash_out = flash_out[: len(ref)]
        d_sdr_2 = [a - b for a, b in zip(_segmentwise(_sdr, flash_out, ref), _segmentwise(_sdr, base16, ref))]
        d_segsnr_2 = [a - b for a, b in zip(_segmentwise(_segsnr, flash_out, ref), _segmentwise(_segsnr, base16, ref))]
        agg["flash_native"].append(min(d_sdr_2))
        print(
            f"  FlashSR nativ: meanΔSDR={np.mean(d_sdr_2):+.2f}  minΔSDR={min(d_sdr_2):+.2f}  "
            f"meanΔSegSNR={np.mean(d_segsnr_2):+.2f}  ({flash_dt:.1f}s)"
        )

        # --- Vergleich 3: gleiche 12-kHz-Quelle (direkt A/B) ---------------
        t0 = time.perf_counter()
        flash_out12 = flash.process(base12, _TARGET_SR, target_sr=_TARGET_SR)
        flash12_dt = time.perf_counter() - t0
        flash_out12 = np.asarray(flash_out12, dtype=np.float32)
        if flash_out12.ndim == 2:
            flash_out12 = flash_out12[0]
        flash_out12 = flash_out12[: len(ref)]
        sdr_aero12 = _sdr(aero_out, ref)
        sdr_flash12 = _sdr(flash_out12, ref)
        sdr_base12 = _sdr(base12, ref)
        agg["shared12"].append(sdr_aero12 - sdr_flash12)
        print(
            f"  Shared-12k: SDR AERO={sdr_aero12:+.2f}  FlashSR={sdr_flash12:+.2f}  "
            f"Baseline={sdr_base12:+.2f}  (Δ={sdr_aero12 - sdr_flash12:+.2f} dB, FlashSR {flash12_dt:.1f}s)"
        )

        results["tracks"].append(
            {
                "track": track.name,
                "aero_native": {
                    "mean_dSDR": float(np.mean(d_sdr_1)),
                    "min_dSDR": float(min(d_sdr_1)),
                    "mean_dSegSNR": float(np.mean(d_segsnr_1)),
                    "wall_s": round(aero_dt, 2),
                },
                "flash_native": {
                    "mean_dSDR": float(np.mean(d_sdr_2)),
                    "min_dSDR": float(min(d_sdr_2)),
                    "mean_dSegSNR": float(np.mean(d_segsnr_2)),
                    "wall_s": round(flash_dt, 2),
                },
                "shared12": {"sdr_aero_db": sdr_aero12, "sdr_flashsr_db": sdr_flash12, "sdr_baseline_db": sdr_base12},
            }
        )

    results["summary"] = {
        "aero_native_min_dSDR_over_tracks": min(agg["aero_native"]),
        "flash_native_min_dSDR_over_tracks": min(agg["flash_native"]),
        "aero_never_worsen": all(v >= 0 for v in agg["aero_native"]),
        "flash_never_worsen": all(v >= 0 for v in agg["flash_native"]),
        "shared12_aero_minus_flash_min_db": min(agg["shared12"]),
        "shared12_aero_minus_flash_mean_db": float(np.mean(agg["shared12"])),
    }
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = _REPORT_DIR / f"{date.today().isoformat()}_bwe_candidates.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {out}")
    print(json.dumps(results["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
