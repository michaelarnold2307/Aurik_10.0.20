#!/usr/bin/env python3
"""SOTA-ML-V3-VORAB: MP-SENet-3-Song-Vorher/Nachher-Benchmark auf MUSDB18HQ.

MP-SENet ist ein SPRACH-Enhancer (Noise-Reduction). Auf Musik (out-of-domain)
erwartet die Roadmap −5,9…−8,5 dB SDR-Veränderung. Dieser Benchmark misst die
Schädigung auf 3 MUSDB-Mixtures (30 s, Seed 42) und liefert damit die
Entscheidungsgrundlage Finetune vs. De-Wiring (§V7 (copilot-instructions.md)).

Metrik je 1-s-Segment: SDR(out, input) — perfekter Passthrough = +inf,
starke Veränderung = niedrig. Zusätzlich SegSNR und SNR-Improvement des
Plugins (dessen interne Schätzung ist für Sprache kalibriert).

Report: docs/reports/current/<datum>_mpsenet_music_damage.json
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


def _sdr(est: np.ndarray, ref: np.ndarray) -> float:
    err = est - ref
    den = float(np.sum(err**2))
    num = float(np.sum(ref**2))
    if den <= 1e-30:
        return float(np.inf) if num > 1e-30 else 0.0
    return float(10.0 * np.log10(num / den))


def _load_mixture(track_dir: Path, seconds: int, seed: int) -> np.ndarray:
    _wf = wavfile.read(str(track_dir / "mixture.wav"))
    if not isinstance(_wf, tuple) or len(_wf) < 2:  # Bug 12: Index-basiert statt Unpacking
        raise ValueError("wavfile.read() ohne (sr, data)-Tupel")
    sr = int(_wf[0])
    wav = np.asarray(_wf[1])
    wav = (wav.astype(np.float32) / 32768.0) if wav.dtype == np.int16 else wav.astype(np.float32)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=3)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, str(_ROOT))
    from plugins.mp_senet_plugin import get_mp_senet_plugin

    tracks = sorted(p for p in _MUSDB.glob("*") if p.is_dir())
    if not tracks:
        print(f"MUSDB18HQ-Testset fehlt: {_MUSDB}")
        return 2
    rng = np.random.default_rng(args.seed)
    chosen = [tracks[i] for i in rng.choice(len(tracks), size=min(args.tracks, len(tracks)), replace=False)]
    chosen.sort()

    plugin = get_mp_senet_plugin()
    results: dict = {"seed": args.seed, "tracks": [], "summary": {}}
    all_d: list[float] = []
    seg = _TARGET_SR  # 1 s

    for track in chosen:
        print(f"Track: {track.name}")
        x = _load_mixture(track, args.seconds, args.seed)
        t0 = time.perf_counter()
        res = plugin.enhance(x, _TARGET_SR)
        dt = time.perf_counter() - t0
        out = np.asarray(res.audio, dtype=np.float32) if res.audio is not None else None
        if out is None or len(out) < len(x):
            print(f"  FEHLER: {res.fail_reason}")
            return 3
        out = out[: len(x)]
        seg_sdr = [_sdr(out[i : i + seg], x[i : i + seg]) for i in range(0, len(x) - seg + 1, seg)]
        seg_sdr = [v for v in seg_sdr if np.isfinite(v)]
        all_d.extend(seg_sdr)
        print(
            f"  mean SDR(out,in)={np.mean(seg_sdr):+.2f} dB  min={min(seg_sdr):+.2f}  "
            f"plugin_snr_imp={res.snr_improvement_db:+.2f} dB  ({dt:.1f}s)"
        )
        results["tracks"].append(
            {
                "track": track.name,
                "mean_sdr_out_in_db": float(np.mean(seg_sdr)),
                "min_sdr_out_in_db": float(min(seg_sdr)),
                "plugin_snr_improvement_db": float(res.snr_improvement_db),
                "model_used": res.model_used,
            }
        )

    results["summary"] = {
        "mean_sdr_out_in_db": float(np.mean(all_d)),
        "min_sdr_out_in_db": float(min(all_d)),
        "max_sdr_out_in_db": float(max(all_d)),
        "passthrough_would_be_better": float(np.mean(all_d)) < 0.0,
    }
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_p = _REPORT_DIR / f"{date.today().isoformat()}_mpsenet_music_damage.json"
    out_p.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {out_p}")
    print(json.dumps(results["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
