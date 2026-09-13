#!/usr/bin/env python3
"""SOTA-VOCAL-INPAINT S1: Baseline-Benchmark für Gesangslücken auf MUSDB18HQ.

Misst, wie gut kurze Lücken in echten Gesangsspuren heute geschlossen werden:

- **DSP-Messlatte**: phase_55 `_inpaint_gap_dsp` (harmonischer AR-Prior +
  Diffusion, adaptive Schritte) — der aktive Pipeline-Pfad.
- **DiffWave Zero-Shot** (Sprach-Checkpoint, LJ-Speech): der HAUPTWEG-Kandidat
  für S2 (Vokal-Finetune) — hier ohne Finetune als Ausgangsbasis.
- **Stille-Baseline**: Lücke bleibt Null (Referenz für ΔSDR).

AudioLDM2-Zero-Shot entfällt: `plugins/audiolm2_plugin.py` ist im aktuellen
Checkout ein Dateisystem-Artefakt (listdir-Eintrag ohne stat) — dokumentiert
im Report.

Metrik je Lücke (300 ms, deterministisch platzierte, Seed 42):
SDR(fill, original) + ΔSDR vs. Stille-Baseline.

Report: docs/reports/current/<datum>_vocal_inpaint_baseline.json
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
_GAP_S = 0.300
_N_GAPS = 5


def _sdr(est: np.ndarray, ref: np.ndarray) -> float:
    err = est - ref
    den = float(np.sum(err**2))
    num = float(np.sum(ref**2))
    if den <= 1e-30:
        return float(np.inf) if num > 1e-30 else 0.0
    return float(10.0 * np.log10(num / den))


def _load_vocals(track_dir: Path, seconds: int, seed: int) -> np.ndarray:
    _wf = wavfile.read(str(track_dir / "vocals.wav"))
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


def _pick_gap_positions(vocals: np.ndarray, n_gaps: int, gap_len: int, seed: int) -> list[int]:
    """Deterministische Lückenpositionen in aktiven Gesangsregionen."""
    rng = np.random.default_rng(seed)
    frame = _TARGET_SR // 10  # 100-ms-Energie-Frames
    energy = np.array([float(np.mean(vocals[i : i + frame] ** 2)) for i in range(0, len(vocals) - frame, frame)])
    thr = float(np.median(energy[energy > 1e-9])) * 0.5 if np.any(energy > 1e-9) else 0.0
    active_frames = [i for i, e in enumerate(energy) if e > max(thr, 1e-7)]
    if not active_frames:
        return []
    chosen = rng.choice(active_frames, size=min(n_gaps, len(active_frames)), replace=False)
    positions = sorted(int(c * frame + frame // 2) for c in chosen)
    # Ränder: Lücke muss vollständig im Signal liegen
    return [p for p in positions if p + gap_len < len(vocals)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=3)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, str(_ROOT))
    from backend.core.phases.phase_55_diffusion_inpainting import _adaptive_steps, _inpaint_gap_dsp
    from plugins.diffwave_plugin import inpaint as diffwave_inpaint

    tracks = sorted(p for p in _MUSDB.glob("*") if p.is_dir())
    rng = np.random.default_rng(args.seed)
    chosen = [tracks[i] for i in rng.choice(len(tracks), size=min(args.tracks, len(tracks)), replace=False)]
    chosen.sort()

    gap_len = int(_GAP_S * _TARGET_SR)
    results: dict = {"seed": args.seed, "gap_ms": _GAP_S * 1000.0, "tracks": [], "summary": {}}
    agg_dsp: list[float] = []
    agg_dw: list[float] = []

    for track in chosen:
        print(f"Track: {track.name}")
        vocals = _load_vocals(track, args.seconds, args.seed)
        positions = _pick_gap_positions(vocals, _N_GAPS, gap_len, args.seed)
        row: dict = {"track": track.name, "gaps": []}
        for p in positions:
            orig = vocals[p : p + gap_len]
            n_steps = _adaptive_steps(_GAP_S * 1000.0)

            # DSP-Messlatte (aktiver Pipeline-Pfad)
            dsp_audio = vocals.copy()
            dsp_audio[p : p + gap_len] = 0.0
            t0 = time.perf_counter()
            dsp_fill = _inpaint_gap_dsp(dsp_audio, p, p + gap_len, n_steps)
            dsp_dt = time.perf_counter() - t0
            dsp_sdr = _sdr(dsp_fill, orig)

            # DiffWave Zero-Shot
            t0 = time.perf_counter()
            dw_out = diffwave_inpaint(vocals, p, p + gap_len, _TARGET_SR, n_steps=50)
            dw_dt = time.perf_counter() - t0
            if dw_out is None or len(dw_out) != len(vocals):
                dw_sdr = float("-inf")
            else:
                dw_sdr = _sdr(dw_out[p : p + gap_len], orig)

            # Stille-Baseline
            sil_sdr = _sdr(np.zeros_like(orig), orig)

            row["gaps"].append(
                {
                    "pos_s": round(p / _TARGET_SR, 2),
                    "sdr_dsp_db": round(dsp_sdr, 2),
                    "sdr_diffwave_db": round(dw_sdr, 2),
                    "sdr_silence_db": round(sil_sdr, 2),
                    "delta_dsp_db": round(dsp_sdr - sil_sdr, 2),
                    "delta_diffwave_db": round(dw_sdr - sil_sdr, 2),
                    "dsp_s": round(dsp_dt, 2),
                    "diffwave_s": round(dw_dt, 2),
                }
            )
            agg_dsp.append(dsp_sdr - sil_sdr)
            agg_dw.append(dw_sdr - sil_sdr)
            print(
                f"  gap@{p / _TARGET_SR:6.2f}s: DSP SDR={dsp_sdr:+6.2f} (Δ{sil_sdr - dsp_sdr:+.1f} vs Stille)  "
                f"DiffWave SDR={dw_sdr:+6.2f} (Δ{sil_sdr - dw_sdr:+.1f})"
            )
        results["tracks"].append(row)

    results["summary"] = {
        "audio_ldm2_zero_shot": "entfällt — plugins/audiolm2_plugin.py ist ein Dateisystem-Artefakt (listdir ohne stat)",
        "dsp_mean_delta_vs_silence_db": float(np.mean(agg_dsp)),
        "dsp_min_delta_vs_silence_db": float(min(agg_dsp)),
        "dsp_better_than_silence_always": bool(all(v > 0 for v in agg_dsp)),
        "diffwave_mean_delta_vs_silence_db": float(np.mean(agg_dw)),
        "diffwave_min_delta_vs_silence_db": float(min(agg_dw)),
        "diffwave_better_than_silence_always": bool(all(v > 0 for v in agg_dw)),
        "diffwave_minus_dsp_mean_db": float(np.mean([w - d for w, d in zip(agg_dw, agg_dsp)])),
    }
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = _REPORT_DIR / f"{date.today().isoformat()}_vocal_inpaint_baseline.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {out}")
    print(json.dumps(results["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
