#!/usr/bin/env python3
"""SOTA-VOCAL-INPAINT S4: Verifikation der phase_55-Kaskade auf Gesangslücken.

§SOTA-VOCAL-INPAINT (Roadmap, Abschnitt 17): VOCAL-INPAINT-S4 — der formale
Abschluss-Beweis für den Langlücken-Repair. Gates je 300-ms-Lücke:

1. **ΔSDR ≥ 0 je Segment**: Der Kaskaden-AUSGANG (erster erfolgreicher Arm in
   phase_55-Priorität: FlowMatching → Consistency → CQTdiff+ → DAC → GaCELA →
   DiffWave-Vokal → DSP-Fallback) muss je Lücke mindestens die
   Stille-Baseline erreichen (Never-worsen, §0).
2. **Sänger-Identität unverändert**: Resemblyzer-Witness (§2.35c) — cos ≥ 0,92
   zwischen Original-Fenster und repariertem Fenster (Witness-Standard der
   F-Reihe, „Resemblyzer cos ≥ 0,92 für Sänger-Identität").
3. **Determinismus (§G5 (GEBOTE.md))**: Der gewinnende Arm reproduziert sich
   bit-identisch (maxdiff == 0) bei identischem Input.

Exit-Codes: 0 = alle Gates bestanden, 1 = Gate-Verletzung, 2 = Setup-Fehler.

Report: docs/reports/current/<datum>_vocal_inpaint_s4.json

Autor: Aurik Testing Team
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
_WITNESS_COS_MIN = 0.92
_WITNESS_WIN_S = 1.9  # ≥ 1,6 s Resemblyzer-VAD-Mindestfenster


def _sdr(est: np.ndarray, ref: np.ndarray) -> float:
    err = np.asarray(est, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    den = float(np.sum(err**2))
    num = float(np.sum(np.asarray(ref, dtype=np.float64) ** 2))
    if den <= 1e-30:
        return float(np.inf) if num > 1e-30 else 0.0
    return float(10.0 * np.log10(num / den))


def _load_vocals(track_dir: Path, seconds: int, seed: int) -> np.ndarray:
    _wf = wavfile.read(str(track_dir / "vocals.wav"))
    if not isinstance(_wf, tuple) or len(_wf) < 2:
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
    rng = np.random.default_rng(seed)
    frame = _TARGET_SR // 10  # 100-ms-Energie-Frames
    energy = np.array([float(np.mean(vocals[i : i + frame] ** 2)) for i in range(0, len(vocals) - frame, frame)])
    thr = float(np.median(energy[energy > 1e-9])) * 0.5 if np.any(energy > 1e-9) else 0.0
    active_frames = [i for i, e in enumerate(energy) if e > max(thr, 1e-7)]
    if not active_frames:
        return []
    chosen = rng.choice(active_frames, size=min(n_gaps, len(active_frames)), replace=False)
    positions = sorted(int(c * frame + frame // 2) for c in chosen)
    return [p for p in positions if p + gap_len < len(vocals)]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _kaskade_fill(vocals: np.ndarray, p: int, gap_len: int) -> tuple[np.ndarray | None, str]:
    """Erster erfolgreicher phase_55-Arm in Produktions-Priorität.

    §SOTA-VOCAL-INPAINT S4 (2026-09-16): Für Gesangslücken (≥ 50 ms) versucht
    phase_55 jetzt CQTdiff+ VOR FlowMatching (A/B-Evidenz S4/Q11); danach
    Consistency → DAC → GaCELA → DiffWave-Vokal → DSP-Fallback.
    """
    from backend.core.phases.phase_55_diffusion_inpainting import (
        _adaptive_steps,
        _inpaint_gap_dsp,
        _try_consistency_model_inpainting,
        _try_cqtdiff_plus_plugin,
        _try_dac_token_inpainting,
        _try_diffwave_vocal,
        _try_flow_matching_plugin,
        _try_gacela_plugin,
    )

    end = p + gap_len
    arms: list[tuple[str, object]] = [
        ("cqtdiff", lambda: _try_cqtdiff_plus_plugin(vocals, p, end, _TARGET_SR)),
        ("flow_matching", lambda: _try_flow_matching_plugin(vocals, p, end, _TARGET_SR)),
        ("consistency", lambda: _try_consistency_model_inpainting(vocals, p, end, _TARGET_SR)),
        ("dac", lambda: _try_dac_token_inpainting(vocals, p, end, _TARGET_SR)),
        ("gacela", lambda: _try_gacela_plugin(vocals, p, end, _TARGET_SR)),
        ("diffwave_vocal", lambda: _try_diffwave_vocal(vocals, p, end, _TARGET_SR)),
    ]
    for name, arm in arms:
        try:
            t0 = time.perf_counter()
            out = arm()
            dt = time.perf_counter() - t0
            if out is not None and len(out) >= gap_len:
                print(f"    Arm {name}: OK ({dt:.1f}s)")
                return np.asarray(out[:gap_len], dtype=np.float32), name
        except Exception as _arm_exc:  # pragma: no cover - nur Protokoll
            print(f"    Arm {name}: nicht verfügbar ({_arm_exc})")
    # DSP-Fallback (produktiver Status quo, nie blockierend)
    gapped = vocals.copy()
    gapped[p:end] = 0.0
    fill = _inpaint_gap_dsp(gapped, p, end, _adaptive_steps(_GAP_S * 1000.0))
    return np.asarray(fill, dtype=np.float32), "dsp_fallback"


def _witness(original: np.ndarray, repaired: np.ndarray, sr: int, emb_fn) -> float | None:
    a = emb_fn(original, sr)
    b = emb_fn(repaired, sr)
    if a is None or b is None:
        return None
    return _cos(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def main() -> int:
    ap = argparse.ArgumentParser(description="§SOTA-VOCAL-INPAINT S4: Kaskaden-Verifikation")
    ap.add_argument("--tracks", type=int, default=3)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-witness", action="store_true", help="Resemblyzer-Witness auslassen (CPU-only-CI)")
    args = ap.parse_args()

    sys.path.insert(0, str(_ROOT))
    from plugins.resemblyzer_plugin import ResemblyzerPlugin

    emb_fn = None
    if not args.skip_witness:
        _rz = ResemblyzerPlugin()
        if _rz.available:
            emb_fn = _rz.embed
            print("Resemblyzer-Witness: verfügbar")
        else:
            print("Resemblyzer-Witness: NICHT verfügbar — Setup-Fehler")
            return 2

    tracks = sorted(p for p in _MUSDB.glob("*") if p.is_dir())
    rng = np.random.default_rng(args.seed)
    chosen = [tracks[i] for i in rng.choice(len(tracks), size=min(args.tracks, len(tracks)), replace=False)]
    chosen.sort()

    gap_len = int(_GAP_S * _TARGET_SR)
    half_win = int(_WITNESS_WIN_S * _TARGET_SR / 2)
    results: dict = {
        "seed": args.seed,
        "gap_ms": _GAP_S * 1000.0,
        "witness_cos_min": _WITNESS_COS_MIN,
        "tracks": [],
        "summary": {},
    }
    deltas: list[float] = []
    cosines: list[float] = []
    arms_used: dict[str, int] = {}
    det_ok: bool | None = None

    for track in chosen:
        print(f"Track: {track.name}")
        vocals = _load_vocals(track, args.seconds, args.seed)
        positions = _pick_gap_positions(vocals, _N_GAPS, gap_len, args.seed)
        row: dict = {"track": track.name, "gaps": []}
        for gi, p in enumerate(positions):
            orig = vocals[p : p + gap_len]
            fill, arm = _kaskade_fill(vocals, p, gap_len)
            arms_used[arm] = arms_used.get(arm, 0) + 1

            fill_sdr = _sdr(fill, orig)
            sil_sdr = _sdr(np.zeros_like(orig), orig)
            delta = fill_sdr - sil_sdr
            deltas.append(delta)

            cos = None
            if emb_fn is not None:
                lo = max(0, p - half_win)
                hi = min(len(vocals), p + gap_len + half_win)
                orig_win = np.asarray(vocals[lo:hi], dtype=np.float32)
                rep_win = np.asarray(vocals[lo:hi].copy(), dtype=np.float32)
                rep_win[p - lo : p - lo + gap_len] = fill
                cos = _witness(orig_win, rep_win, _TARGET_SR, emb_fn)
                if cos is not None:
                    cosines.append(cos)

            if gi == 0:
                fill2, arm2 = _kaskade_fill(vocals, p, gap_len)
                det_ok = arm2 == arm and bool(np.array_equal(fill, fill2))
                print(f"    Determinismus (Lücke 1): {'BIT-IDENTISCH' if det_ok else 'ABWEICHUNG'} ({arm}/{arm2})")

            row["gaps"].append(
                {
                    "pos_s": round(p / _TARGET_SR, 2),
                    "arm": arm,
                    "sdr_fill_db": round(fill_sdr, 2),
                    "sdr_silence_db": round(sil_sdr, 2),
                    "delta_vs_silence_db": round(delta, 2),
                    "witness_cos": None if cos is None else round(cos, 4),
                }
            )
            print(
                f"  gap@{p / _TARGET_SR:6.2f}s: arm={arm:<14} ΔSDR={delta:+6.2f} dB  "
                f"cos={'—' if cos is None else f'{cos:.4f}'}"
            )
        results["tracks"].append(row)

    delta_ok = bool(all(d >= 0.0 for d in deltas))
    cos_ok = bool(all(c >= _WITNESS_COS_MIN for c in cosines)) if cosines else (emb_fn is None)
    verdict_ok = delta_ok and cos_ok and bool(det_ok)

    results["summary"] = {
        "n_gaps": len(deltas),
        "mean_delta_vs_silence_db": float(np.mean(deltas)),
        "min_delta_vs_silence_db": float(min(deltas)),
        "delta_ge_0_per_segment": delta_ok,
        "witness_cos_min_measured": None if not cosines else float(min(cosines)),
        "witness_cos_ge_0_92": cos_ok if cosines else "witness_ausgelassen",
        "determinism_bit_identical": det_ok,
        "arms_used": arms_used,
        "verdict": "S4-BESTANDEN" if verdict_ok else "S4-NICHT-BESTANDEN",
    }
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = _REPORT_DIR / f"{date.today().isoformat()}_vocal_inpaint_s4.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {out}")
    print(json.dumps(results["summary"], indent=2))
    if verdict_ok:
        print("\n✅ S4-VERIFIKATION BESTANDEN — ΔSDR ≥ 0 je Segment, Sänger-Identität erhalten, deterministisch.")
        return 0
    print("\n❌ S4-Gate verletzt (Details oben).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
