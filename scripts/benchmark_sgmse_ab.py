#!/usr/bin/env python3
"""3-Wege-Metrik-A/B SGMSE+ (Defizit D-K0-3): WSJ0-Sprach-Core vs. Musik-Core vs. WPE.

Baut aus einer sauberen Gesangsquelle eine deterministische Degradation
(synthetischer Nachhall + additives Rauschen, §G5 (copilot-instructions.md),
fixe Seeds), fährt die drei Wege über die kanonischen Plugin-Pfade und misst
SI-SDR gegen die saubere Quelle. Der Hörteil (Hörabnahme) bleibt ausdrücklich
offen (Hörordnungs-Sign-off) — dieses Skript liefert nur den Metrik-Beleg für
die A/B-Entscheidung, es aktiviert NICHTS (§V7 (copilot-instructions.md)).

Wege:
  1. input      — degradiertes Signal (Baseline)
  2. wsj0_core  — SGMSE+ Sprach-Core (Legacy, WSJ0/CHiME3)  via Plugin-ONNX-Pfad
  3. musik_core — SGMSE+ Musik-Finetune (MUSDB)             via Plugin-ONNX-Pfad
  4. wpe        — Nara-WPE DSP (klassischer Dereverb)       via WPE-Plugin

Der Core-Pfad wird über einen Prozess-lokalen Monkeypatch von
``plugins.sgmse_plugin._resolve_onnx_path`` gewählt (kein Quelltext-Change,
keine Flag-Änderung).

Nutzung:
  python scripts/benchmark_sgmse_ab.py \
      --clean output/scnet_ab_2026-10-06_fair/AM_Contra_-_Heart_Peripheral__gt_vocals.wav \
      --out docs/reports/current/2026-10-08_sgmse_ab_metrik.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SR = 48_000
_MUSIC_PATH = _ROOT / "models" / "sgmse_plus" / "sgmse_musik_core.onnx"
_SPEECH_PATH = _ROOT / "models" / "sgmse_plus" / "sgmse_plus_core.onnx"


def _si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant SDR in dB (mono, float32)."""
    e = np.asarray(estimate, dtype=np.float64) - float(np.mean(estimate))
    r = np.asarray(reference, dtype=np.float64) - float(np.mean(reference))
    alpha = float(np.dot(e, r) / max(float(np.dot(r, r)), 1e-12))
    target = alpha * r
    noise = e - target
    return float(10.0 * np.log10(max(float(np.dot(target, target)), 1e-12) / max(float(np.dot(noise, noise)), 1e-12)))


def _to_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        return np.mean(arr, axis=0 if arr.shape[0] <= 2 else 1)
    return arr


def _degrade(clean_2ch: np.ndarray, seed: int, snr_db: float) -> np.ndarray:
    """Deterministische Degradation: synthetischer Nachhall (45 %) + Rauschen."""
    from scipy.signal import fftconvolve

    rng = np.random.default_rng(seed)
    n = clean_2ch.shape[1]
    rir_len = int(0.35 * _SR)
    t = np.arange(rir_len) / _SR
    rir = rng.standard_normal(rir_len).astype(np.float64) * np.exp(-6.0 * t)
    rir = (rir / max(np.sqrt(np.sum(rir**2)), 1e-12)).astype(np.float32)

    wet = np.stack([fftconvolve(clean_2ch[ch], rir)[:n] for ch in range(clean_2ch.shape[0])])
    wet = wet / max(float(np.sqrt(np.mean(wet**2))), 1e-12) * float(np.sqrt(np.mean(clean_2ch**2)))
    mix = 0.55 * clean_2ch + 0.45 * wet.astype(np.float32)

    sig_rms = float(np.sqrt(np.mean(mix**2)))
    noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    noise = (rng.standard_normal(mix.shape) * noise_rms).astype(np.float32)
    return np.ascontiguousarray(mix + noise, dtype=np.float32)


def _run_sgmse(core_path: Path, degraded: np.ndarray, sigma: float, max_runtime_s: float) -> tuple[np.ndarray, str]:
    """Fährt den SGMSE+-Plugin-Pfad mit explizit gewähltem ONNX-Core."""
    import plugins.sgmse_plugin as sgmse_mod

    original = sgmse_mod._resolve_onnx_path
    sgmse_mod._resolve_onnx_path = lambda: core_path
    try:
        plugin = sgmse_mod.SGMSEPlusPlugin()
        result = plugin.enhance(degraded, _SR, sigma=sigma, max_runtime_s=max_runtime_s)
        return np.asarray(result.audio, dtype=np.float32), str(result.model_used)
    finally:
        sgmse_mod._resolve_onnx_path = original


def main() -> int:
    parser = argparse.ArgumentParser(description="SGMSE+ 3-Wege-Metrik-A/B (D-K0-3)")
    parser.add_argument("--clean", required=True, help="Saubere Gesangsquelle (WAV)")
    parser.add_argument("--out", default="docs/reports/current/2026-10-08_sgmse_ab_metrik.json")
    parser.add_argument("--start-s", type=float, default=4.0, help="Start des Ausschnitts")
    parser.add_argument("--seconds", type=float, default=8.0, help="Ausschnittlänge")
    parser.add_argument("--snr-db", type=float, default=10.0, help="Rausch-RMS-Abstand")
    parser.add_argument("--sigma", type=float, default=0.5, help="SGMSE+-Sigma (beide Cores identisch)")
    parser.add_argument("--seed", type=int, default=42, help="§G5-Degradations-Seed")
    parser.add_argument("--max-runtime-s", type=float, default=300.0, help="Runtime-Cap je SGMSE-Lauf")
    args = parser.parse_args()

    import soundfile as sf

    src, sr0 = sf.read(args.clean, always_2d=True)
    from scipy.signal import resample_poly

    audio = np.ascontiguousarray(resample_poly(src, _SR, sr0).T, dtype=np.float32)
    n0 = int(args.start_s * _SR)
    n1 = n0 + int(args.seconds * _SR)
    clean = np.ascontiguousarray(audio[:, n0:n1], dtype=np.float32)
    if clean.shape[1] < int(args.seconds * _SR):
        raise SystemExit(f"Ausschnitt zu kurz: {clean.shape[1] / _SR:.2f} s")

    degraded = _degrade(clean, seed=args.seed, snr_db=args.snr_db)
    clean_mono = _to_mono(clean)
    deg_mono = _to_mono(degraded)

    report: dict = {
        "studie": "sgmse_ab_metrik",
        "zweck": "D-K0-3 Metrik-A/B: WSJ0-Sprach-Core vs. Musik-Core vs. WPE (Hörteil offen)",
        "clean_source": str(args.clean),
        "start_s": args.start_s,
        "seconds": args.seconds,
        "snr_db": args.snr_db,
        "sigma": args.sigma,
        "seed": args.seed,
        "wege": {},
    }

    report["wege"]["input"] = {
        "si_sdr_db": round(_si_sdr(deg_mono, clean_mono), 3),
        "delta_vs_input_db": 0.0,
        "sekunden": 0.0,
        "model_used": "keine (Baseline)",
    }
    base_si = report["wege"]["input"]["si_sdr_db"]

    try:
        from plugins.wpe_plugin import get_wpe_plugin

        t0 = time.time()
        wpe_out = get_wpe_plugin().enhance(degraded, _SR)
        dt = time.time() - t0
        si = _si_sdr(_to_mono(wpe_out), clean_mono)
        report["wege"]["wpe"] = {
            "si_sdr_db": round(si, 3),
            "delta_vs_input_db": round(si - base_si, 3),
            "sekunden": round(dt, 2),
            "model_used": "nara-wpe",
        }
    except Exception as exc:  # §V6 (copilot-instructions.md): nie still
        logger.warning("§V6 (copilot-instructions.md) WPE-Weg fehlgeschlagen (%s) — Lücke bleibt sichtbar.", exc)
        report["wege"]["wpe"] = {"error": str(exc)}

    for label, core in (("wsj0_core", _SPEECH_PATH), ("musik_core", _MUSIC_PATH)):
        if not core.exists():
            report["wege"][label] = {"error": f"Core fehlt: {core}"}
            continue
        try:
            t0 = time.time()
            out, used = _run_sgmse(core, degraded, args.sigma, args.max_runtime_s)
            dt = time.time() - t0
            si = _si_sdr(_to_mono(out), clean_mono)
            report["wege"][label] = {
                "si_sdr_db": round(si, 3),
                "delta_vs_input_db": round(si - base_si, 3),
                "sekunden": round(dt, 2),
                "model_used": used,
                "core": str(core),
            }
        except Exception as exc:  # §V6 (copilot-instructions.md): nie still
            logger.warning("§V6 (copilot-instructions.md) SGMSE-Weg '%s' fehlgeschlagen (%s).", label, exc)
            report["wege"][label] = {"error": str(exc)}

    out_path = _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport: {out_path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
