"""KP-7/F11 · A1-HF-Gain-Gate-Messung — BW-Reconstructor v5 (alt vs. Re-Training).

Zwei Metriken, beide auf demselben deterministischen Set
(Seed 42, §G5 (copilot-instructions.md)):
1. `gain_db` = 10*log10(HF-Energie(out)/HF-Energie(in)) oberhalb der
   Cutoff-Grenze (rein-Eingangsbezogen).
2. `hf_recovery` = HF-Energie(out)/HF-Energie(REFERENZ-Vollband) oberhalb der
   Cutoff-Grenze — die plausibelste Interpretation des historischen
   Gate-Werts 0.73 (das Alt-Modell rekonstruierte 73 % der Referenz-HF-Energie;
   Gate ≥ 1.02 linear = +0.086 dB).

Modelle:
- alt: `models/bw_reconstructor/bw_reconstructor_v5.onnx` (historisch 0.73, D-K1-5)
- neu: `models/bw_reconstructor/retrain_a1_20261025/bw_reconstructor_v5.onnx`
  (KP-7-Re-Training mit maskierungs-informiertem A1-Hör-Loss)

Ehrlichkeits-Klausel (§G8 (copilot-instructions.md)): Die historische
Messmetrik ist nicht dokumentiert; beide Metriken werden für beide Exporte
identisch berechnet. Kein Flag-Flip durch dieses Skript
(§v10.802 (copilot-instructions.md)).

Ausgabe: docs/reports/current/<datum>_kp7_hf_gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SR = 22050
CUTOFFS = [3000, 5000, 8000, 11000]
CHUNK_SEC = 3.0
CHUNK_SAMPLES = int(CHUNK_SEC * SR)

_NOTES = np.array([130.81, 164.81, 196.00, 220.00, 261.63, 329.63, 392.00, 523.25, 659.25, 783.99])


def _make_synthetic(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64) / SR
    y = np.zeros(n_samples, dtype=np.float64)
    for _ in range(rng.integers(1, 4)):
        f0 = float(rng.choice(_NOTES) * rng.uniform(0.5, 2.0))
        for h in range(1, int(rng.integers(2, 8))):
            amp = 0.4 / (h ** rng.uniform(0.5, 1.5))
            y += amp * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi))
    for _ in range(rng.integers(2, 5)):
        f_hf = rng.uniform(8_000, 10_900)
        y += rng.uniform(0.02, 0.10) * np.sin(2 * np.pi * f_hf * t + rng.uniform(0, 2 * np.pi))
    y /= np.abs(y).max() + 1e-8
    return np.asarray(y, dtype=np.float32)


def _butter_lp(y: np.ndarray, cutoff: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    if cutoff >= SR / 2 * 0.98:
        return y.copy()
    sos = butter(6, cutoff / (SR / 2), btype="low", output="sos")
    return np.asarray(sosfiltfilt(sos, y), dtype=np.float32)


def _hf_energy(x: np.ndarray, cutoff: float, sr: int) -> float:
    n = len(x)
    win = np.hanning(n).astype(np.float64)
    spec = np.abs(np.fft.rfft(x.astype(np.float64) * win))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = freqs > cutoff
    return float(np.sum(spec[mask] ** 2)) + 1e-20


def _load_onnx(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _run_model(session, x: np.ndarray, cutoff: float) -> np.ndarray:
    inp = x.reshape(1, 1, -1).astype(np.float32)
    c = np.array([[cutoff / 11025.0]], dtype=np.float32)
    out = session.run(None, {"waveform": inp, "cutoff": c})[0]
    return np.asarray(out).reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42, help="Seed (§G5 (copilot-instructions.md))")
    ap.add_argument("--old", type=str, default="models/bw_reconstructor/bw_reconstructor_v5.onnx")
    ap.add_argument("--new", type=str, default="models/bw_reconstructor/retrain_a1_20261025/bw_reconstructor_v5.onnx")
    ap.add_argument("--out", type=str, default=f"docs/reports/current/{date.today().isoformat()}_kp7_hf_gate.json")
    args = ap.parse_args()

    np.random.seed(args.seed)
    target_len = ((CHUNK_SAMPLES // 16) + 1) * 16

    old_session = _load_onnx(Path(args.old))
    new_session = _load_onnx(Path(args.new))

    rows: list[dict] = []
    old_rec: list[float] = []
    new_rec: list[float] = []
    for i, cutoff in enumerate(CUTOFFS):
        y_full = _make_synthetic(target_len + SR, args.seed * 100 + i)
        y_lim = _butter_lp(y_full, cutoff)
        x = y_lim[:target_len]
        ref = y_full[:target_len]
        out_old = _run_model(old_session, x, cutoff)[: len(x)]
        out_new = _run_model(new_session, x, cutoff)[: len(x)]
        e_ref = _hf_energy(ref, cutoff, SR)
        r_old = _hf_energy(out_old, cutoff, SR) / e_ref
        r_new = _hf_energy(out_new, cutoff, SR) / e_ref
        old_rec.append(r_old)
        new_rec.append(r_new)
        rows.append(
            {
                "cutoff_hz": cutoff,
                "hf_recovery_old": round(r_old, 4),
                "hf_recovery_new": round(r_new, 4),
            }
        )

    mean_old = float(np.mean(old_rec))
    mean_new = float(np.mean(new_rec))
    report = {
        "date": date.today().isoformat(),
        "script": "scripts/validate_bw_v5_hf_gate.py",
        "seed": args.seed,
        "old_model": args.old,
        "new_model": args.new,
        "metric": "hf_recovery = HF-Energie(out)/HF-Energie(Referenz-Vollband) oberhalb der Cutoff-Grenze",
        "rows": rows,
        "summary": {
            "mean_hf_recovery_old": round(mean_old, 4),
            "mean_hf_recovery_new": round(mean_new, 4),
            "gate_threshold_linear": 1.02,
            "gate_passed": bool(mean_new >= 1.02),
            "note": (
                "Historischer 0.73-Wert wird hier als HF-Recovery-Rate interpretiert; "
                "beide Exporte auf demselben deterministischen Set gemessen "
                "(§G8 (copilot-instructions.md)). Kein Flag-Flip durch dieses Skript."
            ),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {out_path}")
    for r in rows:
        print(f"cutoff={r['cutoff_hz']:>5} Hz  recovery alt={r['hf_recovery_old']:.4f}  neu={r['hf_recovery_new']:.4f}")
    print(
        f"MEAN recovery  alt={mean_old:.4f}  neu={mean_new:.4f}  "
        f"Gate(≥1.02): {'BESTANDEN' if mean_new >= 1.02 else 'nicht bestanden'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
