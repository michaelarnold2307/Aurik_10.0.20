#!/usr/bin/env python3
"""§SOTA-HR-V1 (F3, 2026-09-15) — BigVGAN-A/B-Validierung (af + HNR).

F3-Aktivierungsvertrag: Der BigVGAN-Repair-Pfad (phase_07 HR-V1) wird nur
aktiviert, wenn er den DSP-Pfad NICHT verschlechtert:

- af (billiger click/pre_echo-Proxy aus ``artifact_freedom_guard``): Delta ≥ −0,02
- HNR (harmonic-to-noise, §0p): nicht schlechter als DSP-Pfad − 0,5 dB

Ablauf: Test-Track laden → DSP-Pfad (phase_07 ohne BigVGAN) → BigVGAN-Pfad
(synthesize_audio + additive_synthesis_gate) → beide messen → Urteil.

Exit-Codes: 0 = BigVGAN-Pfad bestehen (Flag darf gesetzt werden),
1 = verschlechtert (Flag bleibt aus), 2 = Setup-Fehler.

Autor: Aurik Testing Team
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

SR = 48000
AF_TOLERANCE = 0.02
HNR_TOLERANCE_DB = 0.5


def _load(path: str, max_s: float = 20.0) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=SR, mono=True)
    n = min(len(y), int(max_s * SR))
    return np.asarray(y[:n], dtype=np.float32)


def _af_proxy(mono: np.ndarray) -> float:
    from backend.core.dsp.artifact_freedom_guard import fast_af_proxy

    return float(fast_af_proxy(np.asarray(mono, dtype=np.float32), SR))


def _hnr_db(mono: np.ndarray) -> float:
    """HNR via Autokorrelation (Boersma 1993) — kleiner = rauschiger."""
    x = np.asarray(mono, dtype=np.float64)
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
    if ac[0] <= 1e-12:
        return 0.0
    ac /= ac[0]
    lo, hi = int(SR / 400.0), int(SR / 70.0)  # F0-Suchfenster 70–400 Hz
    lo, hi = max(1, lo), min(len(ac) - 1, hi)
    if hi <= lo:
        return 0.0
    peak = float(np.max(ac[lo:hi]))
    return float(10.0 * np.log10(max(peak / (1.0 - peak), 1e-9)))


def dsp_path(mono: np.ndarray) -> np.ndarray:
    from backend.core.defect_scanner import MaterialType
    from backend.core.phases.phase_07_harmonic_restoration import HarmonicRestorationPhase

    res = HarmonicRestorationPhase().process(mono, sample_rate=SR, material_type=MaterialType.VINYL)
    return np.asarray(res.audio, dtype=np.float32)


def bigvgan_path(mono: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    from backend.core.dsp.additive_synthesis_gate import additive_synthesis_gate
    from plugins.bigvgan_v2_plugin import synthesize_audio

    voc = synthesize_audio(mono, SR)
    if str(getattr(voc, "model_used", "none")) == "none":
        return mono, {"model_used": "none"}
    gated, rep = additive_synthesis_gate(np.asarray(voc.audio, dtype=np.float32), mono, SR, model="bigvgan_v2")
    return np.asarray(gated, dtype=np.float32), {
        "model_used": str(getattr(voc, "model_used", "none")),
        "pqs_mos": float(getattr(voc, "pqs_mos", 0.0)),
        "bands_released": int(rep.get("bands_released", 0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="§SOTA-HR-V1 BigVGAN-A/B-Validierung (af + HNR)")
    ap.add_argument("--input", type=str, default="test_audio/Testkünstlerin (Schlager) - 30 Sekunden.mp3")
    ap.add_argument("--max-s", type=float, default=20.0)
    args = ap.parse_args()

    if not Path(args.input).exists():
        print(f"Eingabe nicht gefunden: {args.input}", file=sys.stderr)
        return 2

    mono = _load(args.input, args.max_s)
    print(f"Input: {args.input} ({len(mono) / SR:.1f} s, Mono {SR} Hz)")

    dsp = dsp_path(mono)
    af_dsp = _af_proxy(dsp)
    hnr_dsp = _hnr_db(dsp)

    bvg, meta = bigvgan_path(mono)
    if meta.get("model_used") == "none":
        print("❌ BigVGAN lieferte kein Modell-Ergebnis — Validierung nicht möglich (Setup-Fehler).")
        return 2
    af_bvg = _af_proxy(bvg)
    hnr_bvg = _hnr_db(bvg)

    af_delta = af_bvg - af_dsp
    hnr_delta = hnr_bvg - hnr_dsp
    af_ok = af_delta >= -AF_TOLERANCE
    hnr_ok = hnr_delta >= -HNR_TOLERANCE_DB

    print(
        f"DSP-Pfad:       af={af_dsp:.4f}  HNR={hnr_dsp:+.2f} dB\n"
        f"BigVGAN-Pfad:   af={af_bvg:.4f}  HNR={hnr_bvg:+.2f} dB  "
        f"(PQS={meta.get('pqs_mos', 0.0):.2f}, bands_released={meta.get('bands_released', 0)})"
    )
    print(
        f"Deltas: af {af_delta:+.4f} (Toleranz −{AF_TOLERANCE}) | HNR {hnr_delta:+.2f} dB (Toleranz −{HNR_TOLERANCE_DB})"
    )
    if af_ok and hnr_ok:
        print("✅ BigVGAN-Pfad BESTEHT die F3-Validierung — Aktivierungs-Flag darf gesetzt werden.")
        return 0
    print("❌ BigVGAN-Pfad verschlechtert den DSP-Pfad — Flag bleibt aus (fail-closed, §0).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
