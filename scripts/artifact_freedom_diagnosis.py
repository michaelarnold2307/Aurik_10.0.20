#!/usr/bin/env python3
"""§SOTA-P1 (Residual-Artefakt-Diagnose) — artifact_freedom pro Phase.

Session-Ertrag 2026-09-15, P1: „artifact_freedom pro Phase auf realem Material
messen (welche Phase senkt af unter 0,95?); gezielte Never-worsen-Fixes statt
Raten." Dieses Skript fährt die reparierenden/enhancenden Phasen in
Pipeline-Reihenfolge über einen echten Track (Default: „Testkünstlerin (Schlager) – 30
Sekunden.mp3") und misst nach jeder Phase ``ArtifactDetector.overall_score``
(af). Phasen, die af unter 0,95 drücken, werden als Kandidaten markiert.

Nutzung:

    python3 -B scripts/artifact_freedom_diagnosis.py [--input <pfad>] [--max-s 20] [--out <pfad>]

Deterministisch (kein RNG im Messpfad); ML-lastige Phasen (03, 55) sind bewusst
ausgenommen (Diagnose der DSP-/Guard-Kette; ML-Artefakte werden über die
Hörordnungs-Witnesses abgedeckt).

Autor: Aurik Testing Team
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Repo-Konvention (wie scripts/benchmark_bwe_candidates.py): Projekt-Root in
# sys.path, damit ``import backend`` auch bei direktem Skript-Aufruf funktioniert.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

SR = 48000

# (Anzeigename, Modul, Klasse) — Pipeline-Reihenfolge, DSP-lastige Teilmenge.
PHASES: list[tuple[str, str, str]] = [
    ("phase_01_click_removal", "phase_01_click_removal", "ClickRemovalPhase"),
    ("phase_04_eq_correction", "phase_04_eq_correction", "EQCorrectionPhase"),
    ("phase_07_harmonic_restoration", "phase_07_harmonic_restoration", "HarmonicRestorationPhase"),
    ("phase_08_transient_preservation", "phase_08_transient_preservation", "TransientPreservationPhase"),
    ("phase_13_stereo_enhancement", "phase_13_stereo_enhancement", "StereoEnhancementPhaseV2"),
    ("phase_15_stereo_balance", "phase_15_stereo_balance", "StereoBalancePhaseV2"),
    ("phase_16_final_eq", "phase_16_final_eq", "FinalEQ"),
    ("phase_17_mastering_polish", "phase_17_mastering_polish", "MasteringPolishPhase"),
    ("phase_19_de_esser", "phase_19_de_esser", "DeEsserPhase"),
    ("phase_23_spectral_repair", "phase_23_spectral_repair", "SpectralRepair"),
    ("phase_27_click_pop_removal", "phase_27_click_pop_removal", "ClickPopRemoval"),
    ("phase_28_surface_noise_profiling", "phase_28_surface_noise_profiling", "SurfaceNoiseProfiling"),
    ("phase_33_stereo_width_limiter", "phase_33_stereo_width_limiter", "StereoWidthLimiterPhaseV2"),
    ("phase_34_mid_side_processing", "phase_34_mid_side_processing", "MidSideProcessing"),
    ("phase_36_transient_shaper", "phase_36_transient_shaper", "TransientShaper"),
    ("phase_37_bass_enhancement", "phase_37_bass_enhancement", "BassEnhancement"),
    ("phase_38_presence_boost", "phase_38_presence_boost", "PresenceBoost"),
    ("phase_39_air_band_enhancement", "phase_39_air_band_enhancement", "AirBandEnhancement"),
    ("phase_40_loudness_normalization", "phase_40_loudness_normalization", "LoudnessNormalizationPhase"),
    ("phase_46_spatial_enhancement", "phase_46_spatial_enhancement", "SpatialEnhancementPhase"),
    ("phase_48_stereo_width_enhancer", "phase_48_stereo_width_enhancer", "StereoWidthEnhancerPhase"),
    ("phase_50_spectral_repair", "phase_50_spectral_repair", "SpectralRepairPhase"),
    ("phase_59_modulation_noise_reduction", "phase_59_modulation_noise_reduction", "ModulationNoiseReductionPhase"),
    ("phase_64_tape_splice_repair", "phase_64_tape_splice_repair", "TapeSpliceRepairPhase"),
    (
        "phase_65_vocal_naturalness_restoration",
        "phase_65_vocal_naturalness_restoration",
        "VocalNaturalnessRestorationPhase",
    ),
]

AF_FLOOR = 0.95


def load_audio(path: str, max_s: float) -> tuple[np.ndarray, int]:
    """Lädt Mono 48 kHz float32 aus WAV/MP3 (librosa), begrenzt auf max_s."""
    import librosa

    y, orig_sr = librosa.load(path, sr=SR, mono=True)
    n = min(len(y), int(max_s * SR))
    y = y[:n].astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(y, dtype=np.float32), int(orig_sr)


def measure_af(audio: np.ndarray) -> float:
    from backend.core.artifact_detector import ArtifactDetector

    return float(ArtifactDetector(SR).scan(audio).overall_score)


def run_diagnosis(audio: np.ndarray) -> dict:
    from backend.core.defect_scanner import MaterialType

    report: dict = {
        "input_af": round(measure_af(audio), 4),
        "phases": [],
        "af_floor": AF_FLOOR,
    }
    current = np.asarray(audio, dtype=np.float32).copy()
    for name, module, cls in PHASES:
        try:
            mod = importlib.import_module(f"backend.core.phases.{module}")
            phase = getattr(mod, cls)()
            t0 = time.monotonic()
            # Signatur-Vielfalt bedienen: die meisten Phasen nehmen
            # material_type als Kwarg, manche (z. B. phase_17) verlangen
            # material POSITIONAL.
            try:
                result = phase.process(current, sample_rate=SR, material_type=MaterialType.VINYL)
            except TypeError:
                result = phase.process(current, SR, MaterialType.VINYL)
            wall_ms = (time.monotonic() - t0) * 1000.0
            out = np.asarray(result.audio, dtype=np.float32)
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            if out.ndim == 2:
                out = out.mean(axis=0) if out.shape[0] <= 2 else out.mean(axis=1)
            af = measure_af(out)
            # Delta gegen die letzte ERFOLGREICHE Messung (Fehler-Phasen werden übersprungen)
            _prev_af = report["input_af"]
            for _e in reversed(report["phases"]):
                if _e["af_after"] is not None:
                    _prev_af = _e["af_after"]
                    break
            entry = {
                "phase": name,
                "af_after": round(af, 4),
                "af_delta": round(af - _prev_af, 4),
                "wall_ms": round(wall_ms, 1),
                "below_floor": af < AF_FLOOR,
                "error": None,
            }
            report["phases"].append(entry)
            current = out
            print(
                f"[{name:>38}] af={af:.4f} (Δ{entry['af_delta']:+.4f}) {entry['wall_ms']:7.0f} ms{'  <-- UNTER 0.95' if entry['below_floor'] else ''}"
            )
        except Exception as exc:
            entry = {
                "phase": name,
                "af_after": None,
                "af_delta": None,
                "wall_ms": 0.0,
                "below_floor": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            report["phases"].append(entry)
            print(f"[{name:>38}] FEHLER: {type(exc).__name__}: {exc}")
    report["worst_phase"] = None
    _worst_af: float | None = None
    for entry in report["phases"]:
        if entry["af_after"] is None:
            continue
        if _worst_af is None or entry["af_after"] < _worst_af:
            _worst_af = float(entry["af_after"])
            report["worst_phase"] = entry["phase"]
    report["min_af"] = min((e["af_after"] for e in report["phases"] if e["af_after"] is not None), default=None)
    report["candidates_below_floor"] = [e["phase"] for e in report["phases"] if e["below_floor"]]
    return report


def compute_fail_delta_violations(report: dict, fail_delta: float) -> list[dict]:
    """CI-Gate-Helfer: alle Phasen mit af_delta < -fail_delta (delta-basiert)."""
    return [
        {"phase": e["phase"], "af_delta": round(e["af_delta"], 4)}
        for e in report.get("phases", [])
        if e.get("af_delta") is not None and e["af_delta"] < -fail_delta
    ]


def compute_hot_phases(report: dict, input_seconds: float, hot_rt_threshold: float = 0.5) -> list[dict]:
    """§SOTA-P0-1 (2026-09-15): Hot-Phase-Analyse — rt_factor je Phase aus wall_ms.

    ``hot_rt_threshold`` = RT-Faktor, ab dem eine Phase als Hot-Phase gilt
    (Default 0,5× RT — alles darüber dominiert das End-to-End-Budget).
    """
    hot: list[dict] = []
    for e in report.get("phases", []):
        wall_ms = e.get("wall_ms")
        if wall_ms is None or input_seconds <= 0:
            continue
        rt = float(wall_ms) / 1000.0 / float(input_seconds)
        e["rt_factor"] = round(rt, 3)
        if rt >= float(hot_rt_threshold):
            hot.append({"phase": e["phase"], "rt_factor": round(rt, 3), "wall_ms": round(float(wall_ms), 1)})
    hot.sort(key=lambda h: -h["rt_factor"])
    return hot


def main() -> int:
    parser = argparse.ArgumentParser(description="§SOTA-P1 artifact_freedom-Diagnose je Phase")
    parser.add_argument("--input", type=str, default="test_audio/Testkünstlerin (Schlager) - 30 Sekunden.mp3")
    parser.add_argument("--max-s", type=float, default=20.0)
    parser.add_argument("--out", type=str, default="output/artifact_freedom_diagnosis/af_diagnosis_report.json")
    parser.add_argument(
        "--fail-delta",
        type=float,
        default=None,
        help="CI-Gate: Exit 3, wenn eine Phase af um mehr als diesen Betrag absenkt (delta-basiert, Never-worsen)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Eingabe nicht gefunden: {args.input}", file=sys.stderr)
        return 2

    audio, _ = load_audio(args.input, args.max_s)
    print(f"Input: {args.input} ({len(audio) / SR:.1f} s, Mono {SR} Hz)")
    report = run_diagnosis(audio)
    report["input_file"] = args.input
    report["input_seconds"] = round(len(audio) / SR, 2)
    _hot = compute_hot_phases(report, len(audio) / SR)
    report["hot_phases"] = _hot
    if _hot:
        print("\nHot-Phasen (rt_factor ≥ 0,5× RT):")
        for _h in _hot:
            print(f"   {_h['phase']}: {_h['rt_factor']:.2f}× RT ({_h['wall_ms']:.0f} ms)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nMin-af: {report['min_af']} | Kandidaten < {AF_FLOOR}: {report['candidates_below_floor'] or 'keine'}")
    print(f"Report: {args.out}")
    if args.fail_delta is not None:
        _violations = compute_fail_delta_violations(report, args.fail_delta)
        report["fail_delta"] = args.fail_delta
        report["fail_delta_violations"] = _violations
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        if _violations:
            print(f"\n❌ --fail-delta {args.fail_delta}: {len(_violations)} Phase(n) unterschreiten die Schwelle:")
            for _v in _violations:
                print(f"   {_v['phase']}: Δ{_v['af_delta']:+.4f}")
            return 3
        print(f"\n✅ --fail-delta {args.fail_delta}: keine Phase unterschreitet die Schwelle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
