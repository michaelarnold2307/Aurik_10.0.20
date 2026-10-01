#!/usr/bin/env python3
"""Wohlklang-A/B-Validierung auf echten Songs (§Hörordnung §1–§3).

Beweist den Effekt des Wohlklang-Optimums-Fixes (2026-09-26) hörbar auf
echtem Corpus-Material: identischer Song, identischer Pfad
(`UnifiedRestorerV3.restore` — derselbe Produktionskern wie
`real_audio_execution_golden_gate`), einzige Variable = Zielfunktion der
Phasen-Stärken-Regelung (§v10.600 ClosedLoop):

* **A_alt**: Zielfunktion von vor dem Fix (MR-STFT-Distanz × tanh-Richtung
  — blind in der Praxis: Δ ≈ 1e-5 ⇒ „hold", Stärken blieben Default).
* **B_wohlklang**: `wohlklang_objective_delta` (HPE-Delta = psychoakustische
  Angenehmheit, Hör-Invarianten als Wächter).

Gemessen wird HPE (human_pleasantness_estimator) für Eingang/A/B; die WAVs
liegen zum Hinhören bereit. Pro Lauf ein eigener Worker-Prozess
(Zustands-Isolation §V8 (copilot-instructions.md), resumierbar — vorhandene Ergebnisse bleiben stehen).

Usage:
    python scripts/run_wohlklang_ab_validation.py [--mode quality|fast]
        [--cases corpus/..., ...] [--max-seconds 15] [--workers 1]

Ausgabe: reports/wohlklang_ab/<run_id>/ mit report.json, HINHOEREN.txt und
je Fall input/A_alt/B_wohlklang WAV-Dateien.

Autor: Aurik 10 — Evidenz-Welle Wohlklang-Optimum
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MATERIAL_BY_FAMILY = {
    "vinyl": "vinyl",
    "tape": "reel_tape",
    "reel_tape": "reel_tape",
    "cassette": "cassette",
    "shellac": "shellac",
    "digital": "digital",
    "reverb": "vinyl",
}


def _default_cases() -> list[Path]:
    """Deterministische Auswahl: erste beschädigte Datei je Materialfamilie + vinyl/tape Zweite."""
    corpus = REPO_ROOT / "corpus"
    picks: list[Path] = []
    for family in ("vinyl", "tape", "cassette", "shellac", "digital"):
        damaged = sorted((corpus / family / "damaged").glob("*.wav"))
        if damaged:
            picks.append(damaged[0])
    for family in ("vinyl", "tape"):
        damaged = sorted((corpus / family / "damaged").glob("*.wav"))
        if len(damaged) > 1:
            picks.append(damaged[len(damaged) // 2])
    return picks


def _material_for(path: Path) -> str:
    for family, material in MATERIAL_BY_FAMILY.items():
        if f"/{family}/" in str(path):
            return material
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# A-Bedingung: Zielfunktion VOR dem Wohlklang-Fix (byte-gleich zum alten
# closed_loop_calibrator.measure_phase_quality_delta) — nur im Worker A aktiv.
# ═══════════════════════════════════════════════════════════════════════════


def _legacy_metric_mono(audio: np.ndarray) -> np.ndarray:
    arr: np.ndarray = np.asarray(audio, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.mean(axis=0) if arr.shape[0] <= 2 else arr.mean(axis=1)
    arr = arr.ravel()
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _legacy_crest_db(mono: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(mono**2))) + 1e-12
    peak = float(np.max(np.abs(mono))) + 1e-12
    return float(20.0 * np.log10(peak / rms))


def _legacy_measure_phase_quality_delta(
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    is_repair: bool = False,
    sr: int = 44100,
) -> float:
    """Pre-Fix-Zielfunktion (MR-STFT × tanh-Richtung) — Produktionszustand bis 2026-09-26."""
    try:
        pre = _legacy_metric_mono(audio_before)
        post = _legacy_metric_mono(audio_after)
        n = min(len(pre), len(post))
        if n < 512:
            return 0.0
        pre = pre[:n]
        post = post[:n]

        from backend.core.mert_mushra_proxy import MertMushraProxy

        change = float(np.clip(float(MertMushraProxy._compute_mr_stft_loss(pre, post)), 0.0, 1.0))
        if change < 1e-4:
            return 0.0

        from backend.core.comprehensive_metrics import PsychoAcousticMetrics

        pam = PsychoAcousticMetrics(int(sr or 44100))
        dr = float(pam.calculate_roughness(pre)) - float(pam.calculate_roughness(post))
        df = float(pam.calculate_spectral_flatness(pre)) - float(pam.calculate_spectral_flatness(post))
        dc = _legacy_crest_db(post) - _legacy_crest_db(pre)
        direction = float(
            np.clip(
                0.50 * np.tanh(dr * 20.0) + 0.30 * np.tanh(df * 20.0) + 0.20 * np.clip(dc / 6.0, -1.0, 1.0),
                -1.0,
                1.0,
            )
        )
        delta = float(np.clip(direction * change, -1.0, 1.0))
        return max(delta, 0.0) if is_repair else delta
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Worker: ein Fall × eine Bedingung (eigener Prozess, resumierbar)
# ═══════════════════════════════════════════════════════════════════════════


def _run_worker(
    case_path: Path,
    cond: str,
    out_dir: Path,
    mode: str,
    max_seconds: float,
    material_override: str | None = None,
) -> int:
    import soundfile as sf

    import backend.core.closed_loop_calibrator as clc
    from backend.core.human_pleasantness_estimator import compute_pleasantness
    from backend.core.unified_restorer_v3 import QualityMode, RestorationConfig, UnifiedRestorerV3

    if cond == "A_alt":
        clc.measure_phase_quality_delta = _legacy_measure_phase_quality_delta  # type: ignore[assignment]
    elif cond != "B_wohlklang":
        raise ValueError(f"Unbekannte Bedingung: {cond}")

    audio, sr = sf.read(case_path, dtype="float32", always_2d=True)  # (N, C)
    if max_seconds > 0:
        audio = audio[: int(sr * max_seconds)]
    material = material_override or _material_for(case_path)

    hpe_in = float(compute_pleasantness(audio, sr).score)

    # Hörproben-Referenz: der (gekürzte) Eingang — nur einmal pro Fall schreiben
    _input_wav = out_dir / f"{case_path.stem}__input.wav"
    if not _input_wav.exists():
        sf.write(_input_wav, audio, sr)

    qmode = QualityMode.QUALITY if mode == "quality" else QualityMode.FAST
    cfg = RestorationConfig(
        mode=qmode,
        material_type=None,
        enable_performance_guard=True,
        enable_phase_gate=True,
        enable_phase_skipping=False,
        num_cores=4,
    )
    restorer = UnifiedRestorerV3(config=cfg)
    start = time.time()
    result = restorer.restore(audio, sample_rate=sr, mode=mode, material=material)
    runtime = float(time.time() - start)

    out_audio = getattr(result, "audio", result)
    out_audio = np.asarray(out_audio, dtype=np.float32)
    if out_audio.ndim == 2 and out_audio.shape[0] <= 8 and out_audio.shape[1] > 8:
        out_audio = out_audio.T  # (C,N) → (N,C) für soundfile
    hpe_out = float(compute_pleasantness(out_audio, sr).score)

    stem = f"{case_path.stem}__{cond}"
    wav_path = out_dir / f"{stem}.wav"
    sf.write(wav_path, out_audio, sr)
    report = {
        "case": case_path.name,
        "case_path": str(case_path.relative_to(REPO_ROOT)),
        "condition": cond,
        "material": material,
        "mode": mode,
        "runtime_seconds": round(runtime, 2),
        "hpe_input": round(hpe_in, 4),
        "hpe_output": round(hpe_out, 4),
        "hpe_delta": round(hpe_out - hpe_in, 4),
        "wav": wav_path.name,
    }
    (out_dir / f"report_{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Worker fertig: %s [%s] HPE %.3f → %.3f", case_path.name, cond, hpe_in, hpe_out)
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Driver: alle Fälle × beide Bedingungen, dann Gesamtbericht + Hinhör-Liste
# ═══════════════════════════════════════════════════════════════════════════


def _run_driver(cases: list[Path], out_dir: Path, mode: str, max_seconds: float, material: str | None = None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        for cond in ("B_wohlklang", "A_alt"):
            report_file = out_dir / f"report_{case.stem}__{cond}.json"
            if report_file.exists():
                logger.info("bereits vorhanden (resume) — übersprungen: %s", report_file.name)
                continue
            logger.info("Starte Worker: %s [%s]", case.name, cond)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(case),
                    "--cond",
                    cond,
                    "--out",
                    str(out_dir),
                    "--mode",
                    mode,
                    "--max-seconds",
                    str(max_seconds),
                    *(["--material", material] if material else []),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                logger.error("Worker fehlgeschlagen (%s [%s]):\n%s", case.name, cond, proc.stdout[-2000:])
                logger.error(proc.stderr[-4000:])

    # ── Aggregation ─────────────────────────────────────────────────────────
    rows = []
    for report_file in sorted(out_dir.glob("report_*.json")):
        rows.append(json.loads(report_file.read_text(encoding="utf-8")))
    by_case: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[row["condition"]] = row

    summary = []
    wins = 0
    decided = 0
    for case_name, conds in sorted(by_case.items()):
        a = conds.get("A_alt")
        b = conds.get("B_wohlklang")
        if not a or not b:
            continue
        decided += 1
        d = round(b["hpe_output"] - a["hpe_output"], 4)
        verdict = "B besser (Wohlklang-Optimum)" if d > 0.005 else ("A besser" if d < -0.005 else "gleichauf")
        if d > 0.005:
            wins += 1
        summary.append(
            {
                "case": case_name,
                "hpe_input": a["hpe_input"],
                "hpe_A_alt": a["hpe_output"],
                "hpe_B_wohlklang": b["hpe_output"],
                "delta_B_minus_A": d,
                "verdict": verdict,
            }
        )

    report = {
        "experiment": "Wohlklang-Optimum A/B (ClosedLoop-Zielfunktion alt vs. neu)",
        "mode": mode,
        "cases": summary,
        "b_wins": wins,
        "decided": decided,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "Zum Hinhören (identische Songs, einzige Variable = Zielfunktion der Stärken-Regelung):",
        "",
    ]
    for row in summary:
        lines.append(
            f"== {row['case']}  |  HPE A_alt={row['hpe_A_alt']}  B={row['hpe_B_wohlklang']}  ({row['verdict']})"
        )
        lines.append(f"   Eingang:        {Path(row['case']).stem}__input.wav (siehe reports-Ordner)")
        lines.append(f"   A (alter Bug):  {Path(row['case']).stem}__A_alt.wav")
        lines.append(f"   B (Wohlklang):  {Path(row['case']).stem}__B_wohlklang.wav")
        lines.append("")
    lines.append(f"Ergebnis: B gewinnt {wins}/{decided} Fälle (Δ > 0.005 = hörbar).")
    (out_dir / "HINHOEREN.txt").write_text("\n".join(lines), encoding="utf-8")

    logger.info("A/B fertig: B gewinnt %d/%d Fälle — Bericht: %s", wins, decided, out_dir / "report.json")
    print(f"\nB (Wohlklang-Optimum) gewinnt {wins}/{decided} Fälle. Hinhör-Liste: {out_dir / 'HINHOEREN.txt'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Wohlklang-A/B-Validierung (alt vs. neues Optimum)")
    parser.add_argument("--worker", help="Worker-Modus: Fall-Pfad (nur intern)")
    parser.add_argument("--cond", choices=["A_alt", "B_wohlklang"], help="Bedingung (Worker)")
    parser.add_argument("--out", help="Ausgabeordner (Worker)")
    parser.add_argument("--mode", default="quality", choices=["quality", "fast"])
    parser.add_argument("--max-seconds", type=float, default=15.0)
    parser.add_argument("--cases", nargs="*", default=None, help="Explizite Fall-Pfade (corpus/...)")
    parser.add_argument("--run-id", default=None, help="Run-ID für reports/wohlklang_ab/<run_id>")
    parser.add_argument("--material", default=None, help="Material-Override (z. B. vinyl)")
    args = parser.parse_args()

    if args.worker:
        return _run_worker(Path(args.worker), args.cond, Path(args.out), args.mode, args.max_seconds, args.material)

    cases = [Path(c) for c in args.cases] if args.cases else _default_cases()
    cases = [(c if c.is_absolute() else REPO_ROOT / c) for c in cases]
    missing = [c for c in cases if not c.exists()]
    if missing:
        logger.error("Fälle nicht gefunden: %s", missing)
        return 2
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "reports" / "wohlklang_ab" / run_id
    logger.info("A/B-Kampagne: %d Fälle, Betriebsart=%s → %s", len(cases), args.mode, out_dir)
    return _run_driver(cases, out_dir, args.mode, args.max_seconds, args.material)


if __name__ == "__main__":
    raise SystemExit(main())
