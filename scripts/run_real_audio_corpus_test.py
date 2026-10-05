#!/usr/bin/env python3
"""Real-Audio-Corpus MUSHRA-Test (SOTA Implementation).

Führt eine vollständige E2E-Validierung mit echten Audiodateien durch:
1. Lädt Referenz- und degradierte Audio-Paare
2. Wendet Aurik-Restaurierungs-Pipeline an
3. Bewertet Ergebnis via MUSHRA-Evaluator (objektiv)
4. Generiert detaillierten Qualitätsbericht

Usage:
    python scripts/run_real_audio_corpus_test.py \
        --reference-dir /path/to/clean/audio \
        --degraded-dir /path/to/degraded/audio \
        --output reports/real_audio_corpus_results.json

Autor: Aurik 10.0.0 — SOTA Real-Audio Validation
Referenz: ITU-R BS.1534-3 (MUSHRA), ISO 532-1 (Zwicker-Lautstärke)
Exit-Code: 0 = alle Fälle erfolgreich; 1 = mindestens ein error-Fall (§V6 (VERBOTEN.md)).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _load_audio(path: Path, sr: int = 48000) -> np.ndarray:
    """Lädt Audiodatei und resampelt zu Ziel-SR (deterministisch, mono).

    Kein stiller Ersatzpfad (§V6 (VERBOTEN.md)): Fehler eskalieren an den
    Aufrufer, der den Fall als `error` protokolliert — ein Beweis-Generator
    darf nie still Identität oder Stille messen.
    """
    import soundfile as sf

    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.ndim == 2 and data.shape[1] > 1:
        mono = np.mean(data, axis=1).astype(np.float32)  # (N, C) → (N,)
    else:
        mono = data.reshape(-1).astype(np.float32)

    if file_sr != sr:
        import librosa

        mono = librosa.resample(mono, orig_sr=file_sr, target_sr=sr)  # type: ignore[no-untyped-call]

    return np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)


def _run_restoration(audio: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Führt die kanonische Aurik-Restaurierung aus (UnifiedRestorerV3).

    Muster identisch zu ``scripts/mushra_harness.py::prepare_pair`` (EINE
    Wahrheitsquelle des Hörpfads) inkl. Layout-Normalisierung. Kein stiller
    Identitäts-Fallback: Exceptions eskalieren, der Aufrufer protokolliert
    den Fall als `error` (§V6 (VERBOTEN.md)).
    """
    from backend.core.unified_restorer_v3 import QualityMode, RestorationConfig, UnifiedRestorerV3

    cfg = RestorationConfig(mode=QualityMode.QUALITY)
    engine = UnifiedRestorerV3(cfg)
    result = engine.restore(audio, sample_rate=sr)

    restored = np.asarray(result.audio, dtype=np.float32)
    if restored.ndim == 2 and restored.shape[0] == 2 and restored.shape[1] != 2:
        restored = restored.T  # Kanal-zuerst → (N, C)
    return np.nan_to_num(restored, nan=0.0, posinf=0.0, neginf=0.0)


def _evaluate_mushra(reference: np.ndarray, test: np.ndarray, sr: int = 48000) -> dict:
    """Bewertet via MUSHRA-Evaluator (objektiv). Fehler eskalieren (§V6 (VERBOTEN.md))."""
    from backend.core.mushra_evaluator import evaluate_mushra

    result = evaluate_mushra(reference, test, sr=sr, compute_anchor=True)
    return {
        "mushra_score": result.mushra_score,
        "grade": result.grade,
        "itu_grade": result.itu_grade,
        "nsim": result.nsim,
        "anchor_score": result.anchor_score,
        "musical_goals": result.musical_goals,
        **result.details,
    }


def main() -> int:
    """Hauptfunktion für Real-Audio-Corpus-Test."""
    parser = argparse.ArgumentParser(description="Real-Audio-Corpus MUSHRA-Test (SOTA)")
    parser.add_argument("--reference-dir", required=True, help="Verzeichnis mit sauberen Referenz-WAVs")
    parser.add_argument("--degraded-dir", required=True, help="Verzeichnis mit degradierten WAVs")
    parser.add_argument("--output", default="reports/real_audio_corpus_results.json", help="JSON-Ausgabepfad")
    parser.add_argument("--sr", type=int, default=48000, help="Abtastrate in Hz (Standard: 48000)")
    args = parser.parse_args()

    ref_dir = Path(args.reference_dir)
    deg_dir = Path(args.degraded_dir)
    output_path = Path(args.output)

    # Finde Audio-Paare — Korpus-Konvention wie im Hör-Panel-Setup
    # (scripts/mushra_harness.py::find_pairs): <base>_* (degradiert) ↔
    # <base>_clean.wav (Referenz); der längste Clean-Präfix gewinnt.
    clean_by_stem = {p.stem: p for p in sorted(ref_dir.glob("*.wav"))}
    audio_pairs: list[tuple[Path, Path]] = []
    unmatched: list[Path] = []
    for deg_file in sorted(deg_dir.glob("*.wav")):
        ref_file: Path | None = clean_by_stem.get(deg_file.stem)
        if ref_file is None:
            best: tuple[int, Path] | None = None
            for stem, cand in clean_by_stem.items():
                base = stem[:-6] if stem.endswith("_clean") else stem
                if deg_file.stem == base or deg_file.stem.startswith(base + "_"):
                    if best is None or len(base) > best[0]:
                        best = (len(base), cand)
            ref_file = best[1] if best is not None else None
        if ref_file is None:
            unmatched.append(deg_file)
            continue
        audio_pairs.append((ref_file, deg_file))

    if unmatched:
        logger.warning(
            "Keine Referenz für %d degradierte Datei(en): %s",
            len(unmatched),
            ", ".join(p.name for p in unmatched[:5]),
        )
    if not audio_pairs:
        logger.error("Keine passenden Audio-Paare gefunden!")
        return 1

    logger.info("=" * 80)
    logger.info("Real-Audio-Corpus MUSHRA-Test (SOTA)")
    logger.info("=" * 80)
    logger.info("Audio-Paare: %d", len(audio_pairs))
    logger.info("Referenz-Verzeichnis: %s", ref_dir)
    logger.info("Degradierter Verzeichnis: %s", deg_dir)
    logger.info("Ausgabe: %s", output_path)
    logger.info("=" * 80)

    # Ergebnisse sammeln
    results = []
    error_cases = 0
    total_time = 0.0

    for idx, (ref_path, deg_path) in enumerate(audio_pairs, 1):
        pair_start = time.time()
        logger.info("\n[%d/%d] Verarbeite: %s", idx, len(audio_pairs), deg_path.name)

        try:
            # Lade Audio
            reference = _load_audio(ref_path, sr=args.sr)
            degraded = _load_audio(deg_path, sr=args.sr)

            logger.info(
                "  Referenz: %d Samples (%.1f s)",
                int(np.asarray(reference).size),
                int(np.asarray(reference).size) / args.sr,
            )
            logger.info(
                "  Degradierter: %d Samples (%.1f s)",
                int(np.asarray(degraded).size),
                int(np.asarray(degraded).size) / args.sr,
            )

            # Restaurierung (kanonischer Hörpfad, kein stiller Fallback)
            restored = _run_restoration(degraded, sr=args.sr)

            # MUSHRA-Bewertung
            mushra_result = _evaluate_mushra(reference, restored, sr=args.sr)
        except Exception as exc:
            error_cases += 1
            logger.error(
                "  FALL-FEHLER (%s): %s: %s — als error protokolliert (§V6 (VERBOTEN.md))",
                deg_path.name,
                type(exc).__name__,
                exc,
            )
            results.append(
                {
                    "pair_index": idx,
                    "file_name": ref_path.name,
                    "damaged_file": deg_path.name,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "processing_time_s": round(time.time() - pair_start, 3),
                }
            )
            continue

        pair_time = time.time() - pair_start
        total_time += pair_time

        logger.info(
            "  MUSHRA-Score: %.1f/100 (%s) | NSIM=%.3f | Zeit=%.2fs",
            mushra_result["mushra_score"],
            mushra_result["grade"],
            mushra_result.get("nsim", 0.0),
            pair_time,
        )

        results.append(
            {
                "pair_index": idx,
                "file_name": ref_path.name,
                "damaged_file": deg_path.name,
                "status": "ok",
                "reference_samples": int(np.asarray(reference).size),
                "degraded_samples": int(np.asarray(degraded).size),
                "restored_samples": int(np.asarray(restored).size),
                "processing_time_s": round(pair_time, 3),
                **mushra_result,
            }
        )

    # Zusammenfassung
    scores = [r["mushra_score"] for r in results if r.get("status") == "ok"]
    avg_score = float(np.mean(scores)) if scores else 0.0
    min_score = float(np.min(scores)) if scores else 0.0
    max_score = float(np.max(scores)) if scores else 0.0

    summary = {
        "test_type": "real_audio_corpus_mushra",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_pairs": len(results),
        "error_cases": error_cases,
        "total_time_s": round(total_time, 3),
        "statistics": {
            "mean_score": round(avg_score, 2),
            "min_score": round(min_score, 2),
            "max_score": round(max_score, 2),
            "std_dev": round(float(np.std(scores)), 2) if scores else 0.0,
        },
        "grade_distribution": {
            "excellent": sum(1 for s in scores if s >= 91),
            "good": sum(1 for s in scores if 80 <= s < 91),
            "fair": sum(1 for s in scores if 60 <= s < 80),
            "poor": sum(1 for s in scores if 40 <= s < 60),
            "bad": sum(1 for s in scores if s < 40),
        },
        "results": results,
    }

    # Ausgabe
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 80)
    logger.info("ZUSAMMENFASSUNG")
    logger.info("=" * 80)
    logger.info("Paare verarbeitet: %d", len(results))
    logger.info("Fehlerfälle: %d", error_cases)
    logger.info("Gesamtzeit: %.2f s", total_time)
    logger.info("MUSHRA-Mittelwert: %.2f/100", avg_score)
    logger.info("Min-Wertung: %.2f | Max-Wertung: %.2f", min_score, max_score)
    logger.info(
        "Verteilung: Excellent=%d, Good=%d, Fair=%d, Poor=%d, Bad=%d",
        summary["grade_distribution"]["excellent"],
        summary["grade_distribution"]["good"],
        summary["grade_distribution"]["fair"],
        summary["grade_distribution"]["poor"],
        summary["grade_distribution"]["bad"],
    )
    logger.info("Ergebnisse gespeichert: %s", output_path)
    if error_cases:
        logger.error("Fehlerfälle: %d — Exit ≠ 0 (Beweis-Generator, §V6 (VERBOTEN.md))", error_cases)
    logger.info("=" * 80)

    return 1 if error_cases else 0


if __name__ == "__main__":
    sys.exit(main())
