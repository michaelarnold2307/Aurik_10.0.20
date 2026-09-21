#!/usr/bin/env python3
"""A/B-Evaluierung für Musik-Modelle — §v10.19 §11.2, §v10.25 §6.

Vergleicht eine Referenz (clean) mit beliebig vielen Kandidaten (restaurierte
Versionen, z. B. Speech-Modell vs. Musik-Modell) und schreibt eine
deterministische Metrik-Batterie als JSON fürs Tracking.

Metriken:
  - seg_snr_dB      Fenster-SNR gegen die Referenz (16000-Sample-Fenster)
  - spec_l1         Mittlere L1-Distanz der STFT-Magnituden (spektrale Treue,
                    THD-Ersatz bei Musik)
  - mert_cosine     MERT-QualityGate-Score 0-100 (Kosinus zum Clean-Musik-
                    Zentroid, models/mert/mert_330m.onnx)
  - versa_mos       VERSA-MOS (plugins.versa_plugin.score_mos)
  - hnr_db          Harmonics-to-Noise-Ratio (backend.core.dsp.hnr_guard)
  - brillianz_ratio Energieanteil 8-20 kHz (Luftband-Metrik, §SLR-1e2b)
  - visqol_mos      ViSQOL MOS-LQO (backend.quality_metrics_manager.assess_with_reference)

§V6 (copilot-instructions.md): Jeder Metrik-Ausfall wird mit logger.warning()
+ Grund gemeldet und als "error" im JSON geführt — nie stiller Skip.
Determinismus (§G5 (GEBOTE.md)): keine Zeitstempel in der Entscheidungslogik;
der JSON-Inhalt ist bei gleichen Eingaben bit-identisch.

Usage:
    python3 scripts/evaluate_musik_models.py \
        --ref corpus/vinyl/clean/vinyl_jazz_clean.wav \
        --candidates out_speech.wav out_musik.wav \
        --out output/eval_musik_models.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))


def _load_mono_48k(path: Path) -> tuple[np.ndarray, int]:
    """Lädt WAV/FLAC mono @ 48 kHz (float32)."""
    import librosa  # deferred: langsamer Import, nur hier benötigt

    y, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != 48000:
        y = librosa.resample(y, orig_sr=sr, target_sr=48000)
        sr = 48000
    return y.astype(np.float32), sr


def _align(ref: np.ndarray, est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kürzt beide Signale auf die gemeinsame Länge (keine Zeitverschiebung)."""
    n = min(ref.shape[0], est.shape[0])
    return ref[:n], est[:n]


def _seg_snr_dB(ref: np.ndarray, est: np.ndarray, win: int = 16000) -> float:
    ref, est = _align(ref, est)
    if ref.shape[0] < win:
        win = max(1024, ref.shape[0] // 2)
    frames = [(ref[i : i + win], est[i : i + win]) for i in range(0, ref.shape[0] - win + 1, win)]
    snrs = []
    for r, e in frames:
        noise = e - r
        num = float(np.sum(r**2))
        den = float(np.sum(noise**2)) + 1e-12
        snrs.append(10.0 * np.log10(num / den + 1e-12))
    return float(np.mean(snrs))


def _spec_l1(ref: np.ndarray, est: np.ndarray) -> float:
    ref, est = _align(ref, est)
    n_fft, hop = 2048, 512
    window = np.hanning(n_fft).astype(np.float32)
    frames_r = [ref[i : i + n_fft] * window for i in range(0, ref.shape[0] - n_fft + 1, hop)]
    frames_e = [est[i : i + n_fft] * window for i in range(0, est.shape[0] - n_fft + 1, hop)]
    n_frames = min(len(frames_r), len(frames_e))
    total = 0.0
    for k in range(n_frames):
        mr = np.abs(np.fft.rfft(frames_r[k]))
        me = np.abs(np.fft.rfft(frames_e[k]))
        total += float(np.mean(np.abs(mr - me)))
    return total / max(n_frames, 1)


def _brillianz_ratio(audio: np.ndarray, sr: int, lo: float = 8000.0, hi: float = 20000.0) -> float:
    """Energieanteil des Luftbands 8-20 kHz an der Gesamtenergie (0-1)."""
    spec = np.abs(np.fft.rfft(audio)) ** 2
    freqs = np.fft.rfftfreq(audio.shape[0], 1.0 / sr)
    band = spec[(freqs >= lo) & (freqs <= hi)].sum()
    total = spec.sum() + 1e-12
    return float(band / total)


def _mert_cosine(audio: np.ndarray, sr: int) -> float:
    from plugins.mert_quality_gate import MERTQualityGate  # deferred (§III.9 GPU-/Lade-Kosten)

    gate = MERTQualityGate()
    return float(gate.score_chunk(audio, sr))


def _versa_mos(audio: np.ndarray, sr: int) -> float:
    from plugins.versa_plugin import score_mos  # deferred

    result = score_mos(audio, sr)
    for attr in ("mos", "score", "mean_mos"):
        value = getattr(result, attr, None)
        if isinstance(value, (int, float)):
            return float(value)
    raise ValueError(f"VersaResult ohne MOS-Feld: {type(result).__name__}")


def _hnr_db(audio: np.ndarray, sr: int) -> float:
    from backend.core.dsp.hnr_guard import compute_hnr  # deferred

    return float(compute_hnr(audio.astype(np.float32), sr))


def _visqol(ref_path: Path, candidate: Path) -> float:
    from backend.quality_metrics_manager import assess_with_reference  # deferred

    scores = assess_with_reference(str(ref_path), str(candidate))
    value = scores.get("ViSQOL_MOS")
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"ViSQOL-Ergebnis ohne ViSQOL_MOS: {list(scores)[:5]}")


_METRICS: tuple[tuple[str, object], ...] = (
    ("seg_snr_dB", _seg_snr_dB),
    ("spec_l1", _spec_l1),
    ("mert_cosine", _mert_cosine),
    ("versa_mos", _versa_mos),
    ("hnr_db", _hnr_db),
    ("brillianz_ratio", _brillianz_ratio),
    ("visqol_mos", _visqol),
)


def evaluate_candidate(
    candidate: Path, ref_audio: np.ndarray, sr: int, ref_path: Path | None = None
) -> dict[str, object]:
    """Misst alle Metriken für einen Kandidaten; Ausfälle → logger.warning + "error" (§V6 (copilot-instructions.md))."""
    audio, cand_sr = _load_mono_48k(candidate)
    if cand_sr != sr:
        raise ValueError(f"Sample-Rate-Drift: Ref={sr}, Kandidat={cand_sr}")
    result: dict[str, object] = {"file": str(candidate), "sr": sr}
    for name, fn in _METRICS:
        try:
            if name == "visqol_mos":
                if ref_path is None:
                    raise ValueError("visqol_mos benötigt die Referenz-Datei (ref_path fehlt)")
                value = fn(ref_path, candidate)  # type: ignore[operator]
            elif name in ("seg_snr_dB", "spec_l1"):
                value = fn(ref_audio, audio)  # type: ignore[operator]
            else:
                value = fn(audio, sr)  # type: ignore[operator]
            result[name] = round(float(value), 6)
        except Exception as exc:  # pylint: disable=broad-except — §V6 (copilot-instructions.md): nie still degradieren
            logger.warning("Metrik %s für %s fehlgeschlagen (%s) — mit Begründung übersprungen", name, candidate, exc)
            result[name] = "error"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B-Evaluierung Musik-Modelle (§v10.19 §11.2)")
    parser.add_argument("--ref", required=True, type=Path, help="Clean-Referenz (WAV/FLAC)")
    parser.add_argument(
        "--candidates", required=True, nargs="+", type=Path, help="Zu bewertende Kandidaten (≥2 für A/B)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_PROJECT / "output" / "evaluate_musik_models.json",
        help="Ziel-JSON (Default: output/evaluate_musik_models.json)",
    )
    args = parser.parse_args()

    for cand in args.candidates:
        if not cand.is_file():
            parser.error(f"Kandidat nicht gefunden: {cand}")
    if not args.ref.is_file():
        parser.error(f"Referenz nicht gefunden: {args.ref}")

    ref_audio, sr = _load_mono_48k(args.ref)
    results = [evaluate_candidate(c, ref_audio, sr, ref_path=args.ref) for c in args.candidates]
    payload = {
        "reference": str(args.ref),
        "sample_rate": sr,
        "candidates": results,
        "metrics": [name for name, _ in _METRICS],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    logger.info("Evaluation geschrieben: %s (%d Kandidaten)", args.out, len(results))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
