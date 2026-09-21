#!/usr/bin/env python3
"""A/B-Test: Export-Kandidaten gegen aktive Produktionsmodelle (§v10.25 §6).

Familien:
  mp_senet — Denoise/Enhance: Korpus-damaged → beide Modelle → Metriken vs. clean-Ref
  bigvgan  — Vocoder-Passthrough: clean → beide Vocoder → Metriken vs. Original
  flashsr  — Bandbreiten-Extension: clean 48 kHz ↓ 16 kHz → beide Modelle → vs. 48-kHz-Ref

Metriken über scripts/evaluate_musik_models.py (seg-SNR, spec-L1, MERT-Cosine,
VERSA-MOS, HNR, Brillianz). Ergebnisse: output/ab_tests_20260920/<family>/<family>_ab.json.
Determinismus (§G5 (GEBOTE.md)): feste Dateilisten, keine Zeitstempel in der Logik;
Modell-Sessions werden zwischen A und B vollständig getrennt.

Usage:
    venv_rocm72/bin/python scripts/ab_test_exports.py --family mp_senet
    venv_rocm72/bin/python scripts/ab_test_exports.py --family all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from evaluate_musik_models import _load_mono_48k, evaluate_candidate

logger = logging.getLogger(__name__)

OUT = _ROOT / "output" / "ab_tests_20260920"

_CORPUS_DENOISE = [
    ("vinyl_blues_1950s", "vinyl"),
    ("cassette_pop_1980s", "cassette"),
    ("digital_jazz_2010s", "digital"),
]
_CORPUS_CLEAN = [
    "corpus/vinyl/clean/vinyl_blues_1950s_clean.wav",
    "corpus/digital/clean/digital_pop_2000s_clean.wav",
]


def _write_wav(path: Path, audio: np.ndarray, sr: int = 48000) -> None:
    import soundfile as sf  # deferred

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def _result_audio(result: object) -> np.ndarray | None:
    for attr in ("audio", "enhanced_audio", "waveform"):
        value = getattr(result, attr, None)
        if value is not None:
            return np.asarray(value)
    return None


def _damaged_path(stem: str, material: str) -> Path:
    damaged_dir = _ROOT / "corpus" / material / "damaged"
    candidates = sorted(damaged_dir.glob(f"{stem}*.wav"))
    if not candidates:
        raise FileNotFoundError(f"Keine damaged-Datei für {stem} in {damaged_dir}")
    return candidates[0]


def run_mp_senet() -> dict[str, object]:
    """A=mp_senet.onnx (VoiceBank), B=finetuned/mp_senet_musik.onnx (Musik)."""
    import plugins.mp_senet_plugin as m  # deferred

    models = {
        "A_prod": _ROOT / "models" / "mp_senet" / "mp_senet.onnx",
        "B_musik": _ROOT / "models" / "mp_senet" / "finetuned" / "mp_senet_musik.onnx",
    }
    summary: dict[str, object] = {}
    for stem, material in _CORPUS_DENOISE:
        ref = _ROOT / "corpus" / material / "clean" / f"{stem}_clean.wav"
        damaged = _damaged_path(stem, material)
        ref_audio, sr = _load_mono_48k(ref)
        candidates: list[Path] = []
        for tag, onnx_path in models.items():
            m._ONNX_PATH = onnx_path
            plugin = m.MpSenetPlugin()
            damaged_audio, _ = _load_mono_48k(damaged)
            result = plugin.enhance(damaged_audio, sr)
            audio = _result_audio(result)
            if audio is None:
                logger.warning("MP-SENet %s: kein Audio im Ergebnis (%s) — übersprungen", tag, damaged)
                continue
            out_wav = OUT / "mp_senet" / f"{stem}_{tag}.wav"
            _write_wav(out_wav, audio, sr)
            candidates.append(out_wav)
        summary[stem] = {
            "reference": str(ref),
            "damaged": str(damaged),
            "candidates": [evaluate_candidate(c, ref_audio, sr) for c in candidates],
        }
    return summary


def run_bigvgan() -> dict[str, object]:
    """A=bigvgan_v2.onnx (aktiv), B=bigvgan_v2_f3e29.onnx — direkte ONNX-Läufe.

    Beide Kandidaten erhalten identische Mel-Blöcke (64 Frames, librosa-Mel mit
    den Plugin-Parametern 128 Bänder/Hann-50-ms/Hop-12,5-ms) — der Vergleich
    bleibt fair, unabhängig von den Fallback-Heuristiken des Plugins.
    """
    import librosa  # deferred
    import onnxruntime as ort  # deferred

    variants = {
        "A_prod": _ROOT / "models" / "bigvgan" / "bigvgan_v2.onnx",
        "B_f3e29": _ROOT / "models" / "bigvgan" / "bigvgan_v2_f3e29.onnx",
    }
    mel_frames = 64
    summary: dict[str, object] = {}
    for clean_file in _CORPUS_CLEAN:
        ref = _ROOT / clean_file
        ref_audio, sr = _load_mono_48k(ref)
        # 4.0 s = 320 Mel-Frames = 5 exakte 64er-Blöcke (kein Tail-Chunk)
        n_samples = min(ref_audio.shape[0], int(4.0 * sr))
        ref_audio = ref_audio[:n_samples]
        mel = librosa.feature.melspectrogram(
            y=ref_audio.astype(np.float64),
            sr=sr,
            n_fft=2400,
            hop_length=600,
            win_length=2400,
            window="hann",
            n_mels=128,
            fmin=0.0,
            fmax=sr / 2,
        )
        mel_db = librosa.power_to_db(mel).astype(np.float32)
        mel_blocks = [mel_db[:, i * mel_frames : (i + 1) * mel_frames] for i in range(mel_db.shape[1] // mel_frames)]
        candidates: list[Path] = []
        for tag, onnx_path in variants.items():
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            in_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            blocks_out = []
            for block in mel_blocks:
                inp = block[np.newaxis, :, :]  # [1, 128, 64] — Rank 3 lt. ONNX-Vertrag
                out = sess.run([out_name], {in_name: inp})[0]
                blocks_out.append(np.asarray(out, dtype=np.float32).reshape(-1))
            audio = np.concatenate(blocks_out)
            if audio.shape[0] > n_samples:
                audio = audio[:n_samples]
            out_wav = OUT / "bigvgan" / f"{Path(clean_file).stem}_{tag}.wav"
            _write_wav(out_wav, audio, sr)
            candidates.append(out_wav)
        summary[clean_file] = {
            "reference": str(ref),
            "candidates": [evaluate_candidate(c, ref_audio, sr) for c in candidates],
        }
    return summary


def run_flashsr() -> dict[str, object]:
    """A=Upstream SR48k.pth (Basis des Produktions-ONNX), B=f4-best.pt — Torch-A/B.

    Der f4-ONNX-Export mit dynamischer Zeitachse ist am Torch-2.11-Tracer
    blockiert (LowPassFilter1d-Pad-Constant-Folding; dokumentiert in §v10.25).
    Der Gewichtsvergleich läuft daher direkt in Torch (FASR), 16k → 48k.
    """
    import sys as _sys

    import librosa  # deferred
    import torch  # deferred

    _FLASHSR_DIR = _ROOT / "models" / "flashsr"
    _sys.path.insert(0, str(_FLASHSR_DIR))
    from FastAudioSR import FASR  # defer

    variants = {
        "A_upstream": _FLASHSR_DIR / "FastAudioSR" / "SR48k.pth",
        "B_f4": _ROOT / "output" / "_training_archive_20260920" / "f4_flashsr" / "best.pt",
    }
    summary: dict[str, object] = {}
    for clean_file in _CORPUS_CLEAN:
        ref = _ROOT / clean_file
        ref_audio, sr = _load_mono_48k(ref)
        low = librosa.resample(ref_audio, orig_sr=sr, target_sr=16000).astype(np.float32)
        candidates: list[Path] = []
        for tag, ckpt_path in variants.items():
            fasr = FASR(str(ckpt_path))
            model = fasr.model.eval().to("cpu")
            with np.errstate(all="ignore"):
                out_48k = model(torch.from_numpy(low)[None, None, :]).detach().numpy().reshape(-1)
            out_48k = np.nan_to_num(out_48k, nan=0.0, posinf=0.0, neginf=0.0)
            if out_48k.shape[0] > ref_audio.shape[0]:
                out_48k = out_48k[: ref_audio.shape[0]]
            elif out_48k.shape[0] < ref_audio.shape[0]:
                out_48k = np.pad(out_48k, (0, ref_audio.shape[0] - out_48k.shape[0]))
            out_wav = OUT / "flashsr" / f"{Path(clean_file).stem}_{tag}.wav"
            _write_wav(out_wav, out_48k.astype(np.float32), sr)
            candidates.append(out_wav)
        summary[clean_file] = {
            "reference": str(ref),
            "candidates": [evaluate_candidate(c, ref_audio, sr) for c in candidates],
        }
    return summary


_RUNNERS = {"mp_senet": run_mp_senet, "bigvgan": run_bigvgan, "flashsr": run_flashsr}


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B-Test Export-Kandidaten (§v10.25 §6)")
    parser.add_argument("--family", choices=[*_RUNNERS, "all"], default="all")
    args = parser.parse_args()

    families = list(_RUNNERS) if args.family == "all" else [args.family]
    for family in families:
        logger.info("A/B startet: %s", family)
        summary = _RUNNERS[family]()
        out_json = OUT / family / f"{family}_ab.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        logger.info("A/B abgeschlossen: %s → %s", family, out_json)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
