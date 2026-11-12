#!/usr/bin/env python3
"""F9-Identitäts-Gate (D-K4-5): Vokal-Identitäts-Witness des VoiceEncoder-Finetunes.

Warum: D-K4-5 ist offen mit „Never-worsen + **Identitäts-Gate** +
Hör-Sign-off". Dieses Skript liefert den messbaren Teil des
Identitäts-Gates: Wie gut trennt der deployte VoiceEncoder (resemblyzer,
256-dim d-Vector) **gleiche Sänger** von **verschiedenen Sängern** auf dem
lokalen MUSDB18-HQ-Bestand?

Ehrlichkeits-Vertrag (wie `calibrate_gender_evidence.py`): Es existiert
KEIN Ground-Truth-Sänger-Label im Korpus — deshalb wird keine Schwelle
blind gesetzt. Das Skript weist die Trennbarkeit aus (Kosinus-Similarität
gleicher vs. fremder Sänger) und empfiehlt eine Schwelle NUR, wenn die
Verteilungen klar getrennt sind (Margin-Regel, „empirisch, kein Ground
Truth"). Andernfalls bleibt die Identitäts-Zusage Hör-signierpflichtig
(§V6 (copilot-instructions.md), kein stiller Schwellwert).

Vertrag: CPU-only fail-closed (sichtbare GPU ⇒ Exit 2 — schützt die
Trainingskette, Muster `verify_flashsr_d3_gate.py`); deterministisch
(Seed 42, sortierte Tracks, §G5 (GEBOTE.md)); Report-Only — kein
Flag-Flip, kein Deploy, keine Stärke-Änderung.

Usage:
    python3 -B scripts/verify_f9_identity_gate.py \
        --musdb-root data/musdb18hq \
        --out docs/reports/current/2026-11-12_f9_identity_gate.json \
        --paare 20 --seed 42
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

_SR = 48000
_MARGIN_MIN = 0.10  # Trennbarkeits-Mindestabstand für eine Empfehlung


def _gpu_sichtbar() -> bool:
    """GPU-Präsenz wie in D-K3-74 gepinnt (Produktions-Detektor)."""
    from backend.core.gpu_model_registry import _gpu_device_present

    return bool(_gpu_device_present())


def _decode_48k(pfad: Path) -> np.ndarray:
    import soundfile as sf

    _r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(pfad), "-ac", "1", "-ar", str(_SR), "-f", "wav", "-"],
        capture_output=True,
        check=True,
    )
    _data, _sr = sf.read(io.BytesIO(_r.stdout), always_2d=False)
    return np.asarray(_data, dtype=np.float32)


def _mittel_ausschnitt(audio: np.ndarray, sekunden: float, offset: float) -> np.ndarray:
    arr = np.asarray(audio)
    want = int(round(sekunden * _SR))
    if arr.shape[-1] <= want or sekunden <= 0:
        return arr
    start = int(min(max(offset * _SR, 0), arr.shape[-1] - want))
    return arr[start : start + want]


def kosinus(a: np.ndarray, b: np.ndarray) -> float:
    _a = np.asarray(a, dtype=np.float64)
    _b = np.asarray(b, dtype=np.float64)
    _n = float(np.linalg.norm(_a) * np.linalg.norm(_b))
    if _n <= 1e-12:
        return 0.0
    return float(np.dot(_a, _b) / _n)


def empfehlung(gleiche: list[float], fremde: list[float]) -> dict:
    """Trennbarkeits-Empfehlung — NUR bei klarer Separation (Margin-Regel)."""
    _g = np.asarray(gleiche, dtype=np.float64)
    _f = np.asarray(fremde, dtype=np.float64)
    _mean_g = float(_g.mean())
    _std_g = float(_g.std())
    _mean_f = float(_f.mean())
    _std_f = float(_f.std())
    _margin = (_mean_g - _std_g) - (_mean_f + _std_f)
    _tragfaehig = bool(_margin >= _MARGIN_MIN and _mean_g > _mean_f)
    _schwelle = round(float((_mean_f + _std_f + _mean_g - _std_g) / 2.0), 4) if _tragfaehig else None
    return {
        "n_gleiche": int(len(gleiche)),
        "n_fremde": int(len(fremde)),
        "mean_gleiche": round(_mean_g, 4),
        "std_gleiche": round(_std_g, 4),
        "mean_fremde": round(_mean_f, 4),
        "std_fremde": round(_std_f, 4),
        "margin": round(_margin, 4),
        "empfohlene_schwelle": _schwelle,
        "empfehlung": (
            "tragfaehig (empirisch, kein Ground Truth)"
            if _tragfaehig
            else "keine belastbare Schwelle — Identitaets-Zusage bleibt hoer-signierpflichtig"
        ),
    }


def _vocal_stems(musdb_root: Path) -> list[Path]:
    _out: list[Path] = []
    for _split in ("train", "test"):
        _split_dir = musdb_root / _split
        if not _split_dir.is_dir():
            continue
        for _track in sorted(_split_dir.iterdir()):
            _voc = _track / "vocals.wav"
            if _voc.is_file():
                _out.append(_voc)
    return _out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--musdb-root", type=Path, default=Path("data/musdb18hq"))
    ap.add_argument("--out", type=Path, default=Path("docs/reports/current/2026-11-12_f9_identity_gate.json"))
    ap.add_argument("--paare", type=int, default=20)
    ap.add_argument("--sekunden", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if _gpu_sichtbar():
        logger.error("GPU sichtbar — Abbruch (CPU-only-Vertrag schützt die Trainingskette)")
        return 2

    from plugins.resemblyzer_plugin import get_resemblyzer_plugin

    _plug = get_resemblyzer_plugin()
    if not _plug.available():
        logger.error("VoiceEncoder nicht verfügbar — kein Identitäts-Gate möglich (§V6 (copilot-instructions.md))")
        return 3

    _stems = _vocal_stems(args.musdb_root)
    if len(_stems) < 2:
        logger.error("zu wenige Vocal-Stems (%d) unter %s", len(_stems), args.musdb_root)
        return 4

    _rng = np.random.default_rng(args.seed)
    _gleiche: list[float] = []
    _fremde: list[float] = []
    for _n in range(args.paare):
        _same = _stems[int(_rng.integers(0, len(_stems)))]
        _audio = _decode_48k(_same)
        _emb_a = _plug.embed(_mittel_ausschnitt(_audio, args.sekunden, 0.2), _SR)
        _emb_b = _plug.embed(_mittel_ausschnitt(_audio, args.sekunden, 0.2 + args.sekunden), _SR)
        if _emb_a is None or _emb_b is None:
            logger.warning("Embedding fehlgeschlagen (%s) — Paar übersprungen (§V6 (copilot-instructions.md))", _same)
            continue
        _gleiche.append(kosinus(_emb_a, _emb_b))
        _other = _stems[int(_rng.integers(0, len(_stems)))]
        _emb_c = _plug.embed(_mittel_ausschnitt(_decode_48k(_other), args.sekunden, 0.2), _SR)
        if _emb_c is None:
            logger.warning(
                "Embedding fehlgeschlagen (%s) — Gegenpaar übersprungen (§V6 (copilot-instructions.md))", _other
            )
            continue
        _fremde.append(kosinus(_emb_a, _emb_c))
        logger.info("Paar %d: gleiche=%.4f fremde=%.4f", _n, _gleiche[-1], _fremde[-1])

    if len(_gleiche) < 5 or len(_fremde) < 5:
        logger.error(
            "zu wenige verwertbare Paare (gleiche=%d, fremde=%d) — kein Gate-Report", len(_gleiche), len(_fremde)
        )
        return 5

    _bericht = {
        "schema": "aurik.f9_identity_gate/1",
        "modell": "resemblyzer_voice_encoder.onnx (d-vector, 256-dim)",
        "seed": args.seed,
        "cpu_only": True,
        "flag_flips": "keine",
        **empfehlung(_gleiche, _fremde),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_bericht, indent=2, ensure_ascii=True), encoding="utf-8")
    logger.info("Report geschrieben: %s — %s", args.out, _bericht["empfehlung"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
