"""KP-2 · Musik-MOS-Orakel-Richtungsstabilitäts-Harness (UTMOS vs MuQ+A1 vs SingMOS).

Vergleicht auf Paar-/Degradations-Evidenz drei referenzfreie MOS-Orakel nach
**Richtungsstabilität** (Anteil korrekt geordneter Paare ref > degradiert je
Degradationsklasse) und **Non-Inferiority** gegenüber dem report-only-Zeitraum
UTMOS (Δ-Richtungs-Stabilität).

Degradationsklassen (deterministisch erzeugt, §G5 (copilot-instructions.md)):
Rauschen (SNR), Clipping, Bandlimit, Codec. Auf MUSDB18-test (musiknativ, §6 (copilot-instructions.md)
docs/SOTA_TRAINING_KONZEPT.md) und dem synthetischen Vinyl-Korpus
(corpus/vinyl, CC0). Kein Training, keine Gewichtsänderung, kein Flag-Flip.

Report: docs/reports/current/<datum>_kp2_musik_mos_orakel.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_MUSDB = _ROOT / "data" / "musdb18hq" / "test"
_VINYL_CLEAN = _ROOT / "corpus" / "vinyl" / "clean"
_VINYL_DAMAGED = _ROOT / "corpus" / "vinyl" / "damaged"
_REPORT_DIR = _ROOT / "docs" / "reports" / "current"

_SR = 44100
_CLIP_S = 10  # kurze, stationäre Ausschnitte (MOS-skalierbar, Performance-Budget)


def _load_mono(path: Path, seconds: int, seed: int) -> np.ndarray:
    wf = wavfile.read(str(path))
    sr = int(wf[0])
    wav = np.asarray(wf[1])
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / 32768.0
    elif wav.dtype == np.int32:
        wav = wav.astype(np.float32) / 2147483648.0
    else:
        wav = wav.astype(np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    n_need = seconds * sr
    if wav.size <= n_need:
        return wav
    rng = np.random.RandomState(seed)
    start = rng.randint(0, wav.size - n_need)
    return np.asarray(wav[start : start + n_need], dtype=np.float32)


def _peak_norm(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) + 1e-12
    return np.asarray(x * (peak / m), dtype=np.float32)


def _degrade_noise(x: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    sig_p = float(np.mean(x**2)) + 1e-12
    noise_p = sig_p / (10 ** (snr_db / 10.0))
    noise = rng.randn(x.size).astype(np.float32) * np.sqrt(noise_p)
    return np.asarray(x + noise, dtype=np.float32)


def _degrade_clip(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(np.clip(x, -threshold, threshold), dtype=np.float32)


def _degrade_bandlimit(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    from scipy.signal import butter

    from backend.core.audio_utils import safe_filtfilt  # pylint: disable=import-outside-toplevel

    b, a = butter(5, cutoff_hz / (0.5 * _SR), btype="low", output="ba")
    return np.asarray(safe_filtfilt(b, a, x), dtype=np.float32)


def _degrade_codec(x: np.ndarray, sr: int) -> np.ndarray:
    # Deterministische MP3-artige Degradation via Resample (down-then-up)
    # ohne externen Encoder — repräsentativ für die Codec-Klasse (§G5 (copilot-instructions.md)).
    from scipy.signal import resample_poly

    low = 16000
    g = int(np.gcd(sr, low))
    down = resample_poly(x, low // g, sr // g)
    up = resample_poly(down, sr // g, low // g)
    if len(up) > len(x):
        up = up[: len(x)]
    elif len(up) < len(x):
        up = np.pad(up, (0, len(x) - len(up)))
    return np.asarray(up, dtype=np.float32)


_ORACLES: dict[str, Callable[[np.ndarray, int], float | None]] = {}


def _resample_48k(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Deterministische Resampling auf 48 kHz (UTMOS/SingMOS verlangen 48 k; §G5 (copilot-instructions.md))."""
    if sr == 48000:
        return np.asarray(audio, dtype=np.float32), 48000
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, 48000)
    up = resample_poly(audio, 48000 // g, sr // g)
    return np.asarray(up, dtype=np.float32), 48000


def _register_oracles() -> None:
    _ORACLES.clear()

    def utmos(audio: np.ndarray, sr: int) -> float | None:
        try:
            from plugins.utmos_plugin import estimate_mos  # pylint: disable=import-outside-toplevel

            audio48, sr48 = _resample_48k(audio, sr)
            return float(estimate_mos(audio48, sr48).mos)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[utmos] nicht verfügbar ({exc})", file=sys.stderr)
            return None

    def muq(audio: np.ndarray, sr: int) -> float | None:
        try:
            from plugins.muq_plugin import estimate_muq_mos  # pylint: disable=import-outside-toplevel

            return estimate_muq_mos(audio, sr)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[muq] nicht verfügbar ({exc})", file=sys.stderr)
            return None

    def singmos(audio: np.ndarray, sr: int) -> float | None:
        try:
            from plugins.versa_plugin import get_versa_plugin  # pylint: disable=import-outside-toplevel

            audio48, sr48 = _resample_48k(audio, sr)
            result = get_versa_plugin().score(audio48, sr48)
            if result.model_used != "singmos_pro":
                return None
            return float(result.mos)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[singmos] nicht verfügbar ({exc})", file=sys.stderr)
            return None

    _ORACLES["utmos"] = utmos
    _ORACLES["muq_a1"] = muq
    _ORACLES["singmos_pro"] = singmos


def _degradations(ref: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    return {
        "noise_10dB": _degrade_noise(ref, 10.0, seed),
        "noise_0dB": _degrade_noise(ref, 0.0, seed),
        "clip_0.3": _degrade_clip(ref, 0.3),
        "clip_0.1": _degrade_clip(ref, 0.1),
        "bandlimit_12k": _degrade_bandlimit(ref, 12000.0),
        "bandlimit_8k": _degrade_bandlimit(ref, 8000.0),
        "codec_16k": _degrade_codec(ref, _SR),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42, help="Seed (Determinismus §G5 (copilot-instructions.md))")
    ap.add_argument("--tracks", type=int, default=3, help="Anzahl MUSDB-Test-Tracks")
    ap.add_argument("--seconds", type=int, default=_CLIP_S)
    ap.add_argument("--no-vinyl", action="store_true", help="Vinyl-Korpus überspringen")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    _register_oracles()

    tracks = sorted(p for p in _MUSDB.glob("*") if p.is_dir()) if _MUSDB.is_dir() else []
    if tracks:
        tracks = tracks[: args.tracks]

    # Paare: (ref_resolver, name) — MUSDB-Tracks synthetisch degradiert; Vinyl
    # real-paired (clean ↔ damaged) über identischen Stem.
    pairs: list[tuple[str, np.ndarray, dict[str, np.ndarray]]] = []
    for i, track in enumerate(tracks):
        ref = _peak_norm(_load_mono(track / "mixture.wav", args.seconds, args.seed + i))
        pairs.append((track.name, ref, _degradations(ref, args.seed + i)))

    if not args.no_vinyl and _VINYL_CLEAN.is_dir():
        clean_files = sorted(_VINYL_CLEAN.glob("*.wav"))
        for i, cf in enumerate(clean_files[: len(tracks) if tracks else 3]):
            ref_v = _peak_norm(_load_mono(cf, args.seconds, args.seed + 1000 + i))
            pairs.append((f"vinyl:{cf.stem}", ref_v, _degradations(ref_v, args.seed + 2000 + i)))

    if not pairs:
        print("Keine Paar-Evidenz gefunden (MUSDB und/oder corpus/vinyl fehlt).", file=sys.stderr)
        return 2

    # Richtungsstabilität je Orakel je Klasse.
    report: dict[str, Any] = {
        "date": date.today().isoformat(),
        "script": "scripts/benchmark_music_mos_oracles.py",
        "seed": args.seed,
        "pairs": [],
        "oracles": {},
        "direction_stability": {},
        "non_inferiority_vs_utmos": {},
    }

    # Klassen-Namen deterministisch (einmalig) bestimmen.
    class_names = list(_degradations(np.zeros(1600, np.float32), 0).keys())

    # Sammeln: pro Klasse pro Orakel Liste der korrekten Richtung-Booleans.
    stability: dict[str, dict[str, list[bool]]] = {c: {o: [] for o in _ORACLES} for c in class_names}

    for pname, ref, degs in pairs:
        pair_row: dict[str, Any] = {"name": pname, "classes": {}}
        for cls in class_names:
            deg = degs[cls]
            cls_row: dict[str, Any] = {}
            for oname, fn in _ORACLES.items():
                m_ref = fn(ref, _SR)
                m_deg = fn(deg, _SR)
                if m_ref is not None and m_deg is not None:
                    cls_row[oname] = {"ref": round(m_ref, 4), "deg": round(m_deg, 4), "delta": round(m_ref - m_deg, 4)}
                    stability[cls][oname].append(m_ref > m_deg)
            pair_row["classes"][cls] = cls_row
        report["pairs"].append(pair_row)

    # Aggregation.
    for cls in class_names:
        report["direction_stability"][cls] = {}
        for oname in _ORACLES:
            vals = stability[cls][oname]
            n = len(vals)
            rate = float(np.mean(vals)) if n else None
            report["direction_stability"][cls][oname] = {
                "correct": int(sum(vals)),
                "total": n,
                "rate": round(rate, 4) if rate is not None else None,
            }

    # Non-inferiority: ΔRichtung = rate(oracle) - rate(utmos), je Klasse.
    for cls in class_names:
        utmos_rate = report["direction_stability"][cls].get("utmos", {}).get("rate")
        report["non_inferiority_vs_utmos"][cls] = {}
        for oname in _ORACLES:
            if oname == "utmos":
                continue
            orate = report["direction_stability"][cls][oname]["rate"]
            if utmos_rate is not None and orate is not None:
                report["non_inferiority_vs_utmos"][cls][oname] = round(orate - utmos_rate, 4)
            else:
                report["non_inferiority_vs_utmos"][cls][oname] = None

    out = Path(args.out) if args.out else _REPORT_DIR / f"{date.today().isoformat()}_kp2_musik_mos_orakel.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Report: {out}")
    print(json.dumps(report["direction_stability"], indent=2))
    print(json.dumps(report["non_inferiority_vs_utmos"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
