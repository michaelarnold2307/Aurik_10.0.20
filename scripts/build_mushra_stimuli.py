#!/usr/bin/env python3
"""MUSHRA-Stimuli-Builder (P1-4-Vorbereitung, Protokoll §10).

Baut das Pilot-Set (Seed 42) aus test_audio/ (Referenz = Input) und
output/supervised_run/ (Aurik-Ausgaben):
  - 6 Szenarien (rock worn, cassette wow, mp3 64k, cd clipped, reel dropout,
    jazz scratched als Ersatz für das fehlende e2e_jazz_studio2026 — im
    Manifest dokumentiert)
  - Bedingungen: reference, low_anchor (3,5-kHz-Tiefpass, −6 LUFS), aurik
  - BS.1770-5-Lautheitsabgleich aller Bedingungen auf die Referenz-LUFS
    (pyloudnorm, Block 0,4 s — Repo-Konvention export_transparency.py)
  - 500-ms-Raised-Cosine-Ein/Ausblendung an den Segmenträndern
  - 48 kHz, Mono; deterministisch (§G5 (GEBOTE.md)): Cut-Position per
    Energie-argmax, Reihenfolge per Seed 42, Manifest OHNE Zeitstempel

Quellen-Prüfung (§0c-Konsequenz 2026-09-14): Szenarien, deren Aurik-Ausgabe
fehlt oder leer ist (50-Byte-Header), werden im Manifest als
`quelle_fehlt` markiert und NICHT exportiert — kein stiller Leer-Stimulus.

Ausgabe: output/mushra_study/pilot/<szenario>/<bedingung>.wav + manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SR = 48000
_FADE_S = 0.500
_LOW_ANCHOR_DB = -6.0
_LOWPASS_HZ = 3500.0

# (id, input, aurik) — Pfade relativ zum Repo-Root.
_SCENARIOS: list[tuple[str, str, str]] = [
    ("rock_1970s_worn", "test_audio/vinyl/rock_1970s_worn.wav", "output/supervised_run/rock_1970s_worn.wav"),
    (
        "cassette_1980s_wow",
        "test_audio/tape/cassette_1980s_wow.wav",
        "output/supervised_run/cassette_1980s_wow_fix2.wav",
    ),
    (
        "mp3_64kbps_artifacts",
        "test_audio/digital/mp3_64kbps_artifacts.wav",
        "output/supervised_run/mp3_64kbps_artifacts_fix2.wav",
    ),
    ("cd_clipped_2000s", "test_audio/digital/cd_clipped_2000s.wav", "output/supervised_run/cd_clipped_2000s_fix2.wav"),
    ("reel_1940s_dropout", "test_audio/tape/reel_1940s_dropout.wav", "output/supervised_run/reel_1940s_dropout.wav"),
    (
        "jazz_1950s_scratched",
        "test_audio/vinyl/jazz_1950s_scratched.wav",
        "output/supervised_run/jazz_1950s_scratched_run2.wav",
    ),
]

_BUILD_VERSION = "mushra_pilot_v1_seed42"


def _load_mono_48k(path: Path) -> np.ndarray | None:
    """WAV/FLAC → Mono float32 @ 48 kHz; None bei leerer/defekter Datei."""
    from math import gcd

    import soundfile as sf
    from scipy.signal import resample_poly

    if not path.is_file():
        return None
    try:
        data, sr = sf.read(str(path), dtype="float32")
    except Exception:
        return None
    if data is None or np.asarray(data).size == 0:
        return None
    arr = np.asarray(data, dtype=np.float32)
    mono = arr.mean(axis=-1) if arr.ndim == 2 else arr
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    # Befund 2026-09-14: 50-Byte-Exports (Quality-Gate-Fail) enthalten nur den
    # WAV-Header + 2 Restsamples — als leere Quelle behandeln.
    if len(mono) < int(0.5 * _SR):
        return None
    if sr != _SR:
        g = gcd(int(sr), _SR)
        mono = resample_poly(mono, _SR // g, int(sr) // g).astype(np.float32)
    return mono


def _cut_energetic(x: np.ndarray, duration_s: float, guard_s: float = 2.0) -> np.ndarray:
    """Deterministischer Cut: 1-s-Fenster-Energie-argmax, Guard an den Rändern."""
    n = int(round(duration_s * _SR))
    if len(x) <= n:
        return x
    win = _SR  # 1-s-Energie-Fenster
    energy = np.convolve(x**2, np.ones(win, dtype=np.float64) / win, mode="valid")
    lo, hi = int(guard_s * _SR), max(int(guard_s * _SR) + 1, len(energy) - n - int(guard_s * _SR))
    if hi <= lo:
        start = (len(x) - n) // 2
    else:
        start = lo + int(np.argmax(energy[lo:hi]))
    return x[start : start + n]


def _edge_fade(x: np.ndarray) -> np.ndarray:
    """500-ms-Raised-Cosine-Ein/Ausblendung an den Segmenträndern."""
    n_fade = int(_FADE_S * _SR)
    if len(x) <= 2 * n_fade:
        return x
    fade = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_fade) / n_fade)).astype(np.float32)
    out = x.copy()
    out[:n_fade] *= fade
    out[-n_fade:] *= fade[::-1]
    return out


def lowpass_butter(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """Butterworth-4.-Ordnung, zero-phase (filtfilt)."""
    from scipy.signal import butter, filtfilt

    b, a = butter(4, cutoff_hz / (_SR / 2.0), btype="low")
    return filtfilt(b, a, x).astype(np.float32)


def _measure_lufs(x: np.ndarray) -> float | None:
    try:
        import pyloudnorm as pln

        meter = pln.Meter(rate=_SR, block_size=0.400)
        return float(meter.integrated_loudness(x.astype(np.float64)))
    except Exception:
        return None


def _gain_to_lufs(x: np.ndarray, target: float) -> tuple[np.ndarray, float]:
    """Gain auf Ziel-LUFS; Peak-Deckel 0.99 (erreichte LUFS wird zurückgegeben)."""
    measured = _measure_lufs(x)
    if measured is None:
        return x.astype(np.float32), float("nan")
    gain_db = target - measured
    out = x * (10.0 ** (gain_db / 20.0))
    peak = float(np.max(np.abs(out))) + 1e-12
    if peak > 0.99:
        out = out * (0.99 / peak)
    achieved = _measure_lufs(out)
    return out.astype(np.float32), float(achieved) if achieved is not None else float("nan")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _presentation_orders(seed: int, n_scenarios: int, n_conditions: int) -> list[list[int]]:
    """Latin-Square-Basis pro Szenario (deterministisch, Seed 42)."""
    rng = np.random.default_rng(seed)
    orders = []
    for s in range(n_scenarios):
        order = list(range(n_conditions))
        rng.shuffle(order)
        orders.append(order)
    return orders


def build_study(
    out_dir: Path,
    seed: int = 42,
    duration_s: float = 10.0,
    scenarios: list[tuple[str, str, str]] | None = None,
) -> dict:
    """Baut das Pilot-Set; liefert das Manifest (dict)."""
    import soundfile as sf

    scenarios = scenarios if scenarios is not None else _SCENARIOS
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "build_version": _BUILD_VERSION,
        "seed": seed,
        "target_duration_s": duration_s,
        "sr": _SR,
        "low_anchor": {"lowpass_hz": _LOWPASS_HZ, "lufs_offset_db": _LOW_ANCHOR_DB},
        "scenarios": [],
    }
    conditions = ["reference", "low_anchor", "aurik"]
    orders = _presentation_orders(seed, len(scenarios), len(conditions))

    for idx, (scenario_id, input_rel, aurik_rel) in enumerate(scenarios):
        input_path = _ROOT / input_rel
        aurik_path = _ROOT / aurik_rel
        entry: dict = {
            "id": scenario_id,
            "input": input_rel,
            "aurik": aurik_rel,
            "presentation_order": [conditions[i] for i in orders[idx]],
            "stimuli": [],
        }
        ref = _load_mono_48k(input_path)
        if ref is None:
            entry["status"] = "input_fehlt"
            manifest["scenarios"].append(entry)
            continue
        ref_cut = _edge_fade(_cut_energetic(ref, duration_s))
        ref_lufs = _measure_lufs(ref_cut)
        target = ref_lufs if ref_lufs is not None else -23.0
        sdir = out_dir / scenario_id
        sdir.mkdir(parents=True, exist_ok=True)

        def _emit(cond: str, audio: np.ndarray, source: str, sdir: Path, note: str = "") -> dict:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            wav = sdir / f"{cond}.wav"
            sf.write(str(wav), audio.astype(np.float32), _SR)
            stim = {
                "condition": cond,
                "path": str(wav.relative_to(out_dir)),
                "source": source,
                "duration_s": round(len(audio) / _SR, 3),
                "integrated_lufs": round(float(_measure_lufs(audio) or float("nan")), 1),
                "sha256": _sha256(wav),
            }
            if note:
                stim["note"] = note
            return stim

        entry["stimuli"].append(_emit("reference", ref_cut, input_rel, sdir, note="Ziellautheit"))
        anchor_raw = lowpass_butter(ref_cut, _LOWPASS_HZ)
        anchor, anchor_lufs = _gain_to_lufs(anchor_raw, target + _LOW_ANCHOR_DB)
        entry["stimuli"].append(
            _emit("low_anchor", anchor, input_rel, sdir, note=f"LP {_LOWPASS_HZ} Hz, Ziel {_LOW_ANCHOR_DB} LUFS")
        )
        aurik = _load_mono_48k(aurik_path)
        if aurik is None:
            entry["status"] = "aurik_leer"
        else:
            aurik_cut = _edge_fade(_cut_energetic(aurik, duration_s))
            aurik_aligned, _ = _gain_to_lufs(aurik_cut, target)
            entry["stimuli"].append(_emit("aurik", aurik_aligned, aurik_rel, sdir))
            entry["status"] = "ok"
        manifest["scenarios"].append(entry)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default="output/mushra_study/pilot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out_dir = _ROOT / args.out
    if args.dry_run:
        for scenario_id, inp, aur in _SCENARIOS:
            status_in = "OK" if (_ROOT / inp).is_file() else "FEHLT"
            status_au = "OK" if _load_mono_48k(_ROOT / aur) is not None else "LEER/FEHLT"
            print(f"{scenario_id}: input={status_in} aurik={status_au}")
        return 0
    manifest = build_study(out_dir, seed=args.seed, duration_s=args.duration)
    n_ok = sum(1 for s in manifest["scenarios"] if s.get("status") == "ok")
    n_total = len(manifest["scenarios"])
    print(f"Studien-Set gebaut: {n_ok}/{n_total} Szenarien mit Aurik-Stimulus → {out_dir}")
    for s in manifest["scenarios"]:
        print(f"  {s['id']}: {s.get('status', 'ok')} ({len(s.get('stimuli', []))} Stimuli)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
