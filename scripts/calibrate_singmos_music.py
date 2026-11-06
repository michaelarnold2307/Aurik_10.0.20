"""F10 (D-K4-6, SOTA-M11): SingMOS-Kalibrierung auf verarbeiteten Musik-Vocals.

Ziel (Roadmap F10): Das deployte SingMOS Pro ONNX (`models/singmos/singmos_pro.onnx`,
sprach-trainiert) trifft auf Musik-Vocals mit Effektketten (Hall, Sättigung,
Bandlimit, Codec-Artefakte, Misch-Pegel) nicht die erwartete MOS-Richtung.
Dieses Skript misst den Produktions-Score-Pfad (`plugins.versa_plugin.score_mos`)
auf deterministisch synthetisierten clean/verarbeitet-Paaren aus MUSDB18-HQ-Vocals
und legt ein Kalibrier-Artefakt ab: per-Kette Statistik + affine Abbildung
raw -> kalibriert (Ground-Truth = Effektstärke, sauber >= 0 und monoton
fallend). Der Konsument (VersaSingMOS-Plugin) wird hier NICHT verdrahtet —
Flag-Flip ist verboten; Read-Side-Bedarf dokumentiert der Report.

Vertrag GPU-Kette 4 (scripts/gpu_kette4_modellkoepfe.sh, Job R4):
    nice -n 15 <venv> -B scripts/calibrate_singmos_music.py \
        --musdb-root data/musdb18hq --out-dir models/singmos/calibration

Determinismus (§G5 (copilot-instructions.md)): Seed 42, sortierte Dateilisten,
seeded RNG für Synthese; keine `time.time()`-Entscheidungslogik.
Silent-Failure-Verbot (§V6 (copilot-instructions.md)): jede Degradation wird mit
`logger.warning` + Begründung gemeldet; NaN/Inf-Scores werden wie im
Produktionspfad auf 3.0 gestellt und gezählt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_SCHEMA = "aurik.singmos_calibration/1"
_SR_MOS = 16_000  # Produktionspfad score_mos: I/O @ 16 kHz (Mono)


# ── Effektketten (deterministisch, seed-gebunden) ────────────────────────────
def _mono_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """Stereo (N, C) -> Mono @ 16 kHz; layout-sicher (AGENTS.md §3)."""
    if x.ndim == 2 and x.shape[1] == 2:
        x = x.mean(axis=1)
    elif x.ndim == 2:
        x = x[:, 0]
    x = x.astype(np.float64)
    if sr != _SR_MOS:
        g = max(1, int(np.gcd(sr, _SR_MOS)))
        x = signal.resample_poly(x, _SR_MOS // g, sr // g)
    x -= np.mean(x)
    peak = np.max(np.abs(x))
    if peak > 1e-9:
        x = x / peak * 0.9
    return x.astype(np.float32)


def _hall(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Deterministische FFT-Faltung mit exponentiell fallendem Rauschen."""
    t60 = float(rng.uniform(0.5, 1.6))
    ir_len = int(t60 * _SR_MOS)
    t = np.arange(ir_len) / _SR_MOS
    ir = rng.standard_normal(ir_len) * np.exp(-3.0 * t / t60)
    ir /= np.sqrt(np.sum(ir**2)) + 1e-12
    n = x.shape[0] + ir_len - 1
    n_fft = 1 << (n - 1).bit_length()
    wet = np.fft.irfft(np.fft.rfft(x, n_fft) * np.fft.rfft(ir, n_fft), n_fft)[: x.shape[0]]
    return (0.6 * x + 0.4 * wet).astype(np.float32)


def _saettigung(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Weiche Sättigung (tanh-Drive, seed-gebunden) — kein Hard-Clamp (§III DSP)."""
    drive = float(rng.uniform(2.0, 5.0))
    return np.tanh(drive * x).astype(np.float32) * 0.9


def _bandlimit(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Cutoff strikt unter Nyquist (8 kHz): Wn in (0, 1) — Produktionsbefund: ValueError bei Wn >= 1.
    cutoff = float(rng.uniform(3_000.0, 7_500.0))
    b, a = signal.butter(6, cutoff / (_SR_MOS / 2.0), btype="low")
    return signal.lfilter(b, a, x).astype(np.float32)


def _codec(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bandlimit + Dezimierung/Interpolation als Codec-Ersatz (deterministisch)."""
    target_sr = int(rng.choice([8_000, 11_025]))
    g = int(np.gcd(_SR_MOS, target_sr))
    down = signal.resample_poly(x, target_sr // g, _SR_MOS // g)
    up = signal.resample_poly(down, _SR_MOS // g, target_sr // g)
    return signal.lfilter(*signal.butter(6, 0.9 * target_sr / _SR_MOS, btype="low"), up).astype(np.float32)


def _mixpegel(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gain_db = float(rng.uniform(-18.0, -6.0))
    gain = 10.0 ** (gain_db / 20.0)
    floor = rng.standard_normal(x.shape[0]).astype(np.float32) * 1e-3
    return (x * gain + floor).astype(np.float32)


_EFFECTS = {
    "hall": _hall,
    "saettigung": _saettigung,
    "bandlimit": _bandlimit,
    "codec": _codec,
    "mixpegel": _mixpegel,
}
# Effektstärke je Kette (Ground-Truth-Ordnung, sinkend):
_EFFECT_STRENGTH = {"hall": 1, "saettigung": 2, "bandlimit": 3, "codec": 4, "mixpegel": 5}


# ── Datenzugriff ─────────────────────────────────────────────────────────────
def _vocal_tracks(musdb_root: Path) -> list[Path]:
    train = Path(musdb_root) / "train"
    if not train.is_dir():
        return []
    return sorted(p for p in train.glob("*/vocals.wav"))


def _load_vocal(path: Path) -> tuple[np.ndarray, int]:
    out = wavfile.read(path)
    if isinstance(out, tuple):  # Bug 12, §V36 (VERBOTEN.md): index-basierte Entpackung mit Prüfung
        sr, data = out
    else:
        sr, data = _SR_MOS, out
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    return data.astype(np.float32) / 32768.0, sr


# ── Kern (testbar ohne Modell) ───────────────────────────────────────────────
def score_pair(clean: np.ndarray, processed: np.ndarray, score_fn) -> dict:
    """Misst clean + verarbeitet über den injizierten Produktions-Score.

    `score_fn(audio_mono_16k: np.ndarray) -> float` liefert den rohen MOS.
    NaN/Inf wird wie im Produktionspfad auf 3.0 gestellt (§V6 (copilot-instructions.md) mit Zählung).
    """
    out = {"clean": None, "processed": None, "nan_fallback": 0}
    for key, sig in (("clean", clean), ("processed", processed)):
        mos = float(score_fn(sig))
        if not np.isfinite(mos):
            mos = 3.0
            out["nan_fallback"] += 1
            logger.warning("SingMOS-Score nicht endlich — auf 3.0 gestellt (§V6 (copilot-instructions.md)): %s", key)
        out[key] = mos
    return out


def fit_calibration(pair_rows: list[dict]) -> dict:
    """Affine Abbildung raw -> kalibriert auf Grund-Ordnung (Effektstärke)."""

    def _stats(rows: list[dict], key: str) -> dict:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        vals = np.where(np.isfinite(vals), vals, 3.0)
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": int(vals.size)}

    by_chain: dict[str, list[dict]] = {}
    for r in pair_rows:
        by_chain.setdefault(r["chain"], []).append(r)
    chain_stats = {c: _stats(rows, "processed") for c, rows in by_chain.items()}
    clean_stats = _stats(pair_rows, "clean")

    # Ground-Truth-Richtung: stärkere Kette => Score soll sinken.
    means = [(c, s["mean"]) for c, s in chain_stats.items()]
    means.sort(key=lambda t: _EFFECT_STRENGTH.get(t[0], 99))
    mono_violations = sum(1 for i in range(len(means) - 1) if means[i][1] < means[i + 1][1])
    per_pair_violations = sum(
        1 for r in pair_rows if np.isfinite(r["processed"]) and np.isfinite(r["clean"]) and r["processed"] > r["clean"]
    )

    # Affine Abbildung: kalibriert = a * raw + b, so dass clean-Mittel auf
    # 4.2 und die stärkste Kette auf 1.8 skaliert (Grenzen [1, 5]).
    lo = min((s["mean"] for s in chain_stats.values()), default=3.0)
    hi = clean_stats["mean"]
    span = max(hi - lo, 1e-6)
    a = max(0.0, (4.2 - 1.8) / span)
    b = 4.2 - a * hi
    return {
        "a": float(a),
        "b": float(b),
        "clean": clean_stats,
        "chains": chain_stats,
        "monotonicity_violations": int(mono_violations),
        "per_pair_violations": int(per_pair_violations),
        "n_pairs": int(len(pair_rows)),
    }


def _manifest(artefact: dict, path: Path, seed: int) -> dict:
    sha = hashlib.sha256(json.dumps(artefact, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {
        "schema": _SCHEMA,
        "file": path.name,
        "sha256": sha,
        "seed": seed,
        "source": "scripts/calibrate_singmos_music.py (F10, D-K4-6)",
    }


def build_pairs(
    musdb_root: Path, seed: int, pairs_per_effect: int, rng: np.random.Generator | None = None
) -> list[dict]:
    """Synthetisiert deterministische clean/verarbeitet-Paare (ohne Modell)."""
    tracks = _vocal_tracks(musdb_root)
    if not tracks:
        raise SystemExit("MUSDB18-HQ-Verzeichnis nicht gefunden: --musdb-root zeigt nicht auf data/musdb18hq")
    if rng is None:
        rng = np.random.default_rng(seed)
    chosen = rng.choice(tracks, size=min(len(tracks), max(4, pairs_per_effect)), replace=False)
    rows: list[dict] = []
    for track in sorted(chosen, key=str):
        x, sr = _load_vocal(track)
        clean = _mono_16k(x, sr)
        for chain in sorted(_EFFECTS, key=lambda c: _EFFECT_STRENGTH[c]):
            for _ in range(pairs_per_effect):
                processed = _EFFECTS[chain](clean, rng)
                rows.append(
                    {
                        "track": track.parent.name,
                        "chain": chain,
                        "strength": _EFFECT_STRENGTH[chain],
                        "clean": clean,
                        "processed": processed,
                    }
                )
    return rows


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="SingMOS-Kalibrierung auf Musik-Vocals (F10, D-K4-6)")
    ap.add_argument("--musdb-root", required=True, help="Pfad zu MUSDB18-HQ (data/musdb18hq)")
    ap.add_argument("--out-dir", required=True, help="Zielverzeichnis (models/singmos/calibration)")
    ap.add_argument("--seed", type=int, default=42, help="Determinismus-Seed (§G5 (copilot-instructions.md))")
    ap.add_argument("--pairs-per-effect", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="1 Paar je Kette statt voller Umfang")
    args = ap.parse_args(argv)

    seed = int(args.seed)
    rng = np.random.default_rng(seed)
    rows = build_pairs(Path(args.musdb_root), seed, 1 if args.smoke else args.pairs_per_effect, rng)
    logger.info("Paare synthetisiert: %d (Seed %d)", len(rows), seed)

    # Produktions-Score-Pfad (lazy: ONNX-Session wird nur hier geladen).
    from plugins.versa_plugin import score_mos  # §V4 (copilot-instructions.md): nur Plugin-Pfad

    def _score(sig: np.ndarray) -> float:
        return float(score_mos(sig, _SR_MOS).mos)

    scored = []
    for row in rows:
        res = score_pair(row["clean"], row["processed"], _score)
        scored.append({k: row[k] for k in ("track", "chain", "strength")} | res)

    calibration = fit_calibration(scored)
    calibration["schema"] = _SCHEMA
    calibration["seed"] = seed
    calibration["smoke"] = bool(args.smoke)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    art_path = out_dir / "calibration.json"
    art_path.write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(_manifest(calibration, art_path, seed), indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(
        "Kalibrierung geschrieben: %s (a=%.4f b=%.4f, Monotonie-Verletzungen=%d)",
        art_path,
        calibration["a"],
        calibration["b"],
        calibration["monotonicity_violations"],
    )
    logger.info(
        "KEIN Flag-Flip, KEINE Konsument-Verdrahtung (§V8 (copilot-instructions.md)) — Read-Side-Bedarf im Report dokumentieren"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
