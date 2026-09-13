"""SOTA-MuQ: Richtungs-Validierung Plugin vs. 1:1-MuQ-Eval auf MUSDB-Paaren.

Vergleicht pro Paar (ref vs. Degradation) die MOS-Differenz Δ = MOS(ref) − MOS(deg)
zweier Schätzer auf identischem Material:

  (a) 1:1-Referenz: MusicQualityModel (models/muq_eval/src, A1-Checkpoint
      best_model.pt strict) — Audio-Kette: erste 10 s @ 44,1 kHz → librosa-Resample
      auf 24 kHz. Dies ist die Kette, die auf MUSDB richtungs-korrekt war
      (noise10 Δ+3.337, noise0 Δ+3.270).
  (b) Plugin: plugins.muq_plugin.estimate_muq_mos auf dem vollen 30-s-Clip —
      interne Kette seit dem Audio-Chain-Fix: _mos_eval_window (erste 10 s,
      librosa-Resample, 24 kHz) = Kette (a).

Richtung gilt als korrekt, wenn sign(Δ_a) == sign(Δ_b) und |Δ_a| > 0.1.
Band12 ist bei 24 kHz-Target unhoerbar (Nyquist 12 kHz) — Referenz-Neutralwert.

Determinismus (§G5 (GEBOTE.md)): feste Seed-42-Auswahl, keine Zeitquellen.
Report: docs/reports/current/<datum>_muq_plugin_direction.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MUSDB = _ROOT / "data" / "musdb18hq" / "test"
_REPORT_DIR = _ROOT / "docs" / "reports" / "current"

_SR = 44100
_CLIP_S = 30
_FIRST_S = 10  # 1:1-Kette: erste 10 s
_TARGET_SR = 24000


def _load_mixture(track_dir: Path, seconds: int, seed: int) -> np.ndarray:
    _wf = wavfile.read(str(track_dir / "mixture.wav"))
    if not isinstance(_wf, tuple) or len(_wf) < 2:  # Bug 12: Index-basiert statt Unpacking
        raise ValueError("wavfile.read() ohne (sr, data)-Tupel")
    sr = int(_wf[0])
    wav = np.asarray(_wf[1])
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / 32768.0
    else:
        wav = wav.astype(np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    rng = np.random.RandomState(seed)
    n_need = seconds * sr
    if wav.size <= n_need:
        return wav
    start = rng.randint(0, wav.size - n_need)
    _clip: np.ndarray = np.asarray(wav[start : start + n_need], dtype=np.float32)
    return _clip


def _bandlimit(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    b, a = butter(5, cutoff_hz / (0.5 * _SR), btype="low")
    _y: np.ndarray = np.asarray(filtfilt(b, a, x), dtype=np.float32)
    return _y


def _add_noise(x: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    p_sig = float(np.mean(x**2)) + 1e-12
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    noise = rng.randn(x.size).astype(np.float32) * np.sqrt(p_noise)
    _y: np.ndarray = np.asarray(x + noise, dtype=np.float32)
    return _y


def _peak_normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) + 1e-12
    _y: np.ndarray = np.asarray(x * (peak / m), dtype=np.float32)
    return _y


def _first10_resample24k(x: np.ndarray) -> np.ndarray:
    """1:1-Audio-Kette: erste 10 s @ 44,1 kHz → librosa-Resample auf 24 kHz."""
    import librosa

    seg = x[: _FIRST_S * _SR]
    _y: np.ndarray = np.asarray(librosa.resample(seg, orig_sr=_SR, target_sr=_TARGET_SR), dtype=np.float32)
    return _y


def _build_ref_model() -> tuple[object, object, object]:
    """MusicQualityModel (A1) + torch/cuda-Handles — 1:1 wie validiert."""
    import torch
    from omegaconf import OmegaConf

    import plugins.muq_plugin as _mq

    _ = _mq
    from src.model import MusicQualityModel  # models/muq_eval/src (sys.path via Plugin)

    snap_base = Path.home() / ".cache" / "huggingface" / "hub" / "models--zhudi2825--MuQ-Eval-A1" / "snapshots"
    snap_dirs = sorted(snap_base.glob("*"))
    if not snap_dirs:
        raise FileNotFoundError(f"A1-Snapshot fehlt: {snap_base}")
    snap = snap_dirs[0]
    cfg = OmegaConf.merge(OmegaConf.load(str(snap / "base.yaml")), OmegaConf.load(str(snap / "config.yaml")))
    # best_model.pt enthält die gepickelte Trainings-Config (OmegaConf-Graph) —
    # weights_only=False nach Repo-Konvention für vertrauenswürdige Dritt-Checkpoints
    # (wie bsr317_torch_rocm.py:67); Quelle: HF zhudi2825/MuQ-Eval-A1.
    state = torch.load(str(snap / "best_model.pt"), map_location="cpu", weights_only=False)  # nosec B614
    model = MusicQualityModel(cfg)
    missing, unexpected = model.load_state_dict(state["model_state"], strict=False)
    assert not missing and not unexpected, f"State-Drift: missing={missing}, unexpected={unexpected}"
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    return model, torch, dev


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks", type=int, default=3, help="Anzahl MUSDB-Test-Tracks")
    ap.add_argument("--seed", type=int, default=42, help="Seed fuer Fensterwahl")
    ap.add_argument("--seconds", type=int, default=_CLIP_S)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    if not _MUSDB.is_dir():
        print(f"MUSDB18HQ-Testset fehlt: {_MUSDB}")
        return 2
    tracks = sorted(p for p in _MUSDB.glob("*") if p.is_dir())[: args.tracks]
    if not tracks:
        print("Keine Tracks gefunden")
        return 2

    import plugins.muq_plugin as mq

    model, torch, dev = _build_ref_model()
    variants = ["ref", "band12", "band8", "noise10", "noise0"]

    rows: dict[str, dict[str, float | None]] = {v: {"delta_ref": None, "delta_plugin": None} for v in variants}
    per_track: list[dict[str, object]] = []

    with torch.no_grad():
        for ti, track in enumerate(tracks):
            x = _load_mixture(track, args.seconds, args.seed + ti)
            peak = float(np.max(np.abs(x))) + 1e-12
            ref = np.asarray(x * (0.9 / peak), dtype=np.float32)  # fixe Skalierung fuer alle Varianten
            variant_audio: dict[str, np.ndarray] = {"ref": ref}
            variant_audio["band12"] = _bandlimit(ref, 12000.0)
            variant_audio["band8"] = _bandlimit(ref, 8000.0)
            variant_audio["noise10"] = _add_noise(ref, 10.0, seed=args.seed + ti)
            variant_audio["noise0"] = _add_noise(ref, 0.0, seed=args.seed + ti)

            mos_ref: dict[str, float | None] = {}
            mos_plugin: dict[str, float | None] = {}
            for v in variants:
                aud = variant_audio[v]
                # (a) 1:1-Referenz: erste 10 s → librosa 24 kHz → [1, 240000]
                wav24 = _first10_resample24k(aud)
                wav_t = torch.from_numpy(wav24).unsqueeze(0).to(dev)
                pred = model(wav_t)
                mos_ref[v] = float(pred["MI"].item())
                # (b) Plugin auf dem vollen 30-s-Clip (interne Kette = (a))
                mos_plugin[v] = mq.estimate_muq_mos(aud, _SR)
            row_track: dict[str, object] = {"track": track.name, "ref": mos_ref, "plugin": mos_plugin}
            per_track.append(row_track)
            print(f"[{track.name}] ref-MOS: {mos_ref} | plugin-MOS: {mos_plugin}")

    for v in variants:
        d_ref: list[float] = []
        d_plug: list[float] = []
        for t in per_track:
            r = t["ref"]
            p = t["plugin"]
            assert isinstance(r, dict) and isinstance(p, dict)
            if r["ref"] is not None and r[v] is not None:
                d_ref.append(float(r["ref"]) - float(r[v]))
            if p["ref"] is not None and p[v] is not None:
                d_plug.append(float(p["ref"]) - float(p[v]))
        rows[v] = {
            "delta_ref": float(np.mean(d_ref)) if d_ref else None,
            "delta_plugin": float(np.mean(d_plug)) if d_plug else None,
        }

    direction_ok = 0
    direction_total = 0
    direction_neutral = 0
    for v in variants:
        dr = rows[v]["delta_ref"]
        dp = rows[v]["delta_plugin"]
        if dr is not None and dp is not None:
            if abs(dr) <= 0.1:
                direction_neutral += 1  # neutral (z. B. band12) — keine Richtungswertung
                continue
            direction_total += 1
            direction_ok += 1 if (dr > 0) == (dp > 0) else 0

    report = {
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/validate_muq_plugin_direction.py",
        "ref_model": "MuQ-Eval-A1 (best_model.pt, MusicQualityModel, erste 10 s + librosa 24 kHz)",
        "plugin": "plugins.muq_plugin.estimate_muq_mos (_mos_eval_window, erste 10 s + librosa 24 kHz)",
        "tracks": [str(t.name) for t in tracks],
        "per_track": per_track,
        "deltas": rows,
        "direction_ok": direction_ok,
        "direction_total": direction_total,
        "direction_neutral": direction_neutral,
    }
    out = Path(args.out) if args.out else _REPORT_DIR / f"{_dt.date.today().isoformat()}_muq_plugin_direction.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nDelta-Tabelle: {json.dumps(rows, indent=2)}")
    print(f"Richtung: {direction_ok}/{direction_total} korrekt ({direction_neutral} neutral) — Report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
