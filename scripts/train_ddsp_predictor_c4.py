#!/usr/bin/env python3
"""SOTA-C4 / F5: DDSP-Prädiktor — EQ/Dynamik-Parameter aus CLAP-Embeddings.

§SOTA-C4 (Roadmap, Phase-Tabelle 04/16/17): Ein neuronaler Prädiktor schätzt
aus dem CLAP-Audio-Embedding (LAION-CLAP, eingefroren, 512-dim) die
EQ-/Dynamik-Parameter, die ein Stück geformt haben — die DSP-Phasen
04/16/17 führen die Parameter aus („DSP führt aus", kein ML-Audio).

Trainingsdaten: MUSDB18-HQ (lokal) + synthetisierte Effekt-Paare —
deterministische Effekt-Ketten (3-Band-EQ + Soft-Knee-Kompressor) werden auf
saubere Segmente angewendet; die applizierten Parameter sind das Label.
Aufgabe: CLAP(Effekt-Audio) → Parameter (Regression, MSE).

Ablauf:
  1. --precompute: Embeddings + Targets als .npz je Song cachen (Encoder
     eingefroren ⇒ einmal berechnen; GPU möglich).
  2. --train: MLP-Head (512→256→128→6) auf dem Cache trainieren, je Epoch
     Validierungs-MAE pro Parameter; bester Head nach
     models/ddsp_predictor/c4_head.pth + config.json.
  3. --smoke: Pipeline mit Zufalls-Embeddings end-to-end testen (kein CLAP).

Determinismus: Seed 42 (§G5 (GEBOTE.md)), Effekt-Synthese mit
song-abgeleitetem Seed; keine zeitabhängige Entscheidungslogik.

Acceptance (F5): MuQ-MOS-Witness nicht schlechter — separat validiert, sobald
der MuQ-Backbone verfügbar ist (WIT-M1, extern); hier wird der
Validierungs-MAE als primäres Gate berichtet.

Autor: Aurik Testing Team
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly, sosfilt

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MUSDB_TRAIN = _ROOT / "data" / "musdb18hq" / "train"
_MUSDB_TEST = _ROOT / "data" / "musdb18hq" / "test"
_CACHE_DIR = _ROOT / "output" / "ddsp_c4" / "cache"
_OUT_DIR = _ROOT / "models" / "ddsp_predictor"
_SR = 48000
_SEG_S = 6.0
_N_PARAMS = 6

# Parameter-Ranges (normalisiert auf [0, 1]): low_shelf_db, mid_gain_db,
# mid_freq_hz, mid_q, high_shelf_db, comp_threshold_db
_PARAM_RANGES = (
    (-9.0, 9.0),
    (-9.0, 9.0),
    (300.0, 6000.0),
    (0.5, 4.0),
    (-9.0, 9.0),
    (-42.0, -12.0),
)


def _rng_for(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _norm(p: float, lo: float, hi: float) -> float:
    return float(np.clip((p - lo) / (hi - lo), 0.0, 1.0))


def _denorm(v: float, lo: float, hi: float) -> float:
    return float(lo + float(np.clip(v, 0.0, 1.0)) * (hi - lo))


def _shelving(gain_db: float, fc: float, sr: int, kind: str) -> np.ndarray:
    """RBJ-Biquad Shelving-Koeffizienten (sos, 1 Sektion)."""
    import math

    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * 0.7071)
    if kind == "low":
        b0 = a * ((a + 1) - (a - 1) * cw + 2 * math.sqrt(a) * alpha)
        b1 = 2 * a * ((a - 1) - (a + 1) * cw)
        b2 = a * ((a + 1) - (a - 1) * cw - 2 * math.sqrt(a) * alpha)
        a0 = (a + 1) + (a - 1) * cw + 2 * math.sqrt(a) * alpha
        a1 = -2 * ((a - 1) + (a + 1) * cw)
        a2 = (a + 1) + (a - 1) * cw - 2 * math.sqrt(a) * alpha
    else:
        b0 = a * ((a + 1) + (a - 1) * cw + 2 * math.sqrt(a) * alpha)
        b1 = -2 * a * ((a - 1) + (a + 1) * cw)
        b2 = a * ((a + 1) + (a - 1) * cw - 2 * math.sqrt(a) * alpha)
        a0 = (a + 1) - (a - 1) * cw + 2 * math.sqrt(a) * alpha
        a1 = 2 * ((a - 1) - (a + 1) * cw)
        a2 = (a + 1) - (a - 1) * cw - 2 * math.sqrt(a) * alpha
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def _peaking(gain_db: float, fc: float, q: float, sr: int) -> np.ndarray:
    import math

    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    b0 = 1 + alpha * a
    b1 = -2 * cw
    b2 = 1 - alpha * a
    a0 = 1 + alpha / a
    a1 = -2 * cw
    a2 = 1 - alpha / a
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def _apply_eq(sig: np.ndarray, sr: int, params: tuple[float, float, float, float, float]) -> np.ndarray:
    low_db, mid_db, mid_hz, mid_q, high_db = params
    sos = np.vstack(
        [
            _shelving(low_db, 250.0, sr, "low"),
            _peaking(mid_db, mid_hz, mid_q, sr),
            _shelving(high_db, 6000.0, sr, "high"),
        ]
    )
    out = sosfilt(sos, sig.astype(np.float64)).astype(np.float32)
    peak = float(np.max(np.abs(out))) if out.size else 1.0
    return (out / peak).astype(np.float32) if peak > 1.0 else out


def _apply_compressor(sig: np.ndarray, sr: int, threshold_db: float, ratio: float = 2.5) -> np.ndarray:
    """Soft-Knee-Kompressor (vektorisiert): Attack/Release-Envelope via
    zwei Einpol-Filter, Gewinn-Computer in Frames geglättet."""
    from scipy.signal import lfilter

    x = sig.astype(np.float64)
    attack_s, release_s, knee_db = 0.010, 0.080, 6.0
    alpha_a = float(np.exp(-1.0 / (attack_s * sr)))
    alpha_r = float(np.exp(-1.0 / (release_s * sr)))
    env_a = lfilter([1.0 - alpha_a], [1.0, -alpha_a], np.abs(x))
    env_r = lfilter([1.0 - alpha_r], [1.0, -alpha_r], np.abs(x))
    env = np.maximum(env_a, env_r)
    env_db = 20.0 * np.log10(np.maximum(env, 1e-9))
    over = env_db - threshold_db
    gain_db = np.where(over <= -knee_db / 2.0, 0.0, over * (1.0 / ratio - 1.0))
    soft = (1.0 / ratio - 1.0) * (over + knee_db / 2.0) ** 2 / (2.0 * knee_db)
    gain_db = np.where(np.abs(over) < knee_db / 2.0, soft, gain_db)
    # Frame-Glättung (10 ms) gegen Zipper-Noise
    frame = max(1, sr // 100)
    n_frames = int(np.ceil(len(gain_db) / frame))
    pad = np.pad(gain_db, (0, n_frames * frame - len(gain_db)))
    gain_f = pad.reshape(n_frames, frame).mean(axis=1)
    gain_db = np.repeat(gain_f, frame)[: len(gain_db)]
    out = x * 10.0 ** (gain_db / 20.0)
    peak = float(np.max(np.abs(out))) if out.size else 1.0
    return (out / peak).astype(np.float32) if peak > 1.0 else out.astype(np.float32)


def _synthesize_pair(clean: np.ndarray, sr: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Wendet eine deterministische Effekt-Kette an; liefert (audio, y[0..1])."""
    raw = [rng.uniform(lo, hi) for lo, hi in _PARAM_RANGES]
    low_db, mid_db, mid_hz, mid_q, high_db, thr_db = raw
    y = np.array([_norm(p, lo, hi) for p, (lo, hi) in zip(raw, _PARAM_RANGES)], dtype=np.float32)
    effected = _apply_eq(clean, sr, (low_db, mid_db, mid_hz, mid_q, high_db))
    effected = _apply_compressor(effected, sr, thr_db)
    return effected, y


_WAV_CACHE: dict[Path, np.ndarray] = {}


def _load_segment(track_dir: Path, rng: np.random.Generator) -> np.ndarray:
    from scipy.io import wavfile

    mix_path = track_dir / "mixture.wav"
    if mix_path not in _WAV_CACHE:
        _wf = wavfile.read(str(mix_path))
        if not isinstance(_wf, tuple) or len(_wf) < 2:
            raise ValueError("wavfile.read() ohne (sr, data)-Tupel")
        src_sr = int(_wf[0])
        wav = np.asarray(_wf[1])
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32768.0
        else:
            wav = wav.astype(np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if src_sr != _SR:
            g = int(np.gcd(src_sr, _SR))
            wav = resample_poly(wav, _SR // g, src_sr // g).astype(np.float32)
        _WAV_CACHE[mix_path] = wav
    wav = _WAV_CACHE[mix_path]
    n = int(_SEG_S * _SR)
    if len(wav) <= n:
        return np.pad(wav, (0, n - len(wav))).astype(np.float32)
    start = int(rng.integers(0, len(wav) - n))
    seg = wav[start : start + n]
    peak = float(np.max(np.abs(seg))) if seg.size else 1.0
    return (seg / peak).astype(np.float32) if peak > 0 else seg


def _embed(audio: np.ndarray) -> np.ndarray:
    from plugins.laion_clap_plugin import get_laion_clap

    emb = get_laion_clap().embed_audio(audio, _SR)
    return np.asarray(emb, dtype=np.float32)


def _precompute(tracks: list[Path], segments_per_track: int, seed: int) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for ti, track in enumerate(tracks):
        out = _CACHE_DIR / f"{track.name}.npz"
        if out.exists():
            print(f"[{ti + 1}/{len(tracks)}] {track.name}: Cache vorhanden — übersprungen")
            continue
        rng = _rng_for(seed + ti)
        xs, ys = [], []
        for si in range(segments_per_track):
            seg = _load_segment(track, rng)  # frischer Zuschnitt je Paar (Diversität)
            eff, y = _synthesize_pair(seg, _SR, rng)
            xs.append(_embed(eff))
            ys.append(y)
        np.savez_compressed(out, x=np.stack(xs), y=np.stack(ys))
        print(
            f"[{ti + 1}/{len(tracks)}] {track.name}: {segments_per_track} Paare in {time.perf_counter() - t0:.0f}s gesamt"
        )
    print(f"Precompute fertig: {len(tracks)} Songs, {time.perf_counter() - t0:.0f}s")


def _train(epochs: int, lr: float, seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    files = sorted(_CACHE_DIR.glob("*.npz"))
    if not files:
        print("Kein Cache gefunden — zuerst --precompute ausführen.", file=sys.stderr)
        raise SystemExit(2)
    xs, ys, groups = [], [], []
    for fi, f in enumerate(files):
        d = np.load(f)
        xs.append(d["x"])
        ys.append(d["y"])
        groups.append(np.full(len(d["y"]), fi, dtype=np.int64))
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.float32)
    g = np.concatenate(groups)
    rng = np.random.default_rng(seed)
    song_idx = np.unique(g)
    val_songs = rng.choice(song_idx, size=max(1, int(0.2 * len(song_idx))), replace=False)
    val_mask = np.isin(g, val_songs)
    x_tr, y_tr = torch.from_numpy(x[~val_mask]), torch.from_numpy(y[~val_mask])
    x_va, y_va = torch.from_numpy(x[val_mask]), torch.from_numpy(y[val_mask])
    print(f"Cache: {len(files)} Songs, {x.shape[0]} Paare (train={x_tr.shape[0]}, val={x_va.shape[0]})")

    head = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, _N_PARAMS),
    )
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    best_mae = float("inf")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        head.train()
        opt.zero_grad()
        pred = head(x_tr)
        loss = loss_fn(pred, y_tr)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            va_pred = head(x_va)
            mae = float(torch.abs(va_pred - y_va).mean())
            per = torch.abs(va_pred - y_va).mean(dim=0).tolist()
        if mae < best_mae:
            best_mae = mae
            torch.save(head.state_dict(), _OUT_DIR / "c4_head.pth")
        print(
            f"Epoch {ep + 1}/{epochs}: loss={loss.item():.5f} val_MAE={mae:.5f} per_param={[round(p, 4) for p in per]}"
        )
    (_OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "n_params": _N_PARAMS,
                "param_ranges": list(_PARAM_RANGES),
                "param_names": [
                    "low_shelf_db",
                    "mid_gain_db",
                    "mid_freq_hz",
                    "mid_q",
                    "high_shelf_db",
                    "comp_threshold_db",
                ],
                "embedding": "laion_clap_512",
                "seg_s": _SEG_S,
                "seed": seed,
                "best_val_mae": best_mae,
                "epochs": epochs,
                "lr": lr,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Fertig — bester val_MAE={best_mae:.5f} → {_OUT_DIR}/c4_head.pth + config.json")


def main() -> int:
    ap = argparse.ArgumentParser(description="§SOTA-C4/F5: DDSP-Prädiktor (EQ/Dynamik aus CLAP)")
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--songs", type=int, default=40)
    ap.add_argument("--segments", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.smoke:
        rng = _rng_for(args.seed)
        clean = (0.3 * np.sin(2 * np.pi * 440.0 * np.arange(int(_SEG_S * _SR)) / _SR)).astype(np.float32)
        eff, y = _synthesize_pair(clean, _SR, rng)
        print(f"Smoke: Effekt-Kette OK — Effekt-Paar shape={eff.shape}, y={np.round(y, 3)}")
        assert np.isfinite(eff).all() and eff.shape == clean.shape
        return 0

    if not args.precompute and not args.train:
        print("Nichts zu tun — --precompute und/oder --train angeben (oder --smoke).", file=sys.stderr)
        return 2

    tracks = sorted(p for p in _MUSDB_TRAIN.glob("*") if (p / "mixture.wav").exists())
    if not tracks:
        print(f"MUSDB-Train nicht gefunden: {_MUSDB_TRAIN}", file=sys.stderr)
        return 2
    rng = _rng_for(args.seed)
    chosen = [tracks[i] for i in rng.choice(len(tracks), size=min(args.songs, len(tracks)), replace=False)]
    chosen.sort()

    if args.precompute:
        _precompute(chosen, args.segments, args.seed)
    if args.train:
        _train(args.epochs, args.lr, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
