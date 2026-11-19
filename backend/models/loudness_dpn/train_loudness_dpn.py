"""Loudness-DPN — Dynamik-Pipeline F002 (gelernte Dynamik-Entscheidung).

Gelernte Dynamik-Repraesentation aus dem Log-Mel-Spektrogramm statt der
Schwellwert-/Zielwert-Logik (LUFSNormalizer, deep_loudness_net.py). Regression
auf objektiv messbare Zielfunktionen (integriertes LUFS + Dynamic Range) —
die Labels sind keine Annotation, sondern die gemessene Referenz (§G5 (copilot-instructions.md) deterministisch).

Nutzung:
    python backend/models/loudness_dpn/train_loudness_dpn.py --epochs 40

Integration danach als Challenger im LUFSNormalizer (Default AUS, §V6 (copilot-instructions.md)-Fallback).
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

SEED = 42
N_MELS = 64
FRAMES = 96


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _logmel(audio: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    m = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    logm = librosa.power_to_db(m, ref=np.max)
    if logm.shape[1] >= FRAMES:
        logm = logm[:, :FRAMES]
    else:
        pad = np.full((N_MELS, FRAMES - logm.shape[1]), logm.min(), dtype=np.float32)
        logm = np.concatenate([logm, pad], axis=1)
    return logm.astype(np.float32)


def _measure_lufs_dr(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """Objektive Zielfunktion: integriertes LUFS + Dynamic Range (Peak-RMS)."""
    import pyloudnorm as pyln

    a = audio if audio.ndim == 1 else audio.mean(axis=0)
    try:
        lufs = float(pyln.Meter(sr).integrated_loudness(a.astype(np.float64)))
    except Exception:  # §V6 (copilot-instructions.md)
        lufs = -70.0
    rms = float(np.sqrt(np.mean(a**2)) + 1e-12)
    peak = float(np.max(np.abs(a)) + 1e-12)
    dr = float(20.0 * np.log10(peak / rms))
    return (max(min(lufs, 0.0), -70.0), max(min(dr, 60.0), 0.0))


def _load_dataset(corpus: str, per_file: int):
    import soundfile as sf

    X, y = [], []
    rng = np.random.default_rng(SEED)
    files = sorted(glob.glob(os.path.join(corpus, "*", "damaged", "*")))
    for f in files:
        if not f.lower().endswith((".wav", ".mp3", ".flac")):
            continue
        try:
            a, sr = sf.read(f, dtype="float32", always_2d=True)
            a = a.mean(axis=1)
        except Exception as _e:  # §V6 (copilot-instructions.md)
            logger.debug("Quelle uebersprungen: %s", _e)
            continue
        for _ in range(per_file):
            start = int(rng.uniform(0, max(1, len(a) - sr * 20)))
            seg = a[start : start + sr * 20]
            if len(seg) < sr * 5:
                continue
            lufs, dr = _measure_lufs_dr(seg, sr)
            X.append(_logmel(seg, sr))
            y.append([lufs / 70.0 + 1.0, dr / 60.0])  # normalisiert [-1..1], [0..1]
    return np.array(X), np.array(y, dtype=np.float32)


class LoudnessDPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.gru = nn.GRU(64 * (N_MELS // 8), 96, batch_first=True, bidirectional=True)
        self.head = nn.Linear(192, 2)  # [LUFS_norm, DR_norm]

    def forward(self, x):
        b = x.shape[0]
        h = self.conv(x)
        h = h.permute(0, 3, 1, 2).contiguous().view(b, -1, 64 * (N_MELS // 8))
        o, _ = self.gru(h)
        return self.head(o[:, -1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="models/loudness_dpn.pt")
    ap.add_argument("--per-file", type=int, default=6)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    _seed_all(SEED)

    X, y = _load_dataset(args.corpus, args.per_file)
    logger.info("Daten: %d Samples (LUFS+DR Regression)", len(y))

    idx = np.arange(len(y))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    n_te = max(1, int(0.2 * len(y)))
    te, tr = idx[:n_te], idx[n_te:]
    Xtr = torch.tensor(X[tr], dtype=torch.float32).unsqueeze(1)
    ytr = torch.tensor(y[tr], dtype=torch.float32)
    Xte = torch.tensor(X[te], dtype=torch.float32).unsqueeze(1)
    yte = torch.tensor(y[te], dtype=torch.float32)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = LoudnessDPN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(SEED + ep))
        for i in range(0, len(Xtr), 16):
            b = perm[i : i + 16]
            opt.zero_grad()
            loss = lossf(model(Xtr[b].to(dev)), ytr[b].to(dev))
            loss.backward()
            opt.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                pred = model(Xte.to(dev)).cpu()
            mae_lufs = float((pred[:, 0] - yte[:, 0]).abs().mean() * 70.0)
            mae_dr = float((pred[:, 1] - yte[:, 1]).abs().mean() * 60.0)
            logger.info("Epoche %d: MAE LUFS %.2f dB | MAE DR %.2f dB", ep + 1, mae_lufs, mae_dr)

    model.eval()
    with torch.no_grad():
        pred = model(Xte.to(dev)).cpu()
    mae_lufs = float((pred[:, 0] - yte[:, 0]).abs().mean() * 70.0)
    mae_dr = float((pred[:, 1] - yte[:, 1]).abs().mean() * 60.0)
    logger.info("FINALE MAE LUFS %.2f dB | MAE DR %.2f dB (%d/%d)", mae_lufs, mae_dr, len(yte), len(y))

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "seed": SEED}, args.out)
    logger.info("Modell gespeichert: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
