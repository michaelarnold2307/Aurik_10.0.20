"""CRNN-Defekt-Classifier — D-K1-3 analog / Phase F (multi-label).

Gelernte Defekt-Repraesentation aus dem Log-Mel-Spektrogramm statt 64
handgebaute _detect_*-Detector (DefectScanner). Multi-label: jede Aufnahme
kann mehrere Defekte haben (clicks, crackle, hum, wow, pops, hiss).

Labels: synthetische Defekt-Generierung (clean Audio + simulierte Defekte
aus models/de_crackle/simulate.py) — die Labels sind bekannt, weil die
Defekte kontrolliert injiziert werden. Determinismus (§G5 (copilot-instructions.md)).

Nutzung:
    python backend/models/de_crackle/train_defect_crnn.py --epochs 30

Integration danach als Challenger im DefectScanner (Default AUS, §V6-Fallback).
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
DEFECTS = ["clicks", "crackle", "hum", "wow", "pops", "hiss", "none"]


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
    return np.array(logm, dtype=np.float32, copy=True)


def _inject_defect(audio: np.ndarray, sr: int, rng, kind: str) -> np.ndarray:
    """Deterministische Defekt-Injektion (Label = kind)."""
    y = audio.copy()
    n = len(y)
    if kind == "clicks":
        for _ in range(int(rng.uniform(5, 25))):
            p = int(rng.uniform(0.02, 0.98) * n)
            y[p : p + 32] += rng.standard_normal(32).astype(np.float32) * 0.4
    elif kind == "pops":
        for _ in range(int(rng.uniform(2, 8))):
            p = int(rng.uniform(0.02, 0.98) * n)
            y[p : p + 256] += rng.standard_normal(256).astype(np.float32) * 0.25
    elif kind == "crackle":
        y += (rng.standard_normal(n).astype(np.float32) * 0.03 * (rng.random(n) < 0.01))
    elif kind == "hum":
        t = np.arange(n) / sr
        y += 0.05 * np.sin(2 * np.pi * 50 * t).astype(np.float32)
    elif kind == "hiss":
        y += rng.standard_normal(n).astype(np.float32) * 0.02
    elif kind == "wow":
        t = np.arange(n) / sr
        y *= (1.0 + 0.02 * np.sin(2 * np.pi * 1.0 * t)).astype(np.float32)
    return y


def _load_dataset(corpus: str, per_class: int):
    import soundfile as sf

    clean = []
    for f in sorted(glob.glob(os.path.join(corpus, "*", "damaged", "*"))):
        if not f.lower().endswith((".wav", ".mp3", ".flac")):
            continue
        try:
            a, sr = sf.read(f, dtype="float32", always_2d=True)
            clean.append((a[:, 0][: sr * 20], sr))
        except Exception as _e:  # §V6 (copilot-instructions.md)
            logger.debug("Quelle uebersprungen: %s", _e)
    X, y = [], []
    rng = np.random.default_rng(SEED)
    for kind in DEFECTS:
        for _ in range(per_class):
            if not clean:
                break
            a, sr = clean[int(rng.integers(0, len(clean)))]
            audio = _inject_defect(a, sr, rng, kind) if kind != "none" else a
            X.append(_logmel(audio, sr))
            y.append([1.0 if k == kind else 0.0 for k in DEFECTS])
    return np.array(X), np.array(y, dtype=np.float32)


class DefectCRNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.gru = nn.GRU(64 * (N_MELS // 8), 96, batch_first=True, bidirectional=True)
        self.head = nn.Linear(192, n_classes)

    def forward(self, x):
        b = x.shape[0]
        h = self.conv(x)
        h = h.permute(0, 3, 1, 2).contiguous().view(b, -1, 64 * (N_MELS // 8))
        o, _ = self.gru(h)
        return self.head(o[:, -1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="models/defect_crnn.pt")
    ap.add_argument("--per-class", type=int, default=40)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    _seed_all(SEED)

    X, y = _load_dataset(args.corpus, args.per_class)
    logger.info("Daten: %d Samples, %d Defekt-Klassen", len(y), len(DEFECTS))

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
    model = DefectCRNN(len(DEFECTS)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(SEED + ep))
        for i in range(0, len(Xtr), 16):
            b = perm[i : i + 16]
            opt.zero_grad()
            out = model(Xtr[b].to(dev))
            loss = lossf(out, ytr[b].to(dev))
            loss.backward()
            opt.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                pred = (torch.sigmoid(model(Xte.to(dev))) > 0.5).float().cpu()
            acc = float((pred == yte).float().all(1).float().mean())
            logger.info("Epoche %d: exakte Multi-Label-Acc %.3f", ep + 1, acc)

    model.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(model(Xte.to(dev))) > 0.5).float().cpu()
    acc = float((pred == yte).float().all(1).float().mean())
    ham = float((pred == yte).float().mean())
    logger.info("FINALE exakte Acc %.3f | Hamming-Acc %.3f (%d/%d)", acc, ham, int((pred == yte).all(1).sum()), len(yte))

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "defects": DEFECTS, "seed": SEED}, args.out)
    logger.info("Modell gespeichert: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
