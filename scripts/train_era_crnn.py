"""CRNN-Era-Classifier — D-K1-3 analog / Phase E.

Gelernte Dekaden-Repraesentation aus dem Log-Mel-Spektrogramm statt
handgebaute Bandbreiten-Heuristik (die vom mp3-Codec der Digitalisierung
konfundiert wird — shellac->1980, gemessen). Ein kleines CRNN lernt die
Aufnahme-Epoche (Rausch-/Bandbreiten-/Inhalts-Signatur) direkt aus dem
Spektrogramm.

Determinismus (§G5 (copilot-instructions.md)): feste Seeds.

Nutzung:
    python scripts/train_era_crnn.py --epochs 30 --out models/era_crnn.pt

Der Classifier wird danach in forensics/medium_detector.py als Challenger
zur era-Schaetzung integriert (Default AUS, §V6-Fallback auf den heuristischen
EraClassifier).
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)

SEED = 42
N_MELS = 64
FRAMES = 96
# Dekaden-Buckets (seltene zusammengefasst: 1840/1890 -> 1900)
DECADES = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]


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


def _decade_for(path: str) -> int | None:
    base = os.path.basename(path)
    m = re.search(r"(18|19|20)[0-9]{2}", base)
    year = int(m.group(0)) if m else None
    if year is None:
        try:
            man = yaml.safe_load(open(os.path.join(os.path.dirname(path), "manifest.yaml")))
            for e in man if isinstance(man, list) else []:
                if e.get("file") == base and e.get("era"):
                    year = int(e["era"])
        except Exception as _e:  # §V6 (copilot-instructions.md)
            logger.debug("Manifest-Ära nicht lesbar: %s", _e)
    if year is None:
        return None
    dec = (year // 10) * 10
    if dec < 1900:
        dec = 1900
    return dec if dec in DECADES else None


def _load_dataset(corpus: str):
    X, y = [], []
    import soundfile as sf

    for f in sorted(glob.glob(os.path.join(corpus, "*", "damaged", "*"))):
        if not f.lower().endswith((".wav", ".mp3", ".flac")):
            continue
        dec = _decade_for(f)
        if dec is None:
            continue
        try:
            a, sr = sf.read(f, dtype="float32", always_2d=True)
            mono = a[:, 0][: sr * 25]
            if len(mono) < sr:
                continue
            X.append(_logmel(mono, sr))
            y.append(DECADES.index(dec))
        except Exception as _e:  # §V6 (copilot-instructions.md)
            logger.debug("Datei uebersprungen: %s", _e)
    return np.array(X), np.array(y)


def _augment(X: np.ndarray, y: np.ndarray, factor: int = 3):
    rng = np.random.default_rng(SEED)
    Xa, ya = [X], [y]
    for _ in range(factor):
        z = X + rng.normal(0.0, 2.0, X.shape) + rng.uniform(-3.0, 3.0)
        Xa.append(z)
        ya.append(y)
    return np.concatenate(Xa), np.concatenate(ya)


class EraCRNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
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
    ap.add_argument("--out", default="models/era_crnn.pt")
    ap.add_argument("--aug", type=int, default=3)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    _seed_all(SEED)

    X, y = _load_dataset(args.corpus)
    logger.info("Daten geladen: %d Samples, %d Dekaden", len(y), len(set(y.tolist())))
    if len(y) < 30:
        logger.warning("Zu wenig Daten fuer ein CRNN.")
        return 1

    X, y = _augment(X, y, args.aug)
    logger.info("Mit Augmentation: %d Samples", len(y))

    idx = np.arange(len(y))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    n_te = max(1, int(0.2 * len(y)))
    te, tr = idx[:n_te], idx[n_te:]
    Xtr = torch.tensor(X[tr], dtype=torch.float32).unsqueeze(1)
    ytr = torch.tensor(y[tr], dtype=torch.long)
    Xte = torch.tensor(X[te], dtype=torch.float32).unsqueeze(1)
    yte = torch.tensor(y[te], dtype=torch.long)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = EraCRNN(len(DECADES)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()

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
                pred = model(Xte.to(dev)).argmax(1).cpu()
            logger.info("Epoche %d: Test-Acc %.3f", ep + 1, float((pred == yte).float().mean()))

    model.eval()
    with torch.no_grad():
        pred = model(Xte.to(dev)).argmax(1).cpu()
    acc = float((pred == yte).float().mean())
    logger.info("FINALE Test-Accuracy: %.3f (%d/%d)", acc, int((pred == yte).sum()), len(yte))

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "decades": DECADES, "seed": SEED}, args.out)
    logger.info("Modell gespeichert: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
