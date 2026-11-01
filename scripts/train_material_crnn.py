"""CRNN-Material-Classifier — D-K1-3 Phase D (Typus-Sprung).

Gelernte Material-Repräsentation aus dem Log-Mel-Spektrogramm statt
handgebaute 9 Skalar-Features (die eine ~60%-architektonische Decke haben).
Ein kleines CRNN (Conv2d + GRU + FC) lernt die Material-Signaturen
(Oberflaechenrausch-Textur, Rumble, Hiss, Codec-Artefakte) direkt aus dem
Spektrogramm.

Determinismus (§G5 (copilot-instructions.md)): feste Seeds, gleicher Input ⇒ gleiches Modell/Ergebnis.
Bestimmungsgemaess §G5 (copilot-instructions.md): kein `time.time()` in der
Entscheidungslogik.

Nutzung (Trainings-Pipeline):
    python scripts/train_material_crnn.py --epochs 30 --out models/material_crnn.pt

Der Classifier wird danach in `forensics/medium_detector.py` hinter ein Flag
als Challenger integriert (Default AUS, §V6-Fallback auf den Gauss/Mixture-Pfad).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)

SEED = 42
N_MELS = 64
FRAMES = 96
MATERIALS = [
    "shellac",
    "vinyl",
    "tape",
    "cassette",
    "reel_tape",
    "digital",
    "wax_cylinder",
    "wire_recording",
    "lacquer_disc",
    "minidisc",
]


# ── Determinismus (§G5 (copilot-instructions.md)) ─────────────────────────────────────────────────
def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Daten: Log-Mel + Labels aus dem Korpus ──────────────────────────────
def _logmel(audio: np.ndarray, sr: int) -> np.ndarray:
    """Log-Mel-Spektrogramm [N_MELS, FRAMES] (gecroppt/gepaddet)."""
    import librosa

    m = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    logm = librosa.power_to_db(m, ref=np.max)
    # auf FRAMES croppen/padden
    if logm.shape[1] >= FRAMES:
        logm = logm[:, :FRAMES]
    else:
        pad = np.full((N_MELS, FRAMES - logm.shape[1]), logm.min(), dtype=np.float32)
        logm = np.concatenate([logm, pad], axis=1)
    return logm.astype(np.float32)


def _label_for(path: str) -> str | None:
    base = os.path.basename(path)
    mf = os.path.join(os.path.dirname(path), "manifest.yaml")
    try:
        man = yaml.safe_load(open(mf))
        for e in man if isinstance(man, list) else []:
            if e.get("file") == base:
                return e.get("material")
    except Exception as _e:  # §V6 (copilot-instructions.md)
        logger.debug("_label_for Manifest fehlgeschlagen: %s", _e)
    d = os.path.basename(os.path.dirname(os.path.dirname(path)))
    return d if d in MATERIALS else None


def _load_dataset(corpus: str):
    X, y = [], []
    import soundfile as sf

    for f in sorted(glob.glob(os.path.join(corpus, "*", "damaged", "*"))):
        if not f.lower().endswith((".wav", ".mp3", ".flac")):
            continue
        mat = _label_for(f)
        if mat not in MATERIALS:
            continue
        try:
            a, sr = sf.read(f, dtype="float32", always_2d=True)
            mono = a[:, 0][: sr * 25]
            if len(mono) < sr:
                continue
            X.append(_logmel(mono, sr))
            y.append(MATERIALS.index(mat))
        except Exception as _e:  # §V6 (copilot-instructions.md)
            logger.debug("_load_dataset Datei uebersprungen: %s", _e)
            continue
    return np.array(X), np.array(y)


def _augment(X: np.ndarray, y: np.ndarray, factor: int = 3):
    """Deterministische Augmentation (Gain + Noise + Time-Shift im Mel-Raum)."""
    rng = np.random.default_rng(SEED)
    Xa, ya = [X], [y]
    for _ in range(factor):
        z = X.copy()
        z = z + rng.normal(0.0, 2.0, z.shape)  # Mel-Rauschen
        z = z + rng.uniform(-3.0, 3.0)  # Gain (dB im Mel-Raum)
        Xa.append(z)
        ya.append(y)
    return np.concatenate(Xa), np.concatenate(ya)


# ── Modell: kleines CRNN ────────────────────────────────────────────────
class MaterialCRNN(nn.Module):
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
        h = self.conv(x)  # [B, 64, N_MELS/8, FRAMES/8] (x ist schon [B,1,M,F])
        h = h.permute(0, 3, 1, 2).contiguous().view(b, -1, 64 * (N_MELS // 8))
        o, _ = self.gru(h)
        return self.head(o[:, -1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="models/material_crnn.pt")
    ap.add_argument("--aug", type=int, default=3)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    _seed_all(SEED)

    X, y = _load_dataset(args.corpus)
    logger.info("Daten geladen: %d Samples, %d Klassen", len(y), len(set(y.tolist())))
    if len(y) < 30:
        logger.warning("Zu wenig Daten fuer ein CRNN — braucht >=30 Samples.")
        return 1

    X, y = _augment(X, y, args.aug)
    logger.info("Mit Augmentation: %d Samples", len(y))

    # stratifizierter Split (deterministisch)
    idx = np.arange(len(y))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    n_te = max(1, int(0.2 * len(y)))
    te, tr = idx[:n_te], idx[n_te:]
    Xtr, ytr = torch.tensor(X[tr], dtype=torch.float32).unsqueeze(1), torch.tensor(y[tr], dtype=torch.long)
    Xte, yte = torch.tensor(X[te], dtype=torch.float32).unsqueeze(1), torch.tensor(y[te], dtype=torch.long)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MaterialCRNN(len(MATERIALS)).to(dev)
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
            acc = float((pred == yte).float().mean())
            logger.info("Epoche %d: Test-Acc %.3f", ep + 1, acc)

    # finale Genauigkeit
    model.eval()
    with torch.no_grad():
        pred = model(Xte.to(dev)).argmax(1).cpu()
    acc = float((pred == yte).float().mean())
    logger.info("FINALE Test-Accuracy: %.3f (%d/%d)", acc, int((pred == yte).sum()), len(yte))
    per = {}
    for c in range(len(MATERIALS)):
        m = yte == c
        if m.sum():
            per[MATERIALS[c]] = float((pred[m] == c).float().mean())
    logger.info("Per-Medium: %s", {k: round(v, 3) for k, v in per.items()})

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "materials": MATERIALS, "seed": SEED}, args.out)
    logger.info("Modell gespeichert: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
