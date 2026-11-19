"""Training des gelernten Klickdetektors (ClickNet, v4 mit M/Eingabe).

  python train_detector.py --data ./clean_music --noise-bank ./groove_noise --epochs 30
Ausgabe: runs/detector/detector.pt (bestes Modell nach F1)

v4: ClickNet sieht Mid/Side explizit (feat_ch=8); Seeds fuer Reproduzierbarkeit (§G5).
"""
import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import SR, ClipDataset, find_audio
from losses import detector_counts, detector_loss, f1_from_counts
from model import ClickNet
from simulate import load_noise_bank


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, dl, dev):
    model.eval()
    tp = fp = fn = 0
    for noisy, mask in dl:
        a, b, c = detector_counts(model(noisy.to(dev)), mask.to(dev))
        tp, fp, fn = tp + a, fp + b, fn + c
    return f1_from_counts(tp, fp, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--noise-bank", default=None)
    ap.add_argument("--pattern", default=None, help="Datei-Filter (z.B. mixture.wav)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--epoch-size", type=int, default=200)
    ap.add_argument("--val-size", type=int, default=64)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seg-seconds", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--feat-ch", type=int, default=8, help="Eingabekanäle (v4: 8 = +Mid/Side)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0, help="torch-Threads (0 = Auto)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="runs/detector")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    seed_everything(a.seed)
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    dev = a.device
    files = find_audio(a.data, a.pattern)
    assert files, "Keine Audiodateien gefunden"
    random.Random(0).shuffle(files)
    nval = max(1, int(0.05 * len(files)))
    val_files, train_files = files[:nval], (files[nval:] or files)
    bank = load_noise_bank(a.noise_bank) if a.noise_bank else []
    seg = int(a.seg_seconds * SR)
    cfg = dict(ch=a.ch, layers=a.layers, feat_ch=a.feat_ch)
    model = ClickNet(**cfg).to(dev)
    print(f"ClickNet: {sum(p.numel() for p in model.parameters()) / 1e3:.0f} k Parameter | seed {a.seed}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    train_dl = DataLoader(ClipDataset(train_files, a.epoch_size, seg, bank, mode="detect"), batch_size=a.bs,
                          num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    val_dl = DataLoader(ClipDataset(val_files, a.val_size, seg, bank, train=False, mode="detect"), batch_size=a.bs,
                        num_workers=a.workers)
    best = -1.0
    for ep in range(a.epochs):
        model.train()
        t0, acc, n = time.time(), 0.0, 0
        for noisy, mask in train_dl:
            noisy, mask = noisy.to(dev), mask.to(dev)
            loss = detector_loss(model(noisy), mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            acc += loss.item()
            n += 1
        sched.step()
        p, r, f1 = validate(model, val_dl, dev)
        print(f"ep {ep:03d} loss {acc / max(n, 1):.4f} | val P {p:.3f} R {r:.3f} F1 {f1:.3f} | {time.time() - t0:.0f}s")
        if f1 >= best:
            best = f1
            torch.save(dict(config=cfg, model=model.state_dict(), f1=f1), os.path.join(a.out_dir, "detector.pt"))
    print(f"Fertig. Bestes F1: {best:.3f}")


if __name__ == "__main__":
    main()
