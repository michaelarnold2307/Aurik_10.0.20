#!/usr/bin/env python3
"""scripts/train_flashsr_f4.py — F4: FlashSR-Musik-Finetune (Hochband-Rekonstruktion).

§SOTA-ML-V1/F4 (Roadmap TODOS_SOTA_ROADMAP.md): FlashSR (FastAudioSR,
16 kHz → 48 kHz Sprach-Bandbreitenerweiterung, 132k Parameter) ist die
gewählte Kandidaten-Basis (14× Echtzeit; VORAB-Benchmark 2026-09-13:
beide Kandidaten unter der Never-worsen-Linie auf Musik — der Finetune
ist die Voraussetzung). Der Finetune bildet 16-kHz-Crops auf die
48-kHz-Originale ab (Hochband-Rekonstruktion > 16 kHz — der größte Hebel
aus der Export-Analyse: bandwidth_loss/hf_remanence_loss conf≈0,99).

Rezept (EAR-VAE-Muster, deterministisch §G5 (GEBOTE.md)):
  - Loss: A1 + Multi-Resolution-STFT (n_fft ∈ {512, 1024, 2048} @ 48 kHz)
  - Optimizer: AdamW lr=1e-4, Gradient-Clipping 1.0
  - Validierung: 2 feste Test-Tracks, Early-Stop (Patience 5)
  - Daten: MUSDB18-HQ train/ (44,1 kHz) → 48-kHz-Crops (3 s) → 16-kHz-Input
  - Modell: models/flashsr/FastAudioSR/FASR (Checkpoint models/flashsr/models/upsampler.pth)

Gate (NACH dem Lauf): MUSDB-ΔSDR ≥ +2 dB (Roadmap-F4-Gate) via
Bandbreiten-Extensions-Messung (12-kHz-Basis → Rekonstruktion) — die
Never-worsen-Bedingung gilt je Segment.

Aufruf:
  python scripts/train_flashsr_f4.py --smoke          # 3 Schritte, 1 Track
  python scripts/train_flashsr_f4.py --epochs 30      # voller Lauf (GPU)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_flashsr_f4")

SR_HI = 48000
SR_LO = 16000
CROP_S = 3.0
CROP_N = int(CROP_S * SR_HI)
IN_N = int(CROP_S * SR_LO)
STFT_SIZES = [512, 1024, 2048]
FLASHSR_DIR = "models/flashsr"


def build_model(device: torch.device, checkpoint: Path) -> torch.nn.Module:
    sys.path.insert(0, FLASHSR_DIR)
    from FastAudioSR import FASR

    fasr = FASR(str(checkpoint))
    model = fasr.model
    model = model.to(device)
    return model


def stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = torch.nn.functional.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=torch.hann_window(win, device=x.device),
        return_complex=True,
    )
    return X.abs()


def mr_stft_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sc = torch.zeros((), device=x.device)
    mag = torch.zeros((), device=x.device)
    for n_fft in STFT_SIZES:
        hop = n_fft // 4
        s_x = stft_mag(x, n_fft, hop, n_fft)
        s_y = stft_mag(y, n_fft, hop, n_fft)
        sc = sc + (torch.norm(s_y - s_x, p="fro") / (torch.norm(s_x, p="fro") + 1e-8))
        mag = mag + torch.mean(torch.abs(torch.log(s_y + 1e-4) - torch.log(s_x + 1e-4)))
    return sc / len(STFT_SIZES) + mag / len(STFT_SIZES)


def load_track(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(track_48k, track_16k) — die 16-kHz-Basis wird EINMAL je Track resampelt
    (Kaiser-Sinc, Produktions-identisch) statt je Crop (~0,2 s/Schritt gespart)."""
    import librosa
    import scipy.io.wavfile as wav

    sr, data = wav.read(str(path))
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32) / max(float(np.abs(data).max()), 1e-9)
    if sr != SR_HI:
        data = librosa.resample(data, orig_sr=sr, target_sr=SR_HI)
    lo = librosa.resample(data.astype(np.float32), orig_sr=SR_HI, target_sr=SR_LO)
    return data.astype(np.float32), lo.astype(np.float32)


def crop_pair(track: tuple[np.ndarray, np.ndarray], rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """(input_16k [IN_N], target_48k [CROP_N]) — deterministisch via rng.

    Beide Bänder stammen aus dem EINMALIG resampelten Track-Paar — die naive
    3:1-Dezimation würde HF-Anteile aliasieren und dem Modell die falsche
    Aufgabe stellen; per-Crop-Resampling war der Laufzeit-Befund (~0,2 s).
    """
    hi_arr, lo_arr = track
    s = int(rng.integers(0, max(1, len(hi_arr) - CROP_N)))
    hi = hi_arr[s : s + CROP_N]
    s_lo = int(s * SR_LO / SR_HI)
    lo = lo_arr[s_lo : s_lo + IN_N]
    return torch.from_numpy(lo), torch.from_numpy(hi)


def val_loss(model: torch.nn.Module, device: torch.device, tracks: list[np.ndarray], rng: np.random.Generator) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for tr in tracks:
            for _ in range(4):
                lo, hi = crop_pair(tr, rng)
                y = model(lo.unsqueeze(0).unsqueeze(1).to(device)).squeeze(1).squeeze(1)
                hi_g = hi.to(device)
                n_min = min(hi_g.shape[-1], y.shape[-1])
                y = y[..., :n_min]
                hi_g = hi_g[..., :n_min]
                loss = torch.nn.functional.l1_loss(y, hi_g) + 0.5 * mr_stft_loss(y, hi_g)
                total += float(loss.item())
                count += 1
    model.train()
    return total / max(1, count)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3 Schritte auf 1 Track")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--checkpoint", type=str, default="models/flashsr/models/upsampler.pth")
    ap.add_argument("--data-dir", type=str, default="data/musdb18hq")
    ap.add_argument("--out-dir", type=str, default="output/f4_flashsr")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # Kernel-Autotuning (ROCm/MIOpen-Fallback-Fix)
    logger.info("F4 FlashSR-Finetune — Gerät: %s", device)

    data_root = Path(args.data_dir)
    train_root = data_root / "train"
    test_root = data_root / "test"
    train_files = sorted(train_root.glob("*/mixture.wav"))
    test_files = sorted(test_root.glob("*/mixture.wav"))
    if args.smoke:
        train_files = train_files[:1]
    logger.info("Trainings-Tracks: %d, Test-Tracks: %d", len(train_files), len(test_files))
    if not train_files:
        logger.error("Keine MUSDB-Tracks unter %s gefunden", data_root)
        return 2

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        logger.error("Checkpoint fehlt: %s", checkpoint)
        return 2
    model = build_model(device, checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Modell geladen (%d Parameter) aus %s", n_params, checkpoint)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=1e-2)
    rng = np.random.default_rng(args.seed)
    train_tracks = [load_track(p) for p in train_files]
    val_tracks = [load_track(p) for p in test_files[:2]]

    start_epoch = 0
    best_val = float("inf")
    patience_left = 5
    history: list[dict] = []

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        res = torch.load(args.resume, map_location="cpu", weights_only=True)  # nosec B614
        model.load_state_dict(res["model"])
        opt.load_state_dict(res.get("optimizer", opt.state_dict()))
        start_epoch = int(res.get("epoch", 0)) + 1
        best_val = float(res.get("val_a1", best_val))
        logger.info("Resume ab Epoche %d (best_val=%.4f)", start_epoch, best_val)

    steps_per_epoch = 3 if args.smoke else args.steps_per_epoch
    model.train()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        run_a1 = 0.0
        run_stft = 0.0
        for step in range(steps_per_epoch):
            tr = train_tracks[int(rng.integers(0, len(train_tracks)))]
            lo_b, hi_b = [], []
            for _ in range(args.batch_size):
                lo, hi = crop_pair(tr, rng)
                lo_b.append(lo)
                hi_b.append(hi)
            lo_t = torch.stack(lo_b).to(device)
            hi_t = torch.stack(hi_b).to(device)
            y = model(lo_t.unsqueeze(1)).squeeze(1)
            n_min = min(hi_t.shape[-1], y.shape[-1])
            y = y[..., :n_min]
            hi_t = hi_t[..., :n_min]
            a1 = torch.nn.functional.l1_loss(y, hi_t)
            st = mr_stft_loss(y, hi_t)
            loss = a1 + 0.5 * st
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_a1 += float(a1.item())
            run_stft += float(st.item())
            if device.type == "cuda" and (step + 1) % 50 == 0:
                torch.cuda.empty_cache()
        run_a1 /= max(1, steps_per_epoch)
        run_stft /= max(1, steps_per_epoch)
        v_a1 = val_loss(model, device, val_tracks, rng)
        dt = time.perf_counter() - t0
        logger.info("Epoche %d fertig (%.1f s): A1=%.4f STFT=%.4f Val=%.4f", epoch, dt, run_a1, run_stft, v_a1)
        history.append(
            {
                "epoch": epoch,
                "train_a1": round(run_a1, 6),
                "train_stft": round(run_stft, 6),
                "val_a1": round(v_a1, 6),
                "seconds": round(dt, 1),
            }
        )
        ckpt = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch, "val_a1": v_a1}
        torch.save(ckpt, out_dir / f"checkpoint_epoch{epoch}.pt")
        if v_a1 < best_val:
            best_val = v_a1
            patience_left = 5
            torch.save(ckpt, out_dir / "best.pt")
            logger.info("Neues bestes Modell (Val=%.4f) gespeichert", v_a1)
        else:
            patience_left -= 1
            if patience_left <= 0 and not args.smoke:
                logger.info("Early-Stop nach Epoche %d (Val stagniert)", epoch)
                break
        report = {
            "device": str(device),
            "seed": args.seed,
            "history": history,
            "best_val_a1": round(best_val, 6),
            "gate_hint": "MUSDB-ΔSDR ≥ +2 dB (Bandbreiten-Extension) NACH dem Lauf messen",
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    logger.info("F4-Finetune abgeschlossen — bestes Val-A1: %.4f (Checkpoint: %s)", best_val, out_dir / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
