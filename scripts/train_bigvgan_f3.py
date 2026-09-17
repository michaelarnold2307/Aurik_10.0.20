#!/usr/bin/env python3
"""scripts/train_bigvgan_f3.py — F3: BigVGAN-v2 Musik-Finetune (HR-V1/03/23/50/07).

§SOTA-F3 (Roadmap TODOS_SOTA_ROADMAP.md): BigVGAN-v2 ist sprach-trainiert
(LibriTTS/LJSpeech). Der Musik-Finetune adaptiert den Generator an
MUSDB18-HQ als Rekonstruktions-Vocoder — Ziel: der additive HR-V1-
Synthesepfad (phase_07 + 23/50/03) rekonstruiert Musik-Harmonik korrekt
statt Sprach-Artefakte. A/B-Teilvalidierung liegt vor (af +0,0073,
HNR +4,42 dB); dieses Skript liefert den MUSDB-Finetune für das
Roadmap-Gate (ΔSDR ≥ +2 dB, gemessen NACH dem Lauf via
scripts/validate_hr_v1.py auf dem Finetune-Checkpoint).

Rezept (EAR-VAE-Muster, deterministisch §G5 (GEBOTE.md)):
  - Ziel: Mel → Wellenform-Rekonstruktion (self-supervised, kein Paar-Datensatz)
  - Loss: A1 (Waveform-L1) + Multi-Resolution-STFT (Spectral Convergence +
    Log-Magnitude-L1 bei n_fft ∈ {512, 1024, 2048})
  - Optimizer: AdamW lr=1e-4, Gradient-Clipping 1.0, fp16 (GradScaler)
  - Validierung: 2 feste Test-Tracks (deterministische Fenster), Early-Stop
    auf bestem Val-A1 (Patience 5)
  - Daten: MUSDB18-HQ train/ (44,1 kHz), 2-s-Zufalls-Crops, num_workers=0
    (Fork-Deadlock-Befund F2 2026-09-14), Seed 42

Aufruf:
  python scripts/train_bigvgan_f3.py --smoke           # 3 Schritte, 1 Track
  python scripts/train_bigvgan_f3.py --epochs 30       # voller Lauf (GPU)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_bigvgan_f3")

# Checkpoint-Konfiguration (bigvgan_v2_44khz_128band_512x, aus config.json)
CFG = {
    "resblock": "1",
    "upsample_rates": [8, 4, 2, 2, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 8, 4, 4, 4, 4],
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "activation": "snakebeta",
    "snake_logscale": True,
    "use_tanh_at_final": False,
    "use_bias_at_final": False,
    "num_mels": 128,
    "sampling_rate": 44100,
    "n_fft": 2048,
    "hop_size": 512,
    "win_size": 2048,
    "fmin": 0,
    "fmax": None,
}
SR = CFG["sampling_rate"]
CROP_S = 1.0  # 1536-Kanal-Modell: 2-s-Crops × Batch 4 sprengten 24 GB VRAM (OOM-Befund)
CROP_N = int(CROP_S * SR)
STFT_SIZES = [512, 1024, 2048]


def build_model(device: torch.device, checkpoint: Path) -> torch.nn.Module:
    from bigvgan import AttrDict, BigVGAN

    raw = torch.load(str(checkpoint), map_location="cpu", weights_only=True)  # nosec B614
    if isinstance(raw, dict) and "generator" in raw:
        gen_sd = raw["generator"]
    elif hasattr(raw, "state_dict"):
        gen_sd = raw.state_dict()
    else:
        gen_sd = raw
    # Kanal-Zahl aus dem Checkpoint ableiten (wie plugins/bigvgan_v2_plugin.py):
    # ups.0.0.weight_v → [out_ch, in_ch, ksize]; der Checkpoint hat 24, nicht 512.
    _up0 = gen_sd.get("ups.0.0.weight_v") if isinstance(gen_sd, dict) else None
    if _up0 is not None:
        CFG["upsample_initial_channel"] = int(_up0.shape[0])
    h = AttrDict(dict(CFG))
    model = BigVGAN(h, use_cuda_kernel=False)
    model.load_state_dict(gen_sd)
    model = model.to(device)
    return model


def extract_mel(wave: torch.Tensor) -> torch.Tensor:
    from bigvgan.meldataset import mel_spectrogram

    # Wave [B, T] auf CPU/GPU; mel_spectrogram erwartet [1, T]
    mels = []
    for i in range(wave.shape[0]):
        m = mel_spectrogram(
            wave[i : i + 1],
            n_fft=CFG["n_fft"],
            num_mels=CFG["num_mels"],
            sampling_rate=SR,
            hop_size=CFG["hop_size"],
            win_size=CFG["win_size"],
            fmin=CFG["fmin"],
            fmax=CFG["fmax"],
            center=False,
        )
        mels.append(m)
    return torch.cat(mels, dim=0)  # [B, n_mels, T]


def stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
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
    """Multi-Resolution-STFT-Loss (Spectral Convergence + Log-Mag-L1)."""
    sc = torch.zeros((), device=x.device)
    mag = torch.zeros((), device=x.device)
    for n_fft in STFT_SIZES:
        hop = n_fft // 4
        s_x = stft_mag(x, n_fft, hop, n_fft)
        s_y = stft_mag(y, n_fft, hop, n_fft)
        sc = sc + (torch.norm(s_y - s_x, p="fro") / (torch.norm(s_x, p="fro") + 1e-8))
        mag = mag + torch.mean(torch.abs(torch.log(s_y + 1e-4) - torch.log(s_x + 1e-4)))
    return sc / len(STFT_SIZES) + mag / len(STFT_SIZES)


def load_track(path: Path) -> np.ndarray:
    import scipy.io.wavfile as wav

    sr, data = wav.read(str(path))
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32) / max(float(np.abs(data).max()), 1e-9)
    if sr != SR:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    return data


def random_crops(track: np.ndarray, batch: int, rng: np.random.Generator) -> torch.Tensor:
    n = len(track)
    starts = rng.integers(0, max(1, n - CROP_N), size=batch)
    out = np.stack([track[s : s + CROP_N] for s in starts])
    return torch.from_numpy(out.astype(np.float32))


def val_loss(model: torch.nn.Module, device: torch.device, tracks: list[np.ndarray], rng: np.random.Generator) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for tr in tracks:
            for _ in range(4):
                x = random_crops(tr, 2, rng).to(device)
                mel = extract_mel(x)
                y = model(mel)
                y = y.squeeze(1)  # [B, 1, T] → [B, T]
                _n_min = min(x.shape[-1], y.shape[-1])
                x_a = x[..., :_n_min]
                y_a = y[..., :_n_min]
                loss = torch.nn.functional.l1_loss(y_a, x_a) + 0.5 * mr_stft_loss(y_a.float(), x_a.float())
                total += float(loss.item())
                count += 1
    model.train()
    return total / max(1, count)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3 Schritte auf 1 Track")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--checkpoint", type=str, default="models/bigvgan/bigvgan_v2.pth")
    ap.add_argument("--data-dir", type=str, default="data/musdb18hq")
    ap.add_argument("--out-dir", type=str, default="output/f3_bigvgan")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("F3 BigVGAN-Finetune — Gerät: %s", device)

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
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

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
        model.load_state_dict(res["generator"])
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
            x = random_crops(tr, args.batch_size, rng).to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                mel = extract_mel(x)
                y = model(mel)
                y = y.squeeze(1)  # [B, 1, T] → [B, T]
                _n_min = min(x.shape[-1], y.shape[-1])
                x_a = x[..., :_n_min]
                y_a = y[..., :_n_min]
                a1 = torch.nn.functional.l1_loss(y_a.float(), x_a.float())
                st = mr_stft_loss(y_a.float(), x_a.float())
                loss = a1 + 0.5 * st
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            run_a1 += float(a1.item())
            run_stft += float(st.item())
            if device.type == "cuda" and (step + 1) % 50 == 0:
                torch.cuda.empty_cache()
            if step % 200 == 0 and step > 0:
                logger.info(
                    "Epoche %d Schritt %d/%d: A1=%.4f STFT=%.4f",
                    epoch,
                    step,
                    steps_per_epoch,
                    run_a1 / (step + 1),
                    run_stft / (step + 1),
                )
        run_a1 /= max(1, steps_per_epoch)
        run_stft /= max(1, steps_per_epoch)
        v_a1 = val_loss(model, device, val_tracks, rng)
        dt = time.perf_counter() - t0
        logger.info(
            "Epoche %d fertig (%.1f s): A1=%.4f STFT=%.4f Val=%.4f",
            epoch,
            dt,
            run_a1,
            run_stft,
            v_a1,
        )
        history.append(
            {
                "epoch": epoch,
                "train_a1": round(run_a1, 6),
                "train_stft": round(run_stft, 6),
                "val_a1": round(v_a1, 6),
                "seconds": round(dt, 1),
            }
        )
        ckpt = {
            "generator": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "val_a1": v_a1,
            "cfg": CFG,
        }
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
            "gate_hint": "MUSDB-ΔSDR ≥ +2 dB via scripts/validate_hr_v1.py NACH dem Lauf messen",
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    logger.info("F3-Finetune abgeschlossen — bestes Val-A1: %.4f (Checkpoint: %s)", best_val, out_dir / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
