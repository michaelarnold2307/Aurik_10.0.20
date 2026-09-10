#!/usr/bin/env python3
"""
Fine-tune SGMSE+ (Score-based Generative Model) on MUSDB18 music data (§v10.16).

Replaces speech-only training (VoiceBank-DEMAND, WSJ0, EARS-WHAM at 16kHz)
with music-specific fine-tuning at 48kHz.

§v10.16-Datenvertrag (diese Fassung, 2026-09-10):
  - Musik/Gesang: MUSDB18-HQ stems (vocals, drums, bass, other) — Streaming
  - Rauschen: 50 % synthetisch (weiß/rosa/braun), 50 % Corpus (data/musan,
    Auto-Detection; ohne Corpus → Warnung, synthetisch-only)
  - Hall: 30 % der Samples mit synthetischem Raumhall (anechoic/reverb-Paare;
    RIR: exponentiell abklingendes Rauschen + Direktpfad, deterministisch via Seed)
  - SNR 5–20 dB, 4 s Chunks @ 48 kHz, Joint-Peak-Normalisierung

Prerequisites:
  - Pre-trained checkpoint: models/sgmse_plus/sgmse_plus_src_1.ckpt
  - Ninja-free backbone patch applied (ncsnpp_utils/op/__init__.py)

Architecture:
  - Backbone: NCSNpp (nf=128, identisch zum src-Checkpoint — vollständig
    faltungsbasiert, läuft nativ auf 48-kHz-STFT mit F=256; ACHTUNG: der
    NCSNpp_48k-Backbone ist KEINE gleiche Architektur, sondern strukturell
    verschieden — src-Gewichte laden dort nicht; Befund 2026-09-10, Smoke-Run)
  - SDE: OUVE (Ornstein-Uhlenbeck Variance Exploding)
  - Input: [B, 2, F, T] complex STFT (n_fft=510, hop=128 → F=256,
    identisch zur Plugin-Geometrie sgmse_plugin.py _DEFAULT_N_FFT/_HOP), 48kHz
  - Output: score [B, 1, F, T] complex

Training: ~3-7 days on GPU (200 epochs, batch=4).

Usage:
    python scripts/train_sgmse_musik.py --epochs 200 --batch-size 4 --lr 3e-5
"""

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import fftconvolve
from torch.utils.data import DataLoader, Dataset

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / "models" / "sgmse_plus"))


# §v10.19-Fix: Offline-Lightning-Stub. Die SGMSE+-Release-Checkpoints sind
# PyTorch-Lightning-Dateien, deren Unpickling `pytorch_lightning` referenziert.
# Die venv-torchvision ist ROCm-inkompatibel (zirkulärer Import) — für die
# reine state_dict-Extraktion genügen Platzhalter-Klassen.
def _install_lightning_stub() -> None:
    import types

    if "pytorch_lightning" in sys.modules:
        return
    _pl = types.ModuleType("pytorch_lightning")
    for _n in ("LightningModule", "LightningDataModule", "Callback", "Trainer", "LightningLite"):
        setattr(_pl, _n, type(_n, (), {}))
    sys.modules["pytorch_lightning"] = _pl
    for _sub in (
        "pytorch_lightning.core",
        "pytorch_lightning.core.module",
        "pytorch_lightning.core.saving",
        "pytorch_lightning.callbacks",
        "pytorch_lightning.utilities",
        "pytorch_lightning.plugins",
    ):
        _m = types.ModuleType(_sub)
        _m.LightningModule = _pl.LightningModule
        _m.LightningDataModule = _pl.LightningDataModule
        _m.Callback = _pl.Callback
        sys.modules[_sub] = _m


_install_lightning_stub()


# ── SDE (Ornstein-Uhlenbeck Variance Exploding) ────────────────────────────


class OUVESDE:
    """Minimal OUV SDE for score matching. Matches sgmse.sdes.OUVESDE."""

    def __init__(self, sigma_min=0.05, sigma_max=0.5):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def perturb(self, x, t):
        """Add noise to x according to OUV process at time t."""
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        sigma = sigma.view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        return x + sigma * noise, noise

    def loss_fn(self, model, x_clean, x_noisy, t):
        """Score-matching loss: ||score + noise/sigma||^2

        §v10.19-Fix: SGMSE+-Backbone erwartet [B, 2, F, T] komplex —
        Kanal 0 = perturbierter Zustand x_t, Kanal 1 = Bedingung (verrauschte
        Beobachtung y). Vorher wurde x_noisy ignoriert → IndexError im Forward.
        """
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t.view(-1, 1, 1, 1)
        # Shape-Normalisierung auf [B, 1, F, T] (komplex)
        if x_clean.dim() == 5:
            x_clean = x_clean.squeeze(2)
            x_noisy = x_noisy.squeeze(2)
        if x_clean.dim() == 3:
            x_clean = x_clean.unsqueeze(1)
            x_noisy = x_noisy.unsqueeze(1)
        # §v10.19-Fix: Zeitachse auf Vielfaches von 64 padden — die
        # 6 Downsample-Stufen brauchen gerade Dims (T=96-Befund 2026-09-10:
        # 32er-Padding kann in den Skip-Connections kollidieren).
        _T = x_clean.shape[-1]
        _pad_t = (64 - _T % 64) % 64
        if _pad_t:
            x_clean = F.pad(x_clean, (0, _pad_t))
            x_noisy = F.pad(x_noisy, (0, _pad_t))
        x_t = x_clean + sigma * torch.randn_like(x_clean)
        # cat (nicht stack!) entlang Kanal-Dim → [B, 2, F, T] komplex
        cond = torch.cat([x_t, x_noisy], dim=1)
        score_pred = model(cond, t)
        target = -(x_t - x_clean) / (sigma**2 + 1e-8)
        return F.mse_loss(score_pred.real, target.real) + F.mse_loss(score_pred.imag, target.imag)


# ── Feature Extraction ──────────────────────────────────────────────────────


class STFTExtractor:
    """Compute complex STFT for SGMSE+ input."""

    def __init__(self, n_fft=510, hop=128, device="cpu"):
        self.n_fft = n_fft
        self.hop = hop
        self.window = torch.hann_window(n_fft, device=device)

    def to(self, device):
        self.window = self.window.to(device)
        return self

    def __call__(self, audio):
        """audio: [B, T] → spec: [B, 2, F, T] complex"""
        spec = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop, window=self.window, return_complex=True)
        return spec.unsqueeze(1)  # [B, 1, F, T] → stack as [B, 2, F, T]?


# ── Dataset (streaming from MUSDB18, no pre-generation needed) ──────────────


class SGMSE_Dataset(Dataset):
    """Streaming-Dataset: MUSDB18-Stems direkt laden, Rauschen + Hall on-the-fly.

    §v10.16: 30 % Reverb-Paare (anechoic/reverb), 50 % Corpus-Rauschen
    (data/musan, falls vorhanden), sonst synthetisch weiß/rosa/braun.
    """

    def __init__(self, audio_files: list[Path], musan_files: Optional[list[Path]] = None, reverb_prob: float = 0.30, chunk_samples: int = 192000):
        self.files = audio_files
        self.musan_files = musan_files or []
        self.reverb_prob = reverb_prob
        self.chunk_samples = chunk_samples
        self._musan_durations: dict[Path, float] = {}

    def __len__(self):
        return len(self.files) * 20  # 20 chunks per file per epoch

    def _load(self, path: Path) -> np.ndarray:
        y, orig_sr = librosa.load(str(path), sr=None, mono=True)
        if orig_sr != 48000:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=48000)
        y = y.astype(np.float32)
        if len(y) < self.chunk_samples:
            y = np.pad(y, (0, self.chunk_samples - len(y)), mode="reflect")
        start = random.randint(0, max(0, len(y) - self.chunk_samples))
        return y[start : start + self.chunk_samples]

    def _musan_noise(self) -> np.ndarray:
        """Zufälliges 4s-Segment aus dem MUSAN-Corpus (48 kHz mono)."""
        path = random.choice(self.musan_files)
        if path not in self._musan_durations:
            try:
                self._musan_durations[path] = librosa.get_duration(filename=path)
            except Exception:  # pylint: disable=broad-except
                self._musan_durations[path] = 0.0
        dur = self._musan_durations[path]
        start = random.uniform(0, max(0.0, dur - self.chunk_samples / 48000.0 - 1.0))
        y, sr = librosa.load(str(path), sr=48000, mono=True, offset=start, duration=self.chunk_samples / 48000.0)
        if sr != 48000:
            y = librosa.resample(y, orig_sr=sr, target_sr=48000)
        y = y.astype(np.float32)
        if len(y) < self.chunk_samples:
            y = np.pad(y, (0, self.chunk_samples - len(y)), mode="reflect")
        return y[: self.chunk_samples]

    @staticmethod
    def _rir(taps: int = 2400, tau: float = 0.02) -> np.ndarray:
        """Synthetische Raumimpulsantwort: Direktpfad + exponentiell abklingendes
        Rauschen (RT60 ~ 6.9·tau ≈ 140 ms), L2-normalisiert."""
        t = np.arange(taps, dtype=np.float32) / 48000.0
        ir = np.random.randn(taps).astype(np.float32) * np.exp(-t / tau)
        ir[0] = 1.0  # Direktpfad
        return (ir / (np.sqrt(np.sum(ir**2)) + 1e-8)).astype(np.float32)

    @staticmethod
    def _reverb(clean: np.ndarray) -> np.ndarray:
        """Anechoic → reverberant via synthetischer RIR (fftconvolve, gleiche Länge)."""
        wet = fftconvolve(clean, SGMSE_Dataset._rir())[: len(clean)]
        return wet.astype(np.float32)

    def _synthetic_noise(self, n: int) -> np.ndarray:
        noise = np.random.randn(n).astype(np.float32)
        c = random.choice(["white", "pink", "brown"])
        if c == "pink":
            noise = np.cumsum(noise)
        elif c == "brown":
            noise = np.cumsum(np.cumsum(noise))
        return noise / (np.abs(noise).max() + 1e-8)

    def __getitem__(self, idx):
        clean = self._load(self.files[idx % len(self.files)])
        peak = np.abs(clean).max() + 1e-8
        clean = clean / peak

        # §v10.16: 30 % der Samples reverberant (anechoic/reverb-Paare)
        if random.random() < self.reverb_prob:
            degraded = self._reverb(clean)
        else:
            degraded = clean.copy()

        # Rauschen: 50 % Corpus (falls vorhanden), sonst synthetisch
        if self.musan_files and random.random() < 0.5:
            noise = self._musan_noise()
        else:
            noise = self._synthetic_noise(self.chunk_samples)

        snr_db = random.uniform(5.0, 20.0)
        cr = np.sqrt(np.mean(degraded**2) + 1e-8)
        nr = np.sqrt(np.mean(noise**2) + 1e-8)
        noise = noise * (cr / (10 ** (snr_db / 20))) / (nr + 1e-8)
        degraded = degraded + noise

        dp = np.abs(degraded).max() + 1e-8
        return {"clean": torch.from_numpy(clean / dp), "noisy": torch.from_numpy(degraded / dp)}


# ── Training ────────────────────────────────────────────────────────────────


def train(
    epochs=200,
    batch_size=4,
    lr=3e-5,
    steps_per_epoch=200,
    ckpt_path="models/sgmse_plus/sgmse_plus_src_1.ckpt",
    resume=None,
    seed=42,
    out_dir=None,
    chunk_sec=4.0,
):
    # §G5-Determinismus: Seeds pro Session (Daten-Mixing reproduzierbar)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    chunk_samples = int(chunk_sec * 48000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # Data — streaming directly from MUSDB18, no pre-generation
    musdb = _PROJECT / "data" / "musdb18hq" / "train"
    all_files = sorted(
        f
        for f in musdb.rglob("*.wav")
        if f.is_file() and any(s in f.stem for s in ["vocals", "drums", "bass", "other"])
    )
    if not all_files:
        print(f"ERROR: No stem files found in {musdb}")
        return

    # §v10.16: Corpus-Rauschen (MUSAN) — Auto-Detection, Warnung statt Silent-Failure
    musan_files = sorted(p for p in (_PROJECT / "data" / "musan").rglob("*.wav") if p.is_file())
    if musan_files:
        print(f"MUSAN-Corpus: {len(musan_files)} Noise-Dateien gefunden")
    else:
        print("WARNUNG: data/musan leer — Training läuft synthetisch-only (weiß/rosa/braun). "
              "§v10.16 verlangt zusätzlich Corpus-Rauschen; MUSAN vor dem finalen Lauf bereitstellen.")

    random.shuffle(all_files)
    n_val = max(1, int(len(all_files) * 0.2))
    train_files, val_files = all_files[n_val:], all_files[:n_val]

    train_ds = SGMSE_Dataset(train_files, musan_files=musan_files, chunk_samples=chunk_samples)
    val_ds = SGMSE_Dataset(val_files, musan_files=musan_files, chunk_samples=chunk_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    # Model — Backbone NCSNpp (src-Architektur, 48-kHz-STFT). Nicht NCSNpp_48k:
    # dessen Layer-Struktur weicht vom src-Checkpoint ab (Size-Mismatches im
    # Smoke-Run 2026-09-10) — Fine-Tune-Init wäre unmöglich (§V7: Ursache statt
    # Workaround; der faltungsbasierte NCSNpp verarbeitet F=512/T=768 nativ).
    from sgmse.backbones.ncsnpp import NCSNpp

    model = NCSNpp().to(device)
    sde = OUVESDE()

    # Load pre-trained weights (partial — backbone only, not full Lightning ckpt)
    ckpt = Path(ckpt_path)
    if ckpt.exists():
        print(f"Loading pre-trained weights from {ckpt}")
        # §v10.19-Fix: SGMSE+-Releases sind PyTorch-Lightning-Checkpoints mit
        # Nicht-Tensor-Globals (SpecsDataModule) — weights_only=True scheitert.
        # Lokale, vertrauenswürdige Datei aus models/ → weights_only=False.
        state = torch.load(ckpt, map_location=device, weights_only=False)
        # Lightning checkpoint structure: state_dict has "model."/"dnn." prefix
        if "state_dict" in state:
            sd = {
                k.replace("model.", "").replace("dnn.", "").replace("_orig_mod.", ""): v
                for k, v in state["state_dict"].items()
            }
        elif "model_state_dict" in state:
            sd = state["model_state_dict"]
        else:
            sd = state
        # Filter to backbone parameters only (NCSNpp hält alles in all_modules)
        model_sd = {
            k: v
            for k, v in sd.items()
            if any(k.startswith(p) for p in ["all_modules", "enc", "dec", "output", "act", "norm"])
        }
        if model_sd:
            model.load_state_dict(model_sd, strict=False)
            print(f"  Loaded {len(model_sd)} backbone parameters")
        else:
            print("  WARNING: No backbone parameters found — starting from scratch")
    else:
        print(f"WARNING: No checkpoint at {ckpt} — training from scratch")

    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: NCSNpp ({n_p:.1f}M) | Data: {len(train_ds)} train / {len(val_ds)} val")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)

    out_dir = Path(out_dir) if out_dir else _PROJECT / "models" / "sgmse_plus" / "finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    start_epoch = 0

    if resume:
        rc = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(rc["model_state_dict"])
        start_epoch = rc.get("epoch", 0)
        best_val = rc.get("val_loss", float("inf"))

    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr} → 1e-6")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            clean = batch["clean"].to(device)
            noisy = batch["noisy"].to(device)

            # STFT (complex) — Plugin-Geometrie n_fft=510/hop=128 → F=256
            window = torch.hann_window(510, device=device)
            spec_c = torch.stft(clean, n_fft=510, hop_length=128, window=window, return_complex=True)
            spec_n = torch.stft(noisy, n_fft=510, hop_length=128, window=window, return_complex=True)
            # Stack real+imag as complex: [B, F, T] → [B, 1, F, T] complex → [B, F, T]
            spec_c = spec_c.unsqueeze(1).contiguous()
            spec_n = spec_n.unsqueeze(1).contiguous()

            optimizer.zero_grad()
            t = torch.rand(batch_size, device=device)
            loss = sde.loss_fn(model, spec_c, spec_n, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_loss += loss.item()

            if (step + 1) % 40 == 0:
                e = time.time() - t0
                eta = e / (step + 1) * (steps_per_epoch - step - 1) if step > 0 else 0
                print(
                    f"  Ep {epoch + 1:3d}/{epochs} | St {step + 1:3d}/{steps_per_epoch} | "
                    f"L {train_loss / (step + 1):.4f} | {e:.0f}s/{eta:.0f}s",
                    flush=True,
                )

        scheduler.step()
        avg_train = train_loss / min(steps_per_epoch, len(train_loader))

        # Validation
        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for vb in val_loader:
                if vn >= 20:
                    break
                cv, nv = vb["clean"].to(device), vb["noisy"].to(device)
                sc = torch.stft(cv, n_fft=510, hop_length=128, window=window, return_complex=True).unsqueeze(1)
                sn = torch.stft(nv, n_fft=510, hop_length=128, window=window, return_complex=True).unsqueeze(1)
                val_loss += sde.loss_fn(model, sc, sn, torch.rand(batch_size, device=device)).item()
                vn += 1
        avg_val = val_loss / max(vn, 1)

        print(
            f"Ep {epoch + 1:3d}/{epochs} | Tr {avg_train:.4f} | Val {avg_val:.4f} | "
            f"LR {scheduler.get_last_lr()[0]:.1e} | {time.time() - t0:.0f}s",
            flush=True,
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val,
            },
            out_dir / "checkpoint_latest.ckpt",
        )
        if avg_val < best_val:
            best_val = avg_val
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch + 1, "val_loss": avg_val},
                out_dir / "sgmse_musik_best.ckpt",
            )
            print(f"  >> Best: {best_val:.4f}")

    print(f"\nDone. Best val: {best_val:.4f} | {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune SGMSE+ on music")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--ckpt", type=str, default="models/sgmse_plus/sgmse_plus_src_1.ckpt")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42, help="Determinismus (§G5)")
    p.add_argument("--out-dir", type=str, default=None, help="Ausgabeverzeichnis (Default: models/sgmse_plus/finetuned)")
    p.add_argument("--chunk-sec", type=float, default=4.0, help="Chunk-Länge in Sekunden (Speicher-Knopf: 24-GB-GPU braucht ggf. 2 s oder Batch 1)")
    args = p.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.steps_per_epoch, args.ckpt, args.resume, args.seed, args.out_dir, args.chunk_sec)
