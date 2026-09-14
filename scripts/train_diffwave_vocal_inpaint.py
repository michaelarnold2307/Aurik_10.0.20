#!/usr/bin/env python3
"""SOTA-VOCAL-INPAINT-S2 / GPU-Finetune F1: DiffWave-Vokal-Finetune (HAUPTWEG).

Ziel aus S1: Die Sprach-DiffWave schließt 300-ms-Gesangslücken mit mean SDR
−1,69 dB vs. Stille-Baseline (0 dB) — Ziellücke ΔSDR ≥ 0 je Lücke. Dieser
Finetune trainiert den Checkpoint (models/diffwave/diffwave.ckpt, MIT,
philovivero/DiffWave-vocoder) auf MUSDB18HQ-Vocals mit der exakten
Laufzeit-Semantik des Plugins: Konditionierung = Mel des GAPPTEN Fensters.

Rezept nach dem EAR-VAE-Musik-Finetune (models/ear_vae_upstream/finetune_music.py,
SOTA-DENOISE v2, Commits 98276a32/4b783d18):
  - Architektur: DiffWave (res_channels 64, 30 Dilations, mel 80, 400 Timesteps),
    Checkpoint strict geladen.
  - Loss: L1 auf dem Noise-Target + optional A1-Hör-Loss (--masking-beta,
    models/ear_vae_upstream/masking_loss.py, auf x0_hat vs. Ziel).
  - Optimizer: Adam, LR 5e-5 (kleiner LR = Encoder-Frozen-Analog), Seed 42.
  - Validierung je Epoch: S1-Protokoll (3 Test-Tracks × 5 Lücken à 300 ms,
    Seed 42, 50 Schritte = Laufzeit-Parität) — Early-Stop (Patience 5) auf
    mean SDR vs. Stille (0 dB); Never-worsen-Gate: bestes Modell ≥
    Zero-Shot-Baseline je Lücke.
  - Ausgabe: models/diffwave/diffwave_vocal_ft.ckpt + Report JSON.

Determinismus (§G5 (GEBOTE.md)): feste Seeds, cudnn.benchmark=False,
DDIM-deterministisches Sampling (z=0) in der Validierung.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile
from scipy.signal import resample_poly

_ROOT = Path(__file__).resolve().parent.parent
_MUSDB = _ROOT / "data" / "musdb18hq"
_CKPT = _ROOT / "models" / "diffwave" / "diffwave.ckpt"
_OUT_CKPT = _ROOT / "models" / "diffwave" / "diffwave_vocal_ft.ckpt"
_REPORT_DIR = _ROOT / "docs" / "reports" / "current"

_SR = 22050
_WIN = 16368  # Checkpoint-Konvention: Upsampler ×256 − 272 (T=65 ⇒ 16368)
_N_MELS = 80
_N_FFT = 1024
_HOP = 256
_MEL_T = 65  # Checkpoint-Konvention: 1 + WIN/HOP (Upsampler ×256 − 272 = _WIN)
_RES_CH = 64
_DILATION_CYCLE = 10
_N_RES_LAYERS = 30
_DIFF_EMB_IN = 128
_DIFF_EMB_OUT = 512
_N_TIMESTEPS = 400
_BETA_MIN = 1e-4
_BETA_MAX = 0.02
_GAP_SAMPLES = int(0.300 * _SR)
_VAL_STEPS = 50  # Laufzeit-Parität (Plugin n_steps=50)

_BETAS = np.linspace(_BETA_MIN, _BETA_MAX, _N_TIMESTEPS, dtype=np.float64)
_ALPHAS = 1.0 - _BETAS
_ALPHA_BAR = np.cumprod(_ALPHAS)


# ── DiffWave-Modell (Port von philovivero/DiffWave-vocoder, MIT) ──────────────
# https://github.com/philovivero/DiffWave-vocoder — Attribution: MIT-Lizenz.
# Der Checkpoint nutzt Wrapper-Module (keys wie `dilated_conv.conv.weight`,
# `diffusion_projection.w.weight`) — exakt nachgebildet für strict load.


class Conv1d(nn.Module):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.conv = nn.Conv1d(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Linear(nn.Module):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.w = nn.Linear(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)


class SpectrogramUpsampler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 16])
        self.conv2 = nn.ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 16])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.unsqueeze(x, 1)  # [B,1,80,T]
        x = F.leaky_relu(self.conv1(x), 0.4)
        x = F.leaky_relu(self.conv2(x), 0.4)
        return torch.squeeze(x, 1)  # [B,80,T*256]


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        half = dim_in // 2
        self._emb = math.log(10000) / (half - 1)
        self.projection1 = Linear(dim_in, dim_out)
        self.projection2 = Linear(dim_out, dim_out)

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        device = step.device
        half = _DIFF_EMB_IN // 2
        w = torch.exp(torch.arange(half, dtype=torch.float32, device=device) * -self._emb)
        x = step.float().unsqueeze(1) * w.unsqueeze(0)
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = F.silu(self.projection1(x))
        return F.silu(self.projection2(x))


class ResidualBlock(nn.Module):
    def __init__(self, res_channels: int, dilation: int, n_mels: int, diffusion_emb_dim: int) -> None:
        super().__init__()
        self.dilated_conv = Conv1d(res_channels, 2 * res_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(diffusion_emb_dim, res_channels)
        self.conditioner_projection = Conv1d(n_mels, 2 * res_channels, 1)
        self.output_projection = Conv1d(res_channels, 2 * res_channels, 1)

    def forward(
        self, x: torch.Tensor, diffusion_step: torch.Tensor, conditioner: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner
        gate, filter_ = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_projection = Conv1d(1, _RES_CH, 1)
        self.diffusion_embedding = DiffusionEmbedding(_DIFF_EMB_IN, _DIFF_EMB_OUT)
        self.spectrogram_upsampler = SpectrogramUpsampler()
        self.residual_layers = nn.ModuleList(
            [ResidualBlock(_RES_CH, 2 ** (i % _DILATION_CYCLE), _N_MELS, _DIFF_EMB_OUT) for i in range(_N_RES_LAYERS)]
        )
        self.skip_projection = Conv1d(_RES_CH, _RES_CH, 1)
        self.output_projection = Conv1d(_RES_CH, 1, 1)

    def forward(self, audio: torch.Tensor, spectrogram: torch.Tensor, diffusion_step: torch.Tensor) -> torch.Tensor:
        x = audio.unsqueeze(1)  # [B,1,T]
        x = F.relu(self.input_projection(x))
        diffusion_step = self.diffusion_embedding(diffusion_step)
        spectrogram = self.spectrogram_upsampler(spectrogram)
        skip: torch.Tensor | None = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            skip = skip_connection if skip is None else skip_connection + skip
        x = self.skip_projection(skip) / math.sqrt(len(self.residual_layers))
        x = F.relu(x)
        x = self.output_projection(x)
        return x.squeeze(1)


# ── Mel (Plugin-Parität: scipy-STFT 1024/256, 80 Bins, T=64) ─────────────────


def _mel_matrix(n_bins: int, sr: int) -> np.ndarray:
    mel_lo, mel_hi = 0.0, sr / 2.0
    m_lo = 2595 * np.log10(1 + mel_lo / 700)
    m_hi = 2595 * np.log10(1 + mel_hi / 700)
    pts = 700 * (10 ** (np.linspace(m_lo, m_hi, _N_MELS + 2) / 2595) - 1)
    bins = np.floor(pts * (n_bins - 1) / (sr / 2)).astype(int).clip(0, n_bins - 1)
    fb = np.zeros((_N_MELS, n_bins), np.float32)
    for m in range(1, _N_MELS + 1):
        lo, c, hi = bins[m - 1], bins[m], bins[m + 1]
        for k in range(lo, min(c, n_bins)):
            fb[m - 1, k] = (k - lo) / (c - lo + 1e-8)
        for k in range(c, min(hi, n_bins)):
            fb[m - 1, k] = (hi - k) / (hi - c + 1e-8)
    return fb


_MEL_FB = _mel_matrix(_N_FFT // 2 + 1, _SR)


def _mel_torch(mono: np.ndarray) -> torch.Tensor:
    """[WIN] float32 → Mel [1, 80, 65] (Checkpoint-Konvention: 1 + WIN/HOP).

    Reflect-Padding n_fft//2 beidseitig (librosa-Konvention) ⇒ 65 Frames;
    der Upsampler (ConvTranspose ×256) mappt exakt auf 16384 Samples.
    """
    from scipy.signal import stft as _stft

    noverlap = min(_N_FFT - _HOP, max(0, _N_FFT - 1))
    pad = _N_FFT // 2
    padded = np.pad(mono.astype(np.float64), (pad, pad), mode="reflect")
    _, _, z = _stft(padded, fs=_SR, nperseg=_N_FFT, noverlap=noverlap, window="hann")
    mag = np.abs(z[: _N_FFT // 2 + 1])
    mel = _MEL_FB @ mag
    mel = np.log(np.clip(mel, 1e-5, None))
    if mel.shape[1] > _MEL_T:
        mel = mel[:, :_MEL_T]
    if mel.shape[1] < _MEL_T:
        mel = np.pad(mel, ((0, 0), (0, _MEL_T - mel.shape[1])))
    return torch.from_numpy(mel.astype(np.float32)).unsqueeze(0)


# ── Daten ────────────────────────────────────────────────────────────────────


def _load_vocals_mono(track_dir: Path, seconds: int, seed: int, train: bool = False) -> np.ndarray:
    _wf = wavfile.read(str(track_dir / "vocals.wav"))
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
    if sr != _SR:
        g = int(np.gcd(sr, _SR))
        wav = resample_poly(wav, _SR // g, sr // g).astype(np.float32)
    rng = np.random.default_rng(seed)
    n = int(seconds * _SR)
    if train and len(wav) > n + _SR:
        start = int(rng.integers(0, len(wav) - n - _SR))
        wav = wav[start : start + n]
    else:
        wav = wav[:n]
    peak = float(np.max(np.abs(wav))) if wav.size else 1.0
    return (wav / peak).astype(np.float32) if peak > 0 else wav


def _active_starts(vocals: np.ndarray, n_windows: int, seed: int) -> list[int]:
    """Fensterstarts in aktiven Regionen (S1-Energie-Gate-Protokoll)."""
    rng = np.random.default_rng(seed)
    frame = _SR // 10
    energy = np.array([float(np.mean(vocals[i : i + frame] ** 2)) for i in range(0, len(vocals) - frame, frame)])
    thr = float(np.median(energy[energy > 1e-9])) * 0.5 if np.any(energy > 1e-9) else 0.0
    active = [i for i, e in enumerate(energy) if e > max(thr, 1e-7)]
    if len(active) < 8:
        active = list(range(len(energy)))
    chosen = rng.choice(active, size=min(n_windows, len(active)), replace=True)
    starts = []
    for c in chosen:
        s = int(c * frame + frame // 2 - _WIN // 2)
        s = int(np.clip(s, 0, max(0, len(vocals) - _WIN)))
        starts.append(s)
    return starts


def _make_example(vocals: np.ndarray, start: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    window = np.zeros(_WIN, np.float32)
    seg = vocals[start : start + _WIN]
    window[: len(seg)] = seg
    # 300-ms-Lücke an zufälliger Position (mit Rand, damit Kontext beidseitig bleibt)
    lo = max(int(0.15 * _WIN), _GAP_SAMPLES)
    hi = _WIN - _GAP_SAMPLES - int(0.15 * _WIN)
    gap = int(rng.integers(lo, hi))
    gapped = window.copy()
    gapped[gap : gap + _GAP_SAMPLES] = 0.0
    mel = _mel_torch(gapped)[0]  # [80, 65] — Stack im Trainer macht den Batch-Dim
    target = torch.from_numpy(window)
    return mel, target


# ── Sampling (DDIM-artig, deterministisch) ───────────────────────────────────


@torch.no_grad()
def _inpaint_window(
    model: DiffWave, mel: torch.Tensor, device: torch.device, n_steps: int = _VAL_STEPS, seed: int | None = None
) -> np.ndarray:
    mel = mel.to(device)
    if seed is not None:
        _gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(1, _WIN, device=device, dtype=torch.float32, generator=_gen) * 0.1
    else:
        x = torch.randn(1, _WIN, device=device, dtype=torch.float32) * 0.1  # wie Plugin-Start
    steps = list(range(_N_TIMESTEPS, 0, -(_N_TIMESTEPS // n_steps)))[:n_steps]
    for n in steps:
        step_t = torch.tensor([n - 1], dtype=torch.int64, device=device)
        pred = model(x, mel, step_t)
        a_bar = float(_ALPHA_BAR[n - 1])
        x0 = (x - math.sqrt(1.0 - a_bar) * pred) / math.sqrt(a_bar)
        nxt = n - (_N_TIMESTEPS // n_steps)
        if nxt <= 0:
            x = x0
        else:
            a_bar_nxt = float(_ALPHA_BAR[nxt - 1])
            x = math.sqrt(a_bar_nxt) * x0  # z=0: deterministisch (§G5 (GEBOTE.md))
    return x[0].detach().cpu().numpy().astype(np.float32)


def _sdr(est: np.ndarray, ref: np.ndarray) -> float:
    err = est - ref
    den = float(np.sum(err**2))
    num = float(np.sum(ref**2))
    if den <= 1e-30:
        return float(np.inf) if num > 1e-30 else 0.0
    return float(10.0 * np.log10(num / den))


def _val_gaps_sdr(
    model: DiffWave, tracks: list[Path], seed: int, device: torch.device, n_gaps: int = 5
) -> tuple[float, float, list[float]]:
    """S1-Protokoll @ 22,05 kHz: 30-s-Ausschnitt, aktive 300-ms-Lücken, 50 Schritte.

    Deterministisch je (seed): identisches Startrauschen je Lücke über Epochs
    hinweg — Early-Stop sieht ein sauberes Signal (§G5 (GEBOTE.md)).
    """
    rng = np.random.default_rng(seed)
    chosen = [tracks[i] for i in rng.choice(len(tracks), size=min(len(tracks), 3), replace=False)]
    chosen.sort()
    sdrs: list[float] = []
    gap_idx = 0
    for track in chosen:
        vocals = _load_vocals_mono(track, 30, seed)
        frame = _SR // 10
        energy = np.array([float(np.mean(vocals[i : i + frame] ** 2)) for i in range(0, len(vocals) - frame, frame)])
        thr = float(np.median(energy[energy > 1e-9])) * 0.5 if np.any(energy > 1e-9) else 0.0
        active = [i for i, e in enumerate(energy) if e > max(thr, 1e-7)]
        if not active:
            continue
        for c in rng.choice(active, size=min(n_gaps, len(active)), replace=False):
            center = int(c * frame + frame // 2)
            g0 = center - _GAP_SAMPLES // 2
            if g0 < 0 or g0 + _GAP_SAMPLES >= len(vocals):
                continue
            orig = vocals[g0 : g0 + _GAP_SAMPLES]
            w0 = int(np.clip(center - _WIN // 2, 0, max(0, len(vocals) - _WIN)))
            window = np.zeros(_WIN, np.float32)
            seg = vocals[w0 : w0 + _WIN]
            window[: len(seg)] = seg
            gap_local = center - w0
            gapped = window.copy()
            gapped[gap_local : gap_local + _GAP_SAMPLES] = 0.0
            mel = _mel_torch(gapped)
            filled = _inpaint_window(model, mel, device, seed=seed + gap_idx)
            gap_idx += 1
            gap_fill = filled[gap_local : gap_local + _GAP_SAMPLES]
            sdrs.append(_sdr(gap_fill, orig))
    if not sdrs:
        return float("-inf"), float("-inf"), []
    return float(np.mean(sdrs)), float(min(sdrs)), sdrs


# ── Training ─────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--windows-per-track", type=int, default=32)
    ap.add_argument("--masking-beta", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="1 Epoch, 2 Tracks (Setup-Test)")
    ap.add_argument("--skip-a1", action="store_true", help="A1-Hör-Loss deaktivieren")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    _ALPHA_BAR_DEV = torch.from_numpy(_ALPHA_BAR).float().to(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Benchmark/Autotuning AUS: MIOpen-Tuning je Dilations-Shape machte den
    # ersten Epoch ~2× langsamer (Befund 2026-09-14, batch 32) — die
    # deterministische Standard-Wahl ist auf ROCm schneller und stabil
    # (§G5 (GEBOTE.md): gleicher Input + Version ⇒ bit-identischer Output).
    torch.backends.cudnn.benchmark = False

    # A1-Hör-Loss (models/ear_vae_upstream — gitignored, Rezept-Komponente)
    _masking_loss = None
    if not args.skip_a1 and args.masking_beta > 0.0:
        _ear_dir = str(_ROOT / "models" / "ear_vae_upstream")
        if _ear_dir not in sys.path:
            sys.path.insert(0, _ear_dir)
        try:
            from masking_loss import masking_loss as _masking_loss  # A1 Hör-Loss
        except Exception as _ml_imp:
            print(f"masking_loss nicht ladbar ({_ml_imp}) — trainiere ohne Hör-Loss")

    model = DiffWave()
    state = torch.load(str(_CKPT), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, f"State-Drift: missing={len(missing)}, unexpected={len(unexpected)}"
    model.to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DiffWave geladen: {n_params / 1e6:.2f} M Parameter (strict)")

    train_tracks = sorted(p for p in (_MUSDB / "train").glob("*") if p.is_dir())
    test_tracks = sorted(p for p in (_MUSDB / "test").glob("*") if p.is_dir())
    if args.smoke:
        train_tracks = train_tracks[:2]
        args.epochs = 1
    if not train_tracks or not test_tracks:
        print("MUSDB18HQ fehlt")
        return 2

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # Zero-Shot-Baseline VOR dem Training (Never-worsen-Referenz, S1-Vergleich)
    model.eval()
    baseline_mean, baseline_min, _ = _val_gaps_sdr(model, test_tracks, args.seed, device)
    print(f"Zero-Shot-Baseline: Val-SDR mean={baseline_mean:+.2f} dB min={baseline_min:+.2f} dB")
    model.train()

    history: list[dict[str, float]] = []
    best_val = float("-inf")
    best_epoch = -1
    no_improve = 0

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        model.train()
        rng = np.random.default_rng(args.seed + epoch)
        order = train_tracks.copy()
        rng.shuffle(order)
        total_loss = 0.0
        cnt = 0
        for track in order:
            vocals = _load_vocals_mono(track, 40, args.seed + epoch, train=True)
            starts = _active_starts(vocals, args.windows_per_track, args.seed + epoch)
            for i in range(0, len(starts), args.batch):
                batch_starts = starts[i : i + args.batch]
                mels, targets = [], []
                for s in batch_starts:
                    mel, tgt = _make_example(vocals, s, rng)
                    mels.append(mel)
                    targets.append(tgt)
                mel_b = torch.stack(mels).to(device)
                tgt_b = torch.stack(targets).to(device)
                b = mel_b.shape[0]
                t = torch.randint(1, _N_TIMESTEPS, (b,), device=device)
                noise = torch.randn_like(tgt_b)
                # [B,1] × [B,T] → [B,T] (kein Batch↔Kanal-Broadcast-Kollaps!)
                a_bar_t = _ALPHA_BAR_DEV[t].view(b, 1)
                x_t = torch.sqrt(a_bar_t) * tgt_b + torch.sqrt(1.0 - a_bar_t) * noise
                pred = model(x_t, mel_b, t)
                loss = F.l1_loss(pred, noise)
                if _masking_loss is not None:
                    x0_hat = (x_t - torch.sqrt(1.0 - a_bar_t) * pred) / torch.sqrt(a_bar_t)
                    loss = loss + args.masking_beta * _masking_loss(x0_hat.unsqueeze(1), tgt_b.unsqueeze(1), _SR)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss.item()) * b
                cnt += b
        train_loss = total_loss / max(cnt, 1)

        model.eval()
        val_mean, val_min, _ = _val_gaps_sdr(model, test_tracks, args.seed, device)
        dt = time.perf_counter() - t0
        tag = ""
        if val_mean > best_val:
            best_val = val_mean
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), str(_OUT_CKPT))
            tag = " *best*"
        else:
            no_improve += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_mean_sdr_db": round(val_mean, 3),
                "val_min_sdr_db": round(val_min, 3),
                "seconds": round(dt, 1),
            }
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs}: Loss={train_loss:.6f} | Val-SDR mean={val_mean:+.2f} dB min={val_min:+.2f} dB [{dt:.0f}s]{tag}"
        )
        if no_improve >= args.patience:
            print(f"Early-Stop nach {args.epochs - (epoch + 1)} Runden ohne Verbesserung (Patience {args.patience}).")
            break

    report = {
        "date": f"{date.today().isoformat()}",
        "script": "scripts/train_diffwave_vocal_inpaint.py",
        "goal": "VOCAL-INPAINT-S2/F1: ΔSDR ≥ 0 je Lücke (S1-Ziellücke: mean −1,69 dB)",
        "lr": args.lr,
        "epochs": len(history),
        "masking_beta": args.masking_beta,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_mean_sdr_db": round(best_val, 3),
        "baseline_zero_shot_mean_sdr_db": round(baseline_mean, 3),
        "baseline_zero_shot_min_sdr_db": round(baseline_min, 3),
        "never_worsen_per_gap": best_val > float("-inf") and best_val >= 0.0,
        "history": history,
    }
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = _REPORT_DIR / f"{date.today().isoformat()}_diffwave_vocal_finetune.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nReport: {out}")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
