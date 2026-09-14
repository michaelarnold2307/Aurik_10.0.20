"""DiffWave-Modell (Port von philovivero/DiffWave-vocoder, MIT) — geteiltes Modul.

https://github.com/philovivero/DiffWave-vocoder — Attribution: MIT-Lizenz.

Genutzt von:
  - scripts/train_diffwave_vocal_inpaint.py (§SOTA-VOCAL-INPAINT-S2/F1, Training)
  - backend/core/dsp/diffwave_torch_inpaint.py (§SOTA-VOCAL-INPAINT-S3, Runtime)

Checkpoint-Konventionen (models/diffwave/diffwave.ckpt, strict verifiziert):
  - res_channels 64, 30 Residual-Blocks (Dilations 1..512 ×3), mel 80, 400 Timesteps
  - WIN = 16368 Samples, Mel T = 65 Frames (1 + WIN/HOP) via Reflect-Padding —
    der ConvTranspose-Upsampler (×256) mappt exakt auf 16368
    (256×T − 272 = WIN).
  - Wrapper-Module (`*.conv.weight`, `*.w.weight`) exakt nachgebildet.

Determinismus (§G5 (GEBOTE.md)): reine Module ohne Zufallsquellen.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_RES_CH = 64
_N_MELS = 80
_N_FFT = 1024
_HOP = 256
_MEL_T = 65  # 1 + WIN/HOP
_WIN = 16368  # Upsampler ×256 − 272
_DILATION_CYCLE = 10
_N_RES_LAYERS = 30
_DIFF_EMB_IN = 128
_DIFF_EMB_OUT = 512
_N_TIMESTEPS = 400
_BETA_MIN = 1e-4
_BETA_MAX = 0.02

_BETAS = np.linspace(_BETA_MIN, _BETA_MAX, _N_TIMESTEPS, dtype=np.float64)
_ALPHAS = 1.0 - _BETAS
_ALPHA_BAR = np.cumprod(_ALPHAS)


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
    """DiffWave-Vocoder — Mel-konditionierte Diffusions-Wellenform-Modellierung."""

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


def mel_matrix(n_bins: int, sr: int) -> np.ndarray:
    """Slaney-Mel-Filterbank [80, n_bins] (Plugin-Parität)."""
    mel_lo, mel_hi = 0.0, sr / 2.0
    m_lo = 2595 * np.log10(1 + mel_lo / 700)
    m_hi = 2595 * np.log10(1 + mel_hi / 700)
    pts = 700 * (10 ** (np.linspace(m_lo, m_hi, _N_MELS + 2) / 2595) - 1)
    bins = np.floor(pts * (n_bins - 1) / (sr / 2)).astype(int).clip(0, n_bins - 1)
    fb: np.ndarray = np.zeros((_N_MELS, n_bins), np.float32)
    for m in range(1, _N_MELS + 1):
        lo, c, hi = bins[m - 1], bins[m], bins[m + 1]
        for k in range(lo, min(c, n_bins)):
            fb[m - 1, k] = (k - lo) / (c - lo + 1e-8)
        for k in range(c, min(hi, n_bins)):
            fb[m - 1, k] = (hi - k) / (hi - c + 1e-8)
    return fb


_SR = 22050
_MEL_FB = mel_matrix(_N_FFT // 2 + 1, _SR)


def mel_torch(mono: np.ndarray, sr: int = _SR) -> torch.Tensor:
    """[WIN] float32 → Mel [1, 80, 65] (Reflect-Padding, Checkpoint-Konvention)."""
    from scipy.signal import stft as _stft

    noverlap = min(_N_FFT - _HOP, max(0, _N_FFT - 1))
    pad = _N_FFT // 2
    padded = np.pad(mono.astype(np.float64), (pad, pad), mode="reflect")
    _, _, z = _stft(padded, fs=sr, nperseg=_N_FFT, noverlap=noverlap, window="hann")
    mag = np.abs(z[: _N_FFT // 2 + 1])
    mel = _MEL_FB @ mag
    mel = np.log(np.clip(mel, 1e-5, None))
    if mel.shape[1] > _MEL_T:
        mel = mel[:, :_MEL_T]
    if mel.shape[1] < _MEL_T:
        mel = np.pad(mel, ((0, 0), (0, _MEL_T - mel.shape[1])))
    return torch.from_numpy(mel.astype(np.float32)).unsqueeze(0)
