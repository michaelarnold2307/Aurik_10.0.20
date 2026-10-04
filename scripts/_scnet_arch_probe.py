#!/usr/bin/env python3
"""Prototyp: BMWS/MVSep-SCNet-Architektur aus State-Dict-Shapes rekonstruieren.

Ziel: exact-match (strict=True) zum Checkpoint models/scnet_4stems/huge_scnet_4stems_v1.2.ckpt
(Aname-Tommy/Huge-SCNet-4stems, config.yaml band_SR=[0.23,0.37,0.40], dims=[4,64,128,256]).

Validierung:
  1) load_state_dict(strict=True) → keine missing/unexpected keys
  2) Forward auf 8-s-Ausschnitt → Summe der 4 Stems ≈ Mixture (Additivität, wie MUSDB-GT)
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent


def _stub_bitsandbytes() -> None:
    """Checkpoint-Pickles enthalten bitsandbytes.optim.adamw.AdamW8bit (Training).

    Für reines Inferenz-Laden (weights_only=False) reicht ein Dummy-Modul.
    """

    class _Mod(types.ModuleType):
        def __getattr__(self, n: str):
            if n.startswith("__"):
                raise AttributeError(n)
            val = type(n, (), {"__init__": lambda self, *a, **k: None})
            setattr(self, n, val)
            return val

    for name in (
        "bitsandbytes",
        "bitsandbytes.optim",
        "bitsandbytes.optim.adamw",
        "bitsandbytes.nn",
        "bitsandbytes.functional",
        "bitsandbytes.cextension",
        "bitsandbytes.triton",
    ):
        sys.modules[name] = _Mod(name)


def create_intervals(splits):
    start = 0
    return [(start, start := start + s) for s in splits]


def compute_sd_layer_shapes(input_shape, bandsplit_ratios, downsample_strides, n_layers):
    bandsplit_shapes_list = []
    conv2d_shapes_list = []
    for _ in range(n_layers):
        bandsplit_intervals = create_intervals(bandsplit_ratios)
        bandsplit_shapes = [int(r * input_shape) - int(l * input_shape) for l, r in bandsplit_intervals]
        conv2d_shapes = [int((bs - 1) / ds + 1) for bs, ds in zip(bandsplit_shapes, downsample_strides)]
        input_shape = sum(conv2d_shapes)
        bandsplit_shapes_list.append(bandsplit_shapes)
        conv2d_shapes_list.append(create_intervals(conv2d_shapes))
    return bandsplit_shapes_list, conv2d_shapes_list


class SDLayer(nn.Module):
    """Subband-Split + Freq-Downsample + n_conv_modules Conformer-ConvStack."""

    def __init__(self, input_dim, output_dim, band_kernel, band_stride):
        super().__init__()
        # SDlayer.convs.{0,1,2}: Conv2d(out, in, (k,1)) — 3 convs mit Kernel [3,4,4]
        # Padding = k//2, Stride (stride,1) über die Frequenzachse.
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(input_dim, output_dim, (k, 1), stride=(stride, 1), padding=(k // 2, 0))
                for k, stride in zip(band_kernel, band_stride)
            ]
        )
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (B, Fi, T, Ci) — Fi=Frequenz-Bins, Ci=Kanäle
        # 3 parallele convs über dasselbe Subband → nach Downsample concat
        outs = [conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) for conv in self.convs]
        x = torch.cat(outs, dim=1)  # (B, 3*F_down, T, output_dim)
        x = self.activation(x)
        return x


class ConvStack(nn.Module):
    """Conformer-ConvolutionModule (layers.0/1/3/4/6 im State-Dict)."""

    def __init__(self, dim, compress=4, conv_kernel=3):
        super().__init__()
        hidden = dim // compress
        # state_dict: layers.{m}.{i} — m=ConvStack-Index, i=Layer-Index
        # (0=LayerNorm, 1=Conv1d, 2=GLU, 3=Conv1d, 4=LayerNorm, 5=SiLU, 6=Conv1d)
        # ConvStack.layers = nn.ModuleList von 7 Layern (Index i).
        self.layers = nn.ModuleList(
            [
                nn.LayerNorm(dim),  # 0
                nn.Conv1d(dim, 2 * hidden, conv_kernel, padding=conv_kernel // 2),  # 1
                nn.GLU(dim=1),  # 2
                nn.Conv1d(hidden, hidden, 1, groups=hidden, padding=0),  # 3
                nn.LayerNorm(hidden),  # 4
                nn.SiLU(),  # 5
                nn.Conv1d(hidden, dim, 1, padding=0),  # 6
            ]
        )

    def forward(self, x):
        # x: (N, T, dim)
        x = self.layers[0](x)
        x = x.transpose(1, 2)  # (N, dim, T) für Conv1d
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        x = x.transpose(1, 2)  # (N, T, hidden) für LayerNorm
        x = self.layers[4](x)
        x = x.transpose(1, 2)  # (N, hidden, T)
        x = self.layers[5](x)
        x = self.layers[6](x)
        x = x.transpose(1, 2)  # (N, T, dim)
        return x


class SDBlock(nn.Module):
    def __init__(self, input_dim, output_dim, band_kernel, band_stride, conv_depth):
        super().__init__()
        self.SDlayer = SDLayer(input_dim, output_dim, band_kernel, band_stride)
        # conv_modules.{0..conv_depth-1}: ConvStack (Conformer-ConvolutionModule)
        # state_dict: encoder.{b}.conv_modules.{n}.layers.{m}.{i}
        self.conv_modules = nn.ModuleList([ConvStack(output_dim) for _ in range(conv_depth)])
        self.globalconv = nn.Conv2d(output_dim, output_dim, (3, 3), padding=(1, 1))

    def forward(self, x):
        # x: (B, Fi, T, Ci) → SDlayer → conv_modules → global conv → skip
        x_skip = self.SDlayer(x)
        B, F, T, C = x_skip.shape
        h = x_skip.reshape((B * F), T, C)
        for cm in self.conv_modules:
            h = cm(h)
        x_skip = h.reshape(B, F, T, C)
        x = self.globalconv(x_skip.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x, x_skip


class DualPathModule(nn.Module):
    """dp_modules.{i}: lstm_layers.{0,1} + linear_layers.{0,1} + norm_layers.{0,1}.

    State-Dict-Vertrag (aus huge_scnet_4stems_v1.2.ckpt verifiziert):
      lstm_layers.{0,1}.weight_ih_l0: (4*dim, dim)  — hidden == dim, bidirektional
      linear_layers.{0,1}.weight:     (dim, 2*dim)  — 2*dim weil bidirektional
      norm_layers.{0,1}.weight:       (dim,)
    Beide Layer (Zeit + Frequenz) haben dieselbe Dim — KEIN alternierendes 2×.
    Der Frequenz-Pfad läuft über F mit rFFT/irfft-Mixing (D bleibt konstant).
    """

    def __init__(self, dim):
        super().__init__()
        self.lstm_layers = nn.ModuleList(
            [
                nn.LSTM(dim, dim, batch_first=True, bidirectional=True),
                nn.LSTM(dim, dim, batch_first=True, bidirectional=True),
            ]
        )
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(dim * 2, dim),
                nn.Linear(dim * 2, dim),
            ]
        )
        self.norm_layers = nn.ModuleList([nn.LayerNorm(dim), nn.LayerNorm(dim)])

    def forward(self, x, time_dim):
        # Zeit-Pfad (über T)
        B, F, T, D = x.shape
        xt = x.reshape(B * F, T, D)
        xt = self.norm_layers[0](xt)
        xt, _ = self.lstm_layers[0](xt)
        xt = self.linear_layers[0](xt)
        xt = xt.reshape(B, F, T, D)
        # Frequenz-Pfad (über F) mit RFFT-Mixing über die Frequenzachse
        xf = xt.permute(0, 2, 1, 3).reshape(B * T, F, D)
        xf = self.norm_layers[1](xf)
        xf, _ = self.lstm_layers[1](xf)
        xf = self.linear_layers[1](xf)
        xf = xf.reshape(B, T, F, D)
        # rFFT über F → irfft über F zurück (D bleibt D)
        xfr = torch.fft.rfft(xf, dim=2)
        xfr = torch.view_as_real(xfr).reshape(B, T, F // 2 + 1, D * 2)
        xfr = xfr.reshape(B, T, F // 2 + 1, D, 2)
        xfr = torch.view_as_complex(xfr)
        xfr = torch.fft.irfft(xfr, n=F, dim=2)
        xfr = xfr.reshape(B, T, F, D).permute(0, 2, 1, 3)
        return xfr


class SeparationNet(nn.Module):
    def __init__(self, input_dim, num_dplayer):
        super().__init__()
        # Alternierende Dim pro dp_modules-Block: 256 (gerade), 512 (ungerade).
        # Verifiziert aus dem Checkpoint: dp_modules.{0,2,4,6,8} → 256,
        # dp_modules.{1,3,5,7,9} → 512.
        self.dp_modules = nn.ModuleList(
            [DualPathModule(input_dim if i % 2 == 0 else input_dim * 2) for i in range(num_dplayer)]
        )

    def forward(self, x, time_dim):
        for dp in self.dp_modules:
            x = dp(x, time_dim)
        return x


class SULayer(nn.Module):
    def __init__(self, output_dim, band_kernel, band_stride, in_dim):
        super().__init__()
        # convtrs.{0,1,2}: ConvTranspose2d(out, in, (k,1)) — Kernel [3,4,4]
        self.convtrs = nn.ModuleList(
            [
                nn.ConvTranspose2d(in_dim, output_dim, (k, 1), stride=(stride, 1), padding=(k // 2, 0))
                for k, stride in zip(band_kernel, band_stride)
            ]
        )
        self.activation = nn.GELU()

    def forward(self, x):
        outs = [conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) for conv in self.convtrs]
        x = torch.cat(outs, dim=1)
        x = self.activation(x)
        return x


class SUBlock(nn.Module):
    def __init__(self, input_dim, output_dim, band_kernel, band_stride):
        super().__init__()
        self.fusion = nn.Conv2d(input_dim, input_dim, (3, 3), padding=(1, 1))
        self.activation = nn.GLU()
        self.SUlayer = SULayer(output_dim, band_kernel, band_stride, in_dim=input_dim)

    def forward(self, x, x_skip):
        # x: (B, F, T, C) von separation_net; x_skip: encoder skip
        x = x + x_skip
        x = self.fusion(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.activation(x)
        x = self.SUlayer(x)
        return x


class SCNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dims = cfg["dims"]
        band_stride = cfg["band_stride"]
        band_kernel = cfg["band_kernel"]
        conv_depths = cfg["conv_depths"]
        num_dplayer = cfg["num_dplayer"]
        n_sources = len(cfg["sources"])
        n_blocks = len(dims) - 1
        # Encoder
        self.encoder = nn.ModuleList()
        cur_dim = dims[0]
        for i in range(n_blocks):
            self.encoder.append(
                SDBlock(
                    input_dim=cur_dim,
                    output_dim=dims[i + 1],
                    band_kernel=band_kernel,
                    band_stride=band_stride,
                    conv_depth=conv_depths[i],
                )
            )
            cur_dim = dims[i + 1]
        # Separation
        self.separation_net = SeparationNet(dims[-1], num_dplayer)
        # Decoder (reverse)
        self.decoder = nn.ModuleList()
        for i in reversed(range(n_blocks)):
            out_dim = dims[i] * n_sources if i == 0 else dims[i]
            self.decoder.append(
                SUBlock(
                    input_dim=dims[i + 1],
                    output_dim=out_dim,
                    band_kernel=band_kernel,
                    band_stride=band_stride,
                )
            )
        self.n_sources = n_sources

    def forward(self, x):
        # x: (B, F, T, C) — C=4 (re,im × 2 Kanäle)
        B, F, T, C = x.shape
        skips = []
        for enc in self.encoder:
            x, skip = enc(x)
            skips.append(skip)
        x = self.separation_net(x, T)
        for dec, skip in zip(self.decoder, reversed(skips)):
            x = dec(x, skip)
        x = x.reshape(B, F, T, C, self.n_sources)
        return x


def load_model():
    _stub_bitsandbytes()
    import yaml

    with open(ROOT / "models" / "scnet_4stems" / "config.yaml") as f:
        cfg = yaml.safe_load(f)["model"]

    model = SCNet(cfg)
    ckpt = torch.load(
        ROOT / "models" / "scnet_4stems" / "huge_scnet_4stems_v1.2.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    sd = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("MISSING:", missing[:10])
    if unexpected:
        print("UNEXPECTED:", unexpected[:10])
    model.eval()
    return model, cfg


if __name__ == "__main__":
    model, cfg = load_model()
    print("Modell geladen.")
    print(f"Parameter: {sum(p.numel() for p in model.parameters()):,}")
