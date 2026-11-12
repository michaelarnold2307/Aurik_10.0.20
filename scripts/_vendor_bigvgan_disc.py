#!/usr/bin/env python3
"""Vendored Multi-Period-Diskriminator (BigVGAN-Upstream-Stil, MIT).

Quelle: nvidia/bigvgan (discriminator.py, MIT-Lizenz), auf den für Aurik
benötigten Umfang reduziert (MPD mit Perioden 2/3/5/7/11, keine
Multi-Scale-Stufe). Vendored, weil das installierte ``bigvgan``-Paket nur
``meldataset`` trägt, KEINEN Diskriminator.

Zweck: D-K3-96-Rezept Teil 2 — der F3-Finetune soll optional mit
Diskriminator trainiert werden (``train_bigvgan_f3.py --gan``); Default AUS,
damit der bestehende Trainingspfad bit-identisch bleibt.

MIT-Hinweis: Copyright (c) 2022 NVIDIA CORPORATION. Die Original-Lizenz
liegt dem Upstream-Repo bei; die Reduktion behält die MIT-Terms bei
(LICENSE-Datei des Upstreams im Repo unter docs/licenses/).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

_MPD_PERIODS = (2, 3, 5, 7, 11)


class _PeriodDiscriminator(nn.Module):
    """Ein Perioden-Diskriminator (1-D-Conv-Stapel mit LeakyReLU + WeightNorm)."""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3) -> None:
        super().__init__()
        self.period = period
        _convs = [
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(kernel_size // 2, 0))),
        ]
        self.convs = nn.ModuleList(_convs)
        self.post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        _feats: list[torch.Tensor] = []
        _x = x
        if _x.shape[-1] % self.period != 0:
            _n = self.period - (_x.shape[-1] % self.period)
            _x = F.pad(_x, (0, _n), "reflect")
        _x = _x.view(_x.shape[0], 1, -1, self.period)
        for _conv in self.convs:
            _x = F.leaky_relu(_conv(_x), 0.1)
            _feats.append(_x)
        _x = self.post(_x)
        _feats.append(_x)
        _out = _x.flatten(1).mean(-1)
        return _out, _feats


class MultiPeriodDiscriminator(nn.Module):
    """MPD: Summe der Perioden-Diskriminatoren (BigVGAN-Upstream-Stil)."""

    def __init__(self, periods: tuple[int, ...] = _MPD_PERIODS) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([_PeriodDiscriminator(p) for p in periods])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        _outs: list[torch.Tensor] = []
        _feats: list[torch.Tensor] = []
        for _d in self.discriminators:
            _o, _f = _d(x)
            _outs.append(_o)
            _feats.extend(_f)
        return torch.stack(_outs, dim=0).mean(0), _feats


def generator_hinge_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """Generator-Hinge: -mean(D(fake)) — G will D(fake) → +∞."""
    return -d_fake.mean()


def discriminator_hinge_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Diskriminator-Hinge: relu(1-D(real)) + relu(1+D(fake))."""
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
