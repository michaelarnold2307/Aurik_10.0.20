#!/usr/bin/env python3
"""D-K3-96-Teile 2/3 + D-K4-11-Klassen-Ehrlichkeit (Vertragstests, modell-frei/CPU).

- ``scripts._vendor_bigvgan_disc``: MPD-Form, Hinge-Verluste, Gradientenfluss,
  Determinismus (§G5 (GEBOTE.md)) — der Diskriminator ist Default-AUS, der Alt-Pfad bleibt.
- ``train_bigvgan_f3.random_crops``: Crop-Länge folgt CROP_N (Teil 3 „längere
  Schedule") und ist deterministisch.
- ``train_gender_head._klassen_bilanz``: zählt ehrlich; fehlende Klassen
  werden nicht erfunden (child=0 → 2-Klassen-Training).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import scripts.train_bigvgan_f3 as trainer
from scripts._vendor_bigvgan_disc import (
    MultiPeriodDiscriminator,
    discriminator_hinge_loss,
    generator_hinge_loss,
)
from scripts.train_gender_head import _klassen_bilanz


class TestMultiPeriodDiscriminator:
    def test_forward_form_und_hinge(self) -> None:
        torch.manual_seed(42)
        _disc = MultiPeriodDiscriminator()
        _x = torch.randn(2, 1, 16000)  # [B, C, T]
        _out, _feats = _disc(_x)
        assert _out.shape == (2,)
        assert len(_feats) == 6 * 5  # 5 Perioden × (5 Conv-Features + 1 Post)
        _d_real = _out
        _d_fake = _disc(torch.randn(2, 1, 16000))[0]
        _g = generator_hinge_loss(_d_fake)
        _d = discriminator_hinge_loss(_d_real, _d_fake)
        assert _g.shape == torch.Size([]) and _d.shape == torch.Size([])
        assert float(_d.item()) >= 0.0  # Hinge ist nicht-negativ

    def test_gradientenfluss_und_determinismus(self) -> None:
        _disc = MultiPeriodDiscriminator()
        _x = torch.randn(2, 1, 16000, requires_grad=False)
        _y = torch.randn(2, 1, 16000, requires_grad=True)
        _d_real, _ = _disc(_x)
        _d_fake, _ = _disc(_y)
        _loss = discriminator_hinge_loss(_d_real, _d_fake)
        _loss.backward()
        assert _y.grad is not None and float(_y.grad.abs().sum().item()) > 0.0
        # Determinismus: gleicher Input, gleiche Gewichte → gleicher Output
        _disc2 = MultiPeriodDiscriminator()
        _disc2.load_state_dict(_disc.state_dict())
        _fix = torch.randn(1, 1, 8000)
        _out_a = _disc(_fix)[0]
        _disc2.load_state_dict(_disc.state_dict())
        _out_b = _disc2(_fix)[0]
        assert torch.equal(_out_a, _out_b)


class TestCropSchedule:
    def test_crop_laenge_folgt_crop_n(self) -> None:
        _track = np.random.default_rng(42).standard_normal(48000 * 2).astype(np.float32)
        _rng = np.random.default_rng(42)
        _alt = trainer.CROP_N
        try:
            trainer.CROP_N = int(0.5 * trainer.SR)
            _kurz = trainer.random_crops(_track, 2, _rng)
            trainer.CROP_N = int(2.0 * trainer.SR)
            _lang = trainer.random_crops(_track, 2, _rng)
        finally:
            trainer.CROP_N = _alt
        assert _kurz.shape == (2, int(0.5 * trainer.SR))
        assert _lang.shape == (2, int(2.0 * trainer.SR))

    def test_crop_determinismus(self) -> None:
        _track = np.random.default_rng(7).standard_normal(48000 * 2).astype(np.float32)
        _a = trainer.random_crops(_track, 2, np.random.default_rng(42))
        _b = trainer.random_crops(_track, 2, np.random.default_rng(42))
        assert np.array_equal(_a.numpy(), _b.numpy())


class TestKlassenBilanz:
    def test_bilanz_ehrlich_ohne_child(self) -> None:
        _pairs = [("a.wav", "male"), ("b.wav", "female"), ("c.wav", "male")]
        _b = _klassen_bilanz(_pairs)
        assert _b == {"male": 2, "female": 1, "child": 0}

    def test_unbekanntes_label_wird_gezaehlt_nicht_erfunden(self) -> None:
        _b = _klassen_bilanz([("a.wav", "male")])
        assert _b["child"] == 0
