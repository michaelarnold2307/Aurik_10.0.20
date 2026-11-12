#!/usr/bin/env python3
"""D-K3-96-Mel-Wahrheit + §G5 (GEBOTE.md)-Fingerprint-Determinismus (Vertragstests, modell-frei).

- ``train_bigvgan_f3.extract_mel`` muss das PRODUKTIONS-Mel liefern
  (power_to_db(ref=max), Mittel ≈ −40,7 — nicht das log(abs-STFT)-Mel mit
  Mittel ≈ −3,2, D-K3-96). Geprüft wird die Richtung (Mittel < −20) und die
  Delegation an ``compute_mel_produktion`` (EINE Wahrheit, §G9 (copilot-instructions.md)).
- ``compute_audio_fingerprint`` darf KEIN ``time.time()`` mehr enthalten
  (§G5 (copilot-instructions.md); der Alt-Fallback war zeitbasiert und damit
  nicht-deterministisch — Quelltext-Pin).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

import scripts.train_bigvgan_f3 as trainer
from backend.core.song_strategy_cache import compute_audio_fingerprint
from plugins.bigvgan_v2_plugin import compute_mel_produktion


def _welle(sek: float, sr: int) -> np.ndarray:
    _t = np.arange(int(sek * sr), dtype=np.float64) / sr
    return (0.25 * np.sin(2 * np.pi * 440.0 * _t) + 0.1 * np.sin(2 * np.pi * 110.0 * _t)).astype(np.float32)


class TestMelWahrheit:
    def test_trainer_mel_ist_produktions_mel(self) -> None:
        _np = _welle(2.0, trainer.SR)
        _wave = torch.from_numpy(_np).unsqueeze(0)
        _trainer_mel = trainer.extract_mel(_wave)
        _prod_mel = compute_mel_produktion(_np, trainer.SR)
        assert _trainer_mel.shape[0] == 1
        assert _trainer_mel.shape[1] == 128  # MEL_BANDS
        assert float(_trainer_mel.mean()) < -20.0  # Produktions-Richtung (log-Mel wäre ≈ −3,2)
        # Delegation: identische Werte zur Produktions-Funktion
        assert np.allclose(_trainer_mel[0].numpy(), _prod_mel, atol=0.0)

    def test_mel_clip_und_determinismus(self) -> None:
        _np = _welle(1.0, trainer.SR)
        _a = compute_mel_produktion(_np, trainer.SR)
        _b = compute_mel_produktion(_np, trainer.SR)
        assert np.array_equal(_a, _b)
        assert float(_a.min()) >= -80.0 and float(_a.max()) <= 0.0
        assert np.all(np.isfinite(_a))


class TestFingerprintG5:
    def test_kein_time_time_im_fingerprint(self) -> None:
        _src = inspect.getsource(compute_audio_fingerprint)
        # Der Alt-Fallback war der Aufruf str(time.time()) — genau dieser
        # Aufruf darf nicht mehr vorkommen (§G5 (copilot-instructions.md)).
        assert "str(time.time())" not in _src

    def test_fingerprint_deterministisch(self) -> None:
        _np = _welle(3.0, 48000)
        assert compute_audio_fingerprint(_np, 48000) == compute_audio_fingerprint(_np, 48000)
        _anders = _welle(3.0, 48000)
        _anders += 0.05 * _anders  # echtes anderes Signal → anderer Fingerprint
        assert compute_audio_fingerprint(_np, 48000) != compute_audio_fingerprint(_anders, 48000)
