#!/usr/bin/env python3
"""Vertragstests fuer scripts/trainings_flugschein.py (modell-frei wo möglich).

Trainingsstrategie (Report `2026-11-13_hoerzentrierte_trainingsstrategie.md`):
Stufe A (Flugschein) muss das D-K3-96-Muster (Mel-Drift) in Sekunden ablehnen
und die Produktions-Trainer durchlassen; Stufe B (Epoch-Gate) muss eine
Basis-Degradation erkennen. Zitierdisziplin AGENTS.md §5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import scripts.trainings_flugschein as tf

_ECHTER_TRAINER = Path(__file__).resolve().parents[2] / "scripts" / "train_bigvgan_f3.py"


def _schreibe_trainer(pfad: Path, quelltext: str) -> None:
    pfad.write_text(quelltext, encoding="utf-8")


def test_mel_paritaet_lehnt_d_k3_96_muster_ab(tmp_path) -> None:
    boese = tmp_path / "trainer_boese.py"
    _schreibe_trainer(
        boese,
        "mel = np.log(np.abs(stft(x)))\n# kein compute_mel_produktion\ndef train():\n    pass\n",
    )
    ok, befund = tf.mel_paritaet_pruefen(boese)
    assert ok is False
    assert "Mel-Parität" in befund


def test_mel_paritaet_laesst_produktionstrainer_durch() -> None:
    ok, befund = tf.mel_paritaet_pruefen(_ECHTER_TRAINER)
    assert ok is True, befund
    assert "compute_mel_produktion" in befund


def test_gate_hint_pflicht(tmp_path) -> None:
    ohne = tmp_path / "trainer_ohne_hint.py"
    _schreibe_trainer(ohne, "def train():\n    pass\n")
    ok, _ = tf.gate_hint_pruefen(ohne)
    assert ok is False
    ok2, _ = tf.gate_hint_pruefen(_ECHTER_TRAINER)
    assert ok2 is True


def test_daten_fail_closed(tmp_path) -> None:
    assert tf.daten_pruefen(tmp_path / "fehlt.jsonl")[0] is False
    leer = tmp_path / "leer.jsonl"
    leer.write_text("\n", encoding="utf-8")
    assert tf.daten_pruefen(leer)[0] is False
    synthetik = tmp_path / "synthetik.jsonl"
    synthetik.write_text('{"quelle": "synthetic 4000", "paare": []}\n', encoding="utf-8")
    ok, befund = tf.daten_pruefen(synthetik)
    assert ok is False, "synthetischer Fallback ohne sha256_pcm muss abgelehnt werden"
    assert "synthetische" in befund
    gut = tmp_path / "gut.jsonl"
    gut.write_text('{"pfad": "a.wav", "sha256_pcm": "abc123"}\n', encoding="utf-8")
    assert tf.daten_pruefen(gut)[0] is True


def _sine(n: int, sr: int = 48000) -> np.ndarray:
    t = np.arange(n) / sr
    return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def test_epoch_gate_erkennt_degradation() -> None:
    """Stufe B: Basis = Sinus (hoher HNR), Finetune = Sinus+Rauschen ⇒ Abbruch."""
    import torch

    sr = 48000
    mono = _sine(sr)
    mel_probe = None

    def _n_out(mel: torch.Tensor) -> int:
        return int(mel.shape[-1] * 256)

    def basis_fn(mel: torch.Tensor) -> torch.Tensor:
        n = _n_out(mel)
        return torch.from_numpy(_sine(n)).unsqueeze(0)

    def gut_fn(mel: torch.Tensor) -> torch.Tensor:
        n = _n_out(mel)
        return torch.from_numpy(_sine(n)).unsqueeze(0)

    def boese_fn(mel: torch.Tensor) -> torch.Tensor:
        n = _n_out(mel)
        rng = np.random.default_rng(42)
        rauschig = _sine(n) + 0.1 * rng.standard_normal(n).astype(np.float32)
        return torch.from_numpy(rauschig).unsqueeze(0)

    befunde_gut = tf.epoch_gate_pruefung(gut_fn, basis_fn, mono, sr, torch.device("cpu"))
    assert befunde_gut["abbruch"] is False, befunde_gut
    befunde_boese = tf.epoch_gate_pruefung(boese_fn, basis_fn, mono, sr, torch.device("cpu"))
    assert befunde_boese["degradation_db"] < -1.0, befunde_boese
    assert befunde_boese["abbruch"] is True
