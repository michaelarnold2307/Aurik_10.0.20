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


def test_flugschein_pflicht_nur_mel_familie() -> None:
    assert tf.flugschein_pflicht(_ECHTER_TRAINER) is True  # train_bigvgan_f3.py
    assert tf.flugschein_pflicht(Path("scripts/train_flashsr_f4.py")) is True
    vocos = Path(__file__).resolve().parents[2] / "scripts" / "train_vocos_48khz_finetune.py"
    assert tf.flugschein_pflicht(vocos) is True
    assert tf.flugschein_pflicht(Path("scripts/train_cantus.py")) is False
    assert tf.flugschein_pflicht(Path("scripts/train_gender_head.py")) is False


def test_vocos_flugschein_pinnt_produktionsmel() -> None:
    trainer = Path(__file__).resolve().parents[2] / "scripts" / "train_vocos_48khz_finetune.py"
    ok, befund = tf.flugschein_trainer_pruefen(trainer)
    assert ok, befund
    assert "VocosPlugin._compute_mel" in befund


def test_vocos_train_mel_ist_produktionsmel() -> None:
    from plugins.vocos_plugin import _HOP_48K, _N_FFT_48K, _N_MELS_48K, VocosPlugin
    from scripts.train_vocos_48khz_finetune import _mel_from_wave

    audio = _sine(8192)
    erwartet = VocosPlugin._compute_mel(
        audio,
        48000,
        n_mels=_N_MELS_48K,
        n_fft=_N_FFT_48K,
        hop=_HOP_48K,
    )
    np.testing.assert_array_equal(_mel_from_wave(audio), erwartet)


def test_vocos_flugschein_lehnt_modellinterne_eingangsmel_ab(tmp_path) -> None:
    trainer = tmp_path / "train_vocos_48khz_finetune.py"
    _schreibe_trainer(
        trainer,
        "mel = model.feature_extractor(wt)\n"
        'GATE_HINT = "gate_hint"\n'
        "def _mel_from_wave(wav):\n"
        "    return VocosPlugin._compute_mel(wav, 48000)\n"
        "mel = np.stack([_mel_from_wave(sample) for sample in chunk], axis=0)\n"
        '"gate_hint": GATE_HINT\n',
    )
    ok, befund = tf.vocos_mel_paritaet_pruefen(trainer)
    assert not ok
    assert "Vocos-Mel-Parität" in befund


def test_runner_lehnt_flashsr_ohne_gpu_job_ab(tmp_path) -> None:
    """train_job_runner: Flugschein ABGELEHNT ⇒ Manifest + Exit 3, KEIN Subprozess."""
    import scripts.train_job_runner as runner

    manifest = tmp_path / "job.json"
    exit_code = runner.run_job(
        name="test_flashsr",
        ckpt_dir=tmp_path / "ckpt",
        cmd=["python3", "scripts/train_flashsr_f4.py", "--help"],
        resume_cmd=["python3", "scripts/train_flashsr_f4.py", "--help"],
        phase="test",
        manifest_path=manifest,
        force=True,
    )
    assert exit_code == 3
    inhalt = __import__("json").loads(manifest.read_text(encoding="utf-8"))
    assert inhalt["ergebnis"] == "flugschein_abgelehnt"
    assert "Mel-Parität" in inhalt["begruendung"]


def test_trainer_aus_cmd() -> None:
    import scripts.train_job_runner as runner

    assert (
        runner._trainer_aus_cmd(["env", "X=1", "python3", "-B", "scripts/train_bigvgan_f3.py", "--data", "x"]).name
        == "train_bigvgan_f3.py"
    )
    assert runner._trainer_aus_cmd(["python3", "-B", "scripts/train_cantus.py"]).name == "train_cantus.py"
    assert runner._trainer_aus_cmd(["python3", "-m", "pytest"]) is None
