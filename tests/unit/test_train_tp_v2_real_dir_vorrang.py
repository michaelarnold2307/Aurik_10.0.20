#!/usr/bin/env python3
"""Regressionstests TP-V2-Trainingsdaten: ``--real-dir`` hat Vorrang (§III.11).

Befund 2026-11-06: ``scripts/train_tp_v2_phase.py`` nutzte synthetische
Segmente, sobald ``--synthetic > 0`` — und der argparse-Default ist 16. Die
Kette 4 (``scripts/gpu_kette4_modellkoepfe.sh``) übergab zwar
``--real-dir data/tp2_transient_pairs``, trainierte damit aber unauffällig auf
16 synthetischen Segmenten statt auf MUSDB18-HQ-Musik. Der Fix macht die
explizite Absicht stark: liegen reale ``*.wav`` vor, gewinnt ``--real-dir``
und der synthetische Zähler wird mit Warnung ignoriert (§V7 (copilot-instructions.md)
(copilot-instructions.md) — Ursache statt Symptom).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.train_tp_v2_phase import TransientPhaseDataset

_SR = 48_000


def _schreibe_segmente(root: Path, n: int = 2) -> None:
    root.mkdir(parents=True, exist_ok=True)
    t = np.arange(_SR, dtype=np.float64) / float(_SR)
    for i in range(n):
        seg = (0.4 * np.sin(2 * np.pi * (110.0 + 10.0 * i) * t)).astype(np.float32)
        sf.write(str(root / f"seg_{i:04d}.wav"), seg, _SR, subtype="FLOAT")


def test_real_dir_gewinnt_gegen_synthetic_default(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Mit realen Segmenten darf der Default ``--synthetic 16`` nicht gewinnen."""
    _schreibe_segmente(tmp_path)

    with caplog.at_level(logging.WARNING, logger="scripts.train_tp_v2_phase"):
        ds = TransientPhaseDataset(synthetic=16, real_dir=tmp_path, seed=42)

    assert ds._synthetic == 0, "synthetische Segmente trotz --real-dir aktiv"
    # len() folgt den realen Segmenten (2 Dateien × 32), nicht dem Default 16.
    assert len(ds) == 64
    assert any("ignoriert" in r.message for r in caplog.records), (
        "Vorrang nicht geloggt (§V6 (copilot-instructions.md))"
    )

    log_mag, delta = ds[0]
    assert log_mag.ndim == 2 and delta.ndim == 2
    assert np.all(np.isfinite(log_mag)) and np.all(np.isfinite(delta))


def test_leeres_real_dir_faellt_mit_warnung_zurueck(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Leeres Verzeichnis ⇒ synthetischer Ersatz, aber LAUT (§V6 (copilot-instructions.md))."""
    leer = tmp_path / "leer"
    leer.mkdir()

    with caplog.at_level(logging.WARNING, logger="scripts.train_tp_v2_phase"):
        ds = TransientPhaseDataset(synthetic=8, real_dir=leer, seed=42)

    assert ds._synthetic == 8
    assert len(ds) == 8
    assert any("keine" in r.message for r in caplog.records)


def test_ohne_real_dir_bleibt_synthetisch() -> None:
    ds = TransientPhaseDataset(synthetic=4, real_dir=None, seed=42)
    assert ds._synthetic == 4
    assert len(ds) == 4
