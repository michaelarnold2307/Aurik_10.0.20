"""Vertragstests für das D3-3-Gate (`scripts/verify_flashsr_d3_gate.py`, D-K3-94).

Geprüft wird die **Mess-Logik**, nicht das Modell: Gate-Schwelle,
Never-worsen, Bandtrennung, Residual-Konstruktion (Basis-Passband bleibt),
Diagnose gegen Mess-Artefakte (Peilung/Pegel), Determinismus der Track-Auswahl
und der CPU-Guard (die Trainingskette darf nie belegt werden).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import verify_flashsr_d3_gate as gate


def _rauschen(n: int = 48000, seed: int = 0, amplitude: float = 0.1) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(n).astype(np.float32) * amplitude).astype(np.float32)


def test_gate_urteil_schwelle_und_never_worsen() -> None:
    assert gate._gate_urteil(2.1, 0.1)["bestanden"] is True
    knapp = gate._gate_urteil(1.99, 0.5)
    assert knapp["bestanden"] is False and "Schwelle" in knapp["grund"]
    nie = gate._gate_urteil(3.0, -0.01)
    assert nie["bestanden"] is False and "Never-worsen" in nie["grund"]


def test_residual_erhaelt_das_basis_passband() -> None:
    """Die Mess-Option darf das Passband der Basis nicht verändern."""
    ref = _rauschen(seed=1)
    basis = gate._band(ref, hi=gate.GRENZE_HF).astype(np.float32)
    est = _rauschen(seed=2)
    res = gate._residual(est, basis)
    differenz = gate._band(res, hi=gate.GRENZE_HF) - gate._band(basis, hi=gate.GRENZE_HF)
    assert np.max(np.abs(differenz)) < 1e-6, float(np.max(np.abs(differenz)))
    # das Hochband stammt aus der Schätzung, nicht aus der Basis
    assert np.sqrt(np.mean(gate._band(res, lo=gate.GRENZE_HF) ** 2)) > 0.0


def test_band_trennt_passband_und_hochband() -> None:
    x = _rauschen(seed=3)
    tief = gate._band(x, hi=gate.GRENZE_HF)
    hoch = gate._band(x, lo=gate.GRENZE_HF)
    assert abs(float(np.sum(tief**2) + np.sum(hoch**2) - np.sum(x.astype(np.float64) ** 2))) < 1e-6 * float(
        np.sum(x.astype(np.float64) ** 2)
    )


def test_beste_verzoegerung_erkennt_shift() -> None:
    x = _rauschen(seed=4)
    assert gate._beste_verzoegerung(np.roll(x, 7), x) == 7


def test_optimaler_gain_und_rms() -> None:
    x = _rauschen(seed=5, amplitude=0.2)
    assert gate._optimaler_gain(0.5 * x, x) == pytest.approx(2.0, rel=1e-4)
    assert gate._rms(0.5 * x) == pytest.approx(0.5 * gate._rms(x), rel=1e-6)


def test_messe_liefert_gate_und_diagnose() -> None:
    ref = _rauschen(seed=6)
    basis = gate._band(ref, hi=gate.GRENZE_HF).astype(np.float32)
    # SNR ~= -6 dB: schlechter als die 16-kHz-Basis (~+3 dB bei weissem Rauschen)
    est = (ref + 2.0 * _rauschen(seed=7)).astype(np.float32)
    ergebnis = gate._messe(ref, est, basis)
    for schluessel in (
        "mean_dSDR",
        "min_dSDR",
        "n_segmente",
        "passband_sdr_est_db",
        "hf_sdr_est_db",
        "rms_verhaeltnis",
        "optimale_verstaerkung",
        "beste_verzoegerung_samples",
        "gate",
    ):
        assert schluessel in ergebnis, schluessel
    assert ergebnis["n_segmente"] == 1
    assert ergebnis["mean_dSDR"] < 0.0
    assert ergebnis["gate"]["bestanden"] is False
    # Pegel-Diagnose separat: halbierter Pegel (kein Defekt, nur Skala)
    halb = gate._messe(ref, 0.5 * ref, basis)
    assert halb["rms_verhaeltnis"] == pytest.approx(0.5, rel=1e-3)
    assert halb["optimale_verstaerkung"] == pytest.approx(2.0, rel=1e-3)


def test_waehle_tracks_deterministisch() -> None:
    a = gate._waehle_tracks(3, 42)
    b = gate._waehle_tracks(3, 42)
    if not a:
        pytest.skip("MUSDB18HQ-Testset lokal nicht vorhanden")
    assert a == b
    assert len(a) == min(3, len(sorted(p for p in gate.MUSDB_TEST.glob("*") if p.is_dir())))
    assert a == sorted(a)


def test_schema_konstante() -> None:
    assert gate.SCHEMA == "aurik.flashsr_d3_gate/1"
    assert gate.SCHWELLE_DB == 2.0


def test_cpu_guard_fail_closed_bei_sichtbarer_gpu(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    grund = gate._cpu_guard()
    assert grund is not None and "GPU" in grund

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert gate._cpu_guard() is None
