"""Tests für das C1-UTMOS-Delta-Veto (Zeugen-Prinzip, Hörordnung §8)."""

from __future__ import annotations

import numpy as np

import backend.core.dsp.utmos_delta_gate as ug_mod


def _profile() -> dict:
    return {"global_scalar": 1.0, "family_scalars": {"enhancement": 1.0}}


def test_no_measurement_no_veto(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """UTMOS nicht verfügbar (delta=None) → KEIN Veto, Profil unverändert."""
    monkeypatch.setattr(
        "backend.core.evaluation_system._compute_utmos_delta",
        lambda damaged, restored, sr: None,
    )
    prof = _profile()
    res = ug_mod.utmos_delta_veto(prof, np.zeros(1000), np.zeros(1000), 44100)
    assert res is None
    assert prof["global_scalar"] == 1.0
    assert prof["family_scalars"]["enhancement"] == 1.0


def test_clear_regression_vetoes(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "backend.core.evaluation_system._compute_utmos_delta",
        lambda damaged, restored, sr: -0.25,  # klare MOS-Regression
    )
    prof = _profile()
    res = ug_mod.utmos_delta_veto(prof, np.zeros(1000), np.zeros(1000), 44100)
    assert res is not None and res.applied is True
    assert prof["family_scalars"]["enhancement"] == 0.9
    assert prof["global_scalar"] == 0.95


def test_small_delta_no_veto(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "backend.core.evaluation_system._compute_utmos_delta",
        lambda damaged, restored, sr: -0.02,  # unter der Schwelle
    )
    prof = _profile()
    res = ug_mod.utmos_delta_veto(prof, np.zeros(1000), np.zeros(1000), 44100)
    assert res is None
    assert prof["family_scalars"]["enhancement"] == 1.0


def test_improvement_no_veto(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "backend.core.evaluation_system._compute_utmos_delta",
        lambda damaged, restored, sr: +0.3,
    )
    prof = _profile()
    res = ug_mod.utmos_delta_veto(prof, np.zeros(1000), np.zeros(1000), 44100)
    assert res is None
