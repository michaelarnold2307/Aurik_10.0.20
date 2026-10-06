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
    monkeypatch.setattr(ug_mod, "_domain_veto_allowed", lambda: True)
    monkeypatch.setattr(
        "backend.core.evaluation_system._compute_utmos_delta",
        lambda damaged, restored, sr: -0.25,  # klare MOS-Regression
    )
    prof = _profile()
    res = ug_mod.utmos_delta_veto(prof, np.zeros(1000), np.zeros(1000), 44100)
    assert res is not None and res.applied is True
    assert prof["family_scalars"]["enhancement"] == 0.9
    assert prof["global_scalar"] == 0.95


def test_speech_mos_veto_blocked_without_music_flag(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """§III.11: Sprach-MOS (BVCC) darf bei Musik/Gesang nicht richten.

    Ohne musik-kalibriertes MOS (`use_utmos_music=False`, Default) bleibt die
    klare Regression ein ZEUGNIS — kein Veto, Profil unverändert.
    """
    monkeypatch.setattr("backend.core.music_model_flags.use_utmos_music", False)
    monkeypatch.setattr(
        "backend.core.evaluation_system._compute_utmos_delta",
        lambda damaged, restored, sr: -0.25,
    )
    prof = _profile()
    res = ug_mod.utmos_delta_veto(prof, np.zeros(1000), np.zeros(1000), 44100)
    assert res is None
    assert prof["family_scalars"]["enhancement"] == 1.0
    assert prof["global_scalar"] == 1.0


def test_domain_flag_music_enables_veto(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Mit musik-kalibriertem MOS (F8) greift das Veto wieder."""
    monkeypatch.setattr("backend.core.music_model_flags.use_utmos_music", True)
    monkeypatch.setattr(
        "backend.core.evaluation_system._compute_utmos_delta",
        lambda damaged, restored, sr: -0.25,
    )
    prof = _profile()
    res = ug_mod.utmos_delta_veto(prof, np.zeros(1000), np.zeros(1000), 44100)
    assert res is not None and res.applied is True
    assert prof["family_scalars"]["enhancement"] == 0.9


def test_domain_veto_allowed_default_is_locked() -> None:
    """Das Produktions-Flag steht auf False, bis F8/MUSHRA-Abnahme erfolgt ist."""
    from backend.core.music_model_flags import use_utmos_music  # pylint: disable=import-outside-toplevel

    assert use_utmos_music is False


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
