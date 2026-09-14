"""Unit-Tests für §SOTA-PSY-A8 (hearing_jnd) und §SOTA-PSY-A1 (audibility_gate)."""

from __future__ import annotations

import numpy as np
import pytest

import backend.core.dsp.audibility_gate as ag
import backend.core.dsp.hearing_jnd as hj


def test_jnd_known_values() -> None:
    assert hj.jnd("level_broadband") == 1.0
    assert hj.jnd("time_gap") == 0.005
    assert hj.jnd("iacc") == 0.08
    with pytest.raises(KeyError):
        hj.jnd("nicht_existent")  # fail-closed


def test_below_jnd_semantics() -> None:
    assert hj.below_jnd(0.5, "level_broadband") is True
    assert hj.below_jnd(1.5, "level_broadband") is False
    assert hj.below_jnd(1.5, "level_broadband", factor=2.0) is True  # tolerant
    assert hj.below_jnd(float("nan"), "level_broadband") is False  # NaN → hörbar (fail-safe)


def test_defect_audibility_loud_click_audible() -> None:
    sr = 48000
    rng = np.random.RandomState(0)
    x = (rng.randn(sr) * 0.02).astype(np.float32)  # Rausch-Kontext
    d0, d1 = sr // 2 - 64, sr // 2 + 64
    x[d0:d1] += 0.6  # lauter Klick
    res = ag.defect_audibility(x, sr, d0, d1)
    assert res["audible"] is True
    assert res["skippable"] is False
    assert res["delta_db"] > res["threshold_db"]


def test_defect_audibility_masked_tiny_click_skippable() -> None:
    sr = 48000
    rng = np.random.RandomState(1)
    x = (rng.randn(sr) * 0.05).astype(np.float32)  # kräftiger Masker
    d0, d1 = sr // 2 - 32, sr // 2 + 32
    x[d0:d1] += 0.001  # winziger Defekt
    res = ag.defect_audibility(x, sr, d0, d1)
    assert res["skippable"] is True
    assert res["audible"] is False


def test_defect_audibility_deterministic_and_edge_cases() -> None:
    sr = 48000
    rng = np.random.RandomState(2)
    x = (rng.randn(sr) * 0.03).astype(np.float32)
    x[sr // 2 - 32 : sr // 2 + 32] += 0.5
    r1 = ag.defect_audibility(x, sr, sr // 2 - 32, sr // 2 + 32)
    r2 = ag.defect_audibility(x, sr, sr // 2 - 32, sr // 2 + 32)
    assert r1 == r2  # deterministisch (§G5 (GEBOTE.md))
    edge = ag.defect_audibility(x, sr, 10, 10)  # leeres Intervall
    assert edge["skippable"] is True


def test_defect_audibility_fail_open_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """§V6 (copilot-instructions.md): Fehler im Modell → Reparatur freigegeben (fail-open)."""
    import backend.core.dsp.masking_model as mm

    monkeypatch.setattr(
        mm, "compute_masking_threshold_db", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaputt"))
    )
    x = np.zeros(48000, dtype=np.float32)
    res = ag.defect_audibility(x, 48000, 100, 200)
    assert res["audible"] is True
    assert res["skippable"] is False
