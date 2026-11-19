"""Unit-Tests für den R2-MuQ-MOS-Witness im ExportQualityGate (§0c-Vertrag).

Der Witness entscheidet NIE über den Export: `passed` bleibt unberührt;
der Output wird immer exportiert — nur Status „degraded“ + Warnung werden
annotiert, wenn MOS(out) unter (MOS(in) − Marge) fällt.
"""

from __future__ import annotations

import numpy as np
import pytest

import plugins.muq_plugin as _muq_mod
from backend.core.export_quality_gate import ExportQualityGate


def _audio() -> np.ndarray:
    rng = np.random.RandomState(0)
    return rng.randn(48000).astype(np.float32) * 0.05


def test_mos_witness_degraded_annotates_but_never_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    # Deterministischer Ersatz: Input (1. Aufruf) => 3.5, Output => 3.0.
    calls = []

    def _fake_mos(audio, sr):
        calls.append(float(np.mean(audio)))
        return 3.5 if len(calls) == 1 else 3.0

    monkeypatch.setattr(_muq_mod, "estimate_muq_mos", _fake_mos)
    out = _audio()
    result = ExportQualityGate.check(out, 48000, reference_audio=_audio())
    assert result.passed is True  # NIE Hardstop
    assert result.muq_mos_degraded is True
    assert result.muq_mos_delta == pytest.approx(-0.5, abs=0.01)
    assert any("degraded" in w for w in result.warnings)


def test_mos_witness_ok_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_muq_mod, "estimate_muq_mos", lambda audio, sr: 3.6)
    result = ExportQualityGate.check(_audio(), 48000, reference_audio=_audio())
    assert result.passed is True
    assert result.muq_mos_degraded is False
    assert result.muq_mos_delta == pytest.approx(0.0, abs=0.01)


def test_mos_witness_skipped_without_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []
    monkeypatch.setattr(_muq_mod, "estimate_muq_mos", lambda audio, sr: called.append(1) or 3.0)
    result = ExportQualityGate.check(_audio(), 48000)
    assert result.passed is True
    assert result.muq_mos_in is None
    assert called == []


def test_mos_witness_unavailable_is_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_muq_mod, "estimate_muq_mos", lambda audio, sr: None)
    result = ExportQualityGate.check(_audio(), 48000, reference_audio=_audio())
    assert result.passed is True
    assert result.muq_mos_degraded is False
