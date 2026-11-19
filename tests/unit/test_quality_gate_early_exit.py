from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.unit
def test_check_dsp_skips_audio_probe_when_musical_goals_already_fail(monkeypatch):
    from backend.core.quality_gate import QualityGate

    gate = QualityGate()
    audio_probe_called = False

    def _fail_if_called(_audio, _context):
        nonlocal audio_probe_called
        audio_probe_called = True
        raise AssertionError("_check_audio_array should not run after musical-goal failure")

    monkeypatch.setattr(gate, "_check_audio_array", _fail_if_called)

    result = SimpleNamespace(
        audio=np.ones(4096, dtype=np.float32),
        musical_goals={"natuerlichkeit": 0.10},
    )

    assert gate.check_dsp(result) is False
    assert audio_probe_called is False


def test_check_ml_skips_audio_probe_when_musical_goals_already_fail(monkeypatch):
    from backend.core.quality_gate import QualityGate

    gate = QualityGate()
    audio_probe_called = False

    def _fail_if_called(_audio, _context):
        nonlocal audio_probe_called
        audio_probe_called = True
        raise AssertionError("_check_audio_array should not run after musical-goal failure")

    monkeypatch.setattr(gate, "_check_audio_array", _fail_if_called)

    result = SimpleNamespace(
        audio=np.ones(4096, dtype=np.float32),
        musical_goals={"natuerlichkeit": 0.10},
        authenticity_score=0.99,
    )

    assert gate.check_ml(result) is False
    assert audio_probe_called is False
