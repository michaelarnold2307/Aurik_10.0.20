from __future__ import annotations

import pytest

"""Unit-Tests für adaptive Export-Gate-Profile in cli/aurik_cli.py."""


import math
from types import SimpleNamespace

import numpy as np

from cli.aurik_cli import (
    _compute_export_gate_adjustment,
    _compute_export_signal_signature,
)

SR = 48_000


def _sine(duration_s: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(SR * duration_s), endpoint=False, dtype=np.float32)
    return np.asarray(np.sin(2.0 * np.pi * freq * t), dtype=np.float32)


@pytest.mark.unit
def test_export_signal_signature_returns_finite_metrics() -> None:
    sig = _compute_export_signal_signature(_sine(1.0), SR)
    for key in ("crest_db", "hf_ratio", "transient_ratio", "micro_dynamic_db"):
        assert key in sig
        assert math.isfinite(float(sig[key]))


def test_export_gate_adjustment_relaxes_for_fragile_risk() -> None:
    result = SimpleNamespace(
        stage_notes={"exzellenz_recovery_profile": {"preserve_signal": 0.6}},
    )
    sig = {
        "crest_db": 22.0,
        "hf_ratio": 0.01,
        "transient_ratio": 0.02,
        "micro_dynamic_db": 18.0,
    }
    qe_delta, pegel_delta, reason = _compute_export_gate_adjustment("tape", sig, result)
    assert qe_delta < 0.0
    assert pegel_delta > 0.0
    assert "risk=" in reason


def test_export_gate_adjustment_tightens_for_modern_stable() -> None:
    result = SimpleNamespace(stage_notes={})
    sig = {
        "crest_db": 10.0,
        "hf_ratio": 0.08,
        "transient_ratio": 0.002,
        "micro_dynamic_db": 8.0,
    }
    qe_delta, pegel_delta, reason = _compute_export_gate_adjustment("cd_digital", sig, result)
    assert qe_delta > 0.0
    assert pegel_delta < 0.0
    assert "modern_stable" in reason


def test_export_gate_adjustment_neutral_for_mid_risk() -> None:
    result = SimpleNamespace(stage_notes={"exzellenz_recovery_profile": {"preserve_signal": 0.1}})
    sig = {
        "crest_db": 13.0,
        "hf_ratio": 0.05,
        "transient_ratio": 0.004,
        "micro_dynamic_db": 10.0,
    }
    qe_delta, pegel_delta, reason = _compute_export_gate_adjustment("vinyl", sig, result)
    assert qe_delta == 0.0
    assert pegel_delta == 0.0
    assert "neutral" in reason
