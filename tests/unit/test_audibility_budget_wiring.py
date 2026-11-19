"""Verdrahtungs-Vertragstests für das Audibility-Budget (Hörordnung §4).

Verankert:
- Phase 43 (ML-De-Esser): Flag AUS ⇒ Budget wird NICHT konsultiert, Output
  bleibt bit-identisch; Flag AN ⇒ Boden begrenzt die Dämpfung, hebt nur an,
  nie über 1.0.
- Phase 03 (Denoise): Flag AUS ⇒ ``applied: False`` im Metadaten-Kanal;
  Flag AN ⇒ Budget wird je Zone angewendet und bleibt NaN-frei.
- Quell-Pins: die Budget-Aufrufe liegen im Produktionspfad (kein stilles
  Kapital, §G9 (copilot-instructions.md)).
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core import music_model_flags
from backend.core.dsp import audibility_budget as budget
from backend.core.phases import phase_43_ml_deesser as p43

SR = 48000


def _stats() -> budget.BudgetStats:
    return budget.BudgetStats(True, 12.0, 12.0, 1.0, 0.0, 0.0)


@pytest.fixture
def sib_signal() -> np.ndarray:
    """Programm (220/440 Hz) + lauter 6,5-kHz-Sibilant in der Mitte."""
    t = np.arange(SR) / SR
    programme = 0.15 * np.sin(2 * np.pi * 220.0 * t) + 0.05 * np.sin(2 * np.pi * 440.0 * t)
    sibilant = np.zeros_like(t)
    start, end = int(0.45 * SR), int(0.55 * SR)
    sibilant[start:end] = 0.80 * np.sin(2 * np.pi * 6500.0 * t[start:end])
    return (programme + sibilant).astype(np.float64)


def _deess(x: np.ndarray) -> tuple[np.ndarray, float]:
    return p43._deess_channel(
        x,
        SR,
        threshold_db=-18.0,
        ratio=4.0,
        attack_ms=1.0,
        release_ms=20.0,
        freq_low=5000.0,
        freq_high=8000.0,
        strength_cap=1.0,
    )


def test_flag_off_does_not_consult_budget_and_is_deterministic(monkeypatch, sib_signal):
    monkeypatch.setattr(music_model_flags, "use_audibility_budget", False)
    calls = {"n": 0}

    def _spy(*_args, **_kwargs):
        calls["n"] += 1
        return np.ones(sib_signal.size), _stats()

    monkeypatch.setattr(budget, "band_budget_curve", _spy)
    first, first_gr = _deess(sib_signal)
    second, second_gr = _deess(sib_signal)
    assert calls["n"] == 0
    assert np.array_equal(first, second)
    assert first_gr == pytest.approx(second_gr)


def test_unity_floor_makes_deesser_transparent(monkeypatch, sib_signal):
    monkeypatch.setattr(music_model_flags, "use_audibility_budget", True)
    monkeypatch.setattr(budget, "band_budget_curve", lambda _x, _sr, _band: (np.ones(sib_signal.size), _stats()))
    output, output_gr = _deess(sib_signal)
    # Boden 1.0 ⇒ keine Dämpfung; die Band-Rekombination ist rundungsbedingt
    # nicht bit-identisch, aber im Bereich < 1e-6 (keine hörbare Wirkung).
    assert output_gr == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(output, sib_signal, rtol=0.0, atol=1e-6)


def test_floor_caps_attenuation_of_audible_sibilant(monkeypatch, sib_signal):
    monkeypatch.setattr(music_model_flags, "use_audibility_budget", False)
    reference, reference_gr = _deess(sib_signal)

    monkeypatch.setattr(music_model_flags, "use_audibility_budget", True)
    monkeypatch.setattr(budget, "band_budget_curve", lambda _x, _sr, _band: (np.full(sib_signal.size, 0.5), _stats()))
    output, output_gr = _deess(sib_signal)

    burst = slice(int(0.45 * SR), int(0.55 * SR))
    reference_rms = float(np.sqrt(np.mean(reference[burst] ** 2)))
    output_rms = float(np.sqrt(np.mean(output[burst] ** 2)))
    assert output_rms > reference_rms
    assert output_gr > reference_gr
    assert np.all(np.abs(output) <= 1.5)


def test_real_budget_curve_runs_and_stays_finite(monkeypatch, sib_signal):
    monkeypatch.setattr(music_model_flags, "use_audibility_budget", True)
    output, gr_db = _deess(sib_signal)
    assert output.shape == sib_signal.shape
    assert np.isfinite(output).all()
    assert np.isfinite(gr_db)


# ---------------------------------------------------------------------------
# Phase 03 — Denoise (MRSA/OMLSA): Metadaten-Kanal + Wirkung
# ---------------------------------------------------------------------------
def _phase_03_params():
    from backend.core.phases.phase_03_denoise import DenoisePhase

    phase = DenoisePhase(sample_rate=SR)
    params = dict(phase.MATERIAL_PARAMS["vinyl"])
    params["strength"] = 0.7
    return phase, params


def _noisy_mono(n: int = SR // 2) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(n) / SR
    return (0.2 * np.sin(2 * np.pi * 440.0 * t) + 0.02 * rng.standard_normal(n)).astype(np.float32)


def test_phase03_flag_off_reports_not_applied(monkeypatch):
    monkeypatch.setattr(music_model_flags, "use_audibility_budget", False)
    phase, params = _phase_03_params()
    output, stats = phase._denoise_mono_professional(_noisy_mono(), params, None, None)
    assert np.isfinite(output).all()
    assert stats["audibility_budget"] == {"applied": False}


def test_phase03_flag_on_applies_budget_and_stays_finite(monkeypatch):
    monkeypatch.setattr(music_model_flags, "use_audibility_budget", True)
    phase, params = _phase_03_params()
    output, stats = phase._denoise_mono_professional(_noisy_mono(), params, None, None)
    report = stats["audibility_budget"]
    assert report["applied"] is True
    assert report["zones"] >= 1
    assert np.isfinite(output).all()
    assert np.all(np.abs(output) <= 1.0)


# ---------------------------------------------------------------------------
# Quell-Pins: die Aufrufe liegen wirklich im Produktionspfad (§G9 (copilot-instructions.md))
# ---------------------------------------------------------------------------
def test_phase43_source_pin():
    import inspect

    source = inspect.getsource(p43._deess_channel)
    assert "audibility_budget_enabled" in source
    assert "band_budget_curve" in source
    assert "np.maximum(gr_smooth" in source


def test_phase03_source_pin():
    import inspect

    from backend.core.phases.phase_03_denoise import DenoisePhase

    source = inspect.getsource(DenoisePhase._denoise_mono_mrsa)
    assert "_budget_on" in source
    assert "spectrum_budget_db" in source
    assert "apply_gain_budget" in source
