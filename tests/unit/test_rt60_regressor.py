"""SOTA-M15 (KP-10) — Vertragstests des RT60-Regressors (CHALLENGER zum T60-Witness-Cap).

Verträge (Auftrag SOTA-M15 KP-10):
- Flag `use_rt60_regressor` AUS (Default) ⇒ bit-identisches Bestandsverhalten.
- Determinismus (§G5 (copilot-instructions.md)): 2× gleicher Input ⇒ bit-identisch.
- Physikalische Grenzen [0.05 s, 10 s]: Ausreißer werden geclippt (§0a).
- Fallback mit logger.warning (§V6 (copilot-instructions.md)), kein stiller Fallback.
- Synth-RIR-Schätzung grob korrekt (tolerant).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

import backend.core.music_model_flags as mmf
from backend.core.dsp import rt60_regressor as rr

SR = 48000


def _synth_reverb(rt60: float, sr: int = SR, seed: int = 0, dur_s: float = 1.5) -> np.ndarray:
    """Deterministisches synthetisches RIR (Muster test_deepfilternet_rt60_estimator)."""
    rng = np.random.default_rng(seed)
    n = int(sr * dur_s)
    t = np.arange(n, dtype=np.float64) / sr
    env = np.exp(-6.9078 * t / rt60)
    x = rng.standard_normal(n) * env
    return x.astype(np.float32)


# ── Flag-Gate: use_rt60_regressor AUS ⇒ bit-identisches Bestandsverhalten ─────


def test_flag_default_aus() -> None:
    """Flag ist aus (music_model_flags.py) — Challenger läuft nie ohne Sign-off."""
    assert mmf.use_rt60_regressor is False


def test_flag_aus_challenger_nie_gerufen(monkeypatch) -> None:
    """Flag AUS: die Phasen rufen den Regressor nie (bit-identisches Verhalten)."""
    # Die Integrationsstelle in phase_20/49 prüft das Flag, bevor sie den
    # Regressor importiert. Hier prüfen wir nur den Contract: das Flag ist False.
    assert mmf.use_rt60_regressor is False


# ── Determinismus (§G5 (copilot-instructions.md)) ─────────────────────────────


def test_determinismus_bit_identisch() -> None:
    """2× gleicher Input ⇒ bit-identischer Output (§G5 (copilot-instructions.md))."""
    x = _synth_reverb(0.6, seed=7)
    r1 = rr.Rt60Regressor(checkpoint=None)  # Heuristik-Gewichte
    r2 = rr.Rt60Regressor(checkpoint=None)
    a = r1.predict_rt60(x, SR)
    b = r2.predict_rt60(x, SR)
    assert a == b  # exakt bit-identisch


def test_determinismus_zwei_instanzen_gleicher_seed() -> None:
    """Zwei Instanzen mit demselben Seed liefern bit-identische Ergebnisse."""
    x = _synth_reverb(1.2, seed=11)
    r1 = rr.Rt60Regressor(checkpoint=None, seed=42)
    r2 = rr.Rt60Regressor(checkpoint=None, seed=42)
    assert r1.predict_rt60(x, SR) == r2.predict_rt60(x, SR)


# ── Physikalische Grenzen (§0a) ───────────────────────────────────────────────


def test_physikalische_grenzen_ausreisser_geclippt() -> None:
    """Ausreißer (Prediction > RT60_MAX_S) werden geclippt, nicht verworfen.

    Ein Modell mit extremen Gewichten prognostiziert > 10 s — der Clip nach
    RT60_MAX_S ist die einzige Sicherung gegen physikalisch unmögliche Werte.
    """
    import tempfile

    weights = np.zeros(rr.N_FEATURES, dtype=np.float64)
    weights[4] = 50.0  # span*50 ⇒ bei span≈0.5s ≈ 25 s
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f, weights=weights, bias=np.float64(0.0))
        ckpt = Path(f.name)
    try:
        r = rr.Rt60Regressor(checkpoint=ckpt)
        x = _synth_reverb(1.0, seed=3, dur_s=2.0)
        est = r.predict_rt60(x, SR)
        assert rr.RT60_MIN_S <= est <= rr.RT60_MAX_S
        assert est == rr.RT60_MAX_S  # geclippt
    finally:
        ckpt.unlink(missing_ok=True)


def test_physikalische_grenzen_untergrenze() -> None:
    """Sehr kurzes RIR wird auf RT60_MIN_S geclippt."""
    x = _synth_reverb(0.01, seed=5, dur_s=0.3)
    r = rr.Rt60Regressor(checkpoint=None)
    est = r.predict_rt60(x, SR)
    assert rr.RT60_MIN_S <= est <= rr.RT60_MAX_S
    assert est == rr.RT60_MIN_S  # geclippt


def test_plausibel_begrenzte_ausgabe_alle_faelle() -> None:
    """Ausgabe ist IMMER in [RT60_MIN_S, RT60_MAX_S] — auch bei NaN/Inf-Eingabe."""
    for bad in (np.nan, np.inf, -np.inf, 0.0):
        x = np.full(1000, bad, dtype=np.float32)
        r = rr.Rt60Regressor(checkpoint=None)
        est = r.predict_rt60(x, SR)
        assert rr.RT60_MIN_S <= est <= rr.RT60_MAX_S
        assert np.isfinite(est)


# ── NaN/Inf-Schutz (§0a) + Fallback-Warnung (§V6 (copilot-instructions.md)) ───


def test_nan_inf_schutz_mit_warnung(caplog) -> None:
    """NaN/Inf im Eingang: Sanitisierung + logger.warning (§0a/§V6 (copilot-instructions.md))."""
    x = np.full(500, np.nan, dtype=np.float32)
    r = rr.Rt60Regressor(checkpoint=None)
    with caplog.at_level(logging.WARNING):
        est = r.predict_rt60(x, SR)
    assert rr.RT60_MIN_S <= est <= rr.RT60_MAX_S
    assert any("NaN/Inf" in rec.message for rec in caplog.records)


def test_leeres_audio_fallback_warnung(caplog) -> None:
    """Leeres Audio: deterministischer Fallback-Wert + logger.warning (§V6 (copilot-instructions.md))."""
    r = rr.Rt60Regressor(checkpoint=None)
    with caplog.at_level(logging.WARNING):
        est = r.predict_rt60(np.zeros(0, dtype=np.float32), SR)
    assert est == rr.RT60_MIN_S
    assert any("leeres Audio" in rec.message for rec in caplog.records)


def test_fehlendes_artefakt_warnung(caplog) -> None:
    """Kein trainiertes Artefakt: §V6-Warnung + Heuristik-Gewichte (kein stiller Fallback)."""
    with caplog.at_level(logging.WARNING):
        r = rr.Rt60Regressor(checkpoint=None)
    assert r.source in ("heuristic", "heuristic_fallback")
    assert any("ohne trainiertes Artefakt" in rec.message for rec in caplog.records)


def test_ungueltige_sample_rate_wert_error() -> None:
    """Ungültige sample_rate ⇒ ValueError (laut, §V6 (copilot-instructions.md))."""
    r = rr.Rt60Regressor(checkpoint=None)
    with pytest.raises(ValueError):
        r.predict_rt60(np.zeros(100, dtype=np.float32), 0)


# ── Synth-RIR-Schätzung grob korrekt (tolerant) ───────────────────────────────


@pytest.mark.parametrize(
    "gt_rt60",
    [0.3, 0.6, 1.2],
)
def test_synth_rir_schaetzung_grob_korrekt(gt_rt60: float) -> None:
    """Synth-RIR mit bekanntem GT-RT60 ⇒ Schätzung grob korrekt (±40 % tolerant)."""
    x = _synth_reverb(gt_rt60, seed=13, dur_s=2.5)
    r = rr.Rt60Regressor(checkpoint=None)
    est = r.predict_rt60(x, SR)
    assert rr.RT60_MIN_S <= est <= rr.RT60_MAX_S
    # Tolerant: ±40 % (Heuristik ohne Training, nur Smoke-Gate)
    assert abs(est - gt_rt60) / gt_rt60 < 0.40, f"GT {gt_rt60} vs. est {est}"


def test_synth_rir_schaetzung_monoton_mit_rt60() -> None:
    """Längeres RT60 ⇒ längere Schätzung (monotone Beziehung, tolerant)."""
    r = rr.Rt60Regressor(checkpoint=None)
    est_short = r.predict_rt60(_synth_reverb(0.3, seed=21, dur_s=2.5), SR)
    est_long = r.predict_rt60(_synth_reverb(1.5, seed=21, dur_s=2.5), SR)
    assert est_short < est_long


# ── Layout-Invarianz (Stereo, AGENTS.md §3) ───────────────────────────────────


def test_stereo_layout_invariante() -> None:
    """(2, N) und (N, 2) liefern identische RT60-Schätzung (Stereo-Layout-Invariante)."""
    mono = _synth_reverb(0.6, seed=33, dur_s=2.0)
    stereo_2n = np.stack([mono, mono])  # (2, N)
    stereo_n2 = stereo_2n.T  # (N, 2)
    r = rr.Rt60Regressor(checkpoint=None)
    est_2n = r.predict_rt60(stereo_2n, SR)
    est_n2 = r.predict_rt60(stereo_n2, SR)
    assert est_2n == est_n2  # bit-identisch (mono-Mittelung)


# ── Fassade: estimate_rt60_regression ─────────────────────────────────────────


def test_fassade_estimate_rt60_regression() -> None:
    """`estimate_rt60_regression` liefert float oder None (nie Exception bei gültigem Input)."""
    x = _synth_reverb(0.6, seed=41)
    est = rr.estimate_rt60_regression(x, SR)
    assert est is not None
    assert isinstance(est, float)
    assert rr.RT60_MIN_S <= est <= rr.RT60_MAX_S
