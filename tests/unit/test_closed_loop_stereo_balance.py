#!/usr/bin/env python3
"""Regelkreis-Sichtbarkeit der Stereo-Balance (§III.15, Befund 2026-11-18).

Befund: `measure_phase_quality_delta` ist MONO — der Closed-Loop-Kalibrator
sah Balance-Änderungen nicht, verfiel die Strength (0,69→0,05, „decision=skip“)
und kalibrierte die Stereo-Reparatur weg. Diese Tests pinnen, dass
Balance-Gewinne im Regelkreis-Delta sichtbar sind und No-Ops Null bleiben.
"""

from __future__ import annotations

import numpy as np

from backend.core.closed_loop_calibrator import _stereo_balance_delta


def _signal(imbalance_db: float, n: int = 48000) -> np.ndarray:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((n, 2)) * 0.05
    x[:, 0] *= 10 ** (imbalance_db / 20.0)
    return x


def test_balance_gewinn_ueber_strong_improvement() -> None:
    """3,5 dB Balance-Gewinn MUSS über STRONG_IMPROVEMENT (0,08) liegen."""
    vorher = _signal(-4.0)
    nachher = _signal(-0.5)
    delta = _stereo_balance_delta(vorher, nachher)
    assert delta > 0.08, f"Balance-Gewinn unsichtbar: delta={delta:.4f}"


def test_noop_bleibt_null() -> None:
    x = _signal(-3.0)
    assert _stereo_balance_delta(x, x.copy()) == 0.0


def test_verschlechterung_kein_positives_delta() -> None:
    vorher = _signal(-1.0)
    nachher = _signal(-4.0)  # Imbalance gewachsen
    assert _stereo_balance_delta(vorher, nachher) == 0.0  # kein Bonus


def test_mono_liefert_null() -> None:
    mono = np.zeros((48000,), dtype=np.float32)
    assert _stereo_balance_delta(mono, mono) == 0.0
