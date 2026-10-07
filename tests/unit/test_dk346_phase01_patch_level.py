"""D-K3-46: ML-Click-Patch (phase_01) — Pegel-Erhalt + stufenfreie Flanken.

Produktionsbefund 2026-10-07 21:55: TemporalConsistencyGuard meldete
`4 Energie-Sprünge >6.0dB zwischen 100ms-Fenstern` in phase_01_click_removal
(60-s-Vinyl-Lauf, BANQUET-ML-Konsens: 130 ML-Klick-Regionen, rs≈70, era=1970).

Wurzel: `_repair_click_patch_ml` ersetzte ganze ML-Regionen mit dem
DFN-Rohpegel (systematisch leiser — D-K3-45: −5,5 dB; Bias-Pfad bis −9 dB)
OHNE Pegel-Angleich und blendete nur mit einer auf max. 8 Samples begrenzten
LINEAREN Rampe (0,17 ms @48k; entfiel bei len < 8 ganz — Muster §2.35b).
Lange Regionen kippten damit 100-ms-Fenster an beiden Flanken um >6 dB.

Fix: `level_match` (audio_utils, eine Quelle §G9 (copilot-instructions.md)) + Cosinus-Zügelung
(2…64 Samples). Gepinnt: Region-Pegel-Drift < 1 dB, Guard ohne Sprünge,
kurze Ersetzungen behalten eine gezügelte Flanke.
"""

from __future__ import annotations

import numpy as np

from backend.core.phases.phase_01_click_removal import ClickRemovalPhase
from backend.core.temporal_consistency_guard import TemporalConsistencyGuard

SR = 48_000


class _MockDFN:
    """DFN-Modell-Mock: liefert systematisch leiser (Rohpegel wie Messbefund)."""

    def __init__(self, gain_db: float = -9.0) -> None:
        self._gain = float(10.0 ** (gain_db / 20.0))

    def enhance(self, patch: np.ndarray, sr: int, energy_bias_db: float = -6.0) -> np.ndarray:
        return (np.asarray(patch, dtype=np.float32) * self._gain).astype(np.float32)


def _music(n: int, seed: int = 42) -> np.ndarray:
    t = np.arange(n) / SR
    x = (
        0.30 * np.sin(2 * np.pi * 220.0 * t)
        + 0.15 * np.sin(2 * np.pi * 660.0 * t)
        + 0.05 * np.sin(2 * np.pi * 3000.0 * t)
    )
    rng = np.random.default_rng(seed)
    return (x + 0.008 * rng.standard_normal(n)).astype(np.float32)


def _phase_with_mock(gain_db: float) -> ClickRemovalPhase:
    p = ClickRemovalPhase()
    p._get_deepfilternet_plugin = lambda: _MockDFN(gain_db)  # type: ignore[method-assign]
    return p


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2)) + 1e-12)


def test_ml_patch_preserves_region_level() -> None:
    """150-ms-Region bei −9 dB Rohpegel: level_match hält den Pegel (< 1 dB)."""
    x = _music(4 * SR)
    y = x.copy()
    start = int(1.0 * SR)
    end = start + int(0.15 * SR) - 1
    ok = _phase_with_mock(-9.0)._repair_click_patch_ml(y, SR, {"start": start, "end": end}, panns_singing=0.0)
    assert ok is True, "ML-Patch wurde nicht angewandt (Fixture-Vertrag)"
    drift_db = 20.0 * np.log10(_rms(y[start : end + 1]) / _rms(x[start : end + 1]))
    assert abs(drift_db) < 1.0, f"Pegel-Drift {drift_db:+.2f} dB — level_match nicht wirksam (D-K3-46a)"


def test_ml_patch_leaves_no_temporal_jumps() -> None:
    """Nach dem Angleich misst der TemporalConsistencyGuard keinen Sprung."""
    x = _music(4 * SR)
    y = x.copy()
    start = int(1.0 * SR)
    end = start + int(0.15 * SR) - 1
    ok = _phase_with_mock(-9.0)._repair_click_patch_ml(y, SR, {"start": start, "end": end}, panns_singing=0.0)
    assert ok is True
    result = TemporalConsistencyGuard().check(x, y, "phase_01_click_removal", sr=SR, relative_to_median=True)
    assert result.energy_jumps == 0, f"{result.energy_jumps} Energie-Sprünge trotz Pegel-Angleich (D-K3-46a)"


def test_short_ml_patch_keeps_soft_flank() -> None:
    """1-Sample-Ersetzung (len=4): Flanke gezügelt — alt entfiel der Fade ganz."""
    x = _music(2 * SR)
    y = x.copy()
    start = int(0.5 * SR)
    ok = _phase_with_mock(-9.0)._repair_click_patch_ml(y, SR, {"start": start, "end": start}, panns_singing=0.0)
    assert ok is True
    rs = start - 1  # replace_start = start - min(8, duration=1)
    assert abs(float(y[rs]) - float(x[rs])) <= 1e-3 * max(abs(float(x[rs])), 1e-3), (
        "Ersatzflanke nicht gezügelt — Fade-Ausfall (alt: len//4=1 → `if fade >= 2` übersprang, D-K3-46b)"
    )
