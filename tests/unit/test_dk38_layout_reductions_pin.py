"""D-K3-8-Vertragstests: Layout-Sicherheit der geprüften Top-Module.

Pinnt den Einzelprüfungs-Befund (2026-10-25): die L2-Treffer in den drei
Top-Modulen sind **False Positives** — die `audio[:, 0]`-/`mean(axis=0)`-
Stellen liegen in expliziten Layout-Branches bzw. hinter einer
dokumentierten channels-first-Konvention. Diese Tests lassen fallen, wenn
jemand die Branches so umbaut, dass die Layout-Invarianz bricht
(Lehre D-K3-20: keine `try/except: pass`-Hüllen).
"""

from __future__ import annotations

import numpy as np

from backend.core.stereo_authenticity_invariant import (
    _compute_lr_cross_correlation,
    _compute_ms_correlation,
    _compute_phantom_center_azimuth_deg,
)

SR = 48000


def _stereo_cf(n: int = 4000, seed: int = 42) -> np.ndarray:
    """Deterministisches Stereo-Signal channels-first (2, N)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / SR
    left = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(n)
    right = 0.45 * np.sin(2 * np.pi * 440 * t + 0.3) + 0.05 * rng.standard_normal(n)
    return np.stack([left, right]).astype(np.float32)


def test_ms_correlation_layout_invariant() -> None:
    """(2, N) und (N, 2) müssen identische MS-Korrelation liefern (§11 Spec 02)."""
    cf = _stereo_cf()
    sf = cf.T.copy()
    assert np.isclose(_compute_ms_correlation(cf), _compute_ms_correlation(sf), atol=1e-6)


def test_lr_cross_correlation_layout_invariant() -> None:
    cf = _stereo_cf()
    sf = cf.T.copy()
    assert np.isclose(_compute_lr_cross_correlation(cf), _compute_lr_cross_correlation(sf), atol=1e-6)


def test_phantom_center_azimuth_layout_invariant() -> None:
    cf = _stereo_cf()
    sf = cf.T.copy()
    assert np.isclose(
        _compute_phantom_center_azimuth_deg(cf),
        _compute_phantom_center_azimuth_deg(sf),
        atol=1e-6,
    )


def test_mono_input_does_not_crash() -> None:
    mono = np.sin(2 * np.pi * 440 * np.arange(2000) / SR).astype(np.float32)
    assert np.isfinite(_compute_ms_correlation(mono))
    assert np.isfinite(_compute_lr_cross_correlation(mono))
    assert np.isfinite(_compute_phantom_center_azimuth_deg(mono))


def test_benchmark_stereo_channels_first_contract() -> None:
    """benchmark_stereo rechnet channels-first — Pin der Lade-Konvention."""
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent.parent / "scripts" / "benchmark_stereo.py"
    spec = importlib.util.spec_from_file_location("benchmark_stereo", str(script))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["benchmark_stereo"] = mod  # Dataclass-Dekorator braucht den Eintrag
    spec.loader.exec_module(mod)

    corr_cf = float(mod._channel_correlation(_stereo_cf()))
    # Konvention verletzt (samples-first) ⇒ Funktion würde Kanäle falsch lesen:
    corr_sf = float(mod._channel_correlation(_stereo_cf().T.copy()))
    # Die Werte DÜRFEN sich unterscheiden — gepinnt wird nur: kein Absturz,
    # endliche Werte, und der channels-first-Wert ist der kanonische (≈ hoher
    # Gleichklang der beiden fast identischen Kanäle).
    assert np.isfinite(corr_cf)
    assert np.isfinite(corr_sf)
    assert corr_cf > 0.9
