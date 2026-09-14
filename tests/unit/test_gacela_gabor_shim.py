"""Unit-Tests für scripts/gacela_gabor_shim.py (F2-Entblockung 2026-09-14).

Eigene NumPy-DGT (Truncated-Gaussian, 1024/256) — geprüft: Fenstergeometrie,
Betragsspektrogramm-Form/Energie, Log-Spektrogramm-Bereich, Preprocessing-
Länge, sys.modules-Shim (Upstream-Import läuft ohne ltfatpy).
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location("gacela_gabor_shim", "scripts/gacela_gabor_shim.py")
shim = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(shim)


def test_window_geometry() -> None:
    g = shim.truncated_gauss_window(256, 1024)
    assert g.shape == (2048,)
    assert np.all(np.isfinite(g))
    assert abs(float(np.linalg.norm(g)) - 1.0) < 1e-6
    assert np.argmax(g) == 0  # FFT-zentriert: Maximum bei Index 0 (x=0)
    assert g[1024] < 1e-3 and g[1023] < 1e-3  # fernste Samples praktisch Null


def test_dgt_real_mag_shape_and_energy() -> None:
    g = shim.truncated_gauss_window(256, 1024)
    rng = np.random.RandomState(0)
    x = rng.randn(2048 * 8).astype(np.float64) * 0.1
    mag = shim.dgt_real_mag(x, g, 256, 1024)
    assert mag.shape == (513, 64)
    assert np.all(np.isfinite(mag))
    assert float(np.max(mag)) > 0.0
    # Energie-Plausibilität: Summe der Beträge skaliert mit der Signalenergie.
    assert 1e-2 < float(np.mean(mag)) < 1e2


def test_spectrogram_normalized() -> None:
    obj = shim.GaussTruncTFShim(256, 1024)
    rng = np.random.RandomState(1)
    x = rng.randn(2048 * 8).astype(np.float64)
    spec = obj.spectrogram(x)
    assert np.all(np.isfinite(spec))
    assert spec.shape[0] == 513
    assert abs(float(np.max(spec)) - 1.0) < 1e-6


def test_log_spectrogram_range() -> None:
    spec = np.abs(np.random.RandomState(2).randn(513, 32)).astype(np.float64)
    log_spec = shim.log_spectrogram(spec, dynamic_range_dB=50.0)
    peak = float(np.max(spec))
    assert log_spec.shape == spec.shape
    assert float(np.max(log_spec)) == pytest.approx(10.0 * np.log10(peak), abs=1e-6)
    assert float(np.min(log_spec)) == pytest.approx(10.0 * np.log10(peak) - 50.0, abs=1e-6)


def test_preprocess_signal_length(monkeypatch) -> None:
    import librosa

    monkeypatch.setattr(librosa.effects, "trim", lambda y: (y[:800], None))
    y = np.zeros(1234, dtype=np.float64)
    out = shim.preprocess_signal(y, m=1024)
    assert len(out) % 1024 == 0
    assert len(out) >= 800


def test_install_shim_makes_upstream_import_work(tmp_path, monkeypatch) -> None:
    """Nach dem Shim importiert data.audioLoader ohne ltfatpy."""
    import importlib
    from pathlib import Path

    shim.install_tifresi_shim()
    assert "tifresi" in sys.modules
    assert "tifresi.stft" in sys.modules
    assert "tifresi.transforms" in sys.modules
    assert "tifresi.utils" in sys.modules

    data_dir = Path("models/gacela_upstream/data")
    assert data_dir.is_dir()
    monkeypatch.setattr(sys, "path", [str(data_dir.parent), str(data_dir)] + sys.path)
    try:
        importlib.import_module("audioLoader")
        importlib.import_module("baseDataset")
        importlib.import_module("trainDataset")
    except ImportError as exc:
        if "tifresi" in str(exc) or "ltfat" in str(exc):
            raise AssertionError(f"Shim hat den Upstream-Import nicht entblockt: {exc}") from exc
        raise
