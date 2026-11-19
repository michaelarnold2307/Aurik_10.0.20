"""Unit-Tests für scrape_flutter_rest (WF-CASS, §SOTA-WF-CASS)."""

from __future__ import annotations

import numpy as np

from backend.core.dsp.scrape_flutter_rest import (
    compensate_scrape_flutter,
    detect_scrape_flutter,
)

SR = 48000


def _am_signal(mod_hz: float, depth: float, seconds: float = 4.0, seed: int = 0) -> np.ndarray:
    """3-kHz-Träger (im Mess-Bandpass 1,5–8 kHz) mit Amplitudenmodulation."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    carrier = np.sin(2 * np.pi * 3000.0 * t)
    mod = 1.0 + depth * np.sin(2 * np.pi * mod_hz * t)
    return (carrier * mod + 0.001 * rng.standard_normal(len(t))).astype(np.float32)


def test_detect_finds_modulation():
    x = _am_signal(30.0, 0.4)  # starke 30-Hz-AM
    res = detect_scrape_flutter(x, SR)
    assert res.confidence > 0.5
    assert any(abs(f - 30.0) < 2.0 for f in res.mod_freqs)


def test_compensate_reduces_am():
    x = _am_signal(30.0, 0.4)
    res = detect_scrape_flutter(x, SR)
    out = compensate_scrape_flutter(x, SR, res)
    # Modulationsindex vorher/nachher via Hüllkurven-Varianz
    env = np.abs(__import__("scipy").signal.hilbert(x))
    env_out = np.abs(__import__("scipy").signal.hilbert(out))

    def rel_var(e: np.ndarray) -> float:
        return float(np.std(e) / (np.mean(e) + 1e-9))

    assert rel_var(env_out) < 0.6 * rel_var(env)
    assert out.shape == x.shape


def test_no_finding_passthrough():
    rng = np.random.default_rng(1)
    x = (0.1 * rng.standard_normal(SR * 2)).astype(np.float32)
    res = detect_scrape_flutter(x, SR)
    assert res.confidence < 0.5
    out = compensate_scrape_flutter(x, SR, res)
    assert np.array_equal(out, x)  # unverändert ohne Befund


def test_never_worsen_energy():
    """Energie bleibt nach Kompensation in engen Grenzen (±10 %)."""
    x = _am_signal(22.0, 0.3)
    res = detect_scrape_flutter(x, SR)
    out = compensate_scrape_flutter(x, SR, res)
    assert 0.9 <= float(np.mean(out**2) / np.mean(x**2)) <= 1.1


def test_determinism():
    x = _am_signal(40.0, 0.35)
    r1 = detect_scrape_flutter(x, SR)
    c1 = compensate_scrape_flutter(x, SR, r1)
    r2 = detect_scrape_flutter(x, SR)
    c2 = compensate_scrape_flutter(x, SR, r2)
    assert np.array_equal(c1, c2)
    assert r1.confidence == r2.confidence


def test_stereo_nx2_layout():
    rng = np.random.default_rng(2)
    x = np.stack([_am_signal(30.0, 0.3, seed=3), _am_signal(30.0, 0.3, seed=4)], axis=1)
    res = detect_scrape_flutter(x, SR)
    out = compensate_scrape_flutter(x, SR, res)
    assert out.shape == x.shape
