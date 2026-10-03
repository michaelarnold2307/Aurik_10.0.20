"""Tests für backend/core/dsp/modulation_transfer.py — MTF/STI-Zweitstimme.

Verträge: perfekter Transfer ⇒ sti ≈ 1; zerstörte Modulation ⇒ sti klein;
reine Pegeländerung ist KEIN Verlust (Tiefe normiert); deterministisch und
NaN-sicher (§G5 (GEBOTE.md), §0a (copilot-instructions.md)).
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.dsp.modulation_transfer import measure_modulation_transfer

SR = 48000


def _am_signal(seconds: float = 4.0, rate_hz: float = 6.0, mod_depth: float = 0.8) -> np.ndarray:
    """Mehrband-AM-Testsignal (Träger 500/2000/5000 Hz, gemeinsame Silben-AM)."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    am = 1.0 + mod_depth * np.sin(2 * np.pi * rate_hz * t)
    x = np.zeros(n, dtype=np.float64)
    for fc, amp in ((500.0, 0.3), (2000.0, 0.3), (5000.0, 0.2)):
        x += amp * np.sin(2 * np.pi * fc * t) * am
    return x


def test_perfekter_transfer_gibt_sti_eins() -> None:
    sig = _am_signal()
    rep = measure_modulation_transfer(sig, sig.copy(), SR)
    assert rep["sti"] == pytest.approx(1.0, abs=1e-3)
    assert rep["consonant_modulation"] == pytest.approx(1.0, abs=1e-3)


def test_zerstoerte_modulation_fallt_durch() -> None:
    """Ohne Modulationstiefe bleibt kein Verständlichkeitstransfer übrig."""
    sig = _am_signal()
    n = sig.size
    t = np.arange(n) / SR
    flat = np.zeros(n, dtype=np.float64)
    for fc, amp in ((500.0, 0.3), (2000.0, 0.3), (5000.0, 0.2)):
        flat += amp * np.sin(2 * np.pi * fc * t)  # gleiche Träger, AM entfernt
    rep = measure_modulation_transfer(sig, flat, SR)
    assert rep["sti"] < 0.35, f"AM-Verlust nicht sichtbar: {rep['sti']}"
    assert rep["consonant_modulation"] < 0.35


def test_reine_pegelaenderung_ist_kein_verlust() -> None:
    """Normierung auf Modulationstiefe: −6 dB Gain darf nicht als Fund gelten."""
    sig = _am_signal()
    rep = measure_modulation_transfer(sig, 0.5 * sig, SR)
    assert rep["sti"] == pytest.approx(1.0, abs=1e-3)
    assert rep["consonant_modulation"] == pytest.approx(1.0, abs=1e-3)


def test_deterministisch_und_nan_sicher() -> None:
    sig = _am_signal()
    dirty = sig.copy()
    dirty[100:110] = np.nan
    dirty[200] = np.inf
    r1 = measure_modulation_transfer(dirty, sig, SR)
    r2 = measure_modulation_transfer(dirty, sig, SR)
    assert r1 == r2
    assert np.isfinite(r1["sti"])


def test_zu_kurz_fail_closed() -> None:
    with pytest.raises(ValueError):
        measure_modulation_transfer(np.zeros(100), np.zeros(100), SR)
