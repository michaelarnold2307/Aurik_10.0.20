"""Tests für backend/core/dsp/interaural_cues.py (§HRTF/interaural, SOTA-Basis)."""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.dsp.interaural_cues import (
    InterauralIntegrityResult,
    apply_interaural_cues,
    bmld_advantage_db,
    compute_iacc,
    compute_ild_db,
    compute_itd_us,
    head_shadow_ild_db,
    interaural_cue_integrity,
    woodworth_itd_us,
)


def _make_stereo(
    sr: int = 48000, n: int = 48000, itd_us: float = 0.0, ild_db: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetisches Stereo (2, N): Rauschen mit kontrolliertem ITD/ILD."""
    rng = np.random.default_rng(42)
    left = rng.standard_normal(n).astype(np.float32)
    right = np.roll(left, int(round(itd_us * 1e-6 * sr))) * 10.0 ** (ild_db / 20.0)
    # Gemeinsame Normalisierung — getrennte würde die ILD auslöschen.
    peak = max(np.max(np.abs(left)), np.max(np.abs(right))) + 1e-9
    left = left / peak
    right = right / peak
    return np.stack([left, right], axis=0), np.stack([left, right], axis=0)


def test_woodworth_itd_symmetry() -> None:
    # Geradeaus: 0 µs; 90°: maximal; symmetrisch um 0°.
    assert woodworth_itd_us(0.0) == pytest.approx(0.0, abs=1e-6)
    az = 90.0
    assert woodworth_itd_us(az) == pytest.approx(-woodworth_itd_us(-az), abs=1e-6)
    assert woodworth_itd_us(az) > 600.0  # ~660 µs @ r=8.75 cm


def test_head_shadow_ild_sign() -> None:
    # Kontralaterales Ohr (negative Azimut-Perspektive) → Dämpfung (negativ in dB).
    ild = head_shadow_ild_db(azimuth_deg=90.0, freq_hz=3000.0)
    assert ild < -1.0
    assert head_shadow_ild_db(0.0, 1000.0) == pytest.approx(0.0, abs=0.05)


def test_compute_itd_ild_iacc_recovery() -> None:
    sr = 48000
    audio, _ = _make_stereo(sr, itd_us=200.0, ild_db=-3.0)
    itd = compute_itd_us(audio, sr)
    # right = np.roll(left, +200 µs) → rechts eilt nach → ITD negativ (Konvention: Ankunft links − rechts).
    assert itd == pytest.approx(-200.0, abs=35.0)
    ild = compute_ild_db(audio, sr)
    assert ild == pytest.approx(3.0, abs=1.5)  # |L| - |R| in dB
    iacc = compute_iacc(audio, sr)
    assert 0.9 <= iacc <= 1.0


def test_bmld_monotonic() -> None:
    # BMLD-Vorteil: interaural unkorrelierte Referenz (IACC niedrig) → Vorteil größer.
    assert bmld_advantage_db(iacc_ref=0.2) > bmld_advantage_db(iacc_ref=0.98)


def test_interaural_cue_integrity_detects_drift() -> None:
    sr = 48000
    orig, _ = _make_stereo(sr, itd_us=50.0, ild_db=-1.0)
    # Restaurierung verschiebt ITD +100 µs und verstärkt ILD um 3 dB.
    corrupted = np.stack([np.roll(orig[0], int(round(100e-6 * sr))), orig[1] * 10.0 ** (-3.0 / 20.0)], axis=0)
    res = interaural_cue_integrity(orig, corrupted, sr)
    assert isinstance(res, InterauralIntegrityResult)
    assert res.itd_drift_us > 60.0
    assert res.ild_drift_db > 2.0
    assert not res.itd_ok and not res.ild_ok


def test_interaural_cue_integrity_passthrough_clean() -> None:
    sr = 48000
    orig, _ = _make_stereo(sr, itd_us=100.0, ild_db=-1.0)
    res = interaural_cue_integrity(orig, orig.copy(), sr)
    assert res.itd_drift_us < 5.0
    assert res.ild_drift_db < 0.5
    assert res.itd_ok and res.ild_ok


def test_apply_interaural_cues_places_cues() -> None:
    sr = 48000
    rng = np.random.default_rng(7)
    mono = (rng.standard_normal(sr) / 10.0).astype(np.float32)
    stereo = apply_interaural_cues(mono, sr, azimuth_deg=60.0, distance_m=1.5)
    assert stereo.shape == (2, len(mono))
    itd = compute_itd_us(stereo, sr)
    # ITD-Richtung: bei positivem Azimut (rechts) eilt der rechte Kanal voraus.
    assert itd > 100.0
    # Energie des entfernten Signals ist kleiner als die des nahen Mono-Signals.
    assert float(np.sqrt(np.mean(stereo**2))) < float(np.sqrt(np.mean(mono**2)))
