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


class TestPerfR12ProfileCache:
    """§PERF-R12 (2026-09-19): Content-keyed Interaural-Profil-Cache +
    batched Fenster-Preprocessing — numerisch äquivalent zum alten
    Per-Fenster-Loop, deterministisch nach §G5 (GEBOTE.md), Deckel 4."""

    def test_profile_cache_bit_identical_and_bounded(self) -> None:
        import backend.core.dsp.interaural_cues as ic

        ic._INTERAURAL_PROFILE_CACHE.clear()
        stereo, _ = _make_stereo(sr=48000, n=96000, itd_us=200.0, ild_db=-3.0)
        stereo = np.ascontiguousarray(stereo, dtype=np.float64)
        p1 = ic.compute_interaural_profile(stereo, 48000)
        p2 = ic.compute_interaural_profile(stereo, 48000)
        assert p1 is p2  # exakt derselbe Wert (bit-identische Semantik)
        assert len(ic._INTERAURAL_PROFILE_CACHE) == 1
        for i in range(6):
            ic.compute_interaural_profile(stereo + i * 1e-4, 48000)
        assert len(ic._INTERAURAL_PROFILE_CACHE) <= ic._INTERAURAL_PROFILE_CACHE_MAX
        assert len(ic._INTERAURAL_PROFILE_CACHE) > 0

    def test_profile_equivalent_to_reference_loop(self) -> None:
        import backend.core.dsp.interaural_cues as ic

        ic._INTERAURAL_PROFILE_CACHE.clear()
        sr = 48000
        stereo32, _ = _make_stereo(sr=sr, n=96000, itd_us=250.0, ild_db=2.0)
        stereo64 = np.ascontiguousarray(stereo32, dtype=np.float64)
        prof = ic.compute_interaural_profile(stereo64, sr)

        # Alte Per-Fenster-Schleife (Referenz-Replikation)
        ch = ic.to_channels_first(stereo64)
        _l, _r = ch[0], ch[1]
        win = max(int(round(0.05 * sr)), 256)
        hop = max(win // 2, 1)
        max_lag_samples = int(round(0.001 * sr))
        itds = []
        for start in range(0, max(len(_l) - win + 1, 1), hop):
            _w_l = _l[start : start + win]
            _w_r = _r[start : start + win]
            if float(np.std(_w_l)) < 1e-8 or float(np.std(_w_r)) < 1e-8:
                continue
            _lag, _ = ic._itd_from_cross_correlation(_w_l, _w_r, max_lag_samples)
            itds.append(_lag / sr * 1e6)
        ref_itd = float(np.median(itds))
        ref_jitter = float(np.median(np.abs(np.asarray(itds) - ref_itd)))
        assert abs(prof.itd_us - ref_itd) < 1e-6
        assert abs(prof.itd_jitter_us - ref_jitter) < 1e-6
