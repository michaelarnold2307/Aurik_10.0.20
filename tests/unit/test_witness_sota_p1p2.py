"""Unit-Tests für Maskierungsmodell + Rauigkeit (§Witness-SOTA P1/P2, 2026-09-12)
und P3 (Stereo-Kollaps) + P4 (Pre-Echo) — 2026-09-12.

Prüft: Determinismus, Ton-maskiert-Ton (lauter Masker → leises Delta inaudible),
klares Delta über der Schwelle (audible), Rauigkeit steigt bei 70-Hz-AM,
Stereo→Mono-Kollaps, Energie vor einem Transienten (Pre-Echo).
"""

from __future__ import annotations

import numpy as np

from backend.core.dsp.masking_model import band_audibility, bark_band_edges, compute_masking_threshold_db
from backend.core.dsp.roughness_model import compute_roughness_asper, roughness_rise_asper

SR = 48000


def _tone(freq: float, amp: float, dur_s: float = 1.0) -> np.ndarray:
    n = int(SR * dur_s)
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_bark_edges_deterministic() -> None:
    e1 = bark_band_edges(SR)
    e2 = bark_band_edges(SR)
    np.testing.assert_array_equal(e1, e2)
    assert e1[0] == 0.0
    assert e1[-1] <= SR / 2.0
    assert len(e1) > 20


def test_masking_threshold_deterministic() -> None:
    rng = np.random.default_rng(11)
    x = rng.standard_normal(SR).astype(np.float32) * 0.05
    t1, _ = compute_masking_threshold_db(x, SR)
    t2, _ = compute_masking_threshold_db(x, SR)
    np.testing.assert_allclose(t1, t2, rtol=0, atol=0)
    assert t1.shape[0] > 0
    assert np.isfinite(t1).all()


def test_loud_masker_hides_quiet_delta() -> None:
    # Lauter 1-kHz-Masker verdeckt ein leises 12-kHz-Delta (unter der Schwelle).
    masker = _tone(1000.0, 0.9, dur_s=1.0)
    base = masker + _tone(12000.0, 0.02, dur_s=1.0)
    boosted = masker + _tone(12000.0, 0.021, dur_s=1.0)  # +0.4 dB im Luftband
    res = band_audibility(base, boosted, SR, 8000.0, 16000.0)
    assert res["delta_db"] < res["threshold_db"]
    assert res["audible"] is False


def test_large_delta_is_audible() -> None:
    base = _tone(1000.0, 0.9, dur_s=1.0) + _tone(12000.0, 0.02, dur_s=1.0)
    loud = _tone(1000.0, 0.9, dur_s=1.0) + _tone(12000.0, 0.2, dur_s=1.0)  # +20 dB
    res = band_audibility(base, loud, SR, 8000.0, 16000.0)
    assert res["audible"] is True


def test_roughness_am_increases() -> None:
    carrier = _tone(1000.0, 0.5, dur_s=2.0)
    n = len(carrier)
    t = np.arange(n) / SR
    am = (0.5 + 0.5 * np.sin(2 * np.pi * 70.0 * t)).astype(np.float32)  # 70 Hz AM
    rough = (carrier * am).astype(np.float32)
    assert compute_roughness_asper(rough, SR) > compute_roughness_asper(carrier, SR)


def test_roughness_rise_signed() -> None:
    carrier = _tone(1000.0, 0.5, dur_s=2.0)
    n = len(carrier)
    t = np.arange(n) / SR
    am = (0.5 + 0.5 * np.sin(2 * np.pi * 70.0 * t)).astype(np.float32)
    rough = (carrier * am).astype(np.float32)
    assert roughness_rise_asper(carrier, rough, SR) > 0.0
    assert roughness_rise_asper(rough, carrier, SR) < 0.0
    assert roughness_rise_asper(carrier, carrier.copy(), SR) == 0.0


def test_stereo_collapse_detected_by_witness() -> None:
    """P3: Stereo → Mono-Kollaps meldet stereo_collapse + ILD/IACC-Drift."""
    from backend.core.listening_witness import evaluate_listening_witness

    left = _tone(440.0, 0.4, dur_s=2.0) + _tone(220.0, 0.3, dur_s=2.0)
    right = _tone(660.0, 0.4, dur_s=2.0) + _tone(220.0, 0.2, dur_s=2.0)
    stereo = np.stack([left, right], axis=0).astype(np.float32)  # (2, N)
    collapsed = np.stack([np.mean(stereo, axis=0), np.mean(stereo, axis=0)], axis=0).astype(np.float32)
    res = evaluate_listening_witness(stereo, collapsed, SR, "phase_p3")
    assert res.ild_drift_db > 1.0 or res.iacc_drop > 0.05
    assert "stereo_collapse" in res.findings


def test_pre_echo_detected_by_witness() -> None:
    """P4: Energie vor einem Transienten → pre_echo-Finding."""
    from backend.core.listening_witness import evaluate_listening_witness

    rng = np.random.default_rng(5)
    n = int(SR * 2.0)
    base = (rng.standard_normal(n) * 0.01).astype(np.float32)
    onset = int(SR * 1.0)
    base[onset : onset + 200] += 0.8 * np.hanning(200).astype(np.float32)
    base[onset + 200 : onset + 800] += 0.4
    with_echo = base.copy()
    pre = int(SR * 0.010)
    with_echo[onset - pre : onset] += 0.15 * np.hanning(pre).astype(np.float32)  # 10 ms Vor-Energie
    res = evaluate_listening_witness(base, with_echo, SR, "phase_p4")
    assert res.pre_echo_db > -12.0
    assert "pre_echo" in res.findings
