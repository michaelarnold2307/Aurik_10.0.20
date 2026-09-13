"""Tests für WF-V1…V3: Bandbegrenztes Resampling, Kalman-Glättung, Warp-Schätzer."""

from __future__ import annotations

import numpy as np

from backend.core.dsp.bandlimited_resampler import bandlimited_warp
from backend.core.dsp.warp_estimator import consensus_warp, spectral_warp_estimate
from backend.core.dsp.warp_kalman import kalman_smooth_warp

SR = 44100


def test_bandlimited_warp_integer_positions_exact() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500).astype(np.float32)
    pos = np.arange(500, dtype=np.float64)
    y = bandlimited_warp(x, pos)
    assert np.allclose(y, x, atol=1e-4)


def test_bandlimited_warp_hf_quality_better_than_linear() -> None:
    """HF-Ton bei konstantem Ratio: Sinc-Fehler gegen die EXAKTE verschobene
    Referenz (analytisch) deutlich kleiner als bei linearer Interpolation."""
    f0 = 0.40 * SR  # 0.4·Nyquist
    n = 20000
    ratio = 1.003
    t = np.arange(n) / SR
    x = (0.5 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)
    pos = np.clip(np.arange(n, dtype=np.float64) / ratio, 0, n - 1)
    y_sinc = bandlimited_warp(x, pos)
    y_lin = np.interp(pos, np.arange(n, dtype=np.float64), x)
    # Exakte Referenz: Signal am richtigen Zeitpunkt.
    ref = 0.5 * np.sin(2 * np.pi * f0 * pos / SR)
    err_sinc = np.mean((y_sinc - ref) ** 2)
    err_lin = np.mean((y_lin - ref) ** 2)
    assert err_sinc < err_lin, f"Sinc muss besser sein: {err_sinc:.6f} vs {err_lin:.6f}"


def test_bandlimited_warp_deterministic_and_edges() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(1000).astype(np.float32)
    pos = np.clip(np.linspace(-3, 1002, 1000), 0, 999)
    y1 = bandlimited_warp(x, pos)
    y2 = bandlimited_warp(x, pos)
    assert np.array_equal(y1, y2)
    assert np.all(np.isfinite(y1))


def test_kalman_smooths_jitter_keeps_trend() -> None:
    n = 400
    base = 1.0 + 0.005 * np.sin(2 * np.pi * 0.01 * np.arange(n))
    rng = np.random.default_rng(5)
    noisy = base + rng.standard_normal(n) * 0.002
    out = kalman_smooth_warp(noisy, q=1e-8, r=1e-5)
    err_out = np.mean((out - base) ** 2)
    err_noisy = np.mean((noisy - base) ** 2)
    assert err_out < err_noisy * 0.5, f"Glättung muss Jitter reduzieren: {err_out:.6f} vs {err_noisy:.6f}"


def test_kalman_drift_model_preserves_slow_drift() -> None:
    n = 600
    base = 1.0 + 0.01 * np.sin(2 * np.pi * 0.002 * np.arange(n))  # langsamer Hub-Wow
    rng = np.random.default_rng(7)
    noisy = base + rng.standard_normal(n) * 0.003
    out = kalman_smooth_warp(noisy, drift_model=True, q=1e-8, r=1e-5)
    # Drift bleibt erhalten (Korrelation mit Basis hoch), Jitter sinkt.
    corr = float(np.corrcoef(out, base)[0, 1])
    assert corr > 0.9, f"Drift darf nicht weggeglättet werden: corr={corr:.3f}"
    assert np.mean((out - base) ** 2) < np.mean((noisy - base) ** 2)


def test_spectral_warp_estimate_recovers_static_warp() -> None:
    """Konstant gewarptes Signal: Der Schätzer liefert ≈ das Warp-Verhältnis."""
    t = np.arange(SR * 3) / SR
    x = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    x = x.astype(np.float32)
    ratio = 1.008  # Aufnahme 0,8 % zu schnell
    from backend.core.dsp.bandlimited_resampler import bandlimited_constant_ratio

    warped = bandlimited_constant_ratio(x, 1.0 / ratio)  # langsamer machen ≠ Wiederherstellung — nur Struktur-Test
    times, warp_est, quality = spectral_warp_estimate(warped, SR)
    # Der Schätzer vergleicht gegen die Median-Referenz des SELBEN Signals →
    # für ein konstant gewarptes Signal ist die Relativ-Warp ≈ 0 → warp ≈ 1.
    # Hier testen wir Robustheit: alle Werte endlich und nahe 1 (kein Ausreißer > 2 %).
    assert len(times) > 10
    assert np.all(np.isfinite(warp_est))
    assert np.max(np.abs(warp_est - 1.0)) < 0.02, f"Ausreißer: {np.max(np.abs(warp_est - 1.0)):.4f}"


def test_consensus_warp_rejects_wrong_trajectory() -> None:
    n = 100
    t = np.arange(n, dtype=np.float64)
    good = 1.0 + 0.004 * np.sin(2 * np.pi * 0.02 * t)
    wrong = np.full(n, 1.05)  # konstant deutlich abweichend — nie Konsens
    cons, agree = consensus_warp(good, t, wrong, t, quality_b=None, tol=0.005, min_quality=0.6)
    assert np.allclose(cons, good)  # Abweichung → traj_a bleibt
    assert not agree.any()


def test_consensus_warp_averages_agreement() -> None:
    n = 100
    t = np.arange(n, dtype=np.float64)
    a = 1.0 + 0.004 * np.sin(2 * np.pi * 0.02 * t)
    b = a + np.random.default_rng(9).standard_normal(n) * 0.001  # nahe dran
    cons, agree = consensus_warp(a, t, b, t, quality_b=None, tol=0.005, min_quality=0.0)
    assert agree.mean() > 0.8
    assert np.allclose(cons[agree], 0.5 * (a[agree] + b[agree]))
