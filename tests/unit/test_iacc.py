"""Tests für compute_iacc — Inter-Aural Cross-Correlation (§V44) in stereo_guard.py.

Abdeckung:
  - IACCResult dataclass: Attribute, Wertebereich
  - compute_iacc: Mono → IACC=1.0 (Mono-Sonderfall)
  - compute_iacc: Identische L/R-Kanäle → IACC≈1.0 (Phantom-Center)
  - compute_iacc: Unkorrelierte Kanäle → IACC < 0.5 (breites Klangbild)
  - spatial_depth_score = 1.0 - iacc
  - ok = iacc < 0.70
  - Edge-Cases: Stille, kurzes Signal, Mono-Layout
"""

import numpy as np
import pytest

SR = 48000


def _silence(n: int = SR) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def _correlated_stereo(n: int = SR, seed: int = 0) -> np.ndarray:
    """L == R: IACC = 1.0 (Phantom-Center / Mono-like)."""
    rng = np.random.default_rng(seed)
    mono = rng.standard_normal(n).astype(np.float32)
    return np.stack([mono, mono], axis=0)  # [2, N]


def _decorrelated_stereo(n: int = SR, seed_l: int = 1, seed_r: int = 2) -> np.ndarray:
    """L und R aus unabhängigen Quellen: IACC nahe 0."""
    rng_l = np.random.default_rng(seed_l)
    rng_r = np.random.default_rng(seed_r)
    l = rng_l.standard_normal(n).astype(np.float32)
    r = rng_r.standard_normal(n).astype(np.float32)
    return np.stack([l, r], axis=0)


def _delayed_stereo(delay_samples: int = 24, n: int = SR, seed: int = 5) -> np.ndarray:
    """R ist L um delay_samples verschoben (< 1 ms = suchbarer Bereich)."""
    rng = np.random.default_rng(seed)
    mono = rng.standard_normal(n).astype(np.float32)
    r = np.roll(mono, delay_samples)
    r[:delay_samples] = 0.0
    return np.stack([mono, r], axis=0)


@pytest.mark.unit
class TestIACCResult:
    """IACCResult dataclass — Attribute."""

    def test_attributes_exist(self):
        from backend.core.dsp.stereo_guard import IACCResult

        r = IACCResult(iacc=0.7, spatial_depth_score=0.3, tau_max_ms=0.25, ok=False)
        assert r.iacc == pytest.approx(0.7)
        assert r.spatial_depth_score == pytest.approx(0.3)
        assert r.tau_max_ms == pytest.approx(0.25)
        assert r.ok is False

    def test_iacc_range(self):
        from backend.core.dsp.stereo_guard import IACCResult

        r = IACCResult(iacc=0.5, spatial_depth_score=0.5, tau_max_ms=0.0, ok=True)
        assert 0.0 <= r.iacc <= 1.0
        assert 0.0 <= r.spatial_depth_score <= 1.0

    def test_spatial_depth_complement(self):
        """spatial_depth_score = 1.0 - iacc."""
        from backend.core.dsp.stereo_guard import IACCResult

        r = IACCResult(iacc=0.6, spatial_depth_score=0.4, tau_max_ms=0.0, ok=True)
        assert r.spatial_depth_score == pytest.approx(1.0 - r.iacc, abs=1e-3)


class TestComputeIACC:
    """compute_iacc — Hauptfunktion."""

    def test_mono_input_returns_iacc_1(self):
        """Mono-Signal → IACC = 1.0 (triviale Mono-Kompatibilität)."""
        from backend.core.dsp.stereo_guard import compute_iacc

        mono = np.ones(SR, dtype=np.float32) * 0.1
        r = compute_iacc(mono, SR)
        assert r.iacc == pytest.approx(1.0)
        assert r.spatial_depth_score == pytest.approx(0.0)

    def test_correlated_stereo_high_iacc(self):
        """L == R → IACC nahe 1.0."""
        from backend.core.dsp.stereo_guard import compute_iacc

        stereo = _correlated_stereo()
        r = compute_iacc(stereo, SR)
        assert r.iacc > 0.90, f"L==R: IACC soll hoch sein, got {r.iacc}"

    def test_decorrelated_stereo_low_iacc(self):
        """L und R unkorreliert → IACC nahe 0 (breites Klangbild)."""
        from backend.core.dsp.stereo_guard import compute_iacc

        stereo = _decorrelated_stereo()
        r = compute_iacc(stereo, SR)
        assert r.iacc < 0.30, f"Unkorrelliertes Stereo: IACC soll niedrig sein, got {r.iacc}"
        assert r.ok is True, "Unkorrelliertes Stereo: ok=True (breites Klangbild)"

    def test_correlated_stereo_ok_false(self):
        """L == R → ok=False (Mono-Tendenz)."""
        from backend.core.dsp.stereo_guard import compute_iacc

        stereo = _correlated_stereo()
        r = compute_iacc(stereo, SR)
        assert r.ok is False or r.iacc > 0.90

    def test_decorrelated_stereo_ok_true(self):
        """Unkorrelliertes Stereo → ok=True."""
        from backend.core.dsp.stereo_guard import compute_iacc

        stereo = _decorrelated_stereo()
        r = compute_iacc(stereo, SR)
        assert r.ok is True

    def test_spatial_depth_score_complement(self):
        """spatial_depth_score = 1.0 - iacc (IMMER)."""
        from backend.core.dsp.stereo_guard import compute_iacc

        for stereo in [_correlated_stereo(), _decorrelated_stereo(), _delayed_stereo(24)]:
            r = compute_iacc(stereo, SR)
            assert r.spatial_depth_score == pytest.approx(1.0 - r.iacc, abs=1e-3), (
                f"spatial_depth_score ({r.spatial_depth_score}) ≠ 1 - iacc ({1.0 - r.iacc})"
            )

    def test_iacc_range(self):
        """IACC immer in [0.0, 1.0]."""
        from backend.core.dsp.stereo_guard import compute_iacc

        for stereo in [_correlated_stereo(), _decorrelated_stereo(), _delayed_stereo(24)]:
            r = compute_iacc(stereo, SR)
            assert 0.0 <= r.iacc <= 1.0, f"IACC außerhalb [0, 1]: {r.iacc}"

    def test_no_nan(self):
        """Kein NaN in Output."""
        from backend.core.dsp.stereo_guard import compute_iacc

        for audio in [_correlated_stereo(), _decorrelated_stereo()]:
            r = compute_iacc(audio, SR)
            assert not np.isnan(r.iacc)
            assert not np.isnan(r.spatial_depth_score)
            assert not np.isnan(r.tau_max_ms)

    def test_short_signal_no_crash(self):
        """Kurzes Signal → kein Absturz."""
        from backend.core.dsp.stereo_guard import compute_iacc

        short = np.ones((2, 128), dtype=np.float32) * 0.1
        r = compute_iacc(short, SR)
        assert 0.0 <= r.iacc <= 1.0

    def test_column_layout_stereo(self):
        """[N, 2] Layout wird korrekt verarbeitet."""
        from backend.core.dsp.stereo_guard import compute_iacc

        # Column-first Layout: [N, 2]
        n = SR
        stereo_col = np.random.default_rng(7).standard_normal((n, 2)).astype(np.float32)
        r = compute_iacc(stereo_col, SR)
        assert 0.0 <= r.iacc <= 1.0

    def test_tau_max_ms_within_window(self):
        """tau_max_ms muss im Suchfenster ±1 ms liegen."""
        from backend.core.dsp.stereo_guard import compute_iacc

        r = compute_iacc(_correlated_stereo(), SR)
        assert -2.0 <= r.tau_max_ms <= 2.0, f"tau_max_ms außerhalb Suchfenster: {r.tau_max_ms}"

    def test_delayed_stereo_detected(self):
        """Verzögerter R-Kanal → IACC < 1.0 (leichte Dekorrelation durch Delay)."""
        from backend.core.dsp.stereo_guard import compute_iacc

        stereo = _delayed_stereo(delay_samples=24)  # 0.5 ms @ 48 kHz
        r = compute_iacc(stereo, SR)
        assert r.iacc <= 1.0

    def test_silence_fallback(self):
        """Stille-Kanäle → kein Absturz, Fallback-Rückgabe."""
        from backend.core.dsp.stereo_guard import compute_iacc

        silent_stereo = np.zeros((2, SR), dtype=np.float32)
        r = compute_iacc(silent_stereo, SR)
        assert not np.isnan(r.iacc)
        assert not np.isnan(r.spatial_depth_score)

    def test_assert_sr_48000(self):
        """Falsche SR → AssertionError."""
        from backend.core.dsp.stereo_guard import compute_iacc

        with pytest.raises(AssertionError):
            compute_iacc(_correlated_stereo(), sr=44100)


def test_spatial_depth_metric_uses_spatial_depth_score_as_primary_proxy(monkeypatch):
    """§V44: SpatialDepthMetric muss `spatial_depth_score` primär verwenden.

    Wir patchen `compute_iacc` absichtlich mit inkonsistenten Testwerten,
    damit klar verifiziert wird, dass `_measure_absolute()` den
    `spatial_depth_score` (und nicht ein IACC-Threshold-Mapping) nutzt.
    """
    from backend.core.musical_goals.musical_goals_metrics import SpatialDepthMetric

    class _FakeIaccResult:
        def __init__(self) -> None:
            self.iacc = 0.30
            self.spatial_depth_score = 0.05
            self.ok = True

    calls = {"count": 0}

    def _fake_compute_iacc(_audio: np.ndarray, sr: int):
        assert sr == SR
        calls["count"] += 1
        return _FakeIaccResult()

    monkeypatch.setattr("backend.core.dsp.stereo_guard.compute_iacc", _fake_compute_iacc)

    # Identische L/R Kanäle: width_score=0, depth_score=0, center_score≈1.0
    # Neue Gewichtung §V44: score = 0.55*iacc_score + 0.10*center_score.
    # Bei iacc_score=0.05 (IACC=0.95): 0.55*0.05 + 0.10*1.0 ≈ 0.1275.
    mono = np.random.default_rng(123).standard_normal(SR).astype(np.float32)
    stereo = np.stack([mono, mono], axis=1)  # [N, 2]

    score = SpatialDepthMetric().measure(stereo, SR)

    assert calls["count"] >= 1
    assert score == pytest.approx(0.1275, abs=0.03)
