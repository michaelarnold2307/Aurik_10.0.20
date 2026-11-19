"""Vertragstests für ``backend/core/dsp/audibility_budget.py`` (Hörordnung §4).

Verankert:
- Der Maskierungskern ist bit-identisch zum kanonischen
  ``masking_model.compute_masking_threshold_db`` (eine Wahrheit, §G5 (copilot-instructions.md)).
- Budget = max(0, Defekt - (Schwelle - Marge)); nie negativ, nie unbegrenzt.
- Der Gain-Boden hebt ausschließlich an, nie über 1.0, deterministisch.
- Rausch-only-Material bekommt kein künstliches Kappungsverbot (Masker fehlt).
- Flag-Default AUS (bit-identisches Bestandsverhalten) und fail-safe lesbar.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.dsp.audibility_budget import (
    MAX_BUDGET_DB,
    BudgetStats,
    apply_gain_budget,
    audibility_budget_db,
    audibility_budget_enabled,
    band_budget_curve,
    band_index_for_freqs,
    band_power_db,
    bark_band_centres_z,
    gain_floor_from_budget,
    masking_threshold_db,
    spectrum_budget_db,
    spectrum_gain_floor,
)
from backend.core.dsp.masking_model import (
    _frame_band_energies,
    bark_band_edges,
    compute_masking_threshold_db,
)

SR = 48000
_N_BANDS = len(bark_band_edges(SR)) - 1


def _programme_with_noise(seed: int = 42, n: int = SR) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    clean = 0.30 * np.sin(2 * np.pi * 1000.0 * t) + 0.10 * np.sin(2 * np.pi * 300.0 * t)
    noise = 0.01 * rng.standard_normal(n)
    return (clean + noise).astype(np.float64), noise.astype(np.float64)


class TestMaskingKernel:
    def test_bit_identical_to_canonical_model(self):
        x = _programme_with_noise()[0].astype(np.float32)
        thr_ref, z_ref = compute_masking_threshold_db(x, SR)
        band_db, z_bands = _frame_band_energies(x, SR)
        thr_mine = masking_threshold_db(band_db.T, bark_band_centres_z(bark_band_edges(SR)))
        assert thr_mine.shape == (thr_ref.shape[1], thr_ref.shape[0])
        assert float(np.max(np.abs(thr_ref.T - thr_mine))) == 0.0
        assert float(np.max(np.abs(z_ref - z_bands))) == 0.0

    def test_no_nan_for_empty_and_degenerate_bands(self):
        levels = np.full((_N_BANDS, 1), -200.0)
        thr = masking_threshold_db(levels, bark_band_centres_z(bark_band_edges(SR)))
        assert np.isfinite(thr).all()

    def test_long_input_is_finite_and_deterministic(self):
        rng = np.random.default_rng(7)
        levels = rng.uniform(-100.0, 0.0, size=(_N_BANDS, 4100))
        z = bark_band_centres_z(bark_band_edges(SR))
        first = masking_threshold_db(levels, z)
        second = masking_threshold_db(levels, z)
        assert np.isfinite(first).all()
        assert first.shape == levels.shape
        assert np.array_equal(first, second)


class TestBudgetSemantics:
    def test_below_threshold_needs_no_attenuation(self):
        defect = np.array([[-60.0, -50.0]])
        threshold = np.array([[-40.0, -10.0]])
        assert np.all(audibility_budget_db(defect, threshold, margin_db=3.0) == 0.0)

    def test_excess_plus_margin_is_the_budget(self):
        defect = np.array([[-20.0]])
        threshold = np.array([[-50.0]])
        budget = audibility_budget_db(defect, threshold, margin_db=3.0)
        assert float(budget[0, 0]) == pytest.approx(33.0)

    def test_budget_is_capped(self):
        budget = audibility_budget_db(np.array([[0.0]]), np.array([[-400.0]]))
        assert float(budget[0, 0]) == MAX_BUDGET_DB

    def test_gain_floor_matches_db_definition(self):
        budget = np.array([[0.0, 20.0, 60.0]])
        floor = gain_floor_from_budget(budget)
        assert floor[0, 0] == pytest.approx(1.0)
        assert floor[0, 1] == pytest.approx(0.1)
        assert floor[0, 2] == pytest.approx(0.001)


class TestGainApplication:
    def test_only_lifts_and_never_above_one(self):
        g = np.array([[0.05, 1.0], [0.9, 1.0]], dtype=np.float32)
        idx = np.array([0, 0], dtype=np.int32)
        budget = np.array([[13.0, 0.0]])  # 13 dB ⇒ Boden 0.2239
        out, stats = apply_gain_budget(g, idx, budget)
        assert np.all(out >= g - 1e-9)
        assert np.all(out <= 1.0)
        assert float(out[0, 0]) == pytest.approx(10.0 ** (-13.0 / 20.0), rel=1e-6)
        assert float(out[1, 0]) == pytest.approx(0.9, rel=1e-6)  # war schon über dem Boden
        assert float(out[0, 1]) == pytest.approx(1.0)
        assert stats.applied is True
        assert stats.bound_fraction == pytest.approx(0.25, abs=1e-3)

    def test_deterministic(self):
        g = np.array([[0.01, 0.2]], dtype=np.float32)
        idx = np.array([0], dtype=np.int32)
        budget = np.array([[10.0, 20.0]])
        a, _ = apply_gain_budget(g, idx, budget)
        b, _ = apply_gain_budget(g, idx, budget)
        assert np.array_equal(a, b)

    def test_mismatched_band_index_is_fail_safe(self):
        g = np.ones((3, 4), dtype=np.float32)
        out, stats = apply_gain_budget(g, np.array([0, 1], dtype=np.int32), np.zeros((2, 4)))
        assert np.array_equal(out, g)
        assert stats.applied is False

    def test_time_resampling_keeps_floor_in_bounds(self):
        g = np.full((2, 100), 0.01, dtype=np.float32)
        idx = np.array([0, 1], dtype=np.int32)
        out, _ = apply_gain_budget(g, idx, np.full((2, 7), 20.0))
        assert out.shape == g.shape
        assert np.all(out >= 0.09 - 1e-6)


class TestSpectrumBudget:
    def test_noise_below_signal_is_masked(self):
        tone = _programme_with_noise()[0]
        n = len(tone)
        f = np.fft.rfftfreq(2048, 1.0 / SR)
        # STFT-freie Prüfung: reines Signal, keine Rauschschätzung
        mag = np.abs(np.fft.rfft(tone[:2048] * np.hanning(2048)))
        zero = np.zeros_like(mag)
        budget, band_idx, stats = spectrum_budget_db(mag[:, None], zero[:, None], f, SR)
        assert budget.shape[0] == _N_BANDS
        assert band_idx.shape[0] == f.shape[0]
        assert np.all(budget == 0.0)
        assert stats.limited_fraction == 0.0

    def test_noise_only_gets_large_budget(self):
        rng = np.random.default_rng(11)
        noise = 0.1 * rng.standard_normal(2048)
        f = np.fft.rfftfreq(2048, 1.0 / SR)
        mag = np.abs(np.fft.rfft(noise * np.hanning(2048)))
        budget, _, _ = spectrum_budget_db(mag[:, None], mag[:, None], f, SR)
        active = budget[budget > 0.0]
        assert active.size > 0
        assert float(np.max(budget)) == pytest.approx(MAX_BUDGET_DB)

    def test_gain_floor_facade_is_bounded_and_finite(self):
        x, noise = _programme_with_noise(n=SR // 2)
        f = np.fft.rfftfreq(2048, 1.0 / SR)
        mag = np.abs(np.fft.rfft(x[:2048] * np.hanning(2048)))
        nmag = np.abs(np.fft.rfft(noise[:2048] * np.hanning(2048)))
        floor, stats = spectrum_gain_floor(mag[:, None], nmag[:, None], f, SR)
        assert floor.shape == (f.size, 1)
        assert np.isfinite(floor).all()
        assert float(floor.min()) >= 0.0 and float(floor.max()) <= 1.0
        assert isinstance(stats, BudgetStats)


class TestBandBudgetCurve:
    def test_length_and_bounds(self):
        x, _ = _programme_with_noise(n=SR // 2)
        floor, stats = band_budget_curve(x, SR, (5000.0, 8000.0))
        assert floor.shape == (x.size,)
        assert np.isfinite(floor).all()
        assert float(floor.min()) >= 0.0 and float(floor.max()) <= 1.0
        assert stats.applied is True

    def test_too_short_signal_is_transparent(self):
        floor, stats = band_budget_curve(np.zeros(64), SR, (5000.0, 8000.0))
        assert np.all(floor == 1.0)
        assert stats.applied is False

    def test_silence_is_finite(self):
        floor, _ = band_budget_curve(np.zeros(SR // 4), SR, (5000.0, 8000.0))
        assert np.isfinite(floor).all()


class TestFlag:
    def test_default_is_off(self):
        from backend.core import music_model_flags

        assert music_model_flags.use_audibility_budget is False
        assert audibility_budget_enabled() is False

    def test_reads_flag_at_call_time(self, monkeypatch):
        from backend.core import music_model_flags

        monkeypatch.setattr(music_model_flags, "use_audibility_budget", True, raising=True)
        assert audibility_budget_enabled() is True

    def test_unreadable_flag_is_off(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _boom(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "backend.core" and fromlist and "music_model_flags" in fromlist:
                raise RuntimeError("kaputt")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _boom)
        assert audibility_budget_enabled() is False


class TestHelpers:
    def test_band_index_covers_nyquist(self):
        edges = bark_band_edges(SR)
        idx = band_index_for_freqs(np.array([0.0, 1000.0, SR / 2.0]), edges)
        assert idx.tolist() == [0, int(np.searchsorted(edges, 1000.0, side="right") - 1), len(edges) - 2]

    def test_band_power_db_is_nan_safe(self):
        p = np.array([[np.nan, np.inf], [1.0, 4.0]])
        out = band_power_db(p, np.array([0, 1], dtype=np.int32), 2)
        assert np.isfinite(out).all()
