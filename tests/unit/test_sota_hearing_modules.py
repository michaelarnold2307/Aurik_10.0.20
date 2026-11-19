"""Tests für die SOTA-Hörmodule (Level-2): BMLD, DLM, Gammachirp, HRIR, GO/NO-GO."""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.dsp.binaural_masking import binaural_masking_advantage, binaural_noise_floor_release_db
from backend.core.dsp.gammachirp_filterbank import gammachirp_spectrogram, nsim_gammachirp
from backend.core.dsp.interaural_cues import apply_hrir_pair
from backend.core.dsp.temporal_loudness import stl_adaptive_smoothing, temporal_loudness
from backend.core.go_nogo_export_gate import evaluate

SR = 48000


def _stereo_noise(n: int = SR, corr: float = 1.0) -> np.ndarray:
    """Stereo (2,N): links = Rauschen, rechts = korr·links + (1−corr)·unabhängig."""
    rng = np.random.default_rng(7)
    left = rng.standard_normal(n).astype(np.float32)
    right = corr * left + (1.0 - corr) * rng.standard_normal(n).astype(np.float32)
    peak = max(np.max(np.abs(left)), np.max(np.abs(right))) + 1e-9
    return np.stack([left / peak, right / peak], axis=0)


def test_bmld_uncorrelated_releases_more_than_correlated() -> None:
    corr = _stereo_noise(corr=1.0)
    uncorr = _stereo_noise(corr=0.0)
    assert binaural_noise_floor_release_db(corr, SR) < binaural_noise_floor_release_db(uncorr, SR) + 1e-6
    res = binaural_masking_advantage(uncorr, SR)
    assert res.release_db > 3.0  # Nπ-artiges Rauschen → deutliche Freisetzung
    assert 0.0 <= res.nr_floor_release_db <= 8.0


def test_bmld_mono_is_zero() -> None:
    mono = _stereo_noise(corr=1.0)[0]  # 1-D
    assert binaural_noise_floor_release_db(mono, SR) == 0.0


def test_temporal_loudness_dynamics() -> None:
    n = SR
    t = np.arange(n) / SR
    # Laut/leise-Wechsel: erste Hälfte laut, zweite leise
    x = np.concatenate([0.3 * np.sin(2 * np.pi * 440 * t[: n // 2]), 0.02 * np.sin(2 * np.pi * 440 * t[n // 2 :])])
    res = temporal_loudness(x.astype(np.float32), SR)
    assert res.stl_sone.shape == (n,)
    assert res.peak_stl_sone > 0.0
    assert np.mean(res.stl_sone[: n // 4]) > np.mean(res.stl_sone[3 * n // 4 :])


def test_stl_adaptive_smoothing_bounded() -> None:
    n = SR // 10
    g = np.zeros(n, dtype=np.float64)
    g[n // 2 :] = 6.0
    x = (np.random.default_rng(1).standard_normal(n) * 0.1).astype(np.float32)
    out = stl_adaptive_smoothing(g, x, SR)
    assert out.shape == g.shape
    assert float(np.max(out)) <= 6.0 + 1e-9  # nie über Input-Maximum
    assert np.all(np.isfinite(out))


def test_gammachirp_spec_and_nsim() -> None:
    n = SR // 5
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(n).astype(np.float32)
    spec = gammachirp_spectrogram(ref, SR)
    assert spec.ndim == 2 and spec.shape[0] > 8 and spec.shape[1] > 4
    # identisch → NSIM ≈ 1; verrauscht → niedriger
    assert nsim_gammachirp(ref, ref.copy(), SR) > 0.95
    deg = ref + 0.5 * rng.standard_normal(n).astype(np.float32)
    assert nsim_gammachirp(ref, deg, SR) < nsim_gammachirp(ref, ref.copy(), SR)


def test_apply_hrir_pair_shape_and_determinism() -> None:
    n = SR // 4
    mono = (np.random.default_rng(5).standard_normal(n) * 0.1).astype(np.float32)
    hrir_l = np.zeros(64, dtype=np.float32)
    hrir_r = np.zeros(64, dtype=np.float32)
    hrir_l[0], hrir_l[10] = 0.8, 0.4
    hrir_r[0], hrir_r[20] = 0.8, -0.3
    out = apply_hrir_pair(mono, hrir_l, hrir_r)
    assert out.shape == (2, n)
    out2 = apply_hrir_pair(mono, hrir_l, hrir_r)
    assert np.array_equal(out, out2)  # deterministisch


def test_go_nogo_gate_verdicts() -> None:
    class R:
        metadata: dict = {}
        warnings: list[str] = []
        quality_estimate: float = 0.8

    good = R()
    assert evaluate(good).verdict == "GO"

    bad = R()
    bad.quality_estimate = 0.4
    v = evaluate(bad)
    assert v.verdict == "NO_GO" and any("quality_estimate" in r for r in v.reasons)

    pump = R()
    pump.warnings = ["NR pumping artifact detected"]
    v2 = evaluate(pump)
    assert v2.verdict == "NO_GO"

    nan_audio = np.array([0.0, np.nan], dtype=np.float32)
    assert evaluate(R(), audio=nan_audio).verdict == "NO_GO"
