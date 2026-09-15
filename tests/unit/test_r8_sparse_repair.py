"""§SOTA-R8 (Sparse Repair) — Regressionsschutz für die PERF-D-Infrastruktur.

Deckt ``backend/core/dsp/sparse_repair.py``:
- Passthrough bei leerer Maske (kein repair_fn-Aufruf)
- Voll-Repair bei hoher Coverage
- Fenster-Reparatur nur in Defekt-Nähe + Kontext, Hann-Crossfade ohne Sprünge
- deterministisch, NaN-sicher, Layout-sicher (mono/(C,N)/(N,C))
- Fehlervertrag: repair_fn-Fehler lassen die Region unverändert (§V6 (copilot-instructions.md))

Autor: Aurik Testing Team
"""

import numpy as np
import pytest

from backend.core.dsp.sparse_repair import (
    SparseRepairResult,
    defect_regions,
    sparse_windowed_repair,
)

SR = 48000


def _base(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.05, n).astype(np.float32)


def _noop_repair(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).copy()


def _attenuate_repair(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) * 0.5).astype(np.float32)


class TestDefectRegions:
    def test_single_region(self):
        mask = np.zeros(100, dtype=bool)
        mask[20:40] = True
        assert defect_regions(mask) == [(20, 40)]

    def test_multiple_regions(self):
        mask = np.zeros(100, dtype=bool)
        mask[5:10] = True
        mask[60:70] = True
        assert defect_regions(mask) == [(5, 10), (60, 70)]

    def test_small_gap_merged(self):
        mask = np.zeros(100, dtype=bool)
        mask[10:20] = True
        mask[24:30] = True  # 4 Samples Lücke < min_gap 8 → mergen
        regions = defect_regions(mask, min_gap_samples=8)
        assert len(regions) == 1
        assert regions[0][0] == 10
        assert regions[0][1] == 30

    def test_empty_mask(self):
        assert defect_regions(np.zeros(50, dtype=bool)) == []

    def test_nd_mask_rejected(self):
        with pytest.raises(ValueError):
            defect_regions(np.zeros((2, 50), dtype=bool))


class TestSparseWindowedRepair:
    def test_empty_mask_passthrough_no_call(self):
        audio = _base(4800, 1)
        calls = {"n": 0}

        def fn(x):
            calls["n"] += 1
            return x

        res = sparse_windowed_repair(audio, SR, np.zeros(4800, dtype=bool), fn)
        assert res.skipped and res.regions_repaired == 0 and res.full_repair is False
        assert calls["n"] == 0
        assert np.array_equal(res.audio, audio)

    def test_full_coverage_single_call(self):
        audio = _base(4800, 2)
        calls = {"n": 0}

        def fn(x):
            calls["n"] += 1
            return x * 0.5

        res = sparse_windowed_repair(audio, SR, np.ones(4800, dtype=bool), fn, coverage_threshold=0.85)
        assert res.full_repair and calls["n"] == 1
        assert np.allclose(res.audio, audio * 0.5, atol=1e-7)

    def test_sparse_repair_only_near_defects(self):
        audio = _base(48000, 3)
        mask = np.zeros(48000, dtype=bool)
        mask[10000:10100] = True  # ein 100-Sample-Defekt

        def fn(x):
            return x * 0.25  # Reparatur = starke Dämpfung im Fenster

        res = sparse_windowed_repair(audio, SR, mask, fn, context_ms=0.0, crossfade_ms=0.0)
        # Reparatur nur im Defektfenster (exakt, da Kontext/Fade = 0)
        assert res.regions_repaired == 1 and not res.full_repair
        assert np.allclose(res.audio[:10000], audio[:10000], atol=1e-7)
        assert np.allclose(res.audio[10000:10100], audio[10000:10100] * 0.25, atol=1e-6)
        assert np.allclose(res.audio[10100:], audio[10100:], atol=1e-7)

    def test_context_expands_window(self):
        audio = _base(48000, 4)
        mask = np.zeros(48000, dtype=bool)
        mask[24000:24010] = True

        def fn(x):
            return x * 0.5

        res = sparse_windowed_repair(audio, SR, mask, fn, context_ms=25.0, crossfade_ms=0.0)
        ctx = 25 * 48  # 25 ms @ 48 kHz = 1200 Samples
        assert np.allclose(res.audio[24000 - ctx : 24010 + ctx], audio[24000 - ctx : 24010 + ctx] * 0.5, atol=1e-6)
        assert np.allclose(res.audio[: 24000 - ctx], audio[: 24000 - ctx], atol=1e-7)

    def test_crossfade_has_no_jump(self):
        audio = np.linspace(-0.5, 0.5, 48000, dtype=np.float32)
        mask = np.zeros(48000, dtype=bool)
        mask[24000:24050] = True

        def fn(x):
            return x * 0.1

        res = sparse_windowed_repair(audio, SR, mask, fn, context_ms=10.0, crossfade_ms=5.0)
        # Kein Sprung an den Fenster-Grenzen: max. Sample-zu-Sample-Delta klein halten.
        diff = np.abs(np.diff(res.audio.astype(np.float64)))
        # Basis-Signal hat ~1/48000 Steigung; Crossfade darf nicht springen
        assert float(diff.max()) < 5e-3

    def test_stereo_channels_first(self):
        rng = np.random.default_rng(5)
        audio = rng.normal(0, 0.05, (2, 4800)).astype(np.float32)
        mask = np.zeros(4800, dtype=bool)
        mask[1000:1100] = True
        res = sparse_windowed_repair(audio, SR, mask, _attenuate_repair, context_ms=0.0, crossfade_ms=0.0)
        assert res.audio.shape == (2, 4800)
        assert np.allclose(res.audio[:, 1000:1100], audio[:, 1000:1100] * 0.5, atol=1e-6)
        assert np.allclose(res.audio[:, :1000], audio[:, :1000], atol=1e-7)

    def test_stereo_samples_first(self):
        rng = np.random.default_rng(6)
        audio = rng.normal(0, 0.05, (4800, 2)).astype(np.float32)
        mask = np.zeros(4800, dtype=bool)
        mask[1000:1100] = True
        res = sparse_windowed_repair(audio, SR, mask, _attenuate_repair, context_ms=0.0, crossfade_ms=0.0)
        assert res.audio.shape == (4800, 2)
        assert np.allclose(res.audio[1000:1100, :], audio[1000:1100, :] * 0.5, atol=1e-6)

    def test_deterministic(self):
        audio = _base(48000, 7)
        mask = np.zeros(48000, dtype=bool)
        mask[5000:5200] = True
        mask[20000:20150] = True
        r1 = sparse_windowed_repair(audio, SR, mask, _attenuate_repair, context_ms=20.0, crossfade_ms=5.0)
        r2 = sparse_windowed_repair(audio, SR, mask, _attenuate_repair, context_ms=20.0, crossfade_ms=5.0)
        assert np.array_equal(r1.audio, r2.audio)

    def test_repair_fn_failure_leaves_region(self):
        audio = _base(4800, 8)
        mask = np.zeros(4800, dtype=bool)
        mask[500:600] = True

        def boom(x):
            raise RuntimeError("kaputt")

        res = sparse_windowed_repair(audio, SR, mask, boom)
        assert res.regions_repaired == 0
        assert np.array_equal(res.audio, audio)

    def test_repair_fn_wrong_shape_leaves_region(self):
        audio = _base(4800, 9)
        mask = np.zeros(4800, dtype=bool)
        mask[500:600] = True

        def bad(x):
            return x[:-1]

        res = sparse_windowed_repair(audio, SR, mask, bad)
        assert res.regions_repaired == 0
        assert np.array_equal(res.audio, audio)

    def test_mask_length_mismatch_raises(self):
        audio = _base(4800, 10)
        with pytest.raises(ValueError):
            sparse_windowed_repair(audio, SR, np.zeros(4799, dtype=bool), _noop_repair)

    def test_nan_input_guarded(self):
        audio = _base(4800, 11)
        audio[100] = np.nan
        mask = np.zeros(4800, dtype=bool)
        mask[500:600] = True
        res = sparse_windowed_repair(audio, SR, mask, _noop_repair)
        assert np.isfinite(res.audio).all()
