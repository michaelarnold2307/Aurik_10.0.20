"""StemContext Unit-Tests (§v10.19 First-Class-Stem-Objekt).

Prüft: Rekombinations-Roundtrip (vocal+instr == mix), Layout-Coercion
(channels-first ↔ mono), as_dict-Bookkeeping ohne Audio-Arrays,
Never-worsen-Passthrough der Stems (keine Modifikation der Inputs).
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.dsp.stem_context import StemContext


def _mk_stereo() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n = 48000
    t = np.arange(n) / 48000.0
    vocal = np.stack([np.sin(2 * np.pi * 220 * t) * 0.4, np.sin(2 * np.pi * 220 * t) * 0.4]).astype(np.float32)
    instr = np.stack([rng.standard_normal(n) * 0.05, rng.standard_normal(n) * 0.05]).astype(np.float32)
    return vocal, instr


def test_recombine_roundtrip_stereo():
    vocal, instr = _mk_stereo()
    ctx = StemContext(
        vocal_stem=vocal,
        instrumental_stem=instr,
        sample_rate=48000,
        panns_singing=0.8,
        separation_model="bs_roformer",
        applied_stages=["miipher_dit", "kim_vocal_2", "dfn", "kim_inst"],
    )
    mix = ctx.recombine()
    np.testing.assert_allclose(mix, vocal + instr, atol=1e-6)
    assert mix.shape == vocal.shape
    assert mix.dtype == np.float32


def test_recombine_layout_coercion_to_mono_reference():
    vocal, instr = _mk_stereo()
    ctx = StemContext(vocal_stem=vocal, instrumental_stem=instr, sample_rate=48000)
    ref_mono = np.zeros(48000, dtype=np.float32)
    mix = ctx.recombine(reference=ref_mono)
    assert mix.ndim == 1
    np.testing.assert_allclose(mix, np.mean(vocal + instr, axis=0), atol=1e-6)


def test_recombine_mixed_ndim_stems():
    n = 48000
    vocal = np.sin(np.linspace(0, 2 * np.pi * 220, n)).astype(np.float32) * 0.4  # mono
    instr = np.stack([np.zeros(n), np.zeros(n)]).astype(np.float32)  # stereo
    ctx = StemContext(vocal_stem=vocal, instrumental_stem=instr, sample_rate=48000)
    mix = ctx.recombine()
    assert mix.ndim == 2
    np.testing.assert_allclose(mix[0], vocal, atol=1e-6)
    np.testing.assert_allclose(mix[1], vocal, atol=1e-6)


def test_as_dict_is_lightweight_and_faithful():
    vocal, instr = _mk_stereo()
    ctx = StemContext(
        vocal_stem=vocal,
        instrumental_stem=instr,
        sample_rate=48000,
        panns_singing=0.65,
        separation_model="demucs_v4_htdemucs",
        applied_stages=["dfn", "kim_inst"],
        witness_reports={"kim_inst": {"applied": True, "model": "kim_inst"}},
    )
    summary = ctx.as_dict()
    assert "vocal_stem" not in summary  # keine Audio-Arrays
    assert summary["panns_singing"] == 0.65
    assert summary["separation_model"] == "demucs_v4_htdemucs"
    assert summary["applied_stages"] == ["dfn", "kim_inst"]
    assert summary["vocal_shape"] == (2, 48000)


def test_stems_not_modified_by_context():
    vocal, instr = _mk_stereo()
    vocal_before = vocal.copy()
    instr_before = instr.copy()
    ctx = StemContext(vocal_stem=vocal, instrumental_stem=instr, sample_rate=48000)
    ctx.recombine()
    np.testing.assert_array_equal(ctx.vocal_stem, vocal_before)
    np.testing.assert_array_equal(ctx.instrumental_stem, instr_before)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
