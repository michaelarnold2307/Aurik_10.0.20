"""§PERF-R8: pyin_viterbi_fast == librosa.pyin (bit-identisch) + Viterbi-Semantik."""

import numpy as np
import pytest

librosa = pytest.importorskip("librosa")

from backend.core.dsp.pyin_viterbi_fast import pyin_fast, viterbi_vectorized


@pytest.mark.parametrize("n", [8192, 48000])
def test_pyin_fast_bit_identical_to_librosa(n: int) -> None:
    """pyin_fast liefert identische f0/voiced-Probs wie librosa.pyin (0.11.0)."""
    rng = np.random.RandomState(3)
    y = (rng.randn(n) * 0.15).astype(np.float32)
    y += (0.3 * np.sin(2 * np.pi * 220 * np.arange(n, dtype=np.float64) / 48000)).astype(np.float32)
    f0_a, vf_a, vp_a = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=48000,
        frame_length=2048,
        hop_length=375,
        fill_na=0.0,
    )
    f0_b, vf_b, vp_b = pyin_fast(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=48000,
        frame_length=2048,
        hop_length=375,
        fill_na=0.0,
    )
    assert np.array_equal(f0_a, f0_b)
    assert np.array_equal(vf_a, vf_b)
    assert np.array_equal(vp_a, vp_b)


def test_viterbi_vectorized_bit_identical() -> None:
    """Vektorisiertes Viterbi == librosa.sequence.viterbi auf strukturierten Feeds."""
    rng = np.random.RandomState(5)
    m, t = 97, 64
    prob = rng.rand(m, t).astype(np.float64)
    prob[prob < 0.3] = 0.0
    prob /= prob.sum(axis=0, keepdims=True) + 1e-12
    transition = rng.rand(m, m)
    transition /= transition.sum(axis=1, keepdims=True)
    p_init = rng.rand(m)
    p_init /= p_init.sum()
    a = librosa.sequence.viterbi(prob, transition, p_init=p_init)
    b = viterbi_vectorized(prob, transition, p_init=p_init)
    assert np.array_equal(a, b)
    a2, lp_a = librosa.sequence.viterbi(prob, transition, p_init=p_init, return_logp=True)
    b2, lp_b = viterbi_vectorized(prob, transition, p_init=p_init, return_logp=True)
    assert np.array_equal(a2, b2)
    assert np.array_equal(lp_a, lp_b)


def test_viterbi_vectorized_multichannel() -> None:
    """Mehrkanal-Form [2, m, t] wie im pyin-Pfad dekodiert identisch."""
    rng = np.random.RandomState(7)
    m, t = 51, 48
    prob = rng.rand(2, m, t)
    prob[prob < 0.4] = 0.0
    s = prob.sum(axis=1, keepdims=True)
    prob = prob / (s + 1e-12)
    transition = rng.rand(m, m)
    transition /= transition.sum(axis=1, keepdims=True)
    a = librosa.sequence.viterbi(prob, transition)
    b = viterbi_vectorized(prob, transition)
    assert np.array_equal(a, b)
