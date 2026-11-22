"""§v10.740 (2026-09-09): collect_results Kanal-Layout-Normalisierung.

Befund Lauf 4: Die Pipeline liefert teils (2, N) channels-first — shape[1]
war dann die Chunklänge statt der Kanalzahl → 56.7-TiB-Allokation
(total_samples, chunk_len) und MemoryError im Final-Assembly.
"""

import numpy as np
import pytest

from backend.core.chunked_streaming import ChunkDeltaPatch, ChunkedPipeline, ChunkResult, select_chunk_duration_s


def test_collect_channels_first():
    """(2, N) channels-first — der Lauf-4-Crash-Fall."""
    mgr = ChunkedPipeline(overlap_s=0.05)
    results = [
        ChunkResult(np.zeros((2, 48000), np.float32), 48000, 0, 0, 48000),
        ChunkResult(np.zeros((2, 48000), np.float32), 48000, 1, 48000, 96000),
    ]
    out = mgr.collect_results(results, 48000)
    assert out.shape == (96000, 2)


def test_collect_channels_last():
    """(N, 2) channels-last bleibt unverändert."""
    mgr = ChunkedPipeline(overlap_s=0.05)
    results = [ChunkResult(np.zeros((48000, 2), np.float32), 48000, 0, 0, 48000)]
    out = mgr.collect_results(results, 48000)
    assert out.shape == (48000, 2)


def test_collect_mono():
    """Mono (N,) bleibt mono."""
    mgr = ChunkedPipeline(overlap_s=0.05)
    results = [ChunkResult(np.zeros(48000, np.float32), 48000, 0, 0, 48000)]
    out = mgr.collect_results(results, 48000)
    assert out.shape == (48000,)


def test_chunk_delta_rollback_uses_same_crossfade_and_preserves_unpatched_samples():
    mgr = ChunkedPipeline(overlap_s=0.3)
    results = [
        ChunkResult(np.full(6, 0.1, np.float32), 10, 0, 0, 6),
        ChunkResult(np.full(6, 0.05, np.float32), 10, 1, 3, 9),
    ]
    output = mgr.collect_results(results, 10)
    patches = [
        ChunkDeltaPatch(0, np.array([3, 4, 5]), np.full(3, -0.1, np.float32)),
        ChunkDeltaPatch(1, np.array([0, 1, 2]), np.full(3, -0.05, np.float32)),
    ]

    candidate = mgr.apply_chunk_deltas(output, results, patches, 10)
    phase_off_results = [
        ChunkResult(results[0].audio.copy(), 10, 0, 0, 6),
        ChunkResult(results[1].audio.copy(), 10, 1, 3, 9),
    ]
    phase_off_results[0].audio[3:6] += -0.1
    phase_off_results[1].audio[:3] += -0.05
    expected = mgr.collect_results(phase_off_results, 10)

    np.testing.assert_array_equal(candidate[:3], output[:3])
    np.testing.assert_allclose(candidate[3:6], 0.0, atol=1e-7)
    np.testing.assert_array_equal(candidate[6:], output[6:])
    np.testing.assert_allclose(candidate, expected, atol=1e-7)


def test_chunk_delta_rollback_normalizes_channels_first_patch_layout():
    mgr = ChunkedPipeline(overlap_s=0.0)
    audio = np.full((2, 8), 0.25, np.float32)
    result = ChunkResult(audio, 10, 0, 0, 8)
    output = mgr.collect_results([result], 10)
    patch = ChunkDeltaPatch(0, np.array([3, 4]), np.full((2, 2), -0.25, np.float32))

    candidate = mgr.apply_chunk_deltas(output, [result], [patch], 10)

    assert candidate.shape == (8, 2)
    np.testing.assert_array_equal(candidate[[0, 1, 2, 5, 6, 7]], output[[0, 1, 2, 5, 6, 7]])
    np.testing.assert_allclose(candidate[3:5], 0.0, atol=1e-7)


def test_chunk_delta_rollback_rejects_non_finite_delta():
    mgr = ChunkedPipeline(overlap_s=0.0)
    result = ChunkResult(np.zeros(8, np.float32), 10, 0, 0, 8)
    patch = ChunkDeltaPatch(0, np.array([3]), np.array([np.nan], np.float32))

    with pytest.raises(ValueError, match="NaN/Inf"):
        mgr.apply_chunk_deltas(np.zeros(8, np.float32), [result], [patch], 10)


def test_r16_select_default_heuristic():
    """§PERF-R16 + §P3-1: ohne Override bleibt die Heuristik (60s/120s) unverändert."""
    assert select_chunk_duration_s(200.0) == (60.0, False)
    assert select_chunk_duration_s(300.0) == (60.0, False)
    assert select_chunk_duration_s(300.1) == (120.0, False)
    assert select_chunk_duration_s(225.3, "") == (60.0, False)
    assert select_chunk_duration_s(225.3, "   ") == (60.0, False)


def test_r16_select_override_applied():
    """§PERF-R16: gültiger AURIK_CHUNK_S-Override greift (inkl. Whitespace)."""
    assert select_chunk_duration_s(225.3, "120") == (120.0, True)
    assert select_chunk_duration_s(225.3, " 120.0 ") == (120.0, True)
    assert select_chunk_duration_s(225.3, "60") == (60.0, True)
    assert select_chunk_duration_s(400.0, "120") == (120.0, True)
    # Grenzen inklusive
    assert select_chunk_duration_s(225.3, "10") == (10.0, True)
    assert select_chunk_duration_s(225.3, "600") == (600.0, True)


def test_r16_select_override_invalid_falls_back():
    """§PERF-R16: ungültige Werte fallen auf die Heuristik zurück."""
    assert select_chunk_duration_s(225.3, "abc") == (60.0, False)
    assert select_chunk_duration_s(225.3, "9.9") == (60.0, False)
    assert select_chunk_duration_s(225.3, "600.1") == (60.0, False)
    assert select_chunk_duration_s(400.0, "abc") == (120.0, False)
    assert select_chunk_duration_s(400.0, "1200") == (120.0, False)
