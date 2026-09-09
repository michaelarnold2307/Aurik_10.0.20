"""§v10.740 (2026-09-09): collect_results Kanal-Layout-Normalisierung.

Befund Lauf 4: Die Pipeline liefert teils (2, N) channels-first — shape[1]
war dann die Chunklänge statt der Kanalzahl → 56.7-TiB-Allokation
(total_samples, chunk_len) und MemoryError im Final-Assembly.
"""

import numpy as np

from backend.core.chunked_streaming import ChunkedPipeline, ChunkResult


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
