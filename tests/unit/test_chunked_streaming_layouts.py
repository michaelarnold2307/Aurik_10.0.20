"""§v10.740 (2026-09-09): collect_results Kanal-Layout-Normalisierung.

Befund Lauf 4: Die Pipeline liefert teils (2, N) channels-first — shape[1]
war dann die Chunklänge statt der Kanalzahl → 56.7-TiB-Allokation
(total_samples, chunk_len) und MemoryError im Final-Assembly.
"""

import numpy as np

from backend.core.chunked_streaming import ChunkedPipeline, ChunkResult, select_chunk_duration_s


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


def test_r16_select_default_heuristic():
    """§PERF-R16: ohne Override bleibt die Heuristik (30s/60s) unverändert."""
    assert select_chunk_duration_s(200.0) == (30.0, False)
    assert select_chunk_duration_s(300.0) == (30.0, False)
    assert select_chunk_duration_s(300.1) == (60.0, False)
    assert select_chunk_duration_s(225.3, "") == (30.0, False)
    assert select_chunk_duration_s(225.3, "   ") == (30.0, False)


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
    assert select_chunk_duration_s(225.3, "abc") == (30.0, False)
    assert select_chunk_duration_s(225.3, "9.9") == (30.0, False)
    assert select_chunk_duration_s(225.3, "600.1") == (30.0, False)
    assert select_chunk_duration_s(400.0, "abc") == (60.0, False)
    assert select_chunk_duration_s(400.0, "1200") == (60.0, False)
