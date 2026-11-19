from __future__ import annotations

import numpy as np

from backend.core.defect_scanner import DefectScanner, DefectType


def test_scan_accepts_mono_column_layout_without_stereo_index_error() -> None:
    sr = 48000
    t = np.linspace(0.0, 1.0, sr, endpoint=False, dtype=np.float32)
    mono = 0.1 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    mono_col = mono.reshape(-1, 1)

    scanner = DefectScanner(sample_rate=sr)
    result = scanner.scan(mono_col, sample_rate=sr)

    assert result is not None
    assert DefectType.STEREO_IMBALANCE in result.scores
    assert result.scores[DefectType.STEREO_IMBALANCE].severity == 0.0
