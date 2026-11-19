from __future__ import annotations

import numpy as np
import pytest

from backend.core.phases.phase_01_click_removal import ClickRemovalPhase


@pytest.mark.unit
def test_stereo_sparse_patch_keeps_clean_counter_channel_untouched(monkeypatch) -> None:
    sr = 48_000
    n = sr // 2
    left = np.full(n, 0.1, dtype=np.float32)
    right = np.full(n, 0.1, dtype=np.float32)
    left[1000:1002] = 0.95
    stereo = np.stack([left, right], axis=0)

    phase = ClickRemovalPhase()

    monkeypatch.setattr(phase, "_detect_clicks_multiscale", lambda audio, thresholds: [(1000, 1001)])
    monkeypatch.setattr(
        phase,
        "_classify_clicks",
        lambda audio, click_candidates, preserve_transients, thresholds: [
            {"type": "digital", "start": 1000, "end": 1001, "severity": 0.2}
        ],
    )

    result = phase.process(stereo.copy(), sample_rate=48_000, material_type="vinyl", quality_mode="fast")

    repaired = result.audio
    assert repaired.shape == stereo.shape
    assert np.allclose(repaired[1], right, atol=1e-7)
    assert not np.allclose(repaired[0, 1000:1002], left[1000:1002])
    assert np.allclose(repaired[0, :996], left[:996], atol=1e-7)
    assert np.allclose(repaired[0, 1006:], left[1006:], atol=1e-7)
