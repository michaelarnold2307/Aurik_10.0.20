from __future__ import annotations

import numpy as np
import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_66_stem_targeted_nr import StemTargetedNRPhase


def test_phase66_rejects_non_48000_sample_rate() -> None:
    phase = StemTargetedNRPhase()
    audio = np.zeros(48_000, dtype=np.float32)

    with pytest.raises(AssertionError):
        phase.process(audio, sample_rate=44_100, material_type=MaterialType.VINYL, panns_singing=0.9)


def test_phase66_passthrough_is_nan_safe_and_clipped_when_gate_inactive() -> None:
    phase = StemTargetedNRPhase()
    audio = np.array([0.0, np.nan, 2.0, -2.0], dtype=np.float32)

    result = phase.process(audio, sample_rate=48_000, material_type="vinyl", panns_singing=0.1)

    assert result.success is True
    assert result.metadata["rollback_reason"].startswith("panns_singing=")
    assert not np.isnan(result.audio).any()
    assert float(np.max(np.abs(result.audio))) <= 1.0
