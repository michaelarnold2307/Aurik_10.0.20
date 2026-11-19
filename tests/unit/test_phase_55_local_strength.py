from __future__ import annotations

import numpy as np
import pytest

from backend.core.phases.phase_55_diffusion_inpainting import DiffusionInpaintingPhase


@pytest.mark.unit
def test_local_strength_respects_protected_zone_cap() -> None:
    sr = 48_000
    audio = np.full(sr, 0.5, dtype=np.float32)
    start = int(0.50 * sr)
    end = int(0.55 * sr)
    audio[start:end] = 0.0

    local_strength = DiffusionInpaintingPhase._compute_inpainting_local_strength(
        audio,
        start,
        end,
        sr,
        base_strength=0.8,
        protected_zones=[(0.45, 0.60, 0.20)],
    )

    assert 0.0 <= local_strength <= 0.20


def test_local_strength_is_higher_for_severe_gap_than_for_mild_gap() -> None:
    sr = 48_000
    audio = np.full(sr, 0.5, dtype=np.float32)
    start = int(0.50 * sr)
    end = int(0.55 * sr)

    mild = audio.copy()
    mild[start:end] = 0.40
    severe = audio.copy()
    severe[start:end] = 0.0

    mild_strength = DiffusionInpaintingPhase._compute_inpainting_local_strength(
        mild,
        start,
        end,
        sr,
        base_strength=0.8,
        protected_zones=None,
    )
    severe_strength = DiffusionInpaintingPhase._compute_inpainting_local_strength(
        severe,
        start,
        end,
        sr,
        base_strength=0.8,
        protected_zones=None,
    )

    assert severe_strength > mild_strength
    assert severe_strength <= 0.8
