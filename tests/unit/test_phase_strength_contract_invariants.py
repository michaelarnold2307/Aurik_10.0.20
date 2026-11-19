"""Invarianz-Tests fuer pilotierte Phase-Strength-Contract-Integrationen."""

import numpy as np
import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_10_compression import CompressionPhase
from backend.core.phases.phase_11_limiting import LimitingPhase
from backend.core.phases.phase_16_final_eq import FinalEQ
from backend.core.phases.phase_26_dynamic_range_expansion import DynamicRangeExpansion
from backend.core.phases.phase_32_mono_to_stereo import MonoToStereoPhaseV2
from backend.core.phases.phase_33_stereo_width_limiter import StereoWidthLimiterPhaseV2
from backend.core.phases.phase_34_mid_side_processing import MidSideProcessing
from backend.core.phases.phase_47_truepeak_limiter import TruePeakLimiterPhase
from backend.core.phases.phase_48_stereo_width_enhancer import StereoWidthEnhancerPhase
from backend.core.phases.phase_65_vocal_naturalness_restoration import VocalNaturalnessRestorationPhase


@pytest.mark.parametrize(
    ("phase", "audio", "material"),
    [
        (CompressionPhase(), np.zeros((48000, 2), dtype=np.float32), MaterialType.VINYL),
        (LimitingPhase(), np.zeros((48000, 2), dtype=np.float32), MaterialType.VINYL),
        (FinalEQ(), np.zeros(48000, dtype=np.float32), MaterialType.CD_DIGITAL),
        (DynamicRangeExpansion(), np.zeros((48000, 2), dtype=np.float32), MaterialType.CD_DIGITAL),
        (MonoToStereoPhaseV2(), np.zeros((48000, 2), dtype=np.float32), MaterialType.VINYL),
        (StereoWidthLimiterPhaseV2(), np.zeros((48000, 2), dtype=np.float32), MaterialType.VINYL),
        (MidSideProcessing(), np.zeros((48000, 2), dtype=np.float32), MaterialType.VINYL),
        (TruePeakLimiterPhase(), np.zeros(48000, dtype=np.float32), MaterialType.CD_DIGITAL),
        (StereoWidthEnhancerPhase(), np.zeros((48000, 2), dtype=np.float32), MaterialType.VINYL),
    ],
)
def test_migrated_phases_preserve_zero_strength_skip_metadata(phase, audio, material):
    """Alle pilotierten Phasen muessen den zentralen Skip-Zero-Strength-Vertrag einhalten."""
    result = phase.process(
        audio,
        sample_rate=48000,
        material=material,
        strength=0.0,
        phase_locality_factor=0.1,
    )

    assert result.success
    assert result.audio.shape == audio.shape
    assert np.allclose(result.audio, audio)
    assert abs(float(result.metadata.get("phase_locality_factor", 0.0)) - 0.35) < 1e-6
    assert abs(float(result.metadata.get("effective_strength", -1.0))) < 1e-6


def test_phase65_uses_contract_locality_in_metadata_even_when_vocal_gate_skips():
    """Phase 65 hat eigene Gates, muss aber den Contract fuer locality-Metadaten nutzen."""
    phase = VocalNaturalnessRestorationPhase()
    audio = np.zeros(48000, dtype=np.float32)

    result = phase.process(
        audio,
        sample_rate=48000,
        strength=0.9,
        phase_locality_factor=0.1,
        panns_singing=0.0,
    )

    assert result.success
    assert abs(float(result.metadata.get("phase_locality_factor", 0.0)) - 0.35) < 1e-6
    assert result.metadata.get("algorithm") == "skipped_no_vocal"
