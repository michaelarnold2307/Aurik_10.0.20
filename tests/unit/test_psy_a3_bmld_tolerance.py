"""§SOTA-PSY-A3 (2026-09-15) — BMLD-dynamische Freisetzungs-Toleranz: Tests.

Deckt ``bmld_tolerance_factor`` (binaural_masking.py) + die phase_33-Verdrahtung:
- Faktor ist linear in release_db, Nie < 1,0 (Never-worsen), Cap bei +10 %
- NaN/negativ-sicher (1,0)
- phase_33 meldet den Faktor im Metadatum (≥ 1,0 und ≤ 1,10)

Autor: Aurik Testing Team
"""

import numpy as np

from backend.core.dsp.binaural_masking import (
    BMLD_MAX_TOLERANCE_FACTOR,
    BMLD_TOLERANCE_RELEASE_CAP_DB,
    bmld_tolerance_factor,
)


class TestBmldToleranceFactor:
    def test_zero_release_is_neutral(self):
        assert bmld_tolerance_factor(0.0) == 1.0

    def test_full_release_hits_cap(self):
        assert bmld_tolerance_factor(BMLD_TOLERANCE_RELEASE_CAP_DB) == BMLD_MAX_TOLERANCE_FACTOR
        assert bmld_tolerance_factor(20.0) == BMLD_MAX_TOLERANCE_FACTOR

    def test_linear_midpoint(self):
        mid = bmld_tolerance_factor(BMLD_TOLERANCE_RELEASE_CAP_DB / 2.0)
        assert np.isclose(mid, 1.0 + (BMLD_MAX_TOLERANCE_FACTOR - 1.0) * 0.5, atol=1e-6)

    def test_never_below_one(self):
        assert bmld_tolerance_factor(-3.0) == 1.0
        assert bmld_tolerance_factor(float("nan")) == 1.0

    def test_bounded_above(self):
        assert bmld_tolerance_factor(100.0) <= BMLD_MAX_TOLERANCE_FACTOR


class TestPhase33DynamicTolerance:
    def test_phase33_reports_tolerance_factor(self):
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_33_stereo_width_limiter import StereoWidthLimiterPhaseV2

        t = np.linspace(0, 48000 / 48000, 48000, endpoint=False, dtype=np.float32)
        left = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        right = (0.28 * np.sin(2 * np.pi * 445 * t)).astype(np.float32)
        stereo = np.stack([left, right], axis=0).astype(np.float32)  # channels-first (2, N)

        res = StereoWidthLimiterPhaseV2().process(stereo, sample_rate=48000, material_type=MaterialType.VINYL)
        factor = res.metadata.get("binaural_masking_release_tolerance_factor")
        assert factor is not None, "phase_33 meldet keinen Toleranz-Faktor"
        assert 1.0 <= float(factor) <= BMLD_MAX_TOLERANCE_FACTOR
        assert np.isfinite(np.asarray(res.audio)).all()
