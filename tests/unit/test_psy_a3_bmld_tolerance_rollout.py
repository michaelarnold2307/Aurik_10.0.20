"""§SOTA-PSY-A3 (2026-09-15) — BMLD-Toleranz-Rollout 13/15/46/48: Tests.

Deckt die dynamische Freisetzungs-Toleranz in den Stereo-Phasen
(Muster phase_33): jede Phase meldet
``binaural_masking_release_tolerance_factor`` ≥ 1,0 und ≤ 1,10.

Autor: Aurik Testing Team
"""

import numpy as np

from backend.core.dsp.binaural_masking import BMLD_MAX_TOLERANCE_FACTOR


def _stereo(n: int = 48000) -> np.ndarray:
    t = np.linspace(0, n / 48000, n, endpoint=False, dtype=np.float32)
    left = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    right = (np.roll(left, 3) * 0.9).astype(np.float32)  # korreliert (IACC hoch)
    return np.stack([left, right], axis=0).astype(np.float32)  # channels-first (2, N)


def test_phases_13_15_46_48_report_tolerance_factor():
    from backend.core.defect_scanner import MaterialType

    cases = [
        ("phase_13_stereo_enhancement", "StereoEnhancementPhaseV2"),
        ("phase_15_stereo_balance", "StereoBalancePhaseV2"),
        ("phase_46_spatial_enhancement", "SpatialEnhancementPhase"),
        ("phase_48_stereo_width_enhancer", "StereoWidthEnhancerPhase"),
    ]
    import importlib

    for mod, cls in cases:
        m = importlib.import_module(f"backend.core.phases.{mod}")
        ph = getattr(m, cls)()
        res = ph.process(_stereo(), sample_rate=48000, material_type=MaterialType.VINYL)
        factor = res.metadata.get("binaural_masking_release_tolerance_factor")
        assert factor is not None, f"{mod} meldet keinen Toleranz-Faktor"
        assert 1.0 <= float(factor) <= BMLD_MAX_TOLERANCE_FACTOR
        assert np.isfinite(np.asarray(res.audio)).all()
