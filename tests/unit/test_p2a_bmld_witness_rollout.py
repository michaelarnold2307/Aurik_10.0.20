"""§SOTA-PSY-A3 (BMLD-Witness-Rollout) — Regressionsschutz.

Deckt den PSY-A3-Ausbau (2026-09-15): Die binaurale Maskierungs-Freisetzung
(BMLD/EC) wird in den Stereo-Phasen 13, 15, 46 und 48 als Witness-Metadatum
geführt — Muster phase_33/34, ZEUGE nicht Richter (Hörordnung §8a).

Autor: Aurik Testing Team
"""

import numpy as np
import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.dsp.binaural_masking import binaural_masking_advantage


def _make_stereo(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = rng.normal(0, 0.1, n).astype(np.float32)
    right = 0.7 * left + rng.normal(0, 0.05, n).astype(np.float32)
    return np.column_stack([left, right])


def _assert_witness_shape(bml: dict[str, float]) -> None:
    assert set(bml.keys()) == {"release_db", "nr_floor_release_db", "ec_gain_db"}
    assert 0.0 <= bml["release_db"] <= 15.0
    assert 0.0 <= bml["nr_floor_release_db"] <= 8.0
    assert bml["ec_gain_db"] >= 0.0
    for v in bml.values():
        assert np.isfinite(v)


def _oracle(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    res = binaural_masking_advantage(audio, sample_rate)
    return {
        "release_db": res.release_db,
        "nr_floor_release_db": res.nr_floor_release_db,
        "ec_gain_db": res.ec_gain_db,
    }


class TestPhase13BMLDWitness:
    def test_stereo_witness_recorded(self):
        from backend.core.phases.phase_13_stereo_enhancement import StereoEnhancementPhaseV2

        audio = _make_stereo(48000, 1)
        res = StereoEnhancementPhaseV2().process(audio, sample_rate=48000, material_type=MaterialType.VINYL)
        bml = res.metadata.get("binaural_masking_advantage", {})
        _assert_witness_shape(bml)
        assert bml == pytest.approx(_oracle(audio, 48000), abs=1e-9)

    def test_mono_has_no_witness(self):
        from backend.core.phases.phase_13_stereo_enhancement import StereoEnhancementPhaseV2

        rng = np.random.default_rng(2)
        mono = rng.normal(0, 0.1, 48000).astype(np.float32)
        res = StereoEnhancementPhaseV2().process(mono, sample_rate=48000, material_type=MaterialType.VINYL)
        assert res.metadata.get("binaural_masking_advantage", {}) == {}


class TestPhase15BMLDWitness:
    def test_stereo_witness_recorded(self):
        from backend.core.phases.phase_15_stereo_balance import StereoBalancePhaseV2

        audio = _make_stereo(48000, 3)
        res = StereoBalancePhaseV2().process(audio, sample_rate=48000, material_type=MaterialType.VINYL)
        bml = res.metadata.get("binaural_masking_advantage", {})
        _assert_witness_shape(bml)
        assert bml == pytest.approx(_oracle(audio, 48000), abs=1e-9)


class TestPhase46BMLDWitness:
    def test_stereo_witness_recorded(self):
        from backend.core.phases.phase_46_spatial_enhancement import SpatialEnhancementPhase

        audio = _make_stereo(48000, 5)
        res = SpatialEnhancementPhase().process(audio, sample_rate=48000, material_type=MaterialType.VINYL)
        bml = res.metadata.get("binaural_masking_advantage", {})
        _assert_witness_shape(bml)
        assert bml == pytest.approx(_oracle(audio, 48000), abs=1e-9)


class TestPhase48BMLDWitness:
    def test_stereo_witness_recorded(self):
        from backend.core.phases.phase_48_stereo_width_enhancer import StereoWidthEnhancerPhase

        audio = _make_stereo(48000, 7)
        res = StereoWidthEnhancerPhase().process(audio, sample_rate=48000, material_type=MaterialType.VINYL)
        bml = res.metadata.get("binaural_masking_advantage", {})
        _assert_witness_shape(bml)
        assert bml == pytest.approx(_oracle(audio, 48000), abs=1e-9)
