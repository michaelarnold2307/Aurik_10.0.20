from __future__ import annotations

import pytest

"""Unit-Tests für DR-Ceiling in Phase 26 + BW-Guard Integration (§6.2b, §6.2c)."""


import numpy as np

from backend.core.carrier_transfer_characteristics import (
    get_bw_ceiling_hz,
    get_dr_ceiling_db,
)


@pytest.mark.unit
class TestDRCeilingValues:
    """§6.2b — DR-Ceiling Plausibilitätschecks."""

    def test_shellac_ceiling_45(self):
        assert get_dr_ceiling_db("shellac") == 45

    def test_vinyl_ceiling_70(self):
        assert get_dr_ceiling_db("vinyl") == 70

    def test_cd_ceiling_96(self):
        assert get_dr_ceiling_db("cd_digital") == 96

    def test_analog_lower_than_digital(self):
        """Analoge Materialien haben niedrigere DR-Decken als digitale."""
        for analog in ["vinyl", "shellac", "cassette", "tape"]:
            assert get_dr_ceiling_db(analog) < get_dr_ceiling_db("cd_digital")


class TestBWCeilingValues:
    """§6.2c — BW-Ceiling Plausibilitätschecks."""

    def test_shellac_ceiling_8000(self):
        assert get_bw_ceiling_hz("shellac") == 8000

    def test_wax_cylinder_ceiling_5000(self):
        assert get_bw_ceiling_hz("wax_cylinder") == 5000

    def test_vinyl_ceiling_16000(self):
        assert get_bw_ceiling_hz("vinyl") == 16000

    def test_cd_digital_ceiling_22050(self):
        assert get_bw_ceiling_hz("cd_digital") == 22050

    def test_analog_lower_than_cd(self):
        for analog in ["shellac", "wax_cylinder", "cassette"]:
            assert get_bw_ceiling_hz(analog) < get_bw_ceiling_hz("cd_digital")


class TestPhase26DRCeiling:
    """§6.2b — DR-Ceiling-Enforcement in phase_26."""

    def test_phase26_has_dr_ceiling_metadata(self):
        """phase_26 Ergebnis enthält dr_ceiling_capped-Feld."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_26_dynamic_range_expansion import DynamicRangeExpansion

        phase = DynamicRangeExpansion()
        rng = np.random.default_rng(42)
        # Signal mit moderater Dynamik
        audio = rng.normal(0, 0.3, 48000 * 2).astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)

        result = phase.process(audio, sample_rate=48000, material=MaterialType.VINYL)
        assert result.success
        assert "dr_ceiling_capped" in result.metadata

    def test_phase26_shellac_low_ceiling(self):
        """Shellac DR-Ceiling (45 dB) wird bei starker Expansion respektiert."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_26_dynamic_range_expansion import DynamicRangeExpansion

        phase = DynamicRangeExpansion()
        rng = np.random.default_rng(42)
        audio = rng.normal(0, 0.3, 48000 * 2).astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)

        result = phase.process(
            audio,
            sample_rate=48000,
            material=MaterialType.SHELLAC,
            strength=1.0,
        )
        assert result.success
        # DR After sollte ≤ Shellac-Ceiling (45 dB) + Toleranz sein
        dr_after = result.metadata.get("dynamic_range_after_db", 0)
        assert dr_after <= 55  # 45 + 10 dB Toleranz (Messvarianz)
