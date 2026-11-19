from __future__ import annotations

"""§6.2d Bidirektionale Sync-Invariante: BW/DR-Ceiling-Dicts ↔ CARRIER_TRANSFER_CHARACTERISTICS.

Ensures that:
1. MATERIAL_BW_CEILING_HZ and MATERIAL_DR_CEILING_DB are derived from CARRIER_TRANSFER_CHARACTERISTICS.
2. All 15+ supported materials are present.
3. No material is missing from the derived dicts.
4. Values are physically plausible (BW > 0, DR > 0).
"""


import pytest

from backend.core.carrier_transfer_characteristics import (
    CARRIER_TRANSFER_CHARACTERISTICS,
    MATERIAL_BW_CEILING_HZ,
    MATERIAL_DR_CEILING_DB,
    get_bw_ceiling_hz,
    get_dr_ceiling_db,
)

# §6.1 — 15 Materialtypen (Minimum)
_REQUIRED_MATERIALS = {
    "wax_cylinder",
    "shellac",
    "vinyl",
    "reel_tape",
    "tape",
    "cassette",
    "cd_digital",
    "dat",
    "minidisc",
    "mp3_low",
    "mp3_high",
    "aac",
    "streaming",
    "wire_recording",
    "unknown",
}


@pytest.mark.unit
class TestMaterialCeilingSync:
    """§6.2d bidirectional sync tests."""

    def test_ctc_has_all_required_materials(self):
        missing = _REQUIRED_MATERIALS - set(CARRIER_TRANSFER_CHARACTERISTICS.keys())
        assert not missing, f"CARRIER_TRANSFER_CHARACTERISTICS missing materials: {missing}"

    def test_bw_ceiling_keys_match_ctc(self):
        ctc_keys = set(CARRIER_TRANSFER_CHARACTERISTICS.keys())
        bw_keys = set(MATERIAL_BW_CEILING_HZ.keys())
        assert bw_keys == ctc_keys, f"BW keys mismatch: extra={bw_keys - ctc_keys}, missing={ctc_keys - bw_keys}"

    def test_dr_ceiling_keys_match_ctc(self):
        ctc_keys = set(CARRIER_TRANSFER_CHARACTERISTICS.keys())
        dr_keys = set(MATERIAL_DR_CEILING_DB.keys())
        assert dr_keys == ctc_keys, f"DR keys mismatch: extra={dr_keys - ctc_keys}, missing={ctc_keys - dr_keys}"

    def test_bw_values_consistent_with_ctc(self):
        for mat, ctc_val in CARRIER_TRANSFER_CHARACTERISTICS.items():
            bw_from_ctc = ctc_val[0]
            bw_from_dict = MATERIAL_BW_CEILING_HZ.get(mat)
            assert bw_from_dict == bw_from_ctc, f"{mat}: BW dict={bw_from_dict} but CTC[0]={bw_from_ctc}"

    def test_dr_values_consistent_with_ctc(self):
        for mat, ctc_val in CARRIER_TRANSFER_CHARACTERISTICS.items():
            dr_from_ctc = ctc_val[3]
            dr_from_dict = MATERIAL_DR_CEILING_DB.get(mat)
            assert dr_from_dict == dr_from_ctc, f"{mat}: DR dict={dr_from_dict} but CTC[3]={dr_from_ctc}"

    def test_bw_values_physically_plausible(self):
        for mat, bw in MATERIAL_BW_CEILING_HZ.items():
            assert 1000 <= bw <= 48000, f"{mat}: BW={bw} Hz out of [1000, 48000] range"

    def test_dr_values_physically_plausible(self):
        for mat, dr in MATERIAL_DR_CEILING_DB.items():
            assert 20 <= dr <= 150, f"{mat}: DR={dr} dB out of [20, 150] range"

    def test_helper_functions_consistent(self):
        for mat in CARRIER_TRANSFER_CHARACTERISTICS:
            assert get_bw_ceiling_hz(mat) == MATERIAL_BW_CEILING_HZ[mat]
            assert get_dr_ceiling_db(mat) == MATERIAL_DR_CEILING_DB[mat]

    def test_unknown_fallback(self):
        assert get_bw_ceiling_hz("nonexistent_material") == 20000
        assert get_dr_ceiling_db("nonexistent_material") == 70
