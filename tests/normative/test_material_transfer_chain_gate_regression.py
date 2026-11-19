from __future__ import annotations

"""[RELEASE_MUST] Material × Transfer-Chain Gate-Regressionen.

Ziel:
- Material- und transferkettenbezogene Autosetup-Logik in UV3 robust absichern.
- Regressionsschutz fuer Mehrgenerationsmaterial und konfliktkritische Phase-Caps.
"""


import pytest

from backend.core.unified_restorer_v3 import UnifiedRestorerV3


def _base_profile(material: str = "vinyl") -> dict:
    return {
        "material": material,
        "restorability_tier": "fair",
        "family_scalars": {
            "denoise": 1.0,
            "reconstruction": 1.0,
            "transient": 1.0,
            "dynamics_eq": 1.0,
        },
        "source_fidelity_transfer_chain": [],
    }


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_extract_transfer_chain_normalizes_arrow_string() -> None:
    chain = UnifiedRestorerV3._extract_transfer_chain_from_forensics(
        {"transfer_chain": "shellac -> reel_tape → cassette > mp3_low"}
    )
    assert chain == ["shellac", "reel_tape", "cassette", "mp3_low"]


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_extract_transfer_chain_moves_terminal_codec_after_physical_stage() -> None:
    string_chain = UnifiedRestorerV3._extract_transfer_chain_from_forensics({"transfer_chain": "mp3_low -> cassette"})
    list_chain = UnifiedRestorerV3._extract_transfer_chain_from_forensics(["mp3_low", "cassette"])

    assert string_chain == ["cassette", "mp3_low"]
    assert list_chain == ["cassette", "mp3_low"]


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_inferred_cassette_ancestor_preserves_terminal_mp3_primary_chain_direction() -> None:
    chain = UnifiedRestorerV3._with_inferred_analog_ancestor_chain(
        [],
        inferred_ancestor="cassette",
        primary_material="mp3_low",
    )

    assert chain == ["cassette", "mp3_low"]


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_extract_transfer_chain_from_object_list() -> None:
    class _Forensics:
        transfer_chain = ["Vinyl", "Tape", "MP3_LOW"]

    chain = UnifiedRestorerV3._extract_transfer_chain_from_forensics(_Forensics())
    assert chain == ["vinyl", "tape", "mp3_low"]


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_autosetup_policy_chain_depth_4_biases_family_scalars() -> None:
    profile = _base_profile("vinyl")

    out = UnifiedRestorerV3._apply_song_autosetup_policy(
        profile,
        defect_scores={},
        transfer_chain=["shellac", "reel_tape", "cassette", "mp3_low"],
        max_defect_severity=0.4,
    )

    fam = out["family_scalars"]
    assert fam["reconstruction"] > 1.0
    assert fam["denoise"] > 1.0
    assert fam["transient"] < 1.0
    assert fam["dynamics_eq"] < 1.0

    policy = out["strict_conflict_policy"]
    assert policy["enabled"] is True
    assert policy["transfer_chain_depth"] == 4


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_autosetup_policy_temporal_defects_shift_transient_vs_reconstruction() -> None:
    profile = _base_profile("tape")

    out = UnifiedRestorerV3._apply_song_autosetup_policy(
        profile,
        defect_scores={"wow": 0.55, "flutter": 0.40, "dropouts": 0.30},
        transfer_chain=["tape", "mp3_low"],
        max_defect_severity=0.6,
    )

    fam = out["family_scalars"]
    assert fam["transient"] < 1.0
    assert fam["reconstruction"] > 1.0


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_autosetup_phase_caps_are_bounded_and_material_adaptive() -> None:
    vinyl = UnifiedRestorerV3._apply_song_autosetup_policy(
        _base_profile("vinyl"),
        defect_scores={},
        transfer_chain=["vinyl", "mp3_low"],
        max_defect_severity=0.2,
    )
    shellac = UnifiedRestorerV3._apply_song_autosetup_policy(
        _base_profile("shellac"),
        defect_scores={},
        transfer_chain=["shellac", "reel_tape", "mp3_low"],
        max_defect_severity=0.2,
    )

    v_caps = vinyl["strict_conflict_policy"]["phase_strength_caps"]
    s_caps = shellac["strict_conflict_policy"]["phase_strength_caps"]

    for caps in (v_caps, s_caps):
        for val in caps.values():
            assert 0.28 <= float(val) <= 0.90

    # Material-adaptiv: shellac konservativer als vinyl bei carrier-kritischen Phasen.
    assert s_caps["phase_12_wow_flutter_fix"] <= v_caps["phase_12_wow_flutter_fix"]
    assert s_caps["phase_24_dropout_repair"] <= v_caps["phase_24_dropout_repair"]
    assert s_caps["phase_55_diffusion_inpainting"] <= v_caps["phase_55_diffusion_inpainting"]
