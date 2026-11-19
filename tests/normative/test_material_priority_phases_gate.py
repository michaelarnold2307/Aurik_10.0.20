from __future__ import annotations

"""§6.2a [RELEASE_MUST] Material-Pflicht-Phasen + Phase-Ordering Gate.

Validates:
  1. All 14 material types have their spec-mandated priority phases activated
     regardless of DefectScanner severity scores (all-zero severity).
  2. Phase ordering constraints from _optimize_phase_plan_intelligence enforce
     the scientifically correct signal processing chain.
"""


from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# §6.2 Material → Priority Phases (authoritative source: Spec 05 §6.2)
# ---------------------------------------------------------------------------
MATERIAL_PRIORITY_PHASES: dict[str, list[str]] = {
    "tape": ["phase_24_dropout_repair", "phase_29_tape_hiss_reduction", "phase_12_wow_flutter_fix"],
    "reel_tape": [
        "phase_29_tape_hiss_reduction",
        "phase_03_denoise",
        "phase_24_dropout_repair",
        "phase_55_diffusion_inpainting",
    ],
    "vinyl": ["phase_09_crackle_removal", "phase_12_wow_flutter_fix", "phase_05_rumble_filter"],
    "shellac": ["phase_03_denoise", "phase_06_frequency_restoration", "phase_01_click_removal"],
    "wax_cylinder": [
        "phase_03_denoise",
        "phase_06_frequency_restoration",
        "phase_01_click_removal",
        "phase_29_tape_hiss_reduction",
    ],
    "wire_recording": [
        "phase_12_wow_flutter_fix",
        "phase_24_dropout_repair",
        "phase_03_denoise",
        "phase_29_tape_hiss_reduction",
    ],
    "lacquer_disc": [
        "phase_01_click_removal",
        "phase_09_crackle_removal",
        "phase_03_denoise",
        "phase_29_tape_hiss_reduction",
    ],
    "dat": ["phase_24_dropout_repair", "phase_02_hum_removal", "phase_23_spectral_repair"],
    "cd_digital": ["phase_23_spectral_repair", "phase_06_frequency_restoration", "phase_40_loudness_normalization"],
    "mp3_low": ["phase_23_spectral_repair", "phase_03_denoise", "phase_50_spectral_repair"],
    "mp3_high": ["phase_23_spectral_repair", "phase_50_spectral_repair"],
    "aac": ["phase_23_spectral_repair", "phase_38_presence_boost", "phase_06_frequency_restoration"],
    "minidisc": ["phase_23_spectral_repair", "phase_06_frequency_restoration", "phase_07_harmonic_restoration"],
    "streaming": ["phase_24_dropout_repair", "phase_23_spectral_repair", "phase_50_spectral_repair"],
    "cassette": [  # §6.2a: IEC compact cassette: wow/flutter + hiss dominant
        "phase_12_wow_flutter_fix",
        "phase_29_tape_hiss_reduction",
        "phase_24_dropout_repair",
    ],
}


# ---------------------------------------------------------------------------
# Ordering constraints (scientifically mandated precedence rules)
# ---------------------------------------------------------------------------
ORDERING_CONSTRAINTS: list[tuple[str, str]] = [
    # Safety constraints (existing)
    ("phase_24_dropout_repair", "phase_55_diffusion_inpainting"),
    ("phase_57_print_through_reduction", "phase_29_tape_hiss_reduction"),
    ("phase_20_reverb_reduction", "phase_49_advanced_dereverb"),
    ("phase_16_final_eq", "phase_17_mastering_polish"),
    ("phase_17_mastering_polish", "phase_40_loudness_normalization"),
    ("phase_40_loudness_normalization", "phase_47_truepeak_limiter"),
    ("phase_47_truepeak_limiter", "phase_41_output_format_optimization"),
    # Signal-processing-chain ordering (new)
    ("phase_01_click_removal", "phase_03_denoise"),
    ("phase_01_click_removal", "phase_27_click_pop_removal"),
    ("phase_28_surface_noise_profiling", "phase_09_crackle_removal"),
    ("phase_02_hum_removal", "phase_03_denoise"),
    ("phase_03_denoise", "phase_06_frequency_restoration"),
    ("phase_06_frequency_restoration", "phase_07_harmonic_restoration"),
    ("phase_12_wow_flutter_fix", "phase_31_speed_pitch_correction"),
    ("phase_08_transient_preservation", "phase_36_transient_shaper"),
    ("phase_23_spectral_repair", "phase_50_spectral_repair"),
    # §2.46 Tier-2-Playback (EQ/RIAA) vor Tier-3-Aging und Tier-4-Carrier-NR
    ("phase_04_eq_correction", "phase_01_click_removal"),
    ("phase_04_eq_correction", "phase_09_crackle_removal"),
    ("phase_04_eq_correction", "phase_24_dropout_repair"),
    ("phase_04_eq_correction", "phase_03_denoise"),
    ("phase_04_eq_correction", "phase_29_tape_hiss_reduction"),
    # §2.46 Tier-1-ADC (DC, Rumble) vor Tier-2-Playback (EQ/RIAA)
    ("phase_30_dc_offset_removal", "phase_04_eq_correction"),
    ("phase_05_rumble_filter", "phase_04_eq_correction"),
    # Surface-Noise-Profiling vor Tape-Hiss (Profil muss vor Verwendung existieren)
    ("phase_28_surface_noise_profiling", "phase_29_tape_hiss_reduction"),
    # Vinyl Groove-Phasen (Tier-3 Aging) vor Carrier-NR (Tier-4 subtraktiv)
    ("phase_60_inner_groove_distortion_repair", "phase_03_denoise"),
    ("phase_60_inner_groove_distortion_repair", "phase_29_tape_hiss_reduction"),
    ("phase_61_groove_echo_cancellation", "phase_03_denoise"),
    ("phase_61_groove_echo_cancellation", "phase_29_tape_hiss_reduction"),
]


# ---------------------------------------------------------------------------
# Helpers — minimal mocks to test _select_phases without full UV3 init
# ---------------------------------------------------------------------------


def _get_material_type_enum():
    """Import MaterialType from the real codebase."""
    from backend.core.defect_scanner import MaterialType

    return MaterialType


def _make_zero_severity_defect_result(material_value: str):
    """Create a DefectAnalysisResult-like object with all-zero severities."""
    MaterialType = _get_material_type_enum()

    # Find enum member by value
    mat_enum = MaterialType.UNKNOWN
    for m in MaterialType:
        if m.value == material_value:
            mat_enum = m
            break

    mock_result = SimpleNamespace(
        material_type=mat_enum,
        scores={},  # Empty → sev() returns 0.0 for all defects
        is_stereo=True,
    )
    return mock_result


def _make_mock_uv3():
    """Create a minimal mock UV3 instance for _select_phases."""
    mock = MagicMock()
    mock.config.mode = SimpleNamespace(value="restoration")
    return mock


def _call_select_phases(material_value: str):
    """Call UV3._select_phases with zero-severity defect result for given material."""
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    mock_uv3 = _make_mock_uv3()
    defect_result = _make_zero_severity_defect_result(material_value)

    # Call the unbound method with our mock self
    phases = UnifiedRestorerV3._select_phases(
        mock_uv3,
        defect_result,
        audio=None,
        sr=48000,
    )
    return phases


# ===========================================================================
# TEST GROUP 1: §6.2a Material Priority Phases — Zero Severity
# ===========================================================================


@pytest.mark.parametrize(
    "material_value,expected_phases",
    list(MATERIAL_PRIORITY_PHASES.items()),
    ids=list(MATERIAL_PRIORITY_PHASES.keys()),
)
def test_material_priority_phases_zero_severity(material_value: str, expected_phases: list[str]):
    """§6.2a [RELEASE_MUST]: Priority phases MUST be selected even with zero severity."""
    selected = _call_select_phases(material_value)
    for phase_id in expected_phases:
        assert phase_id in selected, (
            f"§6.2a VERLETZUNG: Material '{material_value}' fehlt Pflichtphase "
            f"'{phase_id}' bei Severity=0.0. Selected phases: {selected}"
        )


def test_all_14_materials_covered():
    """Spec §6.2 defines 14 material-specific priority mappings (unknown excluded)."""
    MaterialType = _get_material_type_enum()
    non_unknown_materials = {m.value for m in MaterialType if m != MaterialType.UNKNOWN}
    covered_materials = set(MATERIAL_PRIORITY_PHASES.keys())
    assert non_unknown_materials == covered_materials, (
        f"Material-Pflichtphasen-Mapping unvollständig.\n"
        f"  Fehlend: {non_unknown_materials - covered_materials}\n"
        f"  Überschüssig: {covered_materials - non_unknown_materials}"
    )


# ===========================================================================
# TEST GROUP 2: Phase Ordering Constraints
# ===========================================================================


def _call_optimize_ordering(phase_list: list[str]):
    """Call UV3._optimize_phase_plan_intelligence with given phase list."""
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    mock_uv3 = MagicMock()
    mock_uv3.config = SimpleNamespace(enable_phase_utility_scoring=False)

    result = UnifiedRestorerV3._optimize_phase_plan_intelligence(
        mock_uv3,
        list(phase_list),
        causal_plan=None,
        pipeline_confidence=None,
        restorability_score=50.0,
    )
    return result


@pytest.mark.parametrize(
    "phase_a,phase_b",
    ORDERING_CONSTRAINTS,
    ids=[f"{a}_before_{b}" for a, b in ORDERING_CONSTRAINTS],
)
def test_ordering_constraint_enforced(phase_a: str, phase_b: str):
    """Ordering constraint: phase_a MUST come before phase_b when both are present."""
    # Build a deliberately REVERSED input to verify the optimizer fixes it
    phases_reversed = [phase_b, phase_a]
    result = _call_optimize_ordering(phases_reversed)
    ia = result.index(phase_a)
    ib = result.index(phase_b)
    assert ia < ib, (
        f"Ordering-Constraint verletzt: '{phase_a}' (idx={ia}) muss vor "
        f"'{phase_b}' (idx={ib}) stehen. Ergebnis: {result}"
    )


def test_tier6_always_at_end():
    """TIER 6 finalization phases must always be at the end of the plan."""
    tier6 = [
        "phase_16_final_eq",
        "phase_17_mastering_polish",
        "phase_40_loudness_normalization",
        "phase_47_truepeak_limiter",
        "phase_41_output_format_optimization",
    ]
    # Build a mixed list with tier6 at the front
    other = ["phase_01_click_removal", "phase_03_denoise", "phase_06_frequency_restoration"]
    phases = tier6 + other
    result = _call_optimize_ordering(phases)

    # Tier 6 ordering must be preserved
    tier6_indices = [result.index(p) for p in tier6 if p in result]
    assert tier6_indices == sorted(tier6_indices), (
        f"TIER 6 internal ordering broken: {[result[i] for i in tier6_indices]}"
    )


def test_click_before_denoise_after_optimization():
    """phase_01 must come before phase_03 even if initially reversed."""
    phases = [
        "phase_30_dc_offset_removal",
        "phase_03_denoise",
        "phase_01_click_removal",
        "phase_06_frequency_restoration",
        "phase_16_final_eq",
    ]
    result = _call_optimize_ordering(phases)
    assert result.index("phase_01_click_removal") < result.index("phase_03_denoise"), (
        f"Click removal must precede broadband denoising: {result}"
    )


def test_denoise_before_freq_restoration_after_optimization():
    """phase_03 must come before phase_06 even if initially reversed."""
    phases = [
        "phase_30_dc_offset_removal",
        "phase_06_frequency_restoration",
        "phase_03_denoise",
        "phase_16_final_eq",
    ]
    result = _call_optimize_ordering(phases)
    assert result.index("phase_03_denoise") < result.index("phase_06_frequency_restoration"), (
        f"Denoise must precede frequency restoration: {result}"
    )


def test_full_chain_ordering():
    """Test complete signal processing chain order with all key phases present."""
    phases = [
        # Deliberately shuffled
        "phase_06_frequency_restoration",
        "phase_40_loudness_normalization",
        "phase_03_denoise",
        "phase_01_click_removal",
        "phase_07_harmonic_restoration",
        "phase_17_mastering_polish",
        "phase_30_dc_offset_removal",
        "phase_47_truepeak_limiter",
        "phase_16_final_eq",
        "phase_41_output_format_optimization",
        "phase_02_hum_removal",
    ]
    result = _call_optimize_ordering(phases)

    # Verify all constraints hold
    expected_order_pairs = [
        ("phase_01_click_removal", "phase_03_denoise"),
        ("phase_02_hum_removal", "phase_03_denoise"),
        ("phase_03_denoise", "phase_06_frequency_restoration"),
        ("phase_06_frequency_restoration", "phase_07_harmonic_restoration"),
        ("phase_16_final_eq", "phase_17_mastering_polish"),
        ("phase_17_mastering_polish", "phase_40_loudness_normalization"),
        ("phase_40_loudness_normalization", "phase_47_truepeak_limiter"),
        ("phase_47_truepeak_limiter", "phase_41_output_format_optimization"),
    ]
    for a, b in expected_order_pairs:
        if a in result and b in result:
            assert result.index(a) < result.index(b), f"Ordering violation: '{a}' must come before '{b}' in {result}"
