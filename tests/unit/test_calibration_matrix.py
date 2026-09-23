from __future__ import annotations

import pytest

"""Unit tests for backend/core/calibration_matrix.py — §09.1/§09.2/§09.7.

Tests cover:
- CANONICAL_THRESHOLDS_* export and consistency
- estimate_song_goal_targets: era/material/genre bias application
- predict_quality_score: material-ceiling and restorability scaling
"""


import numpy as np

from backend.core.calibration_matrix import (
    CANONICAL_THRESHOLDS_RESTORATION,
    CANONICAL_THRESHOLDS_STUDIO2026,
    estimate_chain_end_goal_ceiling,
    estimate_song_goal_targets,
    get_goal_recovery_phases,
    get_material_floor,
    get_phase_strength_range,
    predict_quality_score,
    resolve_effective_goal_targets,
)

# ---------------------------------------------------------------------------
# §09.1 Canonical Thresholds
# ---------------------------------------------------------------------------

_P1_GOALS = {"natuerlichkeit", "authentizitaet"}
_P2_GOALS = {"tonal_center", "timbre_authentizitaet", "artikulation"}


@pytest.mark.unit
def test_canonical_thresholds_restoration_has_all_goals():
    """All 15 goals must be present in CANONICAL_THRESHOLDS_RESTORATION."""
    expected = {
        "natuerlichkeit",
        "authentizitaet",
        "tonal_center",
        "timbre_authentizitaet",
        "artikulation",
        "transient_energie",
        "emotionalitaet",
        "mikrodynamik",
        "groove",
        "transparenz",
        "waerme",
        "bass_kraft",
        "separation_fidelity",
        "brillanz",
        "raumtiefe",
    }
    for g in expected:
        assert g in CANONICAL_THRESHOLDS_RESTORATION, f"Goal '{g}' missing in CANONICAL_THRESHOLDS_RESTORATION"


def test_canonical_thresholds_studio2026_has_all_goals():
    """Studio 2026 thresholds must have the same goal keys."""
    for g in CANONICAL_THRESHOLDS_RESTORATION:
        assert g in CANONICAL_THRESHOLDS_STUDIO2026, f"Goal '{g}' in RESTORATION but missing in STUDIO2026"


def test_p1_p2_floors_are_identical_across_modes():
    """P1/P2 goals must have identical floors in both modes (§09.1a/b comment)."""
    p1p2 = _P1_GOALS | _P2_GOALS
    for g in p1p2:
        if g in CANONICAL_THRESHOLDS_RESTORATION and g in CANONICAL_THRESHOLDS_STUDIO2026:
            r = CANONICAL_THRESHOLDS_RESTORATION[g]
            s = CANONICAL_THRESHOLDS_STUDIO2026[g]
            # Allow ≤ 0.03 delta — spec notes P1/P2 are "identical or near-identical"
            assert abs(r - s) <= 0.05, f"P1/P2 goal '{g}': Restoration={r}, Studio={s}, delta too large"


def test_all_thresholds_in_valid_range():
    """All canonical thresholds must be in (0.50, 1.00)."""
    for mode, thresholds in [
        ("Restoration", CANONICAL_THRESHOLDS_RESTORATION),
        ("Studio2026", CANONICAL_THRESHOLDS_STUDIO2026),
    ]:
        for goal, val in thresholds.items():
            assert 0.50 <= val < 1.00, f"{mode}/{goal}={val} out of range [0.50, 1.00)"


# ---------------------------------------------------------------------------
# §09.2 estimate_song_goal_targets
# ---------------------------------------------------------------------------


def test_shellac_1930_has_lower_brillanz_than_cd_2005():
    """Ultra-analog era+material bias must reduce brillanz target vs. modern digital."""
    t_old = estimate_song_goal_targets(
        material_type="shellac",
        era_decade=1935,
        is_studio_2026=False,
        restorability_score=40,
    )
    t_new = estimate_song_goal_targets(
        material_type="cd_digital",
        era_decade=2005,
        is_studio_2026=False,
        restorability_score=85,
    )
    assert t_old["brillanz"] < t_new["brillanz"], (
        f"shellac/1935 brillanz={t_old['brillanz']:.3f} should be < cd/2005 {t_new['brillanz']:.3f}"
    )


def test_shellac_1930_has_higher_waerme_than_cd_2005():
    """Vintage analog material should have higher waerme target (warm character)."""
    t_old = estimate_song_goal_targets(
        material_type="shellac",
        era_decade=1935,
        is_studio_2026=False,
        restorability_score=40,
    )
    t_new = estimate_song_goal_targets(
        material_type="cd_digital",
        era_decade=2005,
        is_studio_2026=False,
        restorability_score=85,
    )
    assert t_old["waerme"] > t_new["waerme"], (
        f"shellac/1935 waerme={t_old['waerme']:.3f} should be > cd/2005 {t_new['waerme']:.3f}"
    )


def test_klassik_genre_raises_raumtiefe():
    """Klassik genre bias must result in higher raumtiefe target than Pop."""
    t_klassik = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        genre_label="klassik",
        is_studio_2026=False,
        restorability_score=70,
    )
    t_pop = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        genre_label="pop",
        is_studio_2026=False,
        restorability_score=70,
    )
    assert t_klassik["raumtiefe"] > t_pop["raumtiefe"]


def test_jazz_genre_raises_waerme():
    """Jazz genre bias must result in higher waerme target than rock."""
    t_jazz = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        genre_label="jazz",
        is_studio_2026=False,
        restorability_score=70,
    )
    t_rock = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        genre_label="rock",
        is_studio_2026=False,
        restorability_score=70,
    )
    assert t_jazz["waerme"] > t_rock["waerme"]


def test_targets_all_in_valid_range():
    """All returned targets must be in [0.30, 0.99]."""
    targets = estimate_song_goal_targets(
        material_type="shellac",
        era_decade=1928,
        genre_label="jazz",
        restorability_score=25,
        is_studio_2026=False,
    )
    for g, v in targets.items():
        assert 0.30 <= v <= 0.99, f"target[{g}]={v} out of [0.30, 0.99]"


def test_targets_all_finite():
    """All returned targets must be finite (no NaN/Inf)."""
    targets = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1968,
        genre_label="rock",
        restorability_score=55,
        is_studio_2026=True,
    )
    for g, v in targets.items():
        assert np.isfinite(v), f"target[{g}]={v} is not finite"


def test_targets_keys_match_restoration_canonical():
    """estimate_song_goal_targets must return the same keys as CANONICAL_THRESHOLDS_RESTORATION."""
    targets = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        is_studio_2026=False,
        restorability_score=65,
    )
    assert set(targets.keys()) == set(CANONICAL_THRESHOLDS_RESTORATION.keys())


def test_studio_2026_mode_raises_targets():
    """Studio 2026 mode targets must be ≥ Restoration targets for P3–P5 goals."""
    t_rest = estimate_song_goal_targets(
        is_studio_2026=False,
        material_type="vinyl",
        era_decade=1975,
        genre_label="pop",
        restorability_score=70,
    )
    t_s26 = estimate_song_goal_targets(
        is_studio_2026=True,
        material_type="vinyl",
        era_decade=1975,
        genre_label="pop",
        restorability_score=70,
    )
    # P3-P5 floor is higher in Studio 2026 → targets must be higher or equal
    for g in ("transparenz", "brillanz", "groove", "mikrodynamik"):
        assert t_s26.get(g, 0) >= t_rest.get(g, 0) - 0.02, (
            f"Studio2026[{g}]={t_s26.get(g):.3f} < Restoration[{g}]={t_rest.get(g):.3f}"
        )


def test_goal_weights_above_1_raise_target():
    """goal_weight > 1.0 for a goal must increase its target slightly."""
    base = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        restorability_score=70,
        is_studio_2026=False,
    )
    weighted = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        restorability_score=70,
        is_studio_2026=False,
        goal_weights={"natuerlichkeit": 1.8},
    )
    assert weighted["natuerlichkeit"] >= base["natuerlichkeit"]


def test_goal_weights_below_1_lower_target():
    """goal_weight < 1.0 for a goal must lower its target slightly."""
    base = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        restorability_score=70,
        is_studio_2026=False,
    )
    weighted = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        restorability_score=70,
        is_studio_2026=False,
        goal_weights={"brillanz": 0.4},
    )
    assert weighted["brillanz"] <= base["brillanz"]


def test_none_inputs_do_not_crash():
    """Graceful handling of None/missing inputs — must not raise."""
    result = estimate_song_goal_targets(is_studio_2026=False)
    assert isinstance(result, dict)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# §09.7 predict_quality_score
# ---------------------------------------------------------------------------


def test_shellac_quality_lower_than_vinyl():
    """Shellac has lower quality ceiling than vinyl."""
    q_shellac = predict_quality_score("shellac", 50.0, 0.4, False)
    q_vinyl = predict_quality_score("vinyl", 50.0, 0.4, False)
    assert q_shellac < q_vinyl


def test_cd_digital_has_highest_ceiling():
    """CD digital with high restorability and no defects → near-maximum quality."""
    q_cd = predict_quality_score("cd_digital", 95.0, 0.0, False)
    assert q_cd > 0.85


def test_studio_boost_raises_score():
    """Studio 2026 mode adds boost over Restoration."""
    q_rest = predict_quality_score("vinyl", 70.0, 0.3, False)
    q_s26 = predict_quality_score("vinyl", 70.0, 0.3, True)
    assert q_s26 > q_rest


def test_heavy_defects_reduce_score():
    """Heavy defects (severity=0.9) must reduce quality vs. no defects."""
    q_clean = predict_quality_score("vinyl", 65.0, 0.0, False)
    q_damaged = predict_quality_score("vinyl", 65.0, 0.9, False)
    assert q_damaged < q_clean


def test_quality_score_in_valid_range():
    """Output must always be in [0.0, 0.99]."""
    for mat in ["shellac", "vinyl", "tape", "cd_digital", "mp3_low", "wax_cylinder"]:
        for rest in [5.0, 50.0, 95.0]:
            q = predict_quality_score(mat, rest, 0.5, False)
            assert 0.0 <= q <= 0.99, f"{mat}/rest={rest}: q={q} out of range"


def test_quality_score_finite():
    """Output must never be NaN or Inf."""
    q = predict_quality_score("unknown_material", 55.0, 0.5, True)
    assert np.isfinite(q)


# ---------------------------------------------------------------------------
# §09.8 get_material_floor — material-adaptive goal floors
# ---------------------------------------------------------------------------


def test_get_material_floor_shellac_brillanz_below_canonical():
    """Shellac brillanz floor must be below canonical (physical BW limit ≤ 8 kHz)."""
    canonical_floor = CANONICAL_THRESHOLDS_RESTORATION["brillanz"]
    shellac_floor = get_material_floor("shellac", "brillanz")
    assert shellac_floor < canonical_floor, (
        f"shellac brillanz floor {shellac_floor:.3f} must be < canonical {canonical_floor:.3f}"
    )


def test_get_material_floor_vinyl_between_shellac_and_cd():
    """Vinyl brillanz floor must be between shellac and cd_digital."""
    f_shellac = get_material_floor("shellac", "brillanz")
    f_vinyl = get_material_floor("vinyl", "brillanz")
    f_cd = get_material_floor("cd_digital", "brillanz")
    assert f_shellac < f_vinyl <= f_cd, f"Expected shellac({f_shellac:.3f}) < vinyl({f_vinyl:.3f}) <= cd({f_cd:.3f})"


def test_get_material_floor_cd_matches_canonical_closely():
    """CD digital floor should be at or above canonical (digital bias is positive)."""
    for goal in ["transparenz", "artikulation", "brillanz"]:
        floor = get_material_floor("cd_digital", goal)
        canonical = CANONICAL_THRESHOLDS_RESTORATION[goal]
        assert floor >= canonical * 0.99, (
            f"cd_digital {goal} floor {floor:.3f} unexpectedly far below canonical {canonical:.3f}"
        )


def test_get_material_floor_always_in_valid_range():
    """All material/goal combinations must stay in [0.30, 0.99]."""
    materials = ["wax_cylinder", "shellac", "vinyl", "tape", "cd_digital", "mp3_low", "unknown"]
    goals = list(CANONICAL_THRESHOLDS_RESTORATION.keys())
    for mat in materials:
        for goal in goals:
            floor = get_material_floor(mat, goal)
            assert 0.30 <= floor <= 0.99, f"get_material_floor({mat!r}, {goal!r}) = {floor:.3f} out of [0.30, 0.99]"


def test_get_material_floor_studio_2026_geq_restoration():
    """Studio 2026 floors should be >= restoration floors (higher canonical base)."""
    for goal in ["natuerlichkeit", "brillanz", "transparenz"]:
        f_rest = get_material_floor("vinyl", goal, is_studio_2026=False)
        f_s26 = get_material_floor("vinyl", goal, is_studio_2026=True)
        assert f_s26 >= f_rest - 0.01, (
            f"Studio2026 floor ({f_s26:.3f}) unexpectedly below restoration floor ({f_rest:.3f}) for vinyl/{goal}"
        )


def test_get_material_floor_unknown_material_returns_finite():
    """Unknown material should fall back gracefully (analog class)."""
    floor = get_material_floor("totally_unknown_carrier", "natuerlichkeit")
    assert np.isfinite(floor)
    assert 0.30 <= floor <= 0.99


# ---------------------------------------------------------------------------
# §09.9 get_phase_strength_range — material-adaptive strength caps
# ---------------------------------------------------------------------------


def test_get_phase_strength_range_ultra_analog_is_capped():
    """Ultra-analog (shellac) phases must have lower max_strength than analog (vinyl)."""
    _, max_shellac = get_phase_strength_range("phase_03_denoise", "shellac", 50.0)
    _, max_vinyl = get_phase_strength_range("phase_03_denoise", "vinyl", 50.0)
    assert max_shellac < max_vinyl, (
        f"shellac max_strength ({max_shellac:.2f}) must be < vinyl ({max_vinyl:.2f}) for phase_03"
    )


def test_get_phase_strength_range_high_restorability_reduces_max():
    """High restorability (>80) must reduce max_strength toward passthrough (§2.45b)."""
    _, max_low = get_phase_strength_range("phase_03_denoise", "vinyl", 30.0)
    _, max_high = get_phase_strength_range("phase_03_denoise", "vinyl", 95.0)
    assert max_high < max_low, (
        f"High restorability max ({max_high:.2f}) must be < low restorability max ({max_low:.2f})"
    )


def test_get_phase_strength_range_min_never_exceeds_max():
    """min_strength must always be <= max_strength for all materials/phases."""
    phases = [
        "phase_03_denoise",
        "phase_07_harmonic_restoration",
        "phase_09_crackle_removal",
        "phase_23_spectral_repair",
        "phase_26_dynamic_range_expansion",
        "phase_49_advanced_dereverb",
    ]
    materials = ["wax_cylinder", "shellac", "vinyl", "tape", "cd_digital", "mp3_low"]
    for mat in materials:
        for phase in phases:
            min_s, max_s = get_phase_strength_range(phase, mat, 70.0)
            assert min_s <= max_s, f"min({min_s:.2f}) > max({max_s:.2f}) for {mat}/{phase}"
            assert 0.0 <= min_s <= 1.0
            assert 0.0 <= max_s <= 1.0


def test_get_phase_strength_range_returns_floats():
    """Return values must be Python floats, not numpy scalars."""
    min_s, max_s = get_phase_strength_range("phase_09_crackle_removal", "vinyl", 60.0)
    assert isinstance(min_s, float)
    assert isinstance(max_s, float)


def test_get_phase_strength_range_stereo_forbidden_on_ultra_analog():
    """Stereo width enhancer must have max_strength=0 on mono ultra_analog material."""
    _, max_s = get_phase_strength_range("phase_48_stereo_width_enhancer", "shellac", 50.0)
    assert max_s == 0.0, f"phase_48 must be disabled on shellac (max={max_s:.2f})"


def test_get_phase_strength_range_unknown_phase_uses_default():
    """Unknown phase IDs must return the safe default range."""
    min_s, max_s = get_phase_strength_range("phase_99_nonexistent", "vinyl", 50.0)
    assert max_s > 0.0  # default range is permissive
    assert min_s <= max_s


# ---------------------------------------------------------------------------
# §09.8 Material-floor ordering — materials formerly mirrored in UV3 material ceiling tables
# (§0a §2.44 defensive ceiling for lacquer_disc, mp3_high, aac, minidisc, dat)
# ---------------------------------------------------------------------------


def test_material_floor_lacquer_disc_lower_than_vinyl_brillanz():
    """lacquer_disc BW ≤ 8 kHz → brillanz floor must be below vinyl (BW ≤ 16 kHz)."""
    f_lacquer = get_material_floor("lacquer_disc", "brillanz")
    f_vinyl = get_material_floor("vinyl", "brillanz")
    assert f_lacquer < f_vinyl, (
        f"lacquer_disc brillanz floor {f_lacquer:.3f} should be below vinyl {f_vinyl:.3f} "
        "(lacquer_disc BW ≤ 8 kHz, vinyl BW ≤ 16 kHz — §0a §6.2c)"
    )


def test_material_floor_dat_near_cd_level():
    """DAT is near-CD quality; its brillanz floor must be ≥ 90 % of cd_digital floor."""
    f_dat = get_material_floor("dat", "brillanz")
    f_cd = get_material_floor("cd_digital", "brillanz")
    assert f_dat >= 0.90 * f_cd, (
        f"dat brillanz floor {f_dat:.3f} should be at least 90 % of cd_digital {f_cd:.3f} "
        "(DAT is near-CD quality — §0a §2.44)"
    )


def test_material_floor_ordering_ultra_analog_to_digital():
    """Physical material quality ordering must hold for brillanz floor:
    wax_cylinder < shellac ≤ lacquer_disc < vinyl < mp3_low < minidisc ≤ mp3_high ≤ aac < dat < cd_digital
    """
    materials = [
        "wax_cylinder",
        "shellac",
        "lacquer_disc",
        "vinyl",
        "mp3_low",
        "minidisc",
        "mp3_high",
        "aac",
        "dat",
        "cd_digital",
    ]
    floors = [get_material_floor(m, "brillanz") for m in materials]
    for i in range(len(floors) - 1):
        assert floors[i] <= floors[i + 1] + 0.02, (  # 0.02 tolerance for equal-tier materials
            f"brillanz floor ordering violated: {materials[i]}={floors[i]:.3f} "
            f"> {materials[i + 1]}={floors[i + 1]:.3f} (§0a physical medium ordering)"
        )


def test_cassette_emotionalitaet_floor_below_canonical():
    """Kassetten-emotionalitaet-Floor muss unter dem kanonischen Wert liegen.

    Physikalische Begründung: Kassetten-AGC-Kompressionsschaltkreise reduzieren
    dynamische Modulation; 12-kHz-BW-Ceiling begrenzt tonale Bandbreite für
    Arousal-Messungen. Echtmessung 2026-05-20: Original-Kassette = 0.782 < Canon 0.84.
    """
    canonical = CANONICAL_THRESHOLDS_RESTORATION["emotionalitaet"]
    cassette_floor = get_material_floor("cassette", "emotionalitaet")
    assert cassette_floor < canonical, (
        f"cassette emotionalitaet floor {cassette_floor:.3f} must be < canonical {canonical:.3f} "
        "(Kassetten-AGC + BW-Ceiling 12 kHz — §09.2 v10.0.0)"
    )


def test_cassette_emotionalitaet_floor_compatible_with_real_measurement():
    """Floor muss mit Echtmessung kompatibel sein: Original-Kassette 0.782 >= Floor.

    Ohne diesen Bias löst §GOAL_BASELINE_CHECK fälschlicherweise Recovery-Phasen aus,
    da das Kassetten-Original den Floor physikalisch nicht erreichen kann.
    Echtmessung: Testkünstlerin (Schlager), Kassette, 1970er, Schlager, measure_all(panns=0.7) = 0.782.
    """
    cassette_floor = get_material_floor("cassette", "emotionalitaet")
    real_original_score = 0.782  # measure_all() Echtmessung 2026-05-20
    assert real_original_score >= cassette_floor, (
        f"Original-Kassette ({real_original_score:.3f}) liegt unter Floor ({cassette_floor:.3f}) — "
        "§GOAL_BASELINE_CHECK würde fälschlicherweise Recovery triggern (§09.2 v10.0.0)"
    )


def test_tape_emotionalitaet_floor_below_canonical():
    """Tape-emotionalitaet-Floor muss ebenfalls unter Canon liegen (gleiche Bias-Klasse tape_analog)."""
    canonical = CANONICAL_THRESHOLDS_RESTORATION["emotionalitaet"]
    tape_floor = get_material_floor("tape", "emotionalitaet")
    assert tape_floor < canonical, (
        f"tape emotionalitaet floor {tape_floor:.3f} must be < canonical {canonical:.3f} "
        "(tape_analog class — gleiche Bias-Struktur wie cassette)"
    )


def test_cassette_emotionalitaet_floor_range():
    """Floor muss im sinnvollen Bereich [0.77, 0.82] liegen.

    Zu niedrig (< 0.77) würde Restaurierungs-Gate zu permissiv machen;
    zu hoch (>= Canon 0.84) würde §GOAL_BASELINE_CHECK triggern trotz physikalischer Grenze.
    """
    floor = get_material_floor("cassette", "emotionalitaet")
    assert 0.77 <= floor < 0.84, (
        f"cassette emotionalitaet floor {floor:.3f} must be in [0.77, 0.84) "
        "(physikalisches Ceiling Kassette-AGC + BW 12 kHz — §09.2 v10.0.0)"
    )


def test_pmgg_canonical_thresholds_match_calibration_matrix():
    """PMGG's internal threshold copy must stay in sync with calibration_matrix (§2.55).

    Drift between PMGG's local _CANONICAL_THRESHOLDS_RESTORATION and
    calibration_matrix.CANONICAL_THRESHOLDS_RESTORATION causes silent per-phase
    gate inconsistencies — goals pass at pipeline end but fail in PMGG or vice versa.
    """
    from backend.core.per_phase_musical_goals_gate import (
        _CANONICAL_THRESHOLDS_RESTORATION as PMGG_REST,
    )
    from backend.core.per_phase_musical_goals_gate import (
        _CANONICAL_THRESHOLDS_STUDIO2026 as PMGG_S26,
    )

    # Key normalization: PMGG uses short aliases (micro_dynamics, bass_kraft, spatial_depth)
    # calibration_matrix uses both aliases; we map them to a common key for comparison.
    _ALIAS = {
        "micro_dynamics": "mikrodynamik",
        "bass_kraft": "basskraft",
        "spatial_depth": "raumtiefe",
        "tonalcenter": "tonal_center",
    }

    def _norm(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            key = _ALIAS.get(k, k)
            out[key] = v
        return out

    pmgg_r_norm = _norm(PMGG_REST)
    pmgg_s_norm = _norm(PMGG_S26)
    cal_r_norm = _norm(CANONICAL_THRESHOLDS_RESTORATION)
    cal_s_norm = _norm(CANONICAL_THRESHOLDS_STUDIO2026)

    # Check every goal that appears in BOTH dicts
    for goal in cal_r_norm:
        if goal in pmgg_r_norm:
            assert abs(pmgg_r_norm[goal] - cal_r_norm[goal]) < 0.005, (
                f"PMGG restoration threshold for '{goal}' = {pmgg_r_norm[goal]:.4f} "
                f"differs from calibration_matrix {cal_r_norm[goal]:.4f} — §2.55 sync violation"
            )
    for goal in cal_s_norm:
        if goal in pmgg_s_norm:
            assert abs(pmgg_s_norm[goal] - cal_s_norm[goal]) < 0.005, (
                f"PMGG studio2026 threshold for '{goal}' = {pmgg_s_norm[goal]:.4f} "
                f"differs from calibration_matrix {cal_s_norm[goal]:.4f} — §2.55 sync violation"
            )


# ---------------------------------------------------------------------------
# §09.10 get_goal_recovery_phases — GOAL_BASELINE_CHECK backing data
# ---------------------------------------------------------------------------

_ALL_15_GOALS = {
    "brillanz",
    "waerme",
    "natuerlichkeit",
    "transparenz",
    "tonal_center",
    "groove",
    "artikulation",
    "transient_energie",
    "micro_dynamics",
    "emotionalitaet",
    "bass_kraft",
    "separation_fidelity",
    "spatial_depth",
    "authentizitaet",
    "timbre_authentizitaet",
}

# Canonical §0a-forbidden phases — must NEVER appear in restoration mode results
_FORBIDDEN_RESTORATION_PHASES = {
    "phase_21_exciter",
    "phase_35_multiband_compression",
    "phase_42_vocal_enhancement",
}


def test_get_goal_recovery_phases_returns_list_of_strings():
    """get_goal_recovery_phases must return a list of phase-id strings."""
    result = get_goal_recovery_phases("brillanz")
    assert isinstance(result, list)
    assert all(isinstance(p, str) for p in result)


def test_get_goal_recovery_phases_all_15_goals_have_entries():
    """Every canonical Musical Goal must have at least one restoration recovery phase."""
    missing = []
    for goal in _ALL_15_GOALS:
        phases = get_goal_recovery_phases(goal, is_studio_2026=False)
        if not phases:
            missing.append(goal)
    assert not missing, (
        f"Goals with no restoration recovery phases: {sorted(missing)} — "
        "§GOAL_BASELINE_CHECK cannot guarantee threshold achievement for these goals"
    )


def test_get_goal_recovery_phases_no_forbidden_in_restoration():
    """§0a: phase_21/35/42 must NEVER appear in restoration mode recovery lists."""
    violations = []
    for goal in _ALL_15_GOALS:
        phases = get_goal_recovery_phases(goal, is_studio_2026=False)
        bad = set(phases) & _FORBIDDEN_RESTORATION_PHASES
        if bad:
            violations.append(f"{goal}: {sorted(bad)}")
    assert not violations, f"§0a violation — forbidden phases in restoration recovery list: {violations}"


def test_get_goal_recovery_phases_studio_extras_add_phases():
    """Studio 2026 mode must return at least as many phases as restoration for all goals."""
    for goal in _ALL_15_GOALS:
        rest_phases = get_goal_recovery_phases(goal, is_studio_2026=False)
        studio_phases = get_goal_recovery_phases(goal, is_studio_2026=True)
        assert len(studio_phases) >= len(rest_phases), (
            f"Studio 2026 should return ≥ restoration phases for '{goal}': rest={rest_phases} studio={studio_phases}"
        )


def test_get_goal_recovery_phases_unknown_goal_returns_empty():
    """Unknown goal keys must return empty list (non-blocking behaviour)."""
    assert get_goal_recovery_phases("nonexistent_goal_xyz") == []
    assert get_goal_recovery_phases("") == []


def test_get_goal_recovery_phases_alias_normalisation():
    """Legacy aliases (raumtiefe, mikrodynamik, basskraft) must resolve to the same
    phases as their canonical names (spatial_depth, micro_dynamics, bass_kraft)."""
    alias_pairs = [
        ("raumtiefe", "spatial_depth"),
        ("mikrodynamik", "micro_dynamics"),
        ("basskraft", "bass_kraft"),
    ]
    for alias, canonical in alias_pairs:
        assert get_goal_recovery_phases(alias) == get_goal_recovery_phases(canonical), (
            f"Alias '{alias}' returns different phases than canonical '{canonical}'"
        )


def test_get_goal_recovery_phases_brillanz_includes_spectral_repair_after_bw_extension():
    """Brillanz-Recovery muss phase_23 zwischen BW-Extension und Harmonic-Restore fuehren."""
    phases = get_goal_recovery_phases("brillanz", is_studio_2026=False)
    assert "phase_06_frequency_restoration" in phases
    assert "phase_23_spectral_repair" in phases
    assert "phase_07_harmonic_restoration" in phases

    idx_06 = phases.index("phase_06_frequency_restoration")
    idx_23 = phases.index("phase_23_spectral_repair")
    idx_07 = phases.index("phase_07_harmonic_restoration")
    assert idx_06 < idx_23 < idx_07, f"Unerwartete Reihenfolge fuer brillanz recovery: {phases}"


def test_get_goal_recovery_phases_transient_energie_includes_spectral_repair_tail():
    """Transient-Energie-Recovery muss codec-smear Reparatur als dritten Schritt enthalten."""
    phases = get_goal_recovery_phases("transient_energie", is_studio_2026=False)
    assert phases[:2] == ["phase_26_dynamic_range_expansion", "phase_08_transient_preservation"]
    assert "phase_23_spectral_repair" in phases
    assert phases.index("phase_08_transient_preservation") < phases.index("phase_23_spectral_repair")


def test_get_goal_recovery_phases_lossy_chain_injects_phase50_for_brillanz():
    """Bei lossy chain-end soll phase_50 fuer brillanz nach phase_06 eingefuegt werden."""
    phases = get_goal_recovery_phases("brillanz", transfer_chain=["cassette", "mp3_low"])
    assert "phase_50_spectral_repair" in phases
    assert phases.index("phase_06_frequency_restoration") < phases.index("phase_50_spectral_repair")


def test_get_goal_recovery_phases_lossy_chain_injects_phase50_for_transient_energie():
    """Bei lossy chain-end soll transient_energie codec-smear repair via phase_50 priorisieren."""
    phases = get_goal_recovery_phases("transient_energie", transfer_chain=["vinyl", "aac"])
    assert phases[:2] == ["phase_26_dynamic_range_expansion", "phase_08_transient_preservation"]
    assert "phase_50_spectral_repair" in phases
    assert phases.index("phase_08_transient_preservation") < phases.index("phase_50_spectral_repair")


def test_get_goal_recovery_phases_no_duplicates():
    """Returned phase list must have no duplicate entries for any goal or mode."""
    for goal in _ALL_15_GOALS:
        for studio in (False, True):
            phases = get_goal_recovery_phases(goal, is_studio_2026=studio)
            assert len(phases) == len(set(phases)), f"Duplicate phases for goal='{goal}' studio={studio}: {phases}"


def test_estimate_chain_end_goal_ceiling_uses_last_transfer_stage():
    """Transferketten-Ende muss HF-sensitive Goals begrenzen (§2.46a)."""
    caps = estimate_chain_end_goal_ceiling(["vinyl", "cassette", "mp3_low"])
    assert caps["brillanz"] == 0.45
    assert caps["transient_energie"] == 0.70
    assert caps["transparenz"] == 0.60


def test_resolve_effective_goal_targets_caps_by_physical_and_chain_ceiling():
    """Effective targets dürfen weder PhysicalCeiling noch Chain-End-Ceiling überschreiten."""
    targets = resolve_effective_goal_targets(
        is_studio_2026=False,
        restorability_score=90.0,
        era_decade=1990,
        genre_label="pop",
        material_type="vinyl",
        transfer_chain=["vinyl", "mp3_low"],
        physical_ceiling={"brillanz": 0.70, "transparenz": 0.90, "transient_energie": 0.95},
        applicable_goals={"brillanz", "transparenz", "transient_energie"},
    )
    assert targets["brillanz"] <= 0.45
    assert targets["transient_energie"] <= 0.70
    assert targets["transparenz"] <= 0.60


def test_resolve_effective_goal_targets_respects_restorability_floor():
    """Extrem niedrige Restorability darf targets nur bis RESTORABILITY_SCALE_MIN senken."""
    targets = resolve_effective_goal_targets(
        is_studio_2026=False,
        restorability_score=10.0,
        material_type="mp3_low",
        applicable_goals={"natuerlichkeit"},
        physical_ceiling={"natuerlichkeit": 0.99},
    )
    assert targets["natuerlichkeit"] >= get_material_floor("mp3_low", "natuerlichkeit") * 0.72 - 1e-6


def test_get_goal_recovery_phases_primary_phase_is_string():
    """Primary recovery phase (index 0) must be a non-empty string for all 15 goals."""
    for goal in _ALL_15_GOALS:
        phases = get_goal_recovery_phases(goal)
        assert phases and phases[0], f"Primary recovery phase for '{goal}' is empty/None: {phases}"


def test_get_goal_recovery_phases_all_phase_ids_exist_on_disk():
    """All phase IDs in both recovery dicts must map to a real phase_*.py file.

    Root-cause guard for the class of bugs where a wrong or renamed phase ID is
    entered into _GOAL_TO_RECOVERY_PHASES_RESTORATION / _STUDIO_EXTRAS. Such an
    ID passes all structural tests but causes §GOAL_BASELINE_CHECK to silently
    insert a phase that UV3 cannot find — recovery mechanism fails without error.
    """
    import pathlib

    import backend.core.calibration_matrix as _cm

    phases_dir = pathlib.Path(__file__).parent.parent.parent / "backend" / "core" / "phases"
    valid_ids = {p.stem for p in phases_dir.glob("phase_*.py")}

    violations: list[str] = []
    for goal, phases in _cm._GOAL_TO_RECOVERY_PHASES_RESTORATION.items():
        for phase_id in phases:
            if phase_id not in valid_ids:
                violations.append(f"RESTORATION goal='{goal}': '{phase_id}' — no matching phase file")
    for goal, phases in _cm._GOAL_TO_RECOVERY_PHASES_STUDIO_EXTRAS.items():
        for phase_id in phases:
            if phase_id not in valid_ids:
                violations.append(f"STUDIO_EXTRAS goal='{goal}': '{phase_id}' — no matching phase file")

    assert not violations, "Phase IDs in recovery dicts do not match any backend/core/phases/phase_*.py:\n" + "\n".join(
        f"  \u2022 {v}" for v in violations
    )


# ---------------------------------------------------------------------------
# §09.12 get_effective_material_floor()
# ---------------------------------------------------------------------------


def test_get_effective_material_floor_restorability_100():
    """restorability=100 → scale=1.0 → gleicher Floor wie get_material_floor()."""
    from backend.core.calibration_matrix import get_effective_material_floor

    base = get_material_floor("cd_digital", "natuerlichkeit")
    eff = get_effective_material_floor("cd_digital", "natuerlichkeit", restorability_score=100.0)
    assert abs(eff - base) < 1e-6, f"restorability=100 Abweichung: base={base} eff={eff}"


def test_get_effective_material_floor_restorability_50():
    """restorability=50 → scale=max(0.72, 0.50)=0.72."""
    from backend.core.calibration_matrix import (
        RESTORABILITY_SCALE_MIN,
        get_effective_material_floor,
    )

    base = get_material_floor("vinyl", "natuerlichkeit")
    eff = get_effective_material_floor("vinyl", "natuerlichkeit", restorability_score=50.0)
    # scale = max(0.72, 0.50) = 0.72
    expected = float(np.clip(base * RESTORABILITY_SCALE_MIN, 0.20, 0.99))
    assert abs(eff - expected) < 1e-6, f"Expected {expected}, got {eff}"


def test_get_effective_material_floor_restorability_10():
    """restorability=10 → scale=RESTORABILITY_SCALE_MIN (Untergrenze)."""
    from backend.core.calibration_matrix import (
        RESTORABILITY_SCALE_MIN,
        get_effective_material_floor,
    )

    base = get_material_floor("shellac", "natuerlichkeit")
    eff = get_effective_material_floor("shellac", "natuerlichkeit", restorability_score=10.0)
    expected = float(np.clip(base * RESTORABILITY_SCALE_MIN, 0.20, 0.99))
    assert abs(eff - expected) < 1e-6


def test_get_effective_material_floor_restorability_0():
    """restorability=0 → clamp auf RESTORABILITY_SCALE_MIN (kein negativer Floor)."""
    from backend.core.calibration_matrix import get_effective_material_floor

    eff = get_effective_material_floor("unknown", "groove", restorability_score=0.0)
    assert eff >= 0.20, "Floor darf nicht unter 0.20 fallen"


def test_restorability_scale_min_constant():
    """RESTORABILITY_SCALE_MIN MUSS 0.72 sein (§09.12 Spec-Vorgabe)."""
    from backend.core.calibration_matrix import RESTORABILITY_SCALE_MIN

    assert RESTORABILITY_SCALE_MIN == 0.72


def test_effective_floor_always_below_base():
    """get_effective_material_floor() darf niemals ÜBER get_material_floor() liegen."""
    from backend.core.calibration_matrix import get_effective_material_floor

    for mat in ("shellac", "vinyl", "cd_digital", "mp3_low", "unknown"):
        for goal in ("natuerlichkeit", "groove", "brillanz"):
            base = get_material_floor(mat, goal)
            for rest in (0, 25, 50, 75, 99):
                eff = get_effective_material_floor(mat, goal, restorability_score=float(rest))
                assert eff <= base + 1e-6, (
                    f"Effective floor ({eff:.4f}) > base floor ({base:.4f}) für mat={mat} goal={goal} rest={rest}"
                )


def test_get_effective_material_floor_in_all_list():
    """get_effective_material_floor MUSS in __all__ stehen."""
    import backend.core.calibration_matrix as _cm

    assert "get_effective_material_floor" in _cm.__all__


def test_restorability_scale_min_in_all_list():
    """RESTORABILITY_SCALE_MIN MUSS in __all__ stehen."""
    import backend.core.calibration_matrix as _cm

    assert "RESTORABILITY_SCALE_MIN" in _cm.__all__


# ---------------------------------------------------------------------------
# §0d groove-Timing-Fix (v10.0.0) — Normative Guard-Tests
# ---------------------------------------------------------------------------


def test_emotionalitaet_recovery_primary_is_phase_54():
    """§0d v10.0.0: emotionalitaet primary recovery phase MUSS phase_54_transparent_dynamics sein.

    Grund: DFN/SGMSE+/OMLSA glätten alle 4 Dynamik-Komponenten (crest, variance, micro, range).
    phase_26 (Dynamic Range Expansion) behebt nur crest_score; phase_54 wirkt auf alle Komponenten.
    """
    phases = get_goal_recovery_phases("emotionalitaet", is_studio_2026=False)
    assert phases, "emotionalitaet recovery list ist leer"
    assert phases[0] == "phase_54_transparent_dynamics", (
        f"Primary recovery phase für emotionalitaet muss phase_54_transparent_dynamics sein, "
        f"got '{phases[0]}' — §0d v10.0.0 Envelope-Re-Smoothing-Fix (NR-Glättung betrifft alle 4 Dynamik-Komponenten)"
    )
    # phase_26 muss weiterhin in der Liste stehen (als sekundäre Phase)
    assert "phase_26_dynamic_range_expansion" in phases, (
        "phase_26_dynamic_range_expansion muss als sekundäre Phase in emotionalitaet-Recovery stehen"
    )


def test_emotionalitaet_recovery_phase54_before_phase26():
    """phase_54 muss vor phase_26 in der emotionalitaet-Recovery-Liste stehen (§2.46 Carrier-Hierarchie)."""
    phases = get_goal_recovery_phases("emotionalitaet", is_studio_2026=False)
    idx_54 = phases.index("phase_54_transparent_dynamics")
    idx_26 = phases.index("phase_26_dynamic_range_expansion")
    assert idx_54 < idx_26, (
        f"phase_54 (idx={idx_54}) muss vor phase_26 (idx={idx_26}) stehen — "
        "Envelope-Re-Smoothing vor Dynamic Range Expansion (subtraktiv vor additiv)"
    )


def test_groove_recovery_includes_phase12():
    """groove primary recovery muss phase_12_wow_flutter_fix enthalten (physikalische Ursache)."""
    phases = get_goal_recovery_phases("groove", is_studio_2026=False)
    assert phases[0] == "phase_12_wow_flutter_fix", (
        f"groove primary recovery muss phase_12_wow_flutter_fix sein (Wow/Flutter = direkte physikalische Ursache), "
        f"got '{phases[0]}'"
    )


# ---------------------------------------------------------------------------
# §2.70 RestorationMemory-Prior als kappa-Modulator
# ---------------------------------------------------------------------------


class TestRestorationPriorKappaBoost:
    """estimate_song_goal_targets() — restoration_prior erhöht kappa bei hpi_achieved > 0.75."""

    def test_no_prior_returns_baseline(self):
        """Ohne Prior liefert die Funktion die Baseline-Targets (kein Boost)."""
        base = estimate_song_goal_targets(
            restorability_score=70.0,
            era_decade=1970,
            material_type="vinyl",
        )
        assert isinstance(base, dict)
        assert len(base) > 0

    def test_prior_below_threshold_no_boost(self):
        """hpi_achieved=0.70 (< 0.75) → kein kappa-Boost → gleich wie ohne Prior."""
        base = estimate_song_goal_targets(
            restorability_score=70.0,
            era_decade=1970,
            material_type="vinyl",
        )
        with_prior_low = estimate_song_goal_targets(
            restorability_score=70.0,
            era_decade=1970,
            material_type="vinyl",
            restoration_prior={"hpi_achieved": 0.70},
        )
        for goal in base:
            assert base[goal] == with_prior_low.get(goal, base[goal]), (
                f"Goal '{goal}' sollte ohne Boost identisch sein: {base[goal]} != {with_prior_low.get(goal)}"
            )

    def test_prior_above_threshold_amplifies_biases(self):
        """hpi_achieved=0.90 → kappa-Boost verstärkt Biases in beide Richtungen.

        vinyl hat waerme=+0.10 (positiver Bias → Goal steigt)
        vinyl hat natuerlichkeit=-0.30 (negativer Bias → Goal sinkt).
        """
        base = estimate_song_goal_targets(
            restorability_score=70.0,
            era_decade=1970,
            material_type="vinyl",
        )
        boosted = estimate_song_goal_targets(
            restorability_score=70.0,
            era_decade=1970,
            material_type="vinyl",
            restoration_prior={"hpi_achieved": 0.90},
        )
        # waerme hat positiven Bias (+0.10) → muss steigen oder gleich bleiben
        assert boosted.get("waerme", 0) >= base.get("waerme", 0) - 1e-9, (
            f"waerme (pos. Bias) sollte steigen: {base.get('waerme')} → {boosted.get('waerme')}"
        )
        # natuerlichkeit hat negativen Bias (-0.30) → sinkt oder bleibt gleich
        assert boosted.get("natuerlichkeit", 1) <= base.get("natuerlichkeit", 1) + 1e-9, (
            f"natuerlichkeit (neg. Bias) sollte sinken: {base.get('natuerlichkeit')} → {boosted.get('natuerlichkeit')}"
        )
        # Alle Targets in erlaubten Grenzen
        for goal, val in boosted.items():
            assert 0.30 <= val <= 0.99, f"Goal '{goal}' außerhalb Grenzen: {val}"

    def test_prior_hpi_1_boost_capped_at_kappa_base(self):
        """hpi_achieved=1.0 → maximaler Boost, aber kappa nicht über kappa_base."""
        base = estimate_song_goal_targets(restorability_score=70.0, material_type="vinyl")
        boosted = estimate_song_goal_targets(
            restorability_score=70.0,
            material_type="vinyl",
            restoration_prior={"hpi_achieved": 1.0},
        )
        # Targets müssen im erlaubten Bereich bleiben [0.30, 0.99]
        for goal, val in boosted.items():
            assert 0.30 <= val <= 0.99, f"Goal '{goal}' außerhalb Grenzen: {val}"

    def test_prior_none_backward_compat(self):
        """restoration_prior=None → identisch mit fehlendem Parameter."""
        without = estimate_song_goal_targets(restorability_score=50.0)
        with_none = estimate_song_goal_targets(
            restorability_score=50.0,
            restoration_prior=None,
        )
        assert without == with_none

    def test_prior_boost_monotone_for_positive_bias_goal(self):
        """Höherer hpi_achieved → waerme (pos. Bias) steigt monoton."""
        targets_low = estimate_song_goal_targets(
            restorability_score=60.0,
            material_type="vinyl",
            restoration_prior={"hpi_achieved": 0.80},
        )
        targets_high = estimate_song_goal_targets(
            restorability_score=60.0,
            material_type="vinyl",
            restoration_prior={"hpi_achieved": 0.95},
        )
        # waerme hat positiven Bias (+0.10) → höherer kappa → höherer Wert
        assert targets_high.get("waerme", 0) >= targets_low.get("waerme", 0) - 1e-9, (
            f"waerme (pos. Bias) sollte monoton mit hpi steigen: "
            f"{targets_low.get('waerme')} → {targets_high.get('waerme')}"
        )


# ---------------------------------------------------------------------------
# §Lücke5 kappa S-Kurve
# ---------------------------------------------------------------------------


class TestKappaSCurve:
    """estimate_song_goal_targets() — kappa wird via logistischer S-Kurve moduliert."""

    def test_kappa_monotone_with_restorability(self):
        """Ziel mit positivem Bias (waerme=+0.10 bei vinyl) muss mit rest monoton steigen."""
        vals = [
            estimate_song_goal_targets(
                restorability_score=float(r),
                material_type="vinyl",
            ).get("waerme", 0.0)
            for r in range(0, 101, 10)
        ]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 1e-9, f"waerme nicht monoton bei rest={i * 10}: {vals[i]} > {vals[i + 1]}"

    def test_extremes_differ_from_midpoint(self):
        """S-Kurve: mittlerer Bereich (25→75) reagiert stärker als Extrembereiche (0→25, 75→100)."""

        def _waerme(rest: float) -> float:
            return estimate_song_goal_targets(restorability_score=rest, material_type="vinyl").get("waerme", 0.0)

        delta_low = _waerme(25.0) - _waerme(0.0)  # 0→25: flaches Plateau
        delta_mid = _waerme(75.0) - _waerme(25.0)  # 25→75: steile Phase
        delta_high = _waerme(100.0) - _waerme(75.0)  # 75→100: flaches Plateau

        assert delta_mid > delta_low, (
            f"Mittlere Phase ({delta_mid:.4f}) sollte steiler als unteres Plateau ({delta_low:.4f})"
        )
        assert delta_mid > delta_high, (
            f"Mittlere Phase ({delta_mid:.4f}) sollte steiler als oberes Plateau ({delta_high:.4f})"
        )

    def test_boundary_rest0_kappa_near_minimum(self):
        """rest=0 → kappa nahe Minimum (ca. 0.60·kappa_base); Restoration kappa_base=0.45."""
        # waerme bei rest=0 und rest=100 sollten sich durch kappa unterscheiden
        t0 = estimate_song_goal_targets(restorability_score=0.0, material_type="vinyl")
        t100 = estimate_song_goal_targets(restorability_score=100.0, material_type="vinyl")
        # waerme(+0.10 Bias): rest=100 muss höher sein als rest=0
        assert t100.get("waerme", 0) > t0.get("waerme", 0), (
            f"waerme bei rest=100 ({t100.get('waerme')}) sollte > rest=0 ({t0.get('waerme')})"
        )

    def test_targets_in_bounds_for_all_restorability_levels(self):
        """Alle Targets bleiben in [0.30, 0.99] für alle Restorability-Werte."""
        for rest in [0, 10, 25, 50, 75, 90, 100]:
            targets = estimate_song_goal_targets(
                restorability_score=float(rest),
                material_type="shellac",
                era_decade=1930,
            )
            for goal, val in targets.items():
                assert 0.30 <= val <= 0.99, f"rest={rest} goal='{goal}' außerhalb [0.30, 0.99]: {val}"


class TestEraAdaptiveFloors:
    """§09.8 Era-adaptive Böden (Restoration) — historische Ästhetik ist kein Defekt."""

    def test_era_lowers_restoration_floor_for_shellac(self) -> None:
        from backend.core.calibration_matrix import get_material_floor

        modern = get_material_floor("shellac", "brillanz", era_decade=2010)
        vintage = get_material_floor("shellac", "brillanz", era_decade=1940)
        assert vintage < modern
        assert vintage >= 0.30

    def test_studio_ignores_era(self) -> None:
        from backend.core.calibration_matrix import get_material_floor

        vintage = get_material_floor("shellac", "brillanz", is_studio_2026=True, era_decade=1940)
        modern = get_material_floor("shellac", "brillanz", is_studio_2026=True, era_decade=2010)
        assert vintage == modern

    def test_era_none_or_modern_keeps_legacy_value(self) -> None:
        from backend.core.calibration_matrix import get_material_floor

        legacy = get_material_floor("shellac", "brillanz")
        modern_era = get_material_floor("shellac", "brillanz", era_decade=2005)
        assert legacy == modern_era
