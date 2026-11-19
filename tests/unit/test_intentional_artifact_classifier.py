from __future__ import annotations

"""
tests/unit/test_intentional_artifact_classifier.py
Aurik 10.0.0 — Spec §6.5 Tests: IntentionalArtifactClassifier
"""


import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clf():
    from backend.core.intentional_artifact_classifier import get_intentional_artifact_classifier

    return get_intentional_artifact_classifier()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_singleton_returns_same_instance():
    from backend.core.intentional_artifact_classifier import get_intentional_artifact_classifier

    a = get_intentional_artifact_classifier()
    b = get_intentional_artifact_classifier()
    assert a is b


# ---------------------------------------------------------------------------
# AUTHENTIC_CHARACTER dict
# ---------------------------------------------------------------------------


def test_authentic_character_has_shellac():
    from backend.core.intentional_artifact_classifier import AUTHENTIC_CHARACTER

    assert "shellac" in AUTHENTIC_CHARACTER
    assert "surface_noise_texture" in AUTHENTIC_CHARACTER["shellac"]
    assert AUTHENTIC_CHARACTER["shellac"]["surface_noise_texture"] == "PRESERVE"


def test_authentic_character_has_vinyl():
    from backend.core.intentional_artifact_classifier import AUTHENTIC_CHARACTER

    assert "vinyl" in AUTHENTIC_CHARACTER
    assert "riaa_warmth_curve" in AUTHENTIC_CHARACTER["vinyl"]
    assert AUTHENTIC_CHARACTER["vinyl"]["riaa_warmth_curve"] == "PRESERVE"


def test_authentic_character_has_mp3_low():
    from backend.core.intentional_artifact_classifier import AUTHENTIC_CHARACTER

    assert "mp3_low" in AUTHENTIC_CHARACTER
    assert AUTHENTIC_CHARACTER["mp3_low"]["severe_pre_echo"] == "REPAIR"
    assert AUTHENTIC_CHARACTER["mp3_low"]["pre_echo_character_mild"] == "PRESERVE"


def test_authentic_character_has_all_required_materials():
    from backend.core.intentional_artifact_classifier import AUTHENTIC_CHARACTER

    required = {"shellac", "vinyl", "tape", "reel_tape", "wax_cylinder", "cd_digital", "mp3_low", "lacquer_disc"}
    assert required.issubset(set(AUTHENTIC_CHARACTER.keys()))


def test_authentic_character_reel_tape_print_through_is_repair():
    from backend.core.intentional_artifact_classifier import AUTHENTIC_CHARACTER

    assert AUTHENTIC_CHARACTER["reel_tape"]["print_through_ghost"] == "REPAIR"


def test_authentic_character_reel_tape_tape_hiss_is_preserve():
    from backend.core.intentional_artifact_classifier import AUTHENTIC_CHARACTER

    assert AUTHENTIC_CHARACTER["reel_tape"]["tape_hiss_floor_texture"] == "PRESERVE"


# ---------------------------------------------------------------------------
# §6.5b Classify + Strength Caps
# ---------------------------------------------------------------------------


def test_classify_shellac_has_preserve_features(clf):
    result = clf.classify("shellac")
    assert len(result.preserve_features) > 0
    assert "surface_noise_texture" in result.preserve_features


def test_classify_shellac_no_repair_features(clf):
    result = clf.classify("shellac")
    # Shellac hat nur PRESERVE-Merkmale
    assert len(result.repair_features) == 0


def test_classify_reel_tape_has_repair_features(clf):
    result = clf.classify("reel_tape")
    assert "print_through_ghost" in result.repair_features
    assert "tape_head_clog" in result.repair_features


def test_preserve_strength_cap_is_0_10(clf):
    result = clf.classify("shellac")
    cap = result.should_cap_strength("surface_noise_texture")
    from backend.core.intentional_artifact_classifier import PRESERVE_MAX_STRENGTH

    assert cap == PRESERVE_MAX_STRENGTH
    assert cap == pytest.approx(0.10)


def test_repair_strength_cap_is_1_0(clf):
    result = clf.classify("mp3_low")
    cap = result.should_cap_strength("severe_pre_echo")
    assert cap == pytest.approx(1.0)


def test_exception_active_raises_preserve_cap_to_0_20(clf):
    result = clf.classify("shellac", artifact_freedom=0.85)
    assert result.artifact_freedom_override is True
    cap = result.should_cap_strength("surface_noise_texture")
    from backend.core.intentional_artifact_classifier import PRESERVE_EXCEPTION_MAX_STRENGTH

    assert cap == PRESERVE_EXCEPTION_MAX_STRENGTH
    assert cap == pytest.approx(0.20)


def test_no_exception_when_artifact_freedom_above_0_90(clf):
    result = clf.classify("shellac", artifact_freedom=0.92)
    assert result.artifact_freedom_override is False
    cap = result.should_cap_strength("surface_noise_texture")
    assert cap == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# get_strength_cap (schneller Einzelabruf)
# ---------------------------------------------------------------------------


def test_get_strength_cap_shellac_high_freq_noise():
    from backend.core.intentional_artifact_classifier import get_intentional_artifact_classifier

    clf = get_intentional_artifact_classifier()
    cap = clf.get_strength_cap("shellac", "HIGH_FREQ_NOISE")
    assert cap == pytest.approx(0.10)


def test_get_strength_cap_vinyl_no_mapping_returns_1_0():
    from backend.core.intentional_artifact_classifier import get_intentional_artifact_classifier

    clf = get_intentional_artifact_classifier()
    # CLICKS ist kein PRESERVE-Merkmal für vinyl
    cap = clf.get_strength_cap("vinyl", "IMPULSE_NOISE_CLICK")
    assert cap == pytest.approx(1.0)


def test_get_strength_cap_mp3_low_pre_echo_is_0_10():
    from backend.core.intentional_artifact_classifier import get_intentional_artifact_classifier

    clf = get_intentional_artifact_classifier()
    # PRE_ECHO → pre_echo_character_mild → PRESERVE für mp3_low
    cap = clf.get_strength_cap("mp3_low", "PRE_ECHO")
    assert cap == pytest.approx(0.10)


def test_get_strength_cap_unknown_material_returns_1_0():
    from backend.core.intentional_artifact_classifier import get_intentional_artifact_classifier

    clf = get_intentional_artifact_classifier()
    cap = clf.get_strength_cap("aac_streaming", "HIGH_FREQ_NOISE")
    assert cap == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Era-Profile
# ---------------------------------------------------------------------------


def test_classify_with_era_1935_adds_era_features(clf):
    result = clf.classify("shellac", era_decade=1935)
    # Era 1925–1945 fügt Röhren-H2/H4-PRESERVE hinzu
    assert "tube_h2_h4" in result.preserve_features


def test_classify_with_era_2015_adds_era_features(clf):
    result = clf.classify("cd_digital", era_decade=2015)
    # Era 2010+ fügt loudness_war_clip (REPAIR) hinzu
    assert "loudness_war_clip" in result.repair_features


def test_era_features_do_not_override_material_features(clf):
    # Material-Merkmale haben Vorrang; Era soll sie nicht überschreiben
    result = clf.classify("shellac", era_decade=1940)
    # surface_noise_texture bleibt PRESERVE (aus Material), nicht von Era überschrieben
    assert "surface_noise_texture" in result.preserve_features


# ---------------------------------------------------------------------------
# PRESERVE_MAX_STRENGTH + PRESERVE_EXCEPTION_MAX_STRENGTH Konstanten
# ---------------------------------------------------------------------------


def test_preserve_max_strength_constant():
    from backend.core.intentional_artifact_classifier import PRESERVE_MAX_STRENGTH

    assert pytest.approx(0.10) == PRESERVE_MAX_STRENGTH


def test_preserve_exception_max_strength_constant():
    from backend.core.intentional_artifact_classifier import PRESERVE_EXCEPTION_MAX_STRENGTH

    assert pytest.approx(0.20) == PRESERVE_EXCEPTION_MAX_STRENGTH
