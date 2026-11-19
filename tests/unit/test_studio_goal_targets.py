import numpy as np
import pytest

from backend.core.studio_goal_targets import estimate_song_goal_targets


@pytest.mark.unit
def test_legacy_era_targets_reduce_brilliance_in_restoration():
    old = estimate_song_goal_targets(
        is_studio_2026=False,
        goal_weights={"brillanz": 1.2, "waerme": 1.0},
        restorability_score=60.0,
        era_decade=1935,
        genre_label="Jazz",
        material_type="shellac",
        transfer_chain=["shellac", "tape", "mp3_low"],
    )
    modern = estimate_song_goal_targets(
        is_studio_2026=False,
        goal_weights={"brillanz": 1.2, "waerme": 1.0},
        restorability_score=60.0,
        era_decade=2005,
        genre_label="Pop",
        material_type="cd_digital",
        transfer_chain=["cd_digital"],
    )

    assert old.targets["brillanz"] < modern.targets["brillanz"]
    assert old.targets["waerme"] > modern.targets["waerme"]


def test_genre_context_changes_spatial_depth_target():
    klassik = estimate_song_goal_targets(
        is_studio_2026=False,
        goal_weights={"spatial_depth": 1.0, "transparenz": 1.0},
        restorability_score=70.0,
        era_decade=1975,
        genre_label="Klassik",
        material_type="vinyl",
    )
    pop = estimate_song_goal_targets(
        is_studio_2026=False,
        goal_weights={"spatial_depth": 1.0, "transparenz": 1.0},
        restorability_score=70.0,
        era_decade=1975,
        genre_label="Pop",
        material_type="vinyl",
    )

    assert klassik.targets["spatial_depth"] > pop.targets["spatial_depth"]


def test_transfer_chain_depth_reduces_target_confidence():
    shallow = estimate_song_goal_targets(
        is_studio_2026=False,
        goal_weights={"transparenz": 1.0},
        restorability_score=65.0,
        era_decade=1985,
        genre_label="Rock",
        material_type="reel_tape",
        transfer_chain=["reel_tape"],
    )
    deep = estimate_song_goal_targets(
        is_studio_2026=False,
        goal_weights={"transparenz": 1.0},
        restorability_score=65.0,
        era_decade=1985,
        genre_label="Rock",
        material_type="reel_tape",
        transfer_chain=["shellac", "reel_tape", "cassette", "cd_digital", "mp3_low"],
    )

    assert deep.confidence < shallow.confidence
    assert np.isfinite(deep.confidence)


def test_targets_expose_derived_tcci_and_ibs():
    result = estimate_song_goal_targets(
        is_studio_2026=False,
        goal_weights={"transparenz": 1.1},
        restorability_score=55.0,
        era_decade=1960,
        genre_label="Rock",
        material_type="vinyl",
        transfer_chain=["shellac", "reel_tape", "cassette", "cd_digital", "mp3_low"],
    )

    assert result.derived is not None
    assert 0.0 <= float(result.derived.get("tcci", -1.0)) <= 1.0
    assert 0.15 <= float(result.derived.get("ibs", -1.0)) <= 0.95


def test_high_complexity_chain_pulls_targets_toward_floors():
    simple = estimate_song_goal_targets(
        is_studio_2026=False,
        goal_weights={"brillanz": 1.4, "transparenz": 1.4},
        restorability_score=75.0,
        era_decade=1998,
        genre_label="Pop",
        material_type="cd_digital",
        transfer_chain=["cd_digital"],
    )
    complex_chain = estimate_song_goal_targets(
        is_studio_2026=False,
        goal_weights={"brillanz": 1.4, "transparenz": 1.4},
        restorability_score=75.0,
        era_decade=1998,
        genre_label="Pop",
        material_type="cd_digital",
        transfer_chain=["shellac", "reel_tape", "cassette", "cd_digital", "mp3_low"],
    )

    assert complex_chain.targets["brillanz"] <= simple.targets["brillanz"]
    assert complex_chain.targets["transparenz"] <= simple.targets["transparenz"]
