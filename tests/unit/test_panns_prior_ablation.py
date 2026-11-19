from __future__ import annotations

"""Ablation tests for PANNs-prior fusion in GenreClassifier (§open-set rescue).

Three properties are verified:
1. _fuse_non_schlager_with_panns() never reduces DSP scores (max-merge invariant).
2. _panns_open_set_rescue() correctly rescues unambiguous high-confidence priors.
3. Schlager outcome is immune to PANNs prior (open-set rescue must not fire for Schlager).

Literature: Kong et al. (2020) PANNs; Won et al. (2020) – genre-tag correlation.
"""


import numpy as np
import pytest

from backend.core.genre_classifier import GermanSchlagerClassifier as GenreClassifier
from backend.core.genre_classifier import get_genre_classifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def clf() -> GenreClassifier:
    return get_genre_classifier()


# ---------------------------------------------------------------------------
# 1. Fuse: max-merge invariant — PANNs can only raise, never lower DSP scores
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPANNsFuseInvariant:
    """_fuse_non_schlager_with_panns must keep fused >= original at all times."""

    def test_fuse_never_reduces_dsp_score(self, clf):
        """25% PANNs weight must not pull fused score below the original DSP value."""
        dsp = {"Rock": 0.80, "Jazz": 0.55, "Klassik": 0.30, "Electronic": 0.20, "Pop": 0.70}
        # PANNs gives very low Rock/Pop, high Electronic
        panns = {"Rock": 0.10, "Jazz": 0.60, "Klassik": 0.90, "Electronic": 0.95, "Pop": 0.05}
        fused = clf._fuse_non_schlager_with_panns(dsp, panns)
        for genre, orig in dsp.items():
            assert fused[genre] >= orig - 1e-9, f"Fuse reduced {genre}: {orig:.3f} → {fused[genre]:.3f}"

    def test_fuse_can_raise_low_dsp_score(self, clf):
        """PANNs ≥ 0.8 on a low-scoring genre should lift fused score."""
        dsp = {"Klassik": 0.10}
        panns = {"Klassik": 0.90}
        fused = clf._fuse_non_schlager_with_panns(dsp, panns)
        # 0.75*0.10 + 0.25*0.90 = 0.30; max(0.10, 0.30) = 0.30
        assert fused["Klassik"] > 0.10, "PANNs should raise a low DSP Klassik score"
        assert fused["Klassik"] <= 0.30 + 1e-9  # max possible blended value

    def test_fuse_empty_panns_returns_original(self, clf):
        """Empty PANNs prior → fused identical to DSP."""
        dsp = {"Rock": 0.55, "Jazz": 0.40}
        fused = clf._fuse_non_schlager_with_panns(dsp, {})
        for genre in dsp:
            assert fused[genre] == pytest.approx(dsp[genre], abs=1e-9)

    def test_fuse_partial_panns_keys(self, clf):
        """Only genres present in DSP should appear in fused output."""
        dsp = {"Rock": 0.50, "Jazz": 0.45, "Pop": 0.30}
        panns = {"Rock": 0.80}  # only Rock has PANNs evidence
        fused = clf._fuse_non_schlager_with_panns(dsp, panns)
        assert set(fused.keys()) == set(dsp.keys())
        # Rock should be lifted, Jazz/Pop unchanged
        assert fused["Rock"] >= dsp["Rock"] - 1e-9
        assert fused["Jazz"] == pytest.approx(dsp["Jazz"], abs=1e-9)
        assert fused["Pop"] == pytest.approx(dsp["Pop"], abs=1e-9)

    def test_fuse_max_25pct_lift_when_both_equal(self, clf):
        """If DSP == PANNs on a genre, fused == DSP (no distortion)."""
        dsp = {"Electronic": 0.65}
        panns = {"Electronic": 0.65}
        fused = clf._fuse_non_schlager_with_panns(dsp, panns)
        assert fused["Electronic"] == pytest.approx(0.65, abs=0.01)


# ---------------------------------------------------------------------------
# 2. Open-set rescue: unambiguous PANNs signal rescues "Unbekannt"
# ---------------------------------------------------------------------------


class TestPANNsOpenSetRescue:
    """_panns_open_set_rescue must return (genre, conf) for clear evidence."""

    def test_rescue_returns_genre_on_high_unambiguous_prior(self, clf):
        """Single genre > 0.60 with > 0.20 margin from #2 → rescue."""
        panns = {"Jazz": 0.82, "Rock": 0.30, "Klassik": 0.15, "Electronic": 0.10, "Pop": 0.20}
        genre, conf = clf._panns_open_set_rescue(panns)
        assert genre == "Jazz", f"Expected Jazz, got {genre!r}"
        assert 0.40 <= conf <= 0.55, f"Confidence {conf:.3f} out of expected range [0.40, 0.55]"

    def test_rescue_requires_min_0_60(self, clf):
        """PANNs below 0.60 must not trigger rescue."""
        panns = {"Rock": 0.55, "Jazz": 0.10, "Klassik": 0.05, "Electronic": 0.05, "Pop": 0.05}
        genre, conf = clf._panns_open_set_rescue(panns)
        assert genre == "", f"Should be empty, got {genre!r}"
        assert conf == pytest.approx(0.0, abs=1e-9)

    def test_rescue_requires_unambiguous_margin(self, clf):
        """Two genres close together (< 0.20 margin) must not trigger rescue."""
        panns = {"Rock": 0.70, "Jazz": 0.65, "Klassik": 0.20}
        genre, conf = clf._panns_open_set_rescue(panns)
        assert genre == "", f"Ambiguous priors should yield empty rescue, got {genre!r}"

    def test_rescue_empty_prior_returns_empty(self, clf):
        """Empty prior → always returns empty rescue."""
        genre, conf = clf._panns_open_set_rescue({})
        assert genre == ""
        assert conf == pytest.approx(0.0)

    def test_rescue_confidence_in_valid_range(self, clf):
        """Returned confidence must always be in [0.40, 0.50]."""
        for score in [0.60, 0.75, 0.90, 1.00]:
            panns = {"Electronic": score, "Rock": 0.10}
            genre, conf = clf._panns_open_set_rescue(panns)
            if genre:
                assert 0.40 <= conf <= 0.50, f"conf={conf:.3f} at best_score={score} out of [0.40, 0.50]"

    def test_rescue_unknown_key_returns_empty(self, clf):
        """Keys not in internal registry (e.g. 'Hiphop') yield no rescue."""
        panns = {"Hiphop": 0.95, "Schlager": 0.05}
        genre, conf = clf._panns_open_set_rescue(panns)
        assert genre == ""


# ---------------------------------------------------------------------------
# 3. Schlager immunity — PANNs must not override is_schlager=True decision
# ---------------------------------------------------------------------------


class TestSchlagerImmunityToPANNs:
    """When DSP classifies audio as Schlager, PANNs prior must not override."""

    def test_schlager_stays_schlager_with_high_panns_jazz(self, monkeypatch):
        """Even if PANNs reports Jazz=0.90 with high confidence, Schlager must win."""
        clf = get_genre_classifier()
        monkeypatch.setattr(clf, "_is_music_like", lambda _a: True)
        # Activate multiple Schlager tiers
        monkeypatch.setattr(clf, "_compute_clap_score", lambda _a, _sr: 0.80)
        monkeypatch.setattr(clf, "_compute_accordion_score", lambda _a, _sr: 0.80)
        monkeypatch.setattr(clf, "_compute_harmonic_simplicity", lambda _a, _sr: 0.72)
        monkeypatch.setattr(clf, "_classify_rhythm_pattern", lambda _a, _sr: (0.80, "schlager", 120.0))
        monkeypatch.setattr(clf, "_compute_german_vocal_prior", lambda _a, _sr: 0.80)
        monkeypatch.setattr(clf, "_compute_melodic_repetition", lambda _a, _sr: 0.80)
        monkeypatch.setattr(clf, "_detect_vocal_language", lambda _a, _sr: 0.80)
        monkeypatch.setattr(clf, "_compute_lyrics_language_hint", lambda _a, _sr: 0.0)
        monkeypatch.setattr(clf, "_estimate_key", lambda _a, _sr: "C-Dur")
        monkeypatch.setattr(clf, "_spectral_centroid_hz", lambda _a, _sr: 2200.0)
        monkeypatch.setattr(clf, "_onset_rate", lambda _a, _sr: 2.0)
        monkeypatch.setattr(clf, "_dynamic_range_db", lambda _a, _sr: 35.0)
        # Make non-Schlager scores low so they don't accidentally win
        monkeypatch.setattr(clf, "_score_rock", lambda *_a, **_k: 0.05)
        monkeypatch.setattr(clf, "_score_jazz", lambda *_a, **_k: 0.05)
        monkeypatch.setattr(clf, "_score_classical", lambda *_a, **_k: 0.05)
        for m in (
            "_score_pop",
            "_score_blues",
            "_score_soul_rnb",
            "_score_country",
            "_score_folk",
            "_score_funk",
            "_score_electronic",
            "_score_hiphop",
            "_score_metal",
            "_score_latin",
            "_score_gospel",
            "_score_reggae",
        ):
            monkeypatch.setattr(clf, m, lambda *_a, **_k: 0.05)
        # PANNs returns Jazz=0.95 — should be ignored for the Schlager decision
        monkeypatch.setattr(clf, "_compute_panns_genre_prior", lambda _a, _sr: {"Jazz": 0.95, "Rock": 0.05})

        audio = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000).astype(np.float32)
        result = clf.classify(audio, 48000)
        assert result.is_schlager is True, (
            f"PANNs high Jazz prior must not override Schlager (got genre={result.genre_label!r})"
        )
        assert "schlager" in result.genre_label.lower(), (
            f"genre_label must be Schlager variant, not {result.genre_label!r}"
        )
