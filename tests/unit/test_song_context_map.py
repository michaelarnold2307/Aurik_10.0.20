from __future__ import annotations

"""Unit-Tests für backend/core/song_context_map.py (Spec 03 §SCM).

Die Eingaben werden von den ECHTEN Produzenten erzeugt (VFAResult.to_dict,
HarmonicContextResult.to_dict, SongSegment) — so bricht der Test, wenn sich
deren Schlüsselnamen ändern, statt gegen erfundene Namen zu laufen.
"""

import copy
import json
import math

import numpy as np
import pytest

from backend.core.harmonic_context_analyzer import HarmonicContextResult
from backend.core.song_context_map import DIMENSIONS, SCHEMA, TOP_TAGS, build_song_context_map
from backend.core.song_structure_analyzer import SongSegment
from backend.core.vocal_focus_analyzer import VFAResult

pytestmark = [pytest.mark.unit]


def _full_context() -> dict:
    hca = HarmonicContextResult(
        chord_sequence=["Am", "F", "C"],
        chord_confidence=np.array([0.9, 0.8, 0.7], dtype=np.float32),
        key_root="A",
        key_mode="min",
        modulation_frames=[1],
        harmonic_density=np.array([0.5, 0.6, 0.7], dtype=np.float32),
        harmonic_mask=np.ones((4, 3), dtype=bool),
        analysis_confidence=0.62,
    )
    vfa = VFAResult(
        panns_singing=0.81,
        vocal_present=True,
        dominant_register="head",
        singer_school="schlager",
        vibrato_zones=[(1.0, 2.0)],
        tension_zones=[(3.0, 4.0), (5.0, 6.0)],
        release_zones=[(6.0, 7.0)],
        frisson_zones=[(3.5, 3.9)],
        whisper_zones=[],
        climax_type="peak",
        style_confidence=0.2,
    )
    return {
        "decade": 1970,
        "era_confidence": 0.74,
        "primary_material": "vinyl",
        "genre_label": "Schlager",
        "subgenre": "pop",
        "bpm": 98.0,
        "vfa_result": vfa.to_dict(),
        "harmonic_context": hca.to_dict(),
        "panns_tags": {"Singing voice": 0.9, "Guitar": 0.4, "Music": 0.95, "Drum": 0.1, "Bass": 0.3, "Piano": 0.2},
        "arc_protection_weights": object(),
    }


def _segments() -> list[SongSegment]:
    return [
        SongSegment(0.0, 10.0, "intro", 0.2, False, False),
        SongSegment(10.0, 40.0, "verse", 0.5, True, False),
        SongSegment(40.0, 60.0, "chorus", 0.9, True, True),
    ]


def test_full_context_measures_every_dimension() -> None:
    result = build_song_context_map(
        _full_context(), structure_segments=_segments(), structure_bpm=49.0, analysis_window_s=60.0
    ).to_dict()
    assert result["schema"] == SCHEMA
    assert result["report_only"] is True
    assert tuple(result["dimensions"]) == DIMENSIONS
    assert result["coverage"] == 1.0
    assert result["unavailable"] == []
    assert all(d["state"] == "measured" for d in result["dimensions"].values())


def test_missing_everything_is_unavailable_not_zero() -> None:
    result = build_song_context_map({}).to_dict()
    assert result["coverage"] == 0.0
    assert result["weakest_confidence"] is None
    assert set(result["unavailable"]) == set(DIMENSIONS)
    for dim in result["dimensions"].values():
        assert dim["state"] == "unavailable"
        assert dim["value"] == {}
        assert dim["confidence"] is None


def test_none_context_is_handled() -> None:
    assert build_song_context_map(None).coverage == 0.0


def test_confidence_comes_only_from_analyzer_fields() -> None:
    dims = build_song_context_map(_full_context(), structure_segments=_segments(), structure_bpm=49.0).to_dict()[
        "dimensions"
    ]
    assert dims["provenance"]["confidence"] == pytest.approx(0.74)
    assert dims["harmony"]["confidence"] == pytest.approx(0.62)
    # Analysatoren ohne eigenes Konfidenzfeld ⇒ None, nie geschätzt.
    for name in ("voice", "instruments", "rhythm", "structure", "dynamics_arc"):
        assert dims[name]["confidence"] is None, name


def test_style_confidence_is_not_mislabeled_as_voice_confidence() -> None:
    voice = build_song_context_map(_full_context()).to_dict()["dimensions"]["voice"]
    assert voice["confidence"] is None
    assert voice["value"]["style_intent_share"] == pytest.approx(0.2)
    assert voice["value"]["panns_singing"] == pytest.approx(0.81)


def test_era_confidence_default_is_not_a_measurement_without_decade() -> None:
    ctx = {"decade": None, "era_confidence": 0.0, "genre_label": "Schlager", "primary_material": ""}
    dim = build_song_context_map(ctx).to_dict()["dimensions"]["provenance"]
    assert dim["state"] == "measured"
    assert dim["confidence"] is None


def test_unknown_genre_is_not_reported_as_genre() -> None:
    ctx = {"decade": 1980, "era_confidence": 0.5, "genre_label": "Unbekannt", "subgenre": "x"}
    value = build_song_context_map(ctx).to_dict()["dimensions"]["provenance"]["value"]
    assert value["genre"] is None
    assert value["subgenre"] is None


def test_empty_vfa_failure_marker_is_unavailable() -> None:
    # UV3 setzt bei VFA-Fehler setdefault("vfa_result", {}).
    dims = build_song_context_map({"vfa_result": {}}).to_dict()["dimensions"]
    assert dims["voice"]["state"] == "unavailable"
    assert dims["dynamics_arc"]["state"] == "unavailable"


def test_instrument_tags_top_k_is_stable_and_ignores_invalid_values() -> None:
    tags = {"A": 0.5, "B": 0.5, "C": float("nan"), "D": 0.9, "E": True, "F": 0.1, "G": 0.2, "H": 0.3, "I": 0.4}
    value = build_song_context_map({"panns_tags": tags}).to_dict()["dimensions"]["instruments"]["value"]
    labels = [t["label"] for t in value["tags"]]
    assert len(labels) == TOP_TAGS
    assert labels == ["D", "A", "B", "I", "H"]  # Gleichstand A/B: Label aufsteigend
    assert "C" not in labels and "E" not in labels


def test_tags_without_valid_scores_are_unavailable() -> None:
    dim = build_song_context_map({"panns_tags": {"A": float("inf"), "B": "x"}}).to_dict()["dimensions"]["instruments"]
    assert dim["state"] == "unavailable"


def test_rhythm_reports_both_estimates_and_ratio_without_verdict() -> None:
    value = build_song_context_map({"bpm": 98.0}, structure_bpm=49.0).to_dict()["dimensions"]["rhythm"]["value"]
    assert value["bpm_structure"] == 49.0
    assert value["bpm_genre_classifier"] == 98.0
    assert value["bpm_ratio"] == 2.0
    assert "agree" not in value


def test_rhythm_ignores_non_positive_and_non_finite_bpm() -> None:
    dim = build_song_context_map({"bpm": 0.0}, structure_bpm=float("nan")).to_dict()["dimensions"]["rhythm"]
    assert dim["state"] == "unavailable"
    only = build_song_context_map({"bpm": 0.0}, structure_bpm=120.0).to_dict()["dimensions"]["rhythm"]["value"]
    assert only["bpm_genre_classifier"] is None
    assert "bpm_ratio" not in only


def test_structure_summary_from_real_segments() -> None:
    value = build_song_context_map({}, structure_segments=_segments()).to_dict()["dimensions"]["structure"]["value"]
    assert value["n_segments"] == 3
    assert value["n_climax"] == 1
    assert value["covered_s"] == 60.0
    assert value["vocal_fraction"] == pytest.approx(50.0 / 60.0, abs=1e-3)
    assert [s["label"] for s in value["segments"]] == ["intro", "verse", "chorus"]


def test_structure_skips_invalid_segments_and_fails_closed_when_none_valid() -> None:
    bad = [SongSegment(5.0, 1.0, "verse", 0.5, True, False), SongSegment(float("nan"), 2.0, "x", 0.1, False, False)]
    assert build_song_context_map({}, structure_segments=bad).to_dict()["dimensions"]["structure"]["state"] == (
        "unavailable"
    )


def test_unknown_scope_is_rejected_fail_closed() -> None:
    with pytest.raises(ValueError):
        build_song_context_map({}, scope="album")


def test_scope_and_window_are_reported_honestly() -> None:
    result = build_song_context_map({}, scope="chunk", analysis_window_s=12.3456).to_dict()
    assert result["scope"] == "chunk"
    assert result["analysis_window_s"] == 12.35
    assert build_song_context_map({}, analysis_window_s=float("inf")).to_dict()["analysis_window_s"] is None


def test_weakest_confidence_only_among_measured_with_confidence() -> None:
    weakest = build_song_context_map(_full_context()).to_dict()["weakest_confidence"]
    assert weakest == {"dimension": "harmony", "confidence": pytest.approx(0.62)}


def test_result_is_json_serializable_finite_and_deterministic() -> None:
    first = build_song_context_map(_full_context(), structure_segments=_segments(), structure_bpm=49.0).to_dict()
    second = build_song_context_map(_full_context(), structure_segments=_segments(), structure_bpm=49.0).to_dict()
    assert first == second
    text = json.dumps(first, allow_nan=False)  # NaN/Inf wären ein Fehler
    assert math.isfinite(json.loads(text)["coverage"])


def test_inputs_are_not_mutated() -> None:
    ctx = _full_context()
    ctx.pop("arc_protection_weights")  # object() ist nicht deepcopy-vergleichbar
    segments = _segments()
    ctx_before, segments_before = copy.deepcopy(ctx), copy.deepcopy(segments)
    build_song_context_map(ctx, structure_segments=segments, structure_bpm=49.0)
    assert ctx == ctx_before
    assert segments == segments_before


def test_dimension_failure_is_reported_not_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    class _Boom(dict):
        def get(self, *_a, **_k):  # type: ignore[no-untyped-def]
            raise RuntimeError("kaputt")

    with caplog.at_level("WARNING"):
        result = build_song_context_map(_Boom()).to_dict()
    assert result["coverage"] == 0.0
    assert any("nicht ableitbar" in r.message for r in caplog.records)
    failed = {name for name, d in result["dimensions"].items() if "Ableitung fehlgeschlagen" in d["note"]}
    # „structure“ liest den Kontext nicht und ist daher regulär „keine Segmentstruktur“.
    assert failed == set(DIMENSIONS) - {"structure"}
