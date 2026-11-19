"""§P0-1 Slice B (2026-11-05): Song-Level Perceptual-Gates statt per-Chunk.

Scope: die song-globalen Größen HPI (§2.44), EmotionalArc (§8.2/§2.45b) und
VQI + Singer-ID (§2.35c/§0p) laufen im Chunked-Pfad EINMAL auf dem
assemblierten Song (``_restore_chunked``) statt je 30-s-Chunk; nur
lokal-stationäre DSP/ML-Phasen bleiben je Chunk (§V7 (copilot-instructions.md)).

Test-Ebenen:
- Source-Wiring (wie ``test_p0_1_end_gate_chunking.py``): Deferral-Guards im
  ``restore()``-Tail und Song-Level-Aufruf in ``_restore_chunked``.
- Statische Signaturprüfung gegen die echten Modul-Signaturen.
- Synthetische Signale für Determinismus/NaN-Inf-Schutz der EAP-Module.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

_UV3_PATH = Path(__file__).resolve().parents[2] / "backend" / "core" / "unified_restorer_v3.py"


def _uv3_source() -> str:
    return _UV3_PATH.read_text(encoding="utf-8")


def _restore_chunked_body(src: str) -> str:
    """Rumpf von ``_restore_chunked`` isolieren (Muster aus test_p0_1_end_gate_chunking)."""
    assert "def _restore_chunked" in src
    start = src.index("def _restore_chunked")
    rest = src[start:]
    end = rest.find("\n    def ")
    body = rest if end == -1 else rest[:end]
    assert len(body) > 1000, "Rumpf nicht gefunden — Anker 'def _restore_chunked' pruefen"
    return body


def _song_level_block(src: str) -> str:
    """Der Slice-B-Song-Level-Block in ``_restore_chunked`` (zwischen zwei Ankern)."""
    start_anchor = "# §P0-1 Slice B (2026-11-05): Song-Level Perceptual-Gates"
    end_anchor = "# §P1-8 (2026-09-08): Nach der song-globalen m1b"
    assert start_anchor in src, "Song-Level-Block-Anker fehlt"
    assert end_anchor in src, "End-Anker §P1-8 fehlt"
    start = src.index(start_anchor)
    end = src.index(end_anchor, start)
    block = src[start:end]
    assert len(block) > 3000, "Song-Level-Block zu klein — Anker pruefen"
    return block


# ---------------------------------------------------------------------------
# 1) Song-Gate läuft einmal pro Song statt pro Chunk
# ---------------------------------------------------------------------------


def test_hpi_export_gate_runs_once_per_song_not_per_chunk() -> None:
    """§2.44 HPI: im Chunked-Pfad übersprungen, in _restore_chunked genau einmal."""
    src = _uv3_source()
    body = _restore_chunked_body(src)
    # Per-Chunk-Auswertung ist an den Skip-Log gebunden (Guard existiert).
    assert "§P0-1 Slice B: HPI-Ausgabe-Gate übersprungen (Chunk)" in src
    # Song-Level: der Gate wird genau EINMAL beschafft und ausgewertet
    # (Studio- und Restoration-Branch sind mutually exclusive).
    assert body.count("_p01_hg = _p01_get_hg()") == 1, "HPI-Song-Level-Verdrahtung muss genau 1× vorkommen"
    assert body.count("_p01_hg.evaluate_") == 2, "exakt Studio- + Restoration-Branch erwartet"
    # Und er liegt NACH der Assembly (Anker cp.verify_crossfade liegt davor).
    assert body.index("cp.verify_crossfade") < body.index("_p01_hg.evaluate_")


def test_final_hpi_reevaluation_is_deferred_in_the_chunked_path() -> None:
    """Finale HPI-Neubewertung (HPI_FAIL_FINAL) ist Teil des Song-Gates."""
    src = _uv3_source()
    assert "§P0-1 Slice B: Finale HPI-Neubewertung übersprungen (Chunk)" in src
    # _hpi_result_final ist vor dem Guard initialisiert (kein NameError im Skip-Zweig).
    assert "_hpi_result_final = None" in src


def test_emotionalarc_measure_and_correction_run_once_per_song() -> None:
    """§8.2 EmotionalArc: Messung + Makro-Korrektur einmal auf dem assemblierten Song."""
    src = _uv3_source()
    body = _restore_chunked_body(src)
    # Per-Chunk-Post-MDEM wird auf Nicht-Letzten-Chunks deferriert (None-Guard).
    assert "_arc_post_mdem = None" in src
    assert "§P0-1 Slice B: EmotionalArc Post-MDEM übersprungen" in src
    # Song-Level: genau eine Messung, nach der Assembly.
    assert body.count("_p01_measure_arc(") == 1
    assert body.index("cp.verify_crossfade") < body.index("_p01_measure_arc(")


def test_vqi_gate_runs_once_per_song_not_per_chunk() -> None:
    """§2.35c/§0p VQI + Singer-ID: Skip im Chunked-Pfad, Song-Level-Lauf in _restore_chunked."""
    src = _uv3_source()
    body = _restore_chunked_body(src)
    # Per-Chunk-Bedingung ist um not _chunked_tail_skip erweitert.
    assert "and (_vqi_singing_conf >= self._vocal_threshold() or _vqi_prior_active)" in src
    # Song-Level: eine VQI-Messung (+ optionale Rollback-Verifikationsmessung).
    assert body.count("_p01_compute_vqi(") >= 1
    assert "§P0-1 Slice B: §2.35c VQI song-weit=" in body


# ---------------------------------------------------------------------------
# 2) Chunk-Laufzeit sinkt strukturell (measure_all nicht je Chunk)
# ---------------------------------------------------------------------------


def test_pre_excellence_measure_all_is_deferred_on_non_last_chunks() -> None:
    """§8.1 Snapshot/Re-Verifikation: measure_all läuft nicht mehr auf Nicht-Letzten-Chunks."""
    src = _uv3_source()
    assert "§P0-1 Slice B: Pre-Excellence-Messung übersprungen (Nicht-Letzter-Chunk)" in src
    # Der Checker (und damit measure_all) liegt im else-Zweig des Skip-Guards.
    assert "_pre_ex_checker = _PreExMGChecker()" in src
    assert "_pre_excellence_mg_scores = _pre_ex_checker.measure_all(" in src
    # Der Skip-Guard steht vor der Messung.
    guard_at = src.index("§P0-1 Slice B: Pre-Excellence-Messung übersprungen")
    measure_at = src.index("_pre_excellence_mg_scores = _pre_ex_checker.measure_all(")
    assert guard_at < measure_at


def test_goal_scorecard_measure_is_not_repeated_per_chunk() -> None:
    """Block a1 bleibt intakt: measure_all bleibt song-global (kein Rückfall auf Per-Chunk)."""
    src = _uv3_source()
    body = _restore_chunked_body(src)
    # Song-Level-Messung nach Assembly vorhanden …
    assert "🎯 §P0-1 Song-SCORECARD nach Assembly" in body
    # … und der Per-Chunk-Skip aus Block a1 ist unverändert.
    assert "§P0-1: measure_all für Chunk übersprungen" in src


# ---------------------------------------------------------------------------
# 3) Ebene-1-Schutz (Hörordnung §2) + Wohlklang-Enforcer nicht unterlaufen
# ---------------------------------------------------------------------------


def test_ebene1_rollback_uses_verified_snapshot_not_blind_revert() -> None:
    """Song-Level-Ebene-1-Rollback gewinnt nur mit Verifikation (§v10.x Analogie)."""
    src = _uv3_source()
    block = _song_level_block(src)
    assert "_p01_rb_vqi > _song_vqi_score or (_p01_sid < 0.92 and _p01_rb_sid >= 0.92)" in block
    # Der Snapshot ist der Zustand VOR den Song-Gates (Post-Processing-Checkpoint-Analogon).
    assert "_p01_snapshot = output.copy()" in block


def test_vqi_gate_reuses_central_thresholds_without_new_phase_thresholds() -> None:
    """§V7 (copilot-instructions.md): zentrale Schwellwerte (Singer-ID 0.92, get_vqi_material_floor) — keine neuen."""
    src = _uv3_source()
    block = _song_level_block(src)
    # Song-Level nutzt dieselben zentralen Schwellen wie der Per-Chunk-Pfad.
    assert "_p01_sid < 0.92" in block
    assert "_p01_vqi_floor(_p01_mat_str, is_studio_2026=_p01_is_studio)" in block
    # Per-Chunk-Schwelle unverändert (kein Schwellwert-Flip).
    assert "§0p Singer-ID-Rollback (RELEASE_MUST §0p)" in src
    # Der Song-Level-Block führt keine harte VQI-Materialschwelle ein.
    assert "get_vqi_material_floor" in block


def test_wohlklang_enforcer_runs_on_the_final_song_and_stays_intact() -> None:
    """Der Enforcer (wohlklang_ordnung_gate.py) läuft song-weit und bleibt im Tail intakt."""
    src = _uv3_source()
    body = _restore_chunked_body(src)
    block = _song_level_block(src)
    # Song-Level-Audit vorhanden …
    assert "WohlklangOrdnungGate as _P01WoGate" in block
    assert "Song-Wohlklang-Ordnungs-Violation" in block
    # … und der §P1-1 Pipeline-Audit im restore()-Tail ist unverändert vorhanden.
    assert "§P1-1 Wohlklang-Ordnungs-Audit + ENFORCEMENT auf Pipeline-Ebene" in src
    assert "WohlklangOrdnungGate as _WOGate_pipeline" in src
    # Der FC-GPP-Callback (Kandidaten-Verwerfung) bleibt unangetastet.
    assert "§FC-GPP: Wohlklang-Ordnungs-Verstoß" in src


def test_chunked_path_never_resets_audio_on_deferred_hpi() -> None:
    """fail-closed auf Originalsignal greift nur, wenn der Gate ausgewertet wurde."""
    src = _uv3_source()
    assert "if _hpi_result is None and not _chunked_tail_skip:" in src


# ---------------------------------------------------------------------------
# 4) Determinismus (§G5 (copilot-instructions.md)) + keine Flag-Flips (§v10.802-Umfeld) im Song-Level-Block
# ---------------------------------------------------------------------------


def test_song_level_block_is_deterministic_in_its_decisions() -> None:
    """Kein Zeit-/Zufallsbezug in den Song-Level-Entscheidungen (§G5 (copilot-instructions.md))."""
    block = _song_level_block(_uv3_source())
    for _forbidden in ("time.time(", "time.monotonic(", "random.", "np.random", "datetime"):
        assert _forbidden not in block, f"nicht-deterministischer Bezug im Song-Level-Block: {_forbidden}"


def test_song_level_block_introduces_no_env_flag_flips() -> None:
    """Keine neuen Feature-Flags/Env-Schalter im Song-Level-Block."""
    block = _song_level_block(_uv3_source())
    assert "os.environ" not in block
    assert "getenv" not in block


def test_song_level_metadata_reports_gate_results() -> None:
    """Gate-Ergebnisse landen in den Song-Metadaten (Forensik/§V6 (copilot-instructions.md) statt stiller Änderung)."""
    src = _uv3_source()
    body = _restore_chunked_body(src)
    assert '_meta["p0_1_song_perceptual_gates"] = dict(_p01_gates_meta)' in body
    assert "emotional_arc=(" in body


# ---------------------------------------------------------------------------
# 5) Statische Signaturprüfung gegen die echten Module
# ---------------------------------------------------------------------------


def test_real_module_signatures_match_the_song_level_calls() -> None:
    """Die Song-Level-Aufrufe passen exakt zu den echten Modul-Signaturen."""
    from backend.core.emotional_arc_preservation import correct_emotional_arc, measure_emotional_arc
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate
    from backend.core.musical_goals.vocal_quality_index import compute_vqi, get_vqi_material_floor

    vqi_params = set(inspect.signature(compute_vqi).parameters)
    assert {"audio_orig", "audio_restored", "sr", "vocal_segments", "skip_singer_identity", "genre", "era_profile"} <= (
        vqi_params
    )

    floor_params = inspect.signature(get_vqi_material_floor).parameters
    assert "material_type" in floor_params and "is_studio_2026" in floor_params

    arc_params = set(inspect.signature(measure_emotional_arc).parameters)
    assert {"original", "restored", "sr", "frisson_zones"} <= arc_params

    corr_params = set(inspect.signature(correct_emotional_arc).parameters)
    assert {"original", "restored", "sr", "max_gain_db", "damping", "frisson_zones"} <= corr_params

    hpi_params = set(inspect.signature(HolisticPerceptualGate.evaluate_restoration).parameters)
    assert {
        "original",
        "restored",
        "sr",
        "artifact_freedom",
        "emotional_arc_score",
        "restorability_score",
        "genre",
        "material",
        "era_bin",
        "vqi",
        "panns_singing",
        "reference_audio",
        "transfer_chain",
    } <= hpi_params

    studio_params = set(inspect.signature(HolisticPerceptualGate.evaluate_studio).parameters)
    assert {
        "original",
        "restored",
        "sr",
        "pqs_improvement",
        "artifact_freedom",
        "emotional_arc_score",
        "vqi",
        "panns_singing",
    } <= (studio_params)


def test_vqi_material_floor_is_a_pure_deterministic_function() -> None:
    """get_vqi_material_floor: deterministisch, endlich, > 0 (zentrale Schwelle, §V7 (copilot-instructions.md))."""
    from backend.core.musical_goals.vocal_quality_index import get_vqi_material_floor

    for _mat in ("digital", "tape", "vinyl", "cassette"):
        _a = float(get_vqi_material_floor(_mat, is_studio_2026=False))
        _b = float(get_vqi_material_floor(_mat, is_studio_2026=False))
        assert _a == _b, f"Floor nicht deterministisch fuer {_mat}"
        assert np.isfinite(_a) and _a > 0.0, f"Floor unplausibel fuer {_mat}: {_a}"


# ---------------------------------------------------------------------------
# 6) Synthetische Signale: EAP-Determinismus + NaN/Inf-Schutz (§0a)
# ---------------------------------------------------------------------------


def _synthetic_pair(n_samples: int = 1_560_000, sr: int = 48000) -> tuple[np.ndarray, np.ndarray]:
    """Zwei deterministische Mono-Signale (32,5 s @ 48 kHz) mit leicht unterschiedlichem Bogen."""
    t = np.arange(n_samples, dtype=np.float32) / float(sr)
    rng = np.random.default_rng(20261105)
    base = 0.10 * np.sin(2.0 * np.pi * 220.0 * t)
    macro_orig = 0.5 + 0.5 * np.sin(2.0 * np.pi * t / 16.0)
    macro_rest = 0.5 + 0.35 * np.sin(2.0 * np.pi * t / 16.0 + 0.2)
    orig = (base * macro_orig).astype(np.float32)
    rest = (base * macro_rest + 0.002 * rng.standard_normal(n_samples).astype(np.float32)).astype(np.float32)
    return orig, rest


def test_measure_emotional_arc_is_deterministic_on_synthetic_signals() -> None:
    """§G5 (copilot-instructions.md): gleicher Input + gleiche Version ⇒ bit-identischer Mess-Output."""
    from backend.core.emotional_arc_preservation import measure_emotional_arc

    orig, rest = _synthetic_pair()
    r1 = measure_emotional_arc(orig, rest, 48000)
    r2 = measure_emotional_arc(orig, rest, 48000)
    for _field in ("arousal_pearson", "valence_pearson", "preservation_score", "skipped"):
        assert getattr(r1, _field, None) == getattr(r2, _field, None), f"Messung nicht deterministisch: {_field}"


def test_correct_emotional_arc_keeps_shape_and_finite_audio() -> None:
    """Korrektur erhält Shape und NaN/Inf-Freiheit (§0a) auf synthetischem Signal."""
    from backend.core.emotional_arc_preservation import correct_emotional_arc

    orig, rest = _synthetic_pair()
    corrected, result = correct_emotional_arc(orig, rest, 48000, max_gain_db=6.0, damping=0.7)
    assert np.asarray(corrected).shape == rest.shape
    assert np.all(np.isfinite(corrected)), "Korrektur erzeugt NaN/Inf (§0a-Verletzung)"
    assert result is not None
