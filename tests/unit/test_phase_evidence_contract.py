"""Vertragstests für §7.4c (06_phases_system.md) — Kanonischer Phasen-Evidenz-Vertrag.

Sichert die zentrale Normalisierung der Evidenz-Kwarg (defect_scores,
defect_locations, restoration_context-Alias) gegen Regression ab:

- String- UND Enum-Keys lösen sich auf (Wurzel-Fix der L3-Bruch-Klasse
  „String/Enum-Keys", Produktionsbefund 2026-09-25: phase_24 lief nie).
- .items()/Iteration bleibt Enum-only (keine Veränderung für Enum-Konsumenten).
- defect_locations: Sekunden-Fenster, FensterMITTE = Ereignisposition.
- restoration_context (ohne Unterstrich) ist der einzige Kontext-Key;
  _restoration_context wird als deprecated Alias zusammengeführt (Wurzel-Fix
  der L3-Bruch-Klasse „Namensvarianten" — tote Lese-Pfade).
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.defect_scanner import (
    DefectLocationsView,
    DefectScore,
    DefectScoreView,
    DefectType,
)
from backend.core.phases.phase_interface import normalize_evidence_kwargs


def _scores_dict() -> dict:
    return {
        DefectType.WOW: DefectScore(DefectType.WOW, 0.272, 0.9),
        DefectType.CRACKLE: DefectScore(DefectType.CRACKLE, 0.4, 0.8),
    }


class TestDefectScoreViewContract:
    def test_string_key_resolves_to_severity_float(self) -> None:
        view = normalize_evidence_kwargs({"defect_scores": _scores_dict()})["defect_scores"]
        assert isinstance(view, DefectScoreView)
        assert view.get("wow") == 0.272
        assert view["wow"] == 0.272

    def test_enum_key_resolves_to_defect_score(self) -> None:
        view = normalize_evidence_kwargs({"defect_scores": _scores_dict()})["defect_scores"]
        score = view.get(DefectType.WOW)
        assert isinstance(score, DefectScore)
        assert score.severity == 0.272

    def test_unknown_key_returns_default(self) -> None:
        view = normalize_evidence_kwargs({"defect_scores": _scores_dict()})["defect_scores"]
        assert view.get("nicht_vorhanden", 0.0) == 0.0

    def test_contains_and_iteration_semantics(self) -> None:
        view = normalize_evidence_kwargs({"defect_scores": _scores_dict()})["defect_scores"]
        assert "wow" in view
        assert DefectType.WOW in view
        # Iteration bleibt Enum-only (§7.4c)
        assert all(isinstance(k, DefectType) for k, _ in view.items())


class TestDefectLocationsViewContract:
    def test_locations_wrap_and_key_resolution(self) -> None:
        locs = {DefectType.TAPE_SPLICE_ARTIFACT: [(6.98, 7.02)]}
        out = normalize_evidence_kwargs({"defect_locations": locs})["defect_locations"]
        assert isinstance(out, DefectLocationsView)
        assert out.get("tape_splice_artifact") == [(6.98, 7.02)]
        assert out.get(DefectType.TAPE_SPLICE_ARTIFACT) == [(6.98, 7.02)]
        assert "tape_splice_artifact" in out

    def test_window_center_is_event_position(self) -> None:
        # §7.4c: Fenster (t-0.02, t+0.02) → Mitte = Ereignis; Saaten nutzen
        # die Mitte (L3-Befund 2026-09-25: der Anfang liegt 20 ms daneben).
        out = normalize_evidence_kwargs({"defect_locations": {"wow": [(6.98, 7.02)]}})["defect_locations"]
        start, end = out.get("wow")[0]
        assert (start + end) * 0.5 == pytest.approx(7.0)


class TestRestorationContextAlias:
    def test_legacy_alias_is_merged_into_canonical_key(self) -> None:
        out = normalize_evidence_kwargs({"_restoration_context": {"tempo_bpm": 120.0}})
        assert out["restoration_context"] == {"tempo_bpm": 120.0}
        assert out["_restoration_context"] is out["restoration_context"]

    def test_canonical_key_wins_on_conflict(self) -> None:
        out = normalize_evidence_kwargs(
            {
                "restoration_context": {"a": 1},
                "_restoration_context": {"a": 0, "b": 2},
            }
        )
        assert out["restoration_context"] == {"a": 1, "b": 2}
        assert out["_restoration_context"] == {"a": 1, "b": 2}

    def test_no_context_keys_untouched(self) -> None:
        kwargs: dict = {}
        out = normalize_evidence_kwargs(kwargs)
        assert "restoration_context" not in out
        assert "_restoration_context" not in out


class TestIdempotency:
    def test_double_normalization_is_stable(self) -> None:
        kwargs = {
            "defect_scores": _scores_dict(),
            "defect_locations": {"wow": [(1.0, 1.1)]},
            "_restoration_context": {"x": 1},
        }
        first = normalize_evidence_kwargs(kwargs)
        scores_1 = first["defect_scores"]
        ctx_1 = first["restoration_context"]
        second = normalize_evidence_kwargs(kwargs)
        assert second["defect_scores"] is scores_1
        assert second["restoration_context"] is ctx_1
        assert second["defect_scores"].get("wow") == 0.272


class TestSafeProcessNormalizesEvidence:
    def test_safe_process_passes_normalized_kwargs(self) -> None:
        from backend.core.phases.phase_interface import PhaseInterface, PhaseResult

        seen: dict = {}

        class _Probe(PhaseInterface):
            def get_metadata(self):  # type: ignore[override]
                from backend.core.phases.phase_interface import PhaseCategory, PhaseMetadata

                return PhaseMetadata(
                    phase_id="phase_probe_contract",
                    name="Probe",
                    category=PhaseCategory.DEFECT_REMOVAL,
                    priority=1,
                )

            def process(self, audio, sample_rate=48000, material_type="unknown", **kwargs):  # type: ignore[override]
                seen.update(kwargs)
                return PhaseResult(audio=audio)

        probe = _Probe()
        audio = np.zeros(64, dtype=np.float32)
        probe._safe_process(
            audio,
            48000,
            "tape",
            defect_scores=_scores_dict(),
            _restoration_context={"k": 1},
        )
        assert isinstance(seen.get("defect_scores"), DefectScoreView)
        assert seen["defect_scores"].get("wow") == 0.272
        assert seen.get("restoration_context") == {"k": 1}
        assert seen.get("_restoration_context") == {"k": 1}
