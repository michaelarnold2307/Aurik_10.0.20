"""song_context_map — konsolidierte, unsichere Musikverständnis-Karte (Spec 03 §SCM).

Aurik besitzt bereits viele Analysatoren (Ära/Genre, VocalFocusAnalyzer,
HarmonicContextAnalyzer, SongStructureAnalyzer, PANNs-Tags, EmotionalArcPlanner).
Ihre Ergebnisse liegen verstreut in ``_restoration_context`` und im
Phasen-Akkumulator. Diese Karte konsolidiert sie zu EINER Sicht pro Song —
**ohne neu zu messen**:

* Jede Dimension trägt ``state`` (``measured`` | ``unavailable``), ``value``,
  ``confidence`` und ``source``.
* ``confidence`` stammt ausschließlich aus einem vom Analysator gelieferten
  Konfidenzfeld; liefert er keines, ist sie ``None`` — es wird nichts geschätzt
  (§V7 (copilot-instructions.md): keine Ad-hoc-Parameter).
* Fehlende Analysen heißen ``unavailable`` — nie 0, nie „ok“
  (§V6 (copilot-instructions.md), §G10 (copilot-instructions.md): unbewiesen ≠ bestanden).

Rolle (Hörordnung §7 (hoerordnung.instructions.md)): REPORT-ONLY-Zeuge. Die Karte
verändert kein Sample und keine Entscheidung. Sie ist die Grundlage für spätere,
hörkalibrierte Entscheidungen (Spec 03 §SCM) — bis dahin liest keine
Entscheidungslogik sie.

Determinismus (§G5 (copilot-instructions.md)): reine Funktion der Eingaben, keine
Zufalls- oder Zeitabhängigkeit; Eingaben werden nicht verändert.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = "aurik.song_context_map/1"

#: Reihenfolge und Menge der Dimensionen (Spec 03 §SCM).
DIMENSIONS: tuple[str, ...] = (
    "provenance",
    "voice",
    "instruments",
    "harmony",
    "rhythm",
    "structure",
    "dynamics_arc",
)

#: Gültige Geltungsbereiche der Karte. ``chunk``: aus chunk-lokalen Analysen
#: (Chunked-Pfad), ``song``: aus Ganz-Song-Analysen.
SCOPES: tuple[str, ...] = ("song", "chunk")

#: Berichtsgröße der Tag-Liste (kein Entscheidungsschwellwert).
TOP_TAGS = 5

_UNKNOWN_GENRE = "unbekannt"


@dataclass(frozen=True)
class ContextDimension:
    """Eine Dimension der Karte (gemessen oder nicht verfügbar)."""

    name: str
    state: str
    value: dict[str, Any]
    confidence: float | None
    source: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert JSON-sicher (NaN/Inf-frei)."""
        return {
            "state": self.state,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "note": self.note,
        }


@dataclass(frozen=True)
class SongContextMap:
    """Konsolidierte Karte über alle Dimensionen."""

    scope: str
    analysis_window_s: float | None
    dimensions: tuple[ContextDimension, ...]

    @property
    def unavailable(self) -> tuple[str, ...]:
        """Namen der nicht gemessenen Dimensionen."""
        return tuple(d.name for d in self.dimensions if d.state != "measured")

    @property
    def coverage(self) -> float:
        """Anteil gemessener Dimensionen (0..1) — Abdeckung, kein Qualitätsmaß."""
        measured = sum(1 for d in self.dimensions if d.state == "measured")
        return round(measured / len(self.dimensions), 3) if self.dimensions else 0.0

    @property
    def weakest_confidence(self) -> dict[str, Any] | None:
        """Schwächste Dimension unter denen mit analysatoreigener Konfidenz."""
        scored = [(d.confidence, d.name) for d in self.dimensions if d.state == "measured" and d.confidence is not None]
        if not scored:
            return None
        conf, name = min(scored)
        return {"dimension": name, "confidence": conf}

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert die Karte JSON-sicher."""
        return {
            "schema": SCHEMA,
            "report_only": True,
            "scope": self.scope,
            "analysis_window_s": self.analysis_window_s,
            "coverage": self.coverage,
            "unavailable": list(self.unavailable),
            "weakest_confidence": self.weakest_confidence,
            "dimensions": {d.name: d.to_dict() for d in self.dimensions},
        }


def _finite(value: Any) -> float | None:
    """Endlicher Float oder ``None`` (bool/NaN/Inf/Nicht-Zahl sind keine Messwerte)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _unavailable(name: str, source: str, note: str) -> ContextDimension:
    return ContextDimension(name=name, state="unavailable", value={}, confidence=None, source=source, note=note)


def _measured(
    name: str, source: str, value: dict[str, Any], confidence: float | None, note: str = ""
) -> ContextDimension:
    return ContextDimension(name=name, state="measured", value=value, confidence=confidence, source=source, note=note)


def _count(container: Any) -> int | None:
    return len(container) if isinstance(container, (list, tuple)) else None


def _provenance(ctx: Mapping[str, Any]) -> ContextDimension:
    source = "era_classifier+genre_classifier+medium_detector"
    decade = ctx.get("decade")
    decade_value = int(decade) if isinstance(decade, (int, float)) and not isinstance(decade, bool) else None
    genre = str(ctx.get("genre_label") or "").strip()
    genre_known = bool(genre) and genre.lower() != _UNKNOWN_GENRE
    material = str(ctx.get("primary_material") or "").strip()
    if decade_value is None and not genre_known and not material:
        return _unavailable("provenance", source, "Weder Ära, Genre noch Material ermittelt.")
    value: dict[str, Any] = {
        "decade": decade_value,
        "primary_material": material or None,
        "genre": genre if genre_known else None,
        "subgenre": (str(ctx.get("subgenre") or "").strip() or None) if genre_known else None,
    }
    # Ära-Konfidenz nur, wenn die Ära tatsächlich ermittelt wurde (Default 0.0 ≠ Messung).
    confidence = _finite(ctx.get("era_confidence")) if decade_value is not None else None
    return _measured("provenance", source, value, confidence)


def _voice(ctx: Mapping[str, Any]) -> ContextDimension:
    source = "vocal_focus_analyzer"
    vfa = ctx.get("vfa_result")
    if not isinstance(vfa, Mapping) or "vocal_present" not in vfa:
        return _unavailable("voice", source, "VocalFocusAnalyzer lieferte kein Ergebnis.")
    value = {
        "vocal_present": bool(vfa.get("vocal_present")),
        "panns_singing": _finite(vfa.get("panns_singing")),
        "dominant_register": vfa.get("dominant_register"),
        "singer_school": vfa.get("singer_school"),
        "n_vibrato_zones": _count(vfa.get("vibrato_zones")),
        "n_passaggio_zones": _count(vfa.get("passaggio_zones")),
        "n_whisper_zones": _count(vfa.get("whisper_zones")),
        # style_confidence ist laut VFAResult der Anteil intentionaler Voicing-Events —
        # keine Konfidenz der Stimmerkennung, daher als eigenes Maß geführt.
        "style_intent_share": _finite(vfa.get("style_confidence")),
    }
    return _measured(
        "voice",
        source,
        value,
        None,
        "Keine analysatoreigene Konfidenz der Stimmerkennung; panns_singing ist die Gesangs-Wahrscheinlichkeit.",
    )


def _instruments(ctx: Mapping[str, Any]) -> ContextDimension:
    source = "panns_audioset_tags"
    tags = ctx.get("panns_tags")
    if not isinstance(tags, Mapping):
        return _unavailable("instruments", source, "Keine PANNs-Tags vorhanden.")
    scored: list[tuple[float, str]] = []
    for label, score in tags.items():
        number = _finite(score)
        if number is not None and isinstance(label, str):
            scored.append((number, label))
    if not scored:
        return _unavailable("instruments", source, "PANNs-Tags ohne gültige Werte.")
    # Stabil: Score absteigend, bei Gleichstand Label aufsteigend (§G5 (copilot-instructions.md)).
    top = sorted(scored, key=lambda item: (-item[0], item[1]))[:TOP_TAGS]
    value = {"tags": [{"label": label, "score": round(score, 4)} for score, label in top]}
    return _measured(
        "instruments",
        source,
        value,
        None,
        "AudioSet-Tag-Scores; keine Instrumentenerkennung. Jeder Tag trägt seinen eigenen Score.",
    )


def _harmony(ctx: Mapping[str, Any]) -> ContextDimension:
    source = "harmonic_context_analyzer"
    hca = ctx.get("harmonic_context")
    if not isinstance(hca, Mapping) or not hca.get("key"):
        return _unavailable("harmony", source, "HarmonicContextAnalyzer lieferte kein Ergebnis.")
    value = {
        "key": str(hca.get("key")),
        "n_chord_frames": hca.get("n_frames") if isinstance(hca.get("n_frames"), int) else None,
        "n_modulations": _count(hca.get("modulation_frames")),
    }
    return _measured("harmony", source, value, _finite(hca.get("analysis_confidence")))


def _rhythm(ctx: Mapping[str, Any], structure_bpm: float | None) -> ContextDimension:
    source = "musical_structure_analyzer+genre_classifier"
    bpm_structure = _finite(structure_bpm)
    bpm_genre = _finite(ctx.get("bpm"))
    bpm_structure = bpm_structure if bpm_structure is not None and bpm_structure > 0.0 else None
    bpm_genre = bpm_genre if bpm_genre is not None and bpm_genre > 0.0 else None
    if bpm_structure is None and bpm_genre is None:
        return _unavailable("rhythm", source, "Kein Tempo ermittelt.")
    value: dict[str, Any] = {
        "bpm_structure": round(bpm_structure, 1) if bpm_structure is not None else None,
        "bpm_genre_classifier": round(bpm_genre, 1) if bpm_genre is not None else None,
    }
    note = "Tempo ohne analysatoreigene Konfidenz."
    if bpm_structure is not None and bpm_genre is not None:
        # Faktum ohne Urteil: Verhältnis der beiden Schätzungen (Halb-/Doppeltempo bleibt sichtbar).
        value["bpm_ratio"] = round(max(bpm_structure, bpm_genre) / min(bpm_structure, bpm_genre), 3)
    return _measured("rhythm", source, value, None, note)


def _structure(segments: Sequence[Any] | None) -> ContextDimension:
    source = "song_structure_analyzer"
    if not segments:
        return _unavailable("structure", source, "Keine Segmentstruktur ermittelt.")
    items: list[dict[str, Any]] = []
    covered = 0.0
    vocal_covered = 0.0
    for seg in segments:
        start = _finite(getattr(seg, "start_s", None))
        end = _finite(getattr(seg, "end_s", None))
        if start is None or end is None or end < start:
            continue
        duration = end - start
        has_vocals = bool(getattr(seg, "has_vocals", False))
        covered += duration
        vocal_covered += duration if has_vocals else 0.0
        items.append(
            {
                "label": str(getattr(seg, "label", "")),
                "start_s": round(start, 2),
                "end_s": round(end, 2),
                "has_vocals": has_vocals,
                "is_climax": bool(getattr(seg, "is_climax", False)),
            }
        )
    if not items:
        return _unavailable("structure", source, "Segmentstruktur ohne gültige Zeitgrenzen.")
    value = {
        "n_segments": len(items),
        "n_climax": sum(1 for item in items if item["is_climax"]),
        "covered_s": round(covered, 2),
        "vocal_fraction": round(vocal_covered / covered, 3) if covered > 0.0 else None,
        "segments": items,
    }
    return _measured("structure", source, value, None, "Segmentierung ohne analysatoreigene Konfidenz.")


def _dynamics_arc(ctx: Mapping[str, Any]) -> ContextDimension:
    source = "vocal_focus_analyzer+emotional_arc_planner"
    vfa = ctx.get("vfa_result")
    has_vfa = isinstance(vfa, Mapping) and "vocal_present" in vfa
    arc_available = ctx.get("arc_protection_weights") is not None
    if not has_vfa and not arc_available:
        return _unavailable("dynamics_arc", source, "Weder Emotionszonen noch Bogenplan vorhanden.")
    zones = vfa if isinstance(vfa, Mapping) else {}
    value = {
        "n_tension_zones": _count(zones.get("tension_zones")),
        "n_release_zones": _count(zones.get("release_zones")),
        "n_frisson_zones": _count(zones.get("frisson_zones")),
        "climax_type": zones.get("climax_type"),
        "arc_plan_available": arc_available,
    }
    return _measured(
        "dynamics_arc",
        source,
        value,
        None,
        "Zonen sind gesangsbezogen abgeleitet; Instrumentalpassagen sind nicht abgedeckt.",
    )


def build_song_context_map(
    restoration_context: Mapping[str, Any] | None,
    *,
    structure_segments: Sequence[Any] | None = None,
    structure_bpm: float | None = None,
    scope: str = "song",
    analysis_window_s: float | None = None,
) -> SongContextMap:
    """Konsolidiert vorhandene Analyseergebnisse zur Kontextkarte.

    Raises:
        ValueError: ``scope`` ist weder ``song`` noch ``chunk`` (fail-closed).
    """
    if scope not in SCOPES:
        raise ValueError(f"Unbekannter Scope {scope!r}; erlaubt: {SCOPES}")
    ctx: Mapping[str, Any] = restoration_context if isinstance(restoration_context, Mapping) else {}
    builders: tuple[tuple[str, Any], ...] = (
        ("provenance", lambda: _provenance(ctx)),
        ("voice", lambda: _voice(ctx)),
        ("instruments", lambda: _instruments(ctx)),
        ("harmony", lambda: _harmony(ctx)),
        ("rhythm", lambda: _rhythm(ctx, structure_bpm)),
        ("structure", lambda: _structure(structure_segments)),
        ("dynamics_arc", lambda: _dynamics_arc(ctx)),
    )
    dimensions: list[ContextDimension] = []
    for name, build in builders:
        try:
            dimensions.append(build())
        except Exception as exc:  # §V6 (copilot-instructions.md): nie still „alles gut“
            logger.warning(
                "SongContextMap: Dimension '%s' nicht ableitbar (%s) — als nicht verfügbar geführt", name, exc
            )
            dimensions.append(_unavailable(name, "song_context_map", f"Ableitung fehlgeschlagen: {type(exc).__name__}"))
    window = _finite(analysis_window_s)
    return SongContextMap(
        scope=scope,
        analysis_window_s=round(window, 2) if window is not None else None,
        dimensions=tuple(dimensions),
    )


__all__ = [
    "DIMENSIONS",
    "SCHEMA",
    "SCOPES",
    "TOP_TAGS",
    "ContextDimension",
    "SongContextMap",
    "build_song_context_map",
]
