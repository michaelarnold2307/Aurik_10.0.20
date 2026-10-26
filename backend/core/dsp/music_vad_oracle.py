"""Musikadaptierter VAD-Gate-Offset-Orakel (KP-4, report-only — KEIN Training).

Ersetzt rollenmäßig das sprachtrainierte Silero-VAD (English Speech) als
Quelle für den Gate-Schwellen-Offset des Noise-Gate (phase_18) durch einen
**musiknativen** Zeugen: die PANNs-Singing-Wahrscheinlichkeit
(``panns_singing``) bzw. die AudioSet-,„Singing voice"-Tags — nach dem Muster
von phase_19 (Gender-Prior) und KP-2/KP-3 (report-only Zeuge, KEIN Veto).

Hintergrund (§1 „Lage", docs/SOTA_TRAINING_KONZEPT.md §3/KP-4): Silero ist
auf SPRACHE trainiert und steuert heute in phase_18 einen Gate-Schwellen-
Offset (bis −15 dB). Nach §III.11 (copilot-instructions.md) darf ein
sprachtrainiertes Modell Musik/Gesang nicht steuern — deshalb ist
`music_model_flags.use_silero_vad_music == False` (W-11, gesperrt) und das
Gate läuft heute OHNE Offset (neutral, §V7 (copilot-instructions.md)).

Rolle dieses Moduls: **report-only Zeuge**. Es berechnet aus der
PANNs-Singing-Wahrscheinlichkeit einen Offset (dB, auf dasselbe 0…−15-dB-Band
skaliert wie der Silero-Pfad), protokolliert ihn und liefert ihn als
Zusatzfeld zurück — **ohne** das produktive Gate-Verhalten zu ändern. Der
aktive Offset bleibt der bestehende Pfad (bei Flag=False neutral 0). Es übt
NIE ein Veto aus, flippt keine Flags und mutiert kein Kalibrierungs-Profil
(Hörordnung §8: Metriken sind Zeugen, die Hör-Instanz entscheidet). Die
Aktivierung des musiknativen Offsets hängt weiterhin an einem musik-
kalibrierten VAD (KP-4: Never-worsen-Beweis + Hörordnungs-Sign-off) und steht
noch aus (W-11/W‑12-Register).

Determinismus (§G5 (copilot-instructions.md)): keine Zeit-/Umgebungsquellen
in der Offset-Berechnung; die Eingabe ist ein einzelner
PANNs-Singing-Konfidenzwert in [0, 1], die Skalierung ist eine reine
affine Abbildung. Silent-Failure (§V6 (copilot-instructions.md)): fehlt die
PANNs-Evidenz (None, nicht-numerisch, Modell nicht geladen), wird ein
neutraler Offset (0 dB) zurückgegeben — NUR mit ``logger.warning`` +
Begründung; ein fehlender Messwert darf den Gate nie öffnen/schließen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Offset-Band — identisch zum Silero-Pfad in phase_18 („bis −15 dB").
# Der Offset senkt die Gate-Schwelle bei vokal-/musikreichem Material, damit
# der Gate bei Gesang offen bleibt (kein Abschneiden von leisem Gesang).
_MAX_OFFSET_DB = 15.0

# Schwellen der PANNs-Singing-Evidenz. Unter der Mindest-Konfidenz gilt die
# Evidenz als „nicht vorhanden" (konservativ: kein Offset statt Raten) —
# analog zum Gender-Prior _PANNS_GENDER_MIN_SCORE=0.25 (§G9 (copilot-instructions.md)).
_MIN_SINGING_CONFIDENCE = 0.25


@dataclass
class MusicVADOffsetWitness:
    """Deterministisches Ergebnis des report-only Musik-VAD-Gate-Offsets.

    Ein Ergebnis ist immer ein ZEUGNIS — ``applied`` ist konstant False.
    Die Felder transportieren nur den berechneten Offset für die
    Protokollierung (report-only, Hörordnung §8).
    """

    stage: str = ""
    oracle: str = "music_vad_panns"
    panns_singing: float | None = None
    offset_db: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def applied(self) -> bool:
        """Report-only: der Musik-VAD-Zeuge ändert das Gate-Verhalten nie."""
        return False


def _downmix(audio: np.ndarray) -> np.ndarray:
    """Stereo-Layout-Invariante (§11 Spec 02): (C, N) und (N, C) → Mono float32.

    ``audio[:, 0]``/``mean(axis=0)`` ohne Layout-Check kollabiert auf
    C Samples (Produktionsbefund: Export ``(2,)``) — deshalb beide Layouts
    explizit bedienen und auf float32 normalisieren.
    """
    arr: np.ndarray = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    res: np.ndarray
    if arr.ndim <= 1:
        res = arr.astype(np.float32)
    elif arr.shape[0] <= 2 and arr.shape[0] < arr.shape[1]:
        # kanonisch channels-first (C, N); (N, C) ebenfalls bedienen (wie muq_plugin).
        res = arr.mean(axis=0).astype(np.float32)
    else:
        res = arr.mean(axis=1).astype(np.float32)
    return res


def _coerce_singing(value: Any) -> float | None:
    """Coerced die PANNs-Singing-Evidenz zu ''float in [0, 1]'' oder ''None''.

    ``None`` bedeutet „kein glaubwürdiger Messwert" (nicht-numerisch, NaN,
    außerhalb [0, 1]) — der Aufrufer behandelt das als neutralen Offset 0 dB
    mit §V6-Warnung. Bewusst KEIN stummes ``or 0.0``: ein ungültiger Wert
    darf nie als „sicher kein Gesang" umgedeutet werden (das würde den Gate
    auf Instrumental-Material härter schließen — ein stiller Schaden).
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return float(np.clip(f, 0.0, 1.0))


def _singing_from_tags(panns_tags: Any) -> float | None:
    """Liest die „Singing voice"-Konfidenz aus einem PANNs-Tag-Dict.

    Akzeptiert dieselben Schlüssel wie ``plugins/panns_plugin.py`
    (_TAG_INDEX_MAP: „Singing voice" und „Singing"). Gibt ``None`` wenn kein
    Tag-Dict/nicht-numerisch/nicht-vorhanden (§V6 (copilot-instructions.md)).
    """
    if not isinstance(panns_tags, dict):
        return None
    raw = None
    for key in ("Singing voice", "Singing", "Vocals"):
        if key in panns_tags:
            raw = panns_tags.get(key)
            break
    if raw is None:
        return None
    return _coerce_singing(raw)


def compute_music_vad_offset_db(panns_singing: Any, panns_tags: Any = None) -> MusicVADOffsetWitness:
    """Berechnet den report-only Musik-Gate-Offset (dB) aus PANNs-Singing.

    Pure Funktion (mutiert nichts). Der Offset wird auf das bestehende
    Silero-Band skaliert: ``offset_db = -MAX_OFFSET_DB * singing``, also
    0 dB bei singing→0 und −15 dB bei singing→1. Ein Offset **senkt** die
    Gate-Schwelle (negativ = tieferer Schwellwert = Gate hält länger offen),
    identisch zur Semantik des Silero-Pfads in ``_apply_gate``
    (``threshold_db_adapted = threshold_db - 15 * vad_probabilities``).

    Rückgabe ist NIE None: ein Ergebnis-Objekt wird immer geliefert, damit
    der Aufrufer den Zeugen-Status immer protokollieren kann.

    Args:
        panns_singing: PANNs-Singing-Konfidenz (float oder coercible) ODER
            ``None`` (dann fällt der Zeuge auf die ``panns_tags``-Quelle zurück).
        panns_tags: optionales PANNs-Tag-Dict („Singing voice"/„Singing") als
            alternative/ergänzende Evidenzquelle.

    Returns:
        ``MusicVADOffsetWitness`` mit ``applied == False`` (report-only).
    """
    w = MusicVADOffsetWitness(stage="phase_18_noise_gate")

    singing = _coerce_singing(panns_singing)
    if singing is None and panns_tags is not None:
        singing = _singing_from_tags(panns_tags)

    if singing is None:
        # §V6 (copilot-instructions.md): PANNs-Singing-Evidenz fehlt.
        # Neutraler Offset 0 dB + Warnung mit Begründung — ein fehlender
        # Messwert darf das Gate produktiv nicht beeinflussen.
        w.panns_singing = None
        w.offset_db = 0.0
        w.reasons.append(
            "PANNs-Singing-Evidenz nicht verfügbar (kein panns_singing / "
            "nicht-numerisch / Modell nicht geladen) — neutraler Offset 0 dB"
        )
        logger.warning(
            "Musik-VAD-Offset nicht berechenbar (%s) — Gate bleibt neutral (§V6 (copilot-instructions.md))",
            w.reasons[-1],
        )
        return w

    w.panns_singing = singing
    if singing < _MIN_SINGING_CONFIDENCE:
        # Unter der Mindest-Konfidenz gilt die Evidenz als „nicht vorhanden"
        # (konservativ: kein Offset statt Raten, §G9 (copilot-instructions.md)).
        w.offset_db = 0.0
        w.reasons.append(
            f"PANNs-Singing={singing:.3f} unter Mindest-Konfidenz "
            f"{_MIN_SINGING_CONFIDENCE:.2f} — kein Offset (konservativ)"
        )
    else:
        # Deterministische affine Skalierung auf das bestehende Silero-Band.
        w.offset_db = round(-_MAX_OFFSET_DB * singing, 4)

    return w


def compute_music_vad_offset_from_audio(
    audio: np.ndarray,
    sr: int,
    position_ratio: float = 0.5,
) -> MusicVADOffsetWitness:
    """Ersatz-Evidenzpfad: PANNs-Singing direkt aus dem Audio-Signal ableiten.

    Ruft ``plugins.panns_plugin.get_panns_plugin().get_tags()`` auf dem
    downgemischten Signal auf und liest die „Singing voice"-Konfidenz.
    Deterministisch (§G5 (copilot-instructions.md)): gleiche Gewichte +
    gleicher Input. Silent-Failure (§V6 (copilot-instructions.md)): PANNs
    nicht geladen/ONNX nicht verfügbar → None-Konfidenz → neutraler Offset
    0 dB mit Warnung + Begründung.

    Hinweis: Im Produktionspfad (phase_18) wird die PANNs-Evidenz bereits
    zentral als ``panns_singing``/``panns_tags`` in die Phase-kwargs
    verteilt (unified_restorer_v3._profiled_phase_call) — dieser
    Direkt-Inferenz-Pfad dient primär der A/B- und Offline-Evidenz
    (scripts/benchmark_vad_oracles.py) und Unit-Tests mit Stubs.
    """
    w = MusicVADOffsetWitness(stage="phase_18_noise_gate")
    try:
        from plugins.panns_plugin import get_panns_plugin  # pylint: disable=import-outside-toplevel

        _plugin = get_panns_plugin()
        _mono = _downmix(audio)
        _tags = _plugin.get_tags(_mono, sr)
        if not isinstance(_tags, dict) or not _tags:
            w.panns_singing = None
            w.offset_db = 0.0
            w.reasons.append(
                "PANNs get_tags() lieferte kein Tag-Dict (Modell/ONNX nicht verfügbar) — neutraler Offset 0 dB"
            )
            logger.warning(
                "Musik-VAD-Offset: PANNs-Inferenz ohne Ergebnis (%s) — Gate bleibt neutral "
                "(§V6 (copilot-instructions.md))",
                w.reasons[-1],
            )
            return w
        singing = _singing_from_tags(_tags)
        if singing is None:
            w.panns_singing = None
            w.offset_db = 0.0
            w.reasons.append("PANNs-Tag-Dict ohne Singing-Voice-Konfidenz — neutraler Offset 0 dB")
            logger.warning(
                "Musik-VAD-Offset: PANNs-Tags ohne Singing-Klasse (%s) — Gate bleibt neutral "
                "(§V6 (copilot-instructions.md))",
                w.reasons[-1],
            )
            return w
        # Reuse the deterministic scalar scaling.
        return compute_music_vad_offset_db(singing)
    except Exception as _exc:  # pylint: disable=broad-except
        w.panns_singing = None
        w.offset_db = 0.0
        w.reasons.append(f"PANNs-Inferenz fehlgeschlagen ({_exc}) — neutraler Offset 0 dB")
        logger.warning(
            "Musik-VAD-Offset: PANNs-Inferenz fehlgeschlagen (%s) — Gate bleibt neutral "
            "(§V6 (copilot-instructions.md))",
            _exc,
        )
        return w


def music_vad_witness_log(witness: MusicVADOffsetWitness) -> str:
    """Formatiert einen Musik-VAD-Offset-Befund als EINE Log-Zeile (report-only)."""
    panns_s = f"{witness.panns_singing:.3f}" if witness.panns_singing is not None else "n/a"
    reasons = "; ".join(witness.reasons) if witness.reasons else "—"
    return (
        f"musikVAD zeuge (report-only, stage={witness.stage}): "
        f"panns_singing={panns_s}, offset_db={witness.offset_db:+.2f}, applied={witness.applied}, "
        f"reasons=[{reasons}]"
    )
