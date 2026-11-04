"""§Ebene-3 (Hörordnung) Wohlklang-Ordnungs-Gate — Audit/Verifikation.

Prüft, ob eine gegebene Goal-/Kandidaten-Entscheidung (z. B. das Ranking aus
`goal_priority_protocol.py` oder FeedbackChain-Kandidaten) die **lexikografische
Wohlklang-Ordnung** respektiert.

Hörordnung Ebene 3 (§5 (hoerordnung.instructions.md)): Bei Zielkonflikt gilt die
Reihenfolge strikt — ein höherrangiges Ziel darf für kein niederrangiges gesenkt
werden. Konkret dominiert

    1. Natürlichkeit  >  2. Wärme  >  3. Klarheit/Durchhörbarkeit  >  4. Brillanz

(strikte Dominanz, kein weiches Gewichten; „Teamwork- statt Dominanz-Prinzip",
Spec 01 §1.2c). Die Reihenfolge wird hier **nicht neu erfunden**, sondern
ausschließlich aus `GoalPriorityProtocol.HEARING_TIER_MAP`/`hearing_tier()`
und `PRIORITY_MAP` recycelt (§2.34 (specs/01_musical_goals.md)).

Dieses Modul ist ein **Audit-Gate**: Es verifiziert eine bereits getroffene
Entscheidung; es trifft selbst keine. Ein Verstoß liegt vor, wenn ein Kandidat
ein niederrangiges Goal verbessert (positives Delta) und zugleich ein
höherrangiges Goal verschlechtert (negatives Delta), ohne dass die
Dominanzbedingung der Hörordnung greift.

Deterministisch, numpy-frei, rein deklarativ über die übergebenen Deltas.
Reference: .github/instructions/hoerordnung.instructions.md §5, §8 (Ebene 3).

2026-11-04: Ergänzend zur reinen Auswertung stellt dieses Modul die ZENTRALE
Durchsetzung (`WohlklangOrdnungEnforcer`): Verletzende Anteile werden gekappt
(feste Blend-Stufen, verifiziert gegen das Gate) oder zurückgeworfen (Muster:
ExcellenceOptimizer-Rollback), statt sie nur zu protokollieren. Die
Kandidaten-Entscheidung läuft NUR über die Ordnung + die zentrale
Stärke-Logik (§V7 (copilot-instructions.md)) — keine phasen-individuellen
Schwellwerte (§G5 (copilot-instructions.md): deterministisch).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Status-Konstanten ──────────────────────────────────────────────────────
PASS = "PASS"
VIOLATION = "VIOLATION"
NOT_APPLICABLE = "NOT_APPLICABLE"

# Minimale Delta-Magnitude, ab der eine Score-Änderung als „wirksam" zählt.
# Dies ist KEIN psychoakustischer Schwellwert, sondern nur eine numerische
# Untergrenze gegen Gleitkomma-Rauschen (regt keine Reparatur an).
_EPSILON = 1e-9

# ── Zentrale Enforcement-Konstanten ─────────────────────────────────────────
# Aktionen des Enforcers. Die Kandidaten-Entscheidung läuft NUR über die
# Ordnung + diese zentrale Stärke-Logik (§V7 (copilot-instructions.md):
# keine phasen-individuellen Schwellwerte).
ACCEPT = "ACCEPT"
CAP = "CAP"
ROLLBACK = "ROLLBACK"

# Feste, deterministische Blend-Stufen für die Kappung des verletzenden
# Anteils (absteigend — die erste stufen-konforme gewinnt). Zentrale
# Stärke-Logik statt phasen-individueller Schwellwerte (§V7 (copilot-instructions.md));
# §G5 (copilot-instructions.md): kein Sampling,
# kein Zeitbezug, kein RNG in der Entscheidung.
DEFAULT_BLEND_ALPHAS: tuple[float, ...] = (0.5, 0.25, 0.125, 0.0625)


def _clean_deltas(data: Any) -> dict[str, float]:
    """Filtert nicht-numerische/nicht-endliche Deltas (numerische Robustheit)."""
    cleaned: dict[str, float] = {}
    if not isinstance(data, dict):
        return cleaned
    for goal, delta in data.items():
        try:
            value = float(delta)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        cleaned[str(goal)] = value
    return cleaned


def compute_goal_deltas(before: Any, after: Any) -> dict[str, float]:
    """Goal-Deltas ``after − before`` über die Vereinigung beider Key-Mengen.

    Nicht-numerische/nicht-endliche Werte werden übersprungen (identische
    Semantik wie im Gate). Deterministische Key-Reihenfolge (sortiert).
    """
    _before = before if isinstance(before, dict) else {}
    _after = after if isinstance(after, dict) else {}
    deltas: dict[str, float] = {}
    for key in sorted(set(list(_before.keys()) + list(_after.keys()))):
        try:
            a_val = float(_after.get(key, 0.0))  # type: ignore[arg-type]
            b_val = float(_before.get(key, 0.0))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(a_val) and math.isfinite(b_val):
            deltas[str(key)] = a_val - b_val
    return deltas


def blend_with_reference(candidate: Any, reference: Any, blend_factor: float) -> Any:
    """Zentrale Blend-Anwendung für erzwungene Ordnungs-Entscheidungen (§V7 (copilot-instructions.md)).

    ``blend_factor`` 1.0 → Kandidat unverändert (dasselbe Objekt, bit-identisch,
    §G5 (copilot-instructions.md)); 0.0 → Rollback auf eine Kopie der Referenz
    (Muster: ExcellenceOptimizer-Rollback); dazwischen deterministische
    lineare Überblendung in Richtung Referenz.
    """
    alpha = float(blend_factor)
    if alpha >= 1.0:
        return candidate
    if alpha <= 0.0:
        return reference.copy()
    return alpha * candidate + (1.0 - alpha) * reference


@dataclass
class WohlklangOrdnungResult:
    """Ergebnis des Ebene-3-Audit-Gates.

    Attributes:
        status: "PASS" | "VIOLATION" | "NOT_APPLICABLE".
        violated_goals: Liste der höherrangigen Goals, die zugunsten eines
            niederrangigen gesenkt wurden (bei VIOLATION gefüllt).
        detail: Deutschlandsprachige Kurzbeschreibung des Befunds.
        violations: Liste der einzelnen Verstoß-Paare
            ({"improving_goal", "at_cost_of", "improving_tier",
            "cost_tier"}), für Doku/Metadaten.
    """

    status: str = NOT_APPLICABLE
    violated_goals: list[str] = field(default_factory=list)
    detail: str = ""
    violations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Flache JSON-serialisierbare Repräsentation."""
        return {
            "status": self.status,
            "violated_goals": list(self.violated_goals),
            "detail": self.detail,
            "violations": [dict(v) for v in self.violations],
        }


class WohlklangOrdnungGate:
    """Audit-Gate für die lexikografische Wohlklang-Ordnung (Ebene 3).

    Verifiziert Goal-/Kandidaten-Entscheidungen gegen die strikte
    Dominanzreihenfolge aus GoalPriorityProtocol — ohne eigene Schwellwerte
    oder Neu-Implementierung der Priorisierungslogik.
    """

    def __init__(self, protocol: Any | None = None) -> None:
        """Erzeugt das Gate; injizierter Protocol optional (Default: Singleton)."""
        if protocol is not None:
            self._protocol = protocol
            self._hearing_tier = self._tier_from_protocol(protocol)
        else:
            from backend.core.goal_priority_protocol import get_goal_priority_protocol

            self._protocol = get_goal_priority_protocol()
            self._hearing_tier = self._protocol.hearing_tier

    @staticmethod
    def _tier_from_protocol(protocol: Any):
        """Bindet `hearing_tier` an ein injiziertes Protokoll (falls vorhanden)."""
        if hasattr(protocol, "hearing_tier"):
            return protocol.hearing_tier
        # Fallback über HEARING_TIER_MAP (rekursiv recyceln statt neu erfinden).
        tier_map = getattr(protocol, "HEARING_TIER_MAP", {}) or {}

        def _tier(goal: str) -> int:
            key = str(goal or "").lower().replace(" ", "_").replace("-", "_")
            return int(tier_map.get(key, 3))

        return _tier

    def evaluate(
        self,
        goal_deltas: dict[str, float] | None,
        *,
        threshold: float | None = None,
    ) -> WohlklangOrdnungResult:
        """Prüft eine Kandidaten-Entscheidung gegen die Wohlklang-Ordnung.

        Args:
            goal_deltas: Mapping `{goal: delta}` mit der Score-Änderung je Goal
                (positiv = Verbesserung, negativ = Verschlechterung), wie sie
                ein Kandidat/Eingriff erzeugt (z. B. `scores_after - scores_before`
                aus der FeedbackChain oder ein End-Gate-Ranking).
            threshold: Optionaler Delta-Betrag, ab dem eine Änderung als
                relevant gilt. Default: `None` → nur die numerische
                `_EPSILON`-Untergrenze (kein neuer Wahrnehmungs-Schwellwert).

        Returns:
            WohlklangOrdnungResult mit ``status`` (PASS/VIOLATION/
            NOT_APPLICABLE), ``violated_goals`` und ``detail``.
        """
        data = goal_deltas or {}
        if not isinstance(data, dict) or not data:
            return WohlklangOrdnungResult(
                status=NOT_APPLICABLE,
                detail="keine Goal-Deltas übergeben (keine Entscheidung zu prüfen)",
            )

        # Numerische Robustheit: nicht-endliche/unsaubere Einträge filtern.
        cleaned: dict[str, float] = _clean_deltas(data)

        if not cleaned:
            return WohlklangOrdnungResult(
                status=NOT_APPLICABLE,
                detail="keine auswertbaren Goal-Deltas (leer/NaN/Inf)",
            )

        improving: dict[str, float] = {}  # tier -> [goals], wir speichern je Goal
        degrading: dict[str, float] = {}

        for goal, delta in cleaned.items():
            if delta > _EPSILON:
                improving[goal] = delta
            elif delta < -_EPSILON:
                degrading[goal] = delta

        violations: list[dict[str, Any]] = []
        relevant = abs(float(threshold)) if threshold is not None else 0.0

        for imp_goal, imp_delta in improving.items():
            imp_tier = self._hearing_tier(imp_goal)
            for cost_goal, cost_delta in degrading.items():
                cost_tier = self._hearing_tier(cost_goal)
                # Strikte Dominanz: Verbesserung eines NIEDERRANGIGEN Goals
                # (größere Stufen-Zahl) auf Kosten eines HÖHERRANGIGEN
                # (kleinere Stufen-Zahl) ist verboten.
                # Zielkonflikte innerhalb derselben Stufe sind erlaubt (Tie).
                if imp_tier <= cost_tier:
                    continue
                # Dominanzbedingung: nur ein *wirksamer* Verlust am höherrangigen
                # Goal zählt (Betrag über Schwelle). Verbesserung muss nicht
                # relevant sein — die Richtung allein entscheidet (strikt).
                cost_magnitude = abs(cost_delta)
                if cost_magnitude <= relevant:
                    continue
                violations.append(
                    {
                        "improving_goal": imp_goal,
                        "at_cost_of": cost_goal,
                        "improving_tier": imp_tier,
                        "cost_tier": cost_tier,
                    }
                )

        if violations:
            # Eindeutige, deterministische Reihenfolge der verletzten Goals.
            violated = sorted({v["at_cost_of"] for v in violations})
            for v in violations:
                logger.warning(
                    "§Ebene-3 Wohlklang-Ordnung verletzt: %s (Stufe %d) verbessert auf Kosten von %s (Stufe %d)",
                    v["improving_goal"],
                    v["improving_tier"],
                    v["at_cost_of"],
                    v["cost_tier"],
                )
            return WohlklangOrdnungResult(
                status=VIOLATION,
                violated_goals=violated,
                detail=(
                    f"{len(violations)} Verstoß/Verstöße: niederrangige(s) Goal(s) "
                    f"verbessert auf Kosten höherrangiger ({', '.join(violated)})"
                ),
                violations=violations,
            )

        return WohlklangOrdnungResult(
            status=PASS,
            detail="Entscheidung respektiert die lexikografische Wohlklang-Ordnung",
        )


@dataclass
class OrdnungEnforcementResult:
    """Ergebnis der zentralen Ordnungs-Durchsetzung (2026-11-04).

    Attributes:
        action: "ACCEPT" (kein Verstoß — Kandidat bit-identisch unverändert),
            "CAP" (verletzender Anteil gekappt, blend_factor < 1) oder
            "ROLLBACK" (verletzender Anteil zurückgeworfen, blend_factor 0.0,
            Muster: ExcellenceOptimizer-Rollback).
        blend_factor: Zentraler Stärke-Faktor für `blend_with_reference`.
        tier1_violated: True, wenn ein Ebene-1-Goal (Hörstufe 1) verletzt war.
        violated_goals/violations: Verletzungs-Befund VOR der Durchsetzung.
        enforced_deltas: Goal-Deltas des ERZWUNGENEN Zustands (bei ROLLBACK
            alle 0.0) — Basis für die Restverletzungs-Berechnung.
        residual_violations: Ordnungs-Verletzungen NACH der Durchsetzung
            (verifiziert gegen das Gate — Ziel: 0).
        events: Deutsche Enforcement-Events („Ordnung erzwungen: …" mit
            Begründung; §G8 (copilot-instructions.md)).
        detail: Kurzbeschreibung Befund → erzwungene Maßnahme.
    """

    action: str = ACCEPT
    blend_factor: float = 1.0
    tier1_violated: bool = False
    violated_goals: list[str] = field(default_factory=list)
    violations: list[dict[str, Any]] = field(default_factory=list)
    enforced_deltas: dict[str, float] = field(default_factory=dict)
    residual_violations: int = 0
    events: list[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Flache JSON-serialisierbare Repräsentation."""
        return {
            "action": self.action,
            "blend_factor": self.blend_factor,
            "tier1_violated": self.tier1_violated,
            "violated_goals": list(self.violated_goals),
            "violations": [dict(v) for v in self.violations],
            "enforced_deltas": dict(self.enforced_deltas),
            "residual_violations": self.residual_violations,
            "events": list(self.events),
            "detail": self.detail,
        }


class WohlklangOrdnungEnforcer:
    """Zentrale Durchsetzung der lexikografischen Wohlklang-Ordnung (Ebene 3).

    Verletzt ein Ergebnis die Ordnung (niederrangiges Goal verbessert auf
    Kosten eines höherrangigen — Hörordnung §5 (hoerordnung.instructions.md)),
    wird der verletzende Anteil gekappt (feste Blend-Stufen aus
    `DEFAULT_BLEND_ALPHAS`, jede Stufe gegen das Gate verifiziert) oder
    vollständig zurückgeworfen (Muster: ExcellenceOptimizer-Rollback).

    Die Kandidaten-Entscheidung läuft NUR über das Gate + die zentrale
    Stärke-Logik (§V7 (copilot-instructions.md)): keine phasen-individuellen
    Schwellwerte, deterministisch (§G5 (copilot-instructions.md)).

    Akzeptanz-Prädisposition jeder Kappung (streng — „nie gegen Ebene 1",
    Hörordnung §2 (hoerordnung.instructions.md)):

    1. Gate-Verdikt PASS/NOT_APPLICABLE auf den erzwungenen Deltas,
    2. Never-worsen: kein Goal liegt unter seiner Basis.

    Beide Bedingungen gelten für den Rollback-Zustand (Deltas = 0.0)
    trivialerweise — die Durchsetzung erzeugt daher IMMER einen
    ordnungskonformen, never-worsen-Zustand (Restverletzungen = 0).
    """

    def __init__(self, protocol: Any | None = None) -> None:
        """Erzeugt den Enforcer; Tier-Struktur wird aus dem Gate recycelt."""
        self._gate = WohlklangOrdnungGate(protocol=protocol)

    def enforce(
        self,
        goal_deltas: dict[str, float] | None,
        *,
        threshold: float | None = None,
        blend_probe: Callable[[float], dict[str, float] | None] | None = None,
    ) -> OrdnungEnforcementResult:
        """Setzt die Wohlklang-Ordnung für einen Kandidaten durch.

        Args:
            goal_deltas: Score-Änderung des Kandidaten (positiv = Verbesserung),
                gleiche Semantik wie `WohlklangOrdnungGate.evaluate`.
            threshold: Optionale Delta-Schwelle (an das Gate durchgereicht —
                keine neue Schwelle).
            blend_probe: Optionaler Mess-Callback ``alpha → Deltas bei
                Blend-Faktor alpha`` für die Kappung. Ohne Probe greift der
                vollständige Rollback (ExcellenceOptimizer-Muster).

        Returns:
            OrdnungEnforcementResult mit Aktion, Blend-Faktor, erzwungenen
            Deltas, Restverletzungen und deutschen Enforcement-Events.
        """
        cleaned = _clean_deltas(goal_deltas)
        audit = self._gate.evaluate(goal_deltas, threshold=threshold)

        if audit.status != VIOLATION:
            return OrdnungEnforcementResult(
                action=ACCEPT,
                blend_factor=1.0,
                enforced_deltas=dict(cleaned),
                detail=audit.detail or "keine Verletzung der Wohlklang-Ordnung",
            )

        violations = [dict(v) for v in audit.violations]
        violated = sorted({str(v.get("at_cost_of", "")) for v in violations})
        improving = sorted({str(v.get("improving_goal", "")) for v in violations})
        tier1_goals = sorted({str(v.get("at_cost_of", "")) for v in violations if int(v.get("cost_tier", 0)) == 1})
        tier1_violated = bool(tier1_goals)
        _tier1_note = f"; Ebene-1-Schutz aktiv (nie gegen Ebene 1): {', '.join(tier1_goals)}" if tier1_violated else ""

        # Kappung: feste, absteigende Blend-Stufen (zentrale Stärke-Logik).
        if blend_probe is not None:
            for alpha in DEFAULT_BLEND_ALPHAS:
                probe = _clean_deltas(blend_probe(float(alpha)))
                if not probe:
                    continue
                probe_audit = self._gate.evaluate(probe, threshold=threshold)
                if probe_audit.status == VIOLATION:
                    continue
                if any(delta < -_EPSILON for delta in probe.values()):
                    # Never-worsen: keine Kappung, die ein anderes Ziel unter
                    # seine Basis drückt („höhere Ebenen verlieren nie").
                    continue
                return OrdnungEnforcementResult(
                    action=CAP,
                    blend_factor=float(alpha),
                    tier1_violated=tier1_violated,
                    violated_goals=violated,
                    violations=violations,
                    enforced_deltas=dict(probe),
                    residual_violations=len(probe_audit.violations),
                    events=[
                        "Ordnung erzwungen: verletzender Anteil gekappt "
                        f"(blend_factor={float(alpha):.4f}) — niederrangige(s) "
                        f"Goal(s) {', '.join(improving)} auf Kosten von "
                        f"{', '.join(violated)} zurückgestutzt; höhere Ebenen "
                        f"verlieren nicht mehr für niedrigere{_tier1_note}"
                    ],
                    detail=f"{audit.detail} — erzwungen: CAP (blend_factor={float(alpha):.4f})",
                )

        # Rollback (Muster: ExcellenceOptimizer-Rollback): der verletzende
        # Anteil wird vollständig verworfen — Never-worsen hält trivial.
        return OrdnungEnforcementResult(
            action=ROLLBACK,
            blend_factor=0.0,
            tier1_violated=tier1_violated,
            violated_goals=violated,
            violations=violations,
            enforced_deltas=dict.fromkeys(cleaned, 0.0),
            residual_violations=0,
            events=[
                "Ordnung erzwungen: verletzender Anteil zurückgeworfen "
                "(blend_factor=0.0, Muster ExcellenceOptimizer-Rollback) — "
                f"{audit.detail}{_tier1_note}"
            ],
            detail=f"{audit.detail} — erzwungen: ROLLBACK (blend_factor=0.0)",
        )


__all__ = [
    "ACCEPT",
    "CAP",
    "DEFAULT_BLEND_ALPHAS",
    "NOT_APPLICABLE",
    "PASS",
    "ROLLBACK",
    "VIOLATION",
    "OrdnungEnforcementResult",
    "WohlklangOrdnungEnforcer",
    "WohlklangOrdnungGate",
    "WohlklangOrdnungResult",
    "blend_with_reference",
    "compute_goal_deltas",
]
