"""§D-K3-59 (2026-10-07): PMGG-Reject-Actions deterministisch klassifizieren.

Produktionsbefund (Walzer-Lauf „Trio Schweizer – 13 Tage", 10.12.12):
    WARNING: §v10.17 PSS-Gate [phase_50_spectral_repair]: PSS=0.8265 rejected
    WARNING: PMGG unbekannte Action: 'pss_rejected' — klassifiziert als 'other'

Die §v10.17-Gates (LAG, PSS) geben bei einem Veto je eine eigene Reject-Action
zurück (``return audio, effective_scores_before, "..._rejected", 0.0``) — die
Klassifizierung ``_classify_action_decision`` kannte sie nicht und warf sie in
den „other"-Fallback samt Warnung (§G8 (copilot-instructions.md): die Zeile
stand, aber die Telemetrie-Klasse war falsch). ``pss_rejected`` UND
``lag_rejected`` (gleicher Codeblock, gleiche Fehlerklasse) sind jetzt als
Klasse „reject" registriert; der Conflict-Report behandelt „reject"
unverändert als konflikt-relevant (ein abgelehnter Phasen-Output ist ein
echtes Guard-Ereignis, kein pass).

Gepinnt:
- ``pss_rejected`` → ("reject", "pss_perceptual_similarity_rejected") ohne Warnung
- ``lag_rejected`` → ("reject", "stereo_lag_gate_rejected") ohne Warnung
- Wurzel-Wächter: JEDE Reject-Action, die die Gate-Datei selbst erzeugt, ist
  klassifizierbar (kein „other", keine „unbekannte Action"-Warnung) — künftige
  Reject-Actions können so nicht mehr still im Fallback landen.
- Gegenprobe: echt unbekannte Actions laufen weiter in den "other"-Fallback
  mit §G8-Warnung (der deterministische Fallback wurde nicht abgeschafft).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate

_GATE_SRC = Path(__file__).resolve().parents[2] / "backend" / "core" / "per_phase_musical_goals_gate.py"


def _classify(action: str) -> tuple[str, str]:
    return PerPhaseMusicalGoalsGate._classify_action_decision(action)


def test_pss_rejected_ist_als_reject_registriert(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        decision_class, reason = _classify("pss_rejected")
    assert decision_class == "reject"
    assert reason == "pss_perceptual_similarity_rejected"
    assert not any("unbekannte Action" in r.message for r in caplog.records), (
        "pss_rejected darf nicht mehr als unbekannte Action gewarnt werden"
    )


def test_lag_rejected_ist_als_reject_registriert(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        decision_class, reason = _classify("lag_rejected")
    assert decision_class == "reject"
    assert reason == "stereo_lag_gate_rejected"
    assert not any("unbekannte Action" in r.message for r in caplog.records), (
        "lag_rejected darf nicht mehr als unbekannte Action gewarnt werden"
    )


def test_alle_vom_gate_erzeugten_reject_actions_sind_klassifizierbar(caplog) -> None:
    """Wurzel-Wächter: die Registrierung ist vollständig oder sie ist es nicht."""
    src = _GATE_SRC.read_text(encoding="utf-8")
    # Action-Literale stehen als 3. Return-Element (gefolgt von ", <strength>");
    # die Reason-Codes der Klassifizierung selbst (letztes Return-Element) matchen bewusst nicht.
    actions = sorted(set(re.findall(r'return [^\n]*?, "([a-z0-9]+_rejected)", ', src)))
    assert {"pss_rejected", "lag_rejected"} <= set(actions), (
        f"Vorbedingung: die Gate-Datei erzeugt pss_rejected + lag_rejected als Returns, gefunden: {actions}"
    )
    with caplog.at_level(logging.WARNING):
        for action in actions:
            decision_class, reason = _classify(action)
            assert decision_class != "other", f"{action} läuft in den other-Fallback ({reason})"
    assert not any("unbekannte Action" in r.message for r in caplog.records)


def test_unbekannte_action_bleibt_other_mit_warnung(caplog) -> None:
    """Gegenprobe: der deterministische Fallback wurde nicht abgeschafft."""
    with caplog.at_level(logging.WARNING):
        decision_class, reason = _classify("fiktive_neue_action_zzz")
    assert decision_class == "other"
    assert reason == "unclassified_action_fallback"
    assert any("unbekannte Action" in r.message for r in caplog.records)
