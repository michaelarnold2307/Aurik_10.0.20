"""U4 — Ziel-Bedarfs-Kopplung für phase_08_transient_preservation (§G188 (GEBOTE.md)).

Spezifikation (docs/reports/current/2026-11-04_u4_phase08_transient_reparatur.md):
- Reparaturbedarf (gemessene, HÖRBARE Ziel-Lücke) ⇒ Phase läuft mit wirksamer,
  evidenzdosierter Stärke (Stärke = gewichteter Ziel-Gap, Wet/Dry = 1,0).
- Kein Bedarf ⇒ bit-identisches Verhalten wie bisher (§G5 (copilot-instructions.md)):
  die Bedarfs-Dosis ist 0.0 und der neue Pfad greift nicht.
- Deterministisch (§G5 (copilot-instructions.md)): keine Zeit-/Zufallsquellen in der Entscheidung.
- Ebene-1-Schutz (Hörordnung §2): höherrangige Ziele dominieren die Dosis-
  Gewichtung; ein Ebene-1-Defizit allein dosiert keine Transienten-Phase.

Umgebungshinweis: Vom Worker ohne Shell statisch gegen die echten Signaturen
geschrieben und vom Manager nach der Ausführung korrigiert (Dosis-Funktion
rechnet den Ziel-Gap direkt); alle 17 Tests sind ausgeführt grün.
"""

import inspect

import numpy as np
import pytest

from backend.core.joint_calibrator import (
    GOAL_NEED_AUDIBILITY_JND,
    GOAL_WEIGHTS,
    goal_need_strength,
    joint_calibrate,
)
from backend.core.phase_effect_catalog import PHASE_EFFECT_CATALOG

# Harte B3-Fenster (Auftrag 2026-11-04, transient_energie gemessen):
WINDOW_1_SCORE = 0.63
WINDOW_2_SCORE = 0.73
RESTORATION_FLOOR = 0.80  # musical_goals.instructions.md:20 (Boden ≥ 0.80, Restoration)
PHASE_ID = "phase_08_transient_preservation"


# ── Reparaturbedarf ⇒ wirksame Dosis ─────────────────────────────────────────


def test_reparaturbedarf_liefert_dosis_aus_zielluecke():
    """Gemessene Ziel-Lücke ⇒ Dosis = gewichteter Ziel-Gap (§G188, §2.70)."""
    need = goal_need_strength(PHASE_ID, {"transient_energie": WINDOW_1_SCORE}, {"transient_energie": RESTORATION_FLOOR})
    assert need == pytest.approx(RESTORATION_FLOOR - WINDOW_1_SCORE, abs=1e-9)
    assert need > 0.0


def test_harte_fenster_liefern_wirksame_dosis():
    """Beide harten Fenster (0.63/0.73 gemessen) erzeugen eine hörbare Dosis > 0."""
    for score in (WINDOW_1_SCORE, WINDOW_2_SCORE):
        need = goal_need_strength(PHASE_ID, {"transient_energie": score}, {"transient_energie": RESTORATION_FLOOR})
        assert need > 0.0, f"Fenster score={score} liefert keine Dosis"
        assert need >= GOAL_NEED_AUDIBILITY_JND, f"Fenster score={score} unter JND"


def test_dosis_ist_gewichteter_gap_ueber_mehrere_ziele():
    """Mehrere Ziel-Lücken ⇒ gewichtetes Mittel der Gaps (keine phasen-individuellen Schwellwerte)."""
    proxies = {"transient_energie": 0.63, "artikulation": 0.70}
    targets = {"transient_energie": 0.80, "artikulation": 0.80}
    gap_te = 0.80 - 0.63
    gap_art = 0.80 - 0.70
    w_te = abs(PHASE_EFFECT_CATALOG[PHASE_ID].goal_impact["transient_energie"]) * GOAL_WEIGHTS["transient_energie"]
    w_art = abs(PHASE_EFFECT_CATALOG[PHASE_ID].goal_impact["artikulation"]) * GOAL_WEIGHTS["artikulation"]
    expected = (w_te * gap_te + w_art * gap_art) / (w_te + w_art)
    need = goal_need_strength(PHASE_ID, proxies, targets)
    assert need == pytest.approx(expected, abs=1e-9)


# ── Kein Bedarf ⇒ bit-identisch (§G5 (copilot-instructions.md)) ─────────────────────────────


def test_kein_reparaturbedarf_keine_dosis():
    """Alle Ziele erfüllt ⇒ Dosis 0.0 ⇒ der alte Pfad bleibt unverändert."""
    proxies = {"transient_energie": RESTORATION_FLOOR, "artikulation": 1.0, "groove": 1.0, "micro_dynamics": 1.0}
    targets = {"transient_energie": RESTORATION_FLOOR}
    assert goal_need_strength(PHASE_ID, proxies, targets) == 0.0


def test_unhoerbarer_ziel_gap_liefert_keine_dosis():
    """Ziel-Lücke unter der Hörbarkeits-Grenze (JND 0.05) ⇒ kein Reparaturbedarf."""
    gap = 0.03  # < GOAL_NEED_AUDIBILITY_JND
    need = goal_need_strength(
        PHASE_ID, {"transient_energie": RESTORATION_FLOOR - gap}, {"transient_energie": RESTORATION_FLOOR}
    )
    assert need == 0.0


def test_unbekannte_phase_liefert_keine_dosis():
    """Phase ohne Goal-Impact-Profil ⇒ 0.0 (kein Bedarf ⇒ bit-identischer Altpfad)."""
    assert goal_need_strength("phase_99_nonexistent", {"transient_energie": 0.10}, {"transient_energie": 0.80}) == 0.0


def test_robust_gegen_leere_und_untypische_eingaben():
    """Empty/None/bool-Schad-Eingaben ⇒ 0.0, kein Crash (Null-Toleranz der Kopplung)."""
    targets = {"transient_energie": 0.80}
    assert goal_need_strength("", {"transient_energie": 0.1}, targets) == 0.0
    assert goal_need_strength(PHASE_ID, None, targets) == 0.0
    assert goal_need_strength(PHASE_ID, {"transient_energie": 0.1}, None) == 0.0
    assert goal_need_strength(PHASE_ID, {"transient_energie": True}, targets) == 0.0
    assert goal_need_strength(PHASE_ID, {"transient_energie": float("nan")}, targets) == 0.0


# ── Determinismus (§G5 (copilot-instructions.md)) ──────────────────────────────────────────────


def test_dosis_ist_deterministisch_und_veraendert_eingaben_nicht():
    """Gleiche Eingaben ⇒ gleiche Dosis; die Mess-Dicts bleiben unverändert."""
    proxies = {"transient_energie": WINDOW_1_SCORE}
    targets = {"transient_energie": RESTORATION_FLOOR}
    first = goal_need_strength(PHASE_ID, dict(proxies), dict(targets))
    second = goal_need_strength(PHASE_ID, dict(proxies), dict(targets))
    assert first == second
    assert proxies == {"transient_energie": WINDOW_1_SCORE}
    assert targets == {"transient_energie": RESTORATION_FLOOR}


# ── Ebene-1-Schutz (Hörordnung §2) ──────────────────────────────────────────


def test_ebene1_ziele_haben_hoehere_dosisgewichte_als_transient_energie():
    """Hörordnung §2: Ebene-1/2-Ziele dominieren die Dosis-Gewichtung."""
    for goal in ("emotionalitaet", "authentizitaet", "natuerlichkeit", "waerme"):
        assert GOAL_WEIGHTS[goal] > GOAL_WEIGHTS["transient_energie"], f"{goal} muss transient_energie überwiegen"


def test_ebene1_defizit_allein_dosiert_keine_transienten_phase():
    """Ein Ebene-1-Defizit allein erzeugt KEINE Dosis für phase_08 — die Phase
    darf nicht auf Kosten höherer Ziele hochgefahren werden (Hörordnung §5)."""
    proxies = {"natuerlichkeit": 0.50, "emotionalitaet": 0.50}
    targets = {"natuerlichkeit": 0.90, "emotionalitaet": 0.90}
    assert goal_need_strength(PHASE_ID, proxies, targets) == 0.0


def test_wohlklang_ordnung_enforcer_bleibt_verdrahtet():
    """Der Wohlklang-Ordnungs-Enforcer (Ebene-3) bleibt importierbar und
    signatur-kompatibel — der Fix unterläuft ihn nicht."""
    from backend.core.wohlklang_ordnung_gate import WohlklangOrdnungEnforcer, WohlklangOrdnungGate

    assert callable(WohlklangOrdnungEnforcer.enforce)
    assert "goal_deltas" in inspect.signature(WohlklangOrdnungEnforcer.enforce).parameters
    assert "goal_deltas" in inspect.signature(WohlklangOrdnungGate.evaluate).parameters


# ── Katalog-Klassifikation (§2.70/§2.76) ────────────────────────────────────


def test_katalog_profil_klassifiziert_phase_08_nach_transient_energie():
    """phase_08 ist im PHASE_EFFECT_CATALOG als transient_energie-Reparatur
    klassifiziert — sonst kann die Bedarfs-Kopplung die Phase nicht erreichen."""
    profile = PHASE_EFFECT_CATALOG.get(PHASE_ID)
    assert profile is not None, "phase_08 fehlt im PHASE_EFFECT_CATALOG (Bedarfs-Kopplung tot)"
    assert profile.phase_id == PHASE_ID
    assert profile.goal_impact.get("transient_energie", 0.0) > 0.0
    assert profile.time_profile in ("fast", "medium", "slow", "heavy")


def test_joint_calibration_liefert_phase_08_aus_der_zielluecke():
    """§2.70 Ende-zu-Ende: die Ziel-Lücke dosiert phase_08 über die zentrale
    Bedarfs-Funktion; joint_calibrate behält für Defekt-Phasen seinen
    §2.70-Vertrag (Defekt-Tiefe × Detektor-Konfidenz) — getrennte Verträge.
    """
    need = goal_need_strength(PHASE_ID, {"transient_energie": WINDOW_1_SCORE}, {"transient_energie": RESTORATION_FLOOR})
    assert need == pytest.approx(RESTORATION_FLOOR - WINDOW_1_SCORE, abs=1e-9)
    result = joint_calibrate(
        [PHASE_ID], {"transient_energie": WINDOW_1_SCORE}, {"transient_energie": RESTORATION_FLOOR}
    )
    assert result[PHASE_ID] == 0.0  # Defektzweig ohne Defekt-Evidenz (§2.70)


# ── Verdrahtung in der zentralen Stärke-/Wet-Dry-Entscheidung (statisch) ─────


def test_uv3_ziel_bedarfs_kopplung_ist_verdrahtet():
    """Der Ziel-Bedarfs-Posterior speist die zentrale Stärke/Wet-Dry-Entscheidung."""
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    src = inspect.getsource(UnifiedRestorerV3)
    assert "_goal_need_posterior" in src, "§U4: Ziel-Bedarfs-Posterior fehlt in UV3"
    assert "phase_goal_need_strength" in src, "§U4: Bedarfs-Kwarg für das Strength-Floor-Gate fehlt"
    assert "elif _goal_need_posterior > 0.0:" in src, "§U4: Bedarfs-Zweig muss hinter der Defekt-Evidenz liegen (elif)"
    assert "_sev_wet_dry = 1.0" in src, "§U4: Wet/Dry = 1,0 (keine Doppel-Dämpfung) fehlt"


def test_uv3_strength_floor_gate_ueberspringt_bei_bedarf_nicht_still():
    """§V6/§G8 (copilot-instructions.md): Bei hörbarem Ziel-Bedarf kein stilles
    Überspringen; der Skip-Grund wird begründet geloggt."""
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    src = inspect.getsource(UnifiedRestorerV3)
    assert "_sfg_goal_need" in src, "§U4: Strength-Floor-Gate kennt den Ziel-Bedarf nicht"
    assert "_sfg_goal_need <= 0.0" in src, "§U4: Gate-Skip muss an fehlenden Ziel-Bedarf gekoppelt sein"
    assert "kein hörbarer Ziel-Bedarf" in src, "§U4: Skip-Log begründet den Skip"


# ── Wirkung auf synthetischen Transient-Signalen (phase_08, echte Signatur) ──


def _attack_probe_signal() -> np.ndarray:
    """Synthetisches Transient-Signal: Sustain-Körper + reguläre Attacken
    (musikalische Transienten mit Klangkörper, keine Defekt-Impulse)."""
    sr = 48000
    n = sr
    t = np.arange(n, dtype=np.float64) / sr
    sig = 0.15 * np.sin(2 * np.pi * 220.0 * t)
    onset_idx = [int(0.20 * sr), int(0.45 * sr), int(0.70 * sr)]
    seg = np.arange(int(0.05 * sr))
    env = np.exp(-seg / (0.006 * sr))
    for i in onset_idx:
        sig[i : i + len(seg)] += 0.6 * env * np.sin(2 * np.pi * 880.0 * seg / sr)
    return sig.astype(np.float32)


def _onset_window_rms(sig: np.ndarray, idx: int, sr: int = 48000, win_ms: float = 5.0) -> float:
    """Onset-Energie im 5-ms-Fenster (Messfenster der transient_energie-Metrik, §1.4.6a)."""
    w = int(sr * win_ms / 1000.0)
    seg = sig[idx : idx + w]
    return float(np.sqrt(np.mean(np.square(seg))))


def test_bedarfs_dosis_laesst_phase_am_signal_wirken():
    """Mit Dosis wirkt die Phase am Signal; ohne Stärke ist sie bit-identischer
    Passthrough (phase_08_transient_preservation.py: zero effective strength)."""
    from backend.core.phases.phase_08_transient_preservation import TransientPreservationPhase

    phase = TransientPreservationPhase()
    sig = _attack_probe_signal()
    active = phase.process(sig.copy(), sample_rate=48000, material_type="vinyl", strength=0.17)
    passthrough = phase.process(sig.copy(), sample_rate=48000, material_type="vinyl", strength=0.0)
    assert np.isfinite(active.audio).all()
    assert not np.allclose(active.audio, sig), "Dosis 0.17 muss das Signal verändern"
    assert np.array_equal(passthrough.audio, sig), "strength=0.0 muss bit-identischer Passthrough sein"
    assert passthrough.modifications.get("transient_preserved") is False


def test_bedarfs_dosis_wirkt_an_transient_attacken():
    """Die evidenzdosierte Wirkung erhöht die Onset-Energie (5-ms-Fenster) —
    die Ziel-Lücke transient_energie (§1.4.6) wird in die richtige Richtung geschlossen."""
    from backend.core.phases.phase_08_transient_preservation import TransientPreservationPhase

    phase = TransientPreservationPhase()
    sig = _attack_probe_signal()
    onset_idx = [int(0.20 * 48000), int(0.45 * 48000), int(0.70 * 48000)]
    out = phase.process(sig.copy(), sample_rate=48000, material_type="vinyl", strength=0.17)
    assert out.modifications.get("num_transients", 0) >= 3, (
        "Onset-Detektion fand die synthetischen Attacken nicht — Signal/Detektion prüfen"
    )
    for i in onset_idx:
        assert _onset_window_rms(out.audio, i) > _onset_window_rms(sig, i), (
            f"Onset-Energie bei sample {i} gestiegen erwartet (5-ms-Fenster)"
        )
