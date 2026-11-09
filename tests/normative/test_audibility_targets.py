"""
test_audibility_targets.py — Norm-Suite der Reparatur-Zielabnahme.

Hörordnung Ebene 2 (`.github/instructions/hoerordnung.instructions.md`):
Eine Reparatur gilt erst als abgeschlossen, wenn ihr Rest UNTER der
psychoakustischen Maskierungsschwelle liegt — nie bei Messwert 0.

Verankerte Messsemantik (audibility_targets):
  audible       = über der Maskierungsschwelle (Gate-konform, ab Maske hörbar)
  target_db     = threshold_db − margin_db (Sicherheitsziel unter der Maske)
  objective_met = delta_db <= target_db (sicher unter der Hörbarkeitsschwelle)

Klick-Feeds: 1-Sample-Spike (klassischer Declick-Fall des Kerns) und
48-Sample-Klick (1 ms, Spec v10.306 §3.1 „Impulse < 5 ms") für
Hörbarkeitsmessungen. Deterministisch (§G5 (copilot-instructions.md)), fail-closed.
"""

from __future__ import annotations

import numpy as np
import pytest

SR = 48_000


def _tone(freq: float = 440.0, amp: float = 0.5, n: int = SR) -> np.ndarray:
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def _click(sig: np.ndarray, pos: int = 3000, samples: int = 48, amp: float = 0.5) -> np.ndarray:
    """Klick-Feed; samples=1 = klassischer Spike, 48 = 1-ms-Impuls (§3.1)."""
    out = sig.copy()
    out[pos : pos + samples] = amp
    return out


@pytest.mark.timeout(120)
def test_residual_unter_maske_unhoerbar():
    """Schwaches Residual in lauter Umgebung liegt sicher unter Ziel."""
    from backend.core.dsp.audibility_targets import residual_audibility

    context = _tone(amp=0.5)
    residual = _tone(amp=0.002)  # ≈ −48 dB unter dem Masker, gleiche Frequenz
    report = residual_audibility(residual, context, SR, margin_db=6.0)
    assert report["objective_met"] is True, f"unerwartet nicht erreicht: {report}"
    assert report["audible"] is False


@pytest.mark.timeout(120)
def test_klick_ab_maske_hoerbar():
    """Ein 1-ms-Klick über der Tonalmaske ist hörbar — ab Maske, nicht erst ab Maske+Marge."""
    from backend.core.dsp.audibility_targets import residual_audibility

    context = _tone(amp=0.02)  # leise Tonalmaske
    residual = _click(np.zeros(SR), pos=3000, samples=48, amp=0.5)
    report = residual_audibility(residual, context, SR, margin_db=6.0)
    assert report["overshoot_db"] > 0, f"Klick nicht über Maske: {report}"
    assert report["audible"] is True, f"hörbarer Klick übersehen: {report}"
    assert report["objective_met"] is False


@pytest.mark.timeout(120)
def test_marge_ist_sicherheitsziel_nach_unten():
    """Größere Marge senkt das Ziel (strengere Abnahme), ändert die Maske nicht."""
    from backend.core.dsp.audibility_targets import residual_audibility

    context = _tone(amp=0.5)
    residual = _tone(amp=0.01)
    lax = residual_audibility(residual, context, SR, margin_db=0.0)
    strict = residual_audibility(residual, context, SR, margin_db=24.0)
    assert strict["threshold_db"] == pytest.approx(lax["threshold_db"])
    assert strict["target_db"] == pytest.approx(lax["target_db"] - 24.0)
    assert bool(lax["objective_met"]) >= bool(strict["objective_met"])


@pytest.mark.timeout(120)
def test_repair_objective_saubere_declick_reparatur():
    """AR-Reparatur des Declick-Kerns besteht die Doppelabnahme."""
    from backend.core.dsp.audibility_targets import verify_declick_repair
    from backend.core.dsp.declick_core import detect_click_mask, repair_clicks

    sig = _tone(amp=0.3).copy()
    sig[5000] = 0.9  # klassischer 1-Sample-Klick
    mask = detect_click_mask(sig, strictness_k=6.0)
    assert mask[5000], "Klick nicht detektiert"
    repaired = repair_clicks(sig, mask)
    report = verify_declick_repair(sig, repaired, mask, SR)
    assert report["material_preserved"] is True, f"Material verändert: {report}"
    assert report["rest_subaudible"] is True, f"hörbarer Restdefekt: {report}"
    assert report["objective_met"] is True


@pytest.mark.timeout(120)
def test_repair_objective_materialfresser_faellt_durch():
    """Fehlender Oberton neben dem Rest ist hörbarer Materialverlust (Never-worsen).

    (Kalibrierung: gleichfrequente Pegeländerung ist korrekt maskiert —
    Materialverlust zeigt sich an ANDERFREQUENTem fehlendem Material.)
    """
    from backend.core.dsp.audibility_targets import verify_declick_repair

    sig = _tone(440.0, 0.4) + _tone(2000.0, 0.25)  # Grundton + Oberton
    mask = np.zeros(SR, dtype=bool)
    mask[3000] = True  # winzige Defektstelle — Rechtfertigung fürs Anfassen
    repaired = _tone(440.0, 0.4)  # „Reparatur" verschluckt den Oberton komplett
    report = verify_declick_repair(sig, repaired, mask, SR)
    assert report["material_preserved"] is False, f"Materialverlust nicht erkannt: {report}"
    assert report["objective_met"] is False


@pytest.mark.timeout(120)
def test_repair_strength_dosierung():
    """Stärke ∝ Hörbarkeits-Marge, geclippt auf [0.25, 1.0] (Hörordnung §4)."""
    from backend.core.dsp.audibility_targets import repair_strength_for

    assert repair_strength_for(0.0) == 0.25
    assert repair_strength_for(6.0) == 0.5
    assert repair_strength_for(12.0) == 1.0
    assert repair_strength_for(48.0) == 1.0


@pytest.mark.timeout(180)
def test_repair_until_inaudible_minimaler_eingriff_und_determinismus():
    """Kleinstes Stärke-Rung mit Zielerreichung gewinnt; zweiter Lauf identisch."""
    from backend.core.dsp.audibility_targets import repair_until_inaudible
    from backend.core.dsp.declick_core import detect_click_mask, repair_clicks

    sig = _tone(amp=0.3).copy()
    sig[7000] = 0.9  # klassischer 1-Sample-Klick
    mask = detect_click_mask(sig, strictness_k=6.0)

    def fn(signal: np.ndarray, strength: float) -> np.ndarray:
        # Schwache Stärke repariert nur die Hauptstelle, starke die ganze Maske
        m = mask.copy()
        if strength < 0.6:
            m = np.zeros_like(mask)
            m[7000] = True
        return repair_clicks(signal, m)

    out1, rep1 = repair_until_inaudible(fn, sig, mask, SR, margin_db=6.0)
    out2, rep2 = repair_until_inaudible(fn, sig, mask, SR, margin_db=6.0)
    assert np.array_equal(out1, out2), "Repair-Iteration nicht deterministisch (§G5 (copilot-instructions.md))"
    assert rep1 == rep2
    assert rep1["iterations"] <= 5, "Iteration nicht begrenzt"
    assert rep1["objective_met"] is True, f"Ziel verfehlt: {rep1}"
    assert rep1["strength_used"] <= 1.0


@pytest.mark.timeout(120)
def test_fail_closed_ohne_kontext():
    """Ohne gültigen Kontext gilt konservativ 'nicht erreicht' (fail-closed)."""
    from backend.core.dsp.audibility_targets import residual_audibility

    report = residual_audibility(np.ones(1024), np.zeros(0), SR)
    assert report["objective_met"] is False, "Fail-closed verletzt"
    empty = residual_audibility(np.zeros(1024), np.ones(1024) * 0.1, SR)
    assert empty["objective_met"] is True


@pytest.mark.timeout(120)
def test_temporal_masking_zone_hebt_schwelle():
    """Hörordnung §4 (2026-11-08): liegt der Residual-Schwerpunkt in einer
    Forward-Masking-Zone des Kontexts, wird die effektive Schwelle angehoben
    (zeitlich UND spektral lokal); ohne Transient keine Anhebung.
    """
    from backend.core.dsp.audibility_targets import residual_audibility

    rng = np.random.default_rng(7)
    ctx = (rng.standard_normal(SR * 6) * 0.01).astype(np.float64)
    ctx[int(SR * 3.0) : int(SR * 3.0) + 96] += 0.9  # 2-ms-Klick (starker Masker)
    res = np.zeros(SR * 6)
    res[int(SR * 3.05) : int(SR * 3.05) + 480] = 0.05 * rng.standard_normal(480)
    rep = residual_audibility(res, ctx, SR)
    assert rep["in_forward_mask_zone"] is True, "Impuls 50 ms nach Klick muss in der Zone liegen"
    assert rep["temporal_boost_db"] > 0.0
    assert rep["threshold_db_eff"] > rep["threshold_db"]
    # Gleicher Impuls, Kontext ohne Klick: keine Zone, keine Anhebung.
    ctx_plain = (rng.standard_normal(SR * 6) * 0.01).astype(np.float64)
    rep2 = residual_audibility(res, ctx_plain, SR)
    assert rep2["in_forward_mask_zone"] is False
    assert rep2["temporal_boost_db"] == 0.0
    assert rep2["threshold_db_eff"] == rep2["threshold_db"]
    # Determinismus (§G5 (copilot-instructions.md))
    rep3 = residual_audibility(res, ctx, SR)
    assert rep3["threshold_db_eff"] == rep["threshold_db_eff"]


@pytest.mark.timeout(120)
def test_temporal_masking_kippt_audible_konservativ():
    """Wirkungs-Invariante: ein Defekt, dessen Pegel zwischen spektraler und
    effektiver Schwelle liegt, ist in der Forward-Masking-Zone unhörbar
    (audible=False), ohne zeitliche Maskierung aber hörbar (audible=True).
    """
    from backend.core.dsp.audibility_targets import residual_audibility

    rng = np.random.default_rng(7)
    ctx = (rng.standard_normal(SR * 6) * 0.01).astype(np.float64)
    ctx[int(SR * 3.0) : int(SR * 3.0) + 96] += 0.9
    res_ref = np.zeros(SR * 6)
    res_ref[int(SR * 3.05) : int(SR * 3.05) + 480] = 0.05 * rng.standard_normal(480)
    rep_ref = residual_audibility(res_ref, ctx, SR, temporal_masking=False)
    rep_tm = residual_audibility(res_ref, ctx, SR)
    boost_db = float(rep_tm["temporal_boost_db"])
    assert boost_db > 0.0
    # Ziel-Pegel: knapp ÜBER der spektralen Schwelle, UNTER der effektiven.
    target_delta = float(rep_ref["threshold_db"]) + boost_db / 2.0
    gain = 10.0 ** ((target_delta - float(rep_ref["delta_db"])) / 20.0)
    res_new = res_ref * gain
    a_no = residual_audibility(res_new, ctx, SR, temporal_masking=False)
    a_tm = residual_audibility(res_new, ctx, SR)
    assert a_no["audible"] is True, f"ohne temporal: {a_no}"
    assert a_tm["audible"] is False, f"mit temporal: {a_tm}"


@pytest.mark.timeout(120)
def test_cross_ear_bmld_bonus():
    """Hörordnung §4 (L3, 2026-11-09): ein diotischer Rest (beide Ohren
    kohärent) bekommt den Cross-Ear-Bonus (bis 6 dB, §8b-BMLD konservativ);
    ein inkohärenter Rest nicht; ohne stereo_pair bleibt alles beim Alten.
    """
    from backend.core.dsp.audibility_targets import residual_audibility

    rng = np.random.default_rng(7)
    ctx = (rng.standard_normal(SR * 6) * 0.01).astype(np.float64)
    res = np.zeros(SR * 6)
    res[SR * 3 : SR * 3 + 480] = 0.05 * rng.standard_normal(480)
    dio = np.stack([res, res])
    rng2 = np.random.default_rng(8)
    ink = np.stack([res, np.zeros_like(res)])
    ink[1, SR * 3 : SR * 3 + 480] = 0.05 * rng2.standard_normal(480)
    ctx_st = np.stack([ctx, ctx])
    a = residual_audibility(res, ctx, SR, temporal_masking=False, stereo_pair=(dio, ctx_st))
    b = residual_audibility(res, ctx, SR, temporal_masking=False, stereo_pair=(ink, ctx_st))
    assert a["binaural_boost_db"] > 5.0, f"diotisch: {a['binaural_boost_db']}"
    assert a["bmld_rest_corr"] > 0.95
    assert b["binaural_boost_db"] < 2.0, f"inkoherent: {b['binaural_boost_db']}"
    assert a["threshold_db_eff"] > b["threshold_db_eff"]
    # Rückwärtskompatibilität: ohne stereo_pair kein Bonus.
    c = residual_audibility(res, ctx, SR, temporal_masking=False)
    assert c["binaural_boost_db"] == 0.0
    # Determinismus (§G5 (copilot-instructions.md))
    a2 = residual_audibility(res, ctx, SR, temporal_masking=False, stereo_pair=(dio, ctx_st))
    assert a2["binaural_boost_db"] == a["binaural_boost_db"]
