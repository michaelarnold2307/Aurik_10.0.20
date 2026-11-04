from __future__ import annotations

"""Fokussierte Tests für den korrigierten transient_energie-Messkontrakt (2026-11-04).

Ursachen-Report: docs/reports/current/2026-11-04_transient_energie_ursache_fix.md

Drei Korrekturen (alle gegen §1.4.6 geprüft):
1. Timing-Invarianz: Wow-/Flutter-/Speed-Korrekturen dürfen nicht als
   Energieverlust zählen (§1.4.6-Abgrenzung zu artikulation ≤ 10 ms).
2. Onset-Set: isolierte Defekt-Impulse ohne Klangkörper zählen nicht (§V36).
3. Referenz-Vertrag: transient_energie misst gegen das Pre-Pipeline-Original.
"""

import numpy as np

_SR = 48000
_EREIGNIS_POSITIONEN = (0.25, 0.75, 1.25, 1.75)
_KLICK_POSITIONEN = (0.5, 1.0, 1.5)


# ---------------------------------------------------------------------------
# Fixtures: musikalische Anschläge (Attack + Sustain) und Defekt-Impulse
# ---------------------------------------------------------------------------


def _ereignis_signal(
    sr: int = _SR,
    dauer_s: float = 2.0,
    amp: float = 0.5,
    sustain_amp: float = 0.2,
) -> np.ndarray:
    """Musikalische Anschläge: 2 ms Attack + 150 ms Sustain (Klangkörper).

    Der Sustain-Anteil unterscheidet den Anschlag vom isolierten Defekt-Impuls
    (Transienten-Modell: Attack 1–20 ms, Sustain 20–200 ms).
    """
    audio = np.zeros(int(sr * dauer_s), dtype=np.float32)
    attack = int(0.002 * sr)
    sustain = int(0.15 * sr)
    for pos in _EREIGNIS_POSITIONEN:
        i0 = int(pos * sr)
        audio[i0 : i0 + attack] = amp
        audio[i0 + attack : i0 + attack + sustain] = sustain_amp
    return audio


def _klicks_einfuegen(audio: np.ndarray, sr: int = _SR) -> np.ndarray:
    """Isolierte Defekt-Impulse (2 ms, kein Sustain) — §V36-Fall."""
    out = audio.copy()
    for pos in _KLICK_POSITIONEN:
        i0 = int(pos * sr)
        out[i0 : i0 + int(0.002 * sr)] = 0.9
    return out


def _verzoegert(audio: np.ndarray, samples: int) -> np.ndarray:
    """Reine Verzögerung ohne Energieverlust (Wow-/Flutter-Surrogat)."""
    out = np.zeros_like(audio)
    out[samples:] = audio[:-samples]
    return out


# ---------------------------------------------------------------------------
# 1. Timing-Invarianz (§1.4.6-Abgrenzung zu artikulation ≤ 10 ms)
# ---------------------------------------------------------------------------


def test_timing_korrektur_wird_nicht_als_energieverlust_bewertet():
    """Eine 6-ms-Verzögerung (Timing-Korrektur) darf den Score nicht senken."""
    from backend.core.musical_goals.transient_energy_metric import get_transient_energy_metric

    sig = _ereignis_signal()
    delayed = _verzoegert(sig, int(0.006 * _SR))
    result = get_transient_energy_metric().measure_transient_energy(audio_input=sig, audio_restored=delayed, sr=_SR)
    assert result["is_valid"] is True
    assert result["transient_energy_score"] >= 0.9, (
        f"Timing-Korrektur (6 ms) fälschlich als Energieverlust gewertet: score={result['transient_energy_score']}"
    )


def test_starer_messpfad_bestraft_timing_korrektur():
    """Der alte Messpfad (stares Fenster, kein Impuls-Filter) zeigt den Defekt.

    alignment_tolerance_ms=0.0 + exclude_impulse_onsets=False reproduzieren den
    Vertrag vor dem Fix; der Score muss dort deutlich niedriger ausfallen.
    """
    from backend.core.musical_goals.transient_energy_metric import get_transient_energy_metric

    sig = _ereignis_signal()
    delayed = _verzoegert(sig, int(0.006 * _SR))
    legacy = get_transient_energy_metric().measure_transient_energy(
        audio_input=sig,
        audio_restored=delayed,
        sr=_SR,
        alignment_tolerance_ms=0.0,
        exclude_impulse_onsets=False,
    )
    assert legacy["transient_energy_score"] <= 0.5, (
        f"Alter Messpfad sollte die 6-ms-Verzögerung als Verlust zeigen, bekam {legacy['transient_energy_score']}"
    )


# ---------------------------------------------------------------------------
# 2. Onset-Set: isolierte Defekt-Impulse (§V36)
# ---------------------------------------------------------------------------


def test_isolierte_defektimpulse_senken_den_score_nicht():
    """Korrekt entfernte Clicks dürfen transient_energie nicht senken (§V36)."""
    from backend.core.musical_goals.transient_energy_metric import get_transient_energy_metric

    sig = _ereignis_signal()
    degraded = _klicks_einfuegen(sig)
    result = get_transient_energy_metric().measure_transient_energy(audio_input=degraded, audio_restored=sig, sr=_SR)
    assert result["is_valid"] is True
    assert result["n_onsets_excluded_impulse"] >= 1, "Defekt-Impulse müssen als isolierte Impulse erkannt werden"
    assert result["transient_energy_score"] >= 0.9, (
        f"Korrekt entfernte Clicks senkten den Score: {result['transient_energy_score']}"
    )
    # Der unverfilterte Score macht die False-Regression sichtbar (Transparenz).
    assert result["transient_energy_score_unfiltered"] <= 0.5, (
        "Ohne Impuls-Filter muss die §V36-False-Regression sichtbar bleiben"
    )


def test_gemessene_abschwaechung_bleibt_sichtbar():
    """Echte Energie-Verluste (−12 dB) dürfen nicht weggefiltert werden."""
    from backend.core.musical_goals.transient_energy_metric import get_transient_energy_metric

    sig = _ereignis_signal()
    result = get_transient_energy_metric().measure_transient_energy(audio_input=sig, audio_restored=sig * 0.25, sr=_SR)
    assert result["is_valid"] is True
    assert 0.1 <= result["transient_energy_score"] <= 0.45, (
        f"0.25×-Abschwächung sollte Score ≈ 0.25 liefern, bekam {result['transient_energy_score']}"
    )


# ---------------------------------------------------------------------------
# 3. Referenz-Vertrag (§1.4.6 „Pre-Pipeline-Original (Reference)")
# ---------------------------------------------------------------------------


def test_referenzvertrag_misst_gegen_pre_pipeline_original(monkeypatch):
    """transient_energie misst gegen pre_pipeline_reference, nicht reference.

    reference = lautere Carrier-Checkpoint (2×) — ohne Vertrag gäbe das einen
    falsch niedrigen Score; mit Vertrag zählt das Pre-Pipeline-Original.
    """
    import backend.core.musical_goals.musical_goals_metrics as _mgm

    monkeypatch.setattr(_mgm, "_is_fast_validation_context", lambda: False)
    checker = _mgm.MusicalGoalsChecker()
    checker.metrics = {}  # 15-Goal-Loop überspringen — nur transient-Pfad wird geprüft
    checker._htdemucs_warmed = True

    input_sig = _ereignis_signal()
    carrier = input_sig * 2.0

    mit_pre = checker.measure_all(
        input_sig.copy(),
        _SR,
        reference=carrier.copy(),
        pre_pipeline_reference=input_sig.copy(),
    )
    ohne_pre = checker.measure_all(
        input_sig.copy(),
        _SR,
        reference=carrier.copy(),
    )
    assert mit_pre["transient_energie"] >= 0.9, (
        f"Referenz-Vertrag verletzt: gegen Pre-Pipeline-Original gemessen "
        f"erwarten wir ≈ 1.0, bekamen {mit_pre['transient_energie']}"
    )
    # Korridor-Bound statt alt-kalibrierter Festzahl: die ±10-ms-Toleranz
    # (§1.4.6, 02_pipeline_architecture.md:4158) hebt bei geclusterten
    # Anschlägen einzelne Fenster — maßgeblich ist die Vertrags-Wirkung:
    # unter den Zielkorridor 0.70 (backend/core/calibration_matrix.py) drücken.
    assert ohne_pre["transient_energie"] < 0.70, (
        f"Carrier-Kopplung (2× lauter) muss unter den Zielkorridor 0.70 drücken, "
        f"bekamen {ohne_pre['transient_energie']}"
    )
    assert mit_pre["transient_energie"] - ohne_pre["transient_energie"] >= 0.25, (
        "Referenz-Vertrag muss den Score material verbessern (2×-Carrier-Kopplung)"
    )


# ---------------------------------------------------------------------------
# Determinismus (§G5 (copilot-instructions.md))
# ---------------------------------------------------------------------------


def test_messung_ist_deterministisch():
    """Gleicher Input + gleiche Version ⇒ bit-identisches Messergebnis (§G5 (copilot-instructions.md))."""
    from backend.core.musical_goals.transient_energy_metric import get_transient_energy_metric

    sig = _klicks_einfuegen(_ereignis_signal())
    erst = get_transient_energy_metric().measure_transient_energy(audio_input=sig, audio_restored=sig.copy(), sr=_SR)
    zweit = get_transient_energy_metric().measure_transient_energy(audio_input=sig, audio_restored=sig.copy(), sr=_SR)
    assert erst == zweit


# ---------------------------------------------------------------------------
# Mess-Rate: Corpus-Fenster mit 44,1 kHz (B3-Harness, Report 2026-11-04 §7)
# ---------------------------------------------------------------------------


def test_messfenster_resampeln_deterministisch_auf_48k():
    """Corpus-Fenster mit 44,1 kHz werden auf die Mess-Rate 48 kHz gebracht
    (01_musical_goals.md §1.4.6) — deterministisch (§G5 (copilot-instructions.md)),
    Länge korrekt; 48-kHz-Eingabe bleibt unverändert."""
    from scripts.b3_transient_energie_neverworsen import _auf_messrate_48k

    sr = 44100
    t = np.arange(int(sr * 0.5), dtype=np.float32) / sr
    audio = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    out1, sr1 = _auf_messrate_48k(audio, sr)
    out2, sr2 = _auf_messrate_48k(audio, sr)
    assert (sr1, sr2) == (48000, 48000)
    assert out1.dtype == np.float32
    # 0,5 s × 48 kHz = 24000 Samples (±1 Sample der Polyphasen-Filterung)
    assert abs(len(out1) - 24000) <= 2
    assert np.array_equal(out1, out2)  # §G5 (copilot-instructions.md): gleicher Input ⇒ gleicher Output

    out3, sr3 = _auf_messrate_48k(out1, 48000)
    assert sr3 == 48000
    assert np.array_equal(out3, out1)  # Passthrough ohne erneutes Resampeln


def test_resample_trifft_die_sample_achse_in_beiden_stereo_layouts():
    """Stereo-Layout-Invariante (AGENTS.md §3): Resample entlang der
    Sample-Achse — (N, C) wie ``_load_clip`` und (C, N) wie pipeline-intern;
    Kanalzahl bleibt erhalten (keine Schein-Kanäle)."""
    from scripts.b3_transient_energie_neverworsen import _auf_messrate_48k

    sr = 44100
    n = int(sr * 0.5)
    links = np.sin(2 * np.pi * 440.0 * np.arange(n) / sr).astype(np.float32)
    rechts = np.sin(2 * np.pi * 220.0 * np.arange(n) / sr).astype(np.float32)

    nc = np.stack([links, rechts], axis=1)  # (N, C) — Layout von _load_clip
    out_nc, _ = _auf_messrate_48k(nc, sr)
    assert out_nc.ndim == 2 and out_nc.shape[1] == 2  # Kanalzahl erhalten
    assert abs(out_nc.shape[0] - 24000) <= 2  # Sample-Achse resampelt

    cn = np.stack([links, rechts], axis=0)  # (C, N) — pipeline-intern
    out_cn, _ = _auf_messrate_48k(cn, sr)
    assert out_cn.ndim == 2 and out_cn.shape[0] == 2  # Kanalzahl erhalten
    assert abs(out_cn.shape[1] - 24000) <= 2  # Sample-Achse resampelt
