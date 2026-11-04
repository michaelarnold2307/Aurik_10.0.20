from __future__ import annotations

"""Fokussierte Tests für den korrigierten transparenz-Messkontrakt (2026-11-04).

Ursachen-Report: docs/reports/current/2026-11-04_transparenz_ursache_fix.md

Zwei Korrekturen (gegen §9.7.13/§9.7.25 + PMGG-Identität
02_pipeline_architecture.md:4152 „identisch-semantisch zu PMGG/finaler
TransparenzMetric"):
1. Keine Dauer-Konflation: der wall-clock Short-Form-Blend machte den Score von
   der Excerpt-Dauer statt vom Spektralinhalt abhängig. Auf 2,5-s-B3-Fenstern
   (b3_stufenmessung.py:227) war der Score auf [0.475, 0.525] gestaucht
   (reliability 0.05) — der Zielkorridor 0.55
   (backend/core/calibration_matrix.py:1132) war dort mathematisch unerreichbar.
2. Near-Silence neutral (PMGG-Identität): 0.5 statt 0.0 („katastrophal").

Der Alt-Pfad ist über ``short_form_prior=True`` reproduzierbar (B3-Muster:
Vertrag alt vs. neu auf demselben Audio) — die Alt-Pfad-Tests belegen den
Defekt, die Neu-Vertrags-Tests den Fix (Prove-It).
"""

import numpy as np

_SR = 48000
_B3_FENSTER_S = 2.5


# ---------------------------------------------------------------------------
# Fixtures: spektral konzentriertes Signal, Breitband-Mix, Rauschen
# ---------------------------------------------------------------------------


def _konzentriertes_signal(dauer_s: float, sr: int = _SR) -> np.ndarray:
    """Reinton 400 Hz — Energie fast vollständig im Oktavband 250–500 Hz."""
    t = np.arange(int(sr * dauer_s), dtype=np.float32) / sr
    return (0.5 * np.sin(2 * np.pi * 400.0 * t)).astype(np.float32)


def _breitband_mix(dauer_s: float, sr: int = _SR) -> np.ndarray:
    """Harmonischer Stapel + HF-Partial + Rauschen — nicht BW-limitiert.

    HF/LF-Ratio ≥ 0.05 (gemessen am Spektrum: 6-kHz-Partial trägt HF) — damit
    nutzt die Metrik alle 5 Oktavbänder wie PMGG (kein BW-Limited-Drop).
    """
    t = np.arange(int(sr * dauer_s), dtype=np.float32) / sr
    sig = (
        0.5 * np.sin(2 * np.pi * 350.0 * t) + 0.2 * np.sin(2 * np.pi * 700.0 * t) + 0.1 * np.sin(2 * np.pi * 6000.0 * t)
    )
    rng = np.random.default_rng(11)
    sig = sig + 0.05 * rng.standard_normal(len(t))
    return sig.astype(np.float32)


def _weissrauschen(dauer_s: float, sr: int = _SR, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.1 * rng.standard_normal(int(sr * dauer_s))).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Dauer-Konflation: B3-Fenster (2,5 s) — Zielkorridor 0.55 erreichbar
# ---------------------------------------------------------------------------


def test_b3_fenster_korridor_0_55_ist_erreichbar():
    """Auf 2,5-s-Fenstern darf der Score nicht mehr auf 0.525 gedeckelt sein.

    Vor dem Fix: score = 0.05·raw + 0.475 (reliability (2.5−2)/10 = 0.05)
    → Maximalwert 0.525 < Zielkorridor 0.55 — Prove-It: dieser Test schlägt
    mit short_form_prior=True (Alt-Vertrag) fehl.
    """
    from backend.core.musical_goals.musical_goals_metrics import TransparenzMetric

    met = TransparenzMetric()
    sig = _konzentriertes_signal(_B3_FENSTER_S)
    score = met.measure(sig, _SR)
    assert score >= 0.8, (
        f"Konzentriertes 2,5-s-Fenster muss hoch scoren (Korridor 0.55 ist damit erreichbar); bekam {score}"
    )


def test_alt_vertrag_deckelt_b3_fenster_auf_0_525():
    """Prove-It für den Defekt: der Alt-Vertrag staucht 2,5-s-Fenster auf ≤ 0.525.

    short_form_prior=True reproduziert bit-genau den Blend vor dem Fix
    (score = 0.05·raw + 0.475).
    """
    from backend.core.musical_goals.musical_goals_metrics import TransparenzMetric

    met = TransparenzMetric()
    sig = _konzentriertes_signal(_B3_FENSTER_S)
    neu = met.measure(sig, _SR)
    alt = met.measure(sig, _SR, short_form_prior=True)

    assert alt <= 0.525 + 1e-9, f"Alt-Vertrag-Maximum ist 0.525, bekam {alt}"
    assert abs(alt - (0.05 * neu + 0.475)) <= 1e-6, (
        f"Alt-Pfad muss exakt dem Blend 0.05·neu + 0.475 entsprechen: alt={alt}, erwartet={0.05 * neu + 0.475}"
    )
    assert neu - alt >= 0.25, "Blend muss den Score material stauchen (Faktor 20)"


def test_keine_dauerkonflation_gleicher_spektralinhalt():
    """Gleicher Spektralinhalt ⇒ gleicher Score — unabhängig von der Excerpt-Dauer.

    Vor dem Fix: 2,5 s → 0.525, 10 s → 0.9 (Blend-Stärke hängt an der Dauer).
    """
    from backend.core.musical_goals.musical_goals_metrics import TransparenzMetric

    met = TransparenzMetric()
    kurz = met.measure(_konzentriertes_signal(_B3_FENSTER_S), _SR)
    lang = met.measure(_konzentriertes_signal(10.0), _SR)
    assert abs(kurz - lang) <= 0.05, f"Score darf nicht von der Excerpt-Dauer abhängen: 2.5 s={kurz}, 10 s={lang}"


def test_diskriminierung_auf_b3_fenstern():
    """Konzentriert vs. verrauscht muss auf 2,5-s-Fenstern klar unterscheiden.

    Vor dem Fix lagen beide im Blend-Korridor [0.475, 0.525] (Δ ≤ 0.05).
    """
    from backend.core.musical_goals.musical_goals_metrics import TransparenzMetric

    met = TransparenzMetric()
    ton = met.measure(_konzentriertes_signal(_B3_FENSTER_S), _SR)
    rauschen = met.measure(_weissrauschen(_B3_FENSTER_S), _SR)
    assert ton - rauschen >= 0.6, f"Keine Diskriminierung auf B3-Fenstern: ton={ton}, rauschen={rauschen}"


# ---------------------------------------------------------------------------
# 2. PMGG-Identität (02_pipeline_architecture.md:4152)
# ---------------------------------------------------------------------------


def test_pmgg_identitaet_auf_b3_fenstern():
    """Finale Metrik = PMGG-Proxy (_measure_quick) auf demselben Signal.

    _measure_quick hat keinen Short-Form-Blend — vor dem Fix war die
    „identisch-semantisch"-Zusage auf < 8 s Eingaben gebrochen.
    """
    from backend.core.musical_goals.musical_goals_metrics import TransparenzMetric
    from backend.core.per_phase_musical_goals_gate import _measure_quick

    sig = _breitband_mix(_B3_FENSTER_S)
    final = TransparenzMetric().measure(sig, _SR)
    proxy = float(_measure_quick(sig.copy(), _SR, precise_override=False, enable_vocal_guard=False)["transparenz"])
    assert abs(final - proxy) <= 1e-6, f"PMGG-Identität gebrochen: final={final}, PMGG-Proxy={proxy}"


def test_near_silence_neutral_statt_katastrophal():
    """Near-Silence ist unmessbar → 0.5 (PMGG-Identität), nicht 0.0."""
    from backend.core.musical_goals.musical_goals_metrics import TransparenzMetric

    met = TransparenzMetric()
    stille = np.zeros(int(_SR * _B3_FENSTER_S), dtype=np.float32)
    assert met.measure(stille, _SR) == 0.5
    # Alt-Pfad bleibt bit-identisch (Blend auf den Roherwert 0.0 → 0.475;
    # reliability = (2,5−2)/10 ± Float-Rauschen aus sr+1e-9 → Toleranz 1e-9).
    assert abs(met.measure(stille, _SR, short_form_prior=True) - 0.475) <= 1e-9


# ---------------------------------------------------------------------------
# 3. Referenz-Freiheit (§0d-Shift-Immunität, Spec: reference API-compat only)
# ---------------------------------------------------------------------------


def test_referenz_wird_laut_vertrag_ignoriert():
    """transparenz ist reference-free by design — der §0d-Referenz-Shift
    (unified_restorer_v3.py) darf diesen Score nicht beeinflussen."""
    from backend.core.musical_goals.musical_goals_metrics import TransparenzMetric

    met = TransparenzMetric()
    sig = _breitband_mix(_B3_FENSTER_S)
    ohne = met.measure(sig.copy(), _SR)
    mit = met.measure(sig.copy(), _SR, reference=_weissrauschen(_B3_FENSTER_S) * 3.0)
    assert ohne == mit


# ---------------------------------------------------------------------------
# 4. Support-Guard: Fetzen-Eingaben (< 3 STFT-Frames) shrinken zum Neutral-Prior
# ---------------------------------------------------------------------------


def test_fetzen_eingaben_shrinken_zum_neutral_prior():
    """< 3 STFT-Frames (≈ 0,13 s @ 48 kHz) → Estimator-Support fehlt →
    Neutral-Shrinkage; ab Support volle Auflösung (hier: 2,5 s ≈ 56 Frames)."""
    from backend.core.musical_goals.musical_goals_metrics import TransparenzMetric

    met = TransparenzMetric()
    fetzen = _konzentriertes_signal(0.05)  # 2400 Samples → 1 Frame-Support
    ganz = _konzentriertes_signal(_B3_FENSTER_S)
    score_fetzen = met.measure(fetzen, _SR)
    score_ganz = met.measure(ganz, _SR)
    assert 0.5 <= score_fetzen <= 0.9, f"Fetzen-Score erwartet gestaucht, bekam {score_fetzen}"
    assert score_ganz - score_fetzen >= 0.05, "Support-Guard muss bei Fetzen-Eingaben wirken"


# ---------------------------------------------------------------------------
# 5. Determinismus (§G5 (copilot-instructions.md))
# ---------------------------------------------------------------------------


def test_messung_ist_deterministisch():
    """Gleicher Input + gleiche Version ⇒ bit-identisches Messergebnis (§G5 (copilot-instructions.md))."""
    from backend.core.musical_goals.musical_goals_metrics import TransparenzMetric

    met = TransparenzMetric()
    sig = _breitband_mix(_B3_FENSTER_S)
    erst = met.measure(sig.copy(), _SR)
    zweit = met.measure(sig.copy(), _SR)
    assert erst == zweit
    assert met.measure(sig.copy(), _SR, short_form_prior=True) == met.measure(sig.copy(), _SR, short_form_prior=True)
