"""§D-K3-52 (2026-10-07, Nutzer-Vorgabe): Gesangs-Gate für das ConsonantEnhancement.

Aurik ist ein MUSIK-Restaurierungssystem — der Frikativ-Boost wirkt auf die
Konsonanten im GESANG; Sprache hat eine untergeordnete Position. PANNs-Singing
ist der Zeuge (§III.11 (copilot-instructions.md): Zeugen verändern das Signal
nicht); unsichere Präsenz skaliert linear (§III.10-Muster). Unter 0,15 bleibt
das Signal bit-identisch. Der Frikativ-Detektor selbst ist DSP-basiert
(ZCR/HF, kein ML — plugins/consonant_detector.py, „Kein ML-Modell nötig") und
damit domänenneutral; das Gate regelt die ANWENDUNG des Boosts, nicht die
Detektion.

Gepinnt: geschlossenes Gate (bit-identisch), lineare Skalierung 0,15→0,40,
volles Gate, sicherer Default ohne Zeugen.
"""

from __future__ import annotations

import numpy as np

from backend.core.consonant_enhancement import ConsonantEnhancement, enhance_consonants

SR = 48000


def _speech_like(n_s: float = 1.0) -> np.ndarray:
    n = int(n_s * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(11)
    return (0.3 * np.sin(2 * np.pi * 300.0 * t) + 0.03 * rng.standard_normal(n)).astype(np.float32)


def _mask(n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    seg = int(0.025 * SR)
    for k in range(4):
        s = int((0.05 + 0.2 * k) * SR)
        if s + seg > n:
            break
        mask[s : s + seg] = True
    return mask


def _ce_with_mask() -> ConsonantEnhancement:
    ce = ConsonantEnhancement()
    ce._sibilant_mask = lambda mono, sr: _mask(len(mono))  # type: ignore[method-assign]
    return ce


def test_gate_closed_below_threshold_keeps_signal_identical() -> None:
    """panns_singing < 0,15 (z. B. Sprache): kein Boost, Signal bit-identisch."""
    x = _speech_like()
    ce = _ce_with_mask()
    res = ce.enhance(x, SR, voice_gender="female", defect_scores={"bandwidth_loss": 1.0}, panns_singing=0.0)
    assert res.boost_applied_db == 0.0
    assert res.fricative_segments > 0, "Detektion fand Frikative — das Gate blockt nur den Boost"
    assert np.array_equal(np.asarray(res.audio), np.asarray(np.clip(x, -1.0, 1.0)))


def test_linear_singing_scaling_and_full_gate() -> None:
    """Unsichere Präsenz skaliert linear (0,15→0; 0,40→1); voller Gesang → voller Boost."""
    x = _speech_like()
    ce = _ce_with_mask()
    defect = {"bandwidth_loss": 1.0}
    _causal = ce._causal_factor(defect)  # = total/Σ Faktoren (bestehende Semantik)
    r_mid = ce.enhance(x, SR, defect_scores=defect, panns_singing=0.275)  # scale = 0,5
    assert abs(r_mid.boost_applied_db - 6.0 * _causal * 0.5) < 0.05, (
        f"lineare Skalierung verletzt: {r_mid.boost_applied_db}"
    )
    r_hi = ce.enhance(x, SR, defect_scores=defect, panns_singing=0.9)
    assert abs(r_hi.boost_applied_db - 6.0 * _causal) < 0.05, f"volles Gate verletzt: {r_hi.boost_applied_db}"
    assert abs(r_hi.singing_confidence - 0.9) < 1e-9


def test_default_without_singing_witness_is_gate_closed() -> None:
    """Ohne expliziten Gesangs-Zeugen (Default 0,0) bleibt der Boost aus — sicher per Default."""
    x = _speech_like(0.5)
    res = enhance_consonants(x, SR)
    assert res.boost_applied_db == 0.0
    assert res.singing_confidence == 0.0
