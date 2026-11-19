"""§D-K3-50: ConsonantEnhancement — Wirkungs-Invariante statt unerfüllbarer Pseudo-SNR.

Produktionsbefund 2026-10-07 22:12:27: „SNR-Invariante nicht erfüllt (Δ0.4 dB <
3.0 dB)" — die Forderung war auf der Band-vs-Ausserband-Metrik MATHEMATISCH
UNERFÜLLBAR: ein segmentweiser Boost hebt sie höchstens um 10·log10(1+f(g²−1)) —
Messung 2026-10-07: f=0,10 → 0,22 dB; f=1,0 → 2,76 dB bei MAX_BOOST_DB=6; selbst
der Maximal-Boost überall erreicht die 3 dB nicht. Die 3-dB-Zahl stammt aus der
CHAIN-Invariante §2.8 (korrekt implementiert in phase_43_ml_deesser).

Fix: Wirkungs-Invariante — Band-RMS auf den Frikativ-Segmenten vor/nach dem
Boost (dort wirkt er 1:1: out = x + band·(g−1)), Toleranz 0,75 dB. Gepinnt: die
1:1-Wirkung, die Unerreichbarkeit der alten Metrik (Wurzel-Schutz) und
process() mit synthetischer Maske (invariant_met=True, ehrliche Werte).
"""

from __future__ import annotations

import numpy as np

from backend.core.consonant_enhancement import (
    MAX_BOOST_DB,
    ConsonantEnhancement,
    _band_rms_on_mask,
    _snr_in_band,
)

SR = 48000
_F_LO, _F_HI = 6000.0, 12000.0


def _speech_like(n_s: float = 2.0) -> np.ndarray:
    n = int(n_s * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(42)
    return (0.30 * np.sin(2 * np.pi * 300.0 * t) + 0.03 * rng.standard_normal(n)).astype(np.float32)


def _fricative_mask(n: int, n_segments: int = 8, seg_ms: float = 25.0) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    seg = int(seg_ms * 1e-3 * SR)
    for k in range(n_segments):
        s = int((0.05 + 0.2 * k) * SR)
        if s + seg > n:
            break
        mask[s : s + seg] = True
    return mask


def test_boost_effect_is_measured_one_to_one_on_segments() -> None:
    """§D-K3-50-Kern: Wirkung auf den Masken-Samples ≈ Ziel-Boost (±0,75 dB)."""
    ce = ConsonantEnhancement()
    x = _speech_like()
    mask = _fricative_mask(len(x))
    before = _band_rms_on_mask(x, SR, _F_LO, _F_HI, mask)
    for gain_db in (0.5, 3.0, 6.0):
        y = ce._boost_segment(x, SR, mask, _F_LO, _F_HI, gain_db)
        after = _band_rms_on_mask(y, SR, _F_LO, _F_HI, mask)
        effect_db = float(20.0 * np.log10((after + 1e-12) / (before + 1e-12)))
        assert abs(effect_db - gain_db) <= 0.75, f"Boost {gain_db} dB → Wirkung {effect_db:+.2f} dB"


def test_old_metric_cannot_reach_three_db_at_real_fricative_share() -> None:
    """Wurzel-Regressionsschutz: bei realem Frikativ-Zeitanteil (~10 %) ist die
    alte Forderung (Δ_snr ≥ 3 dB) auf der Band-vs-Ausserband-Metrik unerreichbar
    (Produktionsmessung 2026-10-07: f=0,10 → 0,22 dB)."""
    ce = ConsonantEnhancement()
    x = _speech_like(2.0)
    mask = _fricative_mask(len(x))  # 8 × 25 ms auf 2 s ≈ 10 % Zeitanteil
    y = ce._boost_segment(x, SR, mask, _F_LO, _F_HI, MAX_BOOST_DB)
    delta = _snr_in_band(y, SR, _F_LO, _F_HI) - _snr_in_band(x, SR, _F_LO, _F_HI)
    assert delta < 3.0, f"alte Metrik doch erreichbar? Δ={delta:+.2f} dB"
    assert delta > 0.02, f"kein messbarer Effekt (Fixture-Vertrag)? Δ={delta:+.2f} dB"


def test_process_reports_effect_based_invariant(monkeypatch: object) -> None:
    """enhance() mit synthetischer Maske + Gesangs-Zeuge: invariant_met True, Wirkung ≈ Boost."""
    ce = ConsonantEnhancement()
    x = _speech_like()
    mask = _fricative_mask(len(x))
    ce._sibilant_mask = lambda mono, sr: mask  # type: ignore[method-assign]
    defect = {"bandwidth_loss": 1.0}
    _causal = ce._causal_factor(defect)
    res = ce.enhance(x, SR, voice_gender="female", defect_scores=defect, panns_singing=0.8)
    assert res.invariant_met is True
    expected = 6.0 * _causal  # causal = total/Σ Faktoren (bestehende Semantik)
    assert abs(res.boost_applied_db - expected) < 0.05, f"Boost {res.boost_applied_db} ≠ {expected:.3f}"
    assert abs(res.snr_improvement_db - res.boost_applied_db) <= 0.75, (
        f"Wirkung {res.snr_improvement_db:+.2f} dB ≠ Ziel {res.boost_applied_db:+.2f} dB"
    )


def test_process_without_fricatives_is_passthrough() -> None:
    """Ohne Frikativ-Segmente: keine Verarbeitung, keine Warnung, Invariante erfüllt."""
    ce = ConsonantEnhancement()
    x = _speech_like(0.5)
    ce._sibilant_mask = lambda mono, sr: np.zeros(len(mono), dtype=bool)  # type: ignore[method-assign]
    res = ce.enhance(x, SR, voice_gender="female", defect_scores={}, panns_singing=0.8)
    assert res.invariant_met is True
    assert res.boost_applied_db == 0.0
    assert res.fricative_segments == 0
