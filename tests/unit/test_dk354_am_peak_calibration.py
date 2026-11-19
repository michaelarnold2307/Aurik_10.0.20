"""§D-K3-54 (2026-10-27): Peak-Signifikanz-Kalibrierung des Scrape-AM-Kanals.

Register-Vorgabe (SOTA_DEFICIT_REGISTER D-K3-54) 1:1 umgesetzt und belegt:

1. **Extremwert-Regression:** Bei statisch-tonalen Ad-hoc-Trägern (drei exakt
   harmonische Sinustöne + 20 dB Rauschabstand, statisch, KEINE Musik) darf
   die Rausch-Extremwert-Spitze über ~2160 Modulations-Bins KEINE AM-Rate
   vortäuschen (Befund 2026-10-07: Flutter-Severity 0,75 ohne Defekt).
   Kalibrier-Lauf: 48 Rausch-Realisierungen (Ratio Peak/p99,9 max 1,318) —
   dieser Test pinnt 24 davon als Regression (0 Ereignisse).
2. **Never-worsen:** Echte IEC-60386-AM (0,7 % Tiefe) bleibt detektiert —
   Träger/Raten wie die Spec-Matrix tests/unit/test_dk328_scrape_iec.py
   (20/60/150 Hz) plus die FP-relevante 24/30-Hz-Region. dk328-Ratio-Werte
   (rng 7): 20 Hz 4,91 / 60 Hz 3,33 / 150 Hz 1,49 — alle > k=1,40.
3. **Determinismus** (§G5 (copilot-instructions.md)).

Signifikanz-Formel (zentrale Kalibrierung `_AM_PEAK_SIG_K` /
`_AM_PEAK_SHARP_FACTOR` in backend/core/defect_scanner.py, §V7 (copilot-instructions.md)):
    Peak > k·Perzentil(abs(ΣS_w), 99,9 %)  ∧  lokale Schärfe ±2 Bins,
    Perzentil über den Störgrund (Peak-Region ±3 Bins ausgeschlossen).
"""

from __future__ import annotations

import numpy as np

from backend.core.defect_scanner import DefectScanner

SR = 48000
DUR_S = 5.0
SNR_DB = 20.0
N_NOISE_REALISIERUNGEN = 24


def _statisch_tonaler_traeger(seed: int) -> np.ndarray:
    """Dokumentierter Ad-hoc-Träger (D-K3-54): 3 exakt harmonische Sinustöne.

    220/440/660 Hz (exakt harmonisch), statisch, + weißes Rauschen bei
    20 dB SNR — KEINE Musik (Register-Befund 2026-10-07).
    """
    rng = np.random.default_rng(seed)
    n = int(DUR_S * SR)
    t = np.arange(n, dtype=np.float64) / SR
    sines = np.zeros(n, dtype=np.float64)
    for f0 in (220.0, 440.0, 660.0):
        sines += np.sin(2.0 * np.pi * f0 * t)
    sines /= 3.0
    sig_pow = float(np.mean(sines**2))
    noise_pow = sig_pow / (10.0 ** (SNR_DB / 10.0))
    noise = rng.standard_normal(n) * np.sqrt(noise_pow)
    return sines + noise


def _matrix_traeger(duration: float = 12.0) -> np.ndarray:
    """Breitbandiger Musik-Träger wie test_dk328_scrape_iec._carrier."""
    t = np.arange(int(duration * SR)) / SR
    rng = np.random.default_rng(7)
    base = (
        0.20 * np.sin(2 * np.pi * 220.0 * t)
        + 0.10 * np.sin(2 * np.pi * 440.0 * t)
        + 0.05 * np.sin(2 * np.pi * 880.0 * t)
        + 0.02 * rng.standard_normal(t.size)
    )
    return np.asarray(base, dtype=np.float32)


def _am_richtung(carrier: np.ndarray, rate_hz: float, depth: float = 0.007) -> np.ndarray:
    """IEC-60386-artige Amplitudenmodulation (Scrape-Prinzip) auf dem Träger."""
    n = carrier.shape[-1]
    t = np.arange(n, dtype=np.float64) / SR
    return np.asarray(carrier * (1.0 + depth * np.sin(2.0 * np.pi * rate_hz * t)), dtype=np.float32)


def test_statisch_tonaler_traeger_keine_am_ereignisse() -> None:
    """24 Rausch-Realisierungen des dokumentierten Trägers → 0 AM-Ereignisse.

    Vor dem Fix (D-K3-54-Befund): Rausch-Extremwert-Spitze 24–30 Hz über dem
    Tiefen-Gate → Flutter-Severity 0,75. Nach dem Fix: kein AM-Befund und
    keine Rate (Severity unter der dk328-Sauber-Schwelle 0,05).
    """
    sc = DefectScanner(sample_rate=SR)
    fp_am = 0
    max_sev = 0.0
    for seed in range(N_NOISE_REALISIERUNGEN):
        carrier = _statisch_tonaler_traeger(seed)
        if sc._coherent_scrape_am(carrier) is not None:
            fp_am += 1
        sev = float(sc._detect_flutter(carrier).severity)
        max_sev = max(max_sev, sev)
    assert fp_am == 0, f"{fp_am}/{N_NOISE_REALISIERUNGEN} AM-FP auf statisch-tonalem Träger"
    assert max_sev < 0.05, f"statisch-tonaler Träger erzeugt noch Rate {max_sev:.3f} (≥ 0,05)"


def test_echte_iec_am_bleibt_detektiert() -> None:
    """Never-worsen (§D-K3-28-Matrix): echte AM 0,7 % bleibt detektiert.

    Der Peak realer AM übertrifft die Extremwert-Schwelle (dk328-Ratios
    1,49–4,91 > k=1,40) und ist lokal scharf — die Signifikanz-Korrektur
    darf ihn nicht abschneiden.
    """
    sc = DefectScanner(sample_rate=SR)
    carrier = _matrix_traeger()
    for rate in (20.0, 24.0, 30.0, 60.0, 150.0):
        am = _am_richtung(carrier, rate, depth=0.007)
        res = sc._coherent_scrape_am(am)
        assert res is not None, f"echte IEC-AM {rate:.0f} Hz @ 0,7 % nicht erkannt (D-K3-28-Regression)"
        freq, depth, coh = res
        assert depth >= 0.0035, f"{rate:.0f} Hz: Tiefe {depth:.5f} unter §D-K3-28-Gate"
        assert coh >= 0.5, f"{rate:.0f} Hz: Kohärenz {coh:.3f} unter §D-K3-28-Gate"
        assert sc._detect_flutter(am).severity >= 0.5
        # Frequenz-Genauigkeit nur im verlässlichen Bereich ≤ 60 Hz — bei
        # 120–150 Hz liegt die Peak-Ratio nahe der Extremwert-Schwelle und der
        # Peak wandert (gemessene Kanal-Schwäche, s. `_AM_PEAK_SIG_K`-Kommentar;
        # dk328 pinnt dort ebenfalls nur die Severity, nicht die Frequenz).
        if rate <= 60.0:
            assert abs(freq - rate) <= 5.0, f"{rate:.0f} Hz: Peak-Frequenz {freq:.2f} Hz daneben"


def test_dk354_determinismus() -> None:
    """§G5 (copilot-instructions.md): gleicher Seed ⇒ bit-identisches Ergebnis."""
    sc = DefectScanner(sample_rate=SR)
    a = _statisch_tonaler_traeger(7)
    b = _statisch_tonaler_traeger(7)
    ra = sc._coherent_scrape_am(a)
    rb = sc._coherent_scrape_am(b)
    assert ra == rb
