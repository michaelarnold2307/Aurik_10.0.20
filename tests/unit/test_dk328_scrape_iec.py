"""§D-K3-28/D-K3-53 (2026-10-07, Nutzer-Vorgabe): SOTA-Erkennung für
Scrape-Flutter (IEC 60386) und Inner-Groove-Distortion.

Wurzel D-K3-28: Die vier Bestands-Kanäle des Flutter-/Scrape-Detektors waren für
die IEC-60386-Signatur (Amplitudenmodulation 20–200 Hz, 0,2–1 %) blind — gemessen
wurde AM 20–150 Hz @ 0,7 % → 0.000 in ALLEN Kanälen (die FM-Kanäle messen
Centroid/Momentanfrequenz statt AM; die Seitenband-Fenster prüften nur 40–120 Hz).
Fix: kohärenter Mehrband-Hüllkurven-AM-Kanal (1/3-Oktav, 400 fps,
signal-tragende Bänder) + Raten-Erweiterung 20–180 Hz + Fenster 3–52 dB.

Wurzel D-K3-53: Der IGD-Detektor (Kates 1981) fand nur Summtöne polyphoner
Akkorde — ein EINZELTON mit wachsendem Klirr blieb blind; zusätzlich schloss die
Träger-Maske (80–400 Hz) 440-Hz-Träger aus (Argmax landete auf dem Leakage-Randbin
398,4). Fix: H2/H3-Wachstumspfad (r4 > 1,5·r1, ≥ 2 steigende Viertel) +
Trägerfenster 80–600 Hz.

Gepinnt: IEC-AM 20–150 Hz @ 0,7 % → FLUTTER ≥ 0,5; FM-Seitenband → SCRAPE_FLUTTER
≥ 0,3; IGD-Verlauf → ≥ 0,3; saubere Signale → unauffällig (< 0,05).
"""

from __future__ import annotations

import numpy as np

from backend.core.defect_scanner import DefectScanner

SR = 48_000
DUR = 12.0


def _scanner() -> DefectScanner:
    return DefectScanner(sample_rate=SR)


def _carrier(duration: float = DUR) -> np.ndarray:
    """Breitbandiger Musik-Träger (3 Harmonische + dezentes Rauschen, mono —
    die Einzeldetektoren sind 1D; ``scan()`` faltet Stereo vorher zusammen)."""
    t = np.arange(int(duration * SR)) / SR
    rng = np.random.default_rng(7)
    base = (
        0.20 * np.sin(2 * np.pi * 220.0 * t)
        + 0.10 * np.sin(2 * np.pi * 440.0 * t)
        + 0.05 * np.sin(2 * np.pi * 880.0 * t)
        + 0.02 * rng.standard_normal(t.size)
    )
    return np.asarray(base, dtype=np.float32)


def _am_scrape(x: np.ndarray, rate: float, depth: float = 0.007) -> np.ndarray:
    """IEC-60386-Amplitudenmodulation (pH-Modulation der Bandgeschwindigkeit)."""
    t = np.arange(x.shape[0]) / SR
    return np.asarray(x * (1.0 + depth * np.sin(2 * np.pi * rate * t)), dtype=np.float32)


def _fm_scrape(x: np.ndarray, rate: float, cents: float) -> np.ndarray:
    """Frequenzmodulation (±cents/100 %) — die klassische FM-Scrape-Signatur."""
    n = x.shape[0]
    t = np.arange(n) / SR
    warp = t * (1.0 + (cents / 1200.0) * np.sin(2 * np.pi * rate * t))
    return np.interp(warp, t, x).astype(np.float32)


def _igd_progression(duration: float = DUR) -> np.ndarray:
    """Einzelton (440 Hz), dessen H2/H3-Anteil pro Viertel wächst (Kates 1981)."""
    n = int(duration * SR)
    t = np.arange(n) / SR
    quarter = n // 4
    x = np.zeros(n)
    for k in range(4):
        s = k * quarter
        e = n if k == 3 else (k + 1) * quarter
        tk = t[s:e]
        h2 = 0.005 + 0.060 * k
        h3 = 0.003 + 0.040 * k
        x[s:e] = (
            np.sin(2 * np.pi * 440.0 * tk) + h2 * np.sin(2 * np.pi * 880.0 * tk) + h3 * np.sin(2 * np.pi * 1320.0 * tk)
        )
    return np.asarray(x, dtype=np.float32)


def test_clean_carrier_is_unauffaellig() -> None:
    """Anti-FP: reiner Träger ohne Modulation → beide Kanäle still (< 0,05)."""
    sc = _scanner()
    clean = _carrier()
    assert sc._detect_flutter(clean).severity < 0.05
    assert sc._detect_scrape_flutter(clean).severity < 0.05


def test_iec_am_scrape_20_bis_150_hz_wird_erkannt() -> None:
    """IEC 60386: 0,7 % AM bei 20/60/150 Hz → Flutter-Severity ≥ 0,5."""
    sc = _scanner()
    carrier = _carrier()
    for rate in (20.0, 60.0, 150.0):
        sev = sc._detect_flutter(_am_scrape(carrier, rate)).severity
        assert sev >= 0.5, f"IEC-Scrape {rate:.0f} Hz @ 0,7 % nicht erkannt: {sev:.3f}"


def test_fm_sideband_scrape_wird_erkannt() -> None:
    """FM-Seitenbänder (±4 Cent @ 60 Hz) → Scrape-Flutter-Severity ≥ 0,3."""
    sc = _scanner()
    sev = sc._detect_scrape_flutter(_fm_scrape(_carrier(), 60.0, 4.0)).severity
    assert sev >= 0.3, f"FM-Scrape nicht erkannt: {sev:.3f}"


def test_inner_groove_h2h3_wachstum_wird_erkannt() -> None:
    """IGD: wachsender H2/H3-Anteil eines Einzeltons → Severity ≥ 0,3;
    flacher Sinus ohne Wachstum → < 0,1."""
    sc = _scanner()
    sev = sc._detect_inner_groove_distortion(_igd_progression()).severity
    assert sev >= 0.3, f"IGD-Wachstum nicht erkannt: {sev:.3f}"
    t = np.arange(int(DUR * SR)) / SR
    sine = np.asarray(0.6 * np.sin(2 * np.pi * 440.0 * t), dtype=np.float32)
    assert sc._detect_inner_groove_distortion(sine).severity < 0.1
