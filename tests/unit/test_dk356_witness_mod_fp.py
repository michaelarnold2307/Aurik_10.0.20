"""§D-K3-56 (2026-10-07): Reinhör-Witness `pitch_modulation` war nicht
ausreißer-/voicing-robust.

Wurzel (Repro 2026-10-07, Walzer-Lauf „Trio Schweizer – 13 Tage"):
- (a) Die mod-Tiefe wurde je Seite über die EIGENE Voicing-Maske berechnet —
  Klick-Frames machen den F0-Schätzer einseitig unvoiced, die Runs zerfallen,
  und das Vorher/Nachher-Delta verglich unterschiedliche Run-Strukturen.
- (b) F0-Schätzer-Ausreißer (Oktav-/Kanten-Sprünge, Klick-Triggers) speisten
  das 3–8-Hz-Band mit hunderten Cent (sauberes Vibrato: +814 Cent „mehr
  Modulation" als seine klickige Fassung).
- Betriebsfolge: eine SAUBERE Klick-Patch-Entfernung (das Signal wird besser!)
  meldete +220,7 Cent „pitch_modulation" — Warn-Flut auf jeder Klick-Phase.

Fix: (1) mod-Tiefe beidseitig über die GEMEINSAME Voicing-Maske (identische
Run-Grenzen, Muster `voiced_both` der HNR-Messung); (2) Despike der
Cent-Trajektorie (|ΔF0| > 150 Cent/Frame ist physiologisch unmöglich — Vibrato
≤ 48 Cent/Frame gemessen, Oktav-Sprünge ≥ 600) + Median-Filter (5) für
1-Frame-Spikes. Positiv-Kontrolle: echte zusätzliche 6-Hz-Modulation
(±150 Cent) nach der Phase warnt weiterhin.
"""

from __future__ import annotations

import numpy as np

from backend.core.listening_witness import evaluate_listening_witness

SR = 48_000
DUR = 8.0


def _vibrato_signal(extra_cents: float = 0.0, extra_hz: float = 6.0) -> np.ndarray:
    """Walzer-artiges Vibrato (5,5 Hz, ±60 Cent) auf 220 Hz mit Harmonischen."""
    t = np.arange(int(SR * DUR)) / SR
    m = 60.0 / 1200.0 * np.sin(2 * np.pi * 5.5 * t)
    if extra_cents:
        m = m + extra_cents / 1200.0 * np.sin(2 * np.pi * extra_hz * t)
    ph = 2 * np.pi * 220.0 * np.cumsum(np.exp2(m)) / SR
    y = 0.3 * np.sin(ph)
    for k in (2, 3, 4):
        y += (0.3 / k) * np.sin(k * ph)
    rng = np.random.default_rng(3)
    return (y + 0.005 * rng.standard_normal(t.size)).astype(np.float32)


def _with_clicks(sig: np.ndarray) -> np.ndarray:
    out = sig.copy()
    for pos in (1.3, 2.7, 4.1, 5.9):
        i = int(pos * SR)
        w = int(0.002 * SR)
        out[i - w : i + w] += 0.5
    return out


def _patch_clicks(sig: np.ndarray) -> np.ndarray:
    """Mini-phase_01: Klick-Impulse per Interpolation entfernen (sauberer Patch)."""
    out = _with_clicks(sig)
    for pos in (1.3, 2.7, 4.1, 5.9):
        i = int(pos * SR)
        w = int(0.002 * SR)
        out[i - w : i + w] = np.linspace(out[i - w], out[i + w], 2 * w)
    return out


def _pitch_mod(a: np.ndarray, b: np.ndarray) -> tuple[float, bool]:
    r = evaluate_listening_witness(a, b, SR, "phase_01_click_removal")
    return r.pitch_mod_depth_cents, ("pitch_modulation" in r.findings)


def test_identisches_signal_keine_modulation() -> None:
    x = _vibrato_signal()
    mod, warned = _pitch_mod(x, x.copy())
    assert mod == 0.0
    assert not warned


def test_saubere_klick_patch_entfernung_warnt_nicht() -> None:
    """Kern des Fixes: Klicks → saubere Patches (Signal wird besser) → clean."""
    x = _vibrato_signal()
    x_clicks = _with_clicks(x)
    mod, warned = _pitch_mod(x_clicks, _patch_clicks(x))
    assert mod < 25.0, f"Voicing-/Ausreißer-Delta nicht bereinigt: {mod:.1f}c"
    assert not warned


def test_nur_patch_ohne_klicks_warnt_nicht() -> None:
    """SAUBER → gepatcht (positive Verbesserung): kein False-Positive."""
    x = _vibrato_signal()
    mod, warned = _pitch_mod(x, _patch_clicks(x))
    assert not warned


def test_echte_zusaetzliche_modulation_warnt_weiterhin() -> None:
    """Positiv-Kontrolle: echte +6-Hz-Modulation (±150 Cent) bleibt sichtbar."""
    x = _vibrato_signal()
    worse = _vibrato_signal(extra_cents=150.0)
    mod, warned = _pitch_mod(_with_clicks(x), worse)
    assert warned and mod > 100.0, f"echte Verschlechterung verschluckt: {mod:.1f}c"


def test_verbesserung_richtung_ist_clean() -> None:
    """worse → gepatcht (Verschlechterung entfernt): kein Delta."""
    x = _vibrato_signal()
    worse = _vibrato_signal(extra_cents=150.0)
    mod, warned = _pitch_mod(worse, _with_clicks(x))
    assert mod == 0.0
    assert not warned
