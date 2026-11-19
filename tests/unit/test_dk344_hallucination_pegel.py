"""D-K3-44: Hallucination-Guard-Novelty ist pegel-unabhängig (§2.46e-Wurzel-Fix).

Befund 2026-10-07 (User-Log + Messung): `compute_spectral_novelty` verglich
Absolut-Energie — ein reiner Pegel-Restore erzeugte novelty 0.206 (+1 dB) bis
0.369 (+2 dB) und damit FALSCHEN Rollback; Phase 24 macht explizit einen
Pegel-Restore (§2.45a, bis 2.2–4.5 dB je Material). Gemeldeter Prod-Fall:
spectral_novelty=0.385 ≈ +2,1 dB Gain.

Fix: E_after wird auf die Gesamt-Energie von E_before normiert — §2.46e meint
NEUE SPEKTRALANTEILE, nicht mehr Energie. Gepinnt:

  - Pegel-Unabhängigkeit: 0.5×/1.12×/2.0× Gain → novelty < 0.01 (vorher bis 0.37).
  - Perfekte Dropout-Reparatur bleibt << Rollback-Schwelle 0.15.
  - Grobe Halluzination (Fremdsignal ersetzt Signalteil) wird weiter erkannt.
  - Der DSP-Fallback-Metrik rechnet ebenfalls pegel-unabhängig.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.dsp.hallucination_guard import _compute_spectral_novelty_dsp
from backend.core.hallucination_guard import compute_spectral_novelty

SR = 48_000


def _music_mock(n: int, seed: int = 42) -> np.ndarray:
    t = np.arange(n) / SR
    x = (
        0.30 * np.sin(2 * np.pi * 110.0 * t)
        + 0.20 * np.sin(2 * np.pi * 220.0 * t)
        + 0.10 * np.sin(2 * np.pi * 880.0 * t)
    )
    rng = np.random.default_rng(seed)
    return (x + 0.01 * rng.standard_normal(n)).astype(np.float32)


@pytest.mark.parametrize("gain", (0.5, 1.12, 2.0))
def test_novelty_is_level_invariant_primary(gain: float) -> None:
    x = _music_mock(SR * 2)
    nov, _ = compute_spectral_novelty(x, x * gain, sr=SR)
    assert nov < 0.01, f"Pegel x{gain} erzeugt novelty={nov:.4f} — Pegel-Leck (D-K3-44)"


@pytest.mark.parametrize("gain", (0.5, 1.12, 2.0))
def test_novelty_is_level_invariant_dsp_fallback(gain: float) -> None:
    x = _music_mock(SR * 2)
    nov, _ = _compute_spectral_novelty_dsp(x, x * gain, SR)
    assert nov < 0.01, f"DSP-Fallback: Pegel x{gain} erzeugt novelty={nov:.4f}"


def test_perfect_dropout_repair_stays_below_rollback() -> None:
    x = _music_mock(SR * 2)
    d = x.copy()
    d[SR : SR + int(0.12 * SR)] = 0.0
    nov, _ = compute_spectral_novelty(d, x, sr=SR)  # perfekte Reparatur
    assert nov < 0.05, f"perfekte Reparatur novelty={nov:.4f}"


def test_gross_hallucination_still_detected() -> None:
    x = _music_mock(SR * 2)
    rng = np.random.default_rng(7)
    hall = x.copy()
    # Zweite Hälfte durch lautes Fremd-Rauschen ersetzt = grobe Halluzination.
    hall[SR : SR * 2] = (0.2 * rng.standard_normal(SR)).astype(np.float32)
    nov, _ = compute_spectral_novelty(x, hall, sr=SR)
    assert nov > 0.15, f"halbe Signallänge ersetzt → novelty={nov:.4f} (Guard blind?)"
