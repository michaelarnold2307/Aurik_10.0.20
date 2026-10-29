"""Regressionstest §SR-CK-Textur (Spec 03) — dichte Knistern-Textur muss Severity > 0,15 liefern.

Pinnt den Fix aus `docs/reports/current/2026-10-29_crackle_texture_severity_routing_fix.md`:
die `tonal_or_dense_hf`-Guard-Nullung im DefectScanner darf die CRACKLE-Severity bei
echtem dichtem Knistern (Masken-Dichte `severity_raw >= 0,5` UND impulsive HF
`hp_kurtosis >= 6,0`) nicht auf 0 drücken — sonst wird phase_67_crackle_texture_removal
nie selektiert (toter Code), und die Knistern-Textur bleibt hörbar (Hoer-Befund 2026-10-29).

Vorgabe: Spec 03 §SR-CK-Textur (Knistern-Arten-Differenzierung, Severity-Erkennung).
"""

from __future__ import annotations

import numpy as np

SR = 48000


def _scanner():
    from backend.core.defect_scanner import DefectScanner

    return DefectScanner(sample_rate=SR)


def _dense_crackle(duration: float = 3.0, per_sec: int = 900, amp: float = 0.6, seed: int = 7) -> np.ndarray:
    """Dichte impulsive Knistern-Textur: viele kurze HF-Bursts (Bailey et al. 2019)."""
    rng = np.random.default_rng(seed)
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    # leise tonale Basis (Musik) — darf die Textur-Erkennung nicht dominieren
    sig = 0.03 * np.sin(2 * np.pi * 440.0 * t)
    crackle = np.zeros(n)
    for _ in range(int(per_sec * duration)):
        i = int(rng.integers(0, n - 128))
        L = int(rng.integers(24, 96))
        burst = rng.standard_normal(L) * np.hanning(L)
        crackle[i : i + L] += amp * burst
    return (sig + crackle).astype(np.float32)


class TestCrackleTextureSeverity:
    """§SR-CK-Textur: dichte Textur ist ein erkannter Defekt (Severity > 0,15)."""

    def test_dense_texture_severity_above_phase67_threshold(self):
        """Dichte impulsive Knistern-Textur → Severity > 0,15 (→ phase_67-Routing)."""
        sc = _scanner()
        score = sc._detect_crackle(_dense_crackle())
        assert score.severity > 0.15, (
            f"dichte Knistern-Textur muss Severity > 0,15 liefern (phase_67-Routing), bekommen: {score.severity}"
        )

    def test_dense_texture_not_zeroed_by_tonal_guard(self):
        """tonal_or_dense_hf darf dichte impulsive Textur nicht auf Severity 0 nullen."""
        sc = _scanner()
        score = sc._detect_crackle(_dense_crackle())
        md = score.metadata or {}
        # Nur relevant, wenn die Guard-Logik überhaupt griff (tonal_or_dense_hf):
        if md.get("tonal_or_dense_hf_guard") is True:
            assert score.severity > 0.0, (
                "tonal_or_dense_hf hat Severity genullt, obwohl dichte impulsive "
                "Knistern-Textur vorliegt (Widerspruch zur Spec 03 §SR-CK-Textur)"
            )

    def test_dense_texture_routing_condition(self):
        """Phase-67-Routing-Gate: die Severity-Schwelle (> 0,15) ist erfüllt.

        Hinweis: Die zweite Routing-Bedingung `sparse_impulse_rate < 0,0008` (dichte
        Klassifikation) ist eine separate Scanner-Metrik; sie am synthetischen Signal
        zu erzwingen bildet die Messung nicht zuverlässig ab und wird hier nicht
        gepinnt (maßgeblich ist die Severity-Schwelle des Fixes).
        """
        sc = _scanner()
        score = sc._detect_crackle(_dense_crackle())
        assert score.severity > 0.15, (
            f"phase_67-Routing-Schwelle (Severity > 0,15) nicht erfüllt: severity={score.severity}"
        )
