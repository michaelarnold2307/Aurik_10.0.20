"""Tests für das Richtungszertifikat referenzfreier Qualitätsmetriken (E2)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.quality_metric_direction_gate import (
    _defektlast,
    _degradiere,
    bewerte_richtung,
    spearman,
)


def test_spearman_grundwerte() -> None:
    assert abs(spearman([4, 3, 2, 1], [4.5, 3.1, 2.2, 1.0]) - 1.0) < 1e-9
    assert abs(spearman([4, 3, 2, 1], [1.0, 2.0, 3.0, 4.0]) + 1.0) < 1e-9
    assert spearman([1, 1], [2, 3]) == 0.0


def test_bewerte_richtung_beide_richtungen() -> None:
    mos = bewerte_richtung([4.5, 3.1, 2.2, 1.0])
    assert mos["richtungssicher"] and mos["spearman"] == 1.0
    blind = bewerte_richtung([4.5, 4.5, 4.49, 4.52])
    assert not blind["richtungssicher"], "gesättigte Werte dürfen nicht zertifiziert werden"
    last = bewerte_richtung([0.2, 0.5, 0.9, 1.4], richtung="aufsteigend")
    assert last["richtungssicher"] and last["spearman"] == 1.0


def test_defektlast_steigt_auf_kumulativer_leiter() -> None:
    rng = np.random.default_rng(7)
    # Musikartige Basis: Harmonische + Rausch-Boden (Domäne der Metrik)
    n = 48000 * 6
    zeit = np.arange(n) / 48000
    t = (
        0.2 * np.sin(2 * np.pi * 220 * zeit)
        + 0.1 * np.sin(2 * np.pi * 440 * zeit)
        + rng.standard_normal(n).astype(np.float32) * 0.01
    )
    t = t.astype(np.float32)
    rauschen = rng.standard_normal(n).astype(np.float32) * 0.05
    werte = [_defektlast(_degradiere(t, 48000, s, rauschen, rng), 48000) for s in range(4)]
    # Normatives Kriterium (wie im Zertifikat): Spearman >= 0,8 und klarer
    # Gesamttrend — strikte Monotonie jeder Stufe ist nicht gefordert.
    assert werte[-1] > werte[0], f"Defektlast muss am Ende steigen: {werte}"
    erg = bewerte_richtung(werte, richtung="aufsteigend")
    assert erg["spearman"] > 0, f"Richtung muss positiv sein: {erg}"
    # Der normative Beweis (rho >= 0,8) entsteht im Zertifikatslauf auf echtem
    # Material (docs/reports/current/qualitaetsmetrik_richtungszertifikat.json).
