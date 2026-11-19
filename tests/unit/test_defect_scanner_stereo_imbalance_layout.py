"""Vertragstests: Stereo-Imbalance-Detektion bedient BEIDE Layouts (Snowmine-Fix, 2026-11-19).

Befund: `_detect_stereo_imbalance` las `audio[:, 0]`/`audio[:, 1]` (channels-last).
Für channels-first-Input (C, N) — die Pipeline-Invariante — sind das nur die
ersten ZWEI Samples ⇒ 2-Sample-RMS ⇒ Imbalance unsichtbar ⇒ phase_15_stereo_balance
nie geplant (Snowmine-Befund). Muster-Referenz: `_detect_azimuth_error` (bedient
beide Layouts seit jeher). Diese Tests pinnen: gleiche Severity in beiden Layouts,
korrekte Richtung, Mono/unklare Shapes fail-safe 0,0.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.defect_scanner import DefectScanner


@pytest.fixture()
def scanner() -> DefectScanner:
    return DefectScanner(sample_rate=48000)


def _stereo_48k(rng: np.random.Generator, rechts_db: float = -6.0) -> np.ndarray:
    """Deterministisches Stereo-Testsignal: links Referenz, rechts um `rechts_db` leiser."""
    n = 48000
    t = np.arange(n) / 48000.0
    links = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * rng.standard_normal(n)).astype(np.float32)
    gain = 10.0 ** (rechts_db / 20.0)
    rechts = (links * gain).astype(np.float32)
    return np.stack([links, rechts])  # channels-first (C, N)


def test_severity_identisch_in_beiden_layouts(scanner: DefectScanner) -> None:
    rng = np.random.default_rng(42)
    stereo = _stereo_48k(rng, rechts_db=-6.0)
    s_first = scanner._detect_stereo_imbalance(stereo)
    s_last = scanner._detect_stereo_imbalance(np.ascontiguousarray(stereo.T))
    assert s_first.severity > 0.15, "channels-first muss die Imbalance erkennen"
    assert abs(s_first.severity - s_last.severity) < 1e-6, "beide Layouts müssen dieselbe Severity liefern"
    assert abs(s_first.metadata["balance_ratio"] - s_last.metadata["balance_ratio"]) < 1e-6


def test_richtung_der_imbalance_ist_pegelproportional(scanner: DefectScanner) -> None:
    """Größere Imbalance ⇒ höhere Severity — unterhalb der Sättigung gemessen."""
    rng = np.random.default_rng(7)
    s1 = scanner._detect_stereo_imbalance(_stereo_48k(rng, rechts_db=-1.0)).severity
    s2 = scanner._detect_stereo_imbalance(_stereo_48k(rng, rechts_db=-2.0)).severity
    assert s2 > s1, f"Severity muss mit der Imbalance steigen ({s1} vs {s2})"


def test_mono_und_unklare_shapes_fail_safe(scanner: DefectScanner) -> None:
    assert scanner._detect_stereo_imbalance(np.zeros(48000, dtype=np.float32)).severity == 0.0
    # (2, 2) ist kein eindeutiges Stereo-Layout — fail-safe 0,0 statt Zufalls-Index.
    assert scanner._detect_stereo_imbalance(np.zeros((2, 2), dtype=np.float32)).severity == 0.0
