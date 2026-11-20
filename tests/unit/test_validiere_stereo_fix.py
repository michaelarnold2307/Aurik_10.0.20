"""Vertragstests: Stereo-Balance-End-to-End-Validierung (scripts/validiere_stereo_fix.py).

Modell-frei: die Balance-Messung ist deterministisch und richtungsrichtig,
der Lauf ist CPU-only + faulthandler-geschützt, der Beleg trägt das Schema
`aurik.snowmine_stereo_fix/1` (Stereo-Befund 2026-11-19, Causal-Reasoner-Fix).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts import validiere_stereo_fix as v

_ROOT = Path(__file__).resolve().parents[2]


def test_balance_db_richtung_und_betrag() -> None:
    links_laut = np.stack([np.full(1000, 0.5, np.float32), np.full(1000, 0.25, np.float32)], axis=1)
    assert abs(v._balance_db(links_laut) - 6.0206) < 0.01  # links +6 dB
    rechts_laut = np.stack([np.full(1000, 0.25, np.float32), np.full(1000, 0.5, np.float32)], axis=1)
    assert abs(v._balance_db(rechts_laut) + 6.0206) < 0.01  # rechts-lauter = negativ
    assert v._balance_db(np.zeros(1000, np.float32)) == 0.0  # Mono ⇒ 0


def test_schema_und_haertung() -> None:
    quelle = (_ROOT / "scripts" / "validiere_stereo_fix.py").read_text(encoding="utf-8")
    assert v.SCHEMA == "aurik.snowmine_stereo_fix/1"
    assert 'os.environ.setdefault("AURIK_FORCE_CPU", "1")' in quelle
    assert "faulthandler.enable" in quelle
    assert "restauriere_schnipsel" in quelle  # Produktionspfad, keine Umgehung


def test_beleg_besser_semantik() -> None:
    """`besser` = |nachher| < |vorher| — Annäherung an die Mitte zählt, nicht das Vorzeichen."""
    beleg_struktur = '"schema": SCHEMA,'
    quelle = (_ROOT / "scripts" / "validiere_stereo_fix.py").read_text(encoding="utf-8")
    assert beleg_struktur in quelle
    assert '"besser": abs(nachher) < abs(vorher),' in quelle


def test_active_intervention_stereo_benefit_verdrahtet() -> None:
    """Snowmine-Befund 2026-11-20: die ActiveIntervention verwarf phase_15, weil
    die MONO-Targets Balance-Korrekturen nicht sehen (delta +0,0004). Der
    Stereo-Integritäts-Benefit (JND 0,5 dB) muss für phase_15/25 im
    Kandidaten-Score stehen."""
    quelle = (_ROOT / "backend" / "core" / "unified_restorer_v3.py").read_text(encoding="utf-8")
    assert "def _stereo_balance_benefit(" in quelle
    assert 'str(phase_id) in ("phase_15_stereo_balance", "phase_25_azimuth_correction")' in quelle
    assert "_score -= _stereo_balance_benefit(_base, _sig)" in quelle
    assert "/ 0.5, 0.0, 4.0" in quelle  # JND 0,5 dB, Deckel 4 JNDs
