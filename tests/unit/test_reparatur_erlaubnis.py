"""Vertragstests der Feed-Forward-Reparatur-Erlaubnis (2026-11-25, §G14-Muster).

Muster „Maskierungslücke statt Pauschal-Stärke": Die Reparatur-Stärke wird je
Bark×Frame-Kachel VOR der Aktion aus SMR (Defekt-Energie über der
Maskierungsschwelle) und Musik-/Gesangs-Schutz entschieden — statt pauschal
zu wirken und nachher korrektiv zurückzublenden (Voll-Rollback D-K3-109).
Diese Tests pinnen die Dosierungs-Konvention als reine Funktion (§G5).
"""
from __future__ import annotations

import numpy as np

from backend.core.psychoacoustic_masking_model import (
    REPARATUR_ZIEL_MARGE_DB,
    reparatur_erlaubnis_aus_smr,
)


def test_maskiert_bedeutet_kein_eingriff():
    """smr <= 0 → 0.0 (Hörordnung Ebene 2: unter der Maske = kein Defekt)."""
    assert reparatur_erlaubnis_aus_smr(0.0, 0.0) == 0.0
    assert reparatur_erlaubnis_aus_smr(-3.0, 0.0) == 0.0
    assert reparatur_erlaubnis_aus_smr(-12.0, 0.5) == 0.0


def test_voller_schutz_drosselt_vollstaendig():
    """schutz = 1 (z. B. Gesangs-Formanten) → keine Erlaubnis, auch bei klarem SMR."""
    assert reparatur_erlaubnis_aus_smr(12.0, 1.0) == 0.0


def test_proportionalband_zwischen_schwelle_und_marge():
    """0 < smr <= Marge → proportional; Randwerte exakt."""
    assert reparatur_erlaubnis_aus_smr(REPARATUR_ZIEL_MARGE_DB / 2.0, 0.0) == 0.5
    assert reparatur_erlaubnis_aus_smr(REPARATUR_ZIEL_MARGE_DB, 0.0) == 1.0


def test_ueber_marge_volle_staerke():
    assert reparatur_erlaubnis_aus_smr(REPARATUR_ZIEL_MARGE_DB + 1.0, 0.0) == 1.0
    assert reparatur_erlaubnis_aus_smr(30.0, 0.0) == 1.0


def test_schutz_drosselt_multiplikativ():
    """halb Schutz bei halber Band-Erlaubnis → 0.25."""
    assert abs(reparatur_erlaubnis_aus_smr(REPARATUR_ZIEL_MARGE_DB / 2.0, 0.5) - 0.25) < 1e-9


def test_ausgang_immer_in_01():
    for smr in (-20.0, -1.0, 0.01, 3.0, 6.0, 9.0, 40.0):
        for schutz in (-0.5, 0.0, 0.3, 0.7, 1.0, 1.5):
            v = reparatur_erlaubnis_aus_smr(smr, schutz)
            assert 0.0 <= v <= 1.0


def test_determinismus_und_margen_parameter():
    assert reparatur_erlaubnis_aus_smr(3.0, 0.2) == reparatur_erlaubnis_aus_smr(3.0, 0.2)
    # größere Marge = mehr Spielraum vor voller Stärke
    assert reparatur_erlaubnis_aus_smr(4.0, 0.0, ziel_marge_db=10.0) == 0.4
    assert np.isclose(reparatur_erlaubnis_aus_smr(4.0, 0.0, ziel_marge_db=10.0), 0.4)
