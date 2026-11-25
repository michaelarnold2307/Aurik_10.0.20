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
    reparatur_erlaubnis_karte,
    schutz_karte_aus_signalen,
    smr_karte,
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


def test_karte_gleicht_skalar_pro_kachel():
    rng = np.random.default_rng(42)
    smr = rng.uniform(-12.0, 20.0, size=(50, 24))
    schutz = rng.uniform(0.0, 1.0, size=(50, 24))
    karte = reparatur_erlaubnis_karte(smr, schutz)
    assert karte.shape == (50, 24)
    for _ in range(200):
        i = int(rng.integers(0, 50))
        j = int(rng.integers(0, 24))
        erwartet = reparatur_erlaubnis_aus_smr(smr[i, j], schutz[i, j])
        assert abs(float(karte[i, j]) - erwartet) < 1e-6


def test_karte_maskiert_und_schutz():
    smr = np.array([[-5.0, 3.0, 12.0]])
    schutz = np.array([[0.0, 0.5, 1.0]])
    karte = reparatur_erlaubnis_karte(smr, schutz)
    assert float(karte[0, 0]) == 0.0  # maskiert
    assert abs(float(karte[0, 1]) - 0.25) < 1e-6  # 3/6 * (1-0.5)
    assert float(karte[0, 2]) == 0.0  # voll geschützt


def test_karte_shape_mismatch_wirft():
    import pytest

    with pytest.raises(ValueError):
        reparatur_erlaubnis_karte(np.zeros((4, 24)), np.zeros((3, 24)))


def test_karte_determinismus():
    smr = np.array([[0.5, 6.0, 9.0], [-2.0, 3.0, 4.0]])
    schutz = np.array([[0.1, 0.2, 0.3], [0.0, 0.5, 0.9]])
    a = reparatur_erlaubnis_karte(smr, schutz)
    b = reparatur_erlaubnis_karte(smr, schutz)
    assert np.array_equal(a, b)


def test_smr_karte_differenz():
    band = np.array([[40.0, 25.0, 10.0]])
    thr = np.array([[35.0, 25.0, 15.0]])
    karte = smr_karte(band, thr)
    assert np.allclose(karte, [[5.0, 0.0, -5.0]])


def test_smr_karte_shape_mismatch_wirft():
    import pytest

    with pytest.raises(ValueError):
        smr_karte(np.zeros((4, 24)), np.zeros((5, 24)))


def test_schutz_karte_max_kombination():
    vocal = np.array([[0.9, 0.2, 0.0]])
    intent = np.array([[0.1, 0.8, 0.0]])
    karte = schutz_karte_aus_signalen(vocal, intent)
    assert np.allclose(karte, [[0.9, 0.8, 0.0]])


def test_schutz_karte_geclippt_und_shape_wache():
    import pytest

    assert np.allclose(schutz_karte_aus_signalen(np.array([[1.5]]), np.array([[-0.5]])), [[1.0]])
    with pytest.raises(ValueError):
        schutz_karte_aus_signalen(np.zeros((2, 24)), np.zeros((3, 24)))


def test_endekette_smr_schutz_erlaubnis():
    """Gesamt-Kette: Band-Energie − Schwelle → SMR → Erlaubnis je Kachel."""
    band = np.array([[40.0, 25.0, 10.0, 40.0]])
    thr = np.array([[35.0, 25.0, 15.0, 35.0]])
    vocal = np.array([[0.0, 0.0, 0.0, 1.0]])
    intent = np.array([[0.0, 0.0, 0.0, 0.0]])
    smr = smr_karte(band, thr)
    schutz = schutz_karte_aus_signalen(vocal, intent)
    erlaubnis = reparatur_erlaubnis_karte(smr, schutz)
    assert float(erlaubnis[0, 0]) > 0.5  # SMR 5 dB über Schwelle
    assert float(erlaubnis[0, 1]) == 0.0  # exakt an der Schwelle → 0
    assert float(erlaubnis[0, 2]) == 0.0  # maskiert
    assert float(erlaubnis[0, 3]) == 0.0  # voll geschützter Gesang
