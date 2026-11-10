"""Vertragstests für das Schadens-Bewertungswerkzeug (2026-11-11).

Deckt die drei Fehlerklassen ab, die ein Sinn-Beweis falsch machen kann:
(1) Pegel-Artefakt (die Ziel-Lautheit der Pipeline statt Reparatur messen),
(2) Richtungs-Fehler (defektlast niedriger = besser, musik_mos höher = besser),
(3) Aggregat-/Zähl-Fehler („besser" bei nur einer gekippten Metrik).

Modellfrei: `_metriken` ist gefälscht — geprüft wird die Logik, nicht das Netz.
"""

from __future__ import annotations

import numpy as np
import pytest


def _fake_metric(monkeypatch, werte):
    """Ersetzt die Metrik-Auswertung durch eine feste, reihenfolgeabhängige Liste."""
    import scripts.bewerte_schadens_proben as mod

    aufrufe = iter(werte)

    def _f(audio):
        return dict(next(aufrufe))

    monkeypatch.setattr(mod, "_metriken", _f)


def test_pegel_matching_entfernt_die_ziel_lautheit_als_stoergroesse() -> None:
    """Die Pipeline regelt auf Ziel-Lautheit; ohne Matching misst man den
    Regelkreis. Mit Matching muss der Pegel dem Schaden entsprechen."""
    from scripts.bewerte_schadens_proben import _pegel_matching

    x = np.full(1000, 0.5, dtype=np.float32)
    y = _pegel_matching(x, 0.1)
    assert float(np.sqrt((y.astype(np.float64) ** 2).mean())) == pytest.approx(0.1, abs=1e-6)


def test_bewerte_paar_erkennt_echte_verbesserung(monkeypatch) -> None:
    """defektlast sinkt über die JND UND musik_mos steigt ⇒ besser."""
    import scripts.bewerte_schadens_proben as mod

    _fake_metric(
        monkeypatch,
        [
            {"defektlast": 0.20, "musik_mos": 4.8},  # clean
            {"defektlast": 0.50, "musik_mos": 4.0},  # schaden
            {"defektlast": 0.30, "musik_mos": 4.4},  # restauriert
        ],
    )
    r = mod.bewerte_paar(np.zeros(10), np.zeros(10), np.zeros(10))
    assert r["besser"] is True
    assert r["delta_defektlast"] == pytest.approx(-0.20, abs=1e-6)
    assert r["delta_musik_mos"] == pytest.approx(0.4, abs=1e-6)


def test_bewerte_paar_verlangt_dass_keine_metrik_kippt(monkeypatch) -> None:
    """Nur eine Metrik besser, die andere deutlich schlechter ⇒ NICHT besser.

    Das ist die Lehre aus den B3-Einzel-Flips: eine Verbesserung auf einer
    Achse darf eine Verschlechterung auf der anderen nicht verdecken.
    """
    import scripts.bewerte_schadens_proben as mod

    _fake_metric(
        monkeypatch,
        [
            {"defektlast": 0.20, "musik_mos": 4.8},
            {"defektlast": 0.50, "musik_mos": 4.0},
            {"defektlast": 0.30, "musik_mos": 3.0},  # MOS kippt
        ],
    )
    r = mod.bewerte_paar(np.zeros(10), np.ones(10), np.ones(10))
    assert r["besser"] is False


def test_aggregat_zaehlt_je_rezept_und_ignoriert_unbekannte() -> None:
    from scripts.bewerte_schadens_proben import aggregat

    zeilen = [
        {"rezept": "vinyl", "delta_defektlast": -0.10, "delta_musik_mos": 0.02, "besser": True},
        {"rezept": "vinyl", "delta_defektlast": 0.10, "delta_musik_mos": -0.02, "besser": False},
        {"rezept": "clipping", "delta_defektlast": 0.30, "delta_musik_mos": -0.10, "besser": False},
        {"rezept": "unbekannt", "delta_defektlast": 9.0, "delta_musik_mos": 9.0, "besser": True},
    ]
    agg = aggregat(zeilen)
    assert set(agg) == {"vinyl", "clipping"}
    assert agg["vinyl"]["n"] == 2
    assert agg["vinyl"]["besser"] == 1
    assert agg["vinyl"]["delta_defektlast"] == pytest.approx(0.0, abs=1e-9)


def test_main_fail_closed_ohne_manifest(tmp_path, capsys) -> None:
    """Fehlendes Manifest ⇒ Exit 2 mit klarer Meldung (kein stiller Leererfolg, §V6 (copilot-instructions.md))."""
    from scripts.bewerte_schadens_proben import main

    assert main(["--verzeichnis", str(tmp_path)]) == 2
    assert "FEHLT" in capsys.readouterr().out
