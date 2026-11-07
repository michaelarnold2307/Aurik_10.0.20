"""Vertragstests Pfad-Wirksamkeits-Gate (Muster C, W-3-Lehre).

Bezieht sich auf `scripts/b3_pfad_wirksamkeit.py` (Gate-Funktionen) und die
Integration in `scripts/b3_stufenmessung.py` / `scripts/flag_freigabe.py`.
Modell-frei: alle Fälle laufen über synthetische Metadaten (§G5 (GEBOTE.md):
Tests duerfen nicht umgebungsabhaengig kippen). Zitierdisziplin AGENTS.md §5.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.b3_pfad_wirksamkeit import (
    fussabdruck,
    pfad_wirksam_ok,
    zeugnis,
)


def _zeile(**kw) -> dict:
    base = {
        "trial_id": "clip__w0",
        "flag_an_gelesen": True,
        "flag_aus_gelesen": False,
        "fussabdruck_rms": 1e-4,
        "zeuge_an": {"vorhanden": None, "wert": None, "aktiv": None},
        "zeuge_aus": {"vorhanden": None, "wert": None, "aktiv": None},
    }
    base.update(kw)
    return base


def test_fussabdruck_bit_identisch_und_layout_sicher() -> None:
    a = np.arange(480, dtype=np.float32) / 480.0
    assert fussabdruck(a, a) == 0.0
    # Layout-Unterschiede crashen nicht, sind aber ordnungs-sensitiv — die
    # Harness liefert BEIDE Seiten bereits als Mono (§_restore_fenster), daher
    # ist der Vergleich dort layout-frei.
    stereo_cf = np.stack([np.sin(np.arange(480) / 7.0), np.cos(np.arange(480) / 7.0)]).astype(np.float32)
    stereo_cl = stereo_cf.T
    assert np.isfinite(fussabdruck(stereo_cf, stereo_cl))


def test_fussabdruck_differenz_positiv() -> None:
    a = np.zeros(256, dtype=np.float32)
    b = a.copy()
    b[100] = 0.01
    assert fussabdruck(a, b) > 0.0


def test_fussabdruck_shape_mismatch_ist_inf() -> None:
    assert fussabdruck(np.zeros(10), np.zeros(9)) == float("inf")


def test_zeugnis_ohne_konfiguration_traegt_fussabdruck() -> None:
    z = zeugnis({"irgendwas": 1}, "W-3")
    assert z["vorhanden"] is None  # kein Zeuge konfiguriert — KEIN Fehlalarm


def test_zeugnis_w1_varianten() -> None:
    assert zeugnis({"hr_v1": {"attempted": True}}, "W-1")["aktiv"] is True
    assert zeugnis({"hr_v1": {"attempted": False, "reason": "centralized_g9"}}, "W-1")["aktiv"] is False
    z = zeugnis({}, "W-1")
    assert z["vorhanden"] is False and z["aktiv"] is False


def test_gate_noop_ist_ungueltig_w3_lehre() -> None:
    ok, report = pfad_wirksam_ok([_zeile(fussabdruck_rms=0.0), _zeile(trial_id="c__w1", fussabdruck_rms=0.0)])
    assert ok is False
    assert "fussabdruck_null" in " ".join(report["probleme"])


def test_gate_wirksam_ok_ohne_zeugen() -> None:
    # W-3 hat keinen konfigurierten Zeugen — ein einziger Fußabdruck > 0 genügt.
    ok, _ = pfad_wirksam_ok([_zeile(fussabdruck_rms=0.0), _zeile(trial_id="c__w1", fussabdruck_rms=1e-3)])
    assert ok is True


def test_gate_readback_mismatch_ungueltig() -> None:
    ok, report = pfad_wirksam_ok([_zeile(flag_an_gelesen=False)])
    assert ok is False
    assert "nicht als aktiv" in " ".join(report["probleme"])
    ok2, report2 = pfad_wirksam_ok([_zeile(flag_aus_gelesen=True)])
    assert ok2 is False
    assert "nicht als inaktiv" in " ".join(report2["probleme"])


def test_gate_w1_zeuge_fehlt_ungueltig() -> None:
    ok, report = pfad_wirksam_ok([_zeile(zeuge_an={"vorhanden": False, "wert": None, "aktiv": False})])
    assert ok is False
    assert "Modellzeuge auf AN-Seite fehlt" in " ".join(report["probleme"])


def test_gate_w1_zeuge_inaktiv_ungueltig() -> None:
    ok, report = pfad_wirksam_ok([_zeile(zeuge_an={"vorhanden": True, "wert": {"attempted": False}, "aktiv": False})])
    assert ok is False
    assert "inaktiv" in " ".join(report["probleme"])


def test_gate_w1_zeuge_aktiv_ok() -> None:
    ok, _ = pfad_wirksam_ok([_zeile(zeuge_an={"vorhanden": True, "wert": {"attempted": True}, "aktiv": True})])
    assert ok is True


# ── Integration: flag_freigabe lehnt unwirksame Messungen ab ─────────────────
def _beschluss(messung_pfad: str) -> dict:
    return {
        "schema": "aurik.objektivbeschluss/1",
        "flag": "use_test_flag",
        "referenz_songs": 3,
        "never_worsen": True,
        "signifikant": True,
        "defekt_fallbacks": 0,
        "blind_ausgewertet": True,
        "messung": messung_pfad,
        "zielmetrik": {"name": "musik_mos (G-2)"},
    }


def test_flag_freigabe_lehnt_ungueltige_messung_ab(tmp_path) -> None:
    from scripts.flag_freigabe import objektivbeschluss_ok

    messung = tmp_path / "b3_stufenmessung_W-X.json"
    messung.write_text(
        json.dumps({"pfad_wirksam": False, "messung_ungueltig": True, "wirksam_probleme": ["fussabdruck_null"]}),
        encoding="utf-8",
    )
    ok, grund = objektivbeschluss_ok(_beschluss(str(messung)), "use_test_flag")
    assert ok is False
    assert "W-3-Lehre" in grund


def test_flag_freigabe_akzeptiert_wirksame_messung(tmp_path) -> None:
    from scripts.flag_freigabe import objektivbeschluss_ok

    messung = tmp_path / "b3_stufenmessung_W-Y.json"
    messung.write_text(json.dumps({"pfad_wirksam": True, "fussabdruck_max_rms": 0.01}), encoding="utf-8")
    ok, grund = objektivbeschluss_ok(_beschluss(str(messung)), "use_test_flag")
    assert ok is True


def test_flag_freigabe_altformat_akzeptiert_mit_warnung(tmp_path) -> None:
    from scripts.flag_freigabe import objektivbeschluss_ok

    messung = tmp_path / "b3_stufenmessung_W-Z.json"
    messung.write_text(json.dumps({"stufe": "W-Z", "erfuellt": True}), encoding="utf-8")
    ok, _ = objektivbeschluss_ok(_beschluss(str(messung)), "use_test_flag")
    assert ok is True  # Altformat bleibt akzeptiert; Warnung erfolgt (§V6 (copilot-instructions.md))
