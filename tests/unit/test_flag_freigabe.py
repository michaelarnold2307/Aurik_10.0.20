"""Tests für die Flag-Freigabe-Kette (Pfeiler 1: Verdikt → Flag-Flip → Commit)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.flag_freigabe import (
    buche_register,
    finde_register_zeile,
    flippe_flag,
    gleiche_tests_an,
    verdikt_ok,
)

_REG = (
    "# Register\n"
    "| ID | Artefakt | Klasse | Domäne | Evidenz | Status | Blocker | Beleg |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    "| D-K4-1 | `x.py` | K4 | Musik | 4/4 | in-arbeit | Flag | use_aspade_declipper offen |\n"
    "| D-K4-2 | `y.py` | K4 | Musik | 4/4 | in-arbeit | Flag | use_tf_gridnet_quality offen |\n"
)


def test_verdikt_ok_fit_json() -> None:
    ok, grund = verdikt_ok(
        {
            "verdict": "HOERBAR_BELEGT",
            "pooled": {"above_chance_ci95": True, "trials_sufficient": True, "trials": 24},
        }
    )
    assert ok and "HOERBAR_BELEGT" in grund
    ok2, g2 = verdikt_ok({"verdict": "ZU_WENIGE_TRIALS", "pooled": {"trials_sufficient": False}})
    assert not ok2 and "kein Beleg" in g2
    ok3, _ = verdikt_ok(
        {
            "verdict": "HOERBAR_BELEGT",
            "pooled": {"above_chance_ci95": False, "trials_sufficient": True},
        }
    )
    assert not ok3, "CI unter Zufallspegel ist kein Beleg"


def test_verdikt_ok_ab_stats_und_unbekannt() -> None:
    ok, grund = verdikt_ok(
        {
            "schema": "aurik.ab_stats/1",
            "signifikant": True,
            "siege_a": 15,
            "siege_b": 5,
            "p_wert_vorzeichen_test": 0.041389,
        },
        kandidat_seite="A",
    )
    assert ok and "signifikant" in grund
    ok2, _ = verdikt_ok(
        {"schema": "aurik.ab_stats/1", "signifikant": False, "siege_a": 6, "siege_b": 4}, kandidat_seite="A"
    )
    assert not ok2
    ok3, g3 = verdikt_ok({"irgendwas": 1})
    assert not ok3 and "unbekannt" in g3


def test_ab_stats_1_ohne_richtung_ist_nicht_buchbar() -> None:
    """Befund 2026-11-22: /1 wurde richtungsblind akzeptiert (siege_a=15 > siege_b=5 reichte)."""
    daten = {"schema": "aurik.ab_stats/1", "signifikant": True, "siege_a": 15, "siege_b": 5}
    ok, grund = verdikt_ok(daten)
    assert not ok and "Richtung" in grund


def test_ab_stats_1_votum_fuer_den_bestand_ist_nie_buchbar() -> None:
    daten = {"schema": "aurik.ab_stats/1", "signifikant": True, "siege_a": 15, "siege_b": 5}
    ok, grund = verdikt_ok(daten, kandidat_seite="B")  # B = Kandidat hat nur 5 von 20
    assert not ok and "kein Vorteil des Kandidaten" in grund
    ok_a, _ = verdikt_ok(daten, kandidat_seite="A")
    assert ok_a


def test_ab_stats_1_seitenangabe_im_beleg_und_widerspruch() -> None:
    mit_seite = {"schema": "aurik.ab_stats/1", "signifikant": True, "siege_a": 3, "siege_b": 12, "kandidat_seite": "B"}
    assert verdikt_ok(mit_seite)[0]
    ok, grund = verdikt_ok(mit_seite, kandidat_seite="A")
    assert not ok and "widerspricht" in grund


def test_ab_stats_2_traegt_die_richtung_explizit() -> None:
    basis = {"schema": "aurik.ab_stats/2", "signifikant": True, "p_wert_vorzeichen_test": 0.03}
    assert verdikt_ok({**basis, "siege_kandidat": 6, "siege_bestand": 0})[0]
    for kandidat, bestand in ((0, 6), (3, 3)):
        ok, grund = verdikt_ok({**basis, "siege_kandidat": kandidat, "siege_bestand": bestand})
        assert not ok and "Kandidat" in grund
    assert not verdikt_ok({**basis, "siege_kandidat": None, "siege_bestand": 0})[0]


@pytest.mark.parametrize(
    ("feld", "wert", "erwartet"),
    [("simuliert", True, "simuliert"), ("gueltig", False, "Validität"), ("vollstaendig", False, "unvollständig")],
)
def test_ab_stats_2_simuliert_ungueltig_unvollstaendig_nie_buchbar(feld: str, wert: object, erwartet: str) -> None:
    daten = {
        "schema": "aurik.ab_stats/2",
        "signifikant": True,
        "siege_kandidat": 6,
        "siege_bestand": 0,
        feld: wert,
    }
    ok, grund = verdikt_ok(daten)
    assert not ok and erwartet in grund


def test_register_buchung_behaelt_8_zellen() -> None:
    ziel = finde_register_zeile(_REG, "use_aspade_declipper")
    assert ziel is not None and "D-K4-1" in ziel[1]
    assert finde_register_zeile(_REG, "use_unbekannt") is None
    neu = buche_register(_REG, "use_aspade_declipper", Path("docs/a.json"), Path("docs/b.md"))
    zeilen = [z for z in neu.splitlines() if z.startswith("| D-K4-1")]
    assert len(zeilen) == 1
    cells = zeilen[0].strip().strip("|").split("|")
    assert len(cells) == 8, "8 Zellen bleiben 8 Zellen"
    assert cells[5].strip() == "geschlossen"
    assert "docs/a.json" in cells[7] and "docs/b.md" in cells[7]
    assert "use_tf_gridnet_quality offen" in neu, "andere Zeilen bleiben unberührt"


def test_register_buchung_ambiguitaet_abgelehnt() -> None:
    doppelt = _REG.replace("| in-arbeit |", "| offen |")
    assert finde_register_zeile(doppelt, "use_aspade_declipper") is not None
    try:
        buche_register(
            doppelt + "| D-K4-9 | `z.py` | K4 | Musik | 1/1 | in-arbeit | F | use_aspade_declipper x |",
            "use_aspade_declipper",
            Path("a"),
            Path("b"),
        )
    except ValueError as exc:
        assert "eindeutige" in str(exc)
    else:
        raise AssertionError("Doppelbelegung muß abgelehnt werden")


def test_flag_flip_und_test_angleichung() -> None:
    flags = "use_x: bool = False  # Alt-Kommentar\nuse_y: bool = True  # bleibt\n"
    neu = flippe_flag(flags, "use_x", "fit.json")
    assert "use_x: bool = True" in neu
    assert "Alt-Kommentar" in neu and "AKTIV nach Hör-Sign-off" in neu and "fit.json" in neu
    assert "use_y: bool = True" in neu
    try:
        flippe_flag(neu, "use_x", "fit.json")
    except ValueError as exc:
        assert "nicht gefunden" in str(exc)
    else:
        raise AssertionError("zweiter Flip muß abgelehnt werden")
    t = "assert flags.use_x is False, 'gated'\nassert mmf.use_x is False\nassert other.use_z is False\n"
    nt, n = gleiche_tests_an(t, "use_x")
    assert n == 2
    assert "flags.use_x is True" in nt and "mmf.use_x is True" in nt
    assert "other.use_z is False" in nt, "fremde Flags bleiben unangetastet"


def test_objektivbeschluss_validierung(tmp_path: Path) -> None:
    from scripts.flag_freigabe import objektivbeschluss_ok

    messung = tmp_path / "messung.json"
    messung.write_text("{}", encoding="utf-8")
    gueltig = {
        "schema": "aurik.objektivbeschluss/1",
        "flag": "use_x",
        "referenz_songs": 3,
        "never_worsen": True,
        "signifikant": True,
        "defekt_fallbacks": 0,
        "blind_ausgewertet": True,
        "messung": str(messung),
        "zielmetrik": {"name": "G-2"},
    }
    ok, grund = objektivbeschluss_ok(gueltig, "use_x")
    assert ok and "G-2" in grund and "Vorbehalt" in grund

    for feld, wert in (
        ("referenz_songs", 2),
        ("never_worsen", False),
        ("signifikant", False),
        ("defekt_fallbacks", 1),
        ("blind_ausgewertet", False),
        ("schema", "falsch"),
        ("flag", "use_y"),
    ):
        kaputt = dict(gueltig)
        kaputt[feld] = wert
        ok2, grund2 = objektivbeschluss_ok(kaputt, "use_x")
        assert not ok2, f"{feld}={wert} muss abgelehnt werden"
        assert grund2

    ohne = dict(gueltig)
    ohne["messung"] = str(tmp_path / "fehlt.json")
    ok3, grund3 = objektivbeschluss_ok(ohne, "use_x")
    assert not ok3 and "fehlt" in grund3, "Provenienz-Pflicht"


def test_buchung_mit_objektivbeschluss_text() -> None:
    from scripts.flag_freigabe import buche_register, flippe_flag

    zeile = "| D-K4-1 | `x.py` | K4 | Musik | 4/4 | in-arbeit | F | use_x |"
    reg = (
        "# R\n| ID | Artefakt | Klasse | Domäne | Evidenz | Status | Blocker | Beleg |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        + zeile
        + "\n"
    )
    neu = buche_register(
        reg, "use_x", Path("docs/a.json"), Path("docs/b.json"), verfahren="Objektivbeschluss B3, Hör-Vorbehalt"
    )
    assert "Objektivbeschluss B3" in neu and "Hör-Vorbehalt" in neu
    flags = "use_x: bool = False  # Alt\n"
    geflippt = flippe_flag(flags, "use_x", "fit.json", verfahren="Objektivbeschluss B3, Hör-Vorbehalt")
    assert "AKTIV nach Objektivbeschluss B3" in geflippt
