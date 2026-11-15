"""Vertragstests für `scripts/flag_wirkung_gate.py` (D-K3-92).

Das Gate schließt die Klasse „Schalter ohne Wirkung" (D-K3-85/D-K3-90/D-K3-91):
Ein `use_*`-Flag, den kein Produktionsmodul liest, ist ein stiller No-Op — und
eine B3-Messung darauf wäre ein No-Op-Artefakt statt eines Qualitätsurteils.

Die Tests bauen jeweils ein Mini-Repo in `tmp_path`; nur der letzte Test prüft
das echte Repo (die Invariante selbst).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_WURZEL = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("flag_wirkung_gate", _WURZEL / "scripts" / "flag_wirkung_gate.py")
assert _SPEC is not None and _SPEC.loader is not None
fwg = importlib.util.module_from_spec(_SPEC)
sys.modules["flag_wirkung_gate"] = fwg
_SPEC.loader.exec_module(fwg)


def _repo(tmp_path: Path, *, flags: str, dateien: dict[str, str]) -> Path:
    """Baut ein Mini-Repo mit Flag-Datei und den angegebenen Quelldateien."""
    flag_datei = tmp_path / "backend" / "core" / "music_model_flags.py"
    flag_datei.parent.mkdir(parents=True, exist_ok=True)
    flag_datei.write_text(flags, encoding="utf-8")
    for relativ, inhalt in dateien.items():
        ziel = tmp_path / relativ
        ziel.parent.mkdir(parents=True, exist_ok=True)
        ziel.write_text(inhalt, encoding="utf-8")
    return tmp_path


def _status(wurzel: Path, *, ausnahmen: dict[str, str] | None = None) -> list[str]:
    befunde = fwg.pruefe(wurzel, ausnahmen=ausnahmen or {})
    return fwg.verstoesse(befunde, kandidaten=fwg._mess_kandidaten(wurzel))


FLAGS = "use_alpha: bool = False\nuse_beta: bool = False\n"


def test_definiere_flags_erkennt_annassign_und_assign(tmp_path: Path) -> None:
    wurzel = _repo(
        tmp_path,
        flags=("use_alpha: bool = False\nuse_beta = False\nkein_flag: bool = False\nANDERE = 1\n"),
        dateien={},
    )
    assert fwg.definiere_flags(wurzel / fwg.FLAG_DATEI) == ["use_alpha", "use_beta"]


def test_direkter_import_und_name_zaehlt(tmp_path: Path) -> None:
    wurzel = _repo(
        tmp_path,
        flags=FLAGS,
        dateien={
            "backend/core/konsument.py": "from backend.core.music_model_flags import use_alpha\n\nif use_alpha:\n    pass\n",
            "backend/core/konsument2.py": "import backend.core.music_model_flags as mmf\n\nif mmf.use_beta:\n    pass\n",
        },
    )
    assert _status(wurzel) == []


def test_toter_import_zaehlt_nicht(tmp_path: Path) -> None:
    """`import ... as` ohne Verwendung ist keine Lesestelle (D-K3-91-Muster)."""
    wurzel = _repo(
        tmp_path,
        flags=FLAGS,
        dateien={
            "backend/core/konsument.py": (
                "from backend.core.music_model_flags import use_alpha as unbenutzt\n"
                "from backend.core.music_model_flags import use_beta as genutzt\n\n"
                "if genutzt:\n    pass\n"
            ),
        },
    )
    meldungen = _status(wurzel)
    assert len(meldungen) == 1
    assert "use_alpha" in meldungen[0]


def test_getattr_string_zaehlt_als_lesestelle(tmp_path: Path) -> None:
    """Die häufigste Lesestelle im Repo ist `getattr(modul, "use_x", default)`."""
    wurzel = _repo(
        tmp_path,
        flags=FLAGS,
        dateien={
            "backend/core/konsument.py": (
                "import backend.core.music_model_flags as mmf\n\n"
                'def _an() -> bool:\n    return bool(getattr(mmf, "use_alpha", False))\n'
                'def _an2() -> bool:\n    return bool(getattr(mmf, "use_beta", False))\n'
            ),
        },
    )
    assert _status(wurzel) == []


def test_setattr_string_zaehlt_nicht(tmp_path: Path) -> None:
    """Ein Schreibzugriff schaltet nichts — er ist keine Lesestelle."""
    wurzel = _repo(
        tmp_path,
        flags=FLAGS,
        dateien={
            "backend/core/konsument.py": (
                "import backend.core.music_model_flags as mmf\n\n"
                'def _setze() -> None:\n    setattr(mmf, "use_alpha", True)\n    setattr(mmf, "use_beta", True)\n'
            ),
        },
    )
    assert len(_status(wurzel)) == 2


def test_tests_zaehlen_nicht_als_lesestelle(tmp_path: Path) -> None:
    wurzel = _repo(
        tmp_path,
        flags=FLAGS,
        dateien={
            "tests/unit/test_x.py": (
                "from backend.core.music_model_flags import use_alpha, use_beta\n\n"
                "def test_x():\n    assert use_alpha is False and use_beta is False\n"
            ),
        },
    )
    assert len(_status(wurzel)) == 2


def test_nur_mess_skript_ist_verstoss(tmp_path: Path) -> None:
    """Ein nur in `scripts/` gelesener Flag ist im Betrieb wirkungslos."""
    wurzel = _repo(
        tmp_path,
        flags=FLAGS,
        dateien={
            "scripts/messlauf.py": "from backend.core.music_model_flags import use_alpha, use_beta\n\nprint(use_alpha, use_beta)\n",
        },
    )
    meldungen = _status(wurzel)
    assert len(meldungen) == 2
    assert all("Produktionspfad wirkungslos" in m for m in meldungen)


def test_ausnahme_deckt_flag_ab(tmp_path: Path) -> None:
    wurzel = _repo(
        tmp_path,
        flags=FLAGS,
        dateien={
            "scripts/messlauf.py": "from backend.core.music_model_flags import use_alpha, use_beta\n\nprint(use_alpha, use_beta)\n",
        },
    )
    meldungen = _status(wurzel, ausnahmen={"use_alpha": "D-K3-092: Testausnahme", "use_beta": "D-K3-092: Testausnahme"})
    assert meldungen == []


def test_b3_kandidat_ohne_produktionswirkung_ist_verstoss(tmp_path: Path) -> None:
    """Stufe C: Ein Mess-Kandidat ohne Lesestelle wäre ein No-Op-Artefakt."""
    wurzel = _repo(
        tmp_path,
        flags=FLAGS,
        dateien={
            "scripts/b3_stufenmessung.py": 'KANDIDATEN = {"W-1": "use_alpha"}\n',
        },
    )
    meldungen = _status(wurzel, ausnahmen={"use_beta": "D-K3-092: Testausnahme"})
    assert len(meldungen) == 2
    assert any("use_alpha" in m and "No-Op-Artefakt" in m for m in meldungen)


_DISPATCH_FLAGS = (
    "use_alpha: bool = False\n"
    "use_beta: bool = False\n"
    "\n"
    "def auswahl(key: str) -> bool:\n"
    '    flag_map = {"a": use_alpha, "b": use_beta}\n'
    "    return flag_map.get(key, False)\n"
)


def test_dispatch_tabelle_zaehlt_als_indirekte_lesestelle(tmp_path: Path) -> None:
    """D-K3-92: Ein Flag, das nur über eine Dispatch-Tabelle gelesen wird, wirkt im
    Aufrufer der Funktion — hier ein Produktions-Plugin mit Literal-Schlüssel."""
    wurzel = _repo(
        tmp_path,
        flags=_DISPATCH_FLAGS,
        dateien={
            "backend/core/plugin.py": (
                "from backend.core.music_model_flags import auswahl as _waehle\n\n"
                'def lade() -> bool:\n    return _waehle("a")\n'
            ),
        },
    )
    meldungen = _status(wurzel)
    assert len(meldungen) == 1, meldungen
    assert "use_beta" in meldungen[0]


def test_dispatch_schluessel_filtert_fremde_flags(tmp_path: Path) -> None:
    """Nur der Schlüssel, der auf den Flag zeigt, zählt (kein Pauschal-Treffer
    für die ganze Tabelle)."""
    wurzel = _repo(
        tmp_path,
        flags=_DISPATCH_FLAGS,
        dateien={
            "backend/core/plugin.py": (
                "from backend.core.music_model_flags import auswahl as _waehle\n\n"
                'def lade_a() -> bool:\n    return _waehle("a")\n'
                'def lade_b() -> bool:\n    return _waehle("b")\n'
            ),
        },
    )
    assert _status(wurzel) == []


def test_dispatch_dynamischer_schluessel_zaehlt_konservativ(tmp_path: Path) -> None:
    """Ein nicht auflösbarer Schlüssel darf keine Lesestelle verschlucken."""
    wurzel = _repo(
        tmp_path,
        flags=_DISPATCH_FLAGS,
        dateien={
            "backend/core/plugin.py": (
                "from backend.core.music_model_flags import auswahl as _waehle\n\n"
                "def lade(key: str) -> bool:\n    return _waehle(key)\n"
            ),
        },
    )
    assert _status(wurzel) == []


def test_dispatch_ohne_produktionsaufruf_ist_verstoss(tmp_path: Path) -> None:
    """Die Tabelle allein ist keine Lesestelle — der Aufrufer entscheidet."""
    wurzel = _repo(
        tmp_path,
        flags=_DISPATCH_FLAGS,
        dateien={
            "scripts/messlauf.py": (
                'from backend.core.music_model_flags import auswahl as _waehle\n\nprint(_waehle("a"), _waehle("b"))\n'
            )
        },
    )
    meldungen = _status(wurzel)
    assert len(meldungen) == 2, meldungen
    assert all("Produktionspfad wirkungslos" in m for m in meldungen)


def test_dispatch_ohne_rueckgabe_zaehlt_nicht(tmp_path: Path) -> None:
    """Ohne Rückgabe verlässt der Flag-Wert das Modul nicht."""
    flags = (
        "use_alpha: bool = False\n"
        "use_beta: bool = False\n"
        "\n"
        "def anzeigen(key: str) -> None:\n"
        '    flag_map = {"a": use_alpha, "b": use_beta}\n'
        "    print(flag_map.get(key, False))\n"
    )
    wurzel = _repo(
        tmp_path,
        flags=flags,
        dateien={
            "backend/core/plugin.py": (
                "from backend.core.music_model_flags import anzeigen as _zeige\n\n"
                'def lade() -> None:\n    _zeige("a")\n'
            ),
        },
    )
    assert len(_status(wurzel)) == 2


def test_echtes_repo_ist_gruen() -> None:
    """Die Invariante gilt für den echten Bestand — ohne Ausnahmen (D-K3-92)."""
    befunde = fwg.pruefe(_WURZEL)
    meldungen = fwg.verstoesse(befunde, kandidaten=fwg._mess_kandidaten(_WURZEL))
    assert meldungen == [], "\n".join(meldungen)
    assert len(befunde) >= 20
    ausnahmen = {b.flag for b in befunde if b.ausnahme}
    assert ausnahmen == set(), f"Unerwartete Ausnahmen: {ausnahmen}"
    bw = next(b for b in befunde if b.flag == "use_bw_v5")
    assert any("plugins/bw_reconstructor_plugin.py" in stelle for stelle in bw.produktionsstellen)
    assert any("resolve_model_path" in stelle for stelle in bw.produktionsstellen)


@pytest.mark.parametrize("arg", ["--ci", "--json"])
def test_cli_modi_sind_erfolgreich(arg: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    wurzel = _repo(
        tmp_path,
        flags=FLAGS,
        dateien={
            "backend/core/konsument.py": (
                "from backend.core.music_model_flags import use_alpha, use_beta\n\n"
                "if use_alpha and use_beta:\n    pass\n"
            ),
        },
    )
    assert fwg.main([arg, "--wurzel", str(wurzel)]) == 0
    ausgabe = capsys.readouterr().out
    assert ausgabe.strip()
