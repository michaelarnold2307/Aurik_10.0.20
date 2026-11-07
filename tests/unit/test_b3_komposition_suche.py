"""Vertragstests Muster B (scripts/b3_komposition_suche.py).

Modell-frei: die Messfunktion wird injiziert (§G5 (GEBOTE.md): keine
umgebungsabhaengigen Kipp-Tests). Zitierdisziplin AGENTS.md §5.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.b3_komposition_suche import (
    JND_TOLERANZ,
    KANDIDATEN,
    forward_auswahl,
    never_worsen_ok,
    suche,
    zielwert,
)

_SRC = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "b3_komposition_suche.py"
_SRC_TEXT = _SRC.read_text(encoding="utf-8")


def _messung(mos: float, last: float) -> dict:
    return {"musik_mos": mos, "defektlast": last}


def test_zielwert_ist_mos() -> None:
    assert zielwert(_messung(4.2, 0.9)) == 4.2


def test_never_worsen_jnd_schranke() -> None:
    base = _messung(4.0, 0.5)
    assert never_worsen_ok(_messung(4.0 - JND_TOLERANZ, 0.5), base) is True  # genau an der JND-Grenze
    assert never_worsen_ok(_messung(3.9, 0.5), base) is False  # darunter
    assert never_worsen_ok(_messung(4.5, 0.5 + JND_TOLERANZ), base) is True
    assert never_worsen_ok(_messung(4.5, 0.7), base) is False  # Defektlast-Schranke


def test_forward_auswahl_nimmt_nur_verbesserungen() -> None:
    scores = {
        frozenset(): _messung(4.0, 0.5),
        frozenset({"W-2"}): _messung(4.3, 0.5),
        frozenset({"W-2", "W-3"}): _messung(4.35, 0.5),
        frozenset({"W-2", "W-3", "W-8"}): _messung(4.5, 0.9),  # MOS-Gewinn, aber Defektlast-Blowup
    }
    beste, besucht = forward_auswahl(scores, ["W-2", "W-3", "W-8"], ["W-2", "W-3", "W-8"])
    assert beste == ["W-2", "W-3"]
    assert besucht == ["W-2", "W-2,W-3", "W-2,W-3,W-8"]


def test_forward_auswahl_deterministisch_bei_gleicher_reihenfolge() -> None:
    scores = {
        frozenset(): _messung(4.0, 0.5),
        frozenset({"W-2"}): _messung(4.3, 0.5),
        frozenset({"W-3"}): _messung(4.2, 0.5),
    }
    a = forward_auswahl(scores, ["W-2", "W-3"], ["W-2", "W-3"])
    b = forward_auswahl(scores, ["W-2", "W-3"], ["W-2", "W-3"])
    assert a == b


def test_suche_lazy_nimmt_nur_verbesserungen() -> None:
    tabelle = {
        frozenset(): _messung(4.0, 0.5),
        frozenset({"W-2"}): _messung(4.3, 0.5),
        frozenset({"W-3"}): _messung(3.9, 0.5),
        frozenset({"W-2", "W-3"}): _messung(4.4, 0.5),
    }

    def mess_fn(komp):
        return tabelle[komp]

    ergebnis = suche(mess_fn, ["W-2", "W-3"], ["W-2", "W-3"])
    assert ergebnis["beste"] == ["W-2", "W-3"]
    assert len(ergebnis["protokoll"]) == 2  # lazy: nur 2 Kandidat-Messungen


def test_suche_misst_baseline_zuerst() -> None:
    aufrufe = []

    def mess_fn(komp):
        aufrufe.append(komp)
        return _messung(4.0 if not komp else 4.5, 0.5)

    suche(mess_fn, ["W-2"], ["W-2"])
    assert aufrufe[0] == frozenset()  # Baseline zuerst — Never-worsen-Bezug


def test_suche_seed_reihenfolge_deterministisch() -> None:
    tabelle = {
        frozenset(): _messung(4.0, 0.5),
        frozenset({"W-2"}): _messung(4.4, 0.5),
        frozenset({"W-3"}): _messung(4.2, 0.5),
        frozenset({"W-2", "W-3"}): _messung(4.5, 0.5),
    }

    def mess_fn(komp):
        return tabelle[komp]

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    kandidaten = ["W-2", "W-3"]
    r1 = [kandidaten[i] for i in rng1.permutation(len(kandidaten))]
    r2 = [kandidaten[i] for i in rng2.permutation(len(kandidaten))]
    assert r1 == r2
    assert suche(mess_fn, kandidaten, r1)["beste"] == suche(mess_fn, kandidaten, r2)["beste"]


def test_kandidaten_nur_audio_wirksam() -> None:
    # Zeugen-/Release-Flags (W-10/W-11/W-12) duerfen nicht in der Suche sein:
    for w in ("W-10", "W-11", "W-12"):
        assert w not in KANDIDATEN
    assert set(KANDIDATEN) == {"W-2", "W-3", "W-8", "W-9", "W-13"}


def test_flag_restore_vertrag_im_quelltext() -> None:
    # KEIN dauerhafter Flip: try/finally stellt die Ausgangswerte wieder her.
    assert "aktive_urspruenglich" in _SRC_TEXT
    assert "try:" in _SRC_TEXT and "finally:" in _SRC_TEXT
    assert "_flag_setzen(s, wert)" in _SRC_TEXT
