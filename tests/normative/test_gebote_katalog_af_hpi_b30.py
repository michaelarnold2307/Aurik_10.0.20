"""Pins der B30-Familie des Artifact-Freedom-/HPI-Gates (§v10.704, GEBOTE.md, Kategorie XXIII).

Ein Gebot dieser Familie wird nur gepinnt, wenn der Produktionscode seinen Wortlaut an ALLEN
Erzeugern des artifact_freedom-Mindestwerts wörtlich erfüllt. Gepinnt ist hier:

- §G163 (GEBOTE.md, Kategorie XXIII) — Floor-Absolut-Garantie: artifact_freedom-Mindestwert NIEMALS < 0.32.

Die übrigen Gebote der Familie werden erst gepinnt, wenn HPI-Gate und Konstitution die B30-Formel der
Spec §v10.704 tatsächlich tragen (heute rechnen sie noch mit der Formel aus §v10.120).

Methode (kein Mock, kein Textvorkommen-Test):

- Verhalten der echten Funktionen: ``HolisticPerceptualGate._get_depth_adaptive_af_min``,
  ``SpecConstitution.is_export_blocked`` / ``check_paragraph_zero``, ``CalibrationContext.artifact_freedom_min``,
  ``CalibratedConstants.artifact_freedom_min``.
- Die beiden Export-Gates (ArtifactFreedomGate, FinalExportAudioGate) tragen die B30-Formel INLINE in einer
  Methode von ``unified_restorer_v3.py`` (Import ≈ 20 s, Formel nicht als Funktion aufrufbar). Dort werden die
  Zuweisungen ``*af_min = …`` per AST gefunden, samt Eingangs-Anweisungen herausgeschnitten und mit einem kleinen
  AST-Auswerter über ein Extremwert-Gitter ausgewertet (Laufzeitwerte der echten Quelle, keine Kopie der Formel).
- Selbsttest: derselbe Prüfer muss synthetische Verstöße (fehlender Boden, falscher Boden, NaN-unsichere
  Argumentreihenfolge) finden, sonst wäre der Pin wirkungslos.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import inspect
import math
import operator
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_UV3_PFAD = ROOT / "backend" / "core" / "unified_restorer_v3.py"

# Spec §v10.704, Invarianten-Tabelle: „AF-Minimum absolut | 0.32“.
_AF_BODEN = 0.32

# Extremwert-Gitter: Kettenlängen (0 = leere Kette) und Restorability-Werte inkl. <0, >100, ±inf, NaN.
_KETTENLAENGEN = tuple(range(0, 13))
_TIEFEN = (-3, *range(0, 13))
_RS_GITTER = (
    -1e9,
    -50.0,
    -1e-9,
    0.0,
    33.0,
    34.9,
    49.9,
    66.0,
    100.0,
    100.1,
    250.0,
    1e9,
    math.inf,
    -math.inf,
    math.nan,
)


# ── Kleiner AST-Auswerter (nur die Teilsprache der B30-Formel) ───────────────────────────────────────


class _AuswertungsError(Exception):
    """Der extrahierte Code nutzt einen AST-Knoten außerhalb der unterstützten Teilsprache."""


_BINAER: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_VERGLEICH: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
# Echte Python-Builtins: max/min/float verhalten sich bei NaN/±inf exakt wie im Produktionscode.
_FUNKTIONEN: dict[str, Callable[..., Any]] = {"max": max, "min": min, "len": len, "float": float, "int": int}


def _auswerten(knoten: ast.AST, umgebung: dict[str, Any]) -> Any:
    """Wertet einen Ausdrucks-Knoten mit echter Python-Semantik aus."""
    if isinstance(knoten, ast.Constant):
        return knoten.value
    if isinstance(knoten, ast.Name):
        return umgebung[knoten.id]
    if isinstance(knoten, ast.List):
        return [_auswerten(element, umgebung) for element in knoten.elts]
    if isinstance(knoten, ast.UnaryOp) and isinstance(knoten.op, (ast.USub, ast.Not)):
        operand = _auswerten(knoten.operand, umgebung)
        return -operand if isinstance(knoten.op, ast.USub) else not operand
    if isinstance(knoten, ast.BinOp) and type(knoten.op) in _BINAER:
        return _BINAER[type(knoten.op)](_auswerten(knoten.left, umgebung), _auswerten(knoten.right, umgebung))
    if isinstance(knoten, ast.Compare) and all(type(op) in _VERGLEICH for op in knoten.ops):
        links = _auswerten(knoten.left, umgebung)
        for op, rechter_knoten in zip(knoten.ops, knoten.comparators):
            rechts = _auswerten(rechter_knoten, umgebung)
            if not _VERGLEICH[type(op)](links, rechts):
                return False
            links = rechts
        return True
    if isinstance(knoten, ast.BoolOp):
        wert: Any = None
        for teil in knoten.values:
            wert = _auswerten(teil, umgebung)
            if isinstance(knoten.op, ast.And) and not wert:
                return wert
            if isinstance(knoten.op, ast.Or) and wert:
                return wert
        return wert
    if isinstance(knoten, ast.IfExp):
        return _auswerten(knoten.body if _auswerten(knoten.test, umgebung) else knoten.orelse, umgebung)
    if (
        isinstance(knoten, ast.Call)
        and isinstance(knoten.func, ast.Name)
        and knoten.func.id in _FUNKTIONEN
        and not knoten.keywords
    ):
        return _FUNKTIONEN[knoten.func.id](*(_auswerten(arg, umgebung) for arg in knoten.args))
    raise _AuswertungsError(f"nicht unterstützter AST-Knoten: {ast.dump(knoten)[:140]}")


def _ausfuehren(anweisungen: Sequence[ast.stmt], umgebung: dict[str, Any]) -> None:
    """Führt Zuweisungen und if/elif/else-Ketten aus (mehr kennt die B30-Formel nicht)."""
    for anweisung in anweisungen:
        if isinstance(anweisung, ast.Assign) and len(anweisung.targets) == 1:
            ziel = anweisung.targets[0]
            if isinstance(ziel, ast.Name):
                umgebung[ziel.id] = _auswerten(anweisung.value, umgebung)
                continue
        if isinstance(anweisung, ast.If):
            _ausfuehren(anweisung.body if _auswerten(anweisung.test, umgebung) else anweisung.orelse, umgebung)
            continue
        raise _AuswertungsError(f"nicht unterstützte Anweisung: {ast.dump(anweisung)[:140]}")


# ── Fundstellen „*af_min = …“ samt Rückwärts-Slice ───────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class _AfMinFundstelle:
    """Eine Zuweisung ``*af_min = …`` mit den Anweisungen, von denen ihr Wert abhängt."""

    name: str
    zeile: int
    anweisungen: tuple[ast.stmt, ...]
    eingaben: frozenset[str]


def _geladen(knoten: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(knoten) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - set(
        _FUNKTIONEN
    )


def _gespeichert(knoten: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(knoten) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def _ist_eingabe(name: str) -> bool:
    """Konvention der Pipeline: Transfer-Kette und Restorability-Score sind externe Eingaben."""
    klein = name.lower()
    return "chain" in klein or "restorab" in klein


def _definiert_immer(anweisung: ast.stmt, name: str) -> bool:
    """True, wenn die Anweisung ``name`` auf JEDEM Pfad zuweist (eine if-Kette braucht dafür ein else)."""
    if isinstance(anweisung, ast.Assign):
        return any(isinstance(ziel, ast.Name) and ziel.id == name for ziel in anweisung.targets)
    if isinstance(anweisung, ast.If):
        return (
            bool(anweisung.orelse)
            and any(_definiert_immer(s, name) for s in anweisung.body)
            and any(_definiert_immer(s, name) for s in anweisung.orelse)
        )
    return False


def _fundstelle(block: list[ast.stmt], index: int) -> _AfMinFundstelle:
    """Schneidet die Anweisungen heraus, von denen ``block[index]`` abhängt (letzte Definition je Name)."""
    ziel = block[index]
    assert isinstance(ziel, ast.Assign) and isinstance(ziel.targets[0], ast.Name)
    benoetigt = _geladen(ziel.value)
    eingaben = {n for n in benoetigt if _ist_eingabe(n)}
    offen = benoetigt - eingaben
    gewaehlt: list[ast.stmt] = []
    for anweisung in reversed(block[:index]):
        treffer = _gespeichert(anweisung) & offen
        if not treffer:
            continue
        gewaehlt.append(anweisung)
        neu = _geladen(anweisung)
        eingaben |= {n for n in neu if _ist_eingabe(n)}
        offen = (offen - {n for n in treffer if _definiert_immer(anweisung, n)}) | {
            n for n in neu if not _ist_eingabe(n)
        }
    if offen:
        raise _AuswertungsError(
            f"{ziel.targets[0].id} (Zeile {ziel.lineno}): Eingang(e) {sorted(offen)} ohne Definition im selben Block"
        )
    gewaehlt.reverse()
    return _AfMinFundstelle(
        name=ziel.targets[0].id,
        zeile=ziel.lineno,
        anweisungen=(*gewaehlt, ziel),
        eingaben=frozenset(eingaben),
    )


def _af_min_fundstellen(baum: ast.AST) -> Iterator[_AfMinFundstelle]:
    """Findet alle Zuweisungen an einen Namen, der auf ``af_min`` endet (mit ihrem umgebenden Block)."""
    for knoten in ast.walk(baum):
        for feld in ("body", "orelse", "finalbody"):
            block = getattr(knoten, feld, None)
            if not isinstance(block, list):
                continue
            for index, anweisung in enumerate(block):
                if (
                    isinstance(anweisung, ast.Assign)
                    and len(anweisung.targets) == 1
                    and isinstance(anweisung.targets[0], ast.Name)
                    and anweisung.targets[0].id.endswith("af_min")
                ):
                    yield _fundstelle(block, index)


@functools.lru_cache(maxsize=1)
def _uv3_export_gate_fundstellen() -> tuple[_AfMinFundstelle, ...]:
    """Parst unified_restorer_v3.py einmal; der ~260k-Knoten-Baum wird danach freigegeben."""
    baum = ast.parse(_UV3_PFAD.read_text(encoding="utf-8"), filename=str(_UV3_PFAD))
    return tuple(_af_min_fundstellen(baum))


def _wert(fundstelle: _AfMinFundstelle, kettenlaenge: int, rs: float) -> float:
    """Wertet die Fundstelle für Kette der Länge ``kettenlaenge`` und Restorability ``rs`` aus."""
    umgebung: dict[str, Any] = {
        name: ["stufe"] * kettenlaenge if "chain" in name.lower() else rs for name in fundstelle.eingaben
    }
    _ausfuehren(fundstelle.anweisungen, umgebung)
    return float(umgebung[fundstelle.name])


@dataclasses.dataclass(frozen=True)
class _Befund:
    verstoesse: list[str]
    tiefster_wert: float
    eingaben_wirken: bool


def _pruefe_boden(fundstelle: _AfMinFundstelle) -> _Befund:
    """Wertet die Fundstelle über das Extremwert-Gitter aus und sammelt Unterschreitungen des Bodens."""
    verstoesse: list[str] = []
    tiefster = math.inf
    for kettenlaenge in _KETTENLAENGEN:
        for rs in _RS_GITTER:
            wert = _wert(fundstelle, kettenlaenge, rs)
            if not wert >= _AF_BODEN:  # NaN-sicher: NaN zählt als Unterschreitung
                verstoesse.append(
                    f"{fundstelle.name} (Zeile {fundstelle.zeile}): Kette={kettenlaenge} rs={rs!r} → {wert!r}"
                )
            else:
                tiefster = min(tiefster, wert)
    # Extraktions-Integrität: Der Slice muss auf Kettenlänge UND Restorability reagieren, sonst wäre er entartet.
    wirken = _wert(fundstelle, 1, 100.0) != _wert(fundstelle, 5, 100.0) and _wert(fundstelle, 1, 0.0) != _wert(
        fundstelle, 1, 100.0
    )
    return _Befund(verstoesse=verstoesse, tiefster_wert=tiefster, eingaben_wirken=wirken)


# ── Verhaltens-Prüfungen an den echten Funktionen ────────────────────────────────────────────────────


def _hpi_gate_verstoesse() -> list[str]:
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    funktion = HolisticPerceptualGate._get_depth_adaptive_af_min
    nimmt_rs = "restorability_score" in inspect.signature(funktion).parameters
    ketten: list[list[str] | None] = [None, *(["stufe"] * n for n in _KETTENLAENGEN)]
    verstoesse: list[str] = []
    for kette in ketten:
        for rs in _RS_GITTER if nimmt_rs else (None,):
            wert = funktion(kette, restorability_score=rs) if nimmt_rs else funktion(kette)
            if not wert >= _AF_BODEN:
                verstoesse.append(
                    f"HolisticPerceptualGate: Kette={None if kette is None else len(kette)} rs={rs!r} → {wert!r}"
                )
    return verstoesse


def _konstitution_verstoesse() -> list[str]:
    """Ein AF knapp UNTER dem Boden muss die Konstitution bei JEDER Tiefe/Restorability sperren."""
    from backend.core.spec_constitution import get_constitution

    konstitution = get_constitution()
    knapp_unter_boden = math.nextafter(_AF_BODEN, 0.0)
    audio = np.zeros(48, dtype=np.float32)
    verstoesse: list[str] = []
    for tiefe in _TIEFEN:
        for rs in _RS_GITTER:
            gesperrt, grund = konstitution.is_export_blocked(
                knapp_unter_boden, 0.5, chain_depth=tiefe, restorability=rs
            )
            if not (gesperrt and "artifact_freedom" in grund):
                verstoesse.append(f"SpecConstitution.is_export_blocked: Tiefe={tiefe} rs={rs!r} sperrt AF<0.32 nicht")
            probleme = konstitution.check_paragraph_zero(
                audio, 48000, artifact_freedom=knapp_unter_boden, hpi=0.5, chain_depth=tiefe, restorability=rs
            )
            if not any("VETO" in p and "artifact_freedom" in p for p in probleme):
                verstoesse.append(
                    f"SpecConstitution.check_paragraph_zero: Tiefe={tiefe} rs={rs!r} vetoet AF<0.32 nicht"
                )
    return verstoesse


def _kalibrierungs_verstoesse() -> list[str]:
    from backend.core.calibrated_constants import CalibratedConstants
    from backend.core.calibration_context import CalibrationContext

    verstoesse: list[str] = []
    for tiefe in _TIEFEN:
        for rs in _RS_GITTER:
            kontext = CalibrationContext(restorability_score=rs, transfer_chain_depth=tiefe, material_type="cassette")
            for quelle, wert in (
                ("CalibrationContext", kontext.artifact_freedom_min),
                ("CalibratedConstants", CalibratedConstants.from_context(kontext).artifact_freedom_min),
            ):
                if not wert >= _AF_BODEN:
                    verstoesse.append(f"{quelle}.artifact_freedom_min: Tiefe={tiefe} rs={rs!r} → {wert!r}")
    return verstoesse


# ── Pins ─────────────────────────────────────────────────────────────────────────────────────────────


def test_g163_floor_absolut_garantie() -> None:
    """§G163 (GEBOTE.md, Kategorie XXIII): artifact_freedom-Mindestwert NIEMALS < 0.32.

    Norm (Spec §v10.704 B30, Invariante „AF-Minimum absolut 0.32“): Kein Erzeuger eines AF-Mindestwerts darf
    unter 0.32 fallen — auch nicht bei Extremwerten (depth 0…12 bzw. leere Kette, rs < 0, rs > 100, ±inf, NaN).

    - Export-Gates (ArtifactFreedomGate, FinalExportAudioGate): Die B30-Formel steht inline in
      ``unified_restorer_v3.py``; ihr Laufzeitwert ist über das Gitter ≥ 0.32 und der Boden greift mit exakt
      0.32 (``max(0.32, …)`` mit geklemmtem rs — NaN-sicher durch die Argumentreihenfolge).
    - HPI-Gate, Konstitution, Kalibrierung: Der Mindestwert liegt für jede Eingabe ≥ 0.32; ein AF knapp unter
      dem Boden wird von der Konstitution immer gesperrt.
    """
    verstoesse = [*_hpi_gate_verstoesse(), *_konstitution_verstoesse(), *_kalibrierungs_verstoesse()]

    fundstellen = _uv3_export_gate_fundstellen()
    assert len(fundstellen) >= 2, (
        "Erwartet: mindestens die beiden Inline-Export-Gates (ArtifactFreedomGate, FinalExportAudioGate) mit "
        f"``*af_min``-Zuweisung in {_UV3_PFAD.name}; gefunden: {[f.name for f in fundstellen]}"
    )
    for fundstelle in fundstellen:
        befund = _pruefe_boden(fundstelle)
        verstoesse.extend(befund.verstoesse)
        assert befund.eingaben_wirken, (
            f"{fundstelle.name} (Zeile {fundstelle.zeile}): extrahierte Formel reagiert nicht auf Kettenlänge/rs — "
            "Slice entartet, Auswerter/Namenskonvention anpassen"
        )
        if not befund.verstoesse:  # bei Unterschreitungen meldet die Sammel-Assertion unten die Details
            assert befund.tiefster_wert == _AF_BODEN, (
                f"{fundstelle.name} (Zeile {fundstelle.zeile}): Boden greift bei {befund.tiefster_wert!r}, "
                f"Spec §v10.704 verlangt exakt {_AF_BODEN}"
            )

    assert not verstoesse, "AF-Mindestwert unter 0.32 (§G163 (GEBOTE.md, Kategorie XXIII)):\n" + "\n".join(
        verstoesse[:25]
    )


_SYNTHETISCHE_QUELLE = """
def tor(transfer_chain, restorability_score, artifact_freedom):
    tiefe = max(1, len(transfer_chain))
    if tiefe >= 5:
        basis = 0.48
    elif tiefe >= 4:
        basis = 0.58
    elif tiefe == 3:
        basis = 0.72
    elif tiefe == 2:
        basis = 0.82
    else:
        basis = 0.88
    rs = max(0.0, min(100.0, float(restorability_score)))
    af_min = {formel}
    return artifact_freedom < af_min
"""


@pytest.mark.parametrize(
    ("formel", "verstoss_erwartet"),
    [
        pytest.param("max(0.32, basis - (100.0 - rs) * 0.0045)", False, id="konform"),
        pytest.param("basis - (100.0 - rs) * 0.0045", True, id="boden_fehlt"),
        pytest.param("max(0.30, basis - (100.0 - rs) * 0.0045)", True, id="boden_030"),
        pytest.param(
            "max(basis - (100.0 - float(restorability_score)) * 0.0045, 0.32)", True, id="nan_unsichere_reihenfolge"
        ),
    ],
)
def test_g163_selbsttest_ast_pruefer_findet_verstoesse(formel: str, verstoss_erwartet: bool) -> None:
    """§G163 (GEBOTE.md, Kategorie XXIII), Selbsttest: Der AST-Prüfer erkennt synthetische Boden-Verstöße.

    Ohne diesen Selbsttest könnte der Pin auf den Inline-Export-Gates „grün“ sein, weil der Prüfer nie
    anschlägt. Die konforme Variante belegt zugleich, dass er nicht fälschlich anschlägt und der Boden exakt
    0.32 beträgt.
    """
    baum = ast.parse(_SYNTHETISCHE_QUELLE.format(formel=formel))
    fundstellen = list(_af_min_fundstellen(baum))
    assert len(fundstellen) == 1
    befund = _pruefe_boden(fundstellen[0])
    assert befund.eingaben_wirken
    assert bool(befund.verstoesse) is verstoss_erwartet, befund.verstoesse[:3]
    if not verstoss_erwartet:
        assert befund.tiefster_wert == _AF_BODEN
