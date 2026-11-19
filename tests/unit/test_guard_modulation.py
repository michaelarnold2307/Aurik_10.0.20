"""Vertrags- und Regressionstest der zentralen Guard-Modulation (§DENKER).

Befund 2026-10-07 (Register D-K3-14): ``PhaseInteractionDenker.resolve_guard_modulation``
existierte als Klassenattribut **nicht**. Die Definition lag hinter einem
``return`` in der Modulfunktion ``_categories_conflict_with_material`` und war
damit unerreichbarer Code. Der Aufrufer in
``unified_restorer_v3._profiled_phase_call`` verließ sich auf
``# type: ignore[attr-defined]`` und verschluckte den ``AttributeError`` auf
``debug``-Ebene. Folge: die im Docstring zugesagte „zentrale Guard-Modulation"
lief in JEDEM Lauf nicht — die Stärke blieb unmoduliert.

Dieser Test sichert beides: die **Existenz** der Methode (Regression) und ihre
**Zusagen** (Vertrag, §V6/§G5/§III.6 (copilot-instructions.md)).
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "denker" / "phase_interaction_denker.py"


def _nav():
    """Lädt ``denker.phase_interaction_denker``.

    Bewusst ein normaler Import: ein Laden per ``spec_from_file_location`` ohne
    Eintrag in ``sys.modules`` zerbricht ``@dataclass`` unter Python 3.10
    (``_is_type`` greift auf ``sys.modules[cls.__module__]`` zu).
    """
    import denker.phase_interaction_denker as module

    return module


class _Budget:
    """GoalBudget-Attrappe: ``fraction_left`` liefert je Ziel einen Wert."""

    def __init__(self, value: float) -> None:
        self._value = value

    def fraction_left(self, _goal: str) -> float:
        return self._value


class _Wisdom:
    """GuardWisdom-Attrappe."""

    def __init__(self, value: float) -> None:
        self._value = value

    def get_strength_mod(self) -> float:
        return self._value


class _BrokenBudget:
    """GoalBudget, dessen Auslesen scheitert — darf nicht zum Abbruch führen."""

    def fraction_left(self, _goal: str) -> float:
        raise RuntimeError("Budget nicht lesbar")


def _method_node(name: str) -> ast.FunctionDef:
    """Sucht die Methode ``name`` im Syntaxbaum der Datei.

    ``inspect.getsource`` entzieht sich hier einer verlässlichen Prüfung: es
    dedentiert den Quelltext und zerbricht die Auswertung. Der Syntaxbaum der
    ganzen Datei ist die belastbare Quelle.
    """
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PhaseInteractionDenker":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{name} nicht in PhaseInteractionDenker gefunden")


# ── Regression: die Methode existiert und ist erreichbar ────────────────────


def test_method_exists_and_is_reachable_as_class_attribute() -> None:
    """Die Methode MUSS am Klassenobjekt hängen (der ursprüngliche Defekt)."""
    nav = _nav()
    assert hasattr(nav.PhaseInteractionDenker, "resolve_guard_modulation"), (
        "resolve_guard_modulation fehlt am Klassenobjekt — genau der Defekt aus D-K3-14"
    )
    assert isinstance(nav.PhaseInteractionDenker.__dict__["resolve_guard_modulation"], staticmethod), (
        "Die Methode muss statisch sein, weil UV3 sie auf der KLASSE aufruft"
    )


def test_call_site_contract_is_satisfied_keyword_only() -> None:
    """UV3 ruft sie auf der Klasse mit ausschließlich Schlüsselwort-Argumenten auf."""
    nav = _nav()
    result = nav.PhaseInteractionDenker.resolve_guard_modulation(base_strength=0.8, phase_id="phase_03_denoise")
    assert result == pytest.approx(0.8)


def test_no_unreachable_statements_after_return_in_module() -> None:
    """Allgemeiner Wächter gegen die Fehlerklasse: Code hinter ``return``."""
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for index, statement in enumerate(body[:-1]):
            if isinstance(statement, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
                following = body[index + 1]
                if not isinstance(following, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    offenders.append(f"{getattr(node, 'name', '?')}:{statement.lineno}")
    assert not offenders, f"Unerreichbarer Code hinter return/raise: {offenders}"


# ── Vertrag: Zusagen 1–4 ────────────────────────────────────────────────────


@pytest.mark.parametrize("value", [0.0, 0.13, 0.5, 0.999, 1.0])
def test_without_any_guard_influence_the_result_is_identical(value: float) -> None:
    """Zusage 1: ohne Guard-Einfluss keine stille Stärke-Änderung."""
    nav = _nav()
    assert nav.PhaseInteractionDenker.resolve_guard_modulation(base_strength=value) == value


def test_guard_wisdom_throttles_proportionally() -> None:
    """Eine einzige Drosselung wirkt als Faktor."""
    nav = _nav()
    assert nav.PhaseInteractionDenker.resolve_guard_modulation(
        base_strength=0.8, guard_wisdom=_Wisdom(0.5)
    ) == pytest.approx(0.4)


def test_penalties_are_averaged_not_multiplied() -> None:
    """Die Guards werden GEMITTELT, nicht multipliziert (Kern der Verbesserung)."""
    nav = _nav()
    # GoalBudget 0,4 → Faktor 0,5 (Gewicht 0,40); Wisdom 0,6 (0,50); CrossGuard 0,85 (0,10)
    expected = (0.5 * 0.40 + 0.6 * 0.50 + 0.85 * 0.10) / 1.0
    result = nav.PhaseInteractionDenker.resolve_guard_modulation(
        base_strength=1.0,
        goal_budget=_Budget(0.4),
        guard_wisdom=_Wisdom(0.6),
        cross_guard_results={"verdict": "degraded"},
    )
    assert result == pytest.approx(expected)
    assert result != pytest.approx(0.5 * 0.6 * 0.85), "Multiplikation wäre die falsche Semantik"


def test_critical_phase_floor_protects_against_throttling() -> None:
    """Zusage 2: ein kritischer Defekt darf nicht unter den Material-Floor gedrosselt werden."""
    nav = _nav()
    result = nav.PhaseInteractionDenker.resolve_guard_modulation(
        base_strength=0.8,
        guard_wisdom=_Wisdom(0.2),
        phase_id="phase_01_click_removal",
        material="cassette",
    )
    assert result == pytest.approx(0.40)


def test_floor_never_raises_above_the_incoming_strength() -> None:
    """Zusage 2: der Floor schützt nur — er hebt nie über die Eingabe an."""
    nav = _nav()
    result = nav.PhaseInteractionDenker.resolve_guard_modulation(
        base_strength=0.2,
        guard_wisdom=_Wisdom(0.1),
        phase_id="phase_01_click_removal",
        material="cassette",
    )
    assert result <= 0.2


def test_non_critical_phase_has_no_floor() -> None:
    """Nur kritische Phasen haben einen Floor (§G7 (copilot-instructions.md): Stärke pro Defekt)."""
    nav = _nav()
    result = nav.PhaseInteractionDenker.resolve_guard_modulation(
        base_strength=0.8,
        guard_wisdom=_Wisdom(0.2),
        phase_id="phase_39_air_band_enhancement",
        material="cassette",
    )
    assert result == pytest.approx(0.16)


def test_result_never_exceeds_the_input() -> None:
    """Zusage 2 als Eigenschaft: keine versteckte Anhebung über die Eingabe."""
    nav = _nav()
    for base in (0.0, 0.05, 0.25, 0.5, 1.0):
        for wisdom in (0.0, 0.25, 0.75, 1.0):
            result = nav.PhaseInteractionDenker.resolve_guard_modulation(
                base_strength=base,
                guard_wisdom=_Wisdom(wisdom),
                phase_id="phase_01_click_removal",
                material="vinyl",
            )
            assert 0.0 <= result <= base + 1e-12


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_input_returns_zero_and_warns(bad: float, caplog: pytest.LogCaptureFixture) -> None:
    """Zusage 3: kein NaN verlässt die Funktion, und der Fallback wird protokolliert."""
    nav = _nav()
    with caplog.at_level("WARNING"):
        result = nav.PhaseInteractionDenker.resolve_guard_modulation(base_strength=bad)
    assert result == 0.0
    assert math.isfinite(result)
    assert any("resolve_guard_modulation" in record.getMessage() for record in caplog.records)


def test_broken_goal_budget_is_reported_and_does_not_abort(caplog: pytest.LogCaptureFixture) -> None:
    """Ein unlesbares Budget entfällt mit Begründung — die übrigen Guards wirken weiter."""
    nav = _nav()
    with caplog.at_level("WARNING"):
        result = nav.PhaseInteractionDenker.resolve_guard_modulation(
            base_strength=1.0,
            goal_budget=_BrokenBudget(),
            guard_wisdom=_Wisdom(0.5),
        )
    assert result == pytest.approx(0.5)
    assert any("GoalBudget" in record.getMessage() for record in caplog.records)


def test_deterministic_and_free_of_clock_or_randomness() -> None:
    """Zusage 4: dieselbe Eingabe liefert dasselbe Ergebnis, ohne Zufall und Uhr."""
    nav = _nav()
    arguments = {"base_strength": 0.7, "guard_wisdom": _Wisdom(0.63), "phase_id": "phase_09_crackle_removal"}
    first = nav.PhaseInteractionDenker.resolve_guard_modulation(**arguments)
    for _ in range(20):
        assert nav.PhaseInteractionDenker.resolve_guard_modulation(**arguments) == first

    method = _method_node("resolve_guard_modulation")
    imported: set[str] = set()
    for node in ast.walk(method):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"random", "time", "datetime", "uuid", "secrets"}, imported

    used_modules = {
        node.value.id
        for node in ast.walk(method)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
    }
    assert not used_modules & {"random", "time", "datetime", "uuid"}, used_modules


def test_guard_weights_are_documented_and_normalised() -> None:
    """Die Gewichte sind EINE Quelle und summieren sich auf 1,0."""
    nav = _nav()
    weights = nav.PhaseInteractionDenker._GUARD_WEIGHTS
    assert set(weights) == {"goal_budget", "guard_wisdom", "cross_guard"}
    assert sum(weights.values()) == pytest.approx(1.0)


def test_material_floor_table_covers_fragile_carriers() -> None:
    """Die Floor-Tabelle deckt genau die fragilen Träger ab und liegt nie unter dem Default."""
    nav = _nav()
    table = nav.PhaseInteractionDenker._CRITICAL_FLOOR_BY_MATERIAL
    assert {"cassette", "tape", "reel_tape", "vinyl", "shellac"} <= set(table)
    default = nav.PhaseInteractionDenker._CRITICAL_FLOOR_DEFAULT
    for name, value in table.items():
        assert 0.0 < value <= 1.0, name
        assert value >= default, f"{name}: {value} liegt unter dem Default-Floor {default}"
    assert 0.0 < default <= 1.0
