from __future__ import annotations

"""[RELEASE_MUST] Diagnose-Metadaten-Wahrheit (Spec 07 §8.7, Spec 02 §2.2.3a).

Zwei maschinell prüfbare Regeln:

1. Kein „toter Zeuge“: Eine Zuweisung ``x = sys.modules.get(...)`` im
   Produktionscode MUSS ihr Ergebnis auch auswerten (``x`` wird in derselben
   Funktion gelesen). Befund 2026-11-22: 28 Diagnose-Helfer in
   ``UnifiedRestorerV3`` lasen den Modulzustand, werteten ihn nie aus und
   meldeten unbedingt ``{"active": True}``.
2. ``"active": True`` in einem ``_compute_*``-Diagnose-Helfer von
   ``UnifiedRestorerV3`` ist nur nach erfolgreicher Instanziierung zulässig —
   nicht nach bloßem Typ-Import oder Cache-Präsenz.

Der Wächter hat einen Selbsttest (er MUSS an einem synthetischen Verstoß
auslösen), damit er nicht stillschweigend wirkungslos wird.
"""

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.normative]

_ROOT = Path(__file__).resolve().parents[2]
_PRODUCTION_ROOTS = ("backend", "denker", "plugins", "Aurik10", "cli", "scripts")
_UV3 = _ROOT / "backend" / "core" / "unified_restorer_v3.py"


def _is_sys_modules_get(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and ast.unparse(node.func) == "sys.modules.get"


def dead_sys_modules_witnesses(source: str) -> list[tuple[str, int, str]]:
    """Liefert (Funktion, Zeile, Variable) für jeden nie gelesenen sys.modules-Zeugen."""
    violations: list[tuple[str, int, str]] = []
    tree = ast.parse(source)
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        loaded = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        for node in ast.walk(fn):
            name: str | None = None
            value: ast.AST | None = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name, value = node.targets[0].id, node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                name, value = node.target.id, node.value
            if name is not None and value is not None and _is_sys_modules_get(value) and name not in loaded:
                violations.append((fn.name, node.lineno, name))
    return violations


def unbacked_active_claims(source: str) -> list[tuple[str, int]]:
    """Liefert (Funktion, Zeile) für ``"active": True`` in ``_compute_*`` ohne Konstruktor-Aufruf."""
    violations: list[tuple[str, int]] = []
    tree = ast.parse(source)
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("_compute_"):
            continue
        claims = [
            d.lineno
            for d in ast.walk(fn)
            if isinstance(d, ast.Dict)
            for k, v in zip(d.keys, d.values)
            if isinstance(k, ast.Constant) and k.value == "active" and isinstance(v, ast.Constant) and v.value is True
        ]
        if not claims:
            continue
        instantiates = any(
            isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id.lstrip("_")[:1].isupper()
            for c in ast.walk(fn)
        )
        if not instantiates:
            violations.extend((fn.name, line) for line in claims)
    return violations


def _production_sources() -> list[Path]:
    files: list[Path] = []
    for root in _PRODUCTION_ROOTS:
        for path in sorted((_ROOT / root).rglob("*.py")):
            if "_vendor_" in path.as_posix() or ".venv" in path.parts:
                continue
            files.append(path)
    return files


def test_gate_selftest_flags_dead_witness_and_accepts_checked_one() -> None:
    dead = "import sys\n\ndef f():\n    cache = sys.modules.get('x')\n    return {'active': True}\n"
    used = "import sys\n\ndef f():\n    cache = sys.modules.get('x')\n    return cache is not None\n"
    assert dead_sys_modules_witnesses(dead) == [("f", 4, "cache")]
    assert dead_sys_modules_witnesses(used) == []


def test_gate_selftest_flags_active_claim_without_instantiation() -> None:
    bare = "def _compute_x_result(self):\n    return {'class': 'A', 'active': True}\n"
    built = "def _compute_x_result(self):\n    A()\n    return {'class': 'A', 'active': True}\n"
    assert unbacked_active_claims(bare) == [("_compute_x_result", 2)]
    assert unbacked_active_claims(built) == []


def test_no_dead_sys_modules_witness_in_production_code() -> None:
    offenders: list[str] = []
    for path in _production_sources():
        try:
            found = dead_sys_modules_witnesses(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        offenders.extend(f"{path.relative_to(_ROOT)}:{line} {fn}() liest '{var}' nie aus" for fn, line, var in found)
    assert not offenders, "Toter sys.modules-Zeuge (Spec 07 §8.7 Regel 1):\n" + "\n".join(offenders)


def test_active_true_in_unified_restorer_diagnostics_requires_instantiation() -> None:
    offenders = unbacked_active_claims(_UV3.read_text(encoding="utf-8"))
    assert not offenders, (
        "'active': True ohne Instanziierung in UnifiedRestorerV3 (Spec 07 §8.7 Regel 2): "
        + ", ".join(f"{fn}:{line}" for fn, line in offenders)
    )
