"""CI guard: alle try_allocate()-Aufrufstellen müssen float-GB-Werte < 50.0 nutzen.

§ copilot-instructions.md VERBOTEN-Katalog:
  ML-Budget-Größe als MB statt GB | try_allocate("Plugin", 630) → 630 GB statt 630 MB |
  Einheitenpräzision: Argument ist immer GB (float); 630 MB → 0.63

Regression-Guard für April-2026-Bug: GenreClassifier übergab 630 (MB-Wert) als
size_gb-Argument → try_allocate prüfte 630 GB → alle nachfolgenden Plugin-Allokationen
schlugen fehl → kein ML für den gesamten Song.
"""

import ast
import pathlib

import pytest

# Kein Plugin ist ≥ 50 GB → jeder int/float literal ≥ 50 in size_gb ist ein MB/GB-Tippfehler.
_MAX_ALLOWED_SIZE_GB = 50.0

_SEARCH_ROOTS = ("backend", "plugins")


class _TryAllocateSizeFinder(ast.NodeVisitor):
    """AST-Visitor: findet try_allocate()-Aufrufe mit überdimensionierten size_gb-Literals."""

    def __init__(self, filepath: pathlib.Path) -> None:
        self.filepath = filepath
        self.violations: list[tuple[int, object]] = []

    def visit_Call(self, node: ast.Call) -> None:
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name == "try_allocate":
            size_node: ast.expr | None = None

            # Positionaler 2. Argument
            if len(node.args) >= 2:
                size_node = node.args[1]

            # Keyword-Argument size_gb= gewinnt
            for kw in node.keywords:
                if kw.arg == "size_gb":
                    size_node = kw.value

            if size_node is not None and isinstance(size_node, ast.Constant):
                val = size_node.value
                if isinstance(val, (int, float)) and float(val) >= _MAX_ALLOWED_SIZE_GB:
                    self.violations.append((getattr(size_node, "lineno", -1), val))

        self.generic_visit(node)


def _collect_violations(workspace: pathlib.Path) -> list[tuple[pathlib.Path, int, object]]:
    violations: list[tuple[pathlib.Path, int, object]] = []
    for root_name in _SEARCH_ROOTS:
        root = workspace / root_name
        if not root.exists():
            continue
        for py_file in sorted(root.rglob("*.py")):
            # venv, __pycache__, versteckte Verzeichnisse überspringen
            if any(part.startswith(".") or part in {"__pycache__", "node_modules"} for part in py_file.parts):
                continue
            try:
                src = py_file.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(src, filename=str(py_file))
            except (SyntaxError, OSError):
                continue
            finder = _TryAllocateSizeFinder(py_file)
            finder.visit(tree)
            for line, val in finder.violations:
                violations.append((py_file, line, val))
    return violations


@pytest.mark.unit
def test_try_allocate_no_oversized_literal() -> None:
    """Alle try_allocate()-Literals müssen < 50.0 GB sein.

    Fängt MB-statt-GB-Fehler (z.B. 630 statt 0.63): kein Plugin ist ≥ 50 GB;
    ein solcher Wert ist zwangsläufig ein Einheitenirrtum.
    """
    workspace = pathlib.Path(__file__).resolve().parents[2]
    violations = _collect_violations(workspace)

    if violations:
        lines = [
            f"  {p.relative_to(workspace)}:{line} — size_gb={val!r}"
            f"  (≥ {_MAX_ALLOWED_SIZE_GB} GB ist ungültig; verwende float-GB, nicht int-MB)"
            for p, line, val in violations
        ]
        pytest.fail(
            f"try_allocate() mit überdimensioniertem Literal (≥ {_MAX_ALLOWED_SIZE_GB} GB)"
            f" in {len(violations)} Stelle(n):\n" + "\n".join(lines)
        )


@pytest.mark.unit
def test_try_allocate_positive_size() -> None:
    """Alle try_allocate()-Literals müssen > 0 sein (kein 0-GB-Zombie-Eintrag)."""
    workspace = pathlib.Path(__file__).resolve().parents[2]
    zero_violations: list[tuple[pathlib.Path, int, object]] = []
    for root_name in _SEARCH_ROOTS:
        root = workspace / root_name
        if not root.exists():
            continue
        for py_file in sorted(root.rglob("*.py")):
            if any(part.startswith(".") or part in {"__pycache__", "node_modules"} for part in py_file.parts):
                continue
            try:
                src = py_file.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(src, filename=str(py_file))
            except (SyntaxError, OSError):
                continue

            class _ZeroFinder(ast.NodeVisitor):
                _py_file = py_file  # B023-Fix: Loop-Variable beim Klassenaufbau binden

                def visit_Call(self, node: ast.Call) -> None:
                    func_name = ""
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr
                    if func_name == "try_allocate":
                        size_node = None
                        if len(node.args) >= 2:
                            size_node = node.args[1]
                        for kw in node.keywords:
                            if kw.arg == "size_gb":
                                size_node = kw.value
                        if size_node is not None and isinstance(size_node, ast.Constant):
                            val = size_node.value
                            if isinstance(val, (int, float)) and float(val) <= 0.0:
                                zero_violations.append((self._py_file, getattr(size_node, "lineno", -1), val))
                    self.generic_visit(node)

            _ZeroFinder().visit(tree)

    if zero_violations:
        lines = [f"  {p.relative_to(workspace)}:{line} — size_gb={val!r}" for p, line, val in zero_violations]
        pytest.fail(f"try_allocate() mit size_gb <= 0 in {len(zero_violations)} Stelle(n):\n" + "\n".join(lines))
