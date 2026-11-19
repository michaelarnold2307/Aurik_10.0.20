from __future__ import annotations

"""Normative guard: UI layer must not import core/dsp/plugin modules directly.

Frontend modules under Aurik10 must communicate with backend internals via
backend.api.bridge only (Spec 08 §11).
"""


import ast
from pathlib import Path

import pytest

UI_ROOT = Path("Aurik10")
FORBIDDEN_PREFIXES: tuple[str, ...] = ("backend.core", "plugins", "dsp")


def _is_forbidden_module(name: str) -> bool:
    return any(name == p or name.startswith(f"{p}.") for p in FORBIDDEN_PREFIXES)


@pytest.mark.normative
def test_ui_uses_bridge_not_core_imports() -> None:
    violations: list[tuple[str, int, str]] = []

    for py_file in UI_ROOT.rglob("*.py"):
        rel = str(py_file).replace("\\", "/")
        if "/__pycache__/" in rel or rel.endswith("/__init__.py"):
            continue

        try:
            src = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        tree = ast.parse(src, filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden_module(alias.name):
                        violations.append((rel, node.lineno, f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if _is_forbidden_module(module):
                    violations.append((rel, node.lineno, f"from {module} import ..."))

    assert not violations, "Direct UI imports from core/dsp/plugins are forbidden:\n" + "\n".join(
        f"- {path}:{line} -> {stmt}" for path, line, stmt in violations
    )
