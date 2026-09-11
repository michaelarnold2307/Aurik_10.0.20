from __future__ import annotations

"""Normative guard: no Docker invocation in production code paths.

Scope:
- backend/core/
- backend/api/
- backend/lyrics_guided/
- denker/
- Aurik10/

Rationale:
Production runtime must remain offline and out-of-the-box without Docker.
"""


import ast
from pathlib import Path

import pytest

PRODUCTION_DIRS: tuple[str, ...] = (
    "backend/core",
    "backend/api",
    "backend/lyrics_guided",
    "denker",
    "Aurik10",
)
EXCLUDE_CONTAINS: tuple[str, ...] = ("__pycache__", "tests/")

SUBPROCESS_FUNCS: tuple[str, ...] = (
    "run",
    "Popen",
    "check_call",
    "check_output",
    "call",
)


def _contains_docker_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return "docker" in node.value.lower()
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return any(_contains_docker_literal(elt) for elt in node.elts)
    if isinstance(node, ast.JoinedStr):
        return any(
            isinstance(v, ast.Constant) and isinstance(v.value, str) and "docker" in v.value.lower()
            for v in node.values
        )
    return False


def _is_subprocess_docker_call(call: ast.Call) -> bool:
    fn = call.func
    fn_name = ""
    owner = ""
    if isinstance(fn, ast.Name):
        fn_name = fn.id
    elif isinstance(fn, ast.Attribute):
        fn_name = fn.attr
        if isinstance(fn.value, ast.Name):
            owner = fn.value.id

    if fn_name not in SUBPROCESS_FUNCS:
        return False

    if owner not in ("", "subprocess"):
        return False

    if call.args and _contains_docker_literal(call.args[0]):
        return True

    return any(kw.value is not None and _contains_docker_literal(kw.value) for kw in call.keywords)


def _is_os_system_docker_call(call: ast.Call) -> bool:
    fn = call.func
    if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
        if fn.value.id == "os" and fn.attr in ("system", "popen"):
            return bool(call.args) and _contains_docker_literal(call.args[0])
    return False


def _find_docker_invocations(source: str, file_path: str) -> list[tuple[int, str]]:
    tree = ast.parse(source, filename=file_path)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if _is_subprocess_docker_call(node) or _is_os_system_docker_call(node):
                try:
                    snippet = ast.get_source_segment(source, node) or "<call>"
                except Exception:
                    snippet = "<call>"
                hits.append((node.lineno, snippet))
    return hits


@pytest.mark.normative
def test_no_docker_invocation_in_production_paths() -> None:
    violations: list[tuple[str, int, str]] = []

    for root_name in PRODUCTION_DIRS:
        root = Path(root_name)
        if not root.exists():
            continue

        for py_file in root.rglob("*.py"):
            rel = str(py_file).replace("\\", "/")
            if any(token in rel for token in EXCLUDE_CONTAINS):
                continue

            try:
                src = py_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            for line_no, snippet in _find_docker_invocations(src, rel):
                violations.append((rel, line_no, snippet.strip().replace("\n", " ")))

    assert not violations, "Docker invocation found in production code path(s):\n" + "\n".join(
        f"- {path}:{line_no} -> {snippet}" for path, line_no, snippet in violations
    )
