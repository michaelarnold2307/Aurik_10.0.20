from __future__ import annotations

"""Magic-Button/Autopilot CI-Gate — [RELEASE_MUST] One-Button-Vertrag.

Spec-Quelle: .github/copilot-instructions.md
  - Abschnitt "Autonomer Magic-Button-Betrieb + Profi-Highlights"
  - Nutzer wählt nur: RESTORATION oder STUDIO_2026
  - Frontend-Einstieg muss über AurikDenker.denke() laufen (kein UV3-Bypass)
"""


import ast
from pathlib import Path

import pytest

_MODERN_WINDOW = Path("Aurik10/ui/modern_window.py")
_COPILOT_INSTRUCTIONS = Path(".github/copilot-instructions.md")


def _load_ast(path: Path) -> tuple[str, ast.AST]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    return src, tree


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_one_button_release_must_section_exists() -> None:
    """Doku-Gate: RELEASE_MUST-Abschnitt für One-Button-Autopilot muss vorhanden sein."""
    assert _COPILOT_INSTRUCTIONS.exists(), (
        "copilot-instructions.md fehlt. RELEASE_MUST-Vorgaben müssen im Repo versioniert sein."
    )
    text = _COPILOT_INSTRUCTIONS.read_text(encoding="utf-8")
    assert "[RELEASE_MUST] Autonomer Magic-Button-Betrieb" in text, (
        "RELEASE_MUST-Abschnitt für Magic-Button-Autopilot fehlt in .github/copilot-instructions.md."
    )


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_magic_buttons_offer_exactly_two_modes() -> None:
    """UI-Gate: Magic Buttons dürfen nur RESTORATION und STUDIO_2026 anbieten."""
    assert _MODERN_WINDOW.exists(), f"{_MODERN_WINDOW} nicht gefunden."

    _, tree = _load_ast(_MODERN_WINDOW)
    mode_literals: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "_process_with_mode":
            continue
        if not node.args:
            continue
        arg0 = node.args[0]
        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
            mode_literals.add(arg0.value)

    expected = {"RESTORATION", "STUDIO_2026"}
    assert mode_literals == expected, (
        f"Magic-Buttons müssen exakt zwei Modi anbieten: RESTORATION und STUDIO_2026. Gefunden: {sorted(mode_literals)}"
    )


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_batch_thread_uses_aurik_denker_entrypoint_not_uv3_direct() -> None:
    """Pipeline-Gate: Verarbeitung in Modern UI läuft über AurikDenker.denke()."""
    src, _ = _load_ast(_MODERN_WINDOW)

    assert "_bridge_get_aurik_denker_class" in src, (
        "Bridge-Entrypoint _bridge_get_aurik_denker_class fehlt in modern_window.py."
    )
    assert ".denke(" in src, (
        "AurikDenker.denke() wird in modern_window.py nicht aufgerufen; "
        "der verpflichtende Frontend-Einstiegspunkt fehlt."
    )
    assert "get_restorer().restore(" not in src, (
        "Direkter UV3-Aufruf get_restorer().restore() in modern_window.py ist verboten."
    )
    assert "UnifiedRestorerV3.restore(" not in src, (
        "Direkter Aufruf UnifiedRestorerV3.restore() in modern_window.py ist verboten."
    )


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_mode_mapping_from_magic_button_to_aurik_mode_is_canonical() -> None:
    """Modus-Mapping-Gate: STUDIO_2026 -> studio2026, sonst restoration."""
    src, _ = _load_ast(_MODERN_WINDOW)

    # Guard auf die explizite Mapping-Logik im BatchThread:
    assert 'if mode == "STUDIO_2026"' in src, "STUDIO_2026-Branch fehlt im BatchProcessingThread-Mapping."
    assert '_aurik_mode = "studio2026"' in src, "STUDIO_2026 muss auf Aurik-Modus 'studio2026' gemappt werden."
    assert '_aurik_mode = "restoration"' in src, "RESTORATION muss auf Aurik-Modus 'restoration' gemappt werden."


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_batch_thread_uses_singleton_accessor_not_direct_instantiation() -> None:
    """Singleton-Gate: BatchProcessingThread nutzt den Prozess-Singleton — kein neues Objekt pro Run.

    No-Competing-Instances-Protokoll (RELEASE_MUST):
      - _bridge_get_aurik_denker_instance() muss im Source vorhanden sein.
      - _denker_singleton darf NICHT als AurikDenkerClass() direkt instanziiert werden.
    """
    src, tree = _load_ast(_MODERN_WINDOW)

    # Die neue Singleton-Bridge-Funktion muss verankert sein
    assert "_bridge_get_aurik_denker_instance" in src, (
        "Singleton-Accessor _bridge_get_aurik_denker_instance fehlt in modern_window.py. "
        "BatchProcessingThread muss den Prozess-Singleton nutzen, "
        "nicht AurikDenkerClass() neu instanziieren."
    )

    # AurikDenkerClass() darf innerhalb von BatchProcessingThread.run() nicht aufgerufen werden
    class _CallFinder(ast.NodeVisitor):
        def __init__(self) -> None:
            self.violations: list[int] = []
            self._in_batch_class = False
            self._in_run_method = False

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            if node.name == "BatchProcessingThread":
                prev = self._in_batch_class
                self._in_batch_class = True
                self.generic_visit(node)
                self._in_batch_class = prev
            else:
                self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if self._in_batch_class and node.name == "run":
                prev = self._in_run_method
                self._in_run_method = True
                self.generic_visit(node)
                self._in_run_method = prev
            else:
                self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if self._in_run_method:
                # Detect: AurikDenkerClass() — a call whose func is the name "AurikDenkerClass"
                if isinstance(node.func, ast.Name) and node.func.id == "AurikDenkerClass":
                    self.violations.append(node.lineno)
            self.generic_visit(node)

    finder = _CallFinder()
    finder.visit(tree)

    assert not finder.violations, (
        f"BatchProcessingThread.run() instanziiert AurikDenkerClass() direkt in Zeile(n) "
        f"{finder.violations}. Pflicht: _denker_singleton = _bridge_get_aurik_denker_instance() "
        f"nutzen (No-Competing-Instances-Protokoll, RELEASE_MUST)."
    )
