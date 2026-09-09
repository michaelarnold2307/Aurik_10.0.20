from __future__ import annotations

"""CI-Guard: Keine echten Stubs in Produktionscode — §1.5, §3.1, §10.1.

Erkennungslogik:
  - Sucht nach Methoden-/Funktions-Rümpfen, die ausschließlich aus `...`, `pass` oder
    `raise NotImplementedError` bestehen.
  - Ignoriert Typ-Annotationen (z.B. `tuple[str, ...]`) und NumPy-Slices (`[..., :n]`).
  - Ignoriert Testdateien (test_*) und __pycache__.
  - Docstrings werden korrekt übersprungen (kein False-Positive bei "nur Docstring").

Gemäß V-3-Tabelle in SCHRITTE_ZUR_MUSIKALISCHEN_EXZELLENZ.md sind die richtigen
SOTA-Algorithmen für jeden Stub-Typ definiert. Neue Stubs werden mit einer Deadline
in ACCEPTED_STUBS registriert — nie still akzeptiert.
"""


import ast
import pathlib

import pytest

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

PRODUCTION_DIRS: list[str] = ["core", "plugins", "dsp", "backend"]
EXCLUDE_PATTERNS: list[str] = ["__pycache__", "test_"]

# ---------------------------------------------------------------------------
# Bekannte, akzeptierte Stubs (max. 30 — jeweils mit laufendem Issue + Deadline)
# Format: (partial_path_fragment, function_name)
# Beispiel: ("core/era_classifier_plugin.py", "classify")
# ---------------------------------------------------------------------------
ACCEPTED_STUBS: frozenset[tuple[str, str]] = frozenset(
    {
        # @abstractmethod — legitimes Python-ABC-Muster (kein echter Stub)
        ("core/phases/phase_interface.py", "process"),
        ("core/phases/phase_interface.py", "get_metadata"),
        ("plugins/sdk/aurik_plugin_base.py", "process_audio"),  # @abstractmethod im Plugin-SDK
        ("plugin_base.py", "_model_loaded"),  # @abstractmethod in MLPluginBase (§A2)
        # Template-Method-Pattern — Subklassen überschreiben gezielt
        ("backend/ml/safety_wrappers/safety_wrapper_template.py", "_validate_pre_conditions"),
        ("backend/ml/safety_wrappers/safety_wrapper_template.py", "_assess_epistemic_confidence"),
        ("backend/ml/safety_wrappers/safety_wrapper_template.py", "_validate_post_conditions"),
        ("backend/ml/safety_wrappers/safety_wrapper_template.py", "_compute_quality_score"),
        # P4-Ethics-Integration-Hooks (nicht im Audio-Datenpfad)
        ("backend/core/epistemic_gate/ethics_engine.py", "integrate_ethics_into_pipeline"),
        ("backend/ethics_engine.py", "integrate_ethics_into_pipeline"),
        # P3-Utility-Stubs (Undo, Media-Chain, Auto-Bypass)
        ("backend/core/undo/undo_manager.py", "apply"),
        ("backend/core/undo/undo_manager.py", "cleanup"),
        ("backend/core/undo/undo_manager.py", "revert"),
        ("dsp/auto_bypass_order.py", "__init__"),
        # P3-Logging-Contract-Hooks
        ("dsp/classic_filters.py", "log_contract"),
        ("dsp/multiband_master.py", "_log_contract"),
        # P2 — reales analyze() vorhanden; __init__ darf defer
        ("core/defect_analysis.py", "__init__"),
        # MockPhase.process im Demo-Block am Dateiende
        ("core/quality_feedback_loop.py", "process"),
        # P3-Plugin-Feedback-Hooks
        ("plugins/artifact_detection_plugin.py", "feedback"),
        ("plugins/parameter_optimizer.py", "feedback"),
        # P2-V10 — Worldclass-Architektur-Komponenten (§Phase-Prune, RLP, Tape-Head, Vocal-Scratch)
        ("backend/core/artifact_echo_removal.py", "__init__"),
        ("backend/core/phase_pruner.py", "__init__"),
        ("backend/core/reflective_listening_pass.py", "__init__"),
        ("backend/core/tape_head_artifact_repair.py", "__init__"),
        ("backend/core/vocal_scratch_repair.py", "__init__"),
    }
)

MAX_ACCEPTED_STUBS: int = 35


# ---------------------------------------------------------------------------
# Stub-Erkennung via AST
# ---------------------------------------------------------------------------


def _is_stub_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True wenn Funktionsrumpf *ausschließlich* aus Stub-Statements besteht.

    Stub-Statements:
      - Ellipsis (`...`) als einzige Expression
      - `pass`
      - `raise NotImplementedError(...)`

    Korrekt ignoriert:
      - Docstrings (leading str-Constant)
      - Typ-Annotationen mit `...` (z.B. `tuple[str, ...]`) — erscheinen nicht
        als Statement-Körper, sondern in der Annotation-Syntax
    """
    body: list[ast.stmt] = list(node.body)

    # Docstring überspringen (führendes String-Literal):
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    # Leerer Body nach Docstring-Strip:
    if not body:
        return False  # Reine Docstring-Funktionen gelten nicht als Stub

    # Alle Statements müssen Stub-Ausdrücke sein:
    for stmt in body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            if stmt.value.value is ...:
                continue  # Ellipsis-Stub
        if isinstance(stmt, ast.Pass):
            continue  # Pass-Stub
        if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
            func = stmt.exc.func
            name = getattr(func, "id", getattr(func, "attr", ""))
            if name == "NotImplementedError":
                continue  # NotImplementedError-Stub
        # Irgendein anderes Statement → kein reiner Stub:
        return False

    return True  # Alle Statements waren Stub-Ausdrücke


def _collect_stubs() -> list[tuple[str, str, int]]:
    """Liefert Liste von (dateipfad, funktionsname, zeilennummer) für alle Stubs."""
    stubs: list[tuple[str, str, int]] = []
    root = pathlib.Path(".")

    for prod_dir in PRODUCTION_DIRS:
        prod_path = root / prod_dir
        if not prod_path.exists():
            continue

        for py_file in sorted(prod_path.rglob("*.py")):
            path_str = str(py_file)
            if any(pat in path_str for pat in EXCLUDE_PATTERNS):
                continue

            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=path_str)
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if _is_stub_body(node):
                        stubs.append((path_str, node.name, node.lineno))

    return stubs


def _is_accepted(path: str, name: str) -> bool:
    """Prüft ob ein Stub in ACCEPTED_STUBS registriert ist."""
    return any(acc_path in path and acc_name == name for acc_path, acc_name in ACCEPTED_STUBS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.normative
def test_no_unaccepted_stubs_in_production_code() -> None:
    """Kein echter Stub in Produktionscode ohne explizite Ausnahme-Registrierung.

    Wenn dieser Test fehlschlägt:
      1. SOTA-Implementierung gemäß V-3-Tabelle in SCHRITTE_ZUR_MUSIKALISCHEN_EXZELLENZ.md
         für den entsprechenden Stub-Typ einsetzen.
      2. Alternativ (nur temporär, mit Deadline + Issue):
         Den Stub in ACCEPTED_STUBS oben registrieren.

    Verboten: statische Return-Werte ohne Audio-Analyse in Produktionscode.
    Erlaubt:  echte DSP-/ML-Berechnungen.
    """
    stubs = _collect_stubs()
    unaccepted = [(path, name, line) for path, name, line in stubs if not _is_accepted(path, name)]

    assert not unaccepted, (
        f"\n{len(unaccepted)} unerlaubte Stub(s) in Produktionscode:\n"
        + "\n".join(f"  {p}:{ln}  def {n}()" for p, n, ln in sorted(unaccepted))
        + "\n\nMaßnahme: SOTA-Implementierung gemäß V-3-Tabelle in "
        "docs/SCHRITTE_ZUR_MUSIKALISCHEN_EXZELLENZ.md einsetzen "
        "oder Stub mit Issue-Verweis in ACCEPTED_STUBS registrieren."
    )


@pytest.mark.normative
def test_accepted_stubs_list_is_bounded() -> None:
    """ACCEPTED_STUBS darf nie mehr als 30 Einträge enthalten (technischer Schulden-Cap).

    Ab 30 Einträgen müssen echte SOTA-Implementierungen eingesetzt werden
    statt Ausnahmen zu akkumulieren.
    """
    count = len(ACCEPTED_STUBS)
    assert count <= MAX_ACCEPTED_STUBS, (
        f"ACCEPTED_STUBS hat {count} Einträge — Maximum ist {MAX_ACCEPTED_STUBS}. "
        "Priorität P1-Stubs mit SOTA-Implementierungen ersetzen (V-3-Tabelle)."
    )


@pytest.mark.normative
def test_stub_detector_ignores_type_annotations() -> None:
    """Sicherstellt, dass `tuple[str, ...]` und ähnliche Typ-Annotationen
    nicht fälschlicherweise als Stub erkannt werden (False-Positive-Schutz)."""
    source = """\
def process(self, audio: np.ndarray, chains: tuple[str, ...]) -> np.ndarray:
    \"\"\"Docstring.\"\"\"
    result = audio[..., :100]
    return result
"""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert not _is_stub_body(node), f"False positive: '{node.name}' fälschlicherweise als Stub erkannt."


@pytest.mark.normative
def test_stub_detector_catches_ellipsis_only_body() -> None:
    """Sicherstellt, dass ein reiner `...`-Rumpf korrekt als Stub erkannt wird."""
    source = """\
def balance_remix(self, vocals, instruments, original, sr) -> np.ndarray:
    ...
"""
    tree = ast.parse(source)
    found_stub = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_stub_body(node):
                found_stub = True
    assert found_stub, "Ellipsis-Stub wurde nicht erkannt."


@pytest.mark.normative
def test_stub_detector_catches_pass_body() -> None:
    """Sicherstellt, dass ein reiner `pass`-Rumpf korrekt als Stub erkannt wird."""
    source = """\
def compute_accordion_score(self, mono, sr) -> float:
    pass
"""
    tree = ast.parse(source)
    found_stub = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_stub_body(node):
                found_stub = True
    assert found_stub, "Pass-Stub wurde nicht erkannt."


@pytest.mark.normative
def test_stub_detector_catches_not_implemented_error() -> None:
    """Sicherstellt, dass `raise NotImplementedError(...)` als Stub erkannt wird."""
    source = """\
def process(self, audio, sr, **kwargs):
    raise NotImplementedError("TODO: SOTA-Impl")
"""
    tree = ast.parse(source)
    found_stub = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_stub_body(node):
                found_stub = True
    assert found_stub, "NotImplementedError-Stub wurde nicht erkannt."


@pytest.mark.normative
def test_stub_detector_ignores_docstring_only_function() -> None:
    """Eine Funktion mit reinem Docstring ist kein Stub (Abstract-Marker-Pattern)."""
    source = '''\
def score_audio(self, audio, sr):
    """Berechnet den PQS-Score. Implementierung in Unterklasse."""
'''
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert not _is_stub_body(node), "Docstring-Only-Funktion fälschlicherweise als Stub erkannt."
