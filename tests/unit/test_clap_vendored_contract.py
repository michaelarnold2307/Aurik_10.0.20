"""CLAP-Vendored-Vertrag-Guard (Zwischenaufgabe 2026-10-06).

Befund (2026-10-06, Editor-Diagnostik + Code-Lektüre): Der lokale
LAION-CLAP-Clone unter ``models/clap/src/`` ist aus allen Gates
ausgenommen (`.gitignore` `/models/*`, pyright-`exclude`, mypy
``ignore_errors``, ruff-``exclude``, VERBOTEN-``SKIP_DIRS``). Dadurch
blieben drei **echte Laufzeitdefekte** unbemerkt:

1. ``CLAP.audio_infer`` nutzt ``output_dict[key]``, ohne ``key`` als
   Parameter zu führen — die dokumentierte Aufrufkonvention
   (``evaluate/eval_dcase.py``: ``audio_infer(audio, hopsize=..., key="embedding", ...)``)
   verlangt ``key``. Ergebnis: ``NameError``.
2. ``hopsize = min(hopsize, audio_len)`` mit ``hopsize is None`` —
   ``min(None, int)`` ist ein ``TypeError``.
3. ``convert_weights_to_fp16`` schreibt ``attr.data`` auf
   ``text_projection``, das in den Zweigen bert/roberta/bart ein
   ``nn.Sequential`` ist (kein ``.data``) — ``AttributeError``.

Die Defekte sind damit **keine** Editor-Kosmetik: zwei der drei sind
Ausnahmen zur Laufzeit, einer (1) sogar in einem Pfad, der laut
Dokumentation aufgerufen wird.

Dieser Guard prüft den Clone als **Quelltext-Invariante** (läuft überall,
kein Modell/Gewicht nötig) und überspringt sich sauber, wenn der Clone
nicht vorhanden ist — dasselbe Idiom wie
``tests/unit/test_laion_clap_onnx_guard.py`` (§v10.761).
"""

from __future__ import annotations

import ast
import builtins
import symtable
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_MODEL_PY = _PROJECT_ROOT / "models" / "clap" / "src" / "laion_clap" / "clap_module" / "model.py"

_CLONE_MISSING = not _MODEL_PY.exists()
_SKIP_REASON = "lokaler LAION-CLAP-Clone (models/clap/src) nicht vorhanden — Guard übersprungen"

# Vom Import-System bereitgestellte Modul-Attribute: referenziert, aber nicht
# in der symtable-Symboltabelle und keine Builtins (z. B. ``__file__`` in
# ``Path(__file__)``). Keine Defekte.
_MODULE_DUNDERS = frozenset(
    {
        "__file__",
        "__name__",
        "__doc__",
        "__spec__",
        "__loader__",
        "__package__",
        "__builtins__",
        "__debug__",
    }
)


@pytest.fixture(scope="module")
def clap_source() -> str:
    if _CLONE_MISSING:
        pytest.skip(_SKIP_REASON)
    return _MODEL_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def clap_tree(clap_source: str) -> ast.Module:
    return ast.parse(clap_source, filename=str(_MODEL_PY))


def _undefined_names(source: str, filename: str) -> list[str]:
    """Findet referenzierte Namen ohne Definition im Modul- oder Builtin-Raum.

    Nutzt die stdlib-``symtable`` (keine Zusatzabhängigkeit). Gemeldet wird
    ein Name nur, wenn er referenziert, aber weder zugewiesen, noch Parameter,
    noch ``free`` (aus umschließendem Funktionsscope), noch importiert, noch
    im Modul-Globalraum, noch ein Builtin und auch kein modul-immanenter
    Dunder ist — genau die Signatur des ``key``-Defekts.
    """
    top = symtable.symtable(source, filename, "exec")
    module_names = {sym.get_name() for sym in top.get_symbols()}
    problems: list[str] = []

    def walk(table: symtable.SymbolTable) -> None:
        for sym in table.get_symbols():
            name = sym.get_name()
            if not sym.is_referenced():
                continue
            if sym.is_assigned() or sym.is_parameter() or sym.is_free() or sym.is_imported():
                continue
            if name in module_names or name in _MODULE_DUNDERS or hasattr(builtins, name):
                continue
            problems.append(f"{table.get_name()}::{name}")
        for child in table.get_children():
            walk(child)

    walk(top)
    return problems


def _func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Funktion {name!r} im vendored CLAP-Modul nicht gefunden")


class TestKeineUndefiniertenNamen:
    """Defektklasse 1: Referenz ohne Definition (NameError zur Laufzeit)."""

    def test_keine_undefinierten_namen_im_modul(self, clap_source: str) -> None:
        problems = _undefined_names(clap_source, str(_MODEL_PY))
        assert not problems, "Undefinierte Namen im vendored CLAP-Modul (Laufzeit-NameError): " + ", ".join(problems)


class TestGuardSelbsttest:
    """§G176 (copilot-instructions.md): Der Wächter muss nachweislich auslösen.

    Ein Guard ohne erreichbare Aktivierungsbedingung ist toter Code. Hier wird
    die historische Defektform (``key`` ohne Parameter) synthetisch injiziert
    und die Erkennung erzwungen.
    """

    def test_erkennt_defekten_key_selbstbezug(self) -> None:
        buggy = (
            "def audio_infer(self, audio, hopsize=None, device=None):\n"
            "    output_dict = {}\n"
            "    output_dict[key] = self.encode_audio(audio)[key]\n"
            "    return output_dict\n"
        )
        problems = _undefined_names(buggy, "<injected>")
        assert any(p.endswith("::key") for p in problems), f"Wächter erkennt den key-Defekt nicht: {problems}"

    def test_sauberes_pendant_bleibt_still(self) -> None:
        fixed = (
            "def audio_infer(self, audio, hopsize=None, key='embedding', device=None):\n"
            "    output_dict = {}\n"
            "    output_dict[key] = self.encode_audio(audio)[key]\n"
            "    return output_dict\n"
        )
        assert _undefined_names(fixed, "<injected>") == []


class TestAudioInferVertrag:
    """Defektklassen 1+2: Aufrufkonvention von ``audio_infer``."""

    def test_audio_infer_fuehrt_key_parameter(self, clap_tree: ast.Module) -> None:
        fn = _func(clap_tree, "audio_infer")
        args = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
        assert "key" in args, (
            "audio_infer muss den Parameter 'key' führen — der Rumpf nutzt "
            "output_dict[key] und die dokumentierte Aufrufkonvention "
            'übergibt key="embedding" (sonst NameError).'
        )

    def test_audio_infer_hat_keinen_hopsize_selbstbezug(self, clap_source: str) -> None:
        assert "min(hopsize" not in clap_source, (
            "hopsize = min(hopsize, audio_len) mit hopsize=None ist ein TypeError; "
            "der Default muss die Clip-Fensterweite verwenden."
        )


class TestFp16ProjektionsGuard:
    """Defektklasse 3: ``.data`` auf Nicht-Tensoren."""

    def test_convert_weights_guardet_projektionen(self, clap_source: str) -> None:
        fn = _func(ast.parse(clap_source), "convert_weights_to_fp16")
        segment = ast.get_source_segment(clap_source, fn) or ""
        assert "isinstance(attr, torch.Tensor)" in segment, (
            "convert_weights_to_fp16 darf attr.data nur auf Tensoren schreiben — "
            "text_projection ist in bert/roberta/bart ein nn.Sequential "
            "(AttributeError)."
        )


class TestRoBERTaDimensionsPatch:
    """Dokumentierter Produktionsbefund (2026-10-06), den der Patch adressiert.

    Ohne checkpoint-kompatible Dimensionen verwirft ``load_state_dict`` die
    trainierten Positions-Embeddings stillschweigend (3 Tensoren) und der
    Text-Turm rechnet mit Zufallspositionen.
    """

    def test_roberta_dimensionen_checkpoint_kompatibel(self, clap_source: str) -> None:
        assert "max_position_embeddings" in clap_source
        assert "type_vocab_size" in clap_source
        assert "514" in clap_source, (
            "Die checkpoint-kompatiblen RoBERTa-Dimensionen (514 Positions- "
            "Embeddings) müssen im roberta-Zweig erzwungen werden."
        )
