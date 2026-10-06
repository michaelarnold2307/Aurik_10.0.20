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


class _ShapeOnly:
    """Minimaler Tensor-Ersatz — ``classify_checkpoint_params`` liest nur ``.shape``."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class TestCheckpointParamClassification:
    """Produktionsbefund 2026-10-06: ``position_ids``-Falsch-Alarm (§V6 copilot-instructions.md).

    ``text_branch.embeddings.position_ids`` ist ein nicht-parametrischer
    Index-Buffer (arange), den der Checkpoint versionsbedingt mitführt, den das
    aktuelle Modell aber als non-persistenten Buffer nicht in ``state_dict()``
    führt. Er darf NICHT als „nicht übernommen → Zufallsinitialisierung“ gemeldet
    werden — der Falsch-Alarm verdeckt echte Parameter-Verluste.
    """

    def test_benign_position_ids_is_not_a_drop(self) -> None:
        from plugins.laion_clap_plugin import classify_checkpoint_params

        state = {
            "text_branch.embeddings.position_ids": _ShapeOnly((1, 5)),
            "text_branch.embeddings.word_embeddings.weight": _ShapeOnly((4, 3)),
        }
        model_state = {"text_branch.embeddings.word_embeddings.weight": _ShapeOnly((4, 3))}
        kept, dropped, benign = classify_checkpoint_params(state, model_state)
        assert benign == ["text_branch.embeddings.position_ids"]
        assert dropped == [], f"position_ids darf keine Degradierung erzeugen: {dropped}"
        assert set(kept) == {"text_branch.embeddings.word_embeddings.weight"}

    def test_real_loss_is_still_reported(self) -> None:
        from plugins.laion_clap_plugin import classify_checkpoint_params

        state = {"text_branch.encoder.layer.0.attention.weight": _ShapeOnly((2, 2))}
        kept, dropped, benign = classify_checkpoint_params(state, {})
        assert benign == []
        assert kept == {}
        assert dropped and "nicht im Modell" in dropped[0]

    def test_shape_conflict_is_reported(self) -> None:
        from plugins.laion_clap_plugin import classify_checkpoint_params

        state = {"a.weight": _ShapeOnly((3, 3))}
        model_state = {"a.weight": _ShapeOnly((2, 2))}
        kept, dropped, benign = classify_checkpoint_params(state, model_state)
        assert kept == {} and benign == []
        assert dropped and "Checkpoint" in dropped[0]


class TestZeroShotAbsoluteScores:
    """Regressions-Guard: Zero-Shot-Scores sind ABSOLUTE Kosinus-Ähnlichkeiten.

    Befund 2026-10-06: Der Score war der Rest eines gemeinsamen 46-Tag-Softmax
    (Temperatur ×100). Der sättigte, sodass jede Query, die NICHT der Argmax war,
    auf ~0.0 kollabierte — und er ist über Aufrufe nicht vergleichbar (wechselnder
    Nenner), obwohl `genre_classifier` positive und negative Prompt-Sätze saldiert.
    """

    @staticmethod
    def _plugin_with_fake_model(query_cos: float):
        import numpy as np

        from plugins import laion_clap_plugin as L

        dim = L.LAIONCLAPPlugin.EMBEDDING_DIM
        e0 = np.zeros(dim, dtype=np.float32)
        e0[0] = 1.0
        e1 = np.zeros(dim, dtype=np.float32)
        e1[1] = 1.0
        custom = "mein schlager"

        class _FakeClap:
            def get_audio_embedding_from_data(self, x, use_tensor=False):
                return e0[np.newaxis, :].copy()

            def get_text_embedding(self, texts, use_tensor=False):
                rows = []
                for i, t in enumerate(texts):
                    if t == custom:
                        # Kosinus = query_cos gegen e0
                        rows.append(query_cos * e0 + float(np.sqrt(max(0.0, 1.0 - query_cos**2))) * e1)
                    elif i == 0:
                        rows.append(e0.copy())  # Argmax-Standard-Tag (Kosinus 1.0)
                    else:
                        rows.append(e1.copy())  # orthogonal (Kosinus 0.0)
                return np.stack(rows)

        plugin = L.LAIONCLAPPlugin.__new__(L.LAIONCLAPPlugin)
        plugin._clap_model = _FakeClap()
        return plugin, custom

    def test_query_score_is_absolute_cosine_not_softmax_residual(self) -> None:
        import numpy as np

        plugin, custom = self._plugin_with_fake_model(0.5)
        res = plugin._tag_clap_pt(np.zeros(48_000, dtype=np.float32), 48_000, [custom])
        # Nicht der Argmax (ein Standard-Tag hat Kosinus 1.0) — trotzdem 0.5, NICHT 0.0.
        assert res.custom_scores[custom] == pytest.approx(0.5, abs=1e-3)

    def test_query_score_is_monotone_and_clipped(self) -> None:
        import numpy as np

        plugin_lo, custom = self._plugin_with_fake_model(0.2)
        plugin_hi, _ = self._plugin_with_fake_model(0.7)
        lo = plugin_lo._tag_clap_pt(np.zeros(48_000, dtype=np.float32), 48_000, [custom]).custom_scores[custom]
        hi = plugin_hi._tag_clap_pt(np.zeros(48_000, dtype=np.float32), 48_000, [custom]).custom_scores[custom]
        assert 0.0 <= lo < hi <= 1.0


class TestPerCategoryTagScoring:
    """Befund 2026-10-06: gemeinsamer 43-Tag-Softmax verwässerte die Massen.

    Folge: ``top_instruments`` (Schwelle 0.4) war immer leer und das CLAP-Genre-
    Gate (≥ 0.35) feuerte nie — der Genre-Argmax wurde Rauschen (rnb↔opera).
    Korrektur: getrennter Softmax je Kategorie.
    """

    def test_instruments_and_genres_normalized_per_category(self) -> None:
        import numpy as np

        from plugins.laion_clap_plugin import GENRE_TAGS, INSTRUMENT_TAGS, MATERIAL_TAGS, score_tags_by_category

        n = len(INSTRUMENT_TAGS) + len(GENRE_TAGS) + len(MATERIAL_TAGS)
        raw = np.full(n, 0.05, dtype=np.float32)
        raw[0] = 0.6  # Instrument 0 klar dominant
        raw[len(INSTRUMENT_TAGS)] = 0.55  # Genre 0 klar dominant
        instr, genre, _material = score_tags_by_category(raw)
        # Instrumente und Genre sind (je eigene) Entscheidungen → je Summe 1.
        assert abs(sum(instr.values()) - 1.0) < 1e-6
        assert abs(sum(genre.values()) - 1.0) < 1e-6

    def test_material_is_multi_label_not_exclusive(self) -> None:
        """Träger sind NICHT disjunkt — Material muss eine Kette abbilden können.

        Befund 2026-10-06: „Material = tape“ war ein Einzel-Argmax und
        unterschlug die Tonträgerkette (99 % der Restaurierungsfälle).
        """
        import numpy as np

        from plugins.laion_clap_plugin import GENRE_TAGS, INSTRUMENT_TAGS, MATERIAL_TAGS, score_tags_by_category

        n_i, n_g = len(INSTRUMENT_TAGS), len(GENRE_TAGS)
        n = n_i + n_g + len(MATERIAL_TAGS)
        raw = np.zeros(n, dtype=np.float32)
        raw[n_i + n_g + MATERIAL_TAGS.index("vinyl")] = 0.30
        raw[n_i + n_g + MATERIAL_TAGS.index("mp3")] = 0.20
        _i, _g, material = score_tags_by_category(raw)
        # Mehrere Träger gleichzeitig nicht-null (Kette), absolut (NICHT auf 1 normiert).
        assert material["vinyl"] == pytest.approx(0.30, abs=1e-6)
        assert material["mp3"] == pytest.approx(0.20, abs=1e-6)
        assert sum(material.values()) < 1.0

    def test_dominant_tags_clear_operational_gates(self) -> None:
        import numpy as np

        from plugins.laion_clap_plugin import GENRE_TAGS, INSTRUMENT_TAGS, MATERIAL_TAGS, score_tags_by_category

        n = len(INSTRUMENT_TAGS) + len(GENRE_TAGS) + len(MATERIAL_TAGS)
        raw = np.full(n, 0.05, dtype=np.float32)
        raw[0] = 0.6
        raw[len(INSTRUMENT_TAGS)] = 0.55
        instr, genre, _ = score_tags_by_category(raw)
        assert instr[INSTRUMENT_TAGS[0]] >= 0.4  # Instrument-Gate jetzt wirksam
        assert max(genre.values()) >= 0.35  # Genre-Gate jetzt wirksam
        assert genre[GENRE_TAGS[0]] == max(genre.values())

    def test_uniform_category_stays_below_gate(self) -> None:
        import numpy as np

        from plugins.laion_clap_plugin import GENRE_TAGS, INSTRUMENT_TAGS, MATERIAL_TAGS, score_tags_by_category

        n = len(INSTRUMENT_TAGS) + len(GENRE_TAGS) + len(MATERIAL_TAGS)
        raw = np.full(n, 0.2, dtype=np.float32)  # keine Kategorie dominiert
        _, genre, _ = score_tags_by_category(raw)
        # Gleichverteilt ⇒ kein Pseudo-Sieger über dem Gate (kein Fakten-Genre).
        assert max(genre.values()) < 0.35
