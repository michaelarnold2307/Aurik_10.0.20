"""Wohlklang-Vertrag-Gate (§v10.802/§G8/§G9 copilot-instructions.md) — Tests.

Prüft die Regeln des Gates über **synthetische** Vertragstexte (Regel-Nachweis)
und über den **echten** Vertrag (Bestands-Nachweis). Ohne diese Tests wäre das
Gate unbelegt — und ein unbelegtes Gate ist genau der Defekt, den es verhindern
soll.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import scripts.wohlklang_gate as gate

_REPO = Path(__file__).resolve().parents[2]

_HEADER = (
    "| ID | Stufe | Schalter (Datei::Name) | Soll | ≥3 echte Songs | "
    "Produktionspfad | Baseline | Blindes A/B | Budget | Status |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
)
# Echter Schalter im Produktionscode (Literal `use_df_musik: bool = True`).
_TRUE_SWITCH = "backend/core/music_model_flags.py::use_df_musik"
_FALSE_SWITCH = "backend/core/music_model_flags.py::use_bw_v5"
_REAL_PATH = "backend/core/music_model_flags.py"


def _row(*, rid: str = "W-1", switch: str = _TRUE_SWITCH, soll: str = "AN", status: str, songs: str = "3") -> str:
    return (
        f"| {rid} | Stufe | `{switch}` | {soll} | {songs} | Produktionsweg | Baseline | Blind | 1,0 × RT | {status} |\n"
    )


def _contract(*rows: str) -> str:
    return _HEADER + "".join(rows)


def _errors(rows: str, version: str = "10.6.0") -> list[gate.Finding]:
    return [f for f in gate.check_contract(gate.parse_claims(rows), version) if f.severity == "ERROR"]


def _rules(rows: str, version: str = "10.6.0") -> set[str]:
    return {f.rule for f in _errors(rows, version)}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParser:
    def test_parses_real_contract(self):
        claims = gate.parse_claims((_REPO / ".github" / "WOHLKLANG_CLAIMS.md").read_text(encoding="utf-8"))
        assert len(claims) >= 10
        assert all(gate.ROW_ID_RE.match(c.id) for c in claims)
        assert all(c.switch and "`" not in c.switch for c in claims)

    def test_ignores_header_and_prose(self):
        claims = gate.parse_claims(_contract(_row(status="gesperrt", soll="AUS", switch=_FALSE_SWITCH)))
        assert len(claims) == 1
        assert claims[0].id == "W-1"


# ---------------------------------------------------------------------------
# E1–E6 (Regel-Nachweis)
# ---------------------------------------------------------------------------


class TestRules:
    def test_valid_locked_row_passes(self):
        assert _errors(_contract(_row(status="gesperrt", soll="AUS", switch=_FALSE_SWITCH))) == []

    def test_e1_duplicate_id(self):
        rows = _contract(
            _row(status="gesperrt", soll="AUS", switch=_FALSE_SWITCH),
            _row(status="gesperrt", soll="AUS", switch=_FALSE_SWITCH),
        )
        assert "E1" in _rules(rows)

    def test_e1_wrong_cell_count(self):
        rows = _contract("| W-1 | Stufe | `x.py::y` | AN | 3 |\n")
        assert "E1" in _rules(rows)

    def test_e1_unknown_status(self):
        assert "E1" in _rules(_contract(_row(status="vielleicht", soll="AUS", switch=_FALSE_SWITCH)))

    def test_e2_missing_cited_path(self):
        row = _row(status="gesperrt", soll="AUS", switch=_FALSE_SWITCH, songs="3 (`docs/gibt-es-nicht.md`)")
        assert "E2" in _rules(_contract(row))

    def test_e2_existing_path_ok(self):
        row = _row(status="gesperrt", soll="AUS", switch=_FALSE_SWITCH, songs=f"3 (`{_REAL_PATH}`)")
        assert "E2" not in _rules(_contract(row))

    def test_e3_activated_with_offen_fails(self):
        row = _row(status="aktiviert", soll="AN", songs="OFFEN", switch=_TRUE_SWITCH)
        assert "E3" in _rules(_contract(row))

    def test_e3_activated_requires_version_and_budget(self):
        row = f"| W-1 | Stufe | `{_TRUE_SWITCH}` | AN | 3 | Weg | Baseline | Blind | ohne RT-Angabe | aktiviert |\n"
        assert "E3" in _rules(_contract(row))

    def test_e3_activated_complete_passes(self):
        row = f"| W-1 | Stufe | `{_TRUE_SWITCH}` | AN | 3 | Weg | eingefroren 10.6.0 | Blindpaar | 1,0 × RT | aktiviert |\n"
        assert "E3" not in _rules(_contract(row))

    def test_e4_ausnahme_needs_grund_and_review(self):
        assert "E4" in _rules(_contract(_row(status="ausnahme (Grund: zu kurz)", soll="AN")))
        assert "E4" in _rules(_contract(_row(status="ausnahme (Grund: ausreichend lange Begründung hier)", soll="AN")))
        ok = "ausnahme (Grund: ausreichend lange Begründung hier; Review: nächster Release)"
        assert "E4" not in _rules(_contract(_row(status=ok, soll="AN")))

    def test_e5_drift_code_vs_contract(self):
        """Der Kern des Gates: Code sagt True, Vertrag sagt AUS ⇒ Fehler."""
        assert "E5" in _rules(_contract(_row(status="gesperrt", soll="AUS", switch=_TRUE_SWITCH)))

    def test_e5_unknown_switch(self):
        assert "E5" in _rules(
            _contract(_row(status="gesperrt", soll="AUS", switch="backend/core/music_model_flags.py::gibt_es_nicht"))
        )

    def test_e6_gesperrt_muss_aus_sein(self):
        assert "E6" in _rules(_contract(_row(status="gesperrt", soll="AN", switch=_TRUE_SWITCH)))


# ---------------------------------------------------------------------------
# P1/P2 — Phantom-Referenzen
# ---------------------------------------------------------------------------


class TestPhantom:
    def test_p1_harnesses_exist_and_are_phantom_free(self):
        findings, _ = gate.check_harnesses()
        p1_errors = [f for f in findings if f.rule == "P1" and f.severity == "ERROR"]
        assert p1_errors == [], [f.message for f in p1_errors]

    def test_p1_detects_phantom_in_harness(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Ein Harness, der ein nicht existentes Artefakt zitiert, muss auffallen."""
        fake = tmp_path / "harness.py"
        fake.write_text("P = 'models/bigvgan/gibt_es_nicht.onnx'\n", encoding="utf-8")
        monkeypatch.setattr(gate, "EVIDENCE_HARNESSES", (str(fake),))
        findings, _ = gate.check_harnesses()
        assert any(f.rule == "P1" and f.severity == "ERROR" for f in findings)

    def test_model_paths_skipped_without_model_tree(self, monkeypatch: pytest.MonkeyPatch):
        """Ohne models/-Baum (nackter Checkout) dürfen Modellpfade nicht blocken."""
        monkeypatch.setattr(gate, "_model_tree_present", lambda: False)
        assert gate._path_missing("models/gibt/es/nicht.onnx") is False

    def test_p2_is_report_only(self):
        findings, count = gate.check_harnesses()
        assert isinstance(count, int)
        assert all(f.severity == "INFO" for f in findings if f.rule == "P2")


# ---------------------------------------------------------------------------
# P1/P2 — Prosa ist kein Code (Befund D-K3-5, 2026-10-07)
# ---------------------------------------------------------------------------


class TestProseIsNotCode:
    """Das Gate prüft Verdrahtung, nicht Beschreibung.

    Auslöser: P2 meldete 53 „nicht existente ``models/…``-Referenzen in
    Produktionscode"; die Stichprobe zeigte, dass 27 davon **Prosa** waren. Ein
    Pfad in einem Docstring ist keine Verdrahtung — er darf weder den Bericht
    aufblähen noch P1 (fail-closed) blockieren.
    """

    def test_comment_reference_is_not_code(self):
        src = "x = 1  # lädt models/phantom/kommentar.onnx\n"
        assert gate._scan_code_paths(src) == []

    def test_docstring_reference_is_not_code(self):
        src = 'def f():\n    """Modell: models/phantom/docstring.onnx (722 MB)."""\n    return 1\n'
        assert gate._scan_code_paths(src) == []

    def test_orphaned_statement_string_is_not_code(self):
        """Rückbau-Rest: der zweite Statement-String einer Klasse ist Prosa.

        Produktionsbefund ``plugins/apollo_phase0_integration.py``: Die Klasse
        ``ResembleEnhanceGuard`` trägt ihren echten Docstring **und** darunter den
        zurückgelassenen Docstring der in §v10.19 entfernten Methode. Nur
        ``body[0]`` zu prüfen ließ diesen String als Code durchgehen.
        """
        src = (
            "class Guard:\n"
            '    """Echter Docstring."""\n'
            "\n"
            '    """Verwaister Docstring: models/phantom/verwaist.onnx"""\n'
            "\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
        )
        assert gate._scan_code_paths(src) == []

    def test_functional_string_reference_is_code(self):
        """Ein Pfad in einem Aufruf ist die Verdrahtung — er MUSS bleiben."""
        src = 'model = load(Path("models/real/wiring.onnx"))\n'
        assert gate._scan_code_paths(src) == ["models/real/wiring.onnx"]

    def test_assignment_string_reference_is_code(self):
        src = 'CHECKPOINT = "models/train/best_model_v3.pt"\n'
        assert gate._scan_code_paths(src) == ["models/train/best_model_v3.pt"]

    def test_url_suffix_is_not_a_local_path(self):
        """``.../all_public_uvr_models/model_x.ckpt`` beginnt mitten im Wort."""
        src = 'URL = "https://h/…/all_public_uvr_models/model_bs_roformer_ep_317.ckpt"\n'
        assert gate._scan_code_paths(src) == []

    def test_blank_prose_preserves_positions(self):
        """Zeilen/Spalten bleiben erhalten — sonst sind Befunde nicht auffindbar."""
        src = 'x = 1  # models/phantom/a.onnx\n\ndef f():\n    """models/phantom/b.onnx"""\n'
        blanked = gate._blank_prose(src)
        assert len(blanked.splitlines()) == len(src.splitlines())
        assert blanked.splitlines()[0].startswith("x = 1 ")
        # Die Prosa wird zu Leerzeichen gleicher Länge (Position bleibt, Inhalt weg).
        assert blanked.splitlines()[3] == " " * len(src.splitlines()[3])
        assert "models/" not in blanked

    def test_blank_prose_falls_back_conservatively_on_syntax_error(self):
        """Konservativ: nicht parsebarer Text bleibt unverändert (mehr Befunde, nie weniger)."""
        broken = "def f(:\n    models/phantom/broken.onnx\n"
        assert gate._blank_prose(broken) == broken

    def test_p1_ignores_prose_reference_in_harness(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Ein Harness darf ein Phantom-Artefakt in Prosa erwähnen (Doku) — kein Fail."""
        harness = tmp_path / "harness.py"
        harness.write_text(
            'def run():\n    """Vergleich gegen models/bigvgan/nur_erwaehnt.onnx."""\n    return 1\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(gate, "EVIDENCE_HARNESSES", (str(harness),))
        findings, _ = gate.check_harnesses()
        assert [f.message for f in findings if f.rule == "P1" and f.severity == "ERROR"] == []

    def test_p1_still_detects_functional_phantom_in_harness(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Der Fail-closed-Kern bleibt scharf: Code-Verdrahtung auf ein Phantom."""
        harness = tmp_path / "harness.py"
        harness.write_text('P = "models/bigvgan/gibt_es_nicht.onnx"\n', encoding="utf-8")
        monkeypatch.setattr(gate, "EVIDENCE_HARNESSES", (str(harness),))
        monkeypatch.setattr(gate, "_path_missing", lambda _r: True)
        findings, _ = gate.check_harnesses()
        assert any(f.rule == "P1" and f.severity == "ERROR" for f in findings)

    def test_p2_reports_functional_but_not_prose_end_to_end(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """End-to-End: P2 über einen künstlichen Produktionsbaum.

        Pinnt den **Mechanismus** (Prosa raus, Verdrahtung rein) statt einer
        Momentaufnahme der heutigen Trefferzahlen — eine Liste wäre brüchig: sie
        würde fehlschlagen, sobald ein deklariertes Artefakt tatsächlich
        trainiert wird.
        """
        root = tmp_path / "backend"
        root.mkdir()
        (root / "mod.py").write_text(
            '"""Doku: models/phantom/nur_prosa.onnx."""\n'
            "# Kommentar: models/phantom/auch_prosa.onnx\n"
            'ECHT = "models/phantom/verdrahtet.onnx"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(gate, "ROOT", tmp_path)
        monkeypatch.setattr(gate, "PRODUCTION_ROOTS", ("backend",))
        monkeypatch.setattr(gate, "EVIDENCE_HARNESSES", ())
        monkeypatch.setattr(gate, "_path_missing", lambda ref: "phantom" in ref)

        findings, count = gate.check_harnesses()
        hits = [f.message for f in findings if f.rule == "P2" and " → " in f.message]
        assert hits == ["backend/mod.py → models/phantom/verdrahtet.onnx"], hits
        assert count == 1


# ---------------------------------------------------------------------------
# Bestands-Nachweis: der echte Vertrag ist konsistent
# ---------------------------------------------------------------------------


class TestRealContract:
    def test_real_contract_has_no_errors(self):
        text = (_REPO / ".github" / "WOHLKLANG_CLAIMS.md").read_text(encoding="utf-8")
        claims = gate.parse_claims(text)
        errors = [f for f in gate.check_contract(claims, gate.current_version()) if f.severity == "ERROR"]
        assert errors == [], [f.message for f in errors]

    def test_no_row_is_silently_activated(self):
        """Keine Zeile darf `aktiviert` sein, ohne dass alle fünf Belege vorliegen."""
        text = (_REPO / ".github" / "WOHLKLANG_CLAIMS.md").read_text(encoding="utf-8")
        for claim in gate.parse_claims(text):
            if claim.status.split(" ")[0] == "aktiviert":
                assert not any("OFFEN" in c.upper() for c in claim.evidence_cells), claim.id

    def test_version_is_readable_without_import(self):
        """Die Version wird statisch gelesen (kein Import im Pre-Commit-Pfad)."""
        version = gate.current_version()
        assert re.fullmatch(r"\d+\.\d+\.\d+", version), version
