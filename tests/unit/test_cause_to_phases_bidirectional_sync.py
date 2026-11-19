from __future__ import annotations

"""V12 [RELEASE_MUST] — CAUSE_TO_PHASES bidirektionale Sync-Invariante (§2.59)

Spec: jede Ursache in CAUSES muss einen CAUSE_TO_PHASES-Eintrag haben,
und jeder CAUSE_TO_PHASES-Schlüssel muss in CAUSES stehen.
CausalDefectReasoner iteriert ausschließlich über CAUSES (Bayes-Loop) —
orphaned CAUSE_TO_PHASES-Schlüssel werden nie gefunden; fehlende C2P-Einträge
bedeuten, dass die Ursache keine Phasen aktiviert.

Diese Tests werden im CI via pytest ausgeführt UND im aurik_verboten_linter.py
als V12-Modul-Level-Check referenziert.
"""


import ast
from pathlib import Path

import pytest

_CDR_PATH = Path(__file__).resolve().parents[2] / "backend" / "core" / "causal_defect_reasoner.py"


def _parse_causal_defect_reasoner() -> tuple[list[str], dict[str, list[str]]]:
    """Parse CAUSES and CAUSE_TO_PHASES via AST (no import side-effects)."""
    src = _CDR_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_CDR_PATH))

    causes: list[str] = []
    c2p: dict[str, list[str]] = {}

    for node in ast.walk(tree):
        # CAUSES = ["tape_dropout", ...] (simple list assignment)
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "CAUSES":
                    if isinstance(node.value, ast.List):
                        causes = [
                            e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)
                        ]
        # CAUSE_TO_PHASES: dict[str, list[str]] = {...} (annotated assignment)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "CAUSE_TO_PHASES":
                if node.value and isinstance(node.value, ast.Dict):
                    c2p = {
                        k.value: [
                            v.value
                            for v in (val.elts if isinstance(val, ast.List) else [])
                            if isinstance(v, ast.Constant) and isinstance(v.value, str)
                        ]
                        for k, val in zip(node.value.keys, node.value.values)
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)
                    }

    return causes, c2p


@pytest.fixture(scope="module")
def causal_data() -> tuple[list[str], dict[str, list[str]]]:
    return _parse_causal_defect_reasoner()


@pytest.mark.unit
class TestCauseToPhasesBidirectionalSync:
    """V12: Bidirektionale CAUSES ↔ CAUSE_TO_PHASES Invariante."""

    def test_causal_defect_reasoner_file_exists(self) -> None:
        assert _CDR_PATH.exists(), f"causal_defect_reasoner.py not found at {_CDR_PATH}"

    def test_causes_not_empty(self, causal_data: tuple) -> None:
        causes, _ = causal_data
        assert len(causes) > 0, "CAUSES list ist leer — AST-Parser-Fehler?"

    def test_cause_to_phases_not_empty(self, causal_data: tuple) -> None:
        _, c2p = causal_data
        assert len(c2p) > 0, "CAUSE_TO_PHASES dict ist leer — AST-Parser-Fehler?"

    def test_causes_count_plausible(self, causal_data: tuple) -> None:
        """Mindestens 36 Ursachen laut Spec §2.4."""
        causes, _ = causal_data
        assert len(causes) >= 36, f"Zu wenige CAUSES ({len(causes)}) — Spec §2.4 fordert ≥ 36 Ursachen"

    def test_all_causes_have_c2p_entry(self, causal_data: tuple) -> None:
        """V12a: Jede CAUSE muss einen CAUSE_TO_PHASES-Eintrag haben.

        CausalDefectReasoner iteriert über CAUSES (Bayes-Loop §2.59):
        eine Ursache ohne C2P-Eintrag aktiviert nie eine Phase.
        """
        causes, c2p = causal_data
        missing = sorted(set(causes) - set(c2p))
        assert not missing, (
            f"V12a [RELEASE_MUST] §2.59: {len(missing)} CAUSE(S) ohne CAUSE_TO_PHASES-Eintrag "
            f"(werden nie als Phase aktiviert):\n  {missing}\n"
            "Fix: CAUSE_TO_PHASES bidirektional ergänzen."
        )

    def test_all_c2p_keys_are_in_causes(self, causal_data: tuple) -> None:
        """V12b: Jeder CAUSE_TO_PHASES-Schlüssel muss in CAUSES stehen.

        Orphaned CAUSE_TO_PHASES-Schlüssel werden vom Bayes-Loop nie gefunden
        (er iteriert ausschließlich über CAUSES).  Dies ist toter Code.
        """
        causes, c2p = causal_data
        orphaned = sorted(set(c2p) - set(causes))
        assert not orphaned, (
            f"V12b [RELEASE_MUST] §2.59: {len(orphaned)} CAUSE_TO_PHASES-Schlüssel "
            f"ohne CAUSES-Gegenstück (toter Code, wird nie vom Bayes-Loop gefunden):\n  {orphaned}\n"
            "Fix: CAUSES-Eintrag ergänzen oder CAUSE_TO_PHASES-Schlüssel entfernen."
        )

    def test_counts_equal(self, causal_data: tuple) -> None:
        """Schnell-Check: Längen müssen übereinstimmen."""
        causes, c2p = causal_data
        assert len(causes) == len(c2p), (
            f"CAUSES ({len(causes)}) ≠ CAUSE_TO_PHASES keys ({len(c2p)}) — bidirektionale Sync verletzt (§2.59)."
        )

    def test_each_c2p_entry_has_at_least_one_phase(self, causal_data: tuple) -> None:
        """Jede Ursache sollte mindestens eine Phase aktivieren können.

        Ausnahme: `soft_saturation` ist bewusst leer (§0a Vintage Aesthetics — BEWAHREN,
        kein destruktiver Eingriff). Weitere begründete Ausnahmen können hier gelistet werden.
        """
        # Ursachen, die bewusst keine Phasen aktivieren (§0a BEWAHREN-Semantik)
        _INTENTIONALLY_EMPTY: frozenset[str] = frozenset({"soft_saturation"})

        _, c2p = causal_data
        empty = [cause for cause, phases in c2p.items() if not phases and cause not in _INTENTIONALLY_EMPTY]
        assert not empty, (
            f"{len(empty)} CAUSE_TO_PHASES-Einträge ohne Phasen (und nicht in BEWAHREN-Ausnahmen): {empty}\n"
            "Fix: Mindestens eine Phase eintragen oder Ursache in _INTENTIONALLY_EMPTY aufnehmen."
        )
