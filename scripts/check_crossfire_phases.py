"""§0a Crossfire-Guard — CAUSE_TO_PHASES darf keine Restoration-verbotenen Phasen enthalten.

Verhindert Regression: phase_21_exciter, phase_35_multiband_compression, phase_42_vocal_enhancement
wurden in BUG-FIX v10.0.0 aus CAUSE_TO_PHASES entfernt. Dieser Guard stellt sicher, dass sie
nicht wieder eingebaut werden.

Wird von pre-commit auf backend/core/causal_defect_reasoner.py ausgeführt.

Exit-Codes:
    0 = keine §0a-Verstöße (Crossfire-Guard erfüllt)
    1 = §0a-Verletzung gefunden (phase_21/35/42 in CAUSE_TO_PHASES)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# §0a (copilot-instructions.md): Diese Phasen sind in Restoration-Modus absolut verboten.
# Bidirektionales Verbot: auch CausalDefectReasoner darf sie nie vorschlagen.
_FORBIDDEN_IN_CAUSE_TO_PHASES: frozenset[str] = frozenset(
    {
        "phase_21_exciter",
        "phase_35_multiband_compression",
        "phase_42_vocal_enhancement",
    }
)

_TARGET_FILE = "causal_defect_reasoner.py"


def check(path: Path) -> list[tuple[int, str, str]]:
    """Gibt Liste von (lineno, phase_name, cause_key) zurück für alle §0a-Verletzungen."""
    if path.name != _TARGET_FILE:
        return []
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError) as exc:
        print(f"ERROR: Konnte {path} nicht parsen: {exc}", file=sys.stderr)
        return []

    violations: list[tuple[int, str, str]] = []

    for node in ast.walk(tree):
        # Suche CAUSE_TO_PHASES Dict-Literal (AnnAssign oder Assign)
        dict_node: ast.Dict | None = None
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "CAUSE_TO_PHASES":
                if node.value and isinstance(node.value, ast.Dict):
                    dict_node = node.value
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "CAUSE_TO_PHASES":
                    if isinstance(node.value, ast.Dict):
                        dict_node = node.value

        if dict_node is None:
            continue

        # Iteriere über cause → [phase_list] Paare
        for cause_key_node, phase_list_node in zip(dict_node.keys, dict_node.values):
            if not isinstance(cause_key_node, ast.Constant):
                continue
            cause_key = str(cause_key_node.value)

            # phase_list kann List-Literal oder andere Strukturen sein
            phase_names: list[tuple[str, int]] = []
            if isinstance(phase_list_node, ast.List):
                for elt in phase_list_node.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        phase_names.append((elt.value, elt.lineno))
            elif isinstance(phase_list_node, ast.Constant) and isinstance(phase_list_node.value, str):
                phase_names.append((phase_list_node.value, phase_list_node.lineno))

            for phase_name, lineno in phase_names:
                if phase_name in _FORBIDDEN_IN_CAUSE_TO_PHASES:
                    violations.append((lineno, phase_name, cause_key))

    return violations


def main() -> int:
    """Prüft causal_defect_reasoner.py auf §0a-Crossfire-Verstöße.

    Returns:
        0 wenn keine Verstöße gefunden, 1 wenn §0a-Verletzungen vorliegen.
    """
    # pre-commit übergibt keine Dateipfade (pass_filenames: false) — Pfad ist fest
    workspace_root = Path(__file__).parent.parent
    target = workspace_root / "backend" / "core" / _TARGET_FILE

    if not target.exists():
        print(f"WARNING: {target} nicht gefunden — Crossfire-Guard übersprungen", file=sys.stderr)
        return 0

    violations = check(target)

    if not violations:
        print(f"✓ §0a Crossfire-Guard: keine Verstöße in {_TARGET_FILE}")
        return 0

    print(f"\n§0a CROSSFIRE-VERLETZUNG in {target}:", file=sys.stderr)
    print("─" * 70, file=sys.stderr)
    for lineno, phase, cause in violations:
        print(
            f"  Zeile {lineno:4d}: '{phase}' steht in CAUSE_TO_PHASES['{cause}']",
            file=sys.stderr,
        )
        print(
            "           §0a verbietet phase_21/35/42 in CAUSE_TO_PHASES absolut.",
            file=sys.stderr,
        )
        print(
            "           (Restoration-verbotene Phase; BUG-FIX v10.0.0 §0a Crossfire-Invariante)",
            file=sys.stderr,
        )
    print("─" * 70, file=sys.stderr)
    print(f"  {len(violations)} Verstoß/Verstöße gefunden. Commit blockiert.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
