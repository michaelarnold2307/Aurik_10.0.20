"""Abgleich-Check: Enum-Typen vs. Evidenz-Harness vs. Phase-Mapper (fail-closed).

Wurzel der Luecke vom 2026-09-24: Drei Enum-Typen (DISTORTION, DROPOUT,
DROPOUT_SPLICE) hatten nie einen Mess-Fall, weil kein automatischer Abgleich
existierte - der Harness lief nur manuell. Dieses Skript erzwingt dauerhaft:

  1. Jeder DefectType hat einen Harness-Fall (Erwartungs-Eintrag in CASES).
  2. Jeder DefectType hat eine PhaseAssignment im Phase-Mapper.
  3. Pfad-Differenz Scanner vs. ai_framework wird als Befund gelistet
     (kein Fail - die Typen sind dort verdrahtet, nur in anderem Pfad).
  4. Dokumentierte Zaehlwerte ("N DefectTypes", "N Kausal-Ursachen") in
     AKTUELLEN Wahrheitsquellen stimmen mit dem Enum bzw. CAUSES ueberein
     (Befund 2026-10-06: Regeln nannten 62 DefectTypes, das Enum hat 65;
     Spec 03/05 nannten gleichzeitig 54/62/49 und widersprachen sich selbst).

Exit 0 = vollstaendig, 1 = mindestens eine Luecke.
Deterministisch und schnell (nur Enum-/Dict-Abgleiche, kein Audio-Scan).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_harness_cases() -> dict:
    sys.argv = ["harness", "noise"]
    spec = importlib.util.spec_from_file_location("harness", str(_REPO / "scripts" / "defect_evidence_harness.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("Harness nicht ladbar")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CASES


#: Dateien, in denen eine Zahl eine Aussage ueber den AKTUELLEN Ist-Stand ist.
#: Bewusst NICHT enthalten (deren alte Zahlen sind Historie, nicht falsch):
#:   - docs/CHANGELOG_HISTORY.md, CHANGELOG.md (Chronik)
#:   - docs/PROJECT_STATUS.md (Meilenstein-Tabelle nennt je Release den damaligen Stand)
#:   - docs/dev/** (Audit-Momentaufnahmen)
#:   - .agents/skills/** (in AGENTS.md §6 als Stubs/veraltete Kopien deklariert)
#:   - die Wachstums-Notizen im Enum selbst ("ergibt 28 DefectTypes") — die
#:     dokumentieren die Entstehungsreihenfolge des Enum, nicht den Endstand.
_DOC_COUNT_SOURCES: tuple[str, ...] = (
    ".github/copilot-instructions.md",
    ".github/GEBOTE.md",
    ".github/ID_REGISTRY.md",
    ".github/specs/03_cognitive_modules.md",
    ".github/specs/05_material_system.md",
    ".github/specs/25_ambience_match_plugin.md",
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "CONTRIBUTING.md",
    "SPEC.md",
    "scripts/gebote_verifier.py",
    "backend/core/causal_defect_reasoner.py",
    "backend/core/surgical_defect_analyzer.py",
    "backend/core/unified_restorer_v3.py",
    "denker/defekt_denker.py",
)

#: Negatives Lookbehind: "§6.3 DefectType" darf NICHT als "3 DefectTypes" gelesen werden.
_DOC_TYPE_RE = re.compile(r"(?<![\w.])(\d+)\s*DefectTypes?\b")
_DOC_CAUSE_RE = re.compile(r"(?<![\w.])(\d+)\s*Kausal-Ursachen\b")


def _check_documented_counts(n_types: int, n_causes: int) -> list[str]:
    """Vergleicht dokumentierte Zaehlwerte mit der Code-Wahrheit (fail-closed).

    Die Meldung nennt Datei und Zeile, damit die Korrektur mechanisch ist und
    keine Zahl per Suchen-und-Ersetzen geraten werden muss (die "62" bezeich-
    nete an verschiedenen Stellen Verschiedenes: DefectTypes ODER Kausal-Ursachen).
    """
    problems: list[str] = []
    for rel in _DOC_COUNT_SOURCES:
        path = _REPO / rel
        if not path.exists():
            problems.append(f"Doku-Zahl: {rel} fehlt")
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in _DOC_TYPE_RE.finditer(line):
                if int(match.group(1)) != n_types:
                    problems.append(f"Doku-Zahl: {rel}:{lineno} nennt {match.group(1)} DefectTypes, Enum hat {n_types}")
            for match in _DOC_CAUSE_RE.finditer(line):
                if int(match.group(1)) != n_causes:
                    problems.append(
                        f"Doku-Zahl: {rel}:{lineno} nennt {match.group(1)} Kausal-Ursachen, CAUSES hat {n_causes}"
                    )
    return problems


def main() -> int:
    from backend.core.defect_scanner import DefectType

    all_types = set(DefectType.__members__.values())
    failures: list[str] = []

    # 1. Harness-Abdeckung
    cases = _load_harness_cases()
    covered = {
        dt for _, _, _, expects in ((n, g, m, e) for fam in cases.values() for n, g, m, e in fam) for dt in expects
    }
    missing_harness = sorted(all_types - covered, key=lambda t: t.value)
    if missing_harness:
        failures.append(f"Harness: ohne Evidenz-Fall: {[t.value for t in missing_harness]}")

    # 2. Phase-Mapper-Abdeckung
    import backend.core.defect_phase_mapper as mapper

    mapped = set()
    for attr in dir(mapper):
        obj = getattr(mapper, attr)
        if isinstance(obj, dict):
            mapped.update(dt for dt in obj if isinstance(dt, DefectType))
    missing_mapper = sorted(all_types - mapped, key=lambda t: t.value)
    if missing_mapper:
        failures.append(f"Phase-Mapper: ohne PhaseAssignment: {[t.value for t in missing_mapper]}")

    # 3. Pfad-Befund (kein Fail): Scanner vs. ai_framework
    import inspect

    from backend.core import defect_scanner as ds

    scanner_src = inspect.getsource(ds.DefectScanner.scan)
    scanner_types = set()
    for member in DefectType:
        # Word-Boundary: "DROPOUT" darf nicht als Praefix von "DROPOUTS" matchen
        if re.search(rf"DefectType\.{re.escape(member.name)}\b", scanner_src):
            scanner_types.add(member)
    framework_only = sorted((all_types - scanner_types), key=lambda t: t.value)
    print(f"Abgleich: {len(covered)}/{len(all_types)} Harness-Faelle, {len(mapped)}/{len(all_types)} Phase-Mapper")
    print(f"Befund (kein Fail): nur im ai_framework-Pfad befuellt: {[t.value for t in framework_only]}")

    # 4. Dokumentierte Zaehlwerte == Code-Wahrheit (aktuelle Quellen, fail-closed)
    from backend.core.causal_defect_reasoner import CAUSES as _CAUSES

    failures.extend(_check_documented_counts(len(all_types), len(_CAUSES)))

    if failures:
        for f in failures:
            print(f"LUECKE: {f}")
        return 1
    print("OK: Enum/Harness/Phase-Mapper vollstaendig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
