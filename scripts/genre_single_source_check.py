#!/usr/bin/env python3
"""Genre-Single-Source-Guard (§G9 (copilot-instructions.md)) — fail-closed.

Befund 2026-10-06 (Tranche 3): Die Genre-Auflösung war über **fünf** parallele
Tabellen verteilt (genre_registry, genre_goal_profile, song_goal_importance,
tonal_reference_profile, perceptual_tuning). T3.1 hat die Alias-Auflösung auf
`backend/core/genre_registry.py` konsolidiert. Dieser Guard hält das fest —
mechanisch, verhaltensneutral:

  R1  Nur `backend/core/genre_registry.py` darf eine Genre-Alias-Tabelle
      definieren (Name enthält »genre« UND »alias«). Eine zweite Alias-Quelle
      ist eine Vertrags-Drift und wird blockiert.
  R2  Jeder Schlüssel der beiden Ziel-Gewichtstabellen
      (`genre_goal_profile._GENRE_PROFILES`,
      `song_goal_importance._GENRE_WEIGHT_PROFILES`) MUSS über
      `genre_registry.normalize_genre()` auf ein kanonisches Genre auflösen.
      Fängt neue Schlüsselräume ab, die an der Registry vorbeilaufen.
  R3  XOR-Invariante über alle kanonischen Genres:
      `goal_profile_key(g) is None`  ⟺  `g in _GOAL_KEY_ABSENT`.
      Ein absichtlich fehlendes Ziel-Profil muss also DEKLARIERT sein — ein
      stilles Fehlen (None ohne Eintrag in der Ausnahmeliste) ist ein Fehler.
  R5  Jeder Ziel-Schlüssel aus den Ziel-Tabellen (`genre_goal_profile`,
      `song_goal_importance`, `goal_budget`) ist ENTWEDER kanonisch
      (`ALL_GOAL_NAMES`) ODER in `GOAL_DIALECT_MAP` deklariert (abgebildet oder
      ausdrücklich offen). T3.4-Befund 2026-10-06: vier Ziel-Dialekte, 12
      Fremdschlüssel — ein NEUER, undeklarierter Dialekt ist ein Fehler.

R4 ist ein **Bericht**, kein Fail: Für Genres, die BEIDE Tabellen führen, werden
die abweichenden Ziel-Gewichts-Zellen ausgegeben. Die beiden Tabellen haben
verschiedene Rollen (ROLLE_A/ROLLE_B unten) — gleich benannte Ziele tragen
teils unterschiedliche Werte. Eine Vereinheitlichung ist eine KLANG-ÄNDERUNG
und darf nur mit A/B-Nachweis + Hörordnungs-Abnahme + Minor-Bump erfolgen
(§v10.802 Versionierungs-Vertrag); der Guard macht die Divergenz sichtbar
(§G8 (copilot-instructions.md) Transparenz), erzwingt sie aber nicht.

Aufruf:  python scripts/genre_single_source_check.py [--quiet]

Exit-Codes: 0 = konform, 1 = Verletzung (fail-closed).
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: Rollen der beiden Ziel-Gewichtstabellen (T3.4-Analyse 2026-10-06).
ROLLE_A = (
    "song_goal_importance._GENRE_WEIGHT_PROFILES — Per-Song-Wichtigkeit (§2.56), "
    "speist goal_weights in PMGG/CIG/GPP/FeedbackChain; Schlüsselraum 17 Genres, "
    "sparse Deltas (fehlende Ziele = 1.0)."
)
ROLLE_B = (
    "genre_goal_profile._GENRE_PROFILES — Genre-Ziel-Profil für goal_budget.py; "
    "Schlüsselraum 10 Genres mit vollständigen 15-Ziel-Vektoren."
)

#: Verzeichnisse, in denen eine zweite Genre-Alias-Tabelle verboten ist.
_SCAN_DIRS: tuple[str, ...] = (
    "backend",
    "denker",
    "plugins",
    "forensics",
    "dsp",
    "cli",
    "Aurik10",
    "scripts",
)

#: Die einzige zulässige Alias-Quelle.
_ALIAS_OWNER = "backend/core/genre_registry.py"

_ALIAS_NAME_RE = re.compile(r"(?=.*genre)(?=.*alias)", re.IGNORECASE)


def _module_level_dict_names(path: Path) -> list[str]:
    """Namen aller Modul-Ebene-Dict-Zuweisungen (``NAME = {...}``) in *path*."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Dict):
            if isinstance(node.target, ast.Name):
                names.append(node.target.id)
    return names


def _check_alias_single_source() -> list[str]:
    """R1: Genre-Alias-Tabellen nur in der Registry."""
    problems: list[str] = []
    for rel_dir in _SCAN_DIRS:
        base = _REPO / rel_dir
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(_REPO).as_posix()
            if rel == _ALIAS_OWNER:
                continue
            for name in _module_level_dict_names(path):
                if _ALIAS_NAME_RE.search(name):
                    problems.append(
                        f"R1: {rel} definiert eine Genre-Alias-Tabelle ('{name}') — "
                        f"einzige Alias-Quelle ist {_ALIAS_OWNER} "
                        f"(§G9 (copilot-instructions.md))"
                    )
    return problems


def _check_profile_keys_resolvable() -> list[str]:
    """R2: Alle Profil-Schlüssel müssen über die Registry auflösen."""
    from backend.core import genre_registry as registry
    from backend.core.genre_goal_profile import _GENRE_PROFILES as _GOAL_PROFILES
    from backend.core.song_goal_importance import _GENRE_WEIGHT_PROFILES as _WEIGHT_PROFILES

    problems: list[str] = []
    canonical = set(registry.CANONICAL_GENRES)
    for quelle, keys in (
        ("genre_goal_profile._GENRE_PROFILES", sorted(_GOAL_PROFILES)),
        ("song_goal_importance._GENRE_WEIGHT_PROFILES", sorted(_WEIGHT_PROFILES)),
    ):
        for key in keys:
            resolved = registry.normalize_genre(key)
            if resolved not in canonical:
                problems.append(
                    f"R2: {quelle} enthält '{key}', das nicht auf ein kanonisches Genre "
                    f"auflöst (normalize_genre -> {resolved!r}; {len(canonical)} kanonische Genres)"
                )
    return problems


def _check_absent_xor() -> list[str]:
    """R3: goal_profile_key(g) is None ⟺ g in _GOAL_KEY_ABSENT."""
    from backend.core import genre_registry as registry

    absent = registry._GOAL_KEY_ABSENT
    problems: list[str] = []
    for genre in registry.CANONICAL_GENRES:
        is_none = registry.goal_profile_key(genre) is None
        declared = genre in absent
        if is_none and not declared:
            problems.append(
                f"R3: kanonisches Genre '{genre}' hat kein Ziel-Profil, ist aber NICHT in "
                f"_GOAL_KEY_ABSENT deklariert (stilles Fehlen statt Absicht)"
            )
        elif declared and not is_none:
            problems.append(
                f"R3: '{genre}' steht in _GOAL_KEY_ABSENT, hat aber sehr wohl ein "
                f"Ziel-Profil (veralteter Ausnahme-Eintrag)"
            )
    return problems


def _divergence_report() -> tuple[int, list[str]]:
    """R4: Wert-Divergenz der gemeinsam geführten Genres (Bericht, kein Fail)."""
    from backend.core.genre_goal_profile import _GENRE_PROFILES as _GOAL_PROFILES
    from backend.core.song_goal_importance import _GENRE_WEIGHT_PROFILES as _WEIGHT_PROFILES

    shared = sorted(set(_GOAL_PROFILES) & set(_WEIGHT_PROFILES))
    lines: list[str] = []
    cells = 0
    for genre in shared:
        a = _WEIGHT_PROFILES[genre]
        b = _GOAL_PROFILES[genre]
        diff = {
            goal: (a.get(goal, 1.0), b.get(goal, 1.0))
            for goal in set(a) | set(b)
            if abs(a.get(goal, 1.0) - b.get(goal, 1.0)) > 1e-9
        }
        cells += len(diff)
        if diff:
            paar = ", ".join(f"{g}: {x:.2f}/{y:.2f}" for g, (x, y) in sorted(diff.items()))
            lines.append(f"R4: '{genre}' — {len(diff)} abweichende Zellen (SGI/GGP): {paar}")
    return cells, lines


def _undeclared_goal_keys(sources: dict[str, set[str]]) -> list[str]:
    """R5-Kern: Ziel-Schlüssel, die weder kanonisch noch deklariert sind.

    Reine Funktion (testbar) — *sources* bildet Tabellenname → verwendete Schlüssel.
    """
    from backend.core.song_goal_importance import ALL_GOAL_NAMES, GOAL_DIALECT_MAP

    canonical = set(ALL_GOAL_NAMES)
    problems: list[str] = []
    for name, keys in sources.items():
        for key in sorted(keys):
            if key in canonical or key in GOAL_DIALECT_MAP:
                continue
            problems.append(
                f"R5: {name} nutzt das Ziel '{key}', das weder kanonisch noch in "
                f"GOAL_DIALECT_MAP deklariert ist (§G9 copilot-instructions.md)"
            )
    return problems


def _check_goal_dialects_declared() -> list[str]:
    """R5: alle Ziel-Schlüssel der drei Ziel-Tabellen sind deklariert."""
    from backend.core.genre_goal_profile import _GENRE_PROFILES
    from backend.core.goal_budget import _DEFAULT_GOAL_BUDGET
    from backend.core.song_goal_importance import _GENRE_WEIGHT_PROFILES

    sources: dict[str, set[str]] = {
        "genre_goal_profile._GENRE_PROFILES": {g for prof in _GENRE_PROFILES.values() for g in prof},
        "song_goal_importance._GENRE_WEIGHT_PROFILES": {g for prof in _GENRE_WEIGHT_PROFILES.values() for g in prof},
        "goal_budget._DEFAULT_GOAL_BUDGET": set(_DEFAULT_GOAL_BUDGET),
    }
    return _undeclared_goal_keys(sources)


def main() -> int:
    parser = argparse.ArgumentParser(description="Genre-Single-Source-Guard (Alias nur in der Registry)")
    parser.add_argument("--quiet", action="store_true", help="R4-Bericht unterdrücken")
    args = parser.parse_args()

    problems: list[str] = []
    problems.extend(_check_alias_single_source())
    problems.extend(_check_profile_keys_resolvable())
    problems.extend(_check_absent_xor())
    problems.extend(_check_goal_dialects_declared())

    for problem in problems:
        print(f"LUECKE: {problem}")

    if not args.quiet:
        from backend.core.song_goal_importance import ALL_GOAL_NAMES, GOAL_DIALECT_MAP

        cells, lines = _divergence_report()
        mapped = sum(1 for v in GOAL_DIALECT_MAP.values() if v is not None)
        print()
        print("Ziel-Vokabulare (T3.4-Befund 2026-10-06):")
        print(f"  kanonisch (ALL_GOAL_NAMES): {len(ALL_GOAL_NAMES)} Ziele")
        print(
            f"  deklarierte Fremdschlüssel : {len(GOAL_DIALECT_MAP)} "
            f"({mapped} abgebildet, {len(GOAL_DIALECT_MAP) - mapped} offen/sign-off-pflichtig)"
        )
        print()
        print("Rollenteilung der beiden Ziel-Gewichtstabellen (Befund T3.4):")
        print(f"  A) {ROLLE_A}")
        print(f"  B) {ROLLE_B}")
        if lines:
            print(f"\nDivergenz-Bericht ({cells} Zellen — KLANGRELEVANT, Sign-off-pflichtig):")
            for line in lines:
                print(f"  {line}")
        else:
            print("\nDivergenz-Bericht: keine abweichenden Zellen")

    if problems:
        print(f"\nFEHLER: {len(problems)} Verletzung(en) — fail-closed (Exit 1).")
        return 1

    print("\nOK: Genre-Auflösung single-source, Profil-Schlüssel auflösbar, Ausnahmeliste konsistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
