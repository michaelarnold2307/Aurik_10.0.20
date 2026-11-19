"""Übersetzt englische Docstrings in allen Aurik-Python-Dateien ins Deutsche.

Strategie:
1. AST-Analyse: findet Docstrings pro Funktion/Klasse/Modul
2. Heuristik: erkennt englische Docstrings (keine Umlaute + englisches Verbmuster)
3. Regelbasierte Übersetzung: ~100 Verbmuster + Satzstrukturen
4. Komplexe Sätze werden mit TODO-Marker versehen

Aufruf:
    python scripts/translate_docstrings_to_de.py [--dry-run] [--dirs backend forensics ...]
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Callable
from pathlib import Path

# ---------------------------------------------------------------------------
# Heuristik: Ist ein Docstring auf Englisch?
# ---------------------------------------------------------------------------
_DE_CHARS = set("äöüÄÖÜß")
_EN_VERB_PATTERN = re.compile(
    r"""^(
        Accumulate|Add|Allocate|Amplif|Analys|Analyz|Apply|Assert|Attenuate|
        Blend|Boost|Bridge|Build|
        Calibrat|Cap|Check|Clear|Clamp|Clip|Close|Collect|Combin|Comput|Connect|
        Convert|Count|Creat|
        Decod|Detect|Determin|Disable|Disconnect|Dispatch|Display|
        Enable|Encod|Ensur|Estimat|Evaluat|Execut|Extract|Emit|Extend|Expand|
        Fetch|Filter|Find|Flush|Format|
        Generate|Get|
        Handle|Hide|
        Initializ|Init|Interpolat|
        Limit|Listen|List|Load|Log|
        Measur|Merg|Mix|Monitor|
        Normaliz|Notif|
        Open|Optimiz|
        Pad|Pars|Perform|Print|Process|Propagat|
        Rank|Read|Receive|Reconstruct|Register|Remove|Repair|Reset|Reshap|
        Resiz|Resolv|Restor|
        Return|Run|
        Sanitiz|Save|Scal|Scan|Schedul|Score|Select|Send|Serializ|Set|
        Shift|Show|Signal|Smooth|Sort|Subscribe|Suppress|
        Track|Transform|Trigger|Trim|Truncat|
        Update|
        Validat|
        Wrap|Write|
        Whether|If[ ]|The[ ]|This[ ]|An[ ]|A[ ]|
        Unified|Phase|Plugin|Audio|Score|
        Main[ ]|Core[ ]|Helper|Internal|
        Bridge|Facade|Singleton|Factory|
        Lazy|Simple|Lightweight
    )""",
    re.VERBOSE | re.IGNORECASE,
)


def _is_english(docstring: str) -> bool:
    """Gibt True zurück wenn der Docstring wahrscheinlich Englisch ist."""
    if not docstring:
        return False
    if any(c in docstring for c in _DE_CHARS):
        return False
    first_line = docstring.strip().split("\n")[0].strip()
    return bool(_EN_VERB_PATTERN.match(first_line))


# ---------------------------------------------------------------------------
# Übersetzungsregeln
# ---------------------------------------------------------------------------
# Jede Regel ist (regex_pattern, replacement_or_callable).
# Regeln werden der Reihe nach angewendet — erste passende gewinnt.
_RuleReplacement = str | Callable
_RULES: list[tuple[re.Pattern, _RuleReplacement]] = []


def _rule(pattern: str, replacement: str) -> None:
    _RULES.append((re.compile(pattern, re.IGNORECASE), replacement))


# --- "Returns? X" → "Gibt X zurück" (Verb-Umstellung) ---
_RULES.append(
    (
        re.compile(r"^Returns? (.+)", re.IGNORECASE | re.DOTALL),
        lambda m: f"Gibt {_translate_rest(m.group(1))} zurück.",
    )
)

# --- Einfache Verb-Ersetzungen (Verb + Rest bleiben an gleicher Position) ---
_SIMPLE_VERBS = [
    (r"^Accumulates? ", "Akkumuliert "),
    (r"^Adds? ", "Fügt hinzu: "),
    (r"^Allocates? ", "Allokiert "),
    (r"^Amplif(ies|y)s? ", "Verstärkt "),
    (r"^Analys(es|e)s? ", "Analysiert "),
    (r"^Analyz(es|e)s? ", "Analysiert "),
    (r"^Applies? ", "Wendet an: "),
    (r"^Asserts? ", "Stellt sicher: "),
    (r"^Attenuates? ", "Dämpft "),
    (r"^Blends? ", "Mischt "),
    (r"^Boosts? ", "Verstärkt "),
    (r"^Bridges? ", "Verbindet "),
    (r"^Builds? ", "Erstellt "),
    (r"^Calibrates? ", "Kalibriert "),
    (r"^Caps? ", "Begrenzt "),
    (r"^Checks? ", "Prüft "),
    (r"^Clamps? ", "Begrenzt "),
    (r"^Clears? ", "Löscht "),
    (r"^Clips? ", "Begrenzt "),
    (r"^Closes? ", "Schließt "),
    (r"^Collects? ", "Sammelt "),
    (r"^Combines? ", "Kombiniert "),
    (r"^Computes? ", "Berechnet "),
    (r"^Connects? ", "Verbindet "),
    (r"^Converts? ", "Konvertiert "),
    (r"^Counts? ", "Zählt "),
    (r"^Creates? ", "Erstellt "),
    (r"^Decodes? ", "Dekodiert "),
    (r"^Detects? ", "Erkennt "),
    (r"^Determines? ", "Bestimmt "),
    (r"^Disables? ", "Deaktiviert "),
    (r"^Disconnects? ", "Trennt "),
    (r"^Dispatches? ", "Verteilt "),
    (r"^Displays? ", "Zeigt an: "),
    (r"^Emits? ", "Emittiert "),
    (r"^Enables? ", "Aktiviert "),
    (r"^Encodes? ", "Kodiert "),
    (r"^Ensures? ", "Stellt sicher: "),
    (r"^Estimates? ", "Schätzt "),
    (r"^Evaluates? ", "Bewertet "),
    (r"^Executes? ", "Führt aus: "),
    (r"^Expands? ", "Erweitert "),
    (r"^Extends? ", "Erweitert "),
    (r"^Extracts? ", "Extrahiert "),
    (r"^Fetches? ", "Ruft ab: "),
    (r"^Filters? ", "Filtert "),
    (r"^Finds? ", "Findet "),
    (r"^Flushes? ", "Leert "),
    (r"^Formats? ", "Formatiert "),
    (r"^Generates? ", "Generiert "),
    (r"^Gets? ", "Gibt zurück: "),
    (r"^Handles? ", "Verarbeitet "),
    (r"^Hides? ", "Verbirgt "),
    (r"^Initializes? ", "Initialisiert "),
    (r"^Inits? ", "Initialisiert "),
    (r"^Interpolates? ", "Interpoliert "),
    (r"^Limits? ", "Begrenzt "),
    (r"^Listens? ", "Horcht auf: "),
    (r"^Lists? ", "Listet auf: "),
    (r"^Loads? ", "Lädt "),
    (r"^Logs? ", "Protokolliert "),
    (r"^Measures? ", "Misst "),
    (r"^Merges? ", "Führt zusammen: "),
    (r"^Mixes? ", "Mischt "),
    (r"^Monitors? ", "Überwacht "),
    (r"^Normalizes? ", "Normalisiert "),
    (r"^Notifies? ", "Benachrichtigt "),
    (r"^Opens? ", "Öffnet "),
    (r"^Optimizes? ", "Optimiert "),
    (r"^Pads? ", "Füllt auf: "),
    (r"^Parses? ", "Parst "),
    (r"^Performs? ", "Führt durch: "),
    (r"^Prints? ", "Gibt aus: "),
    (r"^Processes? ", "Verarbeitet "),
    (r"^Propagates? ", "Propagiert "),
    (r"^Ranks? ", "Bewertet "),
    (r"^Reads? ", "Liest "),
    (r"^Receives? ", "Empfängt "),
    (r"^Reconstructs? ", "Rekonstruiert "),
    (r"^Registers? ", "Registriert "),
    (r"^Removes? ", "Entfernt "),
    (r"^Repairs? ", "Repariert "),
    (r"^Resets? ", "Setzt zurück: "),
    (r"^Reshapes? ", "Formt um: "),
    (r"^Resizes? ", "Ändert die Größe von "),
    (r"^Resolves? ", "Löst auf: "),
    (r"^Restores? ", "Restauriert "),
    (r"^Runs? ", "Führt aus: "),
    (r"^Sanitizes? ", "Bereinigt "),
    (r"^Saves? ", "Speichert "),
    (r"^Scales? ", "Skaliert "),
    (r"^Scans? ", "Scannt "),
    (r"^Schedules? ", "Plant "),
    (r"^Scores? ", "Bewertet "),
    (r"^Selects? ", "Wählt aus: "),
    (r"^Sends? ", "Sendet "),
    (r"^Serializes? ", "Serialisiert "),
    (r"^Sets? ", "Setzt "),
    (r"^Shifts? ", "Verschiebt "),
    (r"^Shows? ", "Zeigt an: "),
    (r"^Signals? ", "Signalisiert "),
    (r"^Smooths? ", "Glättet "),
    (r"^Sorts? ", "Sortiert "),
    (r"^Subscribes? ", "Abonniert "),
    (r"^Suppresses? ", "Unterdrückt "),
    (r"^Tracks? ", "Verfolgt "),
    (r"^Transforms? ", "Transformiert "),
    (r"^Triggers? ", "Löst aus: "),
    (r"^Trims? ", "Kürzt "),
    (r"^Truncates? ", "Kürzt "),
    (r"^Updates? ", "Aktualisiert "),
    (r"^Validates? ", "Validiert "),
    (r"^Wraps? ", "Kapselt "),
    (r"^Writes? ", "Schreibt "),
    # Satzanfänge ohne Verb
    (r"^Whether ", "Ob "),
    (r"^If ", "Falls "),
    (r"^The ", "Der/Die/Das "),
    (r"^This ", ""),  # "This class ..." → direkt beschreiben
    (r"^An? ", ""),
]

for _pat, _repl in _SIMPLE_VERBS:
    _RULES.append((re.compile(_pat, re.IGNORECASE), _repl))


def _translate_rest(text: str) -> str:
    """Minimale Inline-Bereinigung der Rest-Phrase nach Verbersetzung."""
    text = text.rstrip(".")
    return text


def _translate_first_line(line: str) -> str:
    """Wendet die erste passende Regel auf die erste Zeile an."""
    for pattern, replacement in _RULES:
        m = pattern.match(line)
        if m:
            if callable(replacement):
                return str(replacement(m))
            new_line = pattern.sub(replacement, line, count=1)
            # Satzende sicherstellen
            new_line = new_line.rstrip()
            if new_line and new_line[-1] not in ".!?:":
                new_line += "."
            return new_line
    # Kein Match → als-is belassen, aber mit TODO markieren
    return f"TODO(de): {line}"


def _translate_docstring(docstring: str) -> str:
    """Übersetzt einen englischen Docstring ins Deutsche (erste Zeile + Rest)."""
    lines = docstring.split("\n")
    # Erste inhaltliche Zeile übersetzen
    first_idx = 0
    while first_idx < len(lines) and not lines[first_idx].strip():
        first_idx += 1
    if first_idx >= len(lines):
        return docstring

    indent = len(lines[first_idx]) - len(lines[first_idx].lstrip())
    indent_str = " " * indent
    first_line_stripped = lines[first_idx].strip()
    translated_first = _translate_first_line(first_line_stripped)
    lines[first_idx] = indent_str + translated_first

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Token-basiertes Ersetzen von Docstrings in Quelldateien
# ---------------------------------------------------------------------------


def _replace_docstrings_in_source(source: str) -> tuple[str, int]:
    """
    Ersetzt englische Docstrings in source.
    Gibt (neuer_quelltext, anzahl_ersetzungen) zurück.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, 0

    # Sammle (lineno, col_offset, end_lineno, end_col_offset, new_docstring)
    replacements: list[tuple[int, int, int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            continue
        if not node.body:
            continue
        first_stmt = node.body[0]
        if not isinstance(first_stmt, ast.Expr):
            continue
        val = first_stmt.value
        if not isinstance(val, ast.Constant) or not isinstance(val.value, str):
            continue
        ds = val.value
        if not _is_english(ds):
            continue
        translated = _translate_docstring(ds)
        if translated == ds:
            continue
        if first_stmt.end_lineno is None or first_stmt.end_col_offset is None:
            continue
        replacements.append(
            (
                first_stmt.lineno,
                first_stmt.col_offset,
                first_stmt.end_lineno,
                first_stmt.end_col_offset,
                translated,
            )
        )

    if not replacements:
        return source, 0

    # Ersetze von hinten nach vorne (vermeidet Zeilenverschiebungen)
    lines = source.splitlines(keepends=True)
    replacements.sort(key=lambda r: (r[0], r[1]), reverse=True)

    for lineno, col_offset, end_lineno, _end_col_offset, new_ds in replacements:
        # Extrahiere das originale Docstring-Literal aus dem Quelltext
        # und ersetze es durch das übersetzte
        start_line = lineno - 1  # 0-basiert
        end_line = end_lineno - 1

        orig_segment = "".join(lines[start_line : end_line + 1])

        # Erkennt das Anführungszeichen-Paar (""" oder ''')
        stripped = orig_segment.lstrip()
        if stripped.startswith('"""'):
            quote = '"""'
        elif stripped.startswith("'''"):
            quote = "'''"
        else:
            # Einfaches String-Literal — überspringen (zu fragil)
            continue

        # Baue das neue Literal
        indent_str = " " * col_offset
        if "\n" in new_ds:
            new_literal = f"{indent_str}{quote}{new_ds}{quote}\n"
        else:
            new_literal = f"{indent_str}{quote}{new_ds}{quote}\n"

        lines[start_line : end_line + 1] = [new_literal]

    return "".join(lines), len(replacements)


# ---------------------------------------------------------------------------
# Datei-Verarbeitung
# ---------------------------------------------------------------------------


def process_file(path: Path, dry_run: bool = False) -> int:
    """Verarbeitet eine Datei und gibt die Anzahl der Ersetzungen zurück."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"  FEHLER lesen {path}: {exc}", file=sys.stderr)
        return 0

    new_source, count = _replace_docstrings_in_source(source)
    if count == 0:
        return 0

    if not dry_run:
        path.write_text(new_source, encoding="utf-8")

    return count


def process_directory(
    directory: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> tuple[int, int]:
    """Verarbeitet alle .py-Dateien rekursiv. Gibt (gesamt_ersetzungen, dateien) zurück."""
    total_replacements = 0
    total_files = 0

    for py_file in sorted(directory.rglob("*.py")):
        count = process_file(py_file, dry_run=dry_run)
        if count > 0:
            total_replacements += count
            total_files += 1
            if verbose:
                mode = "[DRY]" if dry_run else "[OK ]"
                print(f"  {mode} {py_file.relative_to(directory.parent)} ({count} Ersetzungen)")

    return total_replacements, total_files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI-Einstiegspunkt: übersetzt englische Docstrings in allen konfigurierten Verzeichnissen."""
    parser = argparse.ArgumentParser(description="Übersetzt englische Docstrings auf Deutsch.")
    parser.add_argument("--dry-run", action="store_true", help="Keine Änderungen schreiben")
    parser.add_argument("--verbose", "-v", action="store_true", help="Zeigt bearbeitete Dateien")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=[
            "backend",
            "forensics",
            "plugins",
            "Aurik10",
            "cli",
            "denker",
            "dsp",
            "export",
            "processing",
            "workflow",
        ],
        help="Verzeichnisse (relativ zum Projekt-Root)",
    )
    args = parser.parse_args()

    # Projekt-Root = Verzeichnis dieser Datei minus scripts/
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    grand_total_repl = 0
    grand_total_files = 0

    for d in args.dirs:
        target = project_root / d
        if not target.exists():
            continue
        repl, files = process_directory(target, dry_run=args.dry_run, verbose=args.verbose)
        if repl > 0:
            print(f"{d:30s}: {files:4d} Dateien, {repl:5d} Ersetzungen")
        grand_total_repl += repl
        grand_total_files += files

    mode = " [DRY RUN]" if args.dry_run else ""
    print(f"\nGesamt{mode}: {grand_total_files} Dateien, {grand_total_repl} Ersetzungen")


if __name__ == "__main__":
    main()
