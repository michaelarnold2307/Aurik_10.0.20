"""code_weakness_scanner.py — Statische Code-Schwachstellen-Prüfung (Watchdog-Erweiterung).

Scannt den Produktions-Code (backend/, denker/, Aurik10/, cli/) auf bekannte
Schwachstellen-Klassen und weist jeden Befund klar aus: Regel-ID, Schweregrad,
Datei:Zeile, Evidenz-Ausschnitt, Spec-Referenz und empfohlene Aktion.

Regelbasis (normative Kette, AGENTS.md §1/§3 — Zitate mit Quelle):
  - bridge_import_violation      §V4 (copilot-instructions.md)
  - dither_missing_int_conversion §V5 (copilot-instructions.md)
  - silent_fallback_no_log       §V6 (copilot-instructions.md)
  - bare_except                  Bug-Klassen (copilot-instructions.md, Silent Failure)
  - module_logger_missing        Logger-Pflicht (§III DSP, AGENTS.md §3)
  - nan_inf_guard_missing        §0a (copilot-instructions.md)
  - determinism_time_usage       §G5 (AGENTS.md §3 / copilot-instructions.md)
  - print_in_production          Logger-Pflicht (§III DSP, AGENTS.md §3)

Regel-Wahrheit (Korrektur 2026-10-05, Bug-Hunt): Jede Regel wird gegen ihren
WORTLAUT geprüft, nicht gegen ein Muster:
  - §V5 (copilot-instructions.md) verbietet „Integer-Quantisierung (bit_depth < 32) ohne Dithering“ —
    also Audio-Quantisierung. Nicht-Audio-Casts (Masken/Indizes/Overflow-
    Prüfungen) sind ausgenommen und tragen im Repo die dokumentierte Ausnahme
    „# §V5 (copilot-instructions.md) … Dither applied at export“ im Kontext-Fenster (siehe _check_dither).
  - §V6 (copilot-instructions.md) verlangt bei Fallbacks/Fallthrough einen Log-Aufruf; jeder Logger-Name
    zählt (logger, _logger, LOGGER, self.logger) — nicht nur „logger“.
  - §G5 (GEBOTE.md) verbietet Wall-Clock-Zeit in ENTSCHEIDUNGSLOGIK. Messungen
    (Dauer/Profiling) und Zeitstempel sind kein Verstoß; persistierte
    Zeitstempel MÜSSEN Wall-Clock bleiben (monotonic ist prozesslokal und
    über Prozess-/Session-Grenzen bedeutungslos). Die Regel meldet daher
    AST-basiert nur Entscheidungspfade und weist den Rest transparent als
    unterdrückt aus.
  - Logger-/Guard-Regeln prüfen CODE, nie Prosa: Kommentare und String-
    Literale werden positionsgetreu geblankt (blank_noncode), sonst entstehen
    Fehlalarme aus Kommentaren (Befund 2026-10-05: 26/33 print()-Treffer waren
    auskommentiert) und falsche Entlastungen (auskommentiertes getLogger()).

Nutzung standalone:
    python audit/code_weakness_scanner.py --workspace . \
        --json-out audit/code_weakness_report.json \
        --md-out audit/code_weakness_report.md

Nutzung als Bibliothek (Live-Watchdog):
    from audit.code_weakness_scanner import scan_workspace, summarize_findings
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import time
import tokenize
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_SEVERITY_RANK: dict[str, int] = {"critical": 3, "high": 2, "medium": 1, "low": 0}

_SKIP_DIRS: frozenset[str] = frozenset(
    {".git", "__pycache__", ".venv", ".venv_aurik", "node_modules", "models", "temp_repro"}
)

# Aggregierte Regeln (Pro-Datei-Zählung statt pro Fund, um Logflut zu vermeiden).
_PRINT_MIN_COUNT = 3
_AGGREGATED_TOP_N = 10
_PER_FILE_AST_CAP = 3

# §V6 (copilot-instructions.md): Jeder Logger-Name zählt — nicht nur „logger".
_LOGGER_ROOTS: frozenset[str] = frozenset({"logger", "log", "logging"})
_LOGGING_LEVELS: frozenset[str] = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical", "fatal", "log"}
)

# §V6 (copilot-instructions.md) ist erfüllt, wenn der Fehlerpfad überhaupt
# protokolliert wird. Im Repo protokolliert ein Teil der Klassen indirekt über
# eine Audit-Fassade (self._audit_log("error", …) → logger.error/_logger.debug);
# solche Aufrufe enden verifiziert (2026-10-05, 33 Definitionen geprüft) in
# einem echten Logger. Deshalb zählt ein Aufruf mit „log" im Methodennamen als
# Logging-Nachweis — sonst meldet die Regel korrekte Fallbacks als stumm.
_LOG_LIKE_MARKER = "log"

# Prosa-Freiheit der Regeln: Diese Dateien sind Werkzeuge/Diagnose-Skripte, ihr
# stdout IST die Schnittstelle (CI/Entwickler) — die Logger-Pflicht (§III DSP)
# gilt für Laufzeit-Module der Pipeline, nicht für CLI-Berichtsskripte.
_PRINT_TOOL_PREFIXES: tuple[str, ...] = ("backend/core/scripts/",)


@dataclass(frozen=True)
class WeaknessRule:
    """Definition einer Schwachstellen-Regel (klar ausweisbar)."""

    rule_id: str
    severity: str
    spec_ref: str
    title: str
    recommendation: str


RULES: dict[str, WeaknessRule] = {
    "bridge_import_violation": WeaknessRule(
        rule_id="bridge_import_violation",
        severity="critical",
        spec_ref="§V4 (copilot-instructions.md)",
        title="UI/Frontend importiert backend/core direkt statt über backend/api/bridge.py",
        recommendation=(
            "Import über backend.api.bridge umleiten; Denker-Schicht ist ausgenommen. (§V4 (copilot-instructions.md) Bridge-Bypass-Verbot)"
        ),
    ),
    "dither_missing_int_conversion": WeaknessRule(
        rule_id="dither_missing_int_conversion",
        severity="high",
        spec_ref="§V5 (copilot-instructions.md)",
        title="Integer-Konversion ohne Dither (bit_depth < 32)",
        recommendation=(
            "POW-r Type 3 (primär) oder TPDF (Fallback) vor der Konversion anwenden; "
            "kein nacktes astype(np.int16). (§V5 (copilot-instructions.md) Truncation-ohne-Dither-Verbot)"
        ),
    ),
    "silent_fallback_no_log": WeaknessRule(
        rule_id="silent_fallback_no_log",
        severity="high",
        spec_ref="§V6 (copilot-instructions.md)",
        title="Fallback/Return in except-Block ohne Logging",
        recommendation=(
            "logger.warning() + Begründung ergänzen, damit ML→DSP-Fallbacks nie "
            "stumm bleiben. (§V6 (copilot-instructions.md) Silent-Failure-Verbot)"
        ),
    ),
    "bare_except": WeaknessRule(
        rule_id="bare_except",
        severity="medium",
        spec_ref="Bug-Klassen (copilot-instructions.md) — Silent Failure",
        title="Bare except: fängt auch SystemExit/KeyboardInterrupt und verschluckt Fehler",
        recommendation="Exception-Typ explizit benennen und den Fehler loggen.",
    ),
    "module_logger_missing": WeaknessRule(
        rule_id="module_logger_missing",
        severity="medium",
        spec_ref="Logger-Pflicht (§III DSP, AGENTS.md §3)",
        title="Modul mit Fehlerpfaden hat keinen Logger (logging.getLogger(__name__))",
        recommendation=("logger = logging.getLogger(__name__) ergänzen; Fehlerpfade müssen logbar sein."),
    ),
    "nan_inf_guard_missing": WeaknessRule(
        rule_id="nan_inf_guard_missing",
        severity="medium",
        spec_ref="§0a (copilot-instructions.md)",
        title="Phase ohne NaN/Inf-Schutz (isfinite/nan_to_num/isnan fehlt komplett)",
        recommendation=("NaN/Inf-Guard in die Phase einbauen (§0a: Schutz in jeder Phase)."),
    ),
    "determinism_time_usage": WeaknessRule(
        rule_id="determinism_time_usage",
        severity="medium",
        spec_ref="§G5 (AGENTS.md §3 / copilot-instructions.md)",
        title="Wall-Clock (time.time()) in Entscheidungslogik — Determinismus-Risiko",
        recommendation=(
            "Entscheidung von einer deterministischen Quelle abhängig machen (Sample-/Budget-basiert, "
            "Session-Seed) oder die Zeitquelle im Entscheidungspfad auf time.monotonic() umstellen — "
            "Wall-Clock springt (NTP/DST) und bricht die Reproduzierbarkeit (§G5 copilot-instructions.md). "
            "Messungen/Zeitstempel sind kein Verstoß und werden transparent als unterdrückt ausgewiesen."
        ),
    ),
    "wallclock_ttl_housekeeping": WeaknessRule(
        rule_id="wallclock_ttl_housekeeping",
        severity="low",
        spec_ref="§G5 (copilot-instructions.md) — Abgrenzung TTL-Haushalt",
        title="Wall-Clock in TTL-/Datei-Lebenszyklus-Entscheidung (kein Audio-Determinismus-Verstoß)",
        recommendation=(
            "Bewusst: Jede TTL-Politik (Prüfpunkt-/Cache-Ablauf) braucht Wall-Clock — der Wert wird "
            "persistiert und muss über Prozess-/Boot-Grenzen vergleichbar bleiben. Kein §G5-Verstoß, "
            "solange die Entscheidung das Restaurierungs-Audio nicht beeinflusst. Sichtbar ausgewiesen, "
            "damit die Prüfung vollständig bleibt."
        ),
    ),
    "walltime_clock_mismatch": WeaknessRule(
        rule_id="walltime_clock_mismatch",
        severity="high",
        spec_ref=".github/VERBOTEN.md (Wall-Time-Referenz-Mismatch) / §G5 (copilot-instructions.md)",
        title="Gemischte Uhr-Epochen (time.time() vs. monotonic()/perf_counter()) in einer Rechnung",
        recommendation=(
            "Beide Seiten des Akkumulators auf dieselbe Uhr stellen — normativ time.monotonic() für "
            "Zeitbudgets, time.time() (Wall-Clock) für persistierte Zeitstempel "
            "(.github/VERBOTEN.md: Epochen divergieren → Pipeline-Übersprung/Beschädigung)."
        ),
    ),
    "print_in_production": WeaknessRule(
        rule_id="print_in_production",
        severity="low",
        spec_ref="Logger-Pflicht (§III DSP, AGENTS.md §3)",
        title="print() in Laufzeit-Modul statt Logger (Kommentare/Prosa zählen nicht)",
        recommendation=(
            "Ausgaben über logging umleiten (Logger-Pflicht). CLI-/Diagnose-Skripte unter "
            "backend/core/scripts/ sind ausgenommen — dort ist stdout die Schnittstelle."
        ),
    ),
}

_BRIDGE_IMPORT_RX = re.compile(
    r"^\s*(?:from\s+backend\.core\b.*\bimport\b|import\s+backend\.core\b|from\s+backend\s+import\s+core\b)",
    re.MULTILINE,
)
_INT_CAST_RX = re.compile(r"\bastype\(\s*(?:dtype\s*=\s*)?(?:(?:np|numpy)\.)?[\"']?(int8|int16|int32)[\"']?\)")
_DITHER_CONTEXT_TOKENS: tuple[str, ...] = ("dither", "tpdf", "powr", "pow-r")
_PRINT_RX = re.compile(r"(?<![\w.])print\(")
_NAN_GUARD_TOKENS: tuple[str, ...] = ("isfinite", "isnan", "nan_to_num", "isinf")


@dataclass
class WeaknessFinding:
    """Ein klar ausgewiesener Schwachstellen-Befund."""

    rule_id: str
    severity: str
    file: str  # relativer POSIX-Pfad zur Workspace-Root
    line: int
    evidence: str
    spec_ref: str
    recommendation: str


@dataclass
class ScanResult:
    """Ergebnis eines kompletten Schwachstellen-Scans."""

    findings: list[WeaknessFinding] = field(default_factory=list)
    files_scanned: int = 0
    duration_s: float = 0.0
    per_rule_counts: dict[str, int] = field(default_factory=dict)
    # Transparenz: Befunde, die unter Meldeschwellen fielen oder durch
    # Kappungen (AST-Cap, Aggregations-Top-N, max_findings) nicht ausgewiesen
    # wurden. Schlüssel: rule_id bzw. "truncated_max_findings".
    suppressed: dict[str, int] = field(default_factory=dict)


def _is_test_path(rel_posix: str) -> bool:
    parts = rel_posix.split("/")
    return "tests" in parts or any(p.startswith("test_") for p in parts)


def _is_vendor_path(rel_posix: str) -> bool:
    return any(part.startswith("_vendor") for part in rel_posix.split("/"))


def iter_python_files(workspace: Path) -> Iterable[tuple[Path, str]]:
    """Yieldet (Pfad, relativer POSIX-Pfad) für alle zu prüfenden Python-Dateien."""
    for path in sorted(workspace.rglob("*.py")):
        rel = path.relative_to(workspace).as_posix()
        if any(part in _SKIP_DIRS or part.startswith(".venv") for part in Path(rel).parts):
            continue
        yield path, rel


def _evidence(lines: list[str], line_no: int) -> str:
    idx = max(0, min(line_no - 1, len(lines) - 1))
    snippet = lines[idx].strip()
    if len(snippet) > 160:
        snippet = snippet[:157] + "..."
    return snippet or f"<Zeile {line_no}>"


def _handler_returns_value(handler: ast.ExceptHandler) -> bool:
    for child in ast.walk(handler):
        if isinstance(child, ast.Return) and child.value is not None:
            return True
        if isinstance(child, ast.Raise):
            return False
    return False


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    """Eltern-Karte für AST-Knoten (Kontext-Klassifikation, §G5 (GEBOTE.md))."""
    par: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            par[child] = parent
    return par


def blank_noncode(text: str) -> str:
    """Ersetzt Kommentare und String-Literale positionsgetreu durch Leerzeichen.

    Die Regeln prüfen CODE, nie Prosa: Ohne diese Trennung entstehen Fehlalarme
    aus Kommentaren/Docstrings (Befund 2026-10-05: 26 von 33 print()-Treffern
    waren auskommentiert — 5 davon in `backend/core/dsp/phase_rotation.py`) und
    falsche Entlastungen (ein auskommentiertes `getLogger()` hätte die
    Logger-Regel entschärft). Zeilen- und Spaltenpositionen bleiben erhalten.

    Fallback bei Tokenizer-Fehler: Rohtext (konservativ — mehr Befunde, nie
    weniger).
    """
    lines = text.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def _offset(row: int, col: int) -> int:
        return starts[row - 1] + col

    chars = list(text)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                for idx in range(_offset(*tok.start), _offset(*tok.end)):
                    if chars[idx] != "\n":
                        chars[idx] = " "
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text
    return "".join(chars)


def _is_logger_expr(node: ast.AST) -> bool:
    """True, wenn der Ausdruck ein Logger-Objekt bezeichnet (§V6 (copilot-instructions.md)).

    Abgedeckt: logger, _logger, LOGGER, self.logger, self._logger,
    logging.getLogger(__name__), modul.log. NICHT abgedeckt (absichtlich):
    eigene Audit-/Trace-Methoden wie `self._audit_log(...)` — sie protokollieren
    in eine Datei/Struktur, erfüllen aber nicht die §V6-Forderung nach einem
    `logger.warning()`.
    """
    cur: ast.AST = node
    while True:
        if isinstance(cur, ast.Attribute):
            if cur.attr.strip("_").lower() in _LOGGER_ROOTS:
                return True
            cur = cur.value
            continue
        if isinstance(cur, ast.Call):
            func = cur.func
            if isinstance(func, ast.Attribute) and func.attr == "getLogger":
                return True
            cur = func
            continue
        break
    return isinstance(cur, ast.Name) and cur.id.strip("_").lower() in _LOGGER_ROOTS


def _is_log_like_call(child: ast.AST) -> bool:
    """True, wenn der Aufruf ein Logging-Nachweis ist (§V6 (copilot-instructions.md)).

    Erfasst drei verifizierte Formen:
    1. `logger/_logger/LOGGER/self.logger.<level>()`,
    2. `logging.getLogger(...).<level>()`,
    3. modulweite Log-Fassaden wie `_audit_log("error", ...)` (shellac_*), die
       nachweislich an `logger.error/warning/info` delegieren.
    Ein name-„log“-Treffer genügt, weil die Regel sonst korrekte Fallbacks als
    stumm meldet (Befund 2026-10-05: 13 direkte + 7 indirekte Fehlalarme).
    """
    if not isinstance(child, ast.Call):
        return False
    func = child.func
    if isinstance(func, ast.Name):
        return _LOG_LIKE_MARKER in func.id.strip("_").lower()
    if not isinstance(func, ast.Attribute):
        return False
    attr = func.attr.lower()
    if attr in _LOGGING_LEVELS and _is_logger_expr(func.value):
        return True
    # Audit-Fassade: self._audit_log(...) / _log_event(...) → echter Logger
    return _LOG_LIKE_MARKER in attr.strip("_")


def _has_logging_call(node: ast.AST) -> bool:
    """True, wenn im Teilbaum ein Logger-Aufruf steht (jeder Logger-Name).

    §V6 (copilot-instructions.md) verlangt Protokollierung des Fallbacks —
    welcher Logger-Name genutzt wird (`logger`, `_logger`, `LOGGER`,
    `self.logger`) ist dafür unerheblich. Die frühere Namensprüfung auf genau
    `logger`/`logging`/`log` erzeugte 13 Fehlalarme (Befund 2026-10-05).
    """
    return any(_is_log_like_call(child) for child in ast.walk(node))


def _wallclock_in_decision(node: ast.AST, par: dict[ast.AST, ast.AST]) -> bool:
    """True, wenn dieses time.time() die Entscheidungslogik speist (§G5 (GEBOTE.md)).

    Zwei Wege werden geprüft:
    1. direkt im Test einer Verzweigung/Schleife/Assertion, oder
    2. an einen Namen gebunden, der in derselben Funktion in einem Vergleich
       steht (Muster `now = time.time()` … `if now - ts > grenze`).
    Messungen (Dauer) und Zeitstempel fallen bewusst NICHT darunter.
    """
    cur: ast.AST = node
    while cur in par:
        parent = par[cur]
        if isinstance(parent, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            test = getattr(parent, "test", None)
            if test is not None and any(sub is node for sub in ast.walk(test)):
                return True
        cur = parent

    # Fall 2: Bindung an einen EINFACHEN Namen, der in derselben Funktion
    # verglichen wird (Muster `elapsed = time.time() - t0` … `if elapsed > budget`).
    # Bewusst NICHT: Attribut-/Subscript-Ziele (`fp.last_updated`,
    # `entry["_last_access"]`) und Container-Werte (`{"ts": time.time()}`) —
    # das sind Zeitstempel bzw. Messwerte für Berichte, keine Entscheidungsgrößen.
    bound: set[str] = set()
    cur = node
    while cur in par:
        parent = par[cur]
        if isinstance(parent, ast.Assign):
            # Nur direkte Messgrößen: `now = time.time()`, `elapsed = time.time() - t0`.
            # Kein Entscheidungsträger: Zeitstempel-Container und Formatierungen
            # (`{"ts": time.time()}`, `str(time.time())`, `ts = round(time.time(), 1)`).
            if parent.value is node or isinstance(parent.value, ast.BinOp):
                for target in parent.targets:
                    if isinstance(target, ast.Name):
                        bound.add(target.id)
            break
        if isinstance(parent, (ast.Expr, ast.Return, ast.Compare, ast.Call, ast.AnnAssign)):
            break
        cur = parent
    if not bound:
        return False

    scope: ast.AST | None = None
    cur = node
    while cur in par:
        cur = par[cur]
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = cur
            break
    if scope is None:
        return False
    return any(
        isinstance(sub, ast.Compare)
        and any(isinstance(inner, ast.Name) and inner.id in bound for inner in ast.walk(sub))
        for sub in ast.walk(scope)
    )


_CLOCK_ATTRS: dict[tuple[str | None, str], str] = {
    ("time", "time"): "wall",
    ("time", "monotonic"): "mono",
    ("time", "perf_counter"): "perf",
    ("timeit", "default_timer"): "perf",
}

# Datei-Lebenszyklus (TTL/Abllauf): Dort ist Wall-Clock die einzig korrekte Uhr,
# weil die verglichenen Werte persistiert sind (Prüfpunkte, Caches, Fingerprints).
_FILE_LIFECYCLE_HINTS: tuple[str, ...] = (
    "cleanup",
    "clean_up",
    "expire",
    "purge",
    "prune",
    "reap",
    "sweep",
    "evict",
    "delete",
    "remove",
    "gc_",
)
_FILE_LIFECYCLE_CALLS: tuple[str, ...] = (
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.listdir",
    "os.scandir",
    "os.path.",
    "shutil.",
    ".glob(",
    ".unlink(",
    "getmtime",
    "getctime",
)


def _clock_call(node: ast.AST) -> str | None:
    """Uhr-Typ eines Aufrufs (wall/mono/perf) oder None."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return _CLOCK_ATTRS.get((getattr(node.func.value, "id", None), node.func.attr))
    return None


def _clock_target_key(node: ast.AST) -> str | None:
    """Schlüssel eines Ziels: Name oder Attributname (Subscripts werden ausgelassen)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"attr:{node.attr}"
    return None


def _enclosing_function(node: ast.AST, par: dict[ast.AST, ast.AST]) -> ast.AST | None:
    cur = node
    while cur in par:
        cur = par[cur]
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur
    return None


def _is_file_lifecycle_scope(node: ast.AST, par: dict[ast.AST, ast.AST]) -> bool:
    """True, wenn die Entscheidung in einer TTL-/Datei-Lebenszyklus-Funktion liegt."""
    fn = _enclosing_function(node, par)
    if fn is None:
        return False
    if any(hint in fn.name.lower() for hint in _FILE_LIFECYCLE_HINTS):
        return True
    source = ast.unparse(fn)
    return any(call in source for call in _FILE_LIFECYCLE_CALLS)


def _classify_wallclock(tree: ast.AST) -> tuple[list[int], list[int], int]:
    """Klassifiziert time.time()-Vorkommen.

    Rückgabe: (Audio-relevante Entscheidungs-Zeilen, TTL-Haushalts-Zeilen,
    Mess-Zähler). Nur die erste Liste ist ein §G5-Verstoß.
    """
    par = _parents(tree)
    decision_lines: list[int] = []
    ttl_lines: list[int] = []
    measurement = 0
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "time"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "time"
        ):
            continue
        if _wallclock_in_decision(node, par):
            if _is_file_lifecycle_scope(node, par):
                ttl_lines.append(node.lineno)
            else:
                decision_lines.append(node.lineno)
        else:
            measurement += 1
    return decision_lines, ttl_lines, measurement


def _clock_epoch_findings(tree: ast.AST, lines: list[str]) -> list[tuple[int, str]]:
    """Findet gemischte Uhr-Epochen in einer Rechnung (.github/VERBOTEN.md).

    Muster: `time.monotonic() - fp.last_updated`, wobei `last_updated` an anderer
    Stelle aus `time.time()` gespeist wird (oder umgekehrt). Epochen sind dann
    nicht vergleichbar — das Ergebnis ist ein sinnloser Wert (Produktionsbefund:
    Ablauf-Prüfungen greifen nie, Budgets kippen).

    Bewusst nicht gemeldet: Namen ohne bekannte Uhr (Parameter, Importe) und
    Aufrufe ohne Epochen-Bezug — die Regel ist eine Epochen-Prüfung, keine
    Namenssuche.
    """
    par = _parents(tree)
    file_map: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            clocks = {c for sub in ast.walk(node.value) if (c := _clock_call(sub))}
            if not clocks:
                continue
            for target in node.targets:
                key = _clock_target_key(target)
                if key:
                    file_map.setdefault(key, set()).update(clocks)

    def _resolve(node: ast.AST) -> set[str] | None:
        clock = _clock_call(node)
        if clock:
            return {clock}
        key = _clock_target_key(node)
        if key is None:
            return None
        fn = _enclosing_function(node, par)
        local: set[str] = set()
        cur = fn
        while cur is not None:
            args = [a.arg for a in getattr(getattr(cur, "args", None), "args", [])]
            args += [a.arg for a in getattr(getattr(cur, "args", None), "posonlyargs", [])]
            if key in args:
                return None  # Parameter: Uhr unbekannt → nicht melden
            for sub in ast.walk(cur):
                if isinstance(sub, ast.Assign):
                    if any(_clock_target_key(t) == key for t in sub.targets):
                        for inner in ast.walk(sub.value):
                            inner_clock = _clock_call(inner)
                            if inner_clock:
                                local.add(inner_clock)
            if local:
                return local
            cur = _enclosing_function(cur, par)
        return set(file_map[key]) if key in file_map else None

    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub)):
            continue
        left, right = _resolve(node.left), _resolve(node.right)
        if not left or not right:
            continue
        if len(left) == 1 and left == right:
            continue
        hits.append(
            (
                node.lineno,
                f"gemischte Uhr-Epochen: {sorted(left)} vs. {sorted(right)} — "
                f"Epochen sind nicht vergleichbar (Zeile: {lines[node.lineno - 1].strip()[:80]})",
            )
        )
    # Deduplizieren (dieselbe Zeile kann mehrfach auftreten)
    seen: set[int] = set()
    unique: list[tuple[int, str]] = []
    for lineno, message in hits:
        if lineno in seen:
            continue
        seen.add(lineno)
        unique.append((lineno, message))
    return unique


def _append_capped(
    findings: list[WeaknessFinding],
    suppressed: dict[str, int],
    rule: WeaknessRule,
    rel_posix: str,
    linenos: list[int],
    lines: list[str],
) -> None:
    """Fügt Befunde gedeckelt hinzu — der Rest bleibt transparent ausgewiesen."""
    for lineno in linenos[:_PER_FILE_AST_CAP]:
        findings.append(
            WeaknessFinding(
                rule_id=rule.rule_id,
                severity=rule.severity,
                file=rel_posix,
                line=lineno,
                evidence=_evidence(lines, lineno),
                spec_ref=rule.spec_ref,
                recommendation=rule.recommendation,
            )
        )
    if len(linenos) > _PER_FILE_AST_CAP:
        suppressed[rule.rule_id] = suppressed.get(rule.rule_id, 0) + (len(linenos) - _PER_FILE_AST_CAP)


def _check_bridge_imports(rel_posix: str, text: str, lines: list[str]) -> list[WeaknessFinding]:
    """§V4 (copilot-instructions.md): UI/Frontend (Aurik10/, cli/) darf backend/core nie direkt importieren."""
    if not rel_posix.startswith(("Aurik10/", "cli/")):
        return []
    rule = RULES["bridge_import_violation"]
    findings: list[WeaknessFinding] = []
    for m in _BRIDGE_IMPORT_RX.finditer(blank_noncode(text)):
        line_no = text.count("\n", 0, m.start()) + 1
        findings.append(
            WeaknessFinding(
                rule_id=rule.rule_id,
                severity=rule.severity,
                file=rel_posix,
                line=line_no,
                evidence=_evidence(lines, line_no),
                spec_ref=rule.spec_ref,
                recommendation=rule.recommendation,
            )
        )
    return findings


def _check_dither(rel_posix: str, text: str, lines: list[str]) -> list[WeaknessFinding]:
    """§V5 (copilot-instructions.md): Audio-Quantisierung (bit_depth < 32) ohne Dither.

    Der Wortlaut verbietet Integer-Quantisierung von AUDIO. Der Kontext-Test
    liest deshalb bewusst den ROHTEXT (±6 Zeilen): im Repo trägt jeder legitime
    Nicht-Audio-Cast (Maske/Index/Overflow-Prüfung) die dokumentierte Ausnahme
    „# §V5 (copilot-instructions.md) Dither applied at export" — diese
    Kommentar-Ausnahme ist der reguläre, sichtbare Freistellungsweg. Die
    CAST-Position selbst wird dagegen im Prosa-befreiten Code gesucht, damit
    Docstring-Beispiele keinen Befund erzeugen.
    """
    if not rel_posix.startswith(("backend/", "denker/")):
        return []
    rule = RULES["dither_missing_int_conversion"]
    findings: list[WeaknessFinding] = []
    for m in _INT_CAST_RX.finditer(blank_noncode(text)):
        line_no = text.count("\n", 0, m.start()) + 1
        window = " ".join(lines[max(0, line_no - 7) : min(len(lines), line_no + 6)]).lower()
        if any(token in window for token in _DITHER_CONTEXT_TOKENS):
            continue
        findings.append(
            WeaknessFinding(
                rule_id=rule.rule_id,
                severity=rule.severity,
                file=rel_posix,
                line=line_no,
                evidence=_evidence(lines, line_no),
                spec_ref=rule.spec_ref,
                recommendation=rule.recommendation,
            )
        )
    return findings


def _check_ast_rules(
    rel_posix: str,
    tree: ast.AST,
    lines: list[str],
    suppressed: dict[str, int] | None = None,
) -> list[WeaknessFinding]:
    """§V6 (copilot-instructions.md) Silent-Fallbacks + bare except (AST-basiert, pro Datei gedeckelt).

    Befunde jenseits des Pro-Datei-Caps werden nicht verschwiegen, sondern
    im ``suppressed``-Zähler ausgewiesen (Transparenz statt Logflut).
    """
    if not rel_posix.startswith(("backend/", "denker/")):
        return []
    findings: list[WeaknessFinding] = []
    per_file_count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            rule = RULES["bare_except"]
        elif _has_logging_call(node) or not _handler_returns_value(node):
            continue
        else:
            rule = RULES["silent_fallback_no_log"]
        if per_file_count >= _PER_FILE_AST_CAP:
            if suppressed is not None:
                suppressed[rule.rule_id] = suppressed.get(rule.rule_id, 0) + 1
            continue
        findings.append(
            WeaknessFinding(
                rule_id=rule.rule_id,
                severity=rule.severity,
                file=rel_posix,
                line=node.lineno,
                evidence=_evidence(lines, node.lineno),
                spec_ref=rule.spec_ref,
                recommendation=rule.recommendation,
            )
        )
        per_file_count += 1
    return findings


def _check_module_logger(rel_posix: str, text: str) -> WeaknessFinding | None:
    """Logger-Pflicht: Fehlerpfade ohne getLogger() im Modul (Prosa zählt nicht)."""
    if not rel_posix.startswith(("backend/", "denker/")):
        return None
    code = blank_noncode(text)
    has_error_path = "except Exception" in code or "subprocess." in code
    if not has_error_path:
        return None
    if "getLogger(" in code:
        return None
    rule = RULES["module_logger_missing"]
    return WeaknessFinding(
        rule_id=rule.rule_id,
        severity=rule.severity,
        file=rel_posix,
        line=1,
        evidence="Modul enthält Fehlerpfade, aber kein logging.getLogger()",
        spec_ref=rule.spec_ref,
        recommendation=rule.recommendation,
    )


def _check_phase_nan_guard(rel_posix: str, text: str) -> WeaknessFinding | None:
    """§0a: Jede Phase braucht NaN/Inf-Schutz auf dem Ausgabe-Audio."""
    if not re.match(r"backend/core/phases/phase_[0-9]+_", rel_posix):
        return None
    code = blank_noncode(text)
    if "numpy" not in code and "np." not in code:
        return None
    if any(token in code for token in _NAN_GUARD_TOKENS):
        return None
    rule = RULES["nan_inf_guard_missing"]
    return WeaknessFinding(
        rule_id=rule.rule_id,
        severity=rule.severity,
        file=rel_posix,
        line=1,
        evidence="Phase nutzt numpy, aber kein isfinite/nan_to_num/isnan/isinf",
        spec_ref=rule.spec_ref,
        recommendation=rule.recommendation,
    )


def scan_workspace(workspace: Path, *, max_findings: int = 200) -> ScanResult:
    """Führt den kompletten Schwachstellen-Scan über die Workspace aus.

    Die Sortierung erfolgt nach Schweregrad (critical zuerst), dann Datei/Zeile,
    damit bei Begrenzung auf ``max_findings`` niemals kritische Befunde verloren gehen.
    """
    workspace = Path(workspace).resolve()
    started = time.perf_counter()
    findings: list[WeaknessFinding] = []
    files_scanned = 0
    print_counts: dict[str, int] = {}
    # Alle Treffer (auch unter Schwelle), um die unterdrückte Anzahl ehrlich
    # auszuweisen statt still zu verschweigen (Transparenz).
    print_hits_all: dict[str, int] = {}
    wallclock_measurements = 0
    suppressed: dict[str, int] = {}

    for path, rel in iter_python_files(workspace):
        if _is_test_path(rel) or _is_vendor_path(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        files_scanned += 1
        lines = text.splitlines()
        code = blank_noncode(text)

        findings.extend(_check_bridge_imports(rel, text, lines))
        findings.extend(_check_dither(rel, text, lines))

        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None
        if tree is not None:
            findings.extend(_check_ast_rules(rel, tree, lines, suppressed))
            if rel.startswith(("backend/core/", "denker/")):
                # §G5 (GEBOTE.md): nur Wall-Clock in ENTSCHEIDUNGSLOGIK ist ein Verstoß; TTL-/
                # Datei-Lebenszyklus-Entscheidungen sind davon abgegrenzt.
                decision_lines, ttl_lines, measurements = _classify_wallclock(tree)
                wallclock_measurements += measurements
                _append_capped(
                    findings,
                    suppressed,
                    RULES["determinism_time_usage"],
                    rel,
                    sorted(decision_lines),
                    lines,
                )
                _append_capped(
                    findings,
                    suppressed,
                    RULES["wallclock_ttl_housekeeping"],
                    rel,
                    sorted(ttl_lines),
                    lines,
                )
                mismatch_rule = RULES["walltime_clock_mismatch"]
                for lineno, message in _clock_epoch_findings(tree, lines):
                    findings.append(
                        WeaknessFinding(
                            rule_id=mismatch_rule.rule_id,
                            severity=mismatch_rule.severity,
                            file=rel,
                            line=lineno,
                            evidence=message,
                            spec_ref=mismatch_rule.spec_ref,
                            recommendation=mismatch_rule.recommendation,
                        )
                    )

        finding = _check_module_logger(rel, text)
        if finding is not None:
            findings.append(finding)
        finding = _check_phase_nan_guard(rel, text)
        if finding is not None:
            findings.append(finding)

        if rel.startswith("backend/core/") and not rel.startswith(_PRINT_TOOL_PREFIXES):
            # print() nur im echten Code zählen (Kommentare/Docstrings sind Prosa).
            print_hits = len(_PRINT_RX.findall(code))
            if print_hits >= 1:
                print_hits_all[rel] = print_hits
            if print_hits >= _PRINT_MIN_COUNT:
                print_counts[rel] = print_hits

    # Transparenz §G5 (GEBOTE.md): Mess-/Zeitstempel-Nutzung von time.time() ist kein Verstoß,
    # wird aber weiterhin ehrlich ausgewiesen (statt still zu verschwinden).
    if wallclock_measurements > 0:
        suppressed["determinism_time_usage"] = suppressed.get("determinism_time_usage", 0) + wallclock_measurements

    # Aggregierte Befunde (Pro-Datei-Zählung, Top-N — keine Logflut).
    rule = RULES["print_in_production"]
    top_files = sorted(print_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:_AGGREGATED_TOP_N]
    reported_occurrences = 0
    for rel, count in top_files:
        reported_occurrences += count
        findings.append(
            WeaknessFinding(
                rule_id=rule.rule_id,
                severity=rule.severity,
                file=rel,
                line=1,
                evidence=f"{count} Vorkommen von print()",
                spec_ref=rule.spec_ref,
                recommendation=rule.recommendation,
            )
        )
    # Transparenz: Treffer unter Schwelle + Treffer jenseits Top-N ausweisen.
    _suppressed_print = sum(print_hits_all.values()) - reported_occurrences
    if _suppressed_print > 0:
        suppressed[rule.rule_id] = suppressed.get(rule.rule_id, 0) + _suppressed_print

    findings.sort(key=lambda f: (-_SEVERITY_RANK.get(f.severity, 0), f.rule_id, f.file, f.line))
    if len(findings) > max_findings:
        suppressed["truncated_max_findings"] = len(findings) - max_findings
        findings = findings[:max_findings]

    per_rule_counts: dict[str, int] = {}
    for finding in findings:
        per_rule_counts[finding.rule_id] = per_rule_counts.get(finding.rule_id, 0) + 1

    return ScanResult(
        findings=findings,
        files_scanned=files_scanned,
        duration_s=round(time.perf_counter() - started, 3),
        per_rule_counts=dict(sorted(per_rule_counts.items())),
        suppressed=dict(sorted(suppressed.items())),
    )


def summarize_findings(result: ScanResult, *, top_n: int = 5) -> dict[str, Any]:
    """Kompakte Zusammenfassung für Live-Snapshots (Watchdog-Payload)."""
    by_severity = dict.fromkeys(("critical", "high", "medium", "low"), 0)
    for finding in result.findings:
        if finding.severity in by_severity:
            by_severity[finding.severity] += 1
    return {
        "total": len(result.findings),
        **by_severity,
        "files_scanned": result.files_scanned,
        "top_findings": [
            {"rule_id": f.rule_id, "severity": f.severity, "file": f.file, "line": f.line}
            for f in result.findings[:top_n]
        ],
    }


def write_json_report(path: Path, result: ScanResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "files_scanned": result.files_scanned,
        "duration_s": result.duration_s,
        "per_rule_counts": result.per_rule_counts,
        "suppressed": dict(result.suppressed),
        "total_findings": len(result.findings),
        "findings": [asdict(f) for f in result.findings],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown_report(path: Path, result: ScanResult) -> None:
    """Menschlich klarer Report: Befunde nach Schweregrad gruppiert."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    by_severity: dict[str, list[WeaknessFinding]] = {"critical": [], "high": [], "medium": [], "low": []}
    for finding in result.findings:
        by_severity.setdefault(finding.severity, []).append(finding)

    lines: list[str] = [
        "# Code-Schwachstellen-Report (Watchdog)",
        "",
        f"- Erzeugt: {datetime.now().isoformat()}",
        f"- Geprüfte Dateien: {result.files_scanned} (Dauer: {result.duration_s}s)",
        f"- Befunde gesamt: **{len(result.findings)}**",
    ]
    for sev in ("critical", "high", "medium", "low"):
        lines.append(f"  - {sev}: {len(by_severity.get(sev, []))}")
    if result.per_rule_counts:
        lines.append("- Pro Regel: " + ", ".join(f"{k}={v}" for k, v in result.per_rule_counts.items()))
    if result.suppressed:
        lines.append(
            "- Unterdrückte Befunde (unter Schwelle/Kappung, bewusst sichtbar): "
            + ", ".join(f"{k}={v}" for k, v in sorted(result.suppressed.items()))
        )
        lines.append(
            "  (Schwellen: AST-Cap 3/Datei, print ≥ 3 im echten Code, Top-N 10, max_findings. "
            "determinism_time_usage listet NUR Wall-Clock in Entscheidungslogik (§G5 (GEBOTE.md)); Messungen und "
            "Zeitstempel sind kein Verstoß — persistierte Zeitstempel MÜSSEN Wall-Clock bleiben, "
            "time.monotonic() ist prozesslokal und über Prozess-/Session-Grenzen bedeutungslos. "
            "Unterdrückt heißt nicht: nicht vorhanden.)"
        )

    for sev in ("critical", "high", "medium", "low"):
        items = by_severity.get(sev, [])
        if not items:
            continue
        lines += ["", f"## {sev.upper()} ({len(items)})", ""]
        for finding in items:
            rule = RULES.get(finding.rule_id)
            title = rule.title if rule else ""
            lines.append(f"- `{finding.file}:{finding.line}` — **{finding.rule_id}** ({finding.spec_ref})")
            if title:
                lines.append(f"  - {title}")
            lines.append(f"  - Evidenz: `{finding.evidence}`")
            lines.append(f"  - Empfehlung: {finding.recommendation}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Statischer Code-Schwachstellen-Scan (Watchdog)")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--json-out", default="audit/code_weakness_report.json")
    parser.add_argument("--md-out", default="audit/code_weakness_report.md")
    parser.add_argument("--max-findings", type=int, default=200)
    parser.add_argument(
        "--fail-on",
        choices=("critical", "high", "none"),
        default="none",
        help="Exit 1, wenn Befunde der angegebenen Schwere existieren (CI-Modus).",
    )
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    result = scan_workspace(workspace, max_findings=args.max_findings)

    json_out = workspace / args.json_out if not Path(args.json_out).is_absolute() else Path(args.json_out)
    md_out = workspace / args.md_out if not Path(args.md_out).is_absolute() else Path(args.md_out)
    write_json_report(json_out, result)
    write_markdown_report(md_out, result)

    summary = summarize_findings(result)
    _supp_total = sum(result.suppressed.values())
    print(
        f"[code-weakness-scan] files={result.files_scanned} "
        f"findings={summary['total']} critical={summary['critical']} high={summary['high']} "
        f"medium={summary['medium']} low={summary['low']} "
        f"suppressed={_supp_total} duration={result.duration_s}s"
    )
    for finding in result.findings[:15]:
        print(f"  [{finding.severity}] {finding.file}:{finding.line} {finding.rule_id}")
    if len(result.findings) > 15:
        print(f"  … {len(result.findings) - 15} weitere Befunde in {json_out.name}/{md_out.name}")

    if args.fail_on == "critical" and summary["critical"] > 0:
        return 1
    if args.fail_on == "high" and (summary["critical"] > 0 or summary["high"] > 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
