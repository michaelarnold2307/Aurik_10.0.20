#!/usr/bin/env python3
"""Wohlklang-Vertrag-Gate (fail-closed) — §v10.802/§G8/§G9 (copilot-instructions.md).

Vertrag: ``.github/WOHLKLANG_CLAIMS.md``

Warum dieses Gate existiert (Produktionsbefund 2026-10-06): Eine
Qualitätsstufe (BigVGAN HR-V1) wurde **produktiv aktiviert**, obwohl

  * der Beleg auf **20 s GPU-Material** beruhte, der Produktionspfad aber
    **CPU/ONNX** ist (13,2× RT je Passage = 330 % des Phasen-Budgets), und
  * die Aktenlage weiter „Flag OFF" sagte.

Der Fehler war nicht das Modell, sondern die **Beweislage**. Dieses Gate
bindet jede klangverändernde Stufe an fünf Belege **und** an den tatsächlichen
Code-Zustand — damit „produktiv aktiv" nie wieder ohne Nachweis und nie wieder
im Widerspruch zur Akte entsteht.

Regeln (ERROR ⇒ Exit 1):

  E1  Jede Datenzeile hat 10 Zellen und eine eindeutige ID ``W-<n>``.
  E2  Jeder in Backticks zitierte **Pfad** (enthält ``/``) muss **existieren**
      — Evidenzpflicht wie im SOTA-Defizit-Register; ``OFFEN``/``—`` sind die
      erlaubten Marker für fehlende Belege.
  E3  Status ``aktiviert`` ⇒ **kein** ``OFFEN`` in einer Belegzelle, Baseline
      nennt eine Version, Budget nennt eine RT-Zahl.
  E4  Status ``ausnahme`` ⇒ Zelle nennt ``Grund:`` (≥ 20 Zeichen) **und**
      ``Review:`` — sichtbar, befristet, kein stiller Freibrief (Warnung).
  E5  **Schalter-Abgleich (§G9 copilot-instructions.md):** der Literalwert
      ``NAME: bool = ...`` im genannten Modul muss dem deklarierten Soll
      entsprechen (``AN`` ↔ True, ``AUS`` ↔ False). Genau dieser Abgleich hätte
      den Drift vom 2026-09-27 verhindert.
  E6  Status ``gesperrt`` ⇒ Soll muss ``AUS`` sein (eine gesperrte Stufe darf
      nicht eingeschaltet sein).
  P1  In den Evidenz-Harnessen muss jedes ``models/…``-Literal existieren — ein
      defekter Harness produziert keine gültigen Belege (Phantomziel D-K3-1).
  P2  Bericht (kein Fail): Anzahl nicht existenter ``models/…``-Literale im
      Produktionscode.

Nutzung:
  python scripts/wohlklang_gate.py            # fail-closed
  python scripts/wohlklang_gate.py --json     # Maschinen-Ausgabe
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import logging
import re
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / ".github" / "WOHLKLANG_CLAIMS.md"
VERSION_FILE = ROOT / "backend" / "core" / "version.py"

STATUS_ENUM: tuple[str, ...] = ("aktiviert", "ausnahme", "gesperrt")
SOLL_ENUM: tuple[str, ...] = ("AN", "AUS")
N_COLUMNS = 10
OPEN_MARKERS: tuple[str, ...] = ("OFFEN", "—")

# Evidenz-Harneske: erzeugen die Belege, gelten daher selbst als beweispflichtig.
EVIDENCE_HARNESSES: tuple[str, ...] = (
    "scripts/ab_test_exports.py",
    "scripts/validate_hr_v1.py",
    "scripts/eval_scnet_vs_mdx23c.py",
)
# Produktionscode für die Phantom-Berichterstattung (P2).
PRODUCTION_ROOTS: tuple[str, ...] = ("backend", "plugins", "denker", "scripts", "cli", "Aurik10")

ROW_ID_RE = re.compile(r"^W-\d{1,3}$")
BACKTICK_RE = re.compile(r"`([^`]+)`")
SWITCH_RE = re.compile(r"^(?P<path>[\w./-]+)::(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")
BOOL_LITERAL_TMPL = r"^\s*{name}\s*:\s*bool\s*=\s*(?P<value>True|False)\b"
VERSION_RE = re.compile(r'__version__\s*=\s*"([^"]+)"')
BUDGET_RT_RE = re.compile(r"\d+[.,]?\d*\s*[x×]\s*RT", re.IGNORECASE)
MODEL_PATH_RE = re.compile(r"models/[\w./-]+\.(?:onnx|pth|pyt|pt|ckpt|ts|joblib|npy|safetensors)")
# Codepfade: Die Negativ-Lookbehind verhindert Treffer am Ende eines längeren
# Namens (Befund 2026-10-07: `.../all_public_uvr_models/model_bs_roformer_…ckpt`
# ist ein URL-Suffix, kein lokaler Pfad — der Treffer begann mitten im Wort).
CODE_MODEL_PATH_RE = re.compile(r"(?<![\w/])models/[\w./-]+\.(?:onnx|pth|pyt|pt|ckpt|ts|joblib|npy|safetensors)")
SEPARATOR_RE = re.compile(r":?-{3,}:?")
_MIN_GRUND = 20


@dataclass
class Finding:
    """Ein Befund des Gates."""

    severity: str  # ERROR | WARNING | INFO
    rule: str
    message: str
    line_no: int = 0


@dataclass
class Claim:
    """Eine Vertragszeile (klangverändernde Stufe)."""

    id: str
    stufe: str
    switch: str
    soll: str
    songs: str
    production: str
    baseline: str
    blind_ab: str
    budget: str
    status: str
    line_no: int
    evidence_cells: list[str] = field(default_factory=list)
    raw_cells: list[str] = field(default_factory=list)


def _cells(line: str) -> list[str]:
    """Zerlegt eine Markdown-Tabellenzeile in ihre Zellen."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _read(path: Path) -> str:
    """Liest eine Textdatei (leer, wenn sie fehlt)."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def current_version() -> str:
    """Kanonische Version aus ``backend/core/version.py`` (kein Import)."""
    match = VERSION_RE.search(_read(VERSION_FILE))
    return match.group(1) if match else ""


def parse_claims(text: str) -> list[Claim]:
    """Liest alle Vertragszeilen ``| W-… |`` aus dem Markdown."""
    claims: list[Claim] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if not line.startswith("|"):
            continue
        cells = _cells(line)
        if not cells or not ROW_ID_RE.match(cells[0]):
            continue
        if len(cells) < 3 or SEPARATOR_RE.match(cells[1]):
            continue
        claims.append(
            Claim(
                id=cells[0],
                stufe=cells[1] if len(cells) > 1 else "",
                switch=(cells[2] if len(cells) > 2 else "").strip().strip("`"),
                soll=cells[3] if len(cells) > 3 else "",
                songs=cells[4] if len(cells) > 4 else "",
                production=cells[5] if len(cells) > 5 else "",
                baseline=cells[6] if len(cells) > 6 else "",
                blind_ab=cells[7] if len(cells) > 7 else "",
                budget=cells[8] if len(cells) > 8 else "",
                status=cells[9] if len(cells) > 9 else "",
                line_no=idx,
                evidence_cells=cells[4:10],
                raw_cells=cells,
            )
        )
    return claims


def backticked_paths(cell: str, *, skip_switch: bool = False) -> list[str]:
    """Alle in Backticks zitierten **Pfade** (enthalten ``/``) einer Zelle."""
    out: list[str] = []
    for raw in BACKTICK_RE.findall(cell):
        candidate = raw.strip()
        if "::" in candidate and skip_switch:
            candidate = candidate.split("::", 1)[0].strip()
        if "/" in candidate and candidate not in OPEN_MARKERS:
            out.append(candidate.rstrip("/"))
    return out


def switch_state(rel_path: str, name: str) -> str | None:
    """Liest den Literalwert ``NAME: bool = True|False`` statisch (kein Import)."""
    text = _read(ROOT / rel_path)
    if not text:
        return None
    match = re.search(BOOL_LITERAL_TMPL.format(name=re.escape(name)), text, flags=re.MULTILINE)
    return match.group("value") if match else None


def _model_tree_present() -> bool:
    """Ist der ``models/``-Baum vorhanden?

    ``models/*`` ist per ``.gitignore`` ausgenommen: Auf einem nackten Checkout
    existieren die Artefakte nicht — dort dürfen Beleg- und Phantom-Prüfungen
    für Modellpfade **nicht** blocken (sonst wäre das Gate maschinen- statt
    vertragsabhängig).
    """
    return (ROOT / "models").is_dir()


def _path_missing(path: str) -> bool:
    """Fehlt der Pfad? (Modellpfade nur bei vorhandenem ``models/``-Baum geprüft.)"""
    if path.startswith("models/") and not _model_tree_present():
        return False
    return not (ROOT / path).exists()


def _missing_paths(paths: list[str]) -> list[str]:
    """Nicht existierende Repo-Pfade (nach ``_path_missing``-Regel)."""
    return [p for p in paths if _path_missing(p)]


def check_contract(claims: list[Claim], version: str) -> list[Finding]:
    """E1–E6: Struktur, Evidenzpflicht und Schalter-Abgleich."""
    findings: list[Finding] = []
    seen: set[str] = set()
    for claim in claims:
        if claim.id in seen:
            findings.append(Finding("ERROR", "E1", f"Doppelte ID {claim.id}", claim.line_no))
        seen.add(claim.id)
        if len(claim.raw_cells) != N_COLUMNS:
            findings.append(
                Finding(
                    "ERROR",
                    "E1",
                    f"{claim.id}: {len(claim.raw_cells)} Zellen statt {N_COLUMNS}",
                    claim.line_no,
                )
            )

        # E2 — Evidenzpflicht für zitierte Pfade
        cited = backticked_paths(claim.switch, skip_switch=True)
        for cell in claim.evidence_cells:
            cited += backticked_paths(cell)
        missing = _missing_paths(cited)
        if missing:
            findings.append(
                Finding(
                    "ERROR",
                    "E2",
                    f"{claim.id}: zitierte Pfade fehlen: {', '.join(sorted(set(missing)))}",
                    claim.line_no,
                )
            )

        # Soll/Status-Vokabular
        if claim.soll not in SOLL_ENUM:
            findings.append(Finding("ERROR", "E1", f"{claim.id}: Soll '{claim.soll}' ∉ {SOLL_ENUM}", claim.line_no))
        if claim.status.split(" ")[0] not in STATUS_ENUM:
            findings.append(
                Finding("ERROR", "E1", f"{claim.id}: Status '{claim.status}' ∉ {STATUS_ENUM}", claim.line_no)
            )
        status = claim.status.split(" ")[0]

        # E3 — aktiviert braucht vollständige Belege
        if status == "aktiviert":
            offen = [i for i, cell in enumerate(claim.evidence_cells, start=5) if "OFFEN" in cell.upper()]
            if offen:
                findings.append(
                    Finding("ERROR", "E3", f"{claim.id}: aktiviert, aber OFFEN in Zelle(n) {offen}", claim.line_no)
                )
            if version and version not in claim.baseline:
                findings.append(
                    Finding("ERROR", "E3", f"{claim.id}: Baseline nennt nicht die Version {version}", claim.line_no)
                )
            if not BUDGET_RT_RE.search(claim.budget):
                findings.append(Finding("ERROR", "E3", f"{claim.id}: Budget ohne RT-Zahl", claim.line_no))

        # E4 — Ausnahme muss begründet und befristet sein
        if status == "ausnahme":
            if "Grund:" not in claim.status:
                findings.append(Finding("ERROR", "E4", f"{claim.id}: ausnahme ohne 'Grund:'", claim.line_no))
            else:
                grund = claim.status.split("Grund:", 1)[1].split("Review:", 1)[0].strip()
                if len(grund) < _MIN_GRUND:
                    findings.append(
                        Finding(
                            "ERROR", "E4", f"{claim.id}: Grund zu kurz ({len(grund)} < {_MIN_GRUND})", claim.line_no
                        )
                    )
            if "Review:" not in claim.status:
                findings.append(Finding("ERROR", "E4", f"{claim.id}: ausnahme ohne 'Review:'", claim.line_no))
            else:
                findings.append(Finding("WARNING", "E4", f"{claim.id}: Ausnahme aktiv — {claim.status}", claim.line_no))

        # E5 — Schalter-Abgleich gegen den Code
        match = SWITCH_RE.match(claim.switch)
        if not match:
            findings.append(
                Finding(
                    "ERROR", "E5", f"{claim.id}: Schalter '{claim.switch}' nicht als Datei::Name lesbar", claim.line_no
                )
            )
        else:
            actual = switch_state(match.group("path"), match.group("name"))
            if actual is None:
                findings.append(
                    Finding(
                        "ERROR",
                        "E5",
                        f"{claim.id}: Schalter '{claim.switch}' nicht im Code gefunden (Literal 'NAME: bool = …')",
                        claim.line_no,
                    )
                )
            else:
                expected = "AN" if actual == "True" else "AUS"
                if expected != claim.soll:
                    findings.append(
                        Finding(
                            "ERROR",
                            "E5",
                            f"{claim.id}: Drift — Code sagt {expected} ({match.group('name')} = {actual}), Vertrag sagt {claim.soll}",
                            claim.line_no,
                        )
                    )

        # E6 — gesperrt ⇒ Soll AUS
        if status == "gesperrt" and claim.soll != "AUS":
            findings.append(
                Finding("ERROR", "E6", f"{claim.id}: gesperrt, aber Soll {claim.soll} ≠ AUS", claim.line_no)
            )
    return findings


def _prose_string_spans(tree: ast.AST) -> list[tuple[int, int, int, int]]:
    """Positionen aller **Statement-Strings** (nackte String-Ausdrücke).

    Kriterium: ``Expr`` mit ``Constant``-String-Wert. Ein solcher Ausdruck hat in
    Python **keine Laufzeitwirkung** — er kann ausschließlich Dokumentation sein
    (Docstring, verwaister Docstring nach einem Methoden-Rückbau, Kommentarblock).
    Damit ist das Kriterium präziser als „``body[0]`` einer Funktion": Befund
    2026-10-07 in ``plugins/apollo_phase0_integration.py`` — die Klasse
    ``ResembleEnhanceGuard`` trägt ihren echten Docstring **und** darunter den
    zurückgelassenen Docstring der in §v10.19 entfernten Methode; nur ``body[0]``
    zu prüfen ließ den zweiten String als Code durchgehen.
    """
    spans: list[tuple[int, int, int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        if value.end_lineno is None or value.end_col_offset is None:
            continue
        spans.append((value.lineno, value.col_offset, value.end_lineno, value.end_col_offset))
    return spans


def _blank_prose(text: str) -> str:
    """Ersetzt **Kommentare und Statement-Strings** positionsgetreu durch Leerzeichen.

    Bewusst **enger** als ``audit.code_weakness_scanner.blank_noncode``: Jene
    Funktion blankt zusätzlich ALLE String-Literale (sie prüft Log-Aufrufe, nicht
    Pfade). Hier wäre das falsch — ein Modellpfad steht **naturgemäß** in einem
    funktionalen String-Literal; wer alle Strings blankt, löscht genau die
    Verdrahtung, die geprüft werden soll (eigener Fehlversuch 2026-10-07: 53 → 0,
    obwohl nur Prosa entfernt werden sollte). Deshalb: Kommentar und
    Statement-String = Prosa, funktionaler String = Code.

    Zeilen-/Spaltenpositionen bleiben erhalten. Fallback bei Parse-/Tokenizer-
    Fehler: Rohtext (konservativ — mehr Befunde, nie weniger).
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return text
    spans = _prose_string_spans(tree)

    lines = text.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def _offset(row: int, col: int) -> int:
        return starts[row - 1] + col

    def _is_prose_string(tok: tokenize.TokenInfo) -> bool:
        return any(
            tok.start >= (start_row, start_col) and tok.end <= (end_row, end_col)
            for start_row, start_col, end_row, end_col in spans
        )

    chars = list(text)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT or (tok.type == tokenize.STRING and _is_prose_string(tok)):
                for idx in range(_offset(*tok.start), _offset(*tok.end)):
                    if chars[idx] != "\n":
                        chars[idx] = " "
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text
    return "".join(chars)


def _scan_code_paths(text: str) -> list[str]:
    """Modell-Pfad-Literale im **ausführbaren Code** (Kommentare/Docstrings geblankt).

    Grund (Befund D-K3-5, 2026-10-07): Der P2-Lauf meldete 53 „nicht existente
    ``models/…``-Referenzen in Produktionscode". Die Stichprobe zeigte, dass der
    Großteil **Prosa** war („Modell: ``models/…``" in Docstrings, Fallback-Listen
    in Kommentaren). Ein Pfad in einer Beschreibung ist keine Verdrahtung. Ohne
    diese Trennung ist der Bericht nicht actionierbar — und P1 (fail-closed) hätte
    bei einer bloßen Docstring-Erwähnung den Commit blockiert (§G9 (copilot-instructions.md):
    eine Technik, keine parallele Umsetzung; `_blank_prose` ist die engere Variante
    der kanonischen `blank_noncode`).
    """
    return sorted(set(CODE_MODEL_PATH_RE.findall(_blank_prose(text))))


def check_harnesses() -> tuple[list[Finding], int]:
    """P1 (fail-closed) und P2 (Bericht): Phantom-Pfade in Harness/Produktion."""
    findings: list[Finding] = []
    for rel in EVIDENCE_HARNESSES:
        path = ROOT / rel
        if not path.exists():
            findings.append(Finding("ERROR", "P1", f"Evidenz-Harness fehlt: {rel}"))
            continue
        missing = _missing_paths(_scan_code_paths(_read(path)))
        for rel_missing in missing:
            findings.append(
                Finding("ERROR", "P1", f"{rel}: referenziert nicht existierendes Modell-Artefakt '{rel_missing}'")
            )

    phantom_paths: list[str] = []
    checked = 0
    for root_name in PRODUCTION_ROOTS:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            checked += 1
            for ref in _scan_code_paths(_read(path)):
                if _path_missing(ref):
                    phantom_paths.append(f"{path.relative_to(ROOT)} → {ref}")
    findings.append(
        Finding(
            "INFO",
            "P2",
            f"{len(phantom_paths)} nicht existente models/-Referenz(en) im AUSFÜHRBAREN Code von "
            f"{checked} Produktionsdateien (Bericht, kein Fail; Kommentare/Docstrings ausgenommen)",
        )
    )
    for entry in sorted(set(phantom_paths)):
        findings.append(Finding("INFO", "P2", entry))
    return findings, len(phantom_paths)


def main(argv: list[str] | None = None) -> int:
    """Führt das Vertrags-Gate aus (Exit 1 bei ERROR)."""
    parser = argparse.ArgumentParser(description="Wohlklang-Vertrag-Gate")
    parser.add_argument("--json", action="store_true", help="Maschinen-Ausgabe")
    args = parser.parse_args(argv)

    text = _read(CONTRACT)
    if not text:
        print(f"FEHLER: Vertrag fehlt: {CONTRACT.relative_to(ROOT)}", file=sys.stderr)
        return 1

    version = current_version()
    claims = parse_claims(text)
    if not claims:
        print(f"FEHLER: keine Vertragszeilen 'W-…' in {CONTRACT.relative_to(ROOT)}", file=sys.stderr)
        return 1

    findings = check_contract(claims, version)
    harness_findings, phantom_count = check_harnesses()
    findings += harness_findings

    errors = [f for f in findings if f.severity == "ERROR"]
    warnings = [f for f in findings if f.severity == "WARNING"]
    status_counts = {status: sum(1 for c in claims if c.status.split(" ")[0] == status) for status in STATUS_ENUM}
    aktiviert_ohne_ausnahme = status_counts["aktiviert"]

    if args.json:
        print(
            json.dumps(
                {
                    "version": version,
                    "claims": len(claims),
                    "status_counts": status_counts,
                    "aktiviert_mit_vollbeleg_und_ohne_ausnahme": aktiviert_ohne_ausnahme,
                    "phantom_model_refs": phantom_count,
                    "errors": [f.message for f in errors],
                    "warnings": [f.message for f in warnings],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 1 if errors else 0

    for finding in findings:
        where = f":{finding.line_no}" if finding.line_no else ""
        print(f"{finding.severity} [{finding.rule}] {finding.message}{where}")
    print(
        f"\nWohlklang-Vertrag: {len(claims)} Stufen — "
        + ", ".join(f"{k}={v}" for k, v in status_counts.items())
        + f"; Phantom-Referenzen={phantom_count}; Version={version}"
    )
    if errors:
        print(f"\n{len(errors)} Vertragsverstoß/-verstöße — Gate fail-closed (§v10.802 copilot-instructions.md).")
        return 1
    print("\nVertrag erfüllt. Ausnahmen sind als Warnung ausgewiesen (kein stiller Freibrief).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
