#!/usr/bin/env python3
"""SOTA-Defizit-Register-Gate (fail-closed) — §G8/§III.13 (copilot-instructions.md).

Prüft `.github/SOTA_DEFICIT_REGISTER.md` gegen die Akzeptanzkriterien des
Roadmap-Arbeitspakets WP-0 und ergänzt einen Domänen-Abgleich gegen das
kuratierte Modell-Manifest (§G9 (copilot-instructions.md) — eine Quelle):

  (a) Statuswechsel: Status ``geschlossen`` nur mit **existierendem** Beleg-Pfad.
  (b) Domäne ``musik``/``sprache``/``gemischt`` nur mit konkreter, prüfbarer
      Evidenz (§III.13 (copilot-instructions.md) Evidenzpflicht — kein Urteil
      ohne Beleg; ``unbekannt`` ist die ehrliche Angabe).
  (c) Defizit-Klasse ohne Klassendeklaration oder ohne Registereintrag.

Zusätzlich muss **jeder** Eintrag mindestens einen auflösbaren Repo-Pfad in
Backticks zitieren — die unbelegte Domänen-Behauptung ist genau der Defekt,
den dieses Gate verhindert (§V7 (copilot-instructions.md) — Ursache statt
Symptom).

Nutzung:
  python scripts/sota_deficit_gate.py             # fail-closed (Exit 1 bei Verstoß)
  python scripts/sota_deficit_gate.py --json      # Maschinen-Ausgabe
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
REGISTER = ROOT / ".github" / "SOTA_DEFICIT_REGISTER.md"

REQUIRED_CLASSES: tuple[str, ...] = ("K0", "K1", "K2", "K3")
STATUS_ENUM: tuple[str, ...] = ("offen", "in-arbeit", "geschlossen", "bewusst-akzeptiert")
DOMAIN_ENUM: tuple[str, ...] = (
    "musik",
    "sprache",
    "gemischt",
    "audio-allgemein",
    "unueberwacht",
    "unbekannt",
    "—",
)
# Domänen, deren Behauptung eine konkrete Evidenz erzwingt (§III.13 (copilot-instructions.md)).
EVIDENCE_DOMAINS: tuple[str, ...] = ("musik", "sprache", "gemischt")
GENERIC_EVIDENCE_MARKER = "nicht einzeln belegt"
_MIN_ACCEPT_BEGRUENDUNG = 20

ENTRY_ID_RE = re.compile(r"^D-K(\d)-\d{1,2}$")
BACKTICK_RE = re.compile(r"`([^`]+)`")
PATHISH_RE = re.compile(r"^[\w./-]+$")
SEPARATOR_RE = re.compile(r":?-{3,}:?")

HEADER_CELLS: tuple[str, ...] = ("ID", "Artefakt", "Klasse", "Domäne", "Evidenz", "Status", "Blocker", "Beleg")


@dataclass(frozen=True)
class RegisterEntry:
    """Ein Registereintrag `D-<Klasse>-<n>` mit allen Pflichtzellen."""

    id: str
    artifact: str
    klass: str
    domain: str
    evidence: str
    status: str
    blocker: str
    beleg: str
    line_no: int


def _cells(line: str) -> list[str]:
    """Zerlegt eine Markdown-Tabellenzeile in ihre Zellen."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator(cells: list[str]) -> bool:
    """Erkennt die `| --- | --- |`-Trennzeile einer Markdown-Tabelle."""
    meaningful = [c for c in cells if c]
    return bool(meaningful) and all(SEPARATOR_RE.fullmatch(c) for c in meaningful)


def parse_register(text: str) -> tuple[list[RegisterEntry], list[str]]:
    """Liest Einträge und deklarierte Defizit-Klassen aus dem Register.

    Rückgabe: ``(entries, declared_classes)``.

    Raises:
        ValueError: wenn eine Eintragszeile nicht genau ``HEADER_CELLS`` Zellen hat.
    """
    entries: list[RegisterEntry] = []
    declared: list[str] = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        if not raw.lstrip().startswith("|"):
            continue
        cells = _cells(raw)
        if _is_separator(cells) or len(cells) < 3:
            continue
        head = cells[0]
        if head in REQUIRED_CLASSES:
            declared.append(head)
            continue
        if not ENTRY_ID_RE.match(head):
            continue
        if len(cells) != len(HEADER_CELLS):
            raise ValueError(f"Zeile {line_no}: Eintrag {head} hat {len(cells)} Zellen, erwartet {len(HEADER_CELLS)}")
        artifact, klass, domain, evidence, status, blocker, beleg = cells[1:]
        entries.append(RegisterEntry(head, artifact, klass, domain, evidence, status, blocker, beleg, line_no))
    return entries, declared


def _backtick_paths(text: str) -> list[str]:
    """Liefert pfad-artige Backtick-Token aus einem Zellentext."""
    return [tok for tok in BACKTICK_RE.findall(text) if "/" in tok and PATHISH_RE.match(tok)]


def _path_exists(token: str, root: Path) -> bool:
    """Prüft einen repo-relativen Pfad (Datei oder Verzeichnis) auf Existenz."""
    return (root / token).exists()


def _has_resolvable_path(text: str, root: Path) -> bool:
    """Wahr, wenn mindestens ein zitierter Backtick-Pfad tatsächlich existiert."""
    return any(_path_exists(tok, root) for tok in _backtick_paths(text))


def _first_path(cell: str) -> str | None:
    """Erster zitierter Pfad einer Zelle (Primär-Artefakt)."""
    paths = _backtick_paths(cell)
    return paths[0] if paths else None


def entry_evidence_ok(entry: RegisterEntry, root: Path = ROOT) -> bool:
    """Wahr, wenn der Eintrag mindestens einen auflösbaren Repo-Pfad zitiert
    (§G8 (copilot-instructions.md))."""
    return _has_resolvable_path(" ".join((entry.artifact, entry.evidence, entry.beleg)), root)


def domain_evidence_ok(entry: RegisterEntry, root: Path = ROOT) -> bool:
    """Wahr, wenn die Domänen-Behauptung konkret belegt ist (§III.13 Evidenzpflicht).

    Nur für Domänen mit Behauptungs-Charakter (``EVIDENCE_DOMAINS``);
    ``unbekannt``/``—`` sind immer zulässig, weil sie nichts behaupten.
    """
    if entry.domain not in EVIDENCE_DOMAINS:
        return True
    if GENERIC_EVIDENCE_MARKER in entry.evidence:
        return False
    return _has_resolvable_path(entry.evidence, root)


def _curated_domains() -> dict[str, str]:
    """Kuratierte Modell-Domänen aus dem Manifest (§G9 (copilot-instructions.md)).

    Ohne Import (fehlende Umgebung) entfällt der Abgleich mit Warnung —
    §V6 (copilot-instructions.md): kein stilles Degradieren.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from scripts.model_inventory import domain_report
    except Exception as exc:  # pragma: no cover - nur bei defekter Umgebung
        logger.warning(
            "model_inventory nicht importierbar (%s) — Domänen-Abgleich entfällt (§V6 (copilot-instructions.md))",
            exc,
        )
        return {}
    return {path: entry["domain"] for path, entry in domain_report()["models"].items()}


def validate(
    entries: list[RegisterEntry],
    declared_classes: list[str],
    root: Path = ROOT,
    curated_domains: dict[str, str] | None = None,
) -> list[str]:
    """Prüft Registereinträge; Rückgabe = Liste der Verstöße (leer = sauber)."""
    problems: list[str] = []
    if curated_domains is None:
        curated_domains = _curated_domains()

    declared = set(declared_classes)
    for klass in REQUIRED_CLASSES:
        if klass not in declared:
            problems.append(f"(c) Defizit-Klasse {klass} fehlt in der Klassentabelle")
        if not any(e.klass == klass for e in entries):
            problems.append(f"(c) Defizit-Klasse {klass} hat keinen Registereintrag")

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.id] = counts.get(entry.id, 0) + 1
    for eid, n in sorted(counts.items()):
        if n > 1:
            problems.append(f"(ID) doppelte Eintrags-ID {eid} ({n}×)")

    for entry in entries:
        where = f"{entry.id} (Zeile {entry.line_no})"
        id_match = ENTRY_ID_RE.match(entry.id)
        if id_match and f"K{id_match.group(1)}" != entry.klass:
            problems.append(f"(ID) {where}: ID-Klasse K{id_match.group(1)} ≠ Spalte Klasse {entry.klass}")
        if not all(
            (entry.artifact, entry.klass, entry.domain, entry.evidence, entry.status, entry.blocker, entry.beleg)
        ):
            problems.append(f"(Zelle) {where}: leere Pflichtzelle")
            continue
        if entry.status not in STATUS_ENUM:
            problems.append(f"(Status) {where}: unbekannter Status '{entry.status}'")
        if entry.domain not in DOMAIN_ENUM:
            problems.append(f"(Domäne) {where}: unbekannte Domäne '{entry.domain}'")

        if not entry_evidence_ok(entry, root):
            problems.append(f"(Evidenz) {where}: kein auflösbarer Repo-Pfad zitiert (§G8 (copilot-instructions.md))")
        if entry.domain in EVIDENCE_DOMAINS and not domain_evidence_ok(entry, root):
            hint = (
                "mit generischer Evidenz"
                if GENERIC_EVIDENCE_MARKER in entry.evidence
                else "ohne konkreten, existierenden Beleg-Pfad"
            )
            problems.append(
                f"(b) {where}: Domäne '{entry.domain}' {hint} — Evidenzpflicht §III.13 (copilot-instructions.md)"
            )
        if entry.status == "geschlossen" and not _has_resolvable_path(entry.beleg, root):
            problems.append(f"(a) {where}: Status 'geschlossen' ohne existierenden Beleg-Pfad")
        if entry.status == "bewusst-akzeptiert" and len(entry.blocker) < _MIN_ACCEPT_BEGRUENDUNG:
            problems.append(f"(Blocker) {where}: 'bewusst-akzeptiert' braucht eine Begründung im Blocker-Feld")

        primary = _first_path(entry.artifact)
        if primary and primary in curated_domains and curated_domains[primary] != entry.domain:
            problems.append(
                f"(§G9 (copilot-instructions.md)) {where}: Domäne '{entry.domain}' widerspricht dem kuratierten "
                f"Manifest ('{curated_domains[primary]}' für {primary})"
            )
    return problems


def audit(register_path: Path = REGISTER, root: Path = ROOT) -> dict:
    """Führt die komplette Register-Prüfung aus (maschinenlesbares Ergebnis)."""
    register_path = Path(register_path)
    if not register_path.is_file():
        return {
            "register": str(register_path),
            "entries": 0,
            "classes": [],
            "violations": [f"(Format) Register fehlt: {register_path}"],
        }
    text = register_path.read_text(encoding="utf-8")
    try:
        entries, declared = parse_register(text)
    except ValueError as exc:
        return {"register": str(register_path), "entries": 0, "classes": [], "violations": [f"(Format) {exc}"]}
    violations = validate(entries, declared, root=root)
    return {
        "register": str(register_path),
        "entries": len(entries),
        "classes": sorted(set(declared)),
        "violations": violations,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI-Einstieg; Exit 1 bei Verstößen (fail-closed)."""
    parser = argparse.ArgumentParser(description="SOTA-Defizit-Register-Gate (WP-0)")
    parser.add_argument("--register", default=str(REGISTER), help="Pfad zum Defizit-Register")
    parser.add_argument("--json", action="store_true", help="Ergebnis als JSON ausgeben")
    parser.add_argument("--quiet", action="store_true", help="Nur Exit-Code, keine Ausgabe")
    args = parser.parse_args(argv)

    result = audit(Path(args.register))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif not args.quiet:
        print(f"SOTA-Defizit-Register: {result['entries']} Einträge, Klassen {', '.join(result['classes']) or '—'}")
        if result["violations"]:
            for violation in result["violations"]:
                print(f"  ❌ {violation}")
        else:
            print("  ✅ keine Verstöße")
    return 1 if result["violations"] else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
