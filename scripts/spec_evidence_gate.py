"""Fail-closed validation for spec-change evidence reports and PR evidence blocks."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.compliance import wettbewerbs_paritaet_gate

_FIELDS: dict[str, tuple[str, ...]] = {
    "Spec-Datei": ("Spec-Datei",),
    "Abschnitt": ("Abschnitt",),
    "Aenderungstyp": ("Aenderungstyp", "Änderungstyp"),
    "Alte Regel": ("Alte Regel",),
    "Neue Regel": ("Neue Regel",),
    "Hypothese": ("Hypothese", "Fachliche Hypothese"),
    "Referenzen/Standards": ("Referenzen/Standards", "Referenzen", "Standards"),
    "Datensaetze/Szenarien": ("Datensätze/Szenarien", "Datensaetze/Szenarien"),
    "Umfang (n)": ("Umfang (n)", "n"),
    "Primarmetrik": ("Primarmetrik", "Primärmetrik"),
    "Effektstaerke": ("Effektstaerke", "Effektstärke"),
    "95 %-CI": ("95 %-CI",),
    "Signifikanztest + p-Wert": ("Signifikanztest + p-Wert", "Signifikanztest und p-Wert"),
    "Seed(s)": ("Seed(s)", "Seed"),
    "Commit": ("Commit",),
    "Skript/Befehl": ("Skript/Befehl", "Skript", "Befehl"),
    "Artefaktpfade": ("Artefaktpfade", "Artefaktpfad"),
    "Risiko fuer P1/P2": ("Risiko für P1/P2", "Risiko fuer P1/P2"),
    "Risiko fuer Artefakte": ("Risiko für Artefakte", "Risiko fuer Artefakte"),
    "Bekannte Unsicherheiten": ("Bekannte Unsicherheiten", "Unsicherheiten"),
    "Rollback-Kriterium": ("Rollback-Kriterium",),
    "Entscheidung": ("Entscheidung",),
    "Maintainer Sign-off": ("Maintainer Sign-off",),
    "Datum": ("Datum",),
}


def _label_key(label: str) -> str:
    normalized = unicodedata.normalize("NFKD", label).casefold().replace("ß", "ss")
    return "".join(char for char in normalized if char.isalnum())


_FIELD_KEYS = {name: {_label_key(alias) for alias in aliases} for name, aliases in _FIELDS.items()}
_EMPTY_VALUES = {"", "-", "—", "n/a", "na", "n.a.", "...", "todo", "tbd", "placeholder"}


def _extract_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        candidate = re.sub(r"^\s*[-*]\s*", "", line.strip())
        if ":" not in candidate:
            continue
        label, value = candidate.split(":", 1)
        key = _label_key(label.strip().strip("*` "))
        for field, aliases in _FIELD_KEYS.items():
            if key in aliases:
                fields[field] = value.strip().strip("`")
                break
    return fields


def _value_problem(field: str, value: str) -> str | None:
    normalized = value.strip().casefold()
    if normalized in _EMPTY_VALUES:
        return "leer oder nur ein Platzhalter"
    if re.match(r"^(?:n/?a|n\.a\.|nicht anwendbar|not applicable)\b", normalized):
        explanation = re.sub(
            r"^(?:n/?a|n\.a\.|nicht anwendbar|not applicable)\b",
            "",
            normalized,
        ).strip(" \t:;—–-")
        if len(explanation) < 12:
            return "N/A braucht eine konkrete Begründung"
    if field == "Maintainer Sign-off" and re.search(
        r"\b(?:pending|ausstehend|offen|n/?a|nicht erteilt|kein sign[- ]?off)\b",
        normalized,
    ):
        return "Sign-off ist nicht erteilt"
    if field == "Maintainer Sign-off" and len(re.findall(r"[^\W\d_]+", value, re.UNICODE)) < 2:
        return "Name der freigebenden Person fehlt"
    return None


def _validate_drt16_claim(text: str, root: Path) -> list[str]:
    if not re.search(r"\bDRT[- ]16\b", text, re.IGNORECASE) or not re.search(
        r"\b(?:PARITAET|PARITÄT)\b", text, re.IGNORECASE
    ):
        return []

    evidence_paths = sorted(set(re.findall(r"`?(docs/reports/current/[A-Za-z0-9._/-]+\.json)`?", text)))
    if not evidence_paths:
        return ["DRT-16-Paritätsbehauptung ohne kanonischen JSON-Beleg unter docs/reports/current/"]

    errors: list[str] = []
    for evidence_path in evidence_paths:
        relative = Path(evidence_path)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:3] != ("docs", "reports", "current"):
            errors.append(f"Unsicherer DRT-16-Belegpfad: {evidence_path}")
            continue
        path = root / relative
        try:
            path.resolve().relative_to(root.resolve())
        except (OSError, RuntimeError) as exc:
            errors.append(f"DRT-16-Belegpfad nicht sicher aufloesbar ({evidence_path}): {exc}")
            continue
        except ValueError:
            errors.append(f"DRT-16-Belegpfad verlässt das Repository: {evidence_path}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"DRT-16-Beleg nicht lesbar ({evidence_path}): {exc}")
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("discipline"), str):
            errors.append(f"DRT-16-Beleg ohne Disziplin-Objekt ({evidence_path})")
            continue
        parity_errors: list[str] = []
        valid, is_rx12, _ = wettbewerbs_paritaet_gate._pruefe_paritaets_beleg(
            evidence_path, payload["discipline"], parity_errors
        )
        if not valid or not is_rx12:
            errors.extend(parity_errors or [f"DRT-16-Beleg weist keine RX-12-Parität nach: {evidence_path}"])
    return errors


def _declared_specs(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z0-9])\.github/specs/[A-Za-z0-9._/-]+\.md", text))


def validate_evidence_text(
    text: str,
    *,
    root: Path | None = None,
    expected_specs: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not re.search(r"^## Evidenzblock\s*$", text, re.MULTILINE):
        errors.append("Pflichtabschnitt '## Evidenzblock' fehlt")

    fields = _extract_fields(text)
    for field in _FIELDS:
        value = fields.get(field)
        if value is None:
            errors.append(f"Pflichtfeld fehlt: {field}")
            continue
        problem = _value_problem(field, value)
        if problem:
            errors.append(f"{field}: {problem}")

    if expected_specs:
        missing_specs = sorted(expected_specs - _declared_specs(text))
        if missing_specs:
            errors.append(f"Geänderte Specs nicht im Evidenzblock zugeordnet: {', '.join(missing_specs)}")

    date = fields.get("Datum", "")
    if date and not re.search(r"\b20\d{2}-\d{2}-\d{2}\b", date):
        errors.append("Datum muss als ISO-Datum YYYY-MM-DD angegeben werden")

    errors.extend(_validate_drt16_claim(text, root or _ROOT))
    return errors


def _changed_paths(root: Path, *git_args: str) -> set[str]:
    result = subprocess.run(
        ["git", *git_args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return {path.strip() for path in result.stdout.splitlines() if path.strip()}


def _validate_changed_files(
    root: Path,
    changed_paths: set[str],
    *,
    pr_body: str | None = None,
) -> int:
    spec_paths = sorted(path for path in changed_paths if path.startswith(".github/specs/"))
    if not spec_paths:
        print("No spec changes detected; evidence gate passed.")
        return 0

    print("Spec files changed:")
    for path in spec_paths:
        print(f"  {path}")

    if pr_body is not None:
        errors = validate_evidence_text(pr_body, root=root, expected_specs=set(spec_paths))
        source = "PR body"
    else:
        reports = sorted(
            path for path in changed_paths if path.startswith("docs/reports/spec_evidence/") and path.endswith(".md")
        )
        if not reports:
            print("Spec changes require a changed report under docs/reports/spec_evidence/.")
            return 1
        errors = []
        declared_specs: set[str] = set()
        for report in reports:
            report_path = Path(report)
            if report_path.is_absolute() or ".." in report_path.parts:
                errors.append(f"Unsicherer Evidenzbericht-Pfad: {report}")
                continue
            try:
                content = (root / report_path).read_text(encoding="utf-8")
            except OSError as exc:
                errors.append(f"Evidenzbericht nicht lesbar ({report}): {exc}")
                continue
            declared_specs.update(_declared_specs(content))
            report_errors = validate_evidence_text(content, root=root)
            errors.extend(f"{report}: {error}" for error in report_errors)
        missing_specs = sorted(set(spec_paths) - declared_specs)
        if missing_specs:
            errors.append(f"Geänderte Specs nicht in den Evidenzberichten zugeordnet: {', '.join(missing_specs)}")
        source = ", ".join(reports)

    if errors:
        print(f"Evidence validation failed for {source}:")
        for error in errors:
            print(f"  ERROR: {error}")
        return 1
    print(f"Evidence validation passed for {source}.")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since-hours", type=float, help="Validate changes within this git-log window.")
    parser.add_argument("--base", help="Base revision for a git diff range.")
    parser.add_argument("--head", help="Head revision for a git diff range.")
    parser.add_argument("--pr-body-env", help="Environment variable containing the PR body.")
    args = parser.parse_args()
    has_since = args.since_hours is not None
    has_range = args.base is not None or args.head is not None
    if has_since == has_range:
        parser.error("provide exactly one of --since-hours or both --base and --head")
    if has_range and (not args.base or not args.head):
        parser.error("--base and --head must be provided together")
    if args.pr_body_env and not has_range:
        parser.error("--pr-body-env requires --base and --head")
    if has_since and args.since_hours <= 0:
        parser.error("--since-hours must be positive")
    return args


def main() -> int:
    args = _parse_args()
    if args.since_hours is not None:
        changed = _changed_paths(
            _ROOT,
            "log",
            f"--since={args.since_hours:g} hours ago",
            "--name-only",
            "--pretty=format:",
        )
    else:
        changed = _changed_paths(_ROOT, "diff", "--name-only", args.base, args.head)

    pr_body = os.environ.get(args.pr_body_env, "") if args.pr_body_env else None
    return _validate_changed_files(_ROOT, changed, pr_body=pr_body)


if __name__ == "__main__":
    raise SystemExit(main())
