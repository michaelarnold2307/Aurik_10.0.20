"""Pfad-E2E-Register-Gate (fail-closed) — Maßnahme 1 gegen Phantom-Pfade.

Befund 2026-10-07 (K2-Inventur): Mehrere Signalpfade existierten im Code,
ohne dass irgendetwas sie erreichte (vocoder_chain.py: null Aufrufer;
sota_speech_superres.py: null Aufrufer) — oder sie wirkten real unter falschem
Etikett (Routing-Etikett „NVSR" = SBR-Kern; echtes nvsr.ts verwaist). Dieser
Zustand — „existiert, scheint verdrahtet, wirkt nie oder falsch" — ist der
verbotene vierte Zustand. Das Register macht fuer JEDEN auditierten Pfad den
Status explizit:

  productive         — E2E im Produktions-/Studio-Pfad; E2E-Test ODER
                       dokumentierte Luecke (e2e_gap + gap_reason) PFLICHT
  challenger         — eigener E2E-Weg (Benchmark/Runde) muss laufen; E2E-Test
                       ODER dokumentierte Luecke PFLICHT
  k0                 — stilles Kapital; Beleg PFLICHT
  archived           — Fingerprint dokumentiert
  phantom_transition — NUR mit removal_plan (Uebergang, kein Endzustand)

Pruefungen (fail-closed, Exit 1 bei Verstoss):
  (a) Schema: Pflichtfelder, Status im Enum.
  (b) Artefakte: alle `artifacts`-Pfade existieren im Repo.
  (c) E2E-Pflicht: productive/challenger brauchen `e2e_test` (Datei existiert)
      oder `e2e_gap=true` mit nicht-leerem `gap_reason`.
  (d) Phantom-Regel: `phantom_transition` braucht `removal_plan`.
  (e) Beleg-Pflicht: `evidence` nicht leer, Beleg-Dateien existieren.
  (f) Duplikat-Ids und unbekannte Schluessel werden gemeldet.

Nutzung:
  python scripts/path_e2e_check.py            # fail-closed
  python scripts/path_e2e_check.py --json     # Maschinen-Ausgabe
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY = _ROOT / "config" / "path_e2e_registry.json"

_STATUS_ENUM = frozenset({"productive", "challenger", "k0", "archived", "phantom_transition"})
_REQUIRED = ("id", "layer", "status", "role", "domain", "artifacts", "call_path", "evidence")
_ALLOWED = frozenset(_REQUIRED) | {"e2e_test", "e2e_gap", "gap_reason", "removal_plan"}


def _check_path_exists(rel: str) -> bool:
    return (_ROOT / rel).exists()


def run_check(registry_path: Path | None = None) -> tuple[list[str], dict[str, int]]:
    """Prueft das Register. Returns (Fehlerliste, Status-Zaehlung)."""
    path = registry_path or _REGISTRY
    errors: list[str] = []
    counts: dict[str, int] = {}
    if not path.exists():
        return [f"Register fehlt: {path}"], counts

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Register ist kein gueltiges JSON: {exc}"], counts

    entries = data.get("paths")
    if not isinstance(entries, list) or not entries:
        return ["Register ohne 'paths'-Liste."], counts

    seen_ids: set[str] = set()
    for idx, entry in enumerate(entries):
        eid = entry.get("id", f"<index {idx}>")
        if eid in seen_ids:
            errors.append(f"[{eid}] doppelte id")
        seen_ids.add(eid)

        unknown = set(entry) - _ALLOWED
        if unknown:
            errors.append(f"[{eid}] unbekannte Schluessel: {sorted(unknown)}")

        for key in _REQUIRED:
            if key not in entry:
                errors.append(f"[{eid}] Pflichtfeld fehlt: {key}")
        if errors and any(e.startswith(f"[{eid}] Pflichtfeld") for e in errors):
            continue

        status = entry["status"]
        counts[status] = counts.get(status, 0) + 1
        if status not in _STATUS_ENUM:
            errors.append(f"[{eid}] ungueltiger Status: {status!r}")

        for rel in entry.get("artifacts", []):
            if not _check_path_exists(rel):
                errors.append(f"[{eid}] Artefakt fehlt: {rel}")

        evidence = entry.get("evidence") or []
        if not evidence:
            errors.append(f"[{eid}] evidence ist leer (Belegpflicht)")
        for rel in evidence:
            rel_clean = str(rel).split(" (")[0].strip("`")
            if "/" in rel_clean and not _check_path_exists(rel_clean):
                errors.append(f"[{eid}] Belegdatei fehlt: {rel_clean}")

        if status in ("productive", "challenger"):
            e2e = entry.get("e2e_test")
            gap = bool(entry.get("e2e_gap", False))
            if e2e:
                if not _check_path_exists(str(e2e)):
                    errors.append(f"[{eid}] e2e_test nicht gefunden: {e2e}")
            elif not gap or not str(entry.get("gap_reason") or "").strip():
                errors.append(f"[{eid}] {status} ohne e2e_test braucht e2e_gap=true + gap_reason")

        if status == "phantom_transition":
            if not str(entry.get("removal_plan") or "").strip():
                errors.append(f"[{eid}] phantom_transition braucht removal_plan")
        if status == "k0":
            if not evidence:
                errors.append(f"[{eid}] k0 braucht Beleg (evidence)")

    return errors, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pfad-E2E-Register-Gate (fail-closed)")
    parser.add_argument("--json", action="store_true", help="Maschinen-Ausgabe")
    parser.add_argument("--registry", type=Path, default=None, help="alternativer Registerpfad")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    errors, counts = run_check(args.registry)

    gaps = 0
    phantoms = 0
    if _REGISTRY.exists():
        try:
            data = json.loads(_REGISTRY.read_text(encoding="utf-8"))
            gaps = sum(1 for e in data.get("paths", []) if e.get("e2e_gap"))
            phantoms = sum(1 for e in data.get("paths", []) if e.get("status") == "phantom_transition")
        except json.JSONDecodeError as exc:
            logger.debug("Pfad-Registry nicht lesbar — Zählung ohne Registry-Werte: %s", exc)

    summary = {
        "errors": errors,
        "counts": counts,
        "total": sum(counts.values()),
        "gaps": gaps,
        "phantoms": phantoms,
    }
    if args.json:
        print(json.dumps(summary, indent=1, ensure_ascii=False))
    else:
        print(f"Pfad-E2E-Register: {summary['total']} Pfade — {dict(sorted(counts.items()))}")
        print(f"  dokumentierte E2E-Luecken: {gaps} | Phantome (Uebergang): {phantoms}")
        for err in errors:
            logger.error("VERSTOSS: %s", err)
        if not errors:
            print("  ✅ keine Verstoesse")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
