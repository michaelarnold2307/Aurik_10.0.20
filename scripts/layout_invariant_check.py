#!/usr/bin/env python3
"""scripts/layout_invariant_check.py — §G9 (copilot-instructions.md): Layout-Entscheidung hat EINE Quelle.

**Warum (Befund-Klasse D-K3-6, 2026-10-07).** Der Wohlklang-Schutz HR-V1 war auf
Stereo **wirkungslos**: eine modul-lokale Layout-Heuristik (``x.shape[0] <= 2``)
wich von der kanonischen Regel ab und transponierte Mehrachser-Signale
(``3…8`` Kanäle) falsch. Die Messung am 2026-10-07 fand **vier** solche Kopien
gegen die kanonische Quelle ``backend/core/audio_layout.py``
(``is_channels_first``: ``shape[0] <= MAX_CHANNELS``):

* ``backend/core/dsp/additive_synthesis_gate.py`` (der D-K3-6-Fundort),
* ``backend/core/dsp/hybrid_denoise_fusion.py``,
* ``backend/core/dsp/scrape_flutter_rest.py``,
* ``backend/core/dsp/silence_mask.py`` (verschachtelt),

dazu ``backend/core/listening_witness.py`` (ZEUGE — bewusst strenger, s. u.).
Alle vier sind am 2026-10-07 an die kanonische Entscheidung gebunden; dieses Gate
verhindert die Rückkehr (die Konsolidierung war nötig, weil die Divergenz im
Stillen wirkte: HR-V1 fiel in einen §V6-Ersatzpfad und niemand sah es).

**Regeln (ERROR ⇒ Exit 1):**

* **L1** Eine Funktion, deren *Name* sie als Layout-Normalisierer ausweist
  (``to_channels*``, ``channels_first*``, ``normalize_channels_first`` …),
  außerhalb der kanonischen Quelle, die ``audio_layout`` **nicht** benutzt.
  Wird darin zusätzlich die Kanalachse per Zahlenvergleich geraten
  (``shape[0] <=/==/< N``), meldet das Gate das ausdrücklich als
  „Muster D-K3-6“.
* **L2** (Bericht, **kein** Fail): Zahl layout-empfindlicher Reduktionen
  (``mean(axis=0)``, ``[:, 0]``, ``shape[0]``-Schleifen) in Modulen **ohne jede**
  Layout-Entscheidung — Triage-Backlog. Eine statische Regel kann hier nicht
  entscheiden: ``mean(axis=0)`` über einen Batch-Stack (``B``) ist korrekt,
  über ein ``(C, N)``-Audiosignal ist es der Kollaps. Deshalb Bericht + Register,
  nicht Fail (Lehre aus D-K3-5: ein Bericht ohne Entscheidbarkeit ist nur Lärm).

**Nicht im Regel-Scope (andere Rolle, bewusst):** ``_restore_layout``
(Rück-Transpose in die *Eingabe*-Orientierung) und ``to_stereo``
(Kanal-*Aufbau*, kein Layout-Normalisieren). Sie entscheiden das Layout nicht —
sie stellen es wieder her.

Aufruf::

    python scripts/layout_invariant_check.py            # Gate (Exit 0/1)
    python scripts/layout_invariant_check.py --report   # L2-Liste ausgeben
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# §G9 (copilot-instructions.md): Produktionswurzeln wie im Wohlklang-Gate (eine
# Definition der „Produktionscode"-Menge, keine zweite Liste) — ebenso die
# Befund-Klasse `Finding`: nur EINE Implementierung im Tooling
# (`aurik-symbol-duplicates` ist fail-closed auf Klassen-Duplikate).
from scripts.wohlklang_gate import PRODUCTION_ROOTS, Finding, _blank_prose

CANONICAL_MODULE = "backend/core/audio_layout.py"

# Namen, die eine Layout-Normalisierung behaupten.
NORMALIZER_RE = re.compile(
    r"^_?(?:to_channels(?:_first|_last)?|channels_first(?:_or_none)?|normalize_channels_first|ensure_channels_first)$"
)

# Zahlenvergleich auf der ersten Achse = geratene Kanalachse (Muster D-K3-6).
SHAPE_GUESS_RE = re.compile(r"\.shape\s*\[\s*(-?1|0)\s*\]\s*(?:<=?|==|>=?)\s*\d+")

# Layout-empfindliche Reduktionen (nur für den L2-Bericht).
# Deckt BEIDE Schreibweisen: `x.mean(axis=0)` UND `np.mean(x, axis=0)` — die
# erste Fassung dieses Musters fand nur die argumentlose Form und untertrieb den
# Backlog dadurch (2026-10-07 gemessen: 920 → 1269).
RISKY_RE = re.compile(
    r"\.mean\(\s*(?:[^()]*,\s*)?axis\s*=\s*0\b"
    r"|\[\s*:\s*,\s*0\s*\]"
    r"|range\(\s*\w+\.shape\s*\[\s*0\s*\]\s*\)"
)

CANONICAL_REF_RE = re.compile(r"audio_layout")

# Ausnahmen MIT Begründung (§G8 copilot-instructions.md — Transparenz).
# Schlüssel: "<pfad>::<funktion>". Leer = kein Ausnahmefall (Sollzustand).
ALLOWLIST: dict[str, str] = {}


def _function_source(path: Path, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Quelltext einer Funktion — **prosa-bereinigt** (Kommentare/Docstrings weg).

    §G9 (copilot-instructions.md): dieselbe Technik wie im Wohlklang-Gate
    (``_blank_prose``), keine zweite Umsetzung; ein ``shape[0]`` in einer
    Beschreibung ist keine Layout-Entscheidung.
    """
    lines = _blank_prose(path.read_text(encoding="utf-8", errors="ignore")).splitlines()
    start = node.lineno - 1
    end = node.end_lineno or start + 1
    return "\n".join(lines[start:end])


def scan(roots: tuple[str, ...] = PRODUCTION_ROOTS, base: Path = ROOT) -> tuple[list[Finding], int]:
    """L1 (fail-closed) und L2 (Bericht) über die Produktionswurzeln."""
    findings: list[Finding] = []
    risky = 0
    for root_name in roots:
        root = base / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            rel = str(path.relative_to(base))
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError as exc:  # fail-closed: nicht prüfbar = Befund
                findings.append(Finding("ERROR", "L1", f"{rel}: nicht parsebar ({exc.__class__.__name__})"))
                continue
            module_code = _blank_prose(path.read_text(encoding="utf-8", errors="ignore"))
            if rel != CANONICAL_MODULE and not CANONICAL_REF_RE.search(module_code):
                count = len(RISKY_RE.findall(module_code))
                if count:
                    risky += count

            if rel == CANONICAL_MODULE:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not NORMALIZER_RE.match(node.name):
                    continue
                key = f"{rel}::{node.name}"
                if key in ALLOWLIST:
                    continue
                body = _function_source(path, node)
                if CANONICAL_REF_RE.search(body):
                    continue
                guess = SHAPE_GUESS_RE.search(body)
                detail = (
                    " — enthält eine geratene Kanalachse (Muster D-K3-6: `shape[...] <=/==/* < N`)" if guess else ""
                )
                findings.append(
                    Finding(
                        "ERROR",
                        "L1",
                        f"{rel}:{node.lineno} {node.name}() entscheidet das Layout ohne `{CANONICAL_MODULE}`{detail}",
                    )
                )

    findings.append(
        Finding(
            "INFO",
            "L2",
            f"{risky} layout-empfindliche Reduktion(en) in Modulen ohne Layout-Entscheidung "
            f"(Bericht/Triage-Backlog, kein Fail — Batch-Achse vs. Kanal-Achse ist statisch nicht entscheidbar)",
        )
    )
    return findings, risky


def main() -> int:
    """CLI-Einstieg: Exit 0 = Regel erfüllt, 1 = Verstoß."""
    parser = argparse.ArgumentParser(description="§G9 (copilot-instructions.md) Layout-EINE-Quelle-Gate")
    parser.add_argument("--report", action="store_true", help="L2-Bericht mitausgeben")
    args = parser.parse_args()

    findings, risky = scan()
    errors = [f for f in findings if f.severity == "ERROR"]
    for finding in findings:
        if finding.severity == "ERROR" or args.report:
            print(f"{finding.severity} [{finding.rule}] {finding.message}")
    if not errors and not args.report:
        print(f"Layout-Invariante erfüllt: 0 lokale Layout-Entscheidung; L2-Backlog={risky}.")
    if errors:
        print(
            f"\n{len(errors)} Verstoß/Verstöße: Layout-Normalisierung MUSS über "
            f"`{CANONICAL_MODULE}` laufen (§G9 copilot-instructions.md)."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
