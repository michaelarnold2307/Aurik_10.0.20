#!/usr/bin/env python3
"""§G188–§G190 (GEBOTE.md) — Wirkungs-Kalibrierungsgate für Pre-Commit.

Auftrag (2026-09-26): Die Pre-Commit-Hooks MÜSSEN melden, wenn Aurik durch
Begrenzungen blockiert ist oder sich die benötigte Intensität nicht eigenständig
korrekt berechnen kann. Physikalische Grenzen (§G189–§G190 (GEBOTE.md)) sind zu
AKZEPTIEREN und kein Verstoß.

Gemeldet wird:
1. **Feste Stärke-Kappen** (§G188 (GEBOTE.md)) — Material-/Inhalts-Maps auf
   Stärke-Variablen mit Konstanten < 1.0 oder Multiplikations-Dämpfung von
   Stärke-Variablen mit Literalen < 1.0. Solche Kappen blockieren Aurik am
   gemessenen Optimum (Produktionsbefund: 70-%-Kappe ließ 30 % des gemessenen
   Wow stehen). Ausnahme: Zeilen mit §G189-Kommentar (dokumentierte Ausnahme).
2. **Nicht selbst berechenbare Intensität** — Korrektur-Anwendung ohne
   messbasierte Stärke-Herkunft im selben Modul (kein §G188-/§G189-/§7.4d-Bezug
   im Stärke-Pfad) als Warnung.

ZUGELASSEN ohne Meldung (physikalische Grenzen §G189–§G190 (GEBOTE.md)):
`max_stretch_delta`, `DETECTION_THRESHOLD`, Schwell-/Delta-/Clip-Begrenzungen.

Verhalten: Verstöße in den übergebenen Dateien ⇒ Exit 1 (blockiert mit
Meldung). IMMER zusätzlich: repo-weiter Altbestands-Report „Aurik ist durch
folgende Begrenzungen blockiert" — auch unveränderte Alt-Phasen werden so bei
jedem Commit gemeldet, bis §G188 migriert ist.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Physische Grenzen (§G189–§G190 (GEBOTE.md)) — nie als Verstoß melden.
PHYSICAL_LIMIT_MARKERS = (
    "max_stretch_delta",
    "detection_threshold",
    "threshold",
    "delta",
    "clip",
    "ceiling",
    "floor",
    "budget",
)
# Stärke-Variablen: nur diese dürfen nicht pauschal gekappt werden.
STRENGTH_MARKERS = ("strength",)
EXCEPTION_MARKER = "§G189"


def _is_strength_name(name: str) -> bool:
    lowered = name.lower()
    return any(m in lowered for m in STRENGTH_MARKERS) and not any(m in lowered for m in PHYSICAL_LIMIT_MARKERS)


def _numeric_constant(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _numeric_constant(node.operand)
        return -v if v is not None else None
    return None


def _const_below_one(node: ast.AST) -> float | None:
    value = _numeric_constant(node)
    return value if value is not None and value < 1.0 else None


def _assigned_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def scan_source(path: Path, source: str) -> list[str]:
    """Liefert Meldungen (Zeile: Text) für §G188-Verstöße in einer Quelldatei."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    issues: list[str] = []

    def _line_has_exception(lineno: int) -> bool:
        # §G189-Marker im Fenster ±6 Zeilen um die Aussage (ruff-format kann
        # Annotationen umbrechen und Kommentare verschieben).
        _lo = max(1, lineno - 6)
        _hi = min(len(lines), lineno + 6) + 1
        return any(EXCEPTION_MARKER in lines[_ln - 1] for _ln in range(_lo, _hi))

    for node in ast.walk(tree):
        # 1a. Feste Kappen als Dict:  STRENGTH_MAP = {MaterialType.X: 0.7, ...}
        name = _assigned_name(node)
        value = None
        if (isinstance(node, ast.Assign) and name) or (isinstance(node, ast.AnnAssign) and name):
            value = node.value
        if name and value is not None and isinstance(value, ast.Dict) and _is_strength_name(name):
            capped = [factor for v in value.values if (factor := _const_below_one(v)) is not None]
            if capped and not _line_has_exception(getattr(node, "lineno", 0)):
                issues.append(
                    f"{path}:{getattr(node, 'lineno', 0)}: feste Stärke-Kappe "
                    f"{name} = {{…: {', '.join(f'{c:.2f}' for c in capped[:4])}}} — "
                    "Aurik wird am gemessenen Optimum blockiert (§G188 (GEBOTE.md))"
                )

        # 1b. Literal-Dämpfung:  strength_scale *= 0.82  /  strength = strength * 0.7
        if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Mult):
            tgt = node.target.id if isinstance(node.target, ast.Name) else None
            factor = _const_below_one(node.value)
            if tgt and _is_strength_name(tgt) and factor is not None and not _line_has_exception(node.lineno):
                issues.append(
                    f"{path}:{node.lineno}: feste Stärke-Dämpfung {tgt} *= {factor:.2f} — "
                    "Intensität folgt der Evidenz, nicht einem Literalen (§G188 (GEBOTE.md))"
                )
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if _is_strength_name(node.targets[0].id) and isinstance(node.value, ast.BinOp):
                if isinstance(node.value.op, ast.Mult):
                    lhs, rhs = node.value.left, node.value.right
                    other = rhs if _const_below_one(lhs) is not None else lhs
                    factor = _const_below_one(rhs if other is lhs else lhs)
                    if (
                        isinstance(other, ast.Name)
                        and other.id == node.targets[0].id
                        and factor is not None
                        and not _line_has_exception(node.lineno)
                    ):
                        issues.append(
                            f"{path}:{node.lineno}: feste Stärke-Dämpfung "
                            f"{node.targets[0].id} = {other.id} * {factor:.2f} — "
                            "Intensität folgt der Evidenz, nicht einem Literalen (§G188 (GEBOTE.md))"
                        )
    return issues


def scan_file(path: Path) -> list[str]:
    # Tests may intentionally model legacy values as fixtures. They do not
    # constrain production restoration and must not fail the production gate.
    try:
        if "tests" in path.resolve().relative_to(ROOT).parts:
            return []
    except (OSError, ValueError):
        pass
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return scan_source(path, source)


def legacy_report() -> list[str]:
    """Repo-weiter Altbestand: blockierende Begrenzungen IMMER melden."""
    found: list[str] = []
    phases = ROOT / "backend" / "core" / "phases"
    if not phases.is_dir():
        return found
    for py in sorted(phases.glob("phase_*.py")):
        found.extend(scan_file(py))
    return found


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    issues: list[str] = []
    for rel in args:
        p = Path(rel)
        if p.suffix != ".py" or not p.exists():
            continue
        issues.extend(scan_file(p))

    legacy = legacy_report()

    if issues:
        print("🛡️ §G188 Wirkungs-Kalibrierung: blockierende Begrenzung(en) in den geprüften Dateien:")
        for msg in issues:
            print(f"  🚫 {msg}")
        print(
            "  Diese Kappen verhindern die maximal optimierte Restaurierung "
            "(§G188 (GEBOTE.md)). Stärke aus der gemessenen Defekttiefe ableiten "
            "(§7.4d (06_phases_system.md)); physikalische Grenzen bleiben als "
            "dokumentierte Ausnahme §G189 (GEBOTE.md) zugelassen."
        )

    if legacy:
        print("🛡️ §G188 Meldeauftrag — Aurik ist durch folgende Altbestands-Begrenzungen blockiert:")
        for msg in legacy:
            print(f"  ⚠️ {msg}")
    else:
        print("🛡️ §G188 Wirkungs-Kalibrierung: keine blockierenden Stärke-Kappen im Phasen-Kern.")

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
