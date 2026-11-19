#!/usr/bin/env python3
"""Frontend-Prosa-Scanner (D-K3-107): indirekte Nutzertext-Kandidaten ohne ``t()``.

Hintergrund (``.github/SOTA_DEFICIT_REGISTER.md`` D-K3-10/107): Die
``hardcoded_texts``-Prüfung des Frontend-Linters sieht nur **direkte**
Setter-Literale (``setText(\"…\")``, ``setToolTip(\"…\")``, ``_show_toast``,
``set_status``). Die Folgewelle sind **indirekte** Kandidaten: Literale, die
erst über Hilfsvariablen/Helfer (``_warn``, ``_badge``, Painter-``drawText``,
Label-Verkettungen) den Nutzer erreichen.

Dieses Skript ist der reproduzierbare Scanner dafür: AST-basiert, positions-
getreu mit der kanonischen Prosa-Blankung (``scripts.wohlklang_gate._blank_prose``,
D-K3-5-validiert), mit Filtern gegen Keys/Pfade/CSS/Formate/Logger.
Ausgabe: JSON-Inventar nach ``$FRONTEND_PROSA_OUT``
(Default ``reports/frontend_prosa_kandidaten.json``) + Konsolenliste.
Exit 0 = Scan erfolgreich (die ZAHL ist ein Monitoring-Wert, kein Gate).

Verwendung:
    python scripts/frontend_prosa_scanner.py [--ui-dir Aurik10/ui] [--limit N]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Filter ────────────────────────────────────────────────────────────────────
_EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u2705\u274C\u26A0\u2699\u2B50]")
# Reine Format-Templates ("{:.2f}", "  - {n} Stück" ist Prosa, aber "{}" nicht)
_NUR_FORMAT_RE = re.compile(r"^[\s{}:,.0-9fds%+\-<>()\[\]]*$")
# Dotted-Keys ("a.b.c", "ui.slide_panel_open") — keine Prosa
_KEY_RE = re.compile(r"^[a-z_][a-z0-9_]*(?:\.[a-z0-9_]+)+$")
# CSS-/Stylesheet-Fragmente
_CSS_RE = re.compile(r"^(?:[^;]*;)+[^;]*$")  # mind. ein ';'
_CSS_WORTE = (
    "px",
    "rgb",
    "background",
    "font-",
    "padding",
    "margin",
    "border",
    "qlineargradient",
    "color:",
    "transparent",
)
# Pfade/URLs/Datei-Endungen
_PFAD_RE = re.compile(r"\.(?:wav|flac|mp3|onnx|pt|json|yaml|yml|png|svg|ckpt|exe|log|txt)$|://|^/|^~|[/\\][a-z_]+[/\\]")
# Reine Zähler-/Code-Fragmente
_NUR_SYMBOL_RE = re.compile(r"^[\W_0-9]*$")
# Einzelnes großgeschriebenes deutsches Wort (z. B. „Knapp", „Zielwert", „Tonart") —
# in UI-Dateien praktisch immer ein sichtbares Label.
_EINZELWORT_LABEL_RE = re.compile(r"^[A-ZÄÖÜ][a-zäöüß]{3,}$")
# Reine CSS-Eigenschafts-Präfixe ("color: ", "background: ") — kein sichtbarer Text.
_CSS_EIGENSCHAFT_RE = re.compile(r"^[a-z-]+:\s*$")


def _blank_prose(text: str) -> str:
    """Kanonische Prosa-Blankung (D-K3-5-validiert) — Kommentare/Statement-Strings blanken."""
    try:
        from scripts.wohlklang_gate import _blank_prose as _bp
    except Exception:
        return text
    return _bp(text)


def _prosa_wertig(literal: str) -> bool:
    """Heuristik: ist das Literal nutzersichtbare Prosa (kein Key/Pfad/CSS/Format)?"""
    s = literal.strip()
    if not s or len(s) < 3:
        return False
    entkernt = _EMOJI_RE.sub("", s).strip()
    if _NUR_SYMBOL_RE.match(entkernt):
        return False
    if _CSS_EIGENSCHAFT_RE.match(entkernt):
        return False
    if not re.search(r"[A-Za-zÄÖÜäöüß]", s):
        return False
    if _KEY_RE.match(entkernt):
        return False
    if _NUR_FORMAT_RE.match(entkernt):
        return False
    if _PFAD_RE.search(entkernt.lower()):
        return False
    # CSS: Semikolon UND ein CSS-Wort
    if ";" in s and any(w in s.lower() for w in _CSS_WORTE):
        return False
    # Muss Wörter enthalten (mind. ein Leerzeichen ODER Umlaut ODER Emoji ODER Satzzeichen)
    if " " in s or _EMOJI_RE.search(s) or any(c in s for c in "äöüÄÖÜß"):
        return True
    if re.search(r"[.,!?…:]", s) and len(entkernt) >= 4:
        return True
    if _EINZELWORT_LABEL_RE.match(entkernt):
        return True
    return False


def _im_t_call(node: ast.AST) -> bool:
    """True, wenn der Knoten direktes Argument von t(...) ist."""
    if not isinstance(node, ast.Constant):
        return False
    parent = getattr(node, "_eltern", None)
    if not isinstance(parent, ast.Call):
        return False
    func = parent.func
    if isinstance(func, ast.Name):
        return func.id == "t"
    if isinstance(func, ast.Attribute):
        return func.attr == "t"
    return False


def _in_logger_call(node: ast.AST) -> bool:
    """True, wenn der Knoten Argument eines logger./log.-Aufrufs ist (nicht nutzersichtbar)."""
    parent = getattr(node, "_eltern", None)
    if not isinstance(parent, ast.Call):
        return False
    func = parent.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id in {"logger", "log", "_logger", "logging"}
    return False


def _ist_docstring(node: ast.AST) -> bool:
    """True, wenn der Knoten ein Modul-/Klassen-/Funktions-Docstring ist."""
    parent = getattr(node, "_eltern", None)
    if isinstance(parent, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        body = getattr(parent, "body", [])
        return bool(body) and body[0] is node
    return False


def _sink_hinweis(node: ast.Constant) -> str:
    """Kontext für den Fix: wohin fließt das Literal?"""
    parent = getattr(node, "_eltern", None)
    if isinstance(parent, ast.Assign) and len(parent.targets) == 1:
        ziel = parent.targets[0]
        if isinstance(ziel, ast.Name):
            return f"zuweisung:{ziel.id}"
    if isinstance(parent, ast.Call):
        func = parent.func
        name = ""
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name:
            return f"argument:{name}"
        return "argument:?"
    if isinstance(parent, (ast.List, ast.Tuple)):
        return "liste"
    if isinstance(parent, ast.JoinedStr):
        return "f-string-teil"
    return "?"


# ── Dokumentierte Ausnahmen (kein Nutzertext) ────────────────────────────
# Nach der D-K3-107-Welle verbleiben ausschließlich technische Literale.
# Ein NEUER Prosa-Kandidat außerhalb dieser Tabelle lässt
# tests/unit/test_frontend_prosa_folgewelle.py fail-closed fallen.
EXEMPT: list[dict] = [
    {"datei": "help_system.py", "zeilen": (60, 95), "grund": "Fehler-Match-Muster (kein Nutzertext)"},
    {"datei": "help_system.py", "text_gleich": "Error", "grund": "Exception-Namens-Matching (kein Nutzertext)"},
    {"datei": "help_system.py", "text_gleich": "Exception", "grund": "Exception-Namens-Matching (kein Nutzertext)"},
    {"datei": "onboarding.py", "text_gleich": "AURIK Professional", "grund": "QSettings-Org-Name (kein Nutzertext)"},
    {
        "datei": "restoration_status_panel.py",
        "text_prefix": "status_panel.dimension.",
        "grund": "i18n-Key-Präfix-Builder (kein Nutzertext)",
    },
    {"datei": "song_prognose_widget.py", "text_prefix": "rgba(", "grund": "Farbkonstante (kein Nutzertext)"},
    {
        "datei": "song_prognose_widget.py",
        "text_prefix": "QScrollArea { background:",
        "grund": "Stylesheet-Fragment (kein sichtbarer Text)",
    },
    {
        "datei": "song_prognose_widget.py",
        "zeilen": (1030, 1085),
        "grund": "Terminal-/Logger-Diagnosebericht (kein GUI-Text)",
    },
    {"datei": "results_summary.py", "text_prefix": "rgba(", "grund": "Farbkonstante (kein Nutzertext)"},
    {
        "datei": "results_summary.py",
        "text_gleich": "material.",
        "grund": "Dynamischer i18n-Key-Präfix-Builder (kein Nutzertext)",
    },
    {"datei": "splash_screen.py", "text_gleich": "Noto Sans", "grund": "Font-Name (technische Font-Auswahl)"},
    {"datei": "splash_screen.py", "text_gleich": "Segoe UI", "grund": "Font-Name (technische Font-Auswahl)"},
    {"datei": "splash_screen.py", "text_gleich": "Helvetica Neue", "grund": "Font-Name (technische Font-Auswahl)"},
    {"datei": "splash_screen.py", "text_gleich": "Arial", "grund": "Font-Name (technische Font-Auswahl)"},
    {"datei": "splash_screen.py", "text_gleich": "Inter", "grund": "Font-Name (technische Font-Auswahl)"},
    {"datei": "splash_screen.py", "text_gleich": "Ubuntu", "grund": "Font-Name (technische Font-Auswahl)"},
    {
        "datei": "musical_goals_radar.py",
        "text_prefix": "<br><br><span style=",
        "grund": "Reines HTML-Attribut-Fragment (sichtbarer Text läuft über t())",
    },
]


def _exempt_match(datei: str, zeile: int, text: str) -> str | None:
    """Gibt den Grund zurück, wenn der Kandidat dokumentiert ausgenommen ist."""
    for e in EXEMPT:
        if e["datei"] not in datei:
            continue
        if "zeilen" in e:
            lo, hi = e["zeilen"]
            if lo <= zeile <= hi:
                return str(e["grund"])
        if "text_gleich" in e and text == e["text_gleich"]:
            return str(e["grund"])
        if "text_prefix" in e and text.startswith(e["text_prefix"]):
            return str(e["grund"])
    return None


def scan(ui_dir: Path) -> tuple[list[dict], list[dict]]:
    """Sammelt Prosa-Kandidaten aus allen .py-Dateien unter ui_dir.

    Rückgabe: (kandidaten, exempt) — exempt = dokumentierte technische
    Nicht-Texte (EXEMPT-Tabelle). ``kandidaten`` MUSS leer sein
    (fail-closed über tests/unit/test_frontend_prosa_folgewelle.py).
    """
    kandidaten: list[dict] = []
    exempt: list[dict] = []
    for py in sorted(ui_dir.rglob("*.py")):
        try:
            quelle = py.read_text(encoding="utf-8")
        except OSError:
            continue
        geblankt = _blank_prose(quelle)
        try:
            baum = ast.parse(geblankt)
        except SyntaxError:
            continue
        # Eltern-Verkettung für Kontext-Filter
        for knoten in ast.walk(baum):
            for kind in ast.iter_child_nodes(knoten):
                kind._eltern = knoten  # type: ignore[attr-defined]
        zeilenzahl = quelle.count("\n") + 1
        for knoten in ast.walk(baum):
            texte: list[str] = []
            if isinstance(knoten, ast.Constant) and isinstance(knoten.value, str):
                texte = [knoten.value]
            elif isinstance(knoten, ast.JoinedStr):
                texte = [
                    teil.value
                    for teil in knoten.values
                    if isinstance(teil, ast.Constant) and isinstance(teil.value, str)
                ]
            for text in texte:
                if not _prosa_wertig(text):
                    continue
                if _im_t_call(knoten) if isinstance(knoten, ast.Constant) else False:
                    continue
                if _in_logger_call(knoten):
                    continue
                if _ist_docstring(knoten):
                    continue
                # t(...) bereits im selben Literal? (f\"…{t('x')}…\")
                if "t(" in text:
                    continue
                zeile = getattr(knoten, "lineno", 0) or 0
                if zeile <= 0 or zeile > zeilenzahl:
                    zeile = 0
                _grund = _exempt_match(str(py.relative_to(ROOT)), zeile, text)
                _eintrag = {
                    "datei": str(py.relative_to(ROOT)),
                    "zeile": zeile,
                    "text": text,
                    "kontext": _sink_hinweis(knoten),
                }
                if _grund is not None:
                    _eintrag["grund"] = _grund
                    exempt.append(_eintrag)
                else:
                    kandidaten.append(_eintrag)
    return kandidaten, exempt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Frontend-Prosa-Scanner (D-K3-107)")
    ap.add_argument("--ui-dir", default="Aurik10/ui")
    ap.add_argument("--limit", type=int, default=0, help="nur die ersten N Kandidaten ausgeben")
    ap.add_argument("--out", default=str(ROOT / "reports" / "frontend_prosa_kandidaten.json"))
    args = ap.parse_args(argv)

    ui_dir = ROOT / args.ui_dir
    kandidaten, exempt = scan(ui_dir)
    inventar = {
        "schema": "aurik.frontend_prosa/1",
        "ui_dir": str(ui_dir.relative_to(ROOT)),
        "anzahl": len(kandidaten),
        "exempt_anzahl": len(exempt),
        "kandidaten": kandidaten,
        "exempt": exempt,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inventar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Frontend-Prosa-Kandidaten: {len(kandidaten)} (exempt: {len(exempt)}) → {out}")
    for eintrag in kandidaten[: args.limit]:
        print(f"  {eintrag['datei']}:{eintrag['zeile']} [{eintrag['kontext']}] {eintrag['text'][:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
