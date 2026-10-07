"""Frontend-VERBOTE-Linter: Magic Numbers + UI-Anti-Patterns.

Scannt Aurik10/ui/ auf:
- Magic Numbers: numerische Literale die UI-Schwellwerte darstellen
- UI-Thread-Blocker: time.sleep() ohne Worker-Thread
- Hardcodierte UI-Texte > 80 Zeichen (sollten in i18n/callbacks)
- **Hartkodierte Nutzertexte jeder Länge ohne `t()`** (4. Prüfung, 2026-10-07)

Baseline-basiert: neue Verstöße → FAIL (§G123-Äquivalent für Frontend).

**Befund 2026-10-07 (Ursache der i18n-Drift):** Die Prüfung "Lange UI-Texte"
sah nur Zeilen > 80 Zeichen und **nur** wenn das Wort `text/label/title` darin
vorkam — deshalb blieben 49 hartkodierte Nutzertexte in `modern_window.py`
unsichtbar (z. B. „Schließen", „Abtastrate", „▶ Vorschau"). Zugleich war der
Test rein informativ (`assert magic_count >= 0` — immer wahr), konnte also
nie fehlschlagen (§V7 copilot-instructions.md: Prüfung, die nichts prüft).
Die vierte Prüfung erfasst user-sichtbare Setter-Literale **jeder** Länge,
die nicht durch `t(...)` laufen, und ist **fail-closed gegen neue Verstöße**.

Prüfsumme `_BASELINE` (Sollzustand = Ist-Zustand am 2026-10-07 nach der
Korrektur des Zeitbudget-Dialogs und der Statuspanel-Labels):

* `hardcoded_texts` — kurz oder lang, ohne `t()`. **Fail-closed.** Jeder neue
  sichtbare String ohne `t()` lässt den Test fallen; Fixes senken den Wert und
  der Sollwert ist im selben Commit nachzuziehen (keine stille Alterung,
  Lehre aus D-K3-5).
* `long_strings`, `magic_numbers`, `thread_blockers` — heuristisch (Wortlisten,
  Zahlen-Schwellen) und deshalb **informativ**: sie werden gemeldet, aber nicht
  erzwungen, weil ein Fehlalarm sonst jeden Commit blockiert.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

import pytest

# §G9 (copilot-instructions.md): Kommentar-/Docstring-Blankung NICHT nachbauen.
# Befund 2026-10-07: eine eigene Kopie war **blind** geworden — ein abschliessendes
# `"""` einer praefixierten Docstring (`r"""...`) setzte den Zustand auf
# "in Docstring", danach wurde der REST DER DATEI geblankt (in
# `modern_window.py` ab Zeile 10444) und das Gate uebersah 28 Fundstellen
# (19 statt 47). Die kanonische `_blank_prose` (in D-K3-5 gegen 1660 Dateien
# validiert) macht das richtig — sie wird importiert, nicht kopiert.
import scripts.wohlklang_gate as _wohlklang_gate

CANONICAL_FILES: set[str] = {
    "Aurik10/ui/ui_constants.py",  # UIConstants (nach Erstellung)
}

UI_DIR = Path("Aurik10/ui")

# ── 4. Prüfung: user-sichtbare Setter-Literale ohne t() ───────────────────────
# Nur Setter, deren Argument der Nutzer SIEHT. `setObjectName`/`styleSheet`
# fehlen bewusst: Objektnamen sind nicht sichtbar, Stylesheets sind keine Texte.
_VISIBLE_SETTERS: tuple[tuple[str, str], ...] = (
    ("setText", r"\.setText\(\s*f?(?P<q>[\"'])(?P<s>(?:\\.|(?!\1).)*)(?P=q)"),
    ("setToolTip", r"\.setToolTip\(\s*f?(?P<q>[\"'])(?P<s>(?:\\.|(?!\1).)*)(?P=q)"),
    ("setWindowTitle", r"\.setWindowTitle\(\s*f?(?P<q>[\"'])(?P<s>(?:\\.|(?!\1).)*)(?P=q)"),
    ("setPlaceholderText", r"\.setPlaceholderText\(\s*f?(?P<q>[\"'])(?P<s>(?:\\.|(?!\1).)*)(?P=q)"),
    ("setTitle", r"\.setTitle\(\s*f?(?P<q>[\"'])(?P<s>(?:\\.|(?!\1).)*)(?P=q)"),
    ("addItem", r"\.addItem\(\s*f?(?P<q>[\"'])(?P<s>(?:\\.|(?!\1).)*)(?P=q)"),
    ("setInformativeText", r"\.setInformativeText\(\s*f?(?P<q>[\"'])(?P<s>(?:\\.|(?!\1).)*)(?P=q)"),
    ("QPushButton", r"QPushButton\(\s*f?(?P<q>[\"'])(?P<s>(?:\\.|(?!\1).)*)(?P=q)"),
    ("QLabel", r"QLabel\(\s*f?(?P<q>[\"'])(?P<s>(?:\\.|(?!\1).)*)(?P=q)"),
    # Nutzertexte, die über einen **Helfer** zur Anzeige kommen. Befund
    # 2026-10-07: `_show_toast(f"… {_reassure or 'Bitte noch kurz warten'}")`
    # war für die erste Fassung dieser Prüfung unsichtbar, weil sie nur direkte
    # Setter ansah — der Text erreicht den Nutzer aber genauso.
    ("_show_toast", r"_show_toast\(\s*f?(?P<q>[\"'])(?P<s>(?:\\.|(?!\1).)*)(?P=q)"),
    ("set_status", r"\.set_status\(\s*f?(?P<q>[\"'])(?P<s>(?:\\.|(?!\1).)*)(?P=q)"),
)

# Nicht-Nutzersichtbares/Triviales: reine Ziffern-/Symbolfolgen, Format-Reste.
_TEXT_NOISE_RE = re.compile(r"^[\W\d_]*$")
_TIME_FORMAT_RE = re.compile(r"^[\d.,:\s%hHmMsS]*$")
_FMT_PLACEHOLDER_RE = re.compile(r"\{[^{}]*\}")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
# Markup, das über mehrere Zeilen geteilt ist (`"<img src=…" \n " width=…"`) —
# ohne schließendes `>` bliebe der Tag-Text als „Prosa" übrig (Fehlalarm).
_HTML_UNBALANCED_RE = re.compile(r"<[^>]*$")
_HAS_LETTERS_RE = re.compile(r"[A-Za-zÄÖÜäöüß]")
# Escape-Folgen im Quelltext zählen nicht als Buchstaben: `"\n"` ist ein
# Zeilenumbruch, nicht das Wort „n" (Fehlalarm 2026-10-07: 7 Treffer), und
# `"\U0001f527"` (🔧) enthält kein „U" als Text.
_ESCAPE_SEQ_RE = re.compile(r"\\[ntrbfav0\\'\"`]|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}|\\N\{[^}]*\}|\\x[0-9a-fA-F]{2}")
# Einheiten und Symbole sind sprachneutral — keine Übersetzung nötig.
_UNIT_ONLY_RE = re.compile(r"^(?:dB|dBFS|dBTP|kHz|Hz|ms|s|%|LUFS|bit|kbit/s|×)$", re.I)
_TRIM_CHARS = " \t·|/:-–—()[]{}.,%+"


def _is_hardcoded_prose(literal: str) -> bool:
    """Trägt das Literal eigenen übersetzbaren Text — oder nur Gerüst?

    Fehlalarm-Klassen werden ausgeschlossen (2026-10-07 am Bestand gemessen):

    * **schon i18n-isierter f-String** — `f"▶  {t('action.play')}"` enthält die
      Übersetzung, nur eben im Platzhalter.
    * **Markup ohne Text** — `"<span style='font-size:9pt;'>"` ist Gestaltung;
      `"<b>⌨ Tastenkürzel</b>"` trägt dagegen Prosa und bleibt ein Befund.
    * **Format-Vorlage** — `"{_m}:{_s:02d}"` setzt Zahlen zusammen; nach dem
      Entfernen der Platzhalter bleibt kein Buchstabe übrig.
    * **Escape-Folgen und Einheiten** — `"\\n"` ist kein Wort, `"dB"` ist
      sprachneutral.

    Vorbereinigt wird immer nur eine **Kopie**; der Befund zeigt das Original.
    """
    if "t(" in literal:
        return False
    stripped = _HTML_TAG_RE.sub("", literal)
    stripped = _HTML_UNBALANCED_RE.sub("", stripped)
    stripped = _FMT_PLACEHOLDER_RE.sub("", stripped)
    stripped = _ESCAPE_SEQ_RE.sub("", stripped).strip(_TRIM_CHARS)
    if not stripped or _UNIT_ONLY_RE.match(stripped):
        return False
    return bool(_HAS_LETTERS_RE.search(stripped))


def _hardcoded_visible_texts(text: str) -> list[dict]:
    """Setter-Literale finden, die **nicht** durch `t(...)` laufen.

    `setText(t("key"))` matcht bewusst nicht: die Muster verlangen ein
    String-Literal direkt nach der Klammer (`t(` beginnt mit einem Namen).
    """
    blanked = _blank_prose_local(text)
    found: list[dict] = []
    for setter, pattern in _VISIBLE_SETTERS:
        for m in re.finditer(pattern, blanked):
            literal = m.group("s")
            if len(literal) < 2:
                continue
            if _TEXT_NOISE_RE.match(literal) or _TIME_FORMAT_RE.match(literal):
                continue
            if not _is_hardcoded_prose(literal):
                continue
            found.append(
                {
                    "line": blanked[: m.start()].count("\n") + 1,
                    "setter": setter,
                    "text": literal[:90],
                }
            )
    return found


def _blank_prose_local(text: str) -> str:
    """Kommentare und Docstrings positionsgetreu blanken (kanonische Quelle).

    §G9 (copilot-instructions.md): Delegiert an `scripts.wohlklang_gate._blank_prose`
    — eine Umsetzung statt zweier.
    """
    return _wohlklang_gate._blank_prose(text)


# Sollzustand (2026-10-07, korrigiert). Nur `hardcoded_texts` ist fail-closed.
#
# `hardcoded_texts` = 47: Die **erste** Zahl (22) war zu niedrig, weil die
# eigene Prosa-Blankung dieses Tests **blind** war — ein abschliessendes `"""`
# einer praefixierten Docstring (`r"""...`) setzte den Zustand dauerhaft auf "in
# Docstring", danach wurde der REST der Datei geblankt (in
# `modern_window.py` ab Zeile 10444) und 28 Fundstellen blieben unsichtbar
# (19 statt 47). Seit 2026-10-07 delegiert `_blank_prose_local` an die
# kanonische `scripts.wohlklang_gate._blank_prose` (§G9 copilot-instructions.md:
# eine Umsetzung; D-K3-5-validiert).
#
# Wahre Ausgangslage: **94** (4 Dateien). 47 sind auf `t()` umgestellt
# (Ergebnisleiste, Dialoge, Fenstertitel, Schadensbehebung, Abtastrate,
# Panel-Buttons, Herz-/Fatigue-Labels des Berichts, Panel-Tooltip).
# Die verbleibenden 47 (44 in `modern_window.py`, 3 in
# `song_prognose_widget.py`) sind im SOTA-Defizit-Register als **D-K3-10**
# gefuehrt. Der Wert darf nur **sinken**; ein Anstieg laesst diesen Test
# fallen.
_BASELINE: dict[str, int] = {
    "hardcoded_texts": 47,
    # Informativ (heuristisch, nicht erzwungen) — Stand 2026-10-07:
    "long_strings": 69,
    "magic_numbers": 196,
    "thread_blockers": 6,
}


def _scan_ui_file(filepath: Path) -> dict[str, list[dict]]:
    """Scannt eine UI-Datei auf Anti-Patterns."""
    results: dict[str, list[dict]] = {
        "magic_numbers": [],
        "thread_blockers": [],
        "long_strings": [],
        "hardcoded_texts": [],
    }
    rel = str(filepath.relative_to(Path.cwd())) if filepath.is_relative_to(Path.cwd()) else str(filepath)
    if any(cf in rel for cf in CANONICAL_FILES):
        return results

    try:
        source = filepath.read_text(encoding="utf-8")
    except Exception:
        return results

    lines = source.split("\n")
    in_docstring = False

    for lineno_1, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if '"""' in stripped or "'''" in stripped:
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue

        # 1. time.sleep() — UI-Thread-Blocker
        if "time.sleep(" in stripped:
            results["thread_blockers"].append(
                {
                    "line": lineno_1,
                    "text": stripped[:120],
                }
            )

        # 2. Magic Numbers: float/int Literale in UI-Kontext
        #    (Zahlen die wie Schwellwerte/Dimensionen aussehen)
        magic = re.findall(
            r"(?:width|height|margin|padding|spacing|size|threshold|"
            r"delay|timeout|interval|duration|opacity|scale|factor|"
            r"ratio|limit|max|min|offset|radius)\s*[:=]\s*([\d.]+)",
            stripped,
        )
        for m in magic:
            val = float(m)
            # Nur nicht-triviale Zahlen (>3, nicht 0/1/100)
            if val > 3 and val not in (10, 20, 30, 50, 100, 200, 500, 1000):
                results["magic_numbers"].append(
                    {
                        "line": lineno_1,
                        "text": stripped.strip()[:120],
                        "value": val,
                    }
                )

        # 3. Lange UI-Texte (sollten Callback-basiert oder in i18n sein)
        if len(stripped) > 80 and ('"' in stripped or "'" in stripped):
            if any(kw in stripped.lower() for kw in ("text", "label", "title", "message", "tooltip")):
                results["long_strings"].append(
                    {
                        "line": lineno_1,
                        "text": stripped[:150],
                    }
                )

        # 4. Nutzertexte JEDER Länge ohne t() (§VI.5 copilot-instructions.md).

    # Eigene Analyse statt Zeilen-Schleife: die sieht mehrzeilige Setter und
    # f-Strings nicht. Läuft einmal pro Datei, nach der Zeilen-Schleife.
    results["hardcoded_texts"] = _hardcoded_visible_texts(source)

    return results


def _scan_all_ui() -> dict[str, dict]:
    """Scannt alle Python-Dateien in Aurik10/ui/."""
    all_results: dict[str, dict] = {}
    if not UI_DIR.exists():
        return all_results

    for py_file in sorted(UI_DIR.rglob("*.py")):
        if any(p.startswith(".") or p in ("__pycache__",) for p in py_file.parts):
            continue
        results = _scan_ui_file(py_file)
        rel = str(py_file.relative_to(Path.cwd())) if py_file.is_relative_to(Path.cwd()) else str(py_file)
        total = sum(len(v) for v in results.values())
        if total > 0:
            all_results[rel] = results

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Pytest-Test
# ═══════════════════════════════════════════════════════════════════════════════


def test_no_new_frontend_anti_patterns() -> None:
    """Frontend-VERBOTE: keine NEUEN Anti-Patterns (§VI.5, fail-closed).

    `hardcoded_texts` ist **fail-closed**: jeder neue nutzersichtbare String
    ohne `t(...)` lässt den Test fallen. Die heuristischen Zähler werden nur
    berichtet (Fehlalarm-Gefahr durch Wortlisten/Zahlen-Schwellen).
    """
    results = _scan_all_ui()

    counts = {
        "magic_numbers": sum(len(v["magic_numbers"]) for v in results.values()),
        "thread_blockers": sum(len(v["thread_blockers"]) for v in results.values()),
        "long_strings": sum(len(v["long_strings"]) for v in results.values()),
        "hardcoded_texts": sum(len(v["hardcoded_texts"]) for v in results.values()),
    }

    print("\nFrontend-VERBOTE Scan:")
    for key, value in counts.items():
        marker = "" if value <= _BASELINE[key] else "  <-- NEU"
        print(f"  {key:<18} {value:4d}  (Sollwert {_BASELINE[key]}){marker}")

    if counts["hardcoded_texts"] > _BASELINE["hardcoded_texts"]:
        offenders: list[str] = []
        for path, by_rule in results.items():
            for entry in by_rule["hardcoded_texts"]:
                offenders.append(f"  {path}:{entry['line']} [{entry['setter']}] {entry['text']!r}")
        pytest.fail(
            "Hartkodierte Nutzertexte ohne t() — §VI.5 (copilot-instructions.md):\n" + "\n".join(sorted(offenders)[:40])
        )

    assert counts["hardcoded_texts"] <= _BASELINE["hardcoded_texts"]


if __name__ == "__main__":
    results = _scan_all_ui()
    magic_count = sum(len(v["magic_numbers"]) for v in results.values())
    blocker_count = sum(len(v["thread_blockers"]) for v in results.values())
    string_count = sum(len(v["long_strings"]) for v in results.values())

    print("Frontend-VERBOTE:")
    print(f"  Magic Numbers:  {magic_count}")
    print(f"  Thread-Blocker: {blocker_count}")
    print(f"  Lange Texte:    {string_count}")
    print()

    ranked = sorted(results.items(), key=lambda x: sum(len(v) for v in x[1].values()), reverse=True)
    for path, counts in ranked[:10]:
        m = len(counts["magic_numbers"])
        t = len(counts["thread_blockers"])
        s = len(counts["long_strings"])
        if m + t + s > 0:
            print(f"  {path}: {m}M {t}T {s}S")
