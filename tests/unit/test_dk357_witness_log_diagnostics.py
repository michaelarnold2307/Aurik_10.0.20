"""§D-K3-57 (2026-10-07): Reinhör-Witness-Logzeile trägt den ECHTEN pre_echo_db-Wert.

Befund (Walzer-Lauf „Trio Schweizer – 13 Tage"): Die Witness meldete nach
phase_01/05/03 `pre_echo` bei komplett neutralen Nachbarwerten
(pitch=0.0c mod=0.0c hnr=0.0dB hf=0.000 loud=0.0dB) — ohne den echten
pre_echo_db-Wert war nicht entscheidbar, wie weit über der −12-dB-Schwelle der
Trigger lag (D-K3-57; Repro weiterhin per AURIK_WITNESS_DEBUG=1 vorgesehen).
Fix: `pre_echo=%.1fdB` als 6. Messwert in der Logzeile
(backend/core/unified_restorer_v3.py, Reinhör-Witness-Block).

Wiring-Guard (Muster D-K3-35, bewusst textuell — die Zeile liegt im 47k-Zeilen-
Rumpf der Pipeline): Der Log-Call muss den Platzhalter UND das Argument
`_lw_res.pre_echo_db` tragen, und die Platzhalter-/Argument-Zählung muss
stimmen — logging formatiert lazy; ein fehlendes Argument beendet die Zeile
zur Laufzeit als „Logging error" und macht genau die Diagnostik wieder blind,
die dieser Fix herstellt (§G8 (copilot-instructions.md)).
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

_UV3 = Path(__file__).resolve().parents[2] / "backend" / "core" / "unified_restorer_v3.py"

# Eindeutiger Anker: nur das Format-Literal, nicht der gleichlautende Kommentar (Zeile 42659).
_FORMAT_ANCHOR = '"👂 Reinhör-Witness %s: %s (pitch='


def _witness_log_call() -> ast.Call:
    """Findet den Reinhör-Witness-logger.warning-Call und liefert seinen AST-Knoten."""
    src = _UV3.read_text(encoding="utf-8")
    idx = src.find(_FORMAT_ANCHOR)
    assert idx != -1, "Witness-Logformat nicht gefunden"
    call_start = src.rfind("logger.warning(", 0, idx)
    assert call_start != -1, "logger.warning-Call vor der Witness-Zeile nicht gefunden"
    attr_end = src.find("_lw_res.pre_echo_db", idx)
    assert attr_end != -1, "pre_echo_db-Argument fehlt in der Witness-Zeile (D-K3-57-Regression)"
    call_end = src.find(")", attr_end) + 1
    call = ast.parse(src[call_start:call_end]).body[0].value
    assert isinstance(call, ast.Call)
    return call


def test_witness_log_traegt_den_echten_pre_echo_wert() -> None:
    call = _witness_log_call()
    fmt = call.args[0]
    assert isinstance(fmt, ast.Constant) and isinstance(fmt.value, str)
    assert "pre_echo=%.1fdB" in fmt.value, "Format-Platzhalter für den pre_echo_db-Wert fehlt"
    attr_names = [a.attr for a in call.args if isinstance(a, ast.Attribute)]
    assert "pre_echo_db" in attr_names, "pre_echo_db wird nicht als Argument übergeben"


def test_witness_log_platzhalter_und_argumente_sind_konsistent() -> None:
    """Laufzeit-Wirkung: der echte Logging-Mechanismus darf nicht scheitern."""
    call = _witness_log_call()
    fmt = call.args[0]
    assert isinstance(fmt, ast.Constant) and isinstance(fmt.value, str)
    n_values = len(call.args) - 1  # ohne den Format-String selbst
    try:
        # Derselbe Pfad wie LogRecord.getMessage() zur Laufzeit.
        logging.LogRecord("t", logging.WARNING, "", 1, fmt.value, (0.0,) * n_values, None).getMessage()
    except TypeError as exc:  # pragma: no cover - Fehlerpfad ist der Testzweck
        raise AssertionError(
            f"Witness-Logzeile: Platzhalter/Argumente inkonsistent → „Logging error“ zur Laufzeit: {exc}"
        ) from exc
