"""Tests für den statischen Code-Schwachstellen-Scanner (audit/code_weakness_scanner.py).

Golden-Tests pro Regel: Jede Regel muss ihre Ziel-Schwachstelle zuverlässig
melden (True Positive) und saubere Gegenbeispiele NICHT melden (kein
False Positive). Zusätzlich: Unterdrückungs-Transparenz (suppressed-Zähler).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Scanner liegt unter audit/ — für Import verfügbar machen
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "audit"))

from code_weakness_scanner import scan_workspace


def _write(tmp: Path, rel: str, content: str) -> None:
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _rules(workspace: Path) -> dict[str, list]:
    result = scan_workspace(workspace)
    out: dict[str, list] = {}
    for f in result.findings:
        out.setdefault(f.rule_id, []).append(f)
    return out


def test_bridge_import_flagged_in_cli(tmp_path: Path) -> None:
    _write(tmp_path, "cli/tool.py", "from backend.core.unified_restorer_v3 import UnifiedRestorerV3\n")
    rules = _rules(tmp_path)
    assert rules.get("bridge_import_violation"), "CLI-Direktimport muss gemeldet werden"
    assert rules["bridge_import_violation"][0].severity == "critical"


def test_bridge_import_not_flagged_via_bridge_or_elsewhere(tmp_path: Path) -> None:
    _write(tmp_path, "cli/tool.py", "from backend.api.bridge import get_restorer_classes\n")
    _write(tmp_path, "backend/core/mod.py", "from backend.core.foo import Bar\n")
    _write(tmp_path, "denker/plan.py", "from backend.core.foo import Bar\n")
    assert "bridge_import_violation" not in _rules(tmp_path)


def test_dither_missing_flagged_without_context(tmp_path: Path) -> None:
    _write(tmp_path, "backend/core/dsp/x.py", "audio = audio.astype(np.int16)\n")
    rules = _rules(tmp_path)
    assert rules.get("dither_missing_int_conversion"), "nacktes astype(int16) muss gemeldet werden"


def test_dither_not_flagged_with_context(tmp_path: Path) -> None:
    _write(tmp_path, "backend/core/dsp/x.py", "audio = dither_powr3(audio).astype(np.int16)\n")
    assert "dither_missing_int_conversion" not in _rules(tmp_path)


def test_silent_fallback_flagged_and_logged_not(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "backend/core/ml_router.py",
        "try:\n    x = ml()\nexcept Exception:\n    return 0.5\n",
    )
    assert "silent_fallback_no_log" in _rules(tmp_path)
    _write(
        tmp_path,
        "backend/core/ml_router.py",
        "try:\n    x = ml()\nexcept Exception:\n    logger.warning('fallback')\n    return 0.5\n",
    )
    assert "silent_fallback_no_log" not in _rules(tmp_path)


def test_bare_except_flagged(tmp_path: Path) -> None:
    _write(tmp_path, "backend/core/mod.py", "try:\n    x = f()\nexcept:\n    return 1\n")
    rules = _rules(tmp_path)
    assert rules.get("bare_except"), "bare except muss gemeldet werden"
    assert rules["bare_except"][0].severity == "medium"


def test_module_logger_missing_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "backend/core/mod.py",
        "def f():\n    try:\n        x = g()\n    except Exception:\n        return None\n",
    )
    assert "module_logger_missing" in _rules(tmp_path)
    _write(
        tmp_path,
        "backend/core/mod.py",
        "import logging\nlogger = logging.getLogger(__name__)\ndef f():\n    try:\n        x = g()\n    except Exception:\n        return None\n",
    )
    assert "module_logger_missing" not in _rules(tmp_path)


def test_nan_inf_guard_missing_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "backend/core/phases/phase_99_test.py",
        "import numpy as np\ndef process(audio):\n    return audio * 2.0\n",
    )
    assert "nan_inf_guard_missing" in _rules(tmp_path)
    _write(
        tmp_path,
        "backend/core/phases/phase_99_test.py",
        "import numpy as np\ndef process(audio):\n    return np.nan_to_num(audio * 2.0)\n",
    )
    assert "nan_inf_guard_missing" not in _rules(tmp_path)


def test_determinism_wallclock_in_decision_flagged(tmp_path: Path) -> None:
    """§G5 (copilot-instructions.md): Wall-Clock in Entscheidungslogik ist der Verstoß (nicht jede Messung)."""
    _write(
        tmp_path,
        "backend/core/mod.py",
        "import time\ndef keep(ts):\n    now = time.time()\n    if now - ts > 86400:\n        return False\n    return True\n",
    )
    rules = _rules(tmp_path)
    assert rules.get("determinism_time_usage"), "Wall-Clock im Vergleich muss gemeldet werden"
    assert rules["determinism_time_usage"][0].severity == "medium"
    assert rules["determinism_time_usage"][0].line == 3


def test_determinism_measurement_not_flagged(tmp_path: Path) -> None:
    """Messung/Zeitstempel sind kein §G5-Verstoß — aber transparent unterdrückt."""
    _write(
        tmp_path,
        "backend/core/mod.py",
        "import time\ndef run():\n    t0 = time.time()\n    do_work()\n    return time.time() - t0\n",
    )
    result = scan_workspace(tmp_path)
    assert "determinism_time_usage" not in {f.rule_id for f in result.findings}
    assert result.suppressed.get("determinism_time_usage", 0) >= 2


def test_determinism_persisted_timestamp_not_flagged(tmp_path: Path) -> None:
    """Persistierte Zeitstempel MÜSSEN Wall-Clock bleiben (monotonic wäre prozesslokal)."""
    _write(
        tmp_path,
        "backend/core/mod.py",
        "import time\ndef state():\n    return {'timestamp': time.time(), 'start_time': time.time()}\n",
    )
    assert "determinism_time_usage" not in _rules(tmp_path)


def test_print_in_production_aggregated(tmp_path: Path) -> None:
    _write(tmp_path, "backend/core/mod.py", "print(1)\nprint(2)\nprint(3)\n")
    rules = _rules(tmp_path)
    assert rules.get("print_in_production"), "≥3 print() müssen aggregiert gemeldet werden"
    # 2 Treffer unter Schwelle → nur suppressed
    _write(tmp_path, "backend/core/mod.py", "print(1)\nprint(2)\n")
    result = scan_workspace(tmp_path)
    assert "print_in_production" not in {f.rule_id for f in result.findings}
    assert result.suppressed.get("print_in_production", 0) >= 1


def test_print_only_in_comments_not_flagged(tmp_path: Path) -> None:
    """Prosa ist kein Code: auskommentierte print()-Zeilen sind kein Befund."""
    _write(
        tmp_path,
        "backend/core/dsp/mod.py",
        '# print(f"a")\n#     print(f"b")\n# print(f"c")\n# print(f"d")\n',
    )
    result = scan_workspace(tmp_path)
    assert "print_in_production" not in {f.rule_id for f in result.findings}
    assert "print_in_production" not in result.suppressed


def test_print_in_docstring_not_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "backend/core/dsp/mod.py",
        'def f() -> None:\n    """Nutze print(1), print(2), print(3) im Debugger."""\n    return None\n',
    )
    assert "print_in_production" not in _rules(tmp_path)


def test_print_in_tool_script_excluded(tmp_path: Path) -> None:
    """backend/core/scripts/ sind Werkzeuge — stdout ist dort die Schnittstelle."""
    _write(tmp_path, "backend/core/scripts/lint_x.py", "print(1)\nprint(2)\nprint(3)\nprint(4)\n")
    assert "print_in_production" not in _rules(tmp_path)


def test_ast_cap_suppressed_counted(tmp_path: Path) -> None:
    # 5 bare excepts: 3 gemeldet (AST-Cap), 2 im suppressed-Zähler sichtbar
    body = "".join("try:\n    x = f()\nexcept:\n    return 1\n" for _ in range(5))
    _write(tmp_path, "backend/core/mod.py", body)
    result = scan_workspace(tmp_path)
    bare = [f for f in result.findings if f.rule_id == "bare_except"]
    assert len(bare) == 3, f"AST-Cap greift bei 3: {len(bare)}"
    assert result.suppressed.get("bare_except", 0) >= 2, "unterdrückte AST-Befunde müssen gezählt werden"


def test_critical_sorted_first_despite_max_findings(tmp_path: Path) -> None:
    # Viele Low-Befunde + ein CRITICAL: bei max_findings=1 darf nur das CRITICAL bleiben
    for i in range(5):
        _write(tmp_path, f"backend/core/mod_{i}.py", "print(1)\nprint(2)\nprint(3)\n")
    _write(tmp_path, "cli/tool.py", "from backend.core.unified_restorer_v3 import UnifiedRestorerV3\n")
    result = scan_workspace(tmp_path, max_findings=1)
    assert len(result.findings) == 1
    assert result.findings[0].severity == "critical"
    assert result.suppressed.get("truncated_max_findings", 0) >= 1


def test_bridge_import_in_comment_not_flagged(tmp_path: Path) -> None:
    """§V4 (copilot-instructions.md) ist CRITICAL: ein Kommentar/Docstring darf keinen Block auslösen."""
    _write(
        tmp_path,
        "cli/tool.py",
        '"""Verboten (§V4 (copilot-instructions.md)): from backend.core.unified_restorer_v3 import UnifiedRestorerV3"""\n'
        "# import backend.core.audio_utils\n"
        "from backend.api.bridge import get_restorer_classes\n",
    )
    assert "bridge_import_violation" not in _rules(tmp_path)


def test_silent_fallback_accepts_underscore_logger(tmp_path: Path) -> None:
    """§V6 (copilot-instructions.md) ist erfüllt, wenn überhaupt geloggt wird — auch als _logger/LOGGER."""
    _write(
        tmp_path,
        "backend/core/mod.py",
        "try:\n    x = ml()\nexcept Exception as e:\n    _logger.warning('Fallback: %s', e)\n    return None\n",
    )
    assert "silent_fallback_no_log" not in _rules(tmp_path)
    _write(
        tmp_path,
        "backend/core/mod.py",
        "try:\n    x = ml()\nexcept Exception as e:\n    LOGGER.error('Fallback: %s', e)\n    return None\n",
    )
    assert "silent_fallback_no_log" not in _rules(tmp_path)


def test_handler_without_any_log_call_flagged(tmp_path: Path) -> None:
    """Ohne jeden Log-Aufruf bleibt der Fallback stumm → Befund (§V6 (copilot-instructions.md))."""
    _write(
        tmp_path,
        "backend/core/mod.py",
        "def run(audio):\n    try:\n        return work(audio)\n    except Exception:\n        return audio\n",
    )
    assert "silent_fallback_no_log" in _rules(tmp_path)


def test_audit_facade_counts_as_logging(tmp_path: Path) -> None:
    """Verifizierte Audit-Fassade (`_audit_log`) erfüllt §V6 (copilot-instructions.md).

    2026-10-05 geprüft: alle `_audit_log`-Definitionen im Repo delegieren an
    einen echten Logger (logger.error/warning/info bzw. _logger.*). Die Regel
    meldete sie vorher fälschlich als stumm.
    """
    _write(
        tmp_path,
        "backend/core/mod.py",
        "import logging\nlogger = logging.getLogger(__name__)\n"
        'def _audit_log(level, message):\n    {"error": logger.error}.get(level, logger.info)(message)\n'
        "def run(audio):\n    try:\n        return work(audio)\n"
        "    except Exception as e:\n        _audit_log('error', str(e))\n        return audio\n",
    )
    assert "silent_fallback_no_log" not in _rules(tmp_path)


def test_walltime_clock_mismatch_detected(tmp_path: Path) -> None:
    """Persistierter Wall-Clock-Wert vs. time.monotonic() → Epochen-Mismatch."""
    _write(
        tmp_path,
        "backend/core/mod.py",
        "import time\ndef store(fp):\n    fp.last_updated = time.time()\n    return fp\n"
        "def load(fp):\n    if time.monotonic() - fp.last_updated > 86400:\n        return None\n    return fp\n",
    )
    rules = _rules(tmp_path)
    assert rules.get("walltime_clock_mismatch"), "Epochen-Mismatch muss gemeldet werden"
    assert rules["walltime_clock_mismatch"][0].severity == "high"


def test_walltime_clock_consistent_not_flagged(tmp_path: Path) -> None:
    """Konsistente Uhr (Dauer-Messung mit perf_counter) ist kein Befund."""
    _write(
        tmp_path,
        "backend/core/mod.py",
        "import time\ndef run():\n    t0 = time.perf_counter()\n    work()\n    return time.perf_counter() - t0\n",
    )
    assert "walltime_clock_mismatch" not in _rules(tmp_path)


def test_wallclock_ttl_housekeeping_classified(tmp_path: Path) -> None:
    """TTL-Ablauf im Datei-Haushalt: kein §G5-Verstoß, aber sichtbar ausgewiesen."""
    _write(
        tmp_path,
        "backend/core/mod.py",
        "import os\nimport time\ndef cleanup_expired(max_age_days=7):\n"
        "    cutoff = time.time() - max_age_days * 86400\n    removed = 0\n"
        "    for name in os.listdir('.'):\n        if os.path.getmtime(name) < cutoff:\n"
        "            os.remove(name)\n            removed += 1\n    return removed\n",
    )
    rules = _rules(tmp_path)
    assert "determinism_time_usage" not in rules, "TTL-Haushalt ist kein Audio-Determinismus-Verstoß"
    assert rules.get("wallclock_ttl_housekeeping"), "TTL-Haushalt muss sichtbar bleiben"


def test_module_logger_in_comment_does_not_count(tmp_path: Path) -> None:
    """Ein auskommentiertes getLogger() entlastet das Modul nicht."""
    _write(
        tmp_path,
        "backend/core/mod.py",
        "# logger = logging.getLogger(__name__)\ndef f():\n    try:\n        x = g()\n    except Exception:\n        return None\n",
    )
    assert "module_logger_missing" in _rules(tmp_path)


def test_nan_guard_token_in_docstring_does_not_count(tmp_path: Path) -> None:
    """§0a verlangt echten Schutz-Code, keinen Docstring-Hinweis."""
    _write(
        tmp_path,
        "backend/core/phases/phase_98_demo.py",
        'import numpy as np\ndef process(audio):\n    """Nutzt früher np.nan_to_num()."""\n    return audio * 2.0\n',
    )
    assert "nan_inf_guard_missing" in _rules(tmp_path)


def test_dither_docstring_example_not_flagged(tmp_path: Path) -> None:
    """§V5 (copilot-instructions.md) prüft Audio-Quantisierung im Code, nicht Docstring-Beispiele."""
    _write(
        tmp_path,
        "backend/core/dsp/x.py",
        'def f(audio):\n    """Historisch: audio.astype(np.int16) ohne Dither."""\n    return audio\n',
    )
    assert "dither_missing_int_conversion" not in _rules(tmp_path)
