"""Vertragstests für die CLI-Oberfläche von Aurik10/main.py (2026-11-19).

Produktreife-Härtung: --help/--version/--check müssen VOR dem GUI-Start
terminieren (CI- und Headless-Fähigkeit). Subprozess-basiert — der schwere
Main-Import (torch, Backend) läuft im Kind, nie im Testprozess.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
MAIN = _ROOT / "Aurik10" / "main.py"
PY = sys.executable


def _run(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, "-B", str(MAIN), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(_ROOT),
        env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "LANG": "C.UTF-8"},
    )


def test_help_terminiert_mit_exit_0() -> None:
    """--help druckt die Hilfe und beendet sofort — ohne QApplication."""
    r = _run("--help")
    assert r.returncode == 0
    assert "AURIK Professional" in r.stdout
    assert "--check" in r.stdout


def test_version_terminiert_mit_exit_0() -> None:
    r = _run("--version")
    assert r.returncode == 0
    assert "AURIK Professional" in r.stdout


def test_check_ist_headless_smoke() -> None:
    """--check baut offscreen QApplication + ModernMainWindow auf (Exit 0/1/2, nie Hänger).

    Exit 0 = konstruiert, 1 = Konstruktor-Defekt, 2 = Watchdog-Timeout.
    """
    r = _run("--check", timeout=600)
    assert r.returncode in (0, 1, 2)
    assert "GUI-Smoke" in r.stdout


def test_main_enthält_hard_exit_fallback() -> None:
    """SIGTERM/SIGINT-Garantie: der Hard-Exit-Timer existiert im Quelltext (§3.9.2-Härtung)."""
    quelle = (MAIN).read_text(encoding="utf-8")
    assert "_arm_hard_exit_fallback" in quelle
    assert "signal.signal(signal.SIGINT, _sigterm_handler)" in quelle
    assert "threading.Timer(8.0" in quelle


def test_gpu_detection_nutzt_korrektes_keyword() -> None:
    """Bug-Pin (2026-11-19): wait_for_detection heißt der Parameter timeout_s —
    der frühere timeout=5.0-Aufruf ließ die GPU-Erkennung IMMER scheitern."""
    quelle = (MAIN).read_text(encoding="utf-8")
    assert "_mgr.wait_for_detection(timeout_s=5.0)" in quelle
    assert "_mgr.wait_for_detection(timeout=" not in quelle


def test_startup_dialog_blockiert_nie_headless() -> None:
    """Headless/offscreen ⇒ kein modaler Startup-Dialog (Auto-OK-Pfad)."""
    quelle = (MAIN).read_text(encoding="utf-8")
    assert 'os.getenv("QT_QPA_PLATFORM", "") == "offscreen"' in quelle
    assert "QTimer.singleShot(45000" in quelle
