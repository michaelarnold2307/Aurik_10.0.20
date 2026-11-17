#!/usr/bin/env python3
"""Fork-Stellen-Finder für den Aurik-Restaurierungs-Pfad.

Zweck
-----
Bei jedem ``os.fork()`` (und damit bei jedem ``multiprocessing``-``fork``-
Start sowie bei jedem ``joblib``/``loky``-Prozess-Pool) wird **sofort** der
Python-Stack des Aufrufers in eine Dump-Datei geschrieben. Der ``before``-Hook
läuft im Vaterprozess direkt vor dem ``fork`` und zeigt damit exakt, *wer* im
Restaurierungs-Pfad forkt (Aufruf-Kette von ``os.fork`` bis zur Ursprungs-
Stelle im Aurik-Code).

Verwendung
----------
    python scripts/fork_site_finder.py --input <datei> --output <ziel> [--dry-run]

Alle übrigen Argumente werden unverändert an ``cli.aurik_cli.main()``
durchgereicht. Die Dumps landen in ``$FORK_DUMP_DIR`` (Standard
``/tmp/aurik_fork_dumps``).

Bewusst NICHT deterministisch / NICHT im Produktionspfad: reines Debug-Werkzeug.
"""

from __future__ import annotations

import os
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DUMP_DIR = Path(os.environ.get("FORK_DUMP_DIR", "/tmp/aurik_fork_dumps"))


def _thread_name(tid: int) -> str:
    for t in threading.enumerate():
        if t.ident == tid:
            return f"{t.name} (daemon={t.daemon})"
    if tid == threading.main_thread().ident:
        return "MainThread"
    return "?"


def _all_thread_stacks() -> str:
    """Stack aller lebenden Threads (nicht nur des aktuellen)."""
    frames = sys._current_frames()
    parts: list[str] = []
    for tid, frame in sorted(frames.items()):
        parts.append(f"\n--- Thread {tid} ({_thread_name(tid)}) ---\n")
        parts.append("".join(traceback.format_stack(frame)))
    return "".join(parts)


def _dump(tag: str, extra: str = "") -> None:
    try:
        DUMP_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        payload = (
            f"===== {tag} =====\n"
            f"timestamp: {ts}\n"
            f"pid: {os.getpid()}\n"
            f"thread: {threading.current_thread().name}\n"
            f"{extra}\n"
            f"{_all_thread_stacks()}\n"
        )
        path = DUMP_DIR / f"fork_{os.getpid()}_{tag}.txt"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(payload)
        # Kurzer Hinweis auch auf stderr, damit er sofort sichtbar ist.
        print(
            f"[fork-site-finder] {tag} pid={os.getpid()} -> {path}",
            file=sys.stderr,
            flush=True,
        )
    except Exception as exc:
        print(f"[fork-site-finder] dump fehlgeschlagen: {exc}", file=sys.stderr, flush=True)


def _before_fork() -> None:
    _dump(
        "before_fork_in_parent",
        "Dieser Stack zeigt WER gerade fork() aufruft (Parent, vor dem fork).",
    )


def _after_in_parent() -> None:
    _dump("after_fork_in_parent", "Parent direkt nach dem fork.")


def _after_in_child() -> None:
    _dump("after_fork_in_child", "Kind direkt nach dem fork (single-threaded).")


def install() -> None:
    """Registriert die at-fork-Hooks. Muss VOR jedem fork aufgerufen werden."""
    if not hasattr(os, "register_at_fork"):
        print("[fork-site-finder] os.register_at_fork nicht verfügbar", file=sys.stderr)
        return
    os.register_at_fork(
        before=_before_fork,
        after_in_parent=_after_in_parent,
        after_in_child=_after_in_child,
    )
    print(f"[fork-site-finder] installiert, Dump-Verzeichnis={DUMP_DIR}", file=sys.stderr)


def _run_cli(cli_args: list[str]) -> int:
    from cli import aurik_cli

    # sys.argv so umbauen, als wäre aurik_cli.main() direkt aufgerufen worden.
    sys.argv = ["aurik_cli", *cli_args]
    aurik_cli.main()
    return 0


def main() -> int:
    args = sys.argv[1:]
    if not args or {"-h", "--help"} & set(args):
        print(__doc__)
        return 0

    install()
    return _run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
