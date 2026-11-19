"""§SFT Signal-Flow-Trace Viewer — zeigt den neuesten oder einen bestimmten Trace an.

Verwendung:
    python scripts/show_signal_trace.py           # neuester Trace
    python scripts/show_signal_trace.py --latest  # explizit neuester
    python scripts/show_signal_trace.py --json    # als JSON ausgeben
    python scripts/show_signal_trace.py --wav     # nur WAV-Pfad ausgeben
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Workspace-Root in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zeigt den neuesten Signal-Flow-Trace an.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="JSON ausgeben (keine Formatierung)")
    parser.add_argument("--wav", action="store_true", help="Nur WAV-Pfad ausgeben")
    parser.add_argument("--latest", action="store_true", help="Explizit neuester Trace (default)")
    parser.add_argument("--file", type=str, default=None, help="Bestimmte Trace-JSON-Datei laden")
    args = parser.parse_args()

    try:
        from backend.core.signal_flow_tracer import _LATEST_SYMLINK, _format_report, get_signal_flow_tracer

        tracer = get_signal_flow_tracer()

        if args.wav:
            wav = tracer.latest_output_wav()
            print(wav or "(kein WAV gefunden)")
            return

        if args.file:
            data = json.loads(Path(args.file).read_text(encoding="utf-8"))
        else:
            if not _LATEST_SYMLINK.exists():
                print(f"Kein Trace vorhanden: {_LATEST_SYMLINK}")
                print("Starte eine Restaurierung, um den ersten Trace zu erzeugen.")
                sys.exit(1)
            data = json.loads(_LATEST_SYMLINK.read_text(encoding="utf-8"))

        if args.json:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(_format_report(data))

    except ImportError as exc:
        print(f"Import-Fehler: {exc}")
        print("Stelle sicher, dass das Aurik-Venv aktiv ist.")
        sys.exit(1)
    except Exception as exc:
        print(f"Fehler: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
