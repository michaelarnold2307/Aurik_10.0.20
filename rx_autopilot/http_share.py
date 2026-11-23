"""rx_autopilot/http_share.py — Host-seitiger Transfer-Kanal zur RX-VM.

Ohne Samba/Root: Die VM erreicht den Host über QEMU-Usernet als
http://10.0.2.2:8090. Endpunkte:

    GET  /agent/<datei>      Agent/Presets/Bootstrap laden
    GET  /queue/next         Nächstes unbearbeitetes Paar (JSON) oder {}
    GET  /files/<name>       Eingabe-WAV laden
    PUT  /result/<name>      RX-Ausgabe hochladen (Body = WAV)
    GET  /status             Queue-/Ergebnis-Übersicht

Reine stdlib (http.server), kein Framework. Läuft als eigener Prozess auf
127.0.0.1 — Port 8090 wird über QEMU-Usernet aus der VM erreicht.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parents[1]
SHARE_DEFAULT = PROJECT.parent / "rx_vm" / "share"
AGENT_DIR = PROJECT / "rx_autopilot"


class Handler(BaseHTTPRequestHandler):
    server_version = "RXAutopilotShare/1"

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        pfad = self.path.split("?", 1)[0]
        if pfad.startswith("/agent/"):
            name = pfad[len("/agent/") :]
            if "/" in name or ".." in name:
                self._send_json({"error": "ungueltiger Name"}, 400)
                return
            if "\\" in name or name != Path(name).name or not name:
                self._send_json({"error": "ungueltiger Name"}, 400)
                return
            basis = (self.server.share / "agent").resolve()  # type: ignore[attr-defined]
            datei = (basis / name).resolve()
            if datei.parent != basis or not datei.is_file():
                self._send_json({"error": "nicht vorhanden"}, 404)
                return
            self._send_bytes(datei.read_bytes(), "application/octet-stream")
            return
        if pfad == "/queue/next":
            queue: Path = self.server.share / "queue"  # type: ignore[attr-defined]
            queue.mkdir(parents=True, exist_ok=True)
            done = queue / "done"
            erledigt = {p.name for p in done.glob("*.wav")} if done.is_dir() else set()
            for eingabe in sorted(queue.glob("*.wav")):
                if eingabe.name in erledigt or "__" not in eingabe.stem:
                    continue
                stamm, _, klasse = eingabe.stem.partition("__")
                self._send_json({"name": eingabe.name, "stamm": stamm, "klasse": klasse})
                return
            self._send_json({})
            return
        if pfad.startswith("/files/"):
            name = pfad[len("/files/") :]
            if "/" in name or ".." in name:
                self._send_json({"error": "ungueltiger Name"}, 400)
                return
            if "\\" in name or name != Path(name).name or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.\- ]*", name):
                self._send_json({"error": "ungueltiger Name"}, 400)
                return
            basis = self.server.share.resolve()  # type: ignore[attr-defined]
            datei = (basis / name).resolve()
            if datei.parent != basis or not datei.is_file():
                self._send_json({"error": "nicht vorhanden"}, 404)
                return
            self._send_bytes(datei.read_bytes(), "audio/wav")
            return
        if pfad == "/status":
            queue = self.server.share / "queue"  # type: ignore[attr-defined]
            out_dir = self.server.share / "output"  # type: ignore[attr-defined]
            self._send_json(
                {
                    "queue_offen": len(list(queue.glob("*.wav"))) if queue.is_dir() else 0,
                    "ergebnisse": len(list(out_dir.glob("*.wav"))) if out_dir.is_dir() else 0,
                }
            )
            return
        self._send_json({"error": "unbekannter Pfad", "pfad": pfad}, 404)

    def do_PUT(self) -> None:
        pfad = self.path.split("?", 1)[0]
        if not pfad.startswith("/result/"):
            self._send_json({"error": "nur /result/<name>"}, 400)
            return
        name = pfad[len("/result/") :]
        if "/" in name or "\\" in name or ".." in name or not name.endswith(".wav") or name != Path(name).name:
            self._send_json({"error": "ungueltiger Name"}, 400)
            return
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.\- ]*\.wav", name):
            self._send_json({"error": "ungueltiger Name"}, 400)
            return
        try:
            laenge = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "ungueltige Laenge"}, 400)
            return
        if laenge < 0:
            self._send_json({"error": "ungueltige Laenge"}, 400)
            return
        daten = self.rfile.read(laenge)
        out_dir = (self.server.share / "output").resolve()  # type: ignore[attr-defined]
        out_dir.mkdir(parents=True, exist_ok=True)
        sicher = os.path.basename(name)
        if sicher != name or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.\- ]*\.wav", sicher):
            self._send_json({"error": "ungueltiger Name"}, 400)
            return
        ziel = Path(os.path.normpath(os.path.join(str(out_dir), sicher)))
        tmp = Path(os.path.normpath(os.path.join(str(out_dir), sicher + ".part")))
        if ziel.parent != out_dir or tmp.parent != out_dir:
            self._send_json({"error": "ungueltiger Name"}, 400)
            return
        tmp.write_bytes(daten)
        os.replace(tmp, ziel)
        self._send_json({"ok": True, "bytes": len(daten)})

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("[http_share] %s\n" % (fmt % args))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="HTTP-Transferkanal zur RX-VM")
    ap.add_argument("--share", type=Path, default=SHARE_DEFAULT)
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # CLI: Meldungen auf stderr
    share = args.share.resolve()
    share.mkdir(parents=True, exist_ok=True)
    (share / "queue").mkdir(exist_ok=True)
    (share / "output").mkdir(exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.share = share  # type: ignore[attr-defined]
    logger.info("RX-Autopilot-Share: http://127.0.0.1:%d -> %s", args.port, share)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        # intentional: Ctrl+C beendet den Share-Server ohne Stacktrace (V74)
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
