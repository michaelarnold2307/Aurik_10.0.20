"""rx_autopilot/windows_agent.py — RX-Bearbeitung OHNE Menschen (Windows-Seite).

Läuft IN der Windows-VM (siehe vm_setup.sh) mit dem von RX installierten
Python. Verarbeitet eine Watch-Queue deterministisch:

    <name>__<defektklasse>.wav  →  <name>_rx12.wav

Die Modul-Ketten je Klasse kommen aus rx_presets.json (nur dokumentierte
iZotope-Standard-Einstellungen). Kein GUI, kein Experten-Know-how: Die
„Bedienung" von RX ist hier ein deterministischer Batch.

Der Agent ist defensiv geschrieben: Beim ersten Start (--selftest) wird die
tatsächlich installierte rxconnect-API introspiziert und als JSON ausgegeben —
damit kann die Adapter-Zuordnung ohne Zugriff auf die VM-Dokumentation
verifiziert werden. Ohne importierbares rxconnect bricht der Agent LAUT ab
(§V6 (copilot-instructions.md): kein stiller Fallback).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

logger = logging.getLogger("rx_autopilot")

_PRESETS = Path(__file__).resolve().parent / "rx_presets.json"


def _lade_presets() -> dict[str, list[dict]]:
    with _PRESETS.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema") != "aurik.rx_autopilot_presets/1":
        raise ValueError("Unbekanntes Preset-Schema")
    presets: dict[str, list[dict]] = data["presets"]
    return presets


def _rxconnect_import() -> object:
    """Importiert rxconnect (nur in der RX-VM verfügbar) oder bricht laut ab."""
    try:
        import rxconnect  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "rxconnect nicht importierbar — läuft dieser Agent in der RX-VM mit dem RX-Python? "
            "Installation: siehe vm_setup.sh. Kein stiller Fallback (§V6 (copilot-instructions.md))."
        ) from exc
    return rxconnect


def _selftest() -> int:
    rx = _rxconnect_import()
    out: dict[str, object] = {"rxconnect_import": "ok", "module": str(rx)}
    surface: list[str] = []
    for name in dir(rx):
        if not name.startswith("_"):
            surface.append(name)
    out["surface"] = sorted(surface)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def _verarbeite(rx: object, eingabe: Path, ausgabe: Path, kette: list[dict]) -> None:
    """Setzt die Preset-Kette headless über die rxconnect-Batch-API um.

    Die Zuordnung nutzt die dokumentierte rxconnect-Oberfläche (BatchProcessor
    + add_processor). Ist die installierte RX-Version abweichend, liefert
    --selftest die API-Oberfläche für die Anpassung — der Agent verarbeitet
    dann trotzdem NIE mit geratenen Parametern.
    """
    bp_klasse = getattr(rx, "BatchProcessor", None)
    if bp_klasse is None:
        raise SystemExit("rxconnect ohne BatchProcessor — --selftest ausführen und Adapter in _verarbeite anpassen.")
    bp = bp_klasse()
    for schritt in kette:
        modul = schritt["module"]
        params = schritt.get("params", {})
        # Namen/Parameter werden als Schlüsselwörter durchgereicht; die
        # Standard-Presets aus rx_presets.json entsprechen der RX-Doku.
        if not hasattr(bp, "add_processor"):
            raise SystemExit("BatchProcessor ohne add_processor — Adapter prüfen (--selftest).")
        bp.add_processor(modul, **params)
    if not hasattr(bp, "process_files") and not hasattr(bp, "process"):
        raise SystemExit("BatchProcessor ohne process-Methode — Adapter prüfen (--selftest).")
    if hasattr(bp, "process_files"):
        verarbeite = bp.process_files
    else:
        verarbeite = bp.process
    verarbeite([str(eingabe)], str(ausgabe.parent))
    if not ausgabe.exists():
        raise SystemExit(f"RX-Ausgabe fehlt nach Verarbeitung: {ausgabe} — Kette {kette}")


def _watch(queue: Path, out: Path, intervall_s: float = 5.0) -> int:
    rx = _rxconnect_import()
    presets = _lade_presets()
    queue.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    done = queue / "done"
    done.mkdir(exist_ok=True)
    logger.info("RX-Autopilot-Watch gestartet: %s -> %s", queue, out)
    while True:
        for eingabe in sorted(queue.glob("*.wav")):
            if "__" not in eingabe.stem:
                shutil.move(str(eingabe), str(done / eingabe.name))
                logger.warning("Übersprungen (kein __-Defektklassen-Suffix): %s", eingabe.name)
                continue
            stamm, _, klasse = eingabe.stem.partition("__")
            kette = presets.get(klasse)
            if kette is None:
                shutil.move(str(eingabe), str(done / eingabe.name))
                logger.warning("Unbekannte Defektklasse %r — übersprungen: %s", klasse, eingabe.name)
                continue
            ausgabe = out / f"{stamm}_rx12.wav"
            if ausgabe.exists():
                shutil.move(str(eingabe), str(done / eingabe.name))
                logger.info("Bereits verarbeitet: %s", ausgabe.name)
                continue
            logger.info("Verarbeite %s (Klasse %s, %d Schritte) …", eingabe.name, klasse, len(kette))
            try:
                _verarbeite(rx, eingabe, ausgabe, kette)
                shutil.move(str(eingabe), str(done / eingabe.name))
                logger.info("Fertig: %s", ausgabe.name)
            except SystemExit:
                raise
            except Exception as exc:  # §V6 (copilot-instructions.md): laut melden, nicht still
                logger.error("Verarbeitung fehlgeschlagen %s: %s", eingabe.name, exc)
                (done / f"{eingabe.name}.failed").write_text(str(exc), encoding="utf-8")
                shutil.move(str(eingabe), str(done / eingabe.name))
        time.sleep(intervall_s)
    return 0


def _watch_http(basis: str, tmp: Path, intervall_s: float = 5.0) -> int:
    """HTTP-Modus: Queue des Hosts pollen (QEMU-Usernet: http://10.0.2.2:8090)."""
    import urllib.parse
    import urllib.request

    rx = _rxconnect_import()
    presets = _lade_presets()
    tmp.mkdir(parents=True, exist_ok=True)
    logger.info("RX-Autopilot-HTTP gestartet: %s", basis)
    while True:
        try:
            with urllib.request.urlopen(basis + "/queue/next", timeout=10) as ant:
                auftrag = json.loads(ant.read().decode("utf-8"))
            if not auftrag:
                time.sleep(intervall_s)
                continue
            # Path-Traversal-Schutz: nur reine Dateinamen aus Remote-Daten zulassen
            name = str(auftrag["name"])
            stamm = str(auftrag["stamm"])
            klasse = auftrag["klasse"]
            if (
                name != Path(name).name
                or stamm != Path(stamm).name
                or "\\" in name
                or "\\" in stamm
                or name in ("", ".", "..")
                or stamm in ("", ".", "..")
            ):
                logger.error("Ungültiger Auftrags-Dateiname abgelehnt: %r / %r", name, stamm)
                time.sleep(intervall_s)
                continue
            kette = presets.get(klasse)
            if kette is None:
                logger.warning("Unbekannte Defektklasse %r — übersprungen: %s", klasse, name)
                continue
            tmp_basis = tmp.resolve()
            eingabe = (tmp_basis / name).resolve()
            ausgabe = (tmp_basis / f"{stamm}_rx12.wav").resolve()
            if eingabe.parent != tmp_basis or ausgabe.parent != tmp_basis:
                logger.error("Pfad außerhalb des Temp-Ordners abgelehnt: %r", name)
                time.sleep(intervall_s)
                continue
            urllib.request.urlretrieve(basis + "/files/" + urllib.parse.quote(name), eingabe)
            logger.info("Verarbeite %s (Klasse %s, %d Schritte) …", name, klasse, len(kette))
            _verarbeite(rx, eingabe, ausgabe, kette)
            with ausgabe.open("rb") as f:
                daten = f.read()
            req = urllib.request.Request(basis + "/result/" + ausgabe.name, data=daten, method="PUT")
            urllib.request.urlopen(req, timeout=60)
            logger.info("Fertig + hochgeladen: %s", ausgabe.name)
        except SystemExit:
            raise
        except Exception as exc:  # §V6 (copilot-instructions.md): laut melden, nicht still
            logger.error("HTTP-Zyklus fehlgeschlagen: %s — nächster Versuch in %ss", exc, intervall_s)
            time.sleep(intervall_s)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RX-Autopilot (headless, deterministisch)")
    ap.add_argument("--selftest", action="store_true", help="rxconnect-API introspizieren und ausgeben")
    ap.add_argument("--watch", action="store_true", help="Watch-Queue-Modus (lokaler Ordner)")
    ap.add_argument("--queue", type=Path, default=Path(r"Z:\queue"), help="Eingabe-Ordner")
    ap.add_argument("--out", type=Path, default=Path(r"Z:\output"), help="Ausgabe-Ordner")
    ap.add_argument("--http", type=str, default=None, help="HTTP-Basis-URL des Host-Shares (z.B. http://10.0.2.2:8090)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.selftest:
        return _selftest()
    if args.http:
        return _watch_http(args.http, Path(__file__).resolve().parent / "_tmp")
    if args.watch:
        return _watch(args.queue, args.out)
    ap.error("--selftest, --watch oder --http erforderlich")
    return 2


if __name__ == "__main__":
    sys.exit(main())
