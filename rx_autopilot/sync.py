"""rx_autopilot/sync.py — Linux-Seite des RX-Autopiloten (ohne Mensch).

Ablauf:
  1. Paar-Dateien als `<name>__<klasse>.wav` in den virtuofs-Share legen
     (der Agent in der Windows-VM verarbeitet sie automatisch).
  2. Auf die `<name>_rx12.wav`-Ausgaben warten.
  3. DRT-16-Manifest erzeugen und den Vergleichs-Harness laufen lassen
     (`benchmarks/competitive/external_competitive_evidence.py`).

Damit ist die komplette RX-Vergleichskette automatisiert: Die „Bedienung"
von RX ist der Windows-Agent; dieser Sync ist der einzige Orchestrator.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SHARE_DEFAULT = PROJECT.parent / "rx_vm" / "share"


def _szenario_aus(name: str) -> tuple[str, str]:
    stamm, sep, klasse = name.partition("__")
    if not sep or not klasse:
        raise ValueError(f"Ungültiger Paar-Name (erwartet <name>__<klasse>): {name}")
    return stamm, klasse


def _warten_auf(ausgaben: dict[str, Path], timeout_s: float, intervall_s: float = 10.0) -> None:
    gestartet = time.monotonic()
    while time.monotonic() - gestartet < timeout_s:
        if all(p.exists() for p in ausgaben.values()):
            return
        time.sleep(intervall_s)
    fehlend = [str(p) for p in ausgaben.values() if not p.exists()]
    raise TimeoutError(f"RX-Ausgaben fehlen nach {timeout_s:.0f}s: {fehlend}")


def _baue_manifest(szenarien: list[dict], share: Path, ziel: Path) -> None:
    items = []
    for s in szenarien:
        stamm, klasse = s["stamm"], s["klasse"]
        items.append(
            {
                "scenario_id": f"{stamm}-{klasse}".upper().replace(" ", "-"),
                "material": s.get("material", "digital"),
                "defect_class": klasse,
                "reference_path": str(s["reference"]),
                "input_path": str(s["input"]),
                "aurik_path": str(s["aurik"]),
                "rx11_path": str((share / "output" / f"{stamm}_rx12.wav").resolve()),
                "cedar_path": str(s.get("cedar", "")),
            }
        )
    manifest = {
        "items": items,
        "notes": "Automatisch erzeugt vom RX-Autopilot-Sync (rx_autopilot/sync.py). rx11_path = RX-12-Ausgaben des Windows-Agenten.",
        "run_id": f"rx_autopilot_{int(time.time())}",
    }
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RX-Autopilot-Sync (Linux-Seite)")
    ap.add_argument(
        "--szenarien",
        type=Path,
        required=True,
        help="JSON: Liste aus {stamm, klasse, input, reference, aurik, material?}",
    )
    ap.add_argument("--share", type=Path, default=SHARE_DEFAULT, help="virtiofs-Share (Host-Verzeichnis)")
    ap.add_argument("--timeout-s", type=float, default=7200.0, help="Maximale Wartezeit auf RX-Ausgaben")
    ap.add_argument("--nur-manifest", action="store_true", help="Nur Manifest bauen (ohne Queue/Warten/Harness)")
    args = ap.parse_args(argv)

    szenarien = json.loads(args.szenarien.read_text(encoding="utf-8"))
    share = args.share.resolve()
    queue = share / "queue"
    out_dir = share / "output"
    queue.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    ausgaben: dict[str, Path] = {}
    if not args.nur_manifest:
        for s in szenarien:
            quelle = Path(s["input"])
            if not quelle.exists():
                raise SystemExit(f"Eingabe fehlt: {quelle}")
            ziel = queue / f"{s['stamm']}__{s['klasse']}.wav"
            if not ziel.exists():
                shutil.copy2(quelle, ziel)
            ausgaben[s["stamm"]] = out_dir / f"{s['stamm']}_rx12.wav"
        print(f"Queue: {len(ausgaben)} Paare → {queue} (Agent in der VM verarbeitet automatisch)")
        _warten_auf(ausgaben, args.timeout_s)
        print("Alle RX-Ausgaben vorhanden.")

    manifest_ziel = PROJECT / "benchmarks" / "competitive" / "manifest_rx_autopilot.json"
    _baue_manifest(szenarien, share, manifest_ziel)
    print(f"Manifest: {manifest_ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
