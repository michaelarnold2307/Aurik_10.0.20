#!/usr/bin/env python3
"""
Export Guard für die CI‑Lite‑Pipeline.
Der Guard führt:
1️⃣ Lädt Konfiguration aus metadata.yaml
2️⃣ Startet backend/core/audio_exporter.py mit Batch‑Support
3️⃣ Loggt die Ergebnisse in export_log.txt
"""

import datetime
import pathlib
import shlex
import subprocess
import sys

import yaml


def run(cmd: str, cwd=None):
    """Helper to run a command and capture output."""
    result = subprocess.run(
        cmd,
        cwd=cwd or pathlib.Path.cwd(),
        shell=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.returncode, result.stdout


def main():
    # 1️⃣ Load metadata — fehlt sie (Solo-Push-Pfad), ist der Guard ein No-Op.
    try:
        with open("metadata.yaml") as f:
            meta = yaml.safe_load(f)
    except FileNotFoundError:
        with open("export_log.txt", "a") as log:
            log.write(f"Export guard skipped at {datetime.datetime.now()} — metadata.yaml not present.\n")
        return
    # 2️⃣ Build command — kanonische API `backend.core.audio_exporter.AudioExporter`.
    #    Produktionsbefund 2026-10-05: der frühere CLI-Aufruf
    #    (`python core/audio_exporter.py --sample-rate … --format …`) lief ins Leere —
    #    der Pfad existiert seit der backend/-Struktur nicht mehr und der Exporter
    #    hat keine CLI-Flags mehr (nur __main__-Demo). Der Guard exportiert deshalb
    #    konfigurationsgetrieben (metadata.yaml) in ein gitignoriertes output/-Unterverzeichnis.
    _formats = ["." + str(x).lstrip(".") for x in meta["formats"]]
    _out_dir = pathlib.Path("output") / "export_guard"
    _out_dir.mkdir(parents=True, exist_ok=True)
    _snippet = (
        "import pathlib, numpy as np;"
        "from backend.core.audio_exporter import AudioExporter;"
        f"sr={int(meta['sample_rate'])}; bd={int(meta['bit_depth'])}; fmts={_formats!r};"
        "e=AudioExporter();"
        "t=np.linspace(0.0, 1.0, sr, endpoint=False);"
        "a=(0.5*np.sin(2*np.pi*440.0*t)).astype(np.float32);"
        f"res=e.batch_export(a, sr, pathlib.Path({str(_out_dir)!r}), formats=fmts, bit_depth=bd, normalize=True);"
        "fehlend=[f for f in fmts if f not in res];"
        "print('EXPORT_GUARD_FORMATE', sorted(res));"
        "print('EXPORT_GUARD_FEHLEND', fehlend);"
        "raise SystemExit(1 if fehlend else 0)"
    )
    cmd = f'"{sys.executable}" -c {shlex.quote(_snippet)}'
    # 3️⃣ Execute
    ret, out = run(cmd)
    print(out)
    # 4️⃣ Log result
    with open("export_log.txt", "a") as log:
        log.write(f"Export finished at {datetime.datetime.now()}\n")
        log.write(out + "\n")


if __name__ == "__main__":
    main()
