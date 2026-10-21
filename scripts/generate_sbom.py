"""scripts/generate_sbom.py — Software Bill of Materials (SBOM) Generator für Aurik 10.0.0.

Erzeugt ein maschinenlesbares SBOM im SPDX-ähnlichen JSON-Format aus:
  - pip-installierten Paketen (pip list --format=json)
  - models/manifest.json (ML-Modell-Gewichte)

Ausführen:
    python scripts/generate_sbom.py --output sbom-9.10.46.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

ROOT = pathlib.Path(__file__).parent.parent


def _pip_packages() -> list[dict[str, str]]:
    """Fragt pip nach installierten Paketen und gibt eine Liste zurück."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"[WARN] pip list fehlgeschlagen: {result.stderr[:200]}", file=sys.stderr)
            return []
        packages = json.loads(result.stdout)
        return [{"name": p["name"], "version": p["version"], "type": "python-package"} for p in packages]
    except Exception as exc:
        print(f"[WARN] pip-Pakete nicht ermittelbar: {exc}", file=sys.stderr)
        return []


def _model_entries() -> list[dict[str, Any]]:
    """Liest models/manifest.json und gibt ML-Modell-Einträge zurück."""
    manifest_path = ROOT / "models" / "manifest.json"
    if not manifest_path.exists():
        print(f"[WARN] {manifest_path} nicht gefunden — keine ML-Modell-Einträge", file=sys.stderr)
        return []

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = []
        for model in manifest.get("models", []):
            entry: dict[str, Any] = {
                "name": model.get("name", "unknown"),
                "type": "ml-model",
                "bundled": model.get("bundled", False),
                "bundled_path": model.get("bundled_path", ""),
                "sha256": model.get("sha256", ""),
                "size_bytes": model.get("size_bytes", 0),
                "license": model.get("license", "unknown"),
                "required": model.get("required", False),
                "fallback": model.get("fallback", ""),
            }
            # Optionale Felder
            if "reference" in model:
                entry["reference"] = model["reference"]
            # SHA256 live verifizieren wenn Datei lokal vorhanden
            if entry["bundled_path"]:
                local = ROOT / entry["bundled_path"]
                if local.exists():
                    h = hashlib.sha256()
                    with open(local, "rb") as f:
                        for chunk in iter(lambda: f.read(65536), b""):
                            h.update(chunk)
                    entry["sha256_verified"] = h.hexdigest()
                    entry["sha256_match"] = entry["sha256_verified"] == entry["sha256"]
                else:
                    entry["sha256_verified"] = None
                    entry["sha256_match"] = None
            entries.append(entry)
        return entries
    except Exception as exc:
        print(f"[WARN] manifest.json nicht parsbar: {exc}", file=sys.stderr)
        return []


def generate_sbom(output_path: pathlib.Path) -> dict[str, Any]:
    """Erstellt das vollständige SBOM-Dokument und schreibt es als JSON."""
    print("SBOM-Generierung gestartet…")

    pip_pkgs = _pip_packages()
    print(f"  pip-Pakete: {len(pip_pkgs)}")

    models = _model_entries()
    print(f"  ML-Modelle (manifest.json): {len(models)}")

    sbom: dict[str, Any] = {
        "sbom_format": "aurik-sbom-v1",
        "spdx_version": "SPDX-2.3",
        "created": datetime.now(timezone.utc).isoformat(),
        "tool": "scripts/generate_sbom.py",
        "project": "Aurik 10.0.0",
        "version": "9.10.46",
        "components": pip_pkgs + models,
        "summary": {
            "python_packages": len(pip_pkgs),
            "ml_models": len(models),
            "total_components": len(pip_pkgs) + len(models),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(sbom, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  SBOM gespeichert: {output_path}")
    return sbom


def main() -> None:
    parser = argparse.ArgumentParser(description="Aurik 10.0.0 SBOM Generator")
    parser.add_argument("--output", default="sbom.json", help="Ausgabepfad für SBOM-JSON")
    args = parser.parse_args()
    generate_sbom(pathlib.Path(args.output))
    print("Fertig.")


if __name__ == "__main__":
    main()
