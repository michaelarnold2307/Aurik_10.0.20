"""Validate new ONNX artifact entries in models/manifest.json.

Checks only the newly integrated artifacts:
- utmosv2_ssl_encoder_onnx
- laion_clap_audio_encoder_onnx
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "models" / "manifest.json"
REQUIRED = {
    "utmosv2_ssl_encoder_onnx": "models/utmosv2/utmosv2_ssl_encoder.onnx",
    "laion_clap_audio_encoder_onnx": "models/clap/audio_encoder.onnx",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not MANIFEST.exists():
        print(f"Fehler: Manifest fehlt: {MANIFEST}")
        return 2

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    models = data.get("models", [])
    by_name = {m.get("name"): m for m in models if isinstance(m, dict)}

    ok = True
    for name, rel in REQUIRED.items():
        entry = by_name.get(name)
        if entry is None:
            print(f"MISSING_ENTRY {name}")
            ok = False
            continue

        path = ROOT / rel
        if not path.exists():
            print(f"MISSING_FILE {name} {rel}")
            ok = False
            continue

        size_actual = path.stat().st_size
        sha_actual = _sha256(path)
        size_manifest = int(entry.get("size_bytes", -1))
        sha_manifest = str(entry.get("sha256", ""))
        path_manifest = str(entry.get("bundled_path", ""))

        if path_manifest != rel:
            print(f"PATH_MISMATCH {name} manifest={path_manifest} actual={rel}")
            ok = False
        if size_manifest != size_actual:
            print(f"SIZE_MISMATCH {name} manifest={size_manifest} actual={size_actual}")
            ok = False
        if sha_manifest != sha_actual:
            print(f"SHA_MISMATCH {name} manifest={sha_manifest} actual={sha_actual}")
            ok = False

        if path_manifest == rel and size_manifest == size_actual and sha_manifest == sha_actual:
            print(f"OK {name}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
